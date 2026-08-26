"""Exact LINGO ground-truth reconstruction and prediction comparison tools."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import shutil
import sys
import uuid
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw
from pytorch3d import transforms
from scipy.interpolate import interp1d

from .blender import BlenderRenderError, _encode_frames, _write_json
from .headless import _sha256, _tree_sha256
from .hsi_lingo import CANONICAL_COORDINATE_FRAME, _load_native_hsi
from .schema import validate_motion_export
from .video import _probe_video


WINDOW_FRAMES = 16
HISTORY_FRAMES = 2
DATA_STEP = 3
WINDOW_STRIDE_RAW = (WINDOW_FRAMES - HISTORY_FRAMES) * DATA_STEP
SMPLX_JOINTS_28 = [
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
    14, 15, 16, 17, 18, 19, 20, 21, 23, 24, 25, 28, 40, 43,
]
RIGHT_ARM_SLOTS = {
    "right_shoulder": 17,
    "right_elbow": 19,
    "right_wrist": 21,
    "right_index1": 26,
}
DATASET_FILES = {
    "joints": "human_joints_aligned.npy",
    "orient": "human_orient.npy",
    "pose": "human_pose.npy",
    "transl": "transl_aligned.npy",
    "betas": "betas.npy",
    "start": "start_idx.npy",
    "end": "end_idx.npy",
    "gender": "gender.pkl",
    "language": "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl",
}


class GroundTruthError(BlenderRenderError):
    """Raised when matched LINGO GT cannot be reconstructed exactly."""


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    try:
        with np.load(path, allow_pickle=False) as loaded:
            return {key: loaded[key] for key in loaded.files}
    except (OSError, TypeError, ValueError) as exc:
        raise GroundTruthError("cannot load NPZ %s" % path) from exc


def _pickle(path: Path, label: str) -> Any:
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)
    except (OSError, pickle.PickleError, EOFError) as exc:
        raise GroundTruthError("cannot load LINGO %s" % label) from exc


def _reference_identity(data: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    required = ("data_idx", "source_sequence_index", "episode_num")
    missing = [key for key in required if key not in data]
    if missing:
        raise GroundTruthError(
            "reference prediction lacks GT identity: %s" % ", ".join(missing)
        )
    return {
        "sequence_id": str(np.asarray(data["sequence_id"]).item()),
        "scene_name": str(np.asarray(data["scene_name"]).item()),
        "caption": str(np.asarray(data["caption"]).item()),
        "data_idx": int(np.asarray(data["data_idx"]).item()),
        "source_sequence_index": int(
            np.asarray(data["source_sequence_index"]).item()
        ),
        "episode_num": int(np.asarray(data["episode_num"]).item()),
        "coarse_frames": int(np.asarray(data["global_jpos"]).shape[0]),
        "interp_scale": int(np.asarray(data["interp_scale"]).item()),
        "coarse_fps": float(np.asarray(data["fps"]).item()),
    }


def _dataset_paths(root: Path) -> Dict[str, Path]:
    paths = {key: root / relative for key, relative in DATASET_FILES.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise GroundTruthError("LINGO dataset files are missing: %s" % ", ".join(missing))
    return paths


def matched_frame_indices(
    reference_data: Mapping[str, np.ndarray], dataset_root: Path | str
) -> Dict[str, Any]:
    """Reproduce ``GroundTruthSource.episode_indices`` and rollout stitching."""

    root = Path(dataset_root).resolve()
    paths = _dataset_paths(root)
    identity = _reference_identity(reference_data)
    starts = np.load(paths["start"], mmap_mode="r")
    ends = np.load(paths["end"], mmap_mode="r")
    language = _pickle(paths["language"], "language mapping")
    required = ("ori_sequence_idx", "start_idx", "text")
    if not isinstance(language, Mapping) or any(key not in language for key in required):
        raise GroundTruthError("LINGO language mapping lacks evaluator identity fields")

    data_idx = identity["data_idx"]
    if data_idx < 0 or data_idx >= len(language["ori_sequence_idx"]):
        raise GroundTruthError("reference data_idx is outside the LINGO language mapping")
    sequence_index = int(language["ori_sequence_idx"][data_idx])
    if sequence_index != identity["source_sequence_index"]:
        raise GroundTruthError("reference source_sequence_index disagrees with LINGO mapping")
    if sequence_index < 0 or sequence_index >= len(starts):
        raise GroundTruthError("source sequence index is outside LINGO sequence bounds")
    start = int(language["start_idx"][data_idx])
    sequence_start = int(starts[sequence_index])
    end = int(ends[sequence_index])
    if start != sequence_start:
        raise GroundTruthError("reference language window is not the source sequence start")
    if end <= start + HISTORY_FRAMES * DATA_STEP:
        raise GroundTruthError("LINGO source sequence is too short for one evaluator window")
    caption_value = language["text"][data_idx]
    caption = str(caption_value[0] if isinstance(caption_value, (list, tuple)) else caption_value)
    if caption != identity["caption"]:
        raise GroundTruthError("reference caption disagrees with the LINGO language mapping")

    expected_windows = int(
        math.ceil((end - start - HISTORY_FRAMES * DATA_STEP) / WINDOW_STRIDE_RAW)
    )
    if expected_windows != identity["episode_num"]:
        raise GroundTruthError("reference episode_num disagrees with LINGO sequence length")
    offsets = np.arange(WINDOW_FRAMES, dtype=np.int64) * DATA_STEP
    windows = np.stack(
        [
            np.minimum(start + window * WINDOW_STRIDE_RAW + offsets, end - 1)
            for window in range(expected_windows)
        ]
    )
    stitched = np.concatenate(
        [windows[0]] + [window[HISTORY_FRAMES:] for window in windows[1:]]
    )
    expected_coarse = WINDOW_FRAMES + (expected_windows - 1) * (
        WINDOW_FRAMES - HISTORY_FRAMES
    )
    if len(stitched) != expected_coarse or expected_coarse != identity["coarse_frames"]:
        raise GroundTruthError("matched GT coarse frame count disagrees with prediction")
    window_lengths = np.asarray(
        [WINDOW_FRAMES]
        + [WINDOW_FRAMES - HISTORY_FRAMES] * (expected_windows - 1),
        dtype=np.int32,
    )
    seams = np.cumsum(window_lengths[:-1], dtype=np.int32)
    return {
        **identity,
        "dataset_root": str(root),
        "sequence_start": sequence_start,
        "sequence_end_exclusive": end,
        "window_indices": windows,
        "stitched_indices": stitched,
        "window_lengths": window_lengths,
        "seams": seams,
    }


def _stitch(array: np.ndarray, windows: np.ndarray) -> np.ndarray:
    values = [np.asarray(array[index]).copy() for index in windows]
    return np.concatenate(
        [values[0]] + [value[HISTORY_FRAMES:] for value in values[1:]], axis=0
    )


def _quaternion_slerp_exact(
    quaternion1: torch.Tensor, quaternion2: torch.Tensor, step: float
) -> torch.Tensor:
    dot = torch.sum(quaternion1 * quaternion2, dim=-1, keepdim=True)
    quaternion1 = torch.where(dot < 0, -quaternion1, quaternion1)
    dot = torch.sum(quaternion1 * quaternion2, dim=-1, keepdim=True)
    use_lerp = dot > (1.0 - 1e-6)
    omega = torch.acos(torch.clamp(dot, -1.0, 1.0))
    sin_omega = torch.sin(omega)
    safe_sin = torch.where(use_lerp, torch.ones_like(sin_omega), sin_omega)
    factor0 = torch.sin((1.0 - step) * omega) / safe_sin
    factor1 = torch.sin(step * omega) / safe_sin
    slerped = quaternion1 * factor0 + quaternion2 * factor1
    # This reversed near-identical LERP is retained to match the evaluator.
    lerped = quaternion1 * step + quaternion2 * (1.0 - step)
    result = torch.where(use_lerp, lerped, slerped)
    return result / torch.linalg.vector_norm(result, dim=-1, keepdim=True)


def _interpolate_local_pose(local_axis_angle: np.ndarray, scale: int) -> np.ndarray:
    local = torch.from_numpy(np.asarray(local_axis_angle, dtype=np.float32))
    matrices = transforms.axis_angle_to_matrix(local.reshape(-1, 22, 3))
    quaternions = transforms.matrix_to_quaternion(matrices)
    frame_count = int(quaternions.shape[0])
    interpolated = torch.zeros(
        (frame_count * scale, 22, 4), dtype=quaternions.dtype
    )
    for frame in range(frame_count - 1):
        for subframe in range(scale):
            interpolated[frame * scale + subframe] = _quaternion_slerp_exact(
                quaternions[frame], quaternions[frame + 1], subframe / scale
            )
    interpolated[-scale:] = quaternions[-1]
    axis = transforms.matrix_to_axis_angle(
        transforms.quaternion_to_matrix(interpolated)
    )
    return axis.detach().cpu().numpy().astype(np.float32)


def _interpolate_linear(values: np.ndarray, scale: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    frame_count = int(values.shape[0])
    flattened = values.reshape(frame_count, -1)
    source = np.arange(frame_count)
    target = np.linspace(0, frame_count - 1, frame_count * scale)
    result = interp1d(source, flattened, axis=0)(target)
    return np.asarray(result, dtype=np.float32).reshape(
        (frame_count * scale,) + values.shape[1:]
    )


def _slice_sha256(fields: Mapping[str, np.ndarray | str | int]) -> str:
    digest = hashlib.sha256()
    for key in sorted(fields):
        digest.update(key.encode("utf-8") + b"\0")
        value = fields[key]
        if isinstance(value, np.ndarray):
            contiguous = np.ascontiguousarray(value)
            digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
            digest.update(str(contiguous.shape).encode("ascii") + b"\0")
            digest.update(contiguous.tobytes())
        else:
            digest.update(str(value).encode("utf-8") + b"\0")
    return digest.hexdigest()


def export_matched_ground_truth(
    reference_motion_path: Path | str,
    *,
    dataset_root: Path | str,
    output_path: Path | str,
    manifest_path: Path | str,
    smpl_models: Path | str,
    scene_mesh: Path | str,
    renderer_commit: str = "local-unrecorded",
    source_evaluator_commit: str = "unavailable-read-only-dataset-adapter",
    command: str = "tools.visualization.hsi_ground_truth export",
) -> Dict[str, Any]:
    """Export the exact GT episode named by a native HSI prediction."""

    reference = Path(reference_motion_path).resolve()
    root = Path(dataset_root).resolve()
    output = Path(output_path).resolve()
    manifest = Path(manifest_path).resolve()
    models = Path(smpl_models).resolve()
    scene = Path(scene_mesh).resolve()
    if output.exists() or manifest.exists():
        raise GroundTruthError("refusing to overwrite matched GT output")
    for label, path, is_dir in (
        ("reference prediction", reference, False),
        ("LINGO dataset", root, True),
        ("SMPL-X models", models, True),
        ("LINGO scene mesh", scene, False),
    ):
        if not (path.is_dir() if is_dir else path.is_file()):
            raise GroundTruthError("%s does not exist: %s" % (label, path))

    reference_data, native = _load_native_hsi(reference)
    identity = matched_frame_indices(reference_data, root)
    paths = _dataset_paths(root)
    windows = identity["window_indices"]
    joints = _stitch(np.load(paths["joints"], mmap_mode="r"), windows).astype(np.float32)
    orient = _stitch(np.load(paths["orient"], mmap_mode="r"), windows)
    pose = _stitch(np.load(paths["pose"], mmap_mode="r"), windows).reshape(-1, 21, 3)
    transl = _stitch(np.load(paths["transl"], mmap_mode="r"), windows)
    local_pose = np.concatenate([orient.reshape(-1, 1, 3), pose], axis=1)
    fine_pose = _interpolate_local_pose(local_pose, identity["interp_scale"])
    fine_transl = _interpolate_linear(transl, identity["interp_scale"])
    betas_all = np.load(paths["betas"], mmap_mode="r")
    genders = _pickle(paths["gender"], "gender mapping")
    sequence_index = identity["source_sequence_index"]
    if sequence_index >= len(betas_all) or sequence_index >= len(genders):
        raise GroundTruthError("GT shape/gender identity is outside dataset bounds")
    betas = np.asarray(betas_all[sequence_index], dtype=np.float32)
    gender = str(genders[sequence_index])
    if joints.shape != (identity["coarse_frames"], 28, 3):
        raise GroundTruthError("matched GT joints have an unexpected shape")
    fine_frames = identity["coarse_frames"] * identity["interp_scale"]
    if fine_pose.shape != (fine_frames, 22, 3) or fine_transl.shape != (fine_frames, 3):
        raise GroundTruthError("matched GT SMPL-X parameter shapes are inconsistent")

    exact_slice = {
        "window_indices": identity["window_indices"],
        "stitched_indices": identity["stitched_indices"],
        "global_jpos": joints,
        "smplx_pose": fine_pose,
        "transl": fine_transl,
        "betas": betas,
        "gender": gender,
        "caption": identity["caption"],
    }
    slice_hash = _slice_sha256(exact_slice)
    payload = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "sequence_id": np.asarray(identity["sequence_id"]),
        "task_family": np.asarray("hsi"),
        "motion_role": np.asarray("ground_truth"),
        "coordinate_frame": np.asarray(CANONICAL_COORDINATE_FRAME),
        "fps": np.asarray(native["fine_fps"], dtype=np.float32),
        "source_rollout_fps": np.asarray(native["coarse_fps"], dtype=np.float32),
        "global_jpos": joints,
        "global_orient": np.ascontiguousarray(fine_pose[:, 0]),
        "body_pose": np.ascontiguousarray(fine_pose[:, 1:]),
        "transl": np.ascontiguousarray(fine_transl),
        "betas": betas,
        "gender": np.asarray(gender),
        "smplx_output_transform": np.asarray("identity"),
        "interp_scale": np.asarray(identity["interp_scale"], dtype=np.int32),
        "window_lengths": identity["window_lengths"],
        "seams": identity["seams"],
        "history_frames": np.asarray(HISTORY_FRAMES, dtype=np.int32),
        "scene_name": np.asarray(identity["scene_name"]),
        "caption": np.asarray(identity["caption"]),
        "data_idx": np.asarray(identity["data_idx"], dtype=np.int64),
        "source_sequence_index": np.asarray(sequence_index, dtype=np.int64),
        "episode_num": np.asarray(identity["episode_num"], dtype=np.int32),
        "gt_window_raw_indices": identity["window_indices"],
        "gt_stitched_raw_indices": identity["stitched_indices"],
        "ground_truth_slice_sha256": np.asarray(slice_hash),
        "reference_prediction_sha256": np.asarray(_sha256(reference)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    summary = validate_motion_export(output)
    protocol = {
        "window_frames": WINDOW_FRAMES,
        "history_frames": HISTORY_FRAMES,
        "data_step": DATA_STEP,
        "window_stride_raw": WINDOW_STRIDE_RAW,
        "interp_scale": identity["interp_scale"],
        "clamp_policy": "minimum(raw_index, source_end_exclusive - 1)",
    }
    record = {
        "export_schema_version": 1,
        "source_git_commit": source_evaluator_commit,
        "source_live_head_at_completion": renderer_commit,
        "resolved_config_sha256": _slice_sha256(
            {"protocol": json.dumps(protocol, sort_keys=True)}
        ),
        "checkpoint_path_and_sha256": "not-applicable-ground-truth",
        "dataset_snapshot_and_sha256": "exact-lingo-selected-slice: %s" % slice_hash,
        "smpl_models_sha256": _tree_sha256(models),
        "object_asset_manifest_sha256": "absent-hsi-ground-truth",
        "scene_asset_manifest_sha256": _sha256(scene),
        "command": command,
        "working_directory": str(Path.cwd()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_sha256": _sha256(output),
        "motion_role": "ground_truth",
        "visualization_only": True,
        "evaluation_forbidden": True,
        "sequence_id": identity["sequence_id"],
        "scene_name": identity["scene_name"],
        "caption": identity["caption"],
        "data_idx": identity["data_idx"],
        "source_sequence_index": sequence_index,
        "episode_num": identity["episode_num"],
        "source_sequence_bounds": [
            identity["sequence_start"], identity["sequence_end_exclusive"]
        ],
        "ground_truth_protocol": protocol,
        "ground_truth_slice_sha256": slice_hash,
        "dataset_root": str(root),
        "dataset_input_paths": {key: str(path) for key, path in paths.items()},
        "reference_prediction_path": str(reference),
        "reference_prediction_sha256": _sha256(reference),
        # Retained for the shared HSI mesh-cache manifest contract.
        "native_source_sha256": _sha256(reference),
        "renderer_commit": renderer_commit,
        "source_evaluator_commit": source_evaluator_commit,
        "scene_mesh_path": str(scene),
        "scene_mesh_sha256": _sha256(scene),
        "timing": {
            "coarse_frames": identity["coarse_frames"],
            "fine_frames": fine_frames,
            "coarse_fps": native["coarse_fps"],
            "fine_fps": native["fine_fps"],
            "duration_seconds": native["duration_seconds"],
        },
    }
    _write_json(manifest, record)
    validate_motion_export(output, manifest_path=manifest)
    return {**summary, **record["timing"], "motion_sha256": record["motion_sha256"]}


def _smplx_joints(
    data: Mapping[str, np.ndarray], smpl_models: Path, batch_size: int = 32
) -> np.ndarray:
    import smplx

    gender = str(np.asarray(data["gender"]).item())
    betas = np.asarray(data["betas"], dtype=np.float32)
    pose = np.concatenate(
        [
            np.asarray(data["global_orient"], dtype=np.float32)[:, None],
            np.asarray(data["body_pose"], dtype=np.float32),
        ],
        axis=1,
    )
    transl = np.asarray(data["transl"], dtype=np.float32)
    try:
        model = smplx.create(
            str(smpl_models), model_type="smplx", gender=gender,
            num_betas=int(betas.size), use_pca=False, flat_hand_mean=True,
            batch_size=1,
        ).to("cpu")
    except Exception as exc:
        raise GroundTruthError("cannot create SMPL-X model for diagnosis") from exc
    result = []
    with torch.no_grad():
        for begin in range(0, len(pose), batch_size):
            end = min(begin + batch_size, len(pose))
            count = end - begin
            orient = torch.from_numpy(pose[begin:end, 0])
            body = torch.from_numpy(pose[begin:end, 1:].reshape(count, -1))
            translation = torch.from_numpy(transl[begin:end])
            beta_batch = torch.from_numpy(np.repeat(betas[None], count, axis=0))
            zeros = lambda width: torch.zeros((count, width), dtype=torch.float32)
            output = model(
                global_orient=orient, body_pose=body, transl=translation,
                betas=beta_batch, left_hand_pose=zeros(45), right_hand_pose=zeros(45),
                jaw_pose=zeros(3), leye_pose=zeros(3), reye_pose=zeros(3),
                expression=zeros(int(model.expression.shape[1])), return_verts=False,
            )
            result.append(output.joints[:, SMPLX_JOINTS_28].cpu().numpy())
    return np.concatenate(result).astype(np.float32)


def _motion_statistics(joints: np.ndarray) -> Dict[str, Any]:
    pelvis = joints[:, 0]
    result: Dict[str, Any] = {"frame_count": int(len(joints)), "joints": {}}
    for name, slot in RIGHT_ARM_SLOTS.items():
        values = joints[:, slot]
        relative_y = values[:, 1] - pelvis[:, 1]
        result["joints"][name] = {
            "slot": slot,
            "world_y_min_m": float(values[:, 1].min()),
            "world_y_max_m": float(values[:, 1].max()),
            "pelvis_relative_y_min_m": float(relative_y.min()),
            "pelvis_relative_y_max_m": float(relative_y.max()),
            "pelvis_relative_y_max_frame": int(np.argmax(relative_y)),
            "range_xyz_m": np.ptp(values, axis=0).astype(float).tolist(),
        }
    wrist = joints[:, RIGHT_ARM_SLOTS["right_wrist"]]
    shoulder = joints[:, RIGHT_ARM_SLOTS["right_shoulder"]]
    result["right_wrist_above_shoulder_fraction"] = float(
        np.mean(wrist[:, 1] > shoulder[:, 1])
    )
    result["right_wrist_minus_shoulder_max_m"] = float(
        np.max(wrist[:, 1] - shoulder[:, 1])
    )
    return result


def _error_statistics(reference: np.ndarray, candidate: np.ndarray) -> Dict[str, Any]:
    if reference.shape != candidate.shape:
        raise GroundTruthError("joint comparison arrays have different shapes")
    error = np.linalg.norm(candidate - reference, axis=-1)
    return {
        "mean_m": float(error.mean()),
        "median_m": float(np.median(error)),
        "p95_m": float(np.quantile(error, 0.95)),
        "max_m": float(error.max()),
        "right_arm_mean_max_m": {
            name: [float(error[:, slot].mean()), float(error[:, slot].max())]
            for name, slot in RIGHT_ARM_SLOTS.items()
        },
    }


def diagnose_prediction_vs_ground_truth(
    prediction_path: Path | str,
    ground_truth_path: Path | str,
    *,
    smpl_models: Path | str,
    output_path: Path | str,
) -> Dict[str, Any]:
    prediction = Path(prediction_path).resolve()
    gt_path = Path(ground_truth_path).resolve()
    models = Path(smpl_models).resolve()
    destination = Path(output_path).resolve()
    if destination.exists():
        raise GroundTruthError("refusing to overwrite HSI diagnosis")
    prediction_data, native = _load_native_hsi(prediction)
    validate_motion_export(gt_path)
    gt_data = _load_npz(gt_path)
    if str(np.asarray(gt_data.get("motion_role", "")).item()) != "ground_truth":
        raise GroundTruthError("comparison input is not labelled ground_truth")
    for key in ("sequence_id", "data_idx", "source_sequence_index", "episode_num"):
        if str(np.asarray(prediction_data[key]).item()) != str(np.asarray(gt_data[key]).item()):
            raise GroundTruthError("prediction and GT disagree on %s" % key)
    pred_coarse = np.asarray(prediction_data["global_jpos"], dtype=np.float32)
    gt_coarse = np.asarray(gt_data["global_jpos"], dtype=np.float32)
    pred_fk = _smplx_joints(prediction_data, models)
    gt_fk = _smplx_joints(gt_data, models)
    pred_interpolated = _interpolate_linear(pred_coarse, native["interp_scale"])
    gt_interpolated = _interpolate_linear(gt_coarse, native["interp_scale"])
    pred_wrist_max = float(
        np.max(pred_coarse[:, 21, 1] - pred_coarse[:, 0, 1])
    )
    gt_wrist_max = float(np.max(gt_coarse[:, 21, 1] - gt_coarse[:, 0, 1]))
    record = {
        "schema": "infbagel-hsi-prediction-gt-diagnosis-v1",
        "sequence_id": native["sequence_id"],
        "caption": native["caption"],
        "prediction_path": str(prediction),
        "prediction_sha256": _sha256(prediction),
        "ground_truth_path": str(gt_path),
        "ground_truth_sha256": _sha256(gt_path),
        "smpl_models_path": str(models),
        "smpl_models_sha256": _tree_sha256(models),
        "coarse_prediction": _motion_statistics(pred_coarse),
        "coarse_ground_truth": _motion_statistics(gt_coarse),
        "fine_prediction_smplx_fk": _motion_statistics(pred_fk),
        "fine_ground_truth_smplx_fk": _motion_statistics(gt_fk),
        "prediction_fk_vs_interpolated_global_jpos": _error_statistics(
            pred_interpolated, pred_fk
        ),
        "ground_truth_fk_vs_interpolated_global_jpos": _error_statistics(
            gt_interpolated, gt_fk
        ),
        "right_wrist_raise_gap_gt_minus_prediction_m": gt_wrist_max - pred_wrist_max,
        "diagnostic_scope": (
            "Separates native coarse model joints from exported SMPL-X FK and renderer; "
            "does not score task success."
        ),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    _write_json(destination, record)
    return record


def _matched_render_identity(
    prediction_manifest: Mapping[str, Any], gt_manifest: Mapping[str, Any]
) -> Dict[str, Any]:
    keys = ("scene_mesh_sha256", "frame_png_count")
    for key in keys:
        if prediction_manifest.get(key) != gt_manifest.get(key):
            raise GroundTruthError("prediction and GT render manifests disagree on %s" % key)
    for key in ("width", "height", "fps", "frame_count"):
        if prediction_manifest["video_probe"].get(key) != gt_manifest["video_probe"].get(key):
            raise GroundTruthError("prediction and GT video probes disagree on %s" % key)
    pred_camera = prediction_manifest["scene_report"]["camera_video"]
    gt_camera = gt_manifest["scene_report"]["camera_video"]
    for key in ("location", "rotation_euler", "ortho_scale"):
        if not np.allclose(pred_camera[key], gt_camera[key], atol=1e-6, rtol=0.0):
            raise GroundTruthError("prediction and GT cameras disagree on %s" % key)
    return {
        "scene_mesh_sha256": prediction_manifest["scene_mesh_sha256"],
        "frame_count": int(prediction_manifest["video_probe"]["frame_count"]),
        "fps": float(prediction_manifest["video_probe"]["fps"]),
        "width": int(prediction_manifest["video_probe"]["width"]),
        "height": int(prediction_manifest["video_probe"]["height"]),
        "camera_video": pred_camera,
    }


def _matched_render_grid_identity(
    manifests: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    if len(manifests) < 2 or len(manifests) > 6:
        raise GroundTruthError("render grid requires 2 to 6 inputs")
    base = manifests[0]
    for candidate in manifests[1:]:
        for key in ("sequence_id", "caption"):
            if base.get(key) != candidate.get(key):
                raise GroundTruthError("render grid inputs disagree on %s" % key)
        _matched_render_identity(base, candidate)
    return {
        **_matched_render_identity(base, manifests[1]),
        "sequence_id": base.get("sequence_id"),
        "caption": base.get("caption"),
    }


def _validate_grid_labels(labels: Sequence[str]) -> List[str]:
    normalized = []
    for label in labels:
        if label != label.strip() or not label:
            raise GroundTruthError("render grid labels must be non-empty and trimmed")
        if len(label) > 48:
            raise GroundTruthError("render grid labels may not exceed 48 characters")
        normalized.append(label)
    if len(set(normalized)) != len(normalized):
        raise GroundTruthError("render grid labels must be unique")
    return normalized


def _parse_labeled_render(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("grid input must be LABEL=RENDER_DIR")
    label, directory = value.split("=", 1)
    try:
        _validate_grid_labels([label])
    except GroundTruthError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not directory:
        raise argparse.ArgumentTypeError("grid render directory must be non-empty")
    return label, Path(directory)


def compose_prediction_ground_truth(
    prediction_render_dir: Path | str,
    ground_truth_render_dir: Path | str,
    *,
    output_dir: Path | str,
    crf: int = 18,
) -> Dict[str, Any]:
    prediction_dir = Path(prediction_render_dir).resolve()
    gt_dir = Path(ground_truth_render_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise GroundTruthError("refusing to overwrite HSI comparison artifact")
    manifests = []
    for directory, label in ((prediction_dir, "prediction"), (gt_dir, "ground truth")):
        path = directory / "render.manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GroundTruthError("cannot read %s render manifest" % label) from exc
        manifests.append((path, value))
    identity = _matched_render_identity(manifests[0][1], manifests[1][1])
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise GroundTruthError("FFmpeg and ffprobe are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(".%s.%s.staging" % (destination.name, uuid.uuid4().hex))
    staging.mkdir()
    frames_dir = staging / "frames"
    frames_dir.mkdir()
    for frame in range(identity["frame_count"]):
        paths = [prediction_dir / "frames" / ("%05d.png" % frame), gt_dir / "frames" / ("%05d.png" % frame)]
        if not all(path.is_file() for path in paths):
            raise GroundTruthError("matched render frame is missing at index %d" % frame)
        with Image.open(paths[0]) as left_source, Image.open(paths[1]) as right_source:
            left = left_source.convert("RGB")
            right = right_source.convert("RGB")
            if left.size != (identity["width"], identity["height"]) or right.size != left.size:
                raise GroundTruthError("matched render frame dimensions disagree")
            combined = Image.new("RGB", (identity["width"] * 2, identity["height"]), "white")
            combined.paste(left, (0, 0))
            combined.paste(right, (identity["width"], 0))
            draw = ImageDraw.Draw(combined)
            for x, label in ((0, "HSIPrior prediction"), (identity["width"], "Ground truth")):
                draw.rectangle((x + 16, 14, x + 170, 38), fill=(255, 255, 255))
                draw.text((x + 22, 20), label, fill=(18, 25, 28))
            combined.save(frames_dir / ("%05d.png" % frame), format="PNG")
    video = staging / "prediction-vs-ground-truth-lingo-scene.mp4"
    _encode_frames(ffmpeg, frames_dir, video, fps=identity["fps"], crf=crf)
    probe = _probe_video(ffprobe, video)
    if (
        probe["frame_count"] != identity["frame_count"]
        or probe["width"] != identity["width"] * 2
        or probe["height"] != identity["height"]
        or not math.isclose(probe["fps"], identity["fps"], abs_tol=1e-6)
    ):
        raise GroundTruthError("side-by-side video failed ffprobe validation")
    record = {
        "schema": "infbagel-hsi-prediction-gt-side-by-side-v1",
        "sequence_id": manifests[0][1]["sequence_id"],
        "caption": manifests[0][1]["caption"],
        "labels": ["HSIPrior prediction", "Ground truth"],
        "matched_render_identity": identity,
        "prediction_render_manifest_path": str(manifests[0][0]),
        "prediction_render_manifest_sha256": _sha256(manifests[0][0]),
        "ground_truth_render_manifest_path": str(manifests[1][0]),
        "ground_truth_render_manifest_sha256": _sha256(manifests[1][0]),
        "video_probe": probe,
        "video_sha256": _sha256(video),
        "visualization_only": True,
        "evaluation_forbidden": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(staging / "comparison.manifest.json", record)
    os.rename(staging, destination)
    return {
        "output_dir": str(destination),
        "video_path": str(destination / video.name),
        "manifest_path": str(destination / "comparison.manifest.json"),
        "video_probe": probe,
    }


def compose_render_grid(
    labeled_render_dirs: Sequence[Tuple[str, Path | str]],
    *,
    output_dir: Path | str,
    columns: int = 2,
    crf: int = 18,
    renderer_commit: str = "local-unrecorded",
    command: Optional[str] = None,
) -> Dict[str, Any]:
    labels = _validate_grid_labels([label for label, _ in labeled_render_dirs])
    if not 1 <= columns <= len(labeled_render_dirs):
        raise GroundTruthError("render grid columns must be in [1, input count]")
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise GroundTruthError("refusing to overwrite HSI render-grid artifact")

    inputs = []
    manifests = []
    for label, directory_value in zip(labels, (item[1] for item in labeled_render_dirs)):
        directory = Path(directory_value).resolve()
        manifest_path = directory / "render.manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise GroundTruthError("cannot read render manifest for %s" % label) from exc
        inputs.append((label, directory, manifest_path, manifest))
        manifests.append(manifest)
    identity = _matched_render_grid_identity(manifests)
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise GroundTruthError("FFmpeg and ffprobe are required")

    rows = int(math.ceil(len(inputs) / columns))
    output_width = identity["width"] * columns
    output_height = identity["height"] * rows
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(".%s.%s.staging" % (destination.name, uuid.uuid4().hex))
    staging.mkdir()
    frames_dir = staging / "frames"
    frames_dir.mkdir()
    for frame in range(identity["frame_count"]):
        frame_name = "%05d.png" % frame
        paths = [directory / "frames" / frame_name for _, directory, _, _ in inputs]
        if not all(path.is_file() for path in paths):
            raise GroundTruthError("render-grid frame is missing at index %d" % frame)
        combined = Image.new("RGB", (output_width, output_height), "white")
        with ExitStack() as stack:
            sources = [stack.enter_context(Image.open(path)) for path in paths]
            for index, ((label, _, _, _), source) in enumerate(zip(inputs, sources)):
                image = source.convert("RGB")
                if image.size != (identity["width"], identity["height"]):
                    raise GroundTruthError("render-grid frame dimensions disagree")
                x = (index % columns) * identity["width"]
                y = (index // columns) * identity["height"]
                combined.paste(image, (x, y))
                draw = ImageDraw.Draw(combined)
                bbox = draw.textbbox((0, 0), label)
                label_width = bbox[2] - bbox[0]
                draw.rectangle(
                    (x + 16, y + 14, x + max(170, label_width + 34), y + 40),
                    fill=(255, 255, 255),
                )
                draw.text((x + 22, y + 20), label, fill=(18, 25, 28))
        combined.save(frames_dir / frame_name, format="PNG")

    video = staging / "render-comparison-grid.mp4"
    _encode_frames(ffmpeg, frames_dir, video, fps=identity["fps"], crf=crf)
    probe = _probe_video(ffprobe, video)
    if (
        probe["frame_count"] != identity["frame_count"]
        or probe["width"] != output_width
        or probe["height"] != output_height
        or not math.isclose(probe["fps"], identity["fps"], abs_tol=1e-6)
    ):
        raise GroundTruthError("render-grid video failed ffprobe validation")
    input_records = []
    for label, directory, manifest_path, manifest in inputs:
        input_records.append(
            {
                "label": label,
                "render_dir": str(directory),
                "render_manifest_path": str(manifest_path),
                "render_manifest_sha256": _sha256(manifest_path),
                "motion_role": manifest.get("motion_role"),
                "native_source_path": manifest.get("native_source_path"),
                "native_source_sha256": manifest.get("native_source_sha256"),
                "canonical_motion_sha256": manifest.get("canonical_motion_sha256"),
                "video_sha256": manifest.get("video_sha256"),
            }
        )
    record = {
        "schema": "infbagel-hsi-render-grid-v1",
        "sequence_id": identity["sequence_id"],
        "caption": identity["caption"],
        "labels": labels,
        "inputs": input_records,
        "matched_render_identity": identity,
        "grid": {
            "columns": columns,
            "rows": rows,
            "tile_dimensions": [identity["width"], identity["height"]],
            "output_dimensions": [output_width, output_height],
        },
        "encoder": {"codec": "libx264", "crf": crf, "pixel_format": "yuv420p"},
        "video_probe": probe,
        "video_sha256": _sha256(video),
        "renderer_commit": renderer_commit,
        "command": command,
        "visualization_only": True,
        "evaluation_forbidden": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    manifest_path = staging / "comparison-grid.manifest.json"
    _write_json(manifest_path, record)
    if destination.exists():
        raise GroundTruthError("render-grid artifact directory appeared during composition")
    os.rename(staging, destination)
    return {
        "output_dir": str(destination),
        "video_path": str(destination / video.name),
        "manifest_path": str(destination / manifest_path.name),
        "video_probe": probe,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    export = sub.add_parser("export")
    export.add_argument("reference_prediction", type=Path)
    export.add_argument("--dataset-root", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--smpl-models", type=Path, required=True)
    export.add_argument("--scene-mesh", type=Path, required=True)
    export.add_argument("--renderer-commit", default="local-unrecorded")
    export.add_argument("--source-evaluator-commit", default="unavailable")
    diagnose = sub.add_parser("diagnose")
    diagnose.add_argument("prediction", type=Path)
    diagnose.add_argument("ground_truth", type=Path)
    diagnose.add_argument("--smpl-models", type=Path, required=True)
    diagnose.add_argument("--output", type=Path, required=True)
    compare = sub.add_parser("compose")
    compare.add_argument("prediction_render_dir", type=Path)
    compare.add_argument("ground_truth_render_dir", type=Path)
    compare.add_argument("--output-dir", type=Path, required=True)
    compare.add_argument("--crf", type=int, default=18)
    grid = sub.add_parser("compose-grid")
    grid.add_argument(
        "--input", dest="inputs", action="append", type=_parse_labeled_render,
        required=True, metavar="LABEL=RENDER_DIR",
    )
    grid.add_argument("--columns", type=int, default=2)
    grid.add_argument("--output-dir", type=Path, required=True)
    grid.add_argument("--crf", type=int, default=18)
    grid.add_argument("--renderer-commit", default="local-unrecorded")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "export":
            result = export_matched_ground_truth(
                args.reference_prediction, dataset_root=args.dataset_root,
                output_path=args.output, manifest_path=args.manifest,
                smpl_models=args.smpl_models, scene_mesh=args.scene_mesh,
                renderer_commit=args.renderer_commit,
                source_evaluator_commit=args.source_evaluator_commit,
                command=" ".join(sys.argv),
            )
        elif args.command == "diagnose":
            result = diagnose_prediction_vs_ground_truth(
                args.prediction, args.ground_truth, smpl_models=args.smpl_models,
                output_path=args.output,
            )
        elif args.command == "compose":
            result = compose_prediction_ground_truth(
                args.prediction_render_dir, args.ground_truth_render_dir,
                output_dir=args.output_dir, crf=args.crf,
            )
        else:
            result = compose_render_grid(
                args.inputs, output_dir=args.output_dir, columns=args.columns,
                crf=args.crf, renderer_commit=args.renderer_commit,
                command=" ".join(sys.argv),
            )
    except (GroundTruthError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
