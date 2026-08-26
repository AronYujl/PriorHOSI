"""Read-only HSI NPZ adapter and LINGO-style Blender orchestrator.

The Phase 1C evaluator exports schema-3 NPZ files whose SMPL-X parameters are
already in LINGO's y-up world frame.  This module normalizes one such file into
the repository's model-independent schema, reconstructs an immutable human
mesh cache, and invokes a Blender-bundled consumer.  It never imports the HSI
training checkout and never modifies the native export.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from .blender import (
    BlenderRenderError,
    _encode_frames,
    _run_blender,
    _select_process_frames,
    _validate_render_settings,
    _write_json,
    _y_up_to_z_up,
)
from .headless import _restore_human_mesh, _sha256, _tree_sha256
from .schema import MotionExportError, validate_motion_export
from .video import _probe_video


DEFAULT_BLENDER_SCRIPT = Path(__file__).with_name("blender_lingo_scene.py")
NATIVE_SCHEMA_VERSION = 3
CANONICAL_COORDINATE_FRAME = "lingo_y_up_world_m"
NATIVE_REQUIRED = (
    "schema_version",
    "sequence_id",
    "scene_name",
    "caption",
    "fps",
    "interp_scale",
    "global_jpos",
    "global_orient",
    "body_pose",
    "transl",
    "betas",
    "gender",
    "smplx_output_transform",
    "window_lengths",
    "seams",
    "history_frames",
)


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise BlenderRenderError("native HSI %s must be a scalar" % key)
    return value.item()


def _text(data: Mapping[str, np.ndarray], key: str) -> str:
    value = _scalar(data, key)
    if not isinstance(value, (str, bytes, np.str_)):
        raise BlenderRenderError("native HSI %s must be text" % key)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    if not value:
        raise BlenderRenderError("native HSI %s must not be empty" % key)
    return value


def _load_native_hsi(path: Path) -> tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    if path.suffix.lower() != ".npz" or not path.is_file():
        raise BlenderRenderError("native HSI input must be an existing .npz file")
    try:
        with np.load(path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except (OSError, TypeError, ValueError) as exc:
        raise BlenderRenderError("cannot load native HSI NPZ %s" % path) from exc
    missing = [key for key in NATIVE_REQUIRED if key not in data]
    if missing:
        raise BlenderRenderError(
            "native HSI NPZ is missing: %s" % ", ".join(missing)
        )
    if int(_scalar(data, "schema_version")) != NATIVE_SCHEMA_VERSION:
        raise BlenderRenderError("native HSI adapter requires schema_version=3")
    if _text(data, "smplx_output_transform") != "identity":
        raise BlenderRenderError(
            "native HSI schema-3 must have smplx_output_transform=identity"
        )
    scene_name = _text(data, "scene_name")
    sequence_id = _text(data, "sequence_id")
    if not sequence_id.startswith(scene_name + ":"):
        raise BlenderRenderError("native HSI sequence_id disagrees with scene_name")
    coarse_fps = float(_scalar(data, "fps"))
    interp_scale = int(_scalar(data, "interp_scale"))
    if not math.isfinite(coarse_fps) or coarse_fps <= 0:
        raise BlenderRenderError("native HSI fps must be finite and positive")
    if interp_scale < 1:
        raise BlenderRenderError("native HSI interp_scale must be positive")

    global_jpos = np.asarray(data["global_jpos"])
    global_orient = np.asarray(data["global_orient"])
    body_pose = np.asarray(data["body_pose"])
    transl = np.asarray(data["transl"])
    betas = np.asarray(data["betas"])
    if global_jpos.ndim != 3 or global_jpos.shape[1:] != (28, 3):
        raise BlenderRenderError("native HSI global_jpos must have shape [T,28,3]")
    fine_frames = int(global_orient.shape[0])
    if (
        global_orient.shape != (fine_frames, 3)
        or body_pose.shape != (fine_frames, 21, 3)
        or transl.shape != (fine_frames, 3)
        or betas.ndim != 1
        or betas.size < 1
    ):
        raise BlenderRenderError("native HSI SMPL-X arrays have incompatible shapes")
    numeric = (global_jpos, global_orient, body_pose, transl, betas)
    if not all(np.issubdtype(value.dtype, np.number) for value in numeric):
        raise BlenderRenderError("native HSI motion arrays must be numeric")
    if not all(np.isfinite(value).all() for value in numeric):
        raise BlenderRenderError("native HSI motion arrays contain non-finite values")
    coarse_frames = int(global_jpos.shape[0])
    if fine_frames != coarse_frames * interp_scale:
        raise BlenderRenderError(
            "native HSI fine frame count is not coarse_frames * interp_scale"
        )
    lengths = np.asarray(data["window_lengths"])
    seams = np.asarray(data["seams"])
    if (
        lengths.ndim != 1
        or not np.issubdtype(lengths.dtype, np.integer)
        or (lengths <= 0).any()
        or int(lengths.sum()) != coarse_frames
    ):
        raise BlenderRenderError("native HSI window_lengths do not cover the timeline")
    if (
        seams.ndim != 1
        or not np.issubdtype(seams.dtype, np.integer)
        or (seams < 0).any()
        or (seams >= coarse_frames).any()
        or (np.diff(seams) <= 0).any()
    ):
        raise BlenderRenderError("native HSI seams are invalid")
    fine_fps = coarse_fps * interp_scale
    summary = {
        "sequence_id": sequence_id,
        "scene_name": scene_name,
        "caption": _text(data, "caption"),
        "coarse_frames": coarse_frames,
        "fine_frames": fine_frames,
        "coarse_fps": coarse_fps,
        "interp_scale": interp_scale,
        "fine_fps": fine_fps,
        "duration_seconds": fine_frames / fine_fps,
        "gender": _text(data, "gender"),
    }
    return data, summary


def _load_json(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise BlenderRenderError("%s does not exist: %s" % (label, path))
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise BlenderRenderError("cannot read %s" % label) from exc
    if not isinstance(value, dict):
        raise BlenderRenderError("%s must contain a JSON object" % label)
    return value


def adapt_native_hsi(
    source_path: Path | str,
    *,
    output_path: Path | str,
    manifest_path: Path | str,
    shard_report_path: Path | str,
    training_metrics_path: Path | str,
    resolved_config_path: Path | str,
    smpl_models: Path | str,
    scene_mesh: Path | str,
    adapter_commit: str = "local-unrecorded",
    command: str = "tools.visualization.hsi_lingo adapt",
) -> Dict[str, Any]:
    """Normalize one native schema-3 HSI export without changing its arrays."""

    source = Path(source_path).resolve()
    output = Path(output_path).resolve()
    manifest = Path(manifest_path).resolve()
    shard_report = Path(shard_report_path).resolve()
    training_metrics = Path(training_metrics_path).resolve()
    resolved_config = Path(resolved_config_path).resolve()
    models = Path(smpl_models).resolve()
    scene = Path(scene_mesh).resolve()
    if output.exists() or manifest.exists():
        raise BlenderRenderError("refusing to overwrite HSI adapter output")
    for label, path, is_dir in (
        ("shard report", shard_report, False),
        ("training metrics", training_metrics, False),
        ("resolved config", resolved_config, False),
        ("SMPL-X models", models, True),
        ("LINGO scene mesh", scene, False),
    ):
        exists = path.is_dir() if is_dir else path.is_file()
        if not exists:
            raise BlenderRenderError("%s does not exist: %s" % (label, path))
    data, native = _load_native_hsi(source)
    report = _load_json(shard_report, "shard report")
    training = _load_json(training_metrics, "training metrics")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or native["sequence_id"] not in metrics:
        raise BlenderRenderError("selected sequence is absent from shard report")
    checkpoint = report.get("checkpoint")
    if not isinstance(checkpoint, dict) or not checkpoint.get("checkpoint_sha256"):
        raise BlenderRenderError("shard report has no checkpoint provenance")
    source_git_commit = training.get("git_commit")
    if not isinstance(source_git_commit, str) or not source_git_commit:
        raise BlenderRenderError("training metrics have no git_commit")

    payload = dict(data)
    payload.update(
        {
            "schema_version": np.asarray(1, dtype=np.int32),
            "task_family": np.asarray("hsi"),
            "coordinate_frame": np.asarray(CANONICAL_COORDINATE_FRAME),
            "fps": np.asarray(native["fine_fps"], dtype=np.float32),
            "source_schema_version": np.asarray(
                NATIVE_SCHEMA_VERSION, dtype=np.int32
            ),
            "source_rollout_fps": np.asarray(
                native["coarse_fps"], dtype=np.float32
            ),
            "source_motion_sha256": np.asarray(_sha256(source)),
        }
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)
    canonical = validate_motion_export(output)
    for key in ("global_jpos", "global_orient", "body_pose", "transl", "betas"):
        with np.load(output, allow_pickle=False) as normalized:
            if not np.array_equal(normalized[key], data[key]):
                raise BlenderRenderError("HSI adapter changed source array %s" % key)
    record = {
        "export_schema_version": 1,
        "source_git_commit": source_git_commit,
        "source_git_commit_role": "checkpoint_training_commit_from_training_metrics",
        "source_live_head_at_completion": "unavailable-in-native-export",
        "native_export_code_commit": "unavailable-in-native-export",
        "resolved_config_sha256": _sha256(resolved_config),
        "checkpoint_path_and_sha256": "%s: %s"
        % (checkpoint.get("checkpoint_path", "unavailable"), checkpoint["checkpoint_sha256"]),
        "dataset_snapshot_and_sha256": "%s: %s" % (scene, _sha256(scene)),
        "smpl_models_sha256": _tree_sha256(models),
        "object_asset_manifest_sha256": "absent-hsi-no-object-stream",
        "scene_asset_manifest_sha256": _sha256(scene),
        "command": command,
        "working_directory": str(Path.cwd()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_sha256": _sha256(output),
        "adapter_commit": adapter_commit,
        "adapter_kind": "phase1c-native-schema3-to-visualization-schema1",
        "native_source_path": str(source),
        "native_source_sha256": _sha256(source),
        "native_schema_version": NATIVE_SCHEMA_VERSION,
        "native_smplx_output_transform": "identity",
        "shard_report_path": str(shard_report),
        "shard_report_sha256": _sha256(shard_report),
        "training_metrics_path": str(training_metrics),
        "training_metrics_sha256": _sha256(training_metrics),
        "resolved_config_path": str(resolved_config),
        "scene_mesh_path": str(scene),
        "scene_mesh_sha256": _sha256(scene),
        "selection_metrics": metrics[native["sequence_id"]],
        "timing": {
            "coarse_frames": native["coarse_frames"],
            "fine_frames": native["fine_frames"],
            "coarse_fps": native["coarse_fps"],
            "interp_scale": native["interp_scale"],
            "fine_fps": native["fine_fps"],
            "duration_seconds": native["duration_seconds"],
        },
        "sequence_id": native["sequence_id"],
        "scene_name": native["scene_name"],
        "caption": native["caption"],
    }
    _write_json(manifest, record)
    validate_motion_export(output, manifest_path=manifest)
    return {**native, **canonical, "motion_sha256": record["motion_sha256"]}


def _obj_geometry_summary(path: Path) -> Dict[str, Any]:
    lower = np.full(3, np.inf, dtype=np.float64)
    upper = np.full(3, -np.inf, dtype=np.float64)
    vertex_count = 0
    face_count = 0
    has_material = False
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line in handle:
                if line.startswith("v "):
                    value = np.fromstring(line[2:], sep=" ", dtype=np.float64)
                    if value.shape != (3,) or not np.isfinite(value).all():
                        raise BlenderRenderError("LINGO OBJ contains an invalid vertex")
                    lower = np.minimum(lower, value)
                    upper = np.maximum(upper, value)
                    vertex_count += 1
                elif line.startswith("f "):
                    face_count += 1
                elif line.startswith(("mtllib ", "usemtl ")):
                    has_material = True
    except OSError as exc:
        raise BlenderRenderError("cannot inspect LINGO scene OBJ") from exc
    if vertex_count < 3 or face_count < 1:
        raise BlenderRenderError("LINGO scene OBJ has no usable geometry")
    transformed = _y_up_to_z_up(np.stack([lower, upper], axis=0))
    blender_lower = transformed.min(axis=0)
    blender_upper = transformed.max(axis=0)
    return {
        "vertex_count": vertex_count,
        "face_count": face_count,
        "has_material_directives": has_material,
        "source_y_up_bounds": [lower.tolist(), upper.tolist()],
        "blender_z_up_bounds": [blender_lower.tolist(), blender_upper.tolist()],
    }


def _prepare_human_cache(
    motion: Path,
    manifest: Path,
    *,
    smpl_models: Path,
    scene_mesh: Path,
    cache_path: Path,
    cache_manifest_path: Path,
    hand_pose_fallback: str,
    renderer_commit: str,
) -> Dict[str, Any]:
    summary = validate_motion_export(motion, manifest_path=manifest)
    if summary["task_family"] != "hsi":
        raise BlenderRenderError("LINGO scene renderer requires task_family=hsi")
    if summary["coordinate_frame"] != CANONICAL_COORDINATE_FRAME:
        raise BlenderRenderError("unsupported HSI coordinate frame")
    with np.load(motion, allow_pickle=False) as loaded:
        data = {key: loaded[key] for key in loaded.files}
    if _text(data, "smplx_output_transform") != "identity":
        raise BlenderRenderError("HSI SMPL-X output transform must remain identity")
    frame_count = int(summary["pose_frames"])
    frames = np.arange(frame_count, dtype=np.int64)
    human, pelvis, faces, hand_source = _restore_human_mesh(
        data, smpl_models, frames, "cpu", hand_pose_fallback
    )
    human_blender = _y_up_to_z_up(human).astype(np.float32)
    pelvis_blender = _y_up_to_z_up(pelvis).astype(np.float32)
    np.savez_compressed(
        cache_path,
        human_vertices=human_blender,
        human_faces=np.asarray(faces, dtype=np.int32),
        pelvis=pelvis_blender,
        frame_index=frames,
    )
    scene_summary = _obj_geometry_summary(scene_mesh)
    source_record = _load_json(manifest, "canonical motion manifest")
    record = {
        "schema": "infbagel-lingo-human-mesh-cache-v1",
        "sequence_id": summary["sequence_id"],
        "scene_name": str(data["scene_name"].item()),
        "caption": str(data["caption"].item()),
        "frame_count": frame_count,
        "fps": float(summary["fps"]),
        "coordinate_frame": "blender_z_up",
        "source_coordinate_frame": summary["coordinate_frame"],
        "coordinate_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        "coordinate_transform_application_count": 1,
        "smplx_output_transform": "identity",
        "hand_pose_source": hand_source,
        "hand_pose_fallback": hand_pose_fallback,
        "hand_pose_note": (
            "No articulated finger parameters are exported. The recorded SMPL-X "
            "mean-hand fallback follows the relaxed-hand convention used by LINGO's "
            "Blender importer and is visualization-only."
            if hand_pose_fallback == "mean"
            else "No articulated finger parameters are exported; zero flat hands reproduce evaluator FK."
        ),
        "source_motion_sha256": _sha256(motion),
        "source_manifest_sha256": _sha256(manifest),
        "native_source_sha256": source_record["native_source_sha256"],
        "smpl_asset_path": str(smpl_models),
        "smpl_asset_sha256": _tree_sha256(smpl_models),
        "scene_mesh_path": str(scene_mesh),
        "scene_mesh_sha256": _sha256(scene_mesh),
        "scene_geometry": scene_summary,
        "renderer_commit": renderer_commit,
        "human_vertex_count": int(human_blender.shape[1]),
        "human_face_count": int(faces.shape[0]),
        "human_bounds_blender_z_up": [
            human_blender.min(axis=(0, 1)).astype(float).tolist(),
            human_blender.max(axis=(0, 1)).astype(float).tolist(),
        ],
        "human_floor_gap_cm": {
            "min": float(human_blender[:, :, 2].min(axis=1).min() * 100.0),
            "max": float(human_blender[:, :, 2].min(axis=1).max() * 100.0),
        },
        "visual_ground_correction": "none",
        "cache_sha256": _sha256(cache_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(cache_manifest_path, record)
    return record


def _blender_command(
    blender_binary: Path, blender_script: Path, config_path: Path
) -> list[str]:
    return [
        str(blender_binary),
        "--background",
        "--factory-startup",
        "--python-exit-code",
        "2",
        "--python",
        str(blender_script),
        "--",
        "--config",
        str(config_path),
    ]


def _validate_frame_subset(
    frames: Optional[Sequence[int]], frame_count: int
) -> list[int]:
    if frames is None:
        return list(range(frame_count))
    selected = [int(frame) for frame in frames]
    if (
        not selected
        or len(set(selected)) != len(selected)
        or selected != sorted(selected)
        or selected[0] < 0
        or selected[-1] >= frame_count
    ):
        raise BlenderRenderError("render frame subset must be unique, sorted, and valid")
    return selected


def render_lingo_hsi(
    native_motion_path: Path | str,
    *,
    output_dir: Path | str,
    shard_report_path: Path | str,
    training_metrics_path: Path | str,
    resolved_config_path: Path | str,
    smpl_models: Path | str,
    scene_mesh: Path | str,
    blender_binary: Path | str,
    width: int = 1280,
    height: int = 720,
    fps: float = 30.0,
    samples: int = 64,
    crf: int = 18,
    figure_count: int = 4,
    figure_width: int = 1800,
    figure_height: int = 1000,
    camera_elev: float = 32.0,
    camera_azim: float = -55.0,
    camera_padding: float = 1.06,
    scene_decimate_ratio: float = 0.20,
    hand_pose_fallback: str = "mean",
    render_frames: Optional[Sequence[int]] = None,
    figure_frames: Optional[Sequence[int]] = None,
    renderer_commit: str = "local-unrecorded",
    blender_script: Path | str = DEFAULT_BLENDER_SCRIPT,
) -> Dict[str, Any]:
    native = Path(native_motion_path).resolve()
    destination = Path(output_dir).resolve()
    shard_report = Path(shard_report_path).resolve()
    training_metrics = Path(training_metrics_path).resolve()
    resolved_config = Path(resolved_config_path).resolve()
    models = Path(smpl_models).resolve()
    scene = Path(scene_mesh).resolve()
    blender = Path(blender_binary).resolve()
    script = Path(blender_script).resolve()
    _validate_render_settings(
        width=width, height=height, fps=fps, samples=samples, figure_columns=1
    )
    _validate_render_settings(
        width=figure_width,
        height=figure_height,
        fps=fps,
        samples=samples,
        figure_columns=1,
    )
    if destination.exists():
        raise BlenderRenderError("refusing to overwrite artifact directory %s" % destination)
    if not 0.01 <= scene_decimate_ratio <= 1.0:
        raise BlenderRenderError("scene decimate ratio must be in [0.01, 1]")
    if hand_pose_fallback not in ("mean", "flat"):
        raise BlenderRenderError("unsupported hand pose fallback")
    for label, path, is_dir in (
        ("native HSI NPZ", native, False),
        ("shard report", shard_report, False),
        ("training metrics", training_metrics, False),
        ("resolved config", resolved_config, False),
        ("SMPL-X models", models, True),
        ("LINGO scene mesh", scene, False),
        ("Blender binary", blender, False),
        ("Blender scene script", script, False),
    ):
        exists = path.is_dir() if is_dir else path.is_file()
        if not exists:
            raise BlenderRenderError("%s does not exist: %s" % (label, path))
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise BlenderRenderError("system FFmpeg and ffprobe are required")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(
        ".%s.%s.staging" % (destination.name, uuid.uuid4().hex)
    )
    staging.mkdir()
    motion = staging / "motion.canonical.npz"
    motion_manifest = staging / "motion.canonical.manifest.json"
    cache = staging / "human-mesh-cache.npz"
    cache_manifest = staging / "human-mesh-cache.manifest.json"
    frames_dir = staging / "frames"
    frames_dir.mkdir()
    config_path = staging / "blender-lingo-config.json"
    scene_report_path = staging / "blender-scene-report.json"
    log_path = staging / "blender.log"
    video_path = staging / "motion-lingo-scene.mp4"
    figure_path = staging / "trajectory-lingo-shared-scene.png"
    render_manifest = staging / "render.manifest.json"

    adapter_summary = adapt_native_hsi(
        native,
        output_path=motion,
        manifest_path=motion_manifest,
        shard_report_path=shard_report,
        training_metrics_path=training_metrics,
        resolved_config_path=resolved_config,
        smpl_models=models,
        scene_mesh=scene,
        adapter_commit=renderer_commit,
        command=" ".join(sys.argv),
    )
    if not math.isclose(float(adapter_summary["fps"]), fps, abs_tol=1e-6):
        raise BlenderRenderError(
            "requested FPS does not match native coarse_fps * interp_scale"
        )
    cache_record = _prepare_human_cache(
        motion,
        motion_manifest,
        smpl_models=models,
        scene_mesh=scene,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        hand_pose_fallback=hand_pose_fallback,
        renderer_commit=renderer_commit,
    )
    frame_count = int(cache_record["frame_count"])
    rendered = _validate_frame_subset(render_frames, frame_count)
    full_timeline = rendered == list(range(frame_count))
    if figure_frames is None:
        selected_figure_frames = _select_process_frames(
            frame_count, figure_count
        ).astype(int).tolist()
        figure_selection_rule = "rounded_linspace_complete_fine_timeline"
    else:
        selected_figure_frames = _validate_frame_subset(figure_frames, frame_count)
        if len(selected_figure_frames) < 2 or len(selected_figure_frames) > 12:
            raise BlenderRenderError("figure frame selection requires 2 to 12 frames")
        figure_selection_rule = "explicit_increasing_source_frames"
    config = {
        "cache": str(cache),
        "scene_mesh": str(scene),
        "frames_dir": str(frames_dir),
        "output_figure": str(figure_path),
        "scene_report": str(scene_report_path),
        "frame_count": frame_count,
        "render_frame_indices": rendered,
        "selected_figure_frames": selected_figure_frames,
        "selection_rule": figure_selection_rule,
        "video": {"width": width, "height": height},
        "figure": {"width": figure_width, "height": figure_height},
        "samples": samples,
        "engine": "CYCLES",
        "camera": {
            "projection": "orthographic",
            "elev_degrees": camera_elev,
            "azim_degrees": camera_azim,
            "padding": camera_padding,
            "fit_source": "complete_lingo_scene_mesh_bounds",
        },
        "scene": {
            "source_coordinate_frame": CANONICAL_COORDINATE_FRAME,
            "coordinate_transform": "obj_import_y_up_then_apply_once",
            "decimate_ratio": scene_decimate_ratio,
            "cutaway": {
                "enabled": True,
                "kind": "camera_side_dollhouse",
                "sides_blender_z_up": ["z_max", "x_max", "y_min"],
                "boundary_depth_m": 0.25,
                "normal_axis_threshold": 0.65,
            },
            "materials": [
                {
                    "name": "floor_stone",
                    "base_color": [0.39, 0.43, 0.42, 1.0],
                    "roughness": 0.78,
                    "specular": 0.24,
                },
                {
                    "name": "wall_warm_white",
                    "base_color": [0.70, 0.72, 0.69, 1.0],
                    "roughness": 0.76,
                    "specular": 0.22,
                },
                {
                    "name": "furniture_sage",
                    "base_color": [0.42, 0.56, 0.43, 1.0],
                    "roughness": 0.66,
                    "specular": 0.26,
                },
            ],
            "surface_palette": {
                "policy": "floor_wall_geometry_plus_uniform_furniture",
                "floor_depth_m": 0.08,
                "wall_depth_m": 0.20,
                "normal_axis_threshold": 0.65,
            },
        },
        "human_material": {
            "base_color": [0.20, 0.42, 0.56, 1.0],
            "roughness": 0.46,
            "specular": 0.35,
        },
        "lighting": {
            "world_color": [0.78, 0.82, 0.88],
            "world_strength": 0.35,
            "key_energy": 850.0,
            "key_size": 5.0,
            "fill_energy": 450.0,
            "fill_size": 4.0,
            "ambient_occlusion_distance": 3.0,
            "ambient_occlusion_factor": 1.15,
        },
        "color_management": {
            "view_transform": "Filmic",
            "look": "Medium High Contrast",
            "exposure": 0.0,
        },
        "composition": {
            "video": "one human pose per fine source frame",
            "figure": "opaque multi-pose shared scene at unmodified world positions",
            "visual_ground_correction": "none",
        },
    }
    _write_json(config_path, config)
    _run_blender(_blender_command(blender, script, config_path), log_path)
    frame_paths = [frames_dir / ("%05d.png" % frame) for frame in rendered]
    if not all(path.is_file() for path in frame_paths):
        raise BlenderRenderError("Blender did not produce all requested frames")
    with Image.open(frame_paths[0]) as first:
        if first.size != (width, height):
            raise BlenderRenderError("Blender video-frame dimensions are incorrect")
    if not figure_path.is_file():
        raise BlenderRenderError("Blender did not produce the shared-scene figure")
    with Image.open(figure_path) as figure:
        if figure.size != (figure_width, figure_height):
            raise BlenderRenderError("Blender figure dimensions are incorrect")

    video_probe = None
    video_sha256 = None
    if full_timeline:
        _encode_frames(ffmpeg, frames_dir, video_path, fps=fps, crf=crf)
        video_probe = _probe_video(ffprobe, video_path)
        if (
            video_probe["frame_count"] != frame_count
            or video_probe["width"] != width
            or video_probe["height"] != height
            or not math.isclose(video_probe["fps"], fps, abs_tol=1e-6)
        ):
            raise BlenderRenderError("encoded LINGO video failed ffprobe validation")
        video_sha256 = _sha256(video_path)
    blender_version = subprocess.run(
        [str(blender), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).stdout.splitlines()[0]
    scene_report = _load_json(scene_report_path, "Blender scene report")
    record = {
        "schema": "infbagel-lingo-hsi-render-v1",
        "sequence_id": cache_record["sequence_id"],
        "scene_name": cache_record["scene_name"],
        "caption": cache_record["caption"],
        "mode": "full" if full_timeline else "alignment_smoke",
        "renderer_commit": renderer_commit,
        "renderer_backend": "blender-cycles-lingo-full-scene",
        "native_source_path": str(native),
        "native_source_sha256": _sha256(native),
        "canonical_motion_sha256": _sha256(motion),
        "canonical_manifest_sha256": _sha256(motion_manifest),
        "mesh_cache_sha256": _sha256(cache),
        "mesh_cache_manifest_sha256": _sha256(cache_manifest),
        "scene_mesh_path": str(scene),
        "scene_mesh_sha256": _sha256(scene),
        "smpl_asset_sha256": cache_record["smpl_asset_sha256"],
        "blender_binary_path": str(blender),
        "blender_binary_sha256": _sha256(blender),
        "blender_version": blender_version,
        "blender_script_sha256": _sha256(script),
        "config": config,
        "scene_report": scene_report,
        "rendered_frame_indices": rendered,
        "frame_png_count": len(frame_paths),
        "frame_png_sha256": {path.name: _sha256(path) for path in frame_paths},
        "video_probe": video_probe,
        "video_sha256": video_sha256,
        "figure": {
            "path": figure_path.name,
            "selected_frame_indices": selected_figure_frames,
            "sha256": _sha256(figure_path),
            "dimensions": [figure_width, figure_height],
        },
        "hand_pose_source": cache_record["hand_pose_source"],
        "visualization_only": True,
        "evaluation_forbidden": True,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(render_manifest, record)
    if destination.exists():
        raise BlenderRenderError("artifact directory appeared during rendering")
    os.rename(staging, destination)
    return {
        "sequence_id": cache_record["sequence_id"],
        "output_dir": str(destination),
        "video_path": str(destination / video_path.name) if full_timeline else None,
        "figure_path": str(destination / figure_path.name),
        "render_manifest_path": str(destination / render_manifest.name),
        "frame_count": frame_count,
        "rendered_frame_count": len(rendered),
        "fps": fps,
        "mode": record["mode"],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Adapt native HSIPrior NPZ and render it in a LINGO scene"
    )
    parser.add_argument("motion", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--shard-report", type=Path, required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--smpl-models", type=Path, required=True)
    parser.add_argument("--scene-mesh", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--figure-count", type=int, default=4)
    parser.add_argument("--figure-width", type=int, default=1800)
    parser.add_argument("--figure-height", type=int, default=1000)
    parser.add_argument("--camera-elev", type=float, default=32.0)
    parser.add_argument("--camera-azim", type=float, default=-55.0)
    parser.add_argument("--camera-padding", type=float, default=1.06)
    parser.add_argument("--scene-decimate-ratio", type=float, default=0.20)
    parser.add_argument("--hand-pose-fallback", choices=("mean", "flat"), default="mean")
    parser.add_argument("--render-frames", nargs="+", type=int, default=None)
    parser.add_argument("--figure-frames", nargs="+", type=int, default=None)
    parser.add_argument("--renderer-commit", default="local-unrecorded")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = render_lingo_hsi(
            args.motion,
            output_dir=args.output_dir,
            shard_report_path=args.shard_report,
            training_metrics_path=args.training_metrics,
            resolved_config_path=args.resolved_config,
            smpl_models=args.smpl_models,
            scene_mesh=args.scene_mesh,
            blender_binary=args.blender,
            width=args.width,
            height=args.height,
            fps=args.fps,
            samples=args.samples,
            crf=args.crf,
            figure_count=args.figure_count,
            figure_width=args.figure_width,
            figure_height=args.figure_height,
            camera_elev=args.camera_elev,
            camera_azim=args.camera_azim,
            camera_padding=args.camera_padding,
            scene_decimate_ratio=args.scene_decimate_ratio,
            hand_pose_fallback=args.hand_pose_fallback,
            render_frames=args.render_frames,
            figure_frames=args.figure_frames,
            renderer_commit=args.renderer_commit,
        )
    except (BlenderRenderError, MotionExportError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
