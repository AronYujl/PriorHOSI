"""OMOMO-style Blender rendering for canonical HOI/HOSI motion exports.

This host-side orchestrator reconstructs immutable mesh vertices with the
verified InfBaGel Python environment, then invokes Blender's bundled Python on
that cache.  Blender never imports model or training code and never performs
inference.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .headless import (
    _load_object_mesh,
    _restore_human_mesh,
    _sha256,
    _timeline_manifest_fields,
    _tree_sha256,
)
from .schema import MotionExportError, validate_motion_export
from .video import VideoRenderError, _probe_video


class BlenderRenderError(MotionExportError):
    """Raised when the high-quality Blender render cannot be produced safely."""


DEFAULT_BLENDER_SCRIPT = Path(__file__).with_name("blender_scene.py")
DEFAULT_FIGURE_COUNT = 6
DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 768
DEFAULT_SAMPLES = 64


def _y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    """Rotate InfBaGel y-up coordinates into Blender's right-handed z-up frame."""

    source = np.asarray(points)
    if source.shape[-1] != 3:
        raise BlenderRenderError("coordinate conversion expects a final xyz axis")
    converted = source[..., [0, 2, 1]].copy()
    converted[..., 1] *= -1
    return converted


def _select_process_frames(frame_count: int, count: int = DEFAULT_FIGURE_COUNT) -> np.ndarray:
    if frame_count < 1:
        raise BlenderRenderError("motion must contain at least one frame")
    if count < 1:
        raise BlenderRenderError("process figure frame count must be positive")
    return np.unique(
        np.rint(np.linspace(0, frame_count - 1, count)).astype(np.int64)
    )


def _contact_ranges(contact: np.ndarray) -> list[list[int]]:
    ranges = []
    start = None
    for index, active in enumerate(np.asarray(contact, dtype=bool).tolist() + [False]):
        if active and start is None:
            start = index
        elif not active and start is not None:
            ranges.append([start, index - 1])
            start = None
    return ranges


def _nearest_mesh_distances(
    human_vertices: np.ndarray, object_vertices: np.ndarray
) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise BlenderRenderError(
            "SciPy is required for contact-aware ground correction"
        ) from exc
    distances = []
    for human, manipulated_object in zip(human_vertices, object_vertices):
        tree = cKDTree(manipulated_object)
        nearest, _ = tree.query(human, k=1, workers=-1)
        distances.append(float(nearest.min()))
    return np.asarray(distances, dtype=np.float64)


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _apply_visual_ground_correction(
    human_vertices: np.ndarray,
    object_vertices: np.ndarray,
    *,
    floor_height: float,
    support_tolerance: float = 0.005,
    contact_threshold: float = 0.04,
    correction_sigma: float = 2.0,
    contact_sigma: float = 3.0,
    lower_blend_height: float = 0.08,
    upper_blend_height: float = 0.83,
) -> tuple[np.ndarray, np.ndarray, Dict[str, Any], Dict[str, np.ndarray]]:
    """Derive a presentation-only grounded mesh while preserving hand contact."""

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as exc:
        raise BlenderRenderError(
            "SciPy is required for smoothed ground correction"
        ) from exc
    human = np.asarray(human_vertices, dtype=np.float32)
    manipulated_object = np.asarray(object_vertices, dtype=np.float32)
    if (
        human.ndim != 3
        or manipulated_object.ndim != 3
        or human.shape[0] != manipulated_object.shape[0]
        or human.shape[-1] != 3
        or manipulated_object.shape[-1] != 3
    ):
        raise BlenderRenderError("ground correction expects paired [F,V,3] meshes")
    if upper_blend_height <= lower_blend_height:
        raise BlenderRenderError("upper blend height must exceed lower blend height")
    human_min = human[:, :, 2].min(axis=1).astype(np.float64)
    object_min = manipulated_object[:, :, 2].min(axis=1).astype(np.float64)
    raw_human_delta = floor_height - human_min
    raw_object_delta = np.maximum(0.0, floor_height - object_min)
    human_delta = gaussian_filter1d(
        raw_human_delta, sigma=correction_sigma, mode="nearest"
    )
    human_residual = human_min + human_delta - floor_height
    target_residual = np.clip(
        human_residual, -support_tolerance, support_tolerance
    )
    human_delta += target_residual - human_residual
    object_delta = gaussian_filter1d(
        raw_object_delta, sigma=correction_sigma, mode="nearest"
    )
    object_delta = np.maximum(object_delta, raw_object_delta)

    pre_contact_distance = _nearest_mesh_distances(human, manipulated_object)
    contact = pre_contact_distance < contact_threshold
    contact_strength = gaussian_filter1d(
        contact.astype(np.float64), sigma=contact_sigma, mode="nearest"
    )
    contact_strength = np.maximum(contact_strength, contact.astype(np.float64))
    contact_strength = np.clip(contact_strength, 0.0, 1.0)
    upper_delta = human_delta + contact_strength * (object_delta - human_delta)
    normalized_height = (
        human[:, :, 2] - (floor_height + lower_blend_height)
    ) / (upper_blend_height - lower_blend_height)
    upper_weight = _smoothstep(normalized_height)
    vertex_delta = human_delta[:, None] + upper_weight * (
        upper_delta - human_delta
    )[:, None]
    corrected_human = human.copy()
    corrected_object = manipulated_object.copy()
    corrected_human[:, :, 2] += vertex_delta.astype(np.float32)
    corrected_object[:, :, 2] += object_delta[:, None].astype(np.float32)

    post_human_gap = corrected_human[:, :, 2].min(axis=1) - floor_height
    post_object_penetration = floor_height - corrected_object[:, :, 2].min(axis=1)
    if float(post_human_gap.max()) > support_tolerance + 1e-5:
        raise BlenderRenderError("ground correction left unsupported human frames")
    if float(post_object_penetration.max()) > 1e-5:
        raise BlenderRenderError("ground correction left object penetration")
    post_contact_distance = _nearest_mesh_distances(
        corrected_human, corrected_object
    )
    contact_change = (
        post_contact_distance[contact] - pre_contact_distance[contact]
        if contact.any()
        else np.asarray([], dtype=np.float64)
    )
    record = {
        "mode": "visual_contact_aware_v1",
        "visualization_only": True,
        "evaluation_forbidden": True,
        "floor_height_m": float(floor_height),
        "support_tolerance_m": float(support_tolerance),
        "contact_threshold_m": float(contact_threshold),
        "correction_gaussian_sigma_frames": float(correction_sigma),
        "contact_gaussian_sigma_frames": float(contact_sigma),
        "contact_frame_count": int(contact.sum()),
        "contact_ranges": _contact_ranges(contact),
        "vertical_weight": {
            "kind": "smoothstep_by_original_vertex_height",
            "zero_below_floor_plus_m": float(lower_blend_height),
            "one_above_floor_plus_m": float(upper_blend_height),
        },
        "pre_human_floor_gap_cm": {
            "min": float((human_min - floor_height).min() * 100.0),
            "max": float((human_min - floor_height).max() * 100.0),
        },
        "pre_object_penetration_cm": {
            "min": float((floor_height - object_min).min() * 100.0),
            "max": float((floor_height - object_min).max() * 100.0),
            "frames_over_1cm": int(((floor_height - object_min) > 0.01).sum()),
        },
        "post_human_floor_gap_cm": {
            "min": float(post_human_gap.min() * 100.0),
            "max": float(post_human_gap.max() * 100.0),
        },
        "post_object_penetration_cm": {
            "min": float(post_object_penetration.min() * 100.0),
            "max": float(post_object_penetration.max() * 100.0),
        },
        "human_foot_delta_cm": {
            "min": float(human_delta.min() * 100.0),
            "max": float(human_delta.max() * 100.0),
        },
        "human_upper_delta_cm": {
            "min": float(upper_delta.min() * 100.0),
            "max": float(upper_delta.max() * 100.0),
        },
        "object_delta_cm": {
            "min": float(object_delta.min() * 100.0),
            "max": float(object_delta.max() * 100.0),
        },
        "max_rigid_vertical_correction_cm": float(
            max(np.abs(human_delta).max(), np.abs(object_delta).max()) * 100.0
        ),
        "max_within_human_vertical_differential_cm": float(
            np.abs(upper_delta - human_delta).max() * 100.0
        ),
        "contact_distance_change_cm": {
            "min": float(contact_change.min() * 100.0) if contact_change.size else 0.0,
            "median": float(np.median(contact_change) * 100.0)
            if contact_change.size
            else 0.0,
            "max": float(contact_change.max() * 100.0) if contact_change.size else 0.0,
        },
    }
    streams = {
        "ground_human_foot_delta_z": human_delta.astype(np.float32),
        "ground_human_upper_delta_z": upper_delta.astype(np.float32),
        "ground_object_delta_z": object_delta.astype(np.float32),
        "ground_contact_strength": contact_strength.astype(np.float32),
        "ground_pre_contact_distance": pre_contact_distance.astype(np.float32),
    }
    return corrected_human, corrected_object, record, streams


def _validate_render_settings(
    *, width: int, height: int, fps: float, samples: int, figure_columns: int
) -> None:
    if width < 256 or height < 256 or width % 2 or height % 2:
        raise BlenderRenderError(
            "Blender dimensions must be even and each at least 256 pixels"
        )
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        raise BlenderRenderError("FPS must be finite and in (0, 240]")
    if samples < 1 or samples > 4096:
        raise BlenderRenderError("Cycles samples must be in [1, 4096]")
    if figure_columns < 1:
        raise BlenderRenderError("process figure columns must be positive")


def _blender_command(
    blender_binary: Path,
    blend_scene: Path,
    blender_script: Path,
    config_path: Path,
) -> list[str]:
    return [
        str(blender_binary),
        "-b",
        str(blend_scene),
        "--python-exit-code",
        "2",
        "--python",
        str(blender_script),
        "--",
        "--config",
        str(config_path),
    ]


def _font(size: int) -> ImageFont.ImageFont:
    candidates = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _compose_process_figure(
    frame_paths: Sequence[Path],
    frame_indices: Sequence[int],
    output: Path,
    *,
    columns: int,
) -> Dict[str, Any]:
    if output.exists():
        raise BlenderRenderError("refusing to overwrite process figure %s" % output)
    if not frame_paths or len(frame_paths) != len(frame_indices):
        raise BlenderRenderError("process figure needs one image per frame index")
    if columns < 1:
        raise BlenderRenderError("process figure columns must be positive")
    images = []
    for path in frame_paths:
        if not path.is_file():
            raise BlenderRenderError("process figure frame is missing: %s" % path)
        with Image.open(path) as loaded:
            images.append(loaded.convert("RGB"))
    panel_width, panel_height = images[0].size
    if any(image.size != (panel_width, panel_height) for image in images):
        raise BlenderRenderError("process figure frames have inconsistent dimensions")
    rows = int(math.ceil(len(images) / columns))
    gutter = max(8, panel_width // 64)
    label_height = max(36, panel_height // 16)
    canvas = Image.new(
        "RGB",
        (
            columns * panel_width + (columns - 1) * gutter,
            rows * (panel_height + label_height) + (rows - 1) * gutter,
        ),
        (247, 245, 239),
    )
    draw = ImageDraw.Draw(canvas)
    font = _font(max(18, label_height // 2))
    font_size = int(getattr(font, "size", max(18, label_height // 2)))
    for index, (image, frame) in enumerate(zip(images, frame_indices)):
        column = index % columns
        row = index // columns
        x = column * (panel_width + gutter)
        y = row * (panel_height + label_height + gutter)
        canvas.paste(image, (x, y))
        label = "frame %d" % int(frame)
        draw.text(
            (x + gutter, y + panel_height + (label_height - font_size) // 2),
            label,
            fill=(35, 35, 35),
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="PNG", optimize=False)
    return {
        "dimensions": list(canvas.size),
        "columns": columns,
        "rows": rows,
        "frame_indices": [int(frame) for frame in frame_indices],
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise BlenderRenderError("refusing to overwrite JSON %s" % path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _prepare_mesh_cache(
    motion: Path,
    *,
    source_manifest: Optional[Path],
    smpl_models: Path,
    object_mesh: Path,
    object_rest_frame: str,
    hand_pose_fallback: str,
    cache_path: Path,
    cache_manifest_path: Path,
    renderer_commit: str,
    ground_correction: str = "none",
    floor_height: float = 0.015,
    uncorrected_cache: Optional[Path] = None,
) -> Dict[str, Any]:
    if cache_path.exists() or cache_manifest_path.exists():
        raise BlenderRenderError("refusing to overwrite Blender mesh cache")
    summary = validate_motion_export(motion, manifest_path=source_manifest)
    if summary["task_family"] not in ("hoi", "hosi"):
        raise BlenderRenderError("Blender mesh renderer expects HOI/HOSI motion")
    try:
        with np.load(motion, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except (OSError, TypeError, ValueError) as exc:
        raise BlenderRenderError("cannot load canonical motion for Blender") from exc
    frame_count = int(summary["pose_frames"])
    frames = np.arange(frame_count, dtype=np.int64)
    human_vertices, _, human_faces, hand_pose_source = _restore_human_mesh(
        data, smpl_models, frames, "cpu", hand_pose_fallback
    )
    rest_vertices, object_faces = _load_object_mesh(
        object_mesh, coordinate_frame=object_rest_frame
    )
    object_trans = np.asarray(data["object_trans"], dtype=np.float32)
    object_rot = np.asarray(data["object_rot_mat"], dtype=np.float32)
    if object_trans.shape[0] != frame_count or object_rot.shape[0] != frame_count:
        raise BlenderRenderError("object stream does not match human frame count")
    object_vertices = np.einsum(
        "fij,vj->fvi", object_rot, rest_vertices
    ) + object_trans[:, None, :]
    human_blender = _y_up_to_z_up(human_vertices).astype(np.float32)
    object_blender = _y_up_to_z_up(object_vertices).astype(np.float32)
    uncorrected_cache_sha256 = "absent"
    if uncorrected_cache is not None:
        if not uncorrected_cache.is_file():
            raise BlenderRenderError(
                "uncorrected reference cache does not exist: %s" % uncorrected_cache
            )
        try:
            with np.load(uncorrected_cache, allow_pickle=False) as reference:
                reference_human = np.asarray(reference["human_vertices"])
                reference_object = np.asarray(reference["object_vertices"])
                reference_human_faces = np.asarray(reference["human_faces"])
                reference_object_faces = np.asarray(reference["object_faces"])
        except (KeyError, OSError, TypeError, ValueError) as exc:
            raise BlenderRenderError("cannot validate uncorrected reference cache") from exc
        if not (
            np.array_equal(reference_human, human_blender)
            and np.array_equal(reference_object, object_blender)
            and np.array_equal(
                reference_human_faces, np.asarray(human_faces, dtype=np.int32)
            )
            and np.array_equal(
                reference_object_faces, np.asarray(object_faces, dtype=np.int32)
            )
        ):
            raise BlenderRenderError(
                "uncorrected reference cache does not match canonical reconstruction"
            )
        uncorrected_cache_sha256 = _sha256(uncorrected_cache)
    correction_streams: Dict[str, np.ndarray] = {}
    if ground_correction == "visual_contact_aware_v1":
        if uncorrected_cache is None:
            raise BlenderRenderError(
                "visual ground correction requires --uncorrected-cache provenance"
            )
        (
            human_blender,
            object_blender,
            ground_correction_record,
            correction_streams,
        ) = _apply_visual_ground_correction(
            human_blender, object_blender, floor_height=floor_height
        )
    elif ground_correction == "none":
        ground_correction_record = {
            "mode": "none",
            "visualization_only": False,
            "evaluation_forbidden": False,
        }
    else:
        raise BlenderRenderError(
            "unsupported Blender ground correction %s" % ground_correction
        )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        human_vertices=human_blender,
        human_faces=np.asarray(human_faces, dtype=np.int32),
        object_vertices=object_blender,
        object_faces=np.asarray(object_faces, dtype=np.int32),
        frame_index=frames,
        **correction_streams,
    )
    source_manifest_sha256 = (
        _sha256(source_manifest) if source_manifest is not None else "absent"
    )
    cache_record = {
        "schema": "infbagel-blender-mesh-cache-v2",
        "sequence_id": summary["sequence_id"],
        "task_family": summary["task_family"],
        "frame_count": frame_count,
        "coordinate_frame": "blender_z_up",
        "source_coordinate_frame": summary["coordinate_frame"],
        "coordinate_transform": [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0],
        ],
        "hand_pose_source": hand_pose_source,
        "hand_pose_fallback": hand_pose_fallback,
        "source_motion_sha256": _sha256(motion),
        "source_manifest_sha256": source_manifest_sha256,
        "smpl_asset_path": str(smpl_models.resolve()),
        "smpl_asset_sha256": _tree_sha256(smpl_models),
        "object_asset_path": str(object_mesh.resolve()),
        "object_asset_sha256": _sha256(object_mesh),
        "object_rest_frame": object_rest_frame,
        "renderer_commit": renderer_commit,
        "ground_correction": ground_correction_record,
        "uncorrected_cache_path": (
            str(uncorrected_cache.resolve())
            if uncorrected_cache is not None
            else "absent"
        ),
        "uncorrected_cache_sha256": uncorrected_cache_sha256,
        "human_vertex_count": int(human_blender.shape[1]),
        "human_face_count": int(human_faces.shape[0]),
        "object_vertex_count": int(object_blender.shape[1]),
        "object_face_count": int(object_faces.shape[0]),
        "cache_sha256": _sha256(cache_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        **_timeline_manifest_fields(data, frames),
    }
    _write_json(cache_manifest_path, cache_record)
    return {**summary, **cache_record}


def _run_blender(command: Sequence[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            list(command),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
        )
        if process.stdout is None:
            process.kill()
            raise BlenderRenderError("cannot capture Blender output")
        for line in process.stdout:
            log.write(line)
            log.flush()
            if line.startswith(
                ("V2B3_RENDER", "V3A_RENDER", "LINGO_RENDER", "LINGO_FIGURE")
            ):
                print(line.rstrip(), flush=True)
        returncode = process.wait()
    if returncode != 0:
        raise BlenderRenderError(
            "Blender exited with status %d; inspect %s" % (returncode, log_path)
        )


def _encode_frames(
    ffmpeg: str,
    frames_dir: Path,
    output: Path,
    *,
    fps: float,
    crf: int,
) -> None:
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-framerate",
        "%g" % fps,
        "-start_number",
        "0",
        "-i",
        str(frames_dir / "%05d.png"),
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    completed = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise BlenderRenderError("FFmpeg failed: %s" % completed.stderr.strip())


def render_blender_motion(
    motion_path: Path | str,
    *,
    output_dir: Path | str,
    smpl_models: Path | str,
    object_mesh: Path | str,
    blend_scene: Path | str,
    blender_binary: Path | str,
    source_manifest_path: Optional[Path | str] = None,
    object_rest_frame: str = "z_up",
    hand_pose_fallback: str = "mean",
    width: int = DEFAULT_WIDTH,
    height: int = DEFAULT_HEIGHT,
    fps: float = 30.0,
    samples: int = DEFAULT_SAMPLES,
    crf: int = 18,
    figure_count: int = DEFAULT_FIGURE_COUNT,
    figure_columns: int = 3,
    camera_elev: float = 18.0,
    camera_azim: float = -58.0,
    camera_padding: float = 1.18,
    ground_correction: str = "none",
    uncorrected_cache_path: Optional[Path | str] = None,
    renderer_commit: str = "local-unrecorded",
    blender_script: Path | str = DEFAULT_BLENDER_SCRIPT,
) -> Dict[str, Any]:
    motion = Path(motion_path).resolve()
    destination = Path(output_dir).resolve()
    models = Path(smpl_models).resolve()
    object_asset = Path(object_mesh).resolve()
    scene_asset = Path(blend_scene).resolve()
    blender = Path(blender_binary).resolve()
    script = Path(blender_script).resolve()
    source_manifest = (
        Path(source_manifest_path).resolve()
        if source_manifest_path is not None
        else None
    )
    uncorrected_cache = (
        Path(uncorrected_cache_path).resolve()
        if uncorrected_cache_path is not None
        else None
    )
    _validate_render_settings(
        width=width,
        height=height,
        fps=fps,
        samples=samples,
        figure_columns=figure_columns,
    )
    if destination.exists():
        raise BlenderRenderError("refusing to overwrite artifact directory %s" % destination)
    for label, path, kind in (
        ("SMPL-X models", models, "directory"),
        ("object mesh", object_asset, "file"),
        ("OMOMO blend scene", scene_asset, "file"),
        ("Blender binary", blender, "file"),
        ("Blender scene script", script, "file"),
    ):
        exists = path.is_dir() if kind == "directory" else path.is_file()
        if not exists:
            raise BlenderRenderError("%s does not exist: %s" % (label, path))
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise BlenderRenderError("system FFmpeg and ffprobe are required")
    destination.parent.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    staging = destination.with_name(".%s.%s.staging" % (destination.name, identity))
    staging.mkdir()
    cache = staging / "mesh-cache.npz"
    cache_manifest = staging / "mesh-cache.manifest.json"
    frames_dir = staging / "frames"
    frames_dir.mkdir()
    config_path = staging / "blender-render-config.json"
    scene_report = staging / "blender-scene-report.json"
    log_path = staging / "blender.log"
    grounded = ground_correction != "none"
    render_identity = "omomo-grounded-style" if grounded else "omomo-style"
    video_path = staging / ("motion-%s.mp4" % render_identity)
    figure_path = staging / ("process-k6-%s.png" % render_identity)
    render_manifest = staging / "render.manifest.json"
    cache_summary = _prepare_mesh_cache(
        motion,
        source_manifest=source_manifest,
        smpl_models=models,
        object_mesh=object_asset,
        object_rest_frame=object_rest_frame,
        hand_pose_fallback=hand_pose_fallback,
        cache_path=cache,
        cache_manifest_path=cache_manifest,
        renderer_commit=renderer_commit,
        ground_correction=ground_correction,
        floor_height=0.015,
        uncorrected_cache=uncorrected_cache,
    )
    frame_count = int(cache_summary["frame_count"])
    figure_frames = _select_process_frames(frame_count, figure_count)
    config = {
        "cache": cache.name,
        "frames_dir": frames_dir.name,
        "scene_report": scene_report.name,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "samples": samples,
        "engine": "CYCLES",
        "device": "CPU",
        "camera": {
            "projection": "orthographic",
            "elev_degrees": camera_elev,
            "azim_degrees": camera_azim,
            "padding": camera_padding,
        },
        "floor": {
            "kind": "procedural_staggered_wood",
            "height": 0.015,
            "margin": 10.0,
            "brick_scale": 18.0,
            "mortar_size": 0.006,
            "roughness": 0.48,
            "bump_strength": 0.06,
        },
        "materials": {
            "human_source": "blue",
            "object_source": "purple",
            "smooth_shading": True,
        },
        "ground_correction": cache_summary["ground_correction"],
        "color_management": {"view_transform": "Filmic", "look": "None"},
    }
    _write_json(config_path, config)
    command = _blender_command(blender, scene_asset, script, config_path)
    _run_blender(command, log_path)
    frame_paths = [frames_dir / ("%05d.png" % frame) for frame in range(frame_count)]
    if not all(path.is_file() for path in frame_paths):
        raise BlenderRenderError("Blender did not produce the complete PNG sequence")
    with Image.open(frame_paths[0]) as first:
        if first.size != (width, height):
            raise BlenderRenderError("Blender frame dimensions do not match the request")
    _encode_frames(ffmpeg, frames_dir, video_path, fps=fps, crf=crf)
    video_probe = _probe_video(ffprobe, video_path)
    if (
        video_probe["frame_count"] != frame_count
        or video_probe["width"] != width
        or video_probe["height"] != height
        or not math.isclose(video_probe["fps"], fps, abs_tol=1e-6)
    ):
        raise BlenderRenderError("encoded Blender video failed probe validation")
    figure = _compose_process_figure(
        [frame_paths[int(frame)] for frame in figure_frames],
        figure_frames,
        figure_path,
        columns=figure_columns,
    )
    version = subprocess.run(
        [str(blender), "--version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    ).stdout.splitlines()[0]
    report = json.loads(scene_report.read_text(encoding="utf-8"))
    render_record = {
        "schema": "infbagel-blender-render-v1",
        "sequence_id": cache_summary["sequence_id"],
        "renderer_commit": renderer_commit,
        "renderer_backend": "blender-cycles-omomo-scene",
        "source_motion_sha256": cache_summary["source_motion_sha256"],
        "source_manifest_sha256": cache_summary["source_manifest_sha256"],
        "mesh_cache_sha256": _sha256(cache),
        "mesh_cache_manifest_sha256": _sha256(cache_manifest),
        "smpl_asset_sha256": cache_summary["smpl_asset_sha256"],
        "object_asset_sha256": cache_summary["object_asset_sha256"],
        "blend_scene_path": str(scene_asset),
        "blend_scene_sha256": _sha256(scene_asset),
        "blender_binary_path": str(blender),
        "blender_binary_sha256": _sha256(blender),
        "blender_version": version,
        "blender_script_sha256": _sha256(script),
        "config": config,
        "scene_report": report,
        "frame_png_count": frame_count,
        "frame_png_tree_sha256": _tree_sha256(frames_dir),
        "video_probe": video_probe,
        "video_sha256": _sha256(video_path),
        "process_figure": {**figure, "sha256": _sha256(figure_path)},
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hand_pose_source": cache_summary["hand_pose_source"],
        "ground_correction": cache_summary["ground_correction"],
        "uncorrected_cache_sha256": cache_summary["uncorrected_cache_sha256"],
        "source_window_lengths": cache_summary.get("source_window_lengths"),
        "source_seams": cache_summary.get("source_seams"),
    }
    _write_json(render_manifest, render_record)
    if destination.exists():
        raise BlenderRenderError("artifact directory appeared during rendering")
    os.rename(staging, destination)
    return {
        "sequence_id": cache_summary["sequence_id"],
        "output_dir": str(destination),
        "video_path": str(destination / video_path.name),
        "figure_path": str(destination / figure_path.name),
        "render_manifest_path": str(destination / render_manifest.name),
        "frame_count": frame_count,
        "fps": fps,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render one canonical motion with the OMOMO Blender scene"
    )
    parser.add_argument("motion", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--smpl-models", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--object-rest-frame", choices=("z_up", "y_up"), default="z_up")
    parser.add_argument("--blend-scene", type=Path, required=True)
    parser.add_argument("--blender", type=Path, required=True)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--figure-count", type=int, default=DEFAULT_FIGURE_COUNT)
    parser.add_argument("--figure-columns", type=int, default=3)
    parser.add_argument("--hand-pose-fallback", choices=("mean", "flat"), default="mean")
    parser.add_argument("--camera-elev", type=float, default=18.0)
    parser.add_argument("--camera-azim", type=float, default=-58.0)
    parser.add_argument("--camera-padding", type=float, default=1.18)
    parser.add_argument(
        "--ground-correction",
        choices=("none", "visual_contact_aware_v1"),
        default="none",
    )
    parser.add_argument("--uncorrected-cache", type=Path, default=None)
    parser.add_argument("--renderer-commit", default="local-unrecorded")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = render_blender_motion(
            args.motion,
            output_dir=args.output_dir,
            smpl_models=args.smpl_models,
            object_mesh=args.object_mesh,
            blend_scene=args.blend_scene,
            blender_binary=args.blender,
            source_manifest_path=args.manifest,
            object_rest_frame=args.object_rest_frame,
            hand_pose_fallback=args.hand_pose_fallback,
            width=args.width,
            height=args.height,
            fps=args.fps,
            samples=args.samples,
            crf=args.crf,
            figure_count=args.figure_count,
            figure_columns=args.figure_columns,
            camera_elev=args.camera_elev,
            camera_azim=args.camera_azim,
            camera_padding=args.camera_padding,
            ground_correction=args.ground_correction,
            uncorrected_cache_path=args.uncorrected_cache,
            renderer_commit=args.renderer_commit,
        )
    except (BlenderRenderError, VideoRenderError, MotionExportError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
