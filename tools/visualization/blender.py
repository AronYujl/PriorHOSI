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
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        human_vertices=human_blender,
        human_faces=np.asarray(human_faces, dtype=np.int32),
        object_vertices=object_blender,
        object_faces=np.asarray(object_faces, dtype=np.int32),
        frame_index=frames,
    )
    source_manifest_sha256 = (
        _sha256(source_manifest) if source_manifest is not None else "absent"
    )
    cache_record = {
        "schema": "infbagel-blender-mesh-cache-v1",
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
            if line.startswith("V2B3_RENDER"):
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
    video_path = staging / "motion-omomo-style.mp4"
    figure_path = staging / "process-k6-omomo-style.png"
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
            renderer_commit=args.renderer_commit,
        )
    except (BlenderRenderError, VideoRenderError, MotionExportError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
