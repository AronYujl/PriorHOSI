"""Linux-headless SMPL-X/object MP4 renderer for canonical motion exports.

The renderer consumes an already generated motion NPZ.  It reconstructs the
human and object meshes once, renders one synchronized pose per source frame
with PyTorch3D's CPU rasterizer, and streams RGB frames to FFmpeg.  Model
inference is deliberately outside this module.
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
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import trimesh

from .headless import (
    HeadlessRenderError,
    _load_object_mesh,
    _restore_human_mesh,
    _sha256,
    _timeline_manifest_fields,
    _tree_sha256,
)
from .schema import MotionExportError, validate_motion_export


class VideoRenderError(HeadlessRenderError):
    """Raised when a canonical motion cannot be rendered as a verified MP4."""


def _validate_video_settings(*, width: int, height: int, fps: float, crf: int) -> None:
    if width < 64 or height < 64:
        raise VideoRenderError("video dimensions must each be at least 64 pixels")
    if width % 2 or height % 2:
        raise VideoRenderError("video dimensions must be even for yuv420p")
    if not math.isfinite(fps) or fps <= 0 or fps > 240:
        raise VideoRenderError("video fps must be finite and in (0, 240]")
    if crf < 0 or crf > 51:
        raise VideoRenderError("H.264 CRF must be in [0, 51]")


def _validate_video_targets(output: Path, render_manifest: Path) -> None:
    if output.suffix.lower() != ".mp4":
        raise VideoRenderError("video output must have an .mp4 suffix")
    if render_manifest.suffix.lower() != ".json":
        raise VideoRenderError("video render manifest must have a .json suffix")
    if output.exists():
        raise VideoRenderError("refusing to overwrite video %s" % output)
    if render_manifest.exists():
        raise VideoRenderError(
            "refusing to overwrite video render manifest %s" % render_manifest
        )


def _camera_direction(camera_elev: float, camera_azim: float) -> np.ndarray:
    """Match the V2b.1 Matplotlib view in the repository's y-up coordinates."""

    if not math.isfinite(camera_elev) or not math.isfinite(camera_azim):
        raise VideoRenderError("camera angles must be finite")
    elev = math.radians(camera_elev)
    azim = math.radians(camera_azim)
    direction = np.asarray(
        [
            math.cos(elev) * math.cos(azim),
            math.sin(elev),
            math.cos(elev) * math.sin(azim),
        ],
        dtype=np.float64,
    )
    return direction / np.linalg.norm(direction)


def _look_at_rotation(
    eye: np.ndarray, target: np.ndarray, up: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Return PyTorch3D-compatible row-vector world-to-view R and T."""

    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(up, forward)
    right_norm = np.linalg.norm(right)
    if right_norm < 1e-8:
        raise VideoRenderError("camera direction is parallel to its up vector")
    right /= right_norm
    camera_up = np.cross(forward, right)
    rotation = np.stack([right, camera_up, forward], axis=1)
    translation = -eye @ rotation
    return rotation, translation


def _fixed_orthographic_camera(
    human_vertices: np.ndarray,
    object_vertices: np.ndarray,
    *,
    width: int,
    height: int,
    camera_elev: float,
    camera_azim: float,
    padding: float,
) -> Dict[str, Any]:
    """Fit one orthographic camera to every mesh vertex in the sequence."""

    if not math.isfinite(padding) or padding <= 1.0:
        raise VideoRenderError("camera padding must be finite and greater than 1")
    arrays = [np.asarray(human_vertices), np.asarray(object_vertices)]
    for array in arrays:
        if array.ndim != 3 or array.shape[-1] != 3 or array.size == 0:
            raise VideoRenderError("camera inputs must be non-empty [F,V,3] arrays")
        if not np.isfinite(array).all():
            raise VideoRenderError("camera inputs contain non-finite coordinates")

    world_lower = np.minimum(
        arrays[0].min(axis=(0, 1)), arrays[1].min(axis=(0, 1))
    ).astype(np.float64)
    world_upper = np.maximum(
        arrays[0].max(axis=(0, 1)), arrays[1].max(axis=(0, 1))
    ).astype(np.float64)
    target = (world_lower + world_upper) / 2.0
    world_extent = np.maximum(world_upper - world_lower, 0.35)
    eye = target + _camera_direction(camera_elev, camera_azim) * (
        float(world_extent.max()) * 4.0
    )
    rotation, translation = _look_at_rotation(
        eye, target, np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    )

    view_lower = np.full(3, np.inf, dtype=np.float64)
    view_upper = np.full(3, -np.inf, dtype=np.float64)
    for array in arrays:
        flat = array.reshape(-1, 3)
        for offset in range(0, flat.shape[0], 250_000):
            view = flat[offset : offset + 250_000] @ rotation + translation
            view_lower = np.minimum(view_lower, view.min(axis=0))
            view_upper = np.maximum(view_upper, view.max(axis=0))

    view_center = (view_lower[:2] + view_upper[:2]) / 2.0
    view_span = np.maximum(view_upper[:2] - view_lower[:2], 0.35) * padding
    aspect = float(width) / float(height)
    if view_span[0] / view_span[1] < aspect:
        view_span[0] = view_span[1] * aspect
    else:
        view_span[1] = view_span[0] / aspect
    znear = max(0.01, float(view_lower[2]) * 0.5)
    zfar = max(znear + 1.0, float(view_upper[2]) * 1.5)
    return {
        "projection": "orthographic",
        "elev_degrees": float(camera_elev),
        "azim_degrees": float(camera_azim),
        "padding": float(padding),
        "eye": eye.tolist(),
        "target": target.tolist(),
        "up": [0.0, 1.0, 0.0],
        "rotation": rotation.tolist(),
        "translation": translation.tolist(),
        "min_x": float(view_center[0] - view_span[0] / 2.0),
        "max_x": float(view_center[0] + view_span[0] / 2.0),
        "min_y": float(view_center[1] - view_span[1] / 2.0),
        "max_y": float(view_center[1] + view_span[1] / 2.0),
        "znear": znear,
        "zfar": zfar,
        "world_bounds": [world_lower.tolist(), world_upper.tolist()],
        "view_bounds_before_padding": [view_lower.tolist(), view_upper.tolist()],
    }


def _ffmpeg_command(
    ffmpeg: str,
    output: Path,
    *,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
) -> list[str]:
    return [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s:v",
        "%dx%d" % (width, height),
        "-r",
        "%g" % fps,
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        preset,
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]


def _probe_video(ffprobe: str, path: Path) -> Dict[str, Any]:
    completed = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=codec_name,pix_fmt,width,height,avg_frame_rate,nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise VideoRenderError("ffprobe failed: %s" % completed.stderr.strip())
    try:
        streams = json.loads(completed.stdout)["streams"]
        if len(streams) != 1:
            raise ValueError("expected one video stream")
        stream = streams[0]
        numerator, denominator = stream["avg_frame_rate"].split("/", 1)
        actual_fps = float(numerator) / float(denominator)
        return {
            "codec_name": str(stream["codec_name"]),
            "pixel_format": str(stream["pix_fmt"]),
            "width": int(stream["width"]),
            "height": int(stream["height"]),
            "fps": actual_fps,
            "frame_count": int(stream["nb_frames"]),
            "duration_seconds": float(stream["duration"]),
        }
    except (KeyError, TypeError, ValueError, ZeroDivisionError, json.JSONDecodeError) as exc:
        raise VideoRenderError("ffprobe returned incomplete video metadata") from exc


def _promote_no_replace(source: Path, destination: Path) -> None:
    """Atomically publish a same-filesystem temporary file without clobbering."""

    try:
        os.link(source, destination)
    except FileExistsError as exc:
        raise VideoRenderError("refusing to overwrite output %s" % destination) from exc
    source.unlink()


def _render_frames_to_ffmpeg(
    human_vertices: np.ndarray,
    human_faces: np.ndarray,
    object_vertices: np.ndarray,
    object_faces: np.ndarray,
    *,
    camera: Mapping[str, Any],
    output: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    preset: str,
    ffmpeg: str,
    progress_every: int,
) -> None:
    try:
        from pytorch3d.renderer import (
            BlendParams,
            FoVOrthographicCameras,
            HardPhongShader,
            Materials,
            MeshRasterizer,
            MeshRenderer,
            PointLights,
            RasterizationSettings,
            TexturesVertex,
        )
        from pytorch3d.structures import Meshes, join_meshes_as_scene
    except ImportError as exc:
        raise VideoRenderError("PyTorch3D is required for headless video rendering") from exc

    device = torch.device("cpu")
    human_tensor = torch.from_numpy(np.asarray(human_vertices, dtype=np.float32))
    object_tensor = torch.from_numpy(np.asarray(object_vertices, dtype=np.float32))
    human_face_tensor = torch.from_numpy(np.asarray(human_faces, dtype=np.int64))
    object_face_tensor = torch.from_numpy(np.asarray(object_faces, dtype=np.int64))
    rotation = torch.tensor(camera["rotation"], dtype=torch.float32)[None]
    translation = torch.tensor(camera["translation"], dtype=torch.float32)[None]
    cameras = FoVOrthographicCameras(
        R=rotation,
        T=translation,
        znear=float(camera["znear"]),
        zfar=float(camera["zfar"]),
        min_x=float(camera["min_x"]),
        max_x=float(camera["max_x"]),
        min_y=float(camera["min_y"]),
        max_y=float(camera["max_y"]),
        device=device,
    )
    lights = PointLights(
        ambient_color=((0.58, 0.58, 0.58),),
        diffuse_color=((0.42, 0.42, 0.42),),
        specular_color=((0.06, 0.06, 0.06),),
        location=(tuple(camera["eye"]),),
        device=device,
    )
    materials = Materials(shininess=24.0, device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(
            cameras=cameras,
            raster_settings=RasterizationSettings(
                image_size=(height, width),
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=None,
                cull_backfaces=False,
            ),
        ),
        shader=HardPhongShader(
            device=device,
            cameras=cameras,
            lights=lights,
            materials=materials,
            blend_params=BlendParams(background_color=(1.0, 1.0, 1.0)),
        ),
    )
    human_color = torch.tensor([0.26, 0.52, 0.86], dtype=torch.float32)
    object_color = torch.tensor([0.34, 0.30, 0.25], dtype=torch.float32)
    command = _ffmpeg_command(
        ffmpeg,
        output,
        width=width,
        height=height,
        fps=fps,
        crf=crf,
        preset=preset,
    )
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
    )
    try:
        if process.stdin is None:
            raise VideoRenderError("cannot open FFmpeg input stream")
        with torch.no_grad():
            for frame in range(human_tensor.shape[0]):
                human_texture = TexturesVertex(
                    verts_features=human_color.view(1, 1, 3).expand(
                        1, human_tensor.shape[1], 3
                    )
                )
                object_texture = TexturesVertex(
                    verts_features=object_color.view(1, 1, 3).expand(
                        1, object_tensor.shape[1], 3
                    )
                )
                human_mesh = Meshes(
                    verts=[human_tensor[frame]],
                    faces=[human_face_tensor],
                    textures=human_texture,
                )
                object_mesh = Meshes(
                    verts=[object_tensor[frame]],
                    faces=[object_face_tensor],
                    textures=object_texture,
                )
                scene = join_meshes_as_scene([human_mesh, object_mesh])
                image = renderer(scene)[0, :, :, :3].clamp(0.0, 1.0)
                process.stdin.write(
                    (image.mul(255.0).byte().cpu().numpy()).tobytes()
                )
                if progress_every > 0 and (
                    (frame + 1) % progress_every == 0
                    or frame + 1 == human_tensor.shape[0]
                ):
                    print(
                        "rendered %d/%d frames" % (frame + 1, human_tensor.shape[0]),
                        flush=True,
                    )
        process.stdin.close()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        returncode = process.wait()
        if returncode != 0:
            raise VideoRenderError("FFmpeg encoding failed: %s" % stderr.strip())
    except BaseException as exc:
        try:
            if process.stdin is not None and not process.stdin.closed:
                process.stdin.close()
        except OSError:
            pass
        try:
            if process.poll() is None:
                process.kill()
        finally:
            process.wait()
        stderr = process.stderr.read().decode("utf-8", errors="replace") if process.stderr else ""
        if isinstance(exc, (BrokenPipeError, OSError)):
            raise VideoRenderError("FFmpeg stream failed: %s" % stderr.strip()) from exc
        raise


def render_motion_video(
    motion_path: Path | str,
    *,
    output_path: Path | str,
    smpl_models: Path | str,
    object_mesh: Path | str,
    object_rest_frame: str = "z_up",
    object_geometry: str = "full",
    manifest_path: Optional[Path | str] = None,
    render_manifest_path: Optional[Path | str] = None,
    width: int = 640,
    height: int = 480,
    fps: float = 30.0,
    crf: int = 18,
    preset: str = "medium",
    renderer_commit: str = "local-unrecorded",
    hand_pose_fallback: str = "mean",
    camera_elev: float = 18.0,
    camera_azim: float = -58.0,
    camera_padding: float = 1.10,
    progress_every: int = 10,
) -> Dict[str, Any]:
    """Render all frames of one canonical HOI/HOSI motion as a verified MP4."""

    motion = Path(motion_path)
    output = Path(output_path)
    models = Path(smpl_models)
    object_asset = Path(object_mesh)
    source_manifest = Path(manifest_path) if manifest_path is not None else None
    render_manifest = (
        Path(render_manifest_path)
        if render_manifest_path is not None
        else output.with_suffix(".render.json")
    )
    _validate_video_settings(width=width, height=height, fps=fps, crf=crf)
    _validate_video_targets(output, render_manifest)
    if preset not in ("ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"):
        raise VideoRenderError("unsupported FFmpeg preset %s" % preset)
    if progress_every < 0:
        raise VideoRenderError("progress interval cannot be negative")
    if not models.is_dir():
        raise VideoRenderError("SMPL-X model directory does not exist: %s" % models)

    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if ffmpeg is None or ffprobe is None:
        raise VideoRenderError("system FFmpeg and ffprobe are required")
    summary = validate_motion_export(motion, manifest_path=source_manifest)
    try:
        with np.load(motion, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError, TypeError) as exc:
        raise VideoRenderError("cannot reload motion export") from exc
    if summary["task_family"] not in ("hoi", "hosi"):
        raise VideoRenderError("mesh video renderer currently expects HOI/HOSI object motion")
    frame_count = int(summary["pose_frames"])
    selected = np.arange(frame_count, dtype=np.int64)
    human_vertices, _, human_faces, hand_pose_source = _restore_human_mesh(
        data, models, selected, "cpu", hand_pose_fallback
    )
    object_rest_vertices, object_faces = _load_object_mesh(
        object_asset, coordinate_frame=object_rest_frame
    )
    if object_geometry == "convex_hull":
        try:
            proxy = trimesh.Trimesh(
                vertices=object_rest_vertices, faces=object_faces, process=False
            ).convex_hull
            object_rest_vertices = np.asarray(proxy.vertices, dtype=np.float32)
            object_faces = np.asarray(proxy.faces, dtype=np.int64)
        except Exception as exc:
            raise VideoRenderError("cannot construct object convex-hull proxy") from exc
    elif object_geometry != "full":
        raise VideoRenderError("unsupported object geometry mode %s" % object_geometry)
    object_trans = np.asarray(data["object_trans"], dtype=np.float32)
    object_rot = np.asarray(data["object_rot_mat"], dtype=np.float32)
    if object_trans.shape[0] != frame_count or object_rot.shape[0] != frame_count:
        raise VideoRenderError("object stream must have one pose for each human frame")
    object_vertices = np.einsum(
        "fij,vj->fvi", object_rot, object_rest_vertices
    ) + object_trans[:, None, :]
    if human_vertices.shape[0] != frame_count or object_vertices.shape[0] != frame_count:
        raise VideoRenderError("reconstructed mesh streams do not match source frame count")
    camera = _fixed_orthographic_camera(
        human_vertices,
        object_vertices,
        width=width,
        height=height,
        camera_elev=camera_elev,
        camera_azim=camera_azim,
        padding=camera_padding,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    render_manifest.parent.mkdir(parents=True, exist_ok=True)
    identity = uuid.uuid4().hex
    temporary_video = output.with_name(".%s.%s.partial.mp4" % (output.stem, identity))
    temporary_manifest = render_manifest.with_name(
        ".%s.%s.partial.json" % (render_manifest.stem, identity)
    )
    try:
        _render_frames_to_ffmpeg(
            human_vertices,
            human_faces,
            object_vertices,
            object_faces,
            camera=camera,
            output=temporary_video,
            width=width,
            height=height,
            fps=fps,
            crf=crf,
            preset=preset,
            ffmpeg=ffmpeg,
            progress_every=progress_every,
        )
        probe = _probe_video(ffprobe, temporary_video)
        if probe["frame_count"] != frame_count:
            raise VideoRenderError(
                "encoded frame count %d does not match source %d"
                % (probe["frame_count"], frame_count)
            )
        if probe["width"] != width or probe["height"] != height:
            raise VideoRenderError("encoded video dimensions do not match the request")
        if not math.isclose(float(probe["fps"]), fps, rel_tol=0.0, abs_tol=1e-6):
            raise VideoRenderError("encoded video FPS does not match the request")
        source_manifest_sha256 = (
            _sha256(source_manifest) if source_manifest is not None else "absent"
        )
        try:
            import pytorch3d

            pytorch3d_version = str(pytorch3d.__version__)
        except (ImportError, AttributeError):
            pytorch3d_version = "unknown"
        render_record = {
            "source_motion_sha256": _sha256(motion),
            "source_manifest_sha256": source_manifest_sha256,
            "renderer_commit": renderer_commit,
            "renderer_backend": "pytorch3d-cpu-ffmpeg-libx264",
            "pytorch_version": str(torch.__version__),
            "pytorch3d_version": pytorch3d_version,
            "hand_pose_source": hand_pose_source,
            "hand_pose_fallback": hand_pose_fallback,
            "smpl_asset_path": str(models.resolve()),
            "smpl_asset_sha256": _tree_sha256(models),
            "object_asset_path": str(object_asset.resolve()),
            "object_asset_sha256": _sha256(object_asset),
            "object_rest_frame": object_rest_frame,
            "object_geometry": object_geometry,
            "camera": camera,
            "source_frame_indices": selected.tolist(),
            "video_dimensions": [width, height],
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": probe["duration_seconds"],
            "encoder": {
                "codec": "libx264",
                "crf": crf,
                "preset": preset,
                "pixel_format": "yuv420p",
            },
            "ffprobe": probe,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "output_sha256": _sha256(temporary_video),
            **_timeline_manifest_fields(data, selected),
        }
        temporary_manifest.write_text(
            json.dumps(render_record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        _promote_no_replace(temporary_video, output)
        _promote_no_replace(temporary_manifest, render_manifest)
    finally:
        temporary_video.unlink(missing_ok=True)
        temporary_manifest.unlink(missing_ok=True)
    return {
        **summary,
        "output_path": str(output),
        "render_manifest_path": str(render_manifest),
        "frame_count": frame_count,
        "fps": fps,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render a canonical HOI/HOSI motion export to MP4 headlessly"
    )
    parser.add_argument("motion", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpl-models", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--object-rest-frame", choices=("z_up", "y_up"), default="z_up")
    parser.add_argument("--object-geometry", choices=("convex_hull", "full"), default="full")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--render-manifest", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--crf", type=int, default=18)
    parser.add_argument("--preset", default="medium")
    parser.add_argument("--renderer-commit", default="local-unrecorded")
    parser.add_argument("--hand-pose-fallback", choices=("mean", "flat"), default="mean")
    parser.add_argument("--camera-elev", type=float, default=18.0)
    parser.add_argument("--camera-azim", type=float, default=-58.0)
    parser.add_argument("--camera-padding", type=float, default=1.10)
    parser.add_argument("--progress-every", type=int, default=10)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = render_motion_video(
            args.motion,
            output_path=args.output,
            smpl_models=args.smpl_models,
            object_mesh=args.object_mesh,
            object_rest_frame=args.object_rest_frame,
            object_geometry=args.object_geometry,
            manifest_path=args.manifest,
            render_manifest_path=args.render_manifest,
            width=args.width,
            height=args.height,
            fps=args.fps,
            crf=args.crf,
            preset=args.preset,
            renderer_commit=args.renderer_commit,
            hand_pose_fallback=args.hand_pose_fallback,
            camera_elev=args.camera_elev,
            camera_azim=args.camera_azim,
            camera_padding=args.camera_padding,
            progress_every=args.progress_every,
        )
    except (VideoRenderError, HeadlessRenderError, MotionExportError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
