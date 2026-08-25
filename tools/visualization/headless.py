"""CPU/headless keyframe renderer for canonical motion exports.

This renderer intentionally consumes only a validated NPZ, SMPL-X assets and a
rest object mesh.  It uses Matplotlib's Agg backend, so it works on the Linux
headless server without Blender, EGL, or an X display.  It is a deterministic
debug/paper-figure renderer, not a replacement for a furnished Blender scene.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import smplx
import torch
import trimesh
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .schema import MotionExportError, validate_motion_export


class HeadlessRenderError(MotionExportError):
    """Raised when a canonical motion cannot be rendered safely."""


PALETTE = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    for path in files:
        digest.update(str(path.relative_to(root)).encode("utf-8"))
        digest.update(_sha256(path).encode("ascii"))
    return digest.hexdigest()


def _project_y_up(points: np.ndarray) -> np.ndarray:
    """Map x/y-up/z world coordinates to Matplotlib x/y/z plot axes."""

    return np.asarray(points)[..., [0, 2, 1]]


def _parse_frames(value: Optional[str], frame_count: int, keyframe_count: int) -> np.ndarray:
    if value:
        try:
            frames = np.asarray([int(item.strip()) for item in value.split(",")], dtype=np.int64)
        except ValueError as exc:
            raise HeadlessRenderError("--frames must be comma-separated integers") from exc
        if frames.size == 0:
            raise HeadlessRenderError("--frames must contain at least one index")
    else:
        if keyframe_count < 1:
            raise HeadlessRenderError("--keyframe-count must be positive")
        frames = np.rint(np.linspace(0, frame_count - 1, keyframe_count)).astype(np.int64)
        frames = np.unique(frames)
    if (frames < 0).any() or (frames >= frame_count).any():
        raise HeadlessRenderError("selected frame is outside the motion export")
    return frames


def _load_object_mesh(path: Path, *, coordinate_frame: str) -> Tuple[np.ndarray, np.ndarray]:
    if not path.is_file():
        raise HeadlessRenderError("object rest mesh does not exist: %s" % path)
    try:
        mesh = trimesh.load_mesh(path, process=False)
    except Exception as exc:
        raise HeadlessRenderError("cannot load object rest mesh %s" % path) from exc
    if not isinstance(mesh, trimesh.Trimesh):
        raise HeadlessRenderError("object asset must contain one triangular mesh: %s" % path)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or faces.ndim != 2 or faces.shape[1] != 3:
        raise HeadlessRenderError("object rest mesh has invalid vertex/face arrays")
    if coordinate_frame == "z_up":
        vertices = vertices[:, [0, 2, 1]]
        vertices[:, 2] *= -1
    elif coordinate_frame != "y_up":
        raise HeadlessRenderError("unsupported object rest coordinate frame %s" % coordinate_frame)
    return vertices, faces


def _restore_human_mesh(
    data: Mapping[str, np.ndarray], smpl_models: Path, frames: np.ndarray, device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    gender = str(np.asarray(data["gender"]).item())
    betas = np.asarray(data["betas"], dtype=np.float32)
    if betas.ndim != 1 or betas.size == 0:
        raise HeadlessRenderError("betas must be a non-empty vector")
    try:
        model = smplx.create(
            str(smpl_models),
            model_type="smplx",
            gender=gender,
            num_betas=int(betas.size),
            use_pca=False,
            flat_hand_mean=True,
            batch_size=int(frames.size),
        ).to(device)
    except Exception as exc:
        raise HeadlessRenderError("cannot create SMPL-X model from %s" % smpl_models) from exc

    global_orient = torch.from_numpy(np.asarray(data["global_orient"], dtype=np.float32)[frames]).to(device)
    body_pose = torch.from_numpy(np.asarray(data["body_pose"], dtype=np.float32)[frames].reshape(frames.size, -1)).to(device)
    transl = torch.from_numpy(np.asarray(data["transl"], dtype=np.float32)[frames]).to(device)
    beta_batch = torch.from_numpy(np.repeat(betas[None], frames.size, axis=0)).to(device)
    zeros = lambda width: torch.zeros((frames.size, width), dtype=global_orient.dtype, device=device)
    with torch.no_grad():
        try:
            output = model(
                global_orient=global_orient,
                body_pose=body_pose,
                betas=beta_batch,
                transl=transl,
                left_hand_pose=zeros(45),
                right_hand_pose=zeros(45),
                jaw_pose=zeros(3),
                leye_pose=zeros(3),
                reye_pose=zeros(3),
                expression=zeros(int(model.expression.shape[1])),
                return_verts=True,
            )
        except Exception as exc:
            raise HeadlessRenderError("SMPL-X forward pass failed") from exc
    vertices = output.vertices.detach().cpu().numpy().astype(np.float32)
    pelvis = output.joints[:, 0].detach().cpu().numpy().astype(np.float32)
    faces = np.asarray(model.faces, dtype=np.int64)
    return vertices, pelvis, faces


def _face_subset(faces: np.ndarray, max_faces: int) -> np.ndarray:
    if max_faces <= 0 or faces.shape[0] <= max_faces:
        return faces
    stride = int(math.ceil(faces.shape[0] / max_faces))
    return faces[::stride]


def _add_mesh(
    axis: Any,
    vertices: np.ndarray,
    faces: np.ndarray,
    *,
    color: str,
    alpha: float,
    max_faces: int,
) -> None:
    projected = _project_y_up(vertices)
    selected = _face_subset(faces, max_faces)
    triangles = projected[selected]
    axis.add_collection3d(
        Poly3DCollection(
            triangles,
            facecolor=color,
            edgecolor="none",
            linewidths=0.0,
            alpha=alpha,
        )
    )


def render_keyframes(
    motion_path: Path | str,
    *,
    output_path: Path | str,
    smpl_models: Path | str,
    object_mesh: Path | str,
    object_rest_frame: str = "z_up",
    object_geometry: str = "convex_hull",
    manifest_path: Optional[Path | str] = None,
    render_manifest_path: Optional[Path | str] = None,
    frames: Optional[Sequence[int]] = None,
    keyframe_count: int = 4,
    device: str = "cpu",
    dpi: int = 180,
    max_faces: int = 6000,
    renderer_commit: str = "local-unrecorded",
) -> Dict[str, Any]:
    """Render selected human/object poses into one deterministic PNG."""

    motion = Path(motion_path)
    output = Path(output_path)
    models = Path(smpl_models)
    object_asset = Path(object_mesh)
    if output.suffix.lower() != ".png":
        raise HeadlessRenderError("headless output must have a .png suffix")
    if output.exists():
        raise HeadlessRenderError("refusing to overwrite render %s" % output)
    if not models.is_dir():
        raise HeadlessRenderError("SMPL-X model directory does not exist: %s" % models)
    render_manifest = (
        Path(render_manifest_path)
        if render_manifest_path is not None
        else output.with_suffix(".render.json")
    )
    if render_manifest.exists():
        raise HeadlessRenderError("refusing to overwrite render manifest %s" % render_manifest)
    summary = validate_motion_export(motion, manifest_path=manifest_path)
    try:
        with np.load(motion, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError, TypeError) as exc:
        raise HeadlessRenderError("cannot reload motion export") from exc
    if summary["task_family"] not in ("hoi", "hosi"):
        raise HeadlessRenderError("headless mesh renderer currently expects HOI/HOSI object motion")
    frame_count = int(summary["pose_frames"])
    if frames is None:
        selected = _parse_frames(None, frame_count, keyframe_count)
    else:
        selected = _parse_frames(",".join(str(int(item)) for item in frames), frame_count, keyframe_count)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise HeadlessRenderError("requested CUDA but torch.cuda.is_available() is false")
    human_vertices, pelvis, human_faces = _restore_human_mesh(data, models, selected, device)
    object_rest_vertices, object_faces = _load_object_mesh(object_asset, coordinate_frame=object_rest_frame)
    if object_geometry == "convex_hull":
        try:
            proxy = trimesh.Trimesh(
                vertices=object_rest_vertices, faces=object_faces, process=False
            ).convex_hull
            object_rest_vertices = np.asarray(proxy.vertices, dtype=np.float32)
            object_faces = np.asarray(proxy.faces, dtype=np.int64)
        except Exception as exc:
            raise HeadlessRenderError("cannot construct object convex-hull proxy") from exc
    elif object_geometry != "full":
        raise HeadlessRenderError("unsupported object geometry mode %s" % object_geometry)
    object_trans = np.asarray(data["object_trans"], dtype=np.float32)
    object_rot = np.asarray(data["object_rot_mat"], dtype=np.float32)
    if object_trans.shape[0] != frame_count or object_rot.shape[0] != frame_count:
        raise HeadlessRenderError("object stream must have one pose for each human frame")
    object_vertices = np.einsum(
        "fij,vj->fvi", object_rot[selected], object_rest_vertices
    ) + object_trans[selected, None, :]
    all_points = np.concatenate(
        [human_vertices.reshape(-1, 3), object_vertices.reshape(-1, 3), pelvis, object_trans], axis=0
    )
    projected_all = _project_y_up(all_points)
    lower = projected_all.min(axis=0)
    upper = projected_all.max(axis=0)
    center = (lower + upper) / 2.0
    radius = max(float((upper - lower).max()) / 2.0, 0.5) * 1.12

    output.parent.mkdir(parents=True, exist_ok=True)
    figure = plt.figure(figsize=(7.2, 7.2), dpi=dpi)
    axis = figure.add_subplot(111, projection="3d")
    for index, frame in enumerate(selected):
        color = PALETTE[index % len(PALETTE)]
        _add_mesh(
            axis,
            human_vertices[index],
            human_faces,
            color=color,
            alpha=0.32,
            max_faces=max_faces,
        )
        _add_mesh(
            axis,
            object_vertices[index],
            object_faces,
            color="#444444",
            alpha=0.26,
            max_faces=max_faces,
        )
        axis.scatter(*_project_y_up(pelvis[index]), color=color, s=10, depthshade=False)
    trajectory = _project_y_up(object_trans)
    axis.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], color="#444444", linewidth=1.0, alpha=0.65)
    axis.scatter(*trajectory[0], color="black", marker="o", s=22, depthshade=False)
    axis.scatter(*trajectory[-1], color="black", marker="X", s=34, depthshade=False)
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.view_init(elev=18, azim=-58)
    axis.set_xlabel("x")
    axis.set_ylabel("z")
    axis.set_zlabel("y (up)")
    axis.set_title("%s | keyframes %s" % (summary["sequence_id"], ",".join(map(str, selected))))
    legend = [
        Line2D([0], [0], color=PALETTE[index % len(PALETTE)], linewidth=5, label="frame %d" % frame)
        for index, frame in enumerate(selected)
    ]
    legend.append(Line2D([0], [0], color="#444444", linewidth=2, label="object trajectory"))
    axis.legend(handles=legend, loc="upper left", fontsize=8)
    figure.tight_layout()
    try:
        figure.savefig(output, format="png", dpi=dpi, facecolor="white")
    finally:
        plt.close(figure)

    source_manifest_sha256 = "absent"
    if manifest_path is not None:
        source_manifest_sha256 = _sha256(Path(manifest_path))
    render_record = {
        "source_motion_sha256": _sha256(motion),
        "source_manifest_sha256": source_manifest_sha256,
        "renderer_commit": renderer_commit,
        "renderer_backend": "matplotlib-agg-smplx",
        "smpl_asset_path": str(models.resolve()),
        "smpl_asset_sha256": _tree_sha256(models),
        "object_asset_path": str(object_asset.resolve()),
        "object_asset_sha256": _sha256(object_asset),
        "object_rest_frame": object_rest_frame,
        "object_geometry": object_geometry,
        "camera_projection": "orthographic-like fixed 3D view",
        "camera_elev": 18,
        "camera_azim": -58,
        "selected_frame_indices": selected.tolist(),
        "image_dimensions": [int(figure.get_figwidth() * dpi), int(figure.get_figheight() * dpi)],
        "dpi": dpi,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_sha256": _sha256(output),
    }
    render_manifest.parent.mkdir(parents=True, exist_ok=True)
    render_manifest.write_text(json.dumps(render_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {**summary, "output_path": str(output), "render_manifest_path": str(render_manifest), "selected_frame_indices": selected.tolist()}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render a canonical HOI motion export headlessly")
    parser.add_argument("motion", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smpl-models", type=Path, required=True)
    parser.add_argument("--object-mesh", type=Path, required=True)
    parser.add_argument("--object-rest-frame", choices=("z_up", "y_up"), default="z_up")
    parser.add_argument("--object-geometry", choices=("convex_hull", "full"), default="convex_hull")
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--render-manifest", type=Path, default=None)
    parser.add_argument("--frames", default=None, help="comma-separated frame indices")
    parser.add_argument("--keyframe-count", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--max-faces", type=int, default=25000)
    parser.add_argument("--renderer-commit", default="local-unrecorded")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        frames = None if args.frames is None else [int(item.strip()) for item in args.frames.split(",")]
        result = render_keyframes(
            args.motion,
            output_path=args.output,
            smpl_models=args.smpl_models,
            object_mesh=args.object_mesh,
            object_rest_frame=args.object_rest_frame,
            object_geometry=args.object_geometry,
            manifest_path=args.manifest,
            render_manifest_path=args.render_manifest,
            frames=frames,
            keyframe_count=args.keyframe_count,
            device=args.device,
            dpi=args.dpi,
            max_faces=args.max_faces,
            renderer_commit=args.renderer_commit,
        )
    except (HeadlessRenderError, MotionExportError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
