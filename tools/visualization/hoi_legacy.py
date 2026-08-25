"""Read-only adapter for the released HOI ``motion_params`` pickle.

The HOI evaluator currently writes a legacy pickle containing one or more
sample trajectories.  This module is the one model-dependent compatibility
step: it reads that file, selects one sample, and writes the canonical NPZ
motion artifact.  Renderers must consume the NPZ and never import this module.

Pickle is executable serialization.  The adapter is therefore intended only
for trusted, locally produced legacy files; all downstream readers use
``allow_pickle=False``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from .schema import MotionExportError, SCHEMA_VERSION, validate_motion_export


class HOILegacyExportError(MotionExportError):
    """Raised when a legacy HOI pickle cannot be converted unambiguously."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HOILegacyExportError("%s must be a mapping" % name)
    return value


def _numeric(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.issubdtype(array.dtype, np.number):
        raise HOILegacyExportError("%s must be numeric" % name)
    if not np.isfinite(array).all():
        raise HOILegacyExportError("%s contains non-finite values" % name)
    return array


def _as_nonempty_text(value: Any, name: str) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if not isinstance(value, str) or not value:
        raise HOILegacyExportError("%s must be a non-empty string" % name)
    return value


def _zup_to_yup(value: np.ndarray) -> np.ndarray:
    """Convert vectors stored by the legacy exporter from z-up to y-up."""

    converted = np.asarray(value).copy()
    converted = converted[..., [0, 2, 1]]
    converted[..., 2] *= -1
    return converted


def _select_text(value: Any, sample_index: int, sample_count: int, name: str) -> str:
    if isinstance(value, (list, tuple, np.ndarray)):
        values = list(value)
        if len(values) != sample_count:
            raise HOILegacyExportError(
                "%s has %d entries but the pickle has %d samples"
                % (name, len(values), sample_count)
            )
        value = values[sample_index]
    return _as_nonempty_text(value, name)


def _normalise_human(
    pose_value: Any,
    root_value: Any,
    *,
    sample_count: int,
    frames: int,
) -> Tuple[np.ndarray, np.ndarray]:
    pose = _numeric(pose_value, "human_motion.pose_pred")
    if pose.ndim == 3:
        if pose.shape[1:] != (22, 3) or pose.shape[0] != sample_count * frames:
            raise HOILegacyExportError(
                "human_motion.pose_pred must have shape [%d,22,3] or [%d,%d,22,3]"
                % (sample_count * frames, sample_count, frames)
            )
        pose = pose.reshape(sample_count, frames, 22, 3)
    elif pose.ndim == 4:
        if pose.shape != (sample_count, frames, 22, 3):
            raise HOILegacyExportError(
                "human_motion.pose_pred has unexpected shape %s" % (pose.shape,)
            )
    else:
        raise HOILegacyExportError("human_motion.pose_pred must be rank 3 or 4")

    root = _numeric(root_value, "human_motion.root_trans")
    if root.ndim == 2:
        if root.shape != (sample_count * frames, 3):
            raise HOILegacyExportError(
                "human_motion.root_trans must have shape [%d,3] or [%d,%d,3]"
                % (sample_count * frames, sample_count, frames)
            )
        root = root.reshape(sample_count, frames, 3)
    elif root.ndim == 3:
        if root.shape != (sample_count, frames, 3):
            raise HOILegacyExportError(
                "human_motion.root_trans has unexpected shape %s" % (root.shape,)
            )
    else:
        raise HOILegacyExportError("human_motion.root_trans must be rank 2 or 3")
    return pose.astype(np.float32), root.astype(np.float32)


def _normalise_object(
    trans_value: Any,
    rot_value: Any,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    trans = _numeric(trans_value, "object_motion.obj_trans")
    if trans.ndim == 2:
        if trans.shape[1:] != (3,):
            raise HOILegacyExportError("object_motion.obj_trans must end in [3]")
        trans = trans[None, ...]
    elif trans.ndim == 3 and trans.shape[2:] == (3,):
        pass
    else:
        raise HOILegacyExportError(
            "object_motion.obj_trans must have shape [F,3] or [S,F,3]"
        )
    sample_count, frames = int(trans.shape[0]), int(trans.shape[1])

    rotation = _numeric(rot_value, "object_motion.obj_rot_mat")
    if rotation.ndim == 2 and rotation.shape[1:] == (9,):
        rotation = rotation[None, ...]
    elif rotation.ndim == 3 and rotation.shape[2:] == (9,):
        pass
    elif rotation.ndim == 3 and rotation.shape[1:] == (3, 3):
        rotation = rotation[None, ...]
    elif rotation.ndim == 4 and rotation.shape[2:] == (3, 3):
        pass
    else:
        raise HOILegacyExportError(
            "object_motion.obj_rot_mat must have shape [F,9], [S,F,9], "
            "[F,3,3], or [S,F,3,3]"
        )

    if rotation.shape[0] != sample_count or rotation.shape[1] != frames:
        raise HOILegacyExportError(
            "object translation and rotation sample/frame counts differ"
        )
    if rotation.shape[-1] == 9:
        rotation = rotation.reshape(sample_count, frames, 3, 3)
    return trans.astype(np.float32), rotation.astype(np.float32), sample_count, frames


def _select_betas(value: Any, sample_index: int, sample_count: int) -> np.ndarray:
    betas = _numeric(value, "human_motion.betas")
    if betas.ndim == 1:
        selected = betas
    elif betas.ndim == 2 and betas.shape[0] == sample_count:
        selected = betas[sample_index]
    elif betas.ndim == 2 and betas.shape[0] == 1:
        selected = betas[0]
    else:
        raise HOILegacyExportError(
            "human_motion.betas must have shape [B] or [S,B]"
        )
    if selected.ndim != 1 or selected.shape[0] <= 0:
        raise HOILegacyExportError("human_motion.betas must contain shape coefficients")
    return selected.astype(np.float32)


def load_legacy_pickle(path: Path | str) -> Mapping[str, Any]:
    """Load one trusted legacy pickle without importing any expert package."""

    source = Path(path)
    if not source.is_file():
        raise HOILegacyExportError("legacy pickle does not exist: %s" % source)
    if source.suffix.lower() != ".pkl":
        raise HOILegacyExportError("legacy HOI input must have a .pkl suffix")
    try:
        with source.open("rb") as handle:
            value = pickle.load(handle)
    except (OSError, EOFError, pickle.PickleError, ValueError, TypeError) as exc:
        raise HOILegacyExportError("cannot load legacy pickle %s" % source) from exc
    return _require_mapping(value, "legacy pickle")


def legacy_to_payload(
    path: Path | str,
    *,
    sample_index: int = 0,
    fps: float = 30.0,
    coordinate_frame: str = "infbagel_y_up",
    legacy_human_frame: str = "z_up",
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Convert one selected legacy HOI sample into canonical NPZ fields.

    The returned metadata records the legacy sample count/index so a renderer
    cannot accidentally treat concatenated candidates as one long trajectory.
    """

    if isinstance(sample_index, bool) or int(sample_index) != sample_index:
        raise HOILegacyExportError("sample_index must be an integer")
    sample_index = int(sample_index)
    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise HOILegacyExportError("fps must be numeric") from exc
    if not np.isfinite(fps) or fps <= 0:
        raise HOILegacyExportError("fps must be finite and positive")
    coordinate_frame = _as_nonempty_text(coordinate_frame, "coordinate_frame")
    if legacy_human_frame not in ("z_up", "y_up"):
        raise HOILegacyExportError(
            "legacy_human_frame must be 'z_up' or 'y_up'"
        )

    source = Path(path)
    root = load_legacy_pickle(source)
    sequence_id = _as_nonempty_text(root.get("seq_name"), "seq_name")
    human = _require_mapping(root.get("human_motion"), "human_motion")
    object_motion = _require_mapping(root.get("object_motion"), "object_motion")
    object_trans, object_rot, sample_count, frames = _normalise_object(
        object_motion.get("obj_trans"), object_motion.get("obj_rot_mat")
    )
    if sample_index < 0 or sample_index >= sample_count:
        raise HOILegacyExportError(
            "sample_index %d is outside [0,%d)" % (sample_index, sample_count)
        )
    pose, root_trans = _normalise_human(
        human.get("pose_pred"),
        human.get("root_trans"),
        sample_count=sample_count,
        frames=frames,
    )
    if legacy_human_frame == "z_up":
        # The released exporter called yup_to_zup on both axis-angle vectors
        # and translations immediately before writing the pickle.  Object
        # translations were not passed through that conversion and remain in
        # the y-up world.  Undo only that legacy human-side conversion here.
        pose = _zup_to_yup(pose)
        root_trans = _zup_to_yup(root_trans)
    gender = _select_text(human.get("gender"), sample_index, sample_count, "gender")
    object_name = _select_text(
        object_motion.get("obj_name"), sample_index, sample_count, "object_name"
    )

    payload = {
        "schema_version": np.asarray(SCHEMA_VERSION, dtype=np.int32),
        "sequence_id": np.asarray(sequence_id),
        "task_family": np.asarray("hoi"),
        "fps": np.asarray(fps, dtype=np.float32),
        "coordinate_frame": np.asarray(coordinate_frame),
        "global_orient": pose[sample_index, :, 0],
        "body_pose": pose[sample_index, :, 1:],
        "transl": root_trans[sample_index],
        "betas": _select_betas(human.get("betas"), sample_index, sample_count),
        "gender": np.asarray(gender),
        "object_name": np.asarray(object_name),
        "object_trans": object_trans[sample_index],
        "object_rot_mat": object_rot[sample_index],
        "legacy_sample_index": np.asarray(sample_index, dtype=np.int32),
        "legacy_sample_count": np.asarray(sample_count, dtype=np.int32),
    }
    metadata = {
        "legacy_source_path": str(source.resolve()),
        "legacy_source_sha256": _sha256(source),
        "legacy_sample_index": sample_index,
        "legacy_sample_count": sample_count,
        "legacy_frame_count": frames,
        "legacy_human_frame": legacy_human_frame,
        "human_frame_conversion": (
            "z_up_to_y_up" if legacy_human_frame == "z_up" else "none"
        ),
        "adapter": "hoi_legacy_v1",
    }
    return payload, metadata


def write_motion_npz(path: Path | str, payload: Mapping[str, np.ndarray]) -> Path:
    """Write a new NPZ artifact and refuse to overwrite an existing one."""

    destination = Path(path)
    if destination.suffix.lower() != ".npz":
        raise HOILegacyExportError("motion output must have a .npz suffix")
    if destination.exists():
        raise HOILegacyExportError("refusing to overwrite motion export %s" % destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    try:
        with temporary.open("wb") as handle:
            np.savez(handle, **payload)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def write_legacy_manifest(
    path: Path | str,
    *,
    motion_path: Path,
    source_path: Path,
    metadata: Mapping[str, Any],
    command: Optional[str] = None,
) -> Path:
    """Write an explicit legacy provenance manifest beside a new NPZ.

    Existing pickles predate the reportable run manifest.  Unknown provenance
    is labelled ``legacy-unrecorded`` rather than presented as a verified
    checkpoint/config hash.  Such an artifact is suitable for a visual smoke,
    not for a scientific result table.
    """

    destination = Path(path)
    if destination.exists():
        raise HOILegacyExportError("refusing to overwrite manifest %s" % destination)
    unknown = "legacy-unrecorded"
    manifest = {
        "export_schema_version": SCHEMA_VERSION,
        "source_git_commit": unknown,
        "source_live_head_at_completion": unknown,
        "resolved_config_sha256": unknown,
        "checkpoint_path_and_sha256": unknown,
        "dataset_snapshot_and_sha256": unknown,
        "smpl_models_sha256": unknown,
        "object_asset_manifest_sha256": unknown,
        "scene_asset_manifest_sha256": unknown,
        "command": command or "legacy HOI adapter",
        "working_directory": str(Path.cwd()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion_sha256": _sha256(motion_path),
        **dict(metadata),
        "legacy_source_path": str(source_path.resolve()),
        "legacy_source_sha256": _sha256(source_path),
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return destination


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert one legacy HOI motion pickle to NPZ")
    parser.add_argument("input", type=Path, help="trusted *_motion_params.pkl")
    parser.add_argument("--output", type=Path, required=True, help="new canonical .npz path")
    parser.add_argument("--manifest", type=Path, default=None, help="optional output provenance JSON")
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--coordinate-frame", default="infbagel_y_up")
    parser.add_argument(
        "--legacy-human-frame",
        choices=("z_up", "y_up"),
        default="z_up",
        help="frame of human fields in the old pickle; old released exports use z_up",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        payload, metadata = legacy_to_payload(
            args.input,
            sample_index=args.sample_index,
            fps=args.fps,
            coordinate_frame=args.coordinate_frame,
            legacy_human_frame=args.legacy_human_frame,
        )
        motion_path = write_motion_npz(args.output, payload)
        manifest_path = None
        if args.manifest is not None:
            manifest_path = write_legacy_manifest(
                args.manifest,
                motion_path=motion_path,
                source_path=args.input,
                metadata=metadata,
                command=" ".join([sys.executable, "-m", "tools.visualization.hoi_legacy"] + list(sys.argv[1:])),
            )
        summary = validate_motion_export(motion_path, manifest_path=manifest_path)
        summary.update(metadata)
        summary["motion_path"] = str(motion_path)
        if manifest_path is not None:
            summary["manifest_path"] = str(manifest_path)
    except (HOILegacyExportError, MotionExportError, OSError, ValueError) as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
