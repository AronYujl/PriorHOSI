"""Validation for the model-independent motion-export contract.

This module intentionally depends only on the Python standard library and
NumPy.  It must remain importable without loading an expert model, a Hydra
configuration, SMPL-X, or a renderer backend.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

import numpy as np


class MotionExportError(ValueError):
    """Raised when an export or its provenance cannot be safely consumed."""


SCHEMA_VERSION = 1
HUMAN_REQUIRED = (
    "schema_version",
    "sequence_id",
    "task_family",
    "fps",
    "coordinate_frame",
    "global_orient",
    "body_pose",
    "transl",
    "betas",
    "gender",
)
OBJECT_REQUIRED = ("object_name", "object_trans", "object_rot_mat")
HAND_POSE_FIELDS = ("left_hand_pose", "right_hand_pose")
MANIFEST_REQUIRED = (
    "export_schema_version",
    "source_git_commit",
    "source_live_head_at_completion",
    "resolved_config_sha256",
    "checkpoint_path_and_sha256",
    "dataset_snapshot_and_sha256",
    "smpl_models_sha256",
    "object_asset_manifest_sha256",
    "scene_asset_manifest_sha256",
    "command",
    "working_directory",
    "created_at",
)
TASK_FAMILIES = frozenset(("hoi", "hsi", "hosi"))


def _scalar(data: Mapping[str, np.ndarray], key: str) -> Any:
    value = np.asarray(data[key])
    if value.ndim != 0:
        raise MotionExportError("%s must be a scalar, got shape %s" % (key, value.shape))
    return value.item()


def _text(data: Mapping[str, np.ndarray], key: str) -> str:
    value = _scalar(data, key)
    if not isinstance(value, (str, bytes, np.str_)):
        raise MotionExportError("%s must be a string scalar" % key)
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value)
    if not value:
        raise MotionExportError("%s must not be empty" % key)
    return value


def _array(
    data: Mapping[str, np.ndarray],
    key: str,
    *,
    ndim: Optional[int] = None,
    shape_tail: Optional[Sequence[int]] = None,
    finite: bool = True,
) -> np.ndarray:
    if key not in data:
        raise MotionExportError("missing required field %s" % key)
    value = np.asarray(data[key])
    if ndim is not None and value.ndim != ndim:
        raise MotionExportError("%s must have ndim=%d, got shape %s" % (key, ndim, value.shape))
    if shape_tail is not None and tuple(value.shape[-len(shape_tail) :]) != tuple(shape_tail):
        raise MotionExportError("%s has invalid shape %s" % (key, value.shape))
    if finite and not np.issubdtype(value.dtype, np.number):
        raise MotionExportError("%s must be numeric" % key)
    if finite and not np.isfinite(value).all():
        raise MotionExportError("%s contains non-finite values" % key)
    return value


def _positive_int(value: Any, key: str) -> int:
    if isinstance(value, bool):
        raise MotionExportError("%s must be a positive integer" % key)
    try:
        integer = int(value)
    except (TypeError, ValueError) as exc:
        raise MotionExportError("%s must be a positive integer" % key) from exc
    if integer <= 0 or float(value) != integer:
        raise MotionExportError("%s must be a positive integer" % key)
    return integer


def _optional_scalar_int(data: Mapping[str, np.ndarray], key: str) -> Optional[int]:
    if key not in data:
        return None
    return _positive_int(_scalar(data, key), key)


def _check_index_vector(data: Mapping[str, np.ndarray], key: str, *, upper: int) -> None:
    if key not in data:
        return
    values = _array(data, key, ndim=1, finite=False)
    if not np.issubdtype(values.dtype, np.integer):
        raise MotionExportError("%s must contain integer indices" % key)
    if (values < 0).any() or (values >= upper).any():
        raise MotionExportError("%s contains an out-of-range index" % key)


def _check_positive_vector(
    data: Mapping[str, np.ndarray], key: str, *, expected_sum: Optional[int] = None
) -> None:
    if key not in data:
        return
    values = _array(data, key, ndim=1, finite=False)
    if not np.issubdtype(values.dtype, np.integer) or (values <= 0).any():
        raise MotionExportError("%s must contain positive integer lengths" % key)
    if expected_sum is not None and int(values.sum()) != expected_sum:
        raise MotionExportError("%s must sum to coarse frame count %d" % (key, expected_sum))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_provenance_manifest(
    manifest_path: Path, *, motion_path: Optional[Path] = None
) -> Dict[str, Any]:
    """Validate the required sidecar fields and an optional motion hash."""

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MotionExportError("cannot read provenance manifest %s" % manifest_path) from exc
    if not isinstance(manifest, dict):
        raise MotionExportError("provenance manifest must be a JSON object")
    missing = [key for key in MANIFEST_REQUIRED if key not in manifest]
    if missing:
        raise MotionExportError("manifest missing required fields: %s" % ", ".join(missing))
    manifest_version = manifest["export_schema_version"]
    if isinstance(manifest_version, bool) or not isinstance(manifest_version, int):
        raise MotionExportError("manifest export_schema_version must be an integer")
    if manifest_version != SCHEMA_VERSION:
        raise MotionExportError("unsupported manifest schema version")
    if motion_path is not None and "motion_sha256" in manifest:
        actual = _sha256(motion_path)
        if str(manifest["motion_sha256"]) != actual:
            raise MotionExportError("motion_sha256 does not match %s" % motion_path)
    return manifest


def validate_motion_export(
    path: Path | str,
    *,
    manifest_path: Optional[Path | str] = None,
    allow_schema_versions: Iterable[int] = (SCHEMA_VERSION,),
) -> Dict[str, Any]:
    """Validate one NPZ export and return a compact, JSON-safe summary.

    Loading always uses ``allow_pickle=False``.  The function performs no
    conversion and never writes to either the source NPZ or its manifest.
    """

    motion_path = Path(path)
    if motion_path.suffix.lower() != ".npz":
        raise MotionExportError("motion export must be a non-pickle .npz file")
    if not motion_path.is_file():
        raise MotionExportError("motion export does not exist: %s" % motion_path)
    try:
        with np.load(motion_path, allow_pickle=False) as loaded:
            data = {key: loaded[key] for key in loaded.files}
    except (OSError, ValueError, TypeError) as exc:
        raise MotionExportError("cannot load motion export %s" % motion_path) from exc

    missing = [key for key in HUMAN_REQUIRED if key not in data]
    if missing:
        raise MotionExportError("missing required fields: %s" % ", ".join(missing))
    schema_version = _positive_int(_scalar(data, "schema_version"), "schema_version")
    if schema_version not in set(allow_schema_versions):
        raise MotionExportError("unsupported schema_version=%d" % schema_version)
    sequence_id = _text(data, "sequence_id")
    task_family = _text(data, "task_family")
    if task_family not in TASK_FAMILIES:
        raise MotionExportError("unknown task_family=%s" % task_family)
    coordinate_frame = _text(data, "coordinate_frame")
    try:
        fps = float(_scalar(data, "fps"))
    except (TypeError, ValueError) as exc:
        raise MotionExportError("fps must be a numeric scalar") from exc
    if not np.isfinite(fps) or fps <= 0:
        raise MotionExportError("fps must be finite and positive")

    global_orient = _array(data, "global_orient", ndim=2, shape_tail=(3,))
    body_pose = _array(data, "body_pose", ndim=3, shape_tail=(21, 3))
    transl = _array(data, "transl", ndim=2, shape_tail=(3,))
    betas = _array(data, "betas", ndim=1)
    _text(data, "gender")
    if betas.shape[0] <= 0:
        raise MotionExportError("betas must contain at least one shape coefficient")
    pose_frames = int(global_orient.shape[0])
    if body_pose.shape[0] != pose_frames or transl.shape[0] != pose_frames:
        raise MotionExportError("global_orient, body_pose, and transl frame counts differ")

    has_hand_pose = any(key in data for key in HAND_POSE_FIELDS)
    if has_hand_pose:
        missing = [key for key in HAND_POSE_FIELDS if key not in data]
        if missing:
            raise MotionExportError("hand pose fields must be supplied together")
        for key in HAND_POSE_FIELDS:
            hand_pose = _array(data, key, ndim=2, shape_tail=(45,))
            if hand_pose.shape[0] != pose_frames:
                raise MotionExportError("%s frame count differs from body pose" % key)

    coarse_frames = None
    if "global_jpos" in data:
        global_jpos = _array(data, "global_jpos", ndim=3, shape_tail=(28, 3))
        coarse_frames = int(global_jpos.shape[0])
    interp_scale = _optional_scalar_int(data, "interp_scale")
    if coarse_frames is not None and interp_scale is not None:
        if pose_frames != coarse_frames * interp_scale:
            raise MotionExportError(
                "pose frame count %d is not coarse frame count %d * interp_scale %d"
                % (pose_frames, coarse_frames, interp_scale)
            )

    has_object = any(key in data for key in OBJECT_REQUIRED)
    if task_family in ("hoi", "hosi"):
        missing = [key for key in OBJECT_REQUIRED if key not in data]
        if missing:
            raise MotionExportError("object task missing fields: %s" % ", ".join(missing))
        has_object = True
    if has_object:
        if not all(key in data for key in OBJECT_REQUIRED):
            raise MotionExportError("object fields must be supplied together")
        _text(data, "object_name")
        object_trans = _array(data, "object_trans", ndim=2, shape_tail=(3,))
        object_rot_mat = _array(data, "object_rot_mat", ndim=3, shape_tail=(3, 3))
        if object_trans.shape[0] != object_rot_mat.shape[0]:
            raise MotionExportError("object translation and rotation frame counts differ")
        allowed_object_frames = {pose_frames}
        if coarse_frames is not None:
            allowed_object_frames.add(coarse_frames)
        if int(object_trans.shape[0]) not in allowed_object_frames:
            raise MotionExportError("object stream has no matching pose/coarse frame rate")

    timeline_frames = coarse_frames if coarse_frames is not None else pose_frames
    _check_positive_vector(data, "window_lengths", expected_sum=timeline_frames)
    _check_index_vector(data, "seams", upper=timeline_frames)
    if "window_id" in data:
        window_id = _array(data, "window_id", ndim=1, finite=False)
        if not np.issubdtype(window_id.dtype, np.integer) or (window_id < 0).any():
            raise MotionExportError("window_id must contain non-negative integers")
        if window_id.shape[0] != timeline_frames:
            raise MotionExportError("window_id length must match the motion timeline")
    history_frames = _optional_scalar_int(data, "history_frames")
    if history_frames is not None and history_frames >= timeline_frames:
        raise MotionExportError("history_frames must be smaller than the coarse sequence")

    manifest = None
    if manifest_path is not None:
        manifest = validate_provenance_manifest(Path(manifest_path), motion_path=motion_path)

    return {
        "schema_version": schema_version,
        "sequence_id": sequence_id,
        "task_family": task_family,
        "coordinate_frame": coordinate_frame,
        "fps": fps,
        "pose_frames": pose_frames,
        "coarse_frames": coarse_frames,
        "interp_scale": interp_scale,
        "has_object": has_object,
        "has_hand_pose": has_hand_pose,
        "manifest_validated": manifest is not None,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate one motion-export NPZ")
    parser.add_argument("motion", type=Path)
    parser.add_argument("--manifest", type=Path, default=None)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = validate_motion_export(args.motion, manifest_path=args.manifest)
    except MotionExportError as exc:
        print("INVALID: %s" % exc)
        return 2
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
