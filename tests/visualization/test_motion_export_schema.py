import hashlib
import json

import numpy as np
import pytest

from tools.visualization.schema import MotionExportError, validate_motion_export


def _write_export(path, *, task_family="hsi", pose_frames=6, coarse_frames=3, **overrides):
    payload = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "sequence_id": np.asarray("fixture:0001"),
        "task_family": np.asarray(task_family),
        "fps": np.asarray(30.0, dtype=np.float32),
        "coordinate_frame": np.asarray("fixture_y_up"),
        "global_orient": np.zeros((pose_frames, 3), dtype=np.float32),
        "body_pose": np.zeros((pose_frames, 21, 3), dtype=np.float32),
        "transl": np.zeros((pose_frames, 3), dtype=np.float32),
        "betas": np.zeros((16,), dtype=np.float32),
        "gender": np.asarray("neutral"),
        "global_jpos": np.zeros((coarse_frames, 28, 3), dtype=np.float32),
        "interp_scale": np.asarray(pose_frames // coarse_frames, dtype=np.int32),
        "window_lengths": np.asarray([coarse_frames], dtype=np.int32),
        "seams": np.asarray([], dtype=np.int32),
        "history_frames": np.asarray(1, dtype=np.int32),
    }
    payload.update(overrides)
    np.savez(path, **payload)


def _write_manifest(path, motion_path):
    digest = hashlib.sha256(motion_path.read_bytes()).hexdigest()
    manifest = {
        "export_schema_version": 1,
        "source_git_commit": "a" * 40,
        "source_live_head_at_completion": "b" * 40,
        "resolved_config_sha256": "c" * 64,
        "checkpoint_path_and_sha256": "checkpoint.pth: " + "d" * 64,
        "dataset_snapshot_and_sha256": "dataset: " + "e" * 64,
        "smpl_models_sha256": "f" * 64,
        "object_asset_manifest_sha256": "0" * 64,
        "scene_asset_manifest_sha256": "1" * 64,
        "command": "synthetic fixture",
        "working_directory": "/tmp/fixture",
        "created_at": "2026-08-25T00:00:00+08:00",
        "motion_sha256": digest,
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")


def test_valid_hsi_export_and_manifest(tmp_path):
    motion = tmp_path / "hsi.npz"
    manifest = tmp_path / "manifest.json"
    _write_export(motion)
    _write_manifest(manifest, motion)

    summary = validate_motion_export(motion, manifest_path=manifest)

    assert summary == {
        "schema_version": 1,
        "sequence_id": "fixture:0001",
        "task_family": "hsi",
        "coordinate_frame": "fixture_y_up",
        "fps": 30.0,
        "pose_frames": 6,
        "coarse_frames": 3,
        "interp_scale": 2,
        "has_object": False,
        "has_hand_pose": False,
        "manifest_validated": True,
    }


def test_valid_hoi_requires_object_stream(tmp_path):
    motion = tmp_path / "hoi.npz"
    _write_export(
        motion,
        task_family="hoi",
        object_name=np.asarray("chair"),
        object_trans=np.zeros((3, 3), dtype=np.float32),
        object_rot_mat=np.broadcast_to(np.eye(3, dtype=np.float32), (3, 3, 3)),
    )

    assert validate_motion_export(motion)["has_object"] is True


def test_optional_hand_pose_is_validated_as_a_pair(tmp_path):
    motion = tmp_path / "hands.npz"
    _write_export(
        motion,
        left_hand_pose=np.zeros((6, 45), dtype=np.float32),
        right_hand_pose=np.zeros((6, 45), dtype=np.float32),
    )

    assert validate_motion_export(motion)["has_hand_pose"] is True

    _write_export(
        motion,
        left_hand_pose=np.zeros((6, 45), dtype=np.float32),
        right_hand_pose=np.zeros((5, 45), dtype=np.float32),
    )
    with pytest.raises(MotionExportError, match="frame count differs"):
        validate_motion_export(motion)


def test_hsi_rejects_partial_object_stream(tmp_path):
    motion = tmp_path / "partial.npz"
    _write_export(motion, object_name=np.asarray("chair"))

    with pytest.raises(MotionExportError, match="object fields must be supplied together"):
        validate_motion_export(motion)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"coordinate_frame": np.asarray("")}, "coordinate_frame must not be empty"),
        ({"fps": np.asarray(0.0)}, "fps must be finite and positive"),
        ({"transl": np.zeros((5, 3), dtype=np.float32)}, "frame counts differ"),
        ({"global_jpos": np.zeros((4, 28, 3), dtype=np.float32)}, "not coarse frame count"),
        ({"global_orient": np.full((6, 3), np.nan, dtype=np.float32)}, "non-finite"),
    ],
)
def test_invalid_human_or_timing_fields_are_rejected(tmp_path, overrides, message):
    motion = tmp_path / "invalid.npz"
    _write_export(motion, **overrides)

    with pytest.raises(MotionExportError, match=message):
        validate_motion_export(motion)


def test_unsupported_schema_is_rejected(tmp_path):
    motion = tmp_path / "future.npz"
    _write_export(motion, schema_version=np.asarray(2, dtype=np.int32))

    with pytest.raises(MotionExportError, match="unsupported schema_version=2"):
        validate_motion_export(motion)


def test_manifest_hash_mismatch_is_rejected(tmp_path):
    motion = tmp_path / "hsi.npz"
    manifest = tmp_path / "manifest.json"
    _write_export(motion)
    _write_manifest(manifest, motion)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["motion_sha256"] = "0" * 64
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(MotionExportError, match="motion_sha256 does not match"):
        validate_motion_export(motion, manifest_path=manifest)


def test_gender_and_manifest_version_are_strict(tmp_path):
    motion = tmp_path / "invalid.npz"
    manifest = tmp_path / "manifest.json"
    _write_export(motion, gender=np.asarray(""))
    with pytest.raises(MotionExportError, match="gender must not be empty"):
        validate_motion_export(motion)

    _write_export(motion)
    _write_manifest(manifest, motion)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    payload["export_schema_version"] = 1.5
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MotionExportError, match="must be an integer"):
        validate_motion_export(motion, manifest_path=manifest)
