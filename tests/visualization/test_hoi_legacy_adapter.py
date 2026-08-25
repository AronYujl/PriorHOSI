import pickle

import numpy as np
import pytest

from tools.visualization.hoi_legacy import (
    HOILegacyExportError,
    legacy_to_payload,
    write_legacy_manifest,
    write_motion_npz,
)
from tools.visualization.schema import validate_motion_export


def _write_legacy(path, *, samples=2, frames=4, **overrides):
    pose = np.arange(samples * frames * 22 * 3, dtype=np.float32).reshape(
        samples * frames, 22, 3
    )
    root = np.arange(samples * frames * 3, dtype=np.float32).reshape(samples * frames, 3)
    obj_trans = np.arange(samples * frames * 3, dtype=np.float64).reshape(samples, frames, 3)
    obj_rot = np.broadcast_to(np.eye(3, dtype=np.float64), (samples, frames, 3, 3)).reshape(
        samples, frames, 9
    )
    value = {
        "seq_name": "sub_fixture_chair_001",
        "human_motion": {
            "pose_pred": pose,
            "root_trans": root,
            "betas": np.arange(16, dtype=np.float32),
            "gender": "neutral",
        },
        "object_motion": {
            "obj_trans": obj_trans,
            "obj_rot_mat": obj_rot,
            "obj_name": "chair",
        },
    }
    for key, value_override in overrides.items():
        if key in ("human_motion", "object_motion"):
            value[key].update(value_override)
        else:
            value[key] = value_override
    with path.open("wb") as handle:
        pickle.dump(value, handle, protocol=4)


def test_flattened_multi_sample_pickle_selects_one_trajectory(tmp_path):
    source = tmp_path / "legacy_motion_params.pkl"
    _write_legacy(source)

    payload, metadata = legacy_to_payload(source, sample_index=1, legacy_human_frame="y_up")

    assert payload["global_orient"].shape == (4, 3)
    assert payload["body_pose"].shape == (4, 21, 3)
    assert payload["transl"].shape == (4, 3)
    assert payload["object_trans"].shape == (4, 3)
    assert payload["object_rot_mat"].shape == (4, 3, 3)
    np.testing.assert_array_equal(payload["global_orient"], np.arange(2 * 4 * 22 * 3, dtype=np.float32).reshape(2, 4, 22, 3)[1, :, 0])
    assert metadata["legacy_sample_index"] == 1
    assert metadata["legacy_sample_count"] == 2


def test_legacy_z_up_human_fields_are_converted_to_canonical_y_up(tmp_path):
    source = tmp_path / "legacy_motion_params.pkl"
    _write_legacy(source, samples=1, frames=1)

    payload, metadata = legacy_to_payload(source)

    raw_pose = np.arange(22 * 3, dtype=np.float32).reshape(1, 22, 3)
    raw_root = np.arange(3, dtype=np.float32).reshape(1, 3)
    expected_pose = raw_pose[..., [0, 2, 1]].copy()
    expected_pose[..., 2] *= -1
    expected_root = raw_root[..., [0, 2, 1]].copy()
    expected_root[..., 2] *= -1
    np.testing.assert_array_equal(payload["global_orient"], expected_pose[:, 0])
    np.testing.assert_array_equal(payload["body_pose"], expected_pose[:, 1:])
    np.testing.assert_array_equal(payload["transl"], expected_root)
    assert metadata["legacy_human_frame"] == "z_up"


def test_adapter_output_round_trips_through_schema_and_manifest(tmp_path):
    source = tmp_path / "legacy_motion_params.pkl"
    motion = tmp_path / "exports" / "sample_0.npz"
    manifest = tmp_path / "exports" / "sample_0.manifest.json"
    _write_legacy(source, samples=1, frames=5)
    payload, metadata = legacy_to_payload(source)
    write_motion_npz(motion, payload)
    write_legacy_manifest(manifest, motion_path=motion, source_path=source, metadata=metadata)

    summary = validate_motion_export(motion, manifest_path=manifest)

    assert summary["task_family"] == "hoi"
    assert summary["pose_frames"] == 5
    assert summary["has_object"] is True
    assert summary["manifest_validated"] is True


def test_adapter_rejects_sample_count_mismatch(tmp_path):
    source = tmp_path / "legacy_motion_params.pkl"
    _write_legacy(source, samples=2, frames=4, object_motion={"obj_trans": np.zeros((3, 4, 3), dtype=np.float32)})

    with pytest.raises(HOILegacyExportError, match="sample/frame counts differ"):
        legacy_to_payload(source)


def test_adapter_refuses_overwrite(tmp_path):
    source = tmp_path / "legacy_motion_params.pkl"
    motion = tmp_path / "sample.npz"
    _write_legacy(source, samples=1)
    payload, _ = legacy_to_payload(source)
    write_motion_npz(motion, payload)

    with pytest.raises(HOILegacyExportError, match="refusing to overwrite"):
        write_motion_npz(motion, payload)
