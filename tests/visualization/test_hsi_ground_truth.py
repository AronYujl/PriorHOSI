import json
import pickle
from pathlib import Path

import numpy as np
import pytest

from tools.visualization.hsi_ground_truth import (
    GroundTruthError,
    _error_statistics,
    _matched_render_identity,
    _matched_render_grid_identity,
    _motion_statistics,
    _parse_labeled_render,
    _validate_grid_labels,
    export_matched_ground_truth,
    matched_frame_indices,
)
from tools.visualization.schema import validate_motion_export


def _reference_payload(**overrides):
    coarse_frames = 16
    interp_scale = 3
    fine_frames = coarse_frames * interp_scale
    payload = {
        "schema_version": np.asarray(3, dtype=np.int32),
        "sequence_id": np.asarray("071-write:fixture"),
        "scene_name": np.asarray("071-write"),
        "caption": np.asarray("write on blackboard with right hand"),
        "data_idx": np.asarray(0, dtype=np.int64),
        "source_sequence_index": np.asarray(0, dtype=np.int64),
        "episode_num": np.asarray(1, dtype=np.int32),
        "fps": np.asarray(10.0, dtype=np.float32),
        "interp_scale": np.asarray(interp_scale, dtype=np.int32),
        "global_jpos": np.zeros((coarse_frames, 28, 3), dtype=np.float32),
        "global_orient": np.zeros((fine_frames, 3), dtype=np.float32),
        "body_pose": np.zeros((fine_frames, 21, 3), dtype=np.float32),
        "transl": np.zeros((fine_frames, 3), dtype=np.float32),
        "betas": np.zeros(16, dtype=np.float32),
        "gender": np.asarray("male"),
        "smplx_output_transform": np.asarray("identity"),
        "window_lengths": np.asarray([16], dtype=np.int32),
        "seams": np.asarray([], dtype=np.int32),
        "history_frames": np.asarray(2, dtype=np.int32),
    }
    payload.update(overrides)
    return payload


def _write_dataset(tmp_path: Path):
    root = tmp_path / "dataset"
    language_dir = root / "language_motion_dict"
    language_dir.mkdir(parents=True)
    frames = 20
    joints = np.arange(frames * 28 * 3, dtype=np.float64).reshape(frames, 28, 3)
    orient = np.zeros((frames, 3), dtype=np.float64)
    pose = np.zeros((frames, 63), dtype=np.float64)
    transl = np.arange(frames * 3, dtype=np.float64).reshape(frames, 3) / 100.0
    np.save(root / "human_joints_aligned.npy", joints)
    np.save(root / "human_orient.npy", orient)
    np.save(root / "human_pose.npy", pose)
    np.save(root / "transl_aligned.npy", transl)
    np.save(root / "betas.npy", np.zeros((1, 16), dtype=np.float64))
    np.save(root / "start_idx.npy", np.asarray([0], dtype=np.int32))
    np.save(root / "end_idx.npy", np.asarray([frames], dtype=np.int32))
    with (root / "gender.pkl").open("wb") as handle:
        pickle.dump(["male"], handle)
    with (
        language_dir / "language_motion_dict__inter_and_loco__16.pkl"
    ).open("wb") as handle:
        pickle.dump(
            {
                "ori_sequence_idx": np.asarray([0]),
                "start_idx": np.asarray([0]),
                "text": [["write on blackboard with right hand"]],
            },
            handle,
        )
    return root, joints


def test_matched_indices_reproduce_stride_clamp_and_identity(tmp_path):
    root, _ = _write_dataset(tmp_path)
    identity = matched_frame_indices(_reference_payload(), root)

    assert identity["window_indices"].shape == (1, 16)
    assert identity["stitched_indices"].tolist() == [
        0, 3, 6, 9, 12, 15, 18, 19, 19, 19, 19, 19, 19, 19, 19, 19
    ]
    assert identity["window_lengths"].tolist() == [16]
    assert identity["seams"].tolist() == []


def test_matched_indices_reject_caption_or_episode_mismatch(tmp_path):
    root, _ = _write_dataset(tmp_path)
    with pytest.raises(GroundTruthError, match="caption disagrees"):
        matched_frame_indices(
            _reference_payload(caption=np.asarray("walk to blackboard")), root
        )
    with pytest.raises(GroundTruthError, match="episode_num disagrees"):
        matched_frame_indices(
            _reference_payload(episode_num=np.asarray(2, dtype=np.int32)), root
        )


def test_export_matched_ground_truth_has_exact_arrays_and_provenance(tmp_path):
    root, source_joints = _write_dataset(tmp_path)
    reference = tmp_path / "prediction.npz"
    np.savez_compressed(reference, **_reference_payload())
    models = tmp_path / "smpl_models"
    models.mkdir()
    (models / "marker.txt").write_text("fixture", encoding="utf-8")
    scene = tmp_path / "mesh_low.obj"
    scene.write_text("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8")
    output = tmp_path / "ground-truth.npz"
    manifest = tmp_path / "ground-truth.manifest.json"

    result = export_matched_ground_truth(
        reference,
        dataset_root=root,
        output_path=output,
        manifest_path=manifest,
        smpl_models=models,
        scene_mesh=scene,
        renderer_commit="a" * 40,
        source_evaluator_commit="b" * 40,
        command="fixture export",
    )

    summary = validate_motion_export(output, manifest_path=manifest)
    assert result["coarse_frames"] == 16
    assert summary["pose_frames"] == 48
    assert summary["fps"] == 30.0
    with np.load(output, allow_pickle=False) as data:
        indices = data["gt_stitched_raw_indices"]
        assert data["motion_role"].item() == "ground_truth"
        assert np.array_equal(
            data["global_jpos"], source_joints[indices].astype(np.float32)
        )
        assert data["global_orient"].shape == (48, 3)
        assert data["body_pose"].shape == (48, 21, 3)
        assert data["transl"].shape == (48, 3)
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["motion_role"] == "ground_truth"
    assert record["evaluation_forbidden"] is True
    assert record["reference_prediction_path"] == str(reference)
    assert record["ground_truth_protocol"]["window_stride_raw"] == 42


def test_joint_statistics_keep_native_and_fk_errors_explicit():
    joints = np.zeros((3, 28, 3), dtype=np.float32)
    joints[:, 17, 1] = 1.0
    joints[:, 21, 1] = [-0.2, 0.4, 1.2]
    stats = _motion_statistics(joints)
    assert stats["joints"]["right_wrist"]["pelvis_relative_y_max_m"] == pytest.approx(1.2)
    assert stats["right_wrist_above_shoulder_fraction"] == pytest.approx(1 / 3)
    errors = _error_statistics(joints, joints + 0.01)
    assert errors["mean_m"] == pytest.approx(np.sqrt(3) * 0.01)


def test_render_identity_requires_exact_camera_and_video_contract():
    camera = {
        "location": [1.0, 2.0, 3.0],
        "rotation_euler": [0.1, 0.2, 0.3],
        "ortho_scale": 9.0,
    }
    base = {
        "scene_mesh_sha256": "a" * 64,
        "frame_png_count": 174,
        "video_probe": {
            "width": 1280, "height": 720, "fps": 30.0, "frame_count": 174
        },
        "scene_report": {"camera_video": camera},
    }
    identity = _matched_render_identity(base, json.loads(json.dumps(base)))
    assert identity["frame_count"] == 174
    changed = json.loads(json.dumps(base))
    changed["scene_report"]["camera_video"]["ortho_scale"] = 9.1
    with pytest.raises(GroundTruthError, match="cameras disagree"):
        _matched_render_identity(base, changed)


def test_render_grid_requires_shared_sequence_caption_and_camera():
    camera = {
        "location": [1.0, 2.0, 3.0],
        "rotation_euler": [0.1, 0.2, 0.3],
        "ortho_scale": 9.0,
    }
    base = {
        "sequence_id": "062:006305",
        "caption": "sit down on chair",
        "scene_mesh_sha256": "a" * 64,
        "frame_png_count": 300,
        "video_probe": {
            "width": 1280, "height": 720, "fps": 30.0, "frame_count": 300
        },
        "scene_report": {"camera_video": camera},
    }
    identity = _matched_render_grid_identity(
        [json.loads(json.dumps(base)) for _ in range(4)]
    )
    assert identity["sequence_id"] == "062:006305"
    assert identity["frame_count"] == 300
    changed = [json.loads(json.dumps(base)) for _ in range(4)]
    changed[2]["caption"] = "walk elsewhere"
    with pytest.raises(GroundTruthError, match="caption"):
        _matched_render_grid_identity(changed)


def test_render_grid_labels_and_cli_inputs_fail_closed():
    assert _validate_grid_labels(["Unguided", "Ground truth"]) == [
        "Unguided", "Ground truth"
    ]
    assert _parse_labeled_render("Guided=/tmp/render") == (
        "Guided", Path("/tmp/render")
    )
    with pytest.raises(GroundTruthError, match="unique"):
        _validate_grid_labels(["Guided", "Guided"])
    with pytest.raises(GroundTruthError, match="trimmed"):
        _validate_grid_labels([" Guided"])


def test_blender_consumer_supports_reference_fit_bounds_lock():
    source = Path("tools/visualization/blender_lingo_scene.py").read_text(
        encoding="utf-8"
    )
    assert '"reuse_reference_prediction_fit_bounds"' in source
    assert '"locked_fit_bounds_blender_z_up"' in source
