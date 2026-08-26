import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tools.visualization.hsi_lingo import (
    CANONICAL_COORDINATE_FRAME,
    BlenderRenderError,
    _blender_command,
    _load_native_hsi,
    _obj_geometry_summary,
    _validate_frame_subset,
    adapt_native_hsi,
)
from tools.visualization.schema import validate_motion_export


def _native_payload(**overrides):
    coarse_frames = 4
    interp_scale = 3
    fine_frames = coarse_frames * interp_scale
    payload = {
        "schema_version": np.asarray(3, dtype=np.int32),
        "global_jpos": np.arange(
            coarse_frames * 28 * 3, dtype=np.float32
        ).reshape(coarse_frames, 28, 3),
        "global_orient": np.arange(fine_frames * 3, dtype=np.float32).reshape(
            fine_frames, 3
        )
        / 100.0,
        "body_pose": np.arange(
            fine_frames * 21 * 3, dtype=np.float32
        ).reshape(fine_frames, 21, 3)
        / 1000.0,
        "transl": np.arange(fine_frames * 3, dtype=np.float32).reshape(
            fine_frames, 3
        )
        / 10.0,
        "betas": np.linspace(-0.1, 0.1, 16, dtype=np.float32),
        "gender": np.asarray("male"),
        "smplx_output_transform": np.asarray("identity"),
        "interp_scale": np.asarray(interp_scale, dtype=np.int32),
        "window_lengths": np.asarray([2, 2], dtype=np.int32),
        "seams": np.asarray([2], dtype=np.int32),
        "history_frames": np.asarray(1, dtype=np.int32),
        "scene_name": np.asarray("071-write"),
        "sequence_id": np.asarray("071-write:fixture"),
        "caption": np.asarray("write on blackboard with right hand"),
        "fps": np.asarray(10.0, dtype=np.float32),
    }
    payload.update(overrides)
    return payload


def _write_inputs(tmp_path: Path):
    source = tmp_path / "native.npz"
    np.savez_compressed(source, **_native_payload())
    report = tmp_path / "per_sequence_metrics.json"
    report.write_text(
        json.dumps(
            {
                "checkpoint": {
                    "checkpoint_path": "/external/checkpoint.pth",
                    "checkpoint_sha256": "a" * 64,
                },
                "metrics": {
                    "071-write:fixture": {
                        "pen_ratio": 0.01,
                        "skate_ratio": 0.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    training = tmp_path / "metrics.json"
    training.write_text(json.dumps({"git_commit": "b" * 40}), encoding="utf-8")
    config = tmp_path / "resolved.yaml"
    config.write_text("model: fixture\n", encoding="utf-8")
    models = tmp_path / "smpl_models"
    models.mkdir()
    (models / "marker.txt").write_text("fixture", encoding="utf-8")
    scene = tmp_path / "mesh_low.obj"
    scene.write_text(
        "v -1 0 -2\nv 2 3 4\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    return source, report, training, config, models, scene


def test_native_hsi_adapter_preserves_arrays_and_derives_fine_fps(tmp_path):
    source, report, training, config, models, scene = _write_inputs(tmp_path)
    output = tmp_path / "canonical.npz"
    manifest = tmp_path / "canonical.manifest.json"

    result = adapt_native_hsi(
        source,
        output_path=output,
        manifest_path=manifest,
        shard_report_path=report,
        training_metrics_path=training,
        resolved_config_path=config,
        smpl_models=models,
        scene_mesh=scene,
        adapter_commit="c" * 40,
        command="fixture adapter",
    )

    assert result["sequence_id"] == "071-write:fixture"
    assert result["coarse_fps"] == 10.0
    assert result["fine_fps"] == 30.0
    summary = validate_motion_export(output, manifest_path=manifest)
    assert summary["task_family"] == "hsi"
    assert summary["coordinate_frame"] == CANONICAL_COORDINATE_FRAME
    assert summary["fps"] == 30.0
    assert summary["pose_frames"] == 12
    assert summary["coarse_frames"] == 4
    with np.load(source, allow_pickle=False) as native, np.load(
        output, allow_pickle=False
    ) as canonical:
        for key in ("global_jpos", "global_orient", "body_pose", "transl", "betas"):
            assert np.array_equal(native[key], canonical[key])
        assert canonical["source_rollout_fps"].item() == 10.0
        assert canonical["fps"].item() == 30.0
        assert canonical["smplx_output_transform"].item() == "identity"
    record = json.loads(manifest.read_text(encoding="utf-8"))
    assert record["native_source_sha256"] == hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    assert record["checkpoint_path_and_sha256"].endswith("a" * 64)


@pytest.mark.parametrize(
    "overrides,message",
    [
        ({"schema_version": np.asarray(2)}, "schema_version=3"),
        (
            {"smplx_output_transform": np.asarray("zup_to_yup")},
            "must have smplx_output_transform=identity",
        ),
        ({"global_orient": np.zeros((11, 3), dtype=np.float32)}, "incompatible shapes"),
        ({"window_lengths": np.asarray([3], dtype=np.int32)}, "window_lengths"),
    ],
)
def test_native_hsi_validation_fails_closed(tmp_path, overrides, message):
    path = tmp_path / "invalid.npz"
    np.savez_compressed(path, **_native_payload(**overrides))
    with pytest.raises(BlenderRenderError, match=message):
        _load_native_hsi(path)


def test_adapter_refuses_overwrite(tmp_path):
    source, report, training, config, models, scene = _write_inputs(tmp_path)
    output = tmp_path / "canonical.npz"
    output.write_bytes(b"occupied")
    with pytest.raises(BlenderRenderError, match="overwrite"):
        adapt_native_hsi(
            source,
            output_path=output,
            manifest_path=tmp_path / "manifest.json",
            shard_report_path=report,
            training_metrics_path=training,
            resolved_config_path=config,
            smpl_models=models,
            scene_mesh=scene,
        )


def test_obj_summary_records_single_yup_to_zup_bounds(tmp_path):
    path = tmp_path / "scene.obj"
    path.write_text(
        "v -1 0 -2\nv 2 3 4\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
    )
    summary = _obj_geometry_summary(path)
    assert summary["vertex_count"] == 3
    assert summary["face_count"] == 1
    assert summary["has_material_directives"] is False
    assert np.allclose(summary["source_y_up_bounds"], [[-1, 0, -2], [2, 3, 4]])
    assert np.allclose(summary["blender_z_up_bounds"], [[-1, -4, 0], [2, 2, 3]])


def test_frame_subset_and_blender_command_are_deterministic(tmp_path):
    assert _validate_frame_subset(None, 4) == [0, 1, 2, 3]
    assert _validate_frame_subset([0, 2, 3], 4) == [0, 2, 3]
    with pytest.raises(BlenderRenderError, match="unique, sorted"):
        _validate_frame_subset([2, 1], 4)
    command = _blender_command(
        tmp_path / "blender", tmp_path / "scene.py", tmp_path / "config.json"
    )
    assert command[1:3] == ["--background", "--factory-startup"]
    assert command[-2:] == ["--config", str(tmp_path / "config.json")]


def test_blender_consumer_creates_world_for_empty_factory_scene():
    source = Path("tools/visualization/blender_lingo_scene.py").read_text(
        encoding="utf-8"
    )
    assert 'bpy.data.worlds.new("PriorHOSI.LINGO.World")' in source
    assert "scene.world = world" in source
    assert 'scene.cycles.device = "CPU"' in source
    assert "scene.cycles.use_denoising = True" in source
    orchestrator = Path("tools/visualization/hsi_lingo.py").read_text(
        encoding="utf-8"
    )
    assert '"kind": "camera_side_dollhouse"' in orchestrator
    assert "floor_wall_geometry_plus_uniform_furniture" in orchestrator
    assert 'bmesh.ops.delete(bm, geom=remove, context="FACES")' in source
