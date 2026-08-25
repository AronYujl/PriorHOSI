import json
from pathlib import Path

import numpy as np
import pytest

from tools.visualization.blender import BlenderRenderError
from tools.visualization.blender_trajectory import (
    _build_config,
    _validate_keyframes,
    _validate_sources,
)
from tools.visualization.headless import _sha256


def test_multi_pose_keyframes_preserve_explicit_temporal_order():
    assert _validate_keyframes([0, 31, 63, 94, 125], 126) == [0, 31, 63, 94, 125]


@pytest.mark.parametrize(
    "frames",
    ([0], [0, 10, 10], [10, 0], [-1, 10], [0, 126], [False, 10]),
)
def test_multi_pose_keyframes_fail_closed(frames):
    with pytest.raises(BlenderRenderError):
        _validate_keyframes(frames, 126)


def test_multi_pose_config_uses_shared_scene_and_unaltered_world_layout():
    source = {
        "camera": {"projection": "orthographic"},
        "color_management": {"view_transform": "Filmic", "look": "None"},
        "device": "CPU",
        "engine": "CYCLES",
        "floor": {"height": 0.015},
        "materials": {"human_source": "blue", "object_source": "purple"},
    }

    config = _build_config(
        source,
        cache=Path("/immutable/mesh-cache.npz"),
        frame_count=126,
        frames=[0, 31, 63, 94, 125],
        output_image="trajectory-k5.png",
        width=1600,
        height=800,
        samples=64,
    )

    assert config["render_mode"] == "multi_pose"
    assert config["selected_frame_indices"] == [0, 31, 63, 94, 125]
    assert config["composition"] == {
        "mode": "opaque_multi_pose_shared_scene",
        "selected_frame_indices": [0, 31, 63, 94, 125],
        "selection_rule": "explicit_increasing_source_frames",
        "pose_layout": "unaltered_source_world_coordinates",
        "camera_fit_source": "complete_mesh_cache_all_frames",
        "human_object_pair_count": 5,
    }
    assert config["camera"] is source["camera"]


def test_multi_pose_source_validation_binds_both_manifests(tmp_path):
    cache = tmp_path / "mesh-cache.npz"
    np.savez_compressed(
        cache,
        human_vertices=np.zeros((3, 4, 3), dtype=np.float32),
        human_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        object_vertices=np.zeros((3, 3, 3), dtype=np.float32),
        object_faces=np.asarray([[0, 1, 2]], dtype=np.int32),
        frame_index=np.arange(3, dtype=np.int64),
    )
    scene = tmp_path / "scene.blend"
    scene.write_bytes(b"fixed-scene")
    correction = {
        "mode": "visual_contact_aware_v1",
        "visualization_only": True,
        "evaluation_forbidden": True,
    }
    cache_manifest = tmp_path / "mesh-cache.manifest.json"
    cache_manifest.write_text(
        json.dumps(
            {
                "sequence_id": "sequence",
                "frame_count": 3,
                "cache_sha256": _sha256(cache),
                "ground_correction": correction,
            }
        ),
        encoding="utf-8",
    )
    render_manifest = tmp_path / "render.manifest.json"
    render_manifest.write_text(
        json.dumps(
            {
                "sequence_id": "sequence",
                "mesh_cache_sha256": _sha256(cache),
                "mesh_cache_manifest_sha256": _sha256(cache_manifest),
                "blend_scene_sha256": _sha256(scene),
                "ground_correction": correction,
            }
        ),
        encoding="utf-8",
    )

    cache_record, render_record, frame_count = _validate_sources(
        cache, cache_manifest, render_manifest, scene
    )

    assert frame_count == 3
    assert cache_record["sequence_id"] == render_record["sequence_id"]

    render_manifest.write_text("{}", encoding="utf-8")
    with pytest.raises(BlenderRenderError, match="source render"):
        _validate_sources(cache, cache_manifest, render_manifest, scene)
