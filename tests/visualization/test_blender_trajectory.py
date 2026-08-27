import json
from pathlib import Path

import numpy as np
import pytest

from tools.visualization.blender import BlenderRenderError
from tools.visualization.blender_trajectory import (
    LINGO_FOREGROUND_MATERIALS,
    ORANGE_TIME_GRADIENT_LINGO_OBJECT_MATERIALS,
    ORANGE_TIME_GRADIENT_MATERIAL_STYLE,
    TIME_GRADIENT_LINGO_OBJECT_MATERIALS,
    TIME_GRADIENT_MATERIAL_STYLE,
    _build_config,
    _build_parser,
    _resolve_materials,
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
        "material_style": "omomo",
        "material_change_scope": "foreground_human_and_object_only",
    }
    assert config["camera"] is source["camera"]


def test_multi_pose_default_preserves_omomo_source_materials():
    source_materials = {
        "human_source": "blue",
        "object_source": "purple",
        "smooth_shading": True,
    }

    resolved = _resolve_materials(source_materials, "omomo")

    assert resolved == source_materials
    assert resolved is not source_materials
    resolved["human_source"] = "changed"
    assert source_materials["human_source"] == "blue"

    with pytest.raises(BlenderRenderError, match="source OMOMO materials"):
        _resolve_materials({"human_source": "blue"}, "omomo")


def test_multi_pose_lingo_style_uses_exact_hsi_foreground_palette():
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
        frames=[0, 42, 83, 125],
        output_image="trajectory-k4-lingo.png",
        width=1600,
        height=800,
        samples=64,
        material_style="lingo",
    )

    assert config["materials"] == LINGO_FOREGROUND_MATERIALS
    assert config["materials"] is not LINGO_FOREGROUND_MATERIALS
    assert config["composition"]["material_style"] == "lingo"
    assert config["camera"] is source["camera"]
    assert config["floor"] is source["floor"]
    assert config["color_management"] is source["color_management"]


def test_multi_pose_time_gradient_uses_white_to_yellow_human_and_lingo_object():
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
        frames=[0, 42, 83, 125],
        output_image="trajectory-k4-time.png",
        width=1600,
        height=800,
        samples=64,
        material_style=TIME_GRADIENT_MATERIAL_STYLE,
    )

    assert config["materials"] == TIME_GRADIENT_LINGO_OBJECT_MATERIALS
    assert config["materials"]["object"] == LINGO_FOREGROUND_MATERIALS["object"]
    assert config["materials"]["human"]["start_color"] == [0.92, 0.92, 0.90, 1.0]
    assert config["materials"]["human"]["end_color"] == [1.0, 0.62, 0.03, 1.0]
    encoding = config["composition"]["temporal_color_encoding"]
    assert encoding["target"] == "human"
    assert encoding["timeline_normalization"] == "source_frame_index/(frame_count-1)"
    assert encoding["interpolation"] == "linear_rgba"


def test_multi_pose_orange_time_gradient_is_darker_and_keeps_lingo_object():
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
        frames=[0, 42, 83, 125],
        output_image="trajectory-k4-orange.png",
        width=1600,
        height=800,
        samples=64,
        material_style=ORANGE_TIME_GRADIENT_MATERIAL_STYLE,
    )

    assert config["materials"] == ORANGE_TIME_GRADIENT_LINGO_OBJECT_MATERIALS
    assert config["materials"]["object"] == LINGO_FOREGROUND_MATERIALS["object"]
    assert config["materials"]["human"]["end_color"] == [0.82, 0.32, 0.055, 1.0]
    assert config["materials"]["human"]["roughness"] == 0.62
    assert config["materials"]["human"]["specular"] == 0.20
    assert config["composition"]["temporal_color_encoding"]["target"] == "human"


@pytest.mark.parametrize("style", ["", "unknown", "LINGO"])
def test_multi_pose_material_style_fails_closed(style):
    with pytest.raises(BlenderRenderError, match="material style"):
        _resolve_materials(
            {"human_source": "blue", "object_source": "purple"}, style
        )


def test_multi_pose_parser_requires_explicit_material_opt_in():
    required = [
        "cache.npz",
        "--cache-manifest",
        "cache.json",
        "--source-render-manifest",
        "render.json",
        "--output-dir",
        "output",
        "--blend-scene",
        "scene.blend",
        "--blender",
        "blender",
        "--frames",
        "0",
        "125",
    ]

    assert _build_parser().parse_args(required).material_style == "omomo"
    lingo_args = _build_parser().parse_args(
        required + ["--material-style", "lingo"]
    )
    assert lingo_args.material_style == "lingo"
    time_args = _build_parser().parse_args(
        required + ["--material-style", TIME_GRADIENT_MATERIAL_STYLE]
    )
    assert time_args.material_style == TIME_GRADIENT_MATERIAL_STYLE
    orange_args = _build_parser().parse_args(
        required + ["--material-style", ORANGE_TIME_GRADIENT_MATERIAL_STYLE]
    )
    assert orange_args.material_style == ORANGE_TIME_GRADIENT_MATERIAL_STYLE


def test_blender_consumer_dispatches_lingo_principled_materials():
    consumer = (
        Path(__file__).parents[2] / "tools/visualization/blender_scene.py"
    ).read_text(encoding="utf-8")

    assert 'settings.get("style", "omomo_source_copy")' in consumer
    assert 'style == "lingo_principled_v1"' in consumer
    assert '"timeline_white_to_yellow_lingo_object_v1"' in consumer
    assert '"timeline_white_to_orange_lingo_object_v1"' in consumer
    assert '"PriorHOSI.Human.LINGOBlue"' in consumer
    assert '"PriorHOSI.Object.LINGOSage"' in consumer
    assert "human_material, object_material = _foreground_materials" in consumer
    assert "_timeline_human_settings" in consumer


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
