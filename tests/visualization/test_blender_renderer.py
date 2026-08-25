from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.visualization.blender import (
    BlenderRenderError,
    _apply_visual_ground_correction,
    _blender_command,
    _compose_process_figure,
    _select_process_frames,
    _validate_render_settings,
    _y_up_to_z_up,
)


def test_y_up_to_z_up_is_right_handed_rotation():
    source = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    converted = _y_up_to_z_up(source)

    np.testing.assert_allclose(converted, [[1.0, -3.0, 2.0]])
    rotation = np.asarray(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]]
    )
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    np.testing.assert_allclose(source @ rotation, converted)


def test_process_frame_selection_matches_accepted_long_window():
    np.testing.assert_array_equal(
        _select_process_frames(126, 6), [0, 25, 50, 75, 100, 125]
    )


def test_visual_ground_correction_plants_support_and_preserves_upper_contact():
    human = np.repeat(
        np.asarray([[[0.0, 0.0, 0.04], [0.0, 0.0, 1.0], [0.2, 0.0, 0.5]]]),
        5,
        axis=0,
    ).astype(np.float32)
    manipulated_object = np.repeat(
        np.asarray([[[1.0, 0.0, -0.02], [0.01, 0.0, 1.0], [1.0, 0.2, 0.3]]]),
        5,
        axis=0,
    ).astype(np.float32)

    corrected_human, corrected_object, record, streams = (
        _apply_visual_ground_correction(
            human,
            manipulated_object,
            floor_height=0.015,
            correction_sigma=1.0,
            contact_sigma=1.0,
        )
    )

    np.testing.assert_allclose(corrected_human[:, :, 2].min(1), 0.015, atol=1e-5)
    np.testing.assert_allclose(corrected_object[:, :, 2].min(1), 0.015, atol=1e-5)
    assert record["visualization_only"] is True
    assert record["evaluation_forbidden"] is True
    assert record["contact_ranges"] == [[0, 4]]
    assert record["contact_distance_change_cm"]["max"] == pytest.approx(0.0, abs=1e-4)
    assert record["max_rigid_vertical_correction_cm"] == pytest.approx(3.5)
    assert set(streams) == {
        "ground_human_foot_delta_z",
        "ground_human_upper_delta_z",
        "ground_object_delta_z",
        "ground_contact_strength",
        "ground_pre_contact_distance",
    }


@pytest.mark.parametrize(
    "settings",
    [
        {"width": 255, "height": 768, "fps": 30.0, "samples": 32, "figure_columns": 3},
        {"width": 1025, "height": 768, "fps": 30.0, "samples": 32, "figure_columns": 3},
        {"width": 1024, "height": 768, "fps": 0.0, "samples": 32, "figure_columns": 3},
        {"width": 1024, "height": 768, "fps": 30.0, "samples": 0, "figure_columns": 3},
        {"width": 1024, "height": 768, "fps": 30.0, "samples": 32, "figure_columns": 0},
    ],
)
def test_blender_settings_fail_closed(settings):
    with pytest.raises(BlenderRenderError):
        _validate_render_settings(**settings)


def test_blender_command_is_headless_and_uses_explicit_config(tmp_path):
    command = _blender_command(
        Path("/opt/blender/blender"),
        Path("/assets/floor.blend"),
        Path("/repo/blender_scene.py"),
        tmp_path / "config.json",
    )

    assert command[:3] == [
        "/opt/blender/blender",
        "-b",
        "/assets/floor.blend",
    ]
    assert command[command.index("--python-exit-code") + 1] == "2"
    assert command[command.index("--python") + 1] == "/repo/blender_scene.py"
    assert command[-2:] == ["--config", str(tmp_path / "config.json")]


def test_process_figure_composes_recorded_frames_and_refuses_overwrite(tmp_path):
    frame_paths = []
    colors = [(210, 20, 20), (20, 210, 20), (20, 20, 210), (180, 80, 20)]
    for index, color in enumerate(colors):
        path = tmp_path / ("%05d.png" % index)
        Image.new("RGB", (64, 48), color).save(path)
        frame_paths.append(path)
    output = tmp_path / "process.png"
    summary = _compose_process_figure(
        frame_paths, [0, 10, 20, 30], output, columns=2
    )

    assert summary["frame_indices"] == [0, 10, 20, 30]
    assert summary["columns"] == 2
    assert summary["rows"] == 2
    assert Image.open(output).size == tuple(summary["dimensions"])
    with pytest.raises(BlenderRenderError, match="refusing to overwrite"):
        _compose_process_figure(frame_paths, [0, 10, 20, 30], output, columns=2)
