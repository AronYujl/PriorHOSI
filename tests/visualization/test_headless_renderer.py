import numpy as np
import pytest
from PIL import Image, ImageDraw

from tools.visualization.headless import (
    HeadlessRenderError,
    _face_subset,
    _crop_white_margins,
    _parse_frames,
    _style_parameters,
    _timeline_manifest_fields,
)


def test_keyframe_selection_is_deterministic_and_unique():
    np.testing.assert_array_equal(_parse_frames(None, 42, 4), [0, 14, 27, 41])
    np.testing.assert_array_equal(_parse_frames("0, 14, 14, 41", 42, 4), [0, 14, 14, 41])


@pytest.mark.parametrize("value", [",", "-1", "0,42", "abc"])
def test_invalid_keyframes_are_rejected(value):
    with pytest.raises(HeadlessRenderError):
        _parse_frames(value, 42, 4)


def test_face_subset_can_request_full_mesh():
    faces = np.arange(30, dtype=np.int64).reshape(10, 3)
    np.testing.assert_array_equal(_face_subset(faces, 0), faces)
    assert _face_subset(faces, 4).shape[0] <= 4


def test_paper_style_is_axis_free_and_orthographic():
    style = _style_parameters("paper")

    assert style["axes_visible"] is False
    assert style["orthographic"] is True
    assert style["bounds_mode"] == "content"


def test_unknown_style_is_rejected():
    with pytest.raises(HeadlessRenderError, match="unsupported render style"):
        _style_parameters("unknown")


def test_paper_crop_removes_white_margin_with_fixed_padding(tmp_path):
    output = tmp_path / "render.png"
    image = Image.new("RGB", (100, 80), "white")
    ImageDraw.Draw(image).rectangle((30, 20, 69, 59), fill="black")
    image.save(output)

    dimensions = _crop_white_margins(output, padding=5)

    assert dimensions == (50, 50)
    assert Image.open(output).size == (50, 50)


def test_render_manifest_keeps_window_semantics_for_selected_frames():
    fields = _timeline_manifest_fields(
        {
            "window_lengths": np.asarray([4, 4, 4]),
            "seams": np.asarray([4, 8]),
            "window_id": np.repeat(np.arange(3), 4),
        },
        np.asarray([0, 3, 4, 8, 11]),
    )

    assert fields == {
        "source_window_lengths": [4, 4, 4],
        "source_seams": [4, 8],
        "selected_window_ids": [0, 0, 1, 2, 2],
    }
