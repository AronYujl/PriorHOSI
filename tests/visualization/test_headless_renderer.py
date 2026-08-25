import numpy as np
import pytest

from tools.visualization.headless import HeadlessRenderError, _face_subset, _parse_frames


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
