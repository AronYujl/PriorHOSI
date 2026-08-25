from pathlib import Path

import numpy as np
import pytest

from tools.visualization.video import (
    VideoRenderError,
    _ffmpeg_command,
    _fixed_orthographic_camera,
    _promote_no_replace,
    _validate_video_settings,
    _validate_video_targets,
)


@pytest.mark.parametrize(
    "settings",
    [
        {"width": 63, "height": 480, "fps": 30.0, "crf": 18},
        {"width": 641, "height": 480, "fps": 30.0, "crf": 18},
        {"width": 640, "height": 481, "fps": 30.0, "crf": 18},
        {"width": 640, "height": 480, "fps": 0.0, "crf": 18},
        {"width": 640, "height": 480, "fps": 30.0, "crf": 52},
    ],
)
def test_video_settings_fail_closed(settings):
    with pytest.raises(VideoRenderError):
        _validate_video_settings(**settings)


def test_video_targets_refuse_overwrite(tmp_path):
    output = tmp_path / "motion.mp4"
    manifest = tmp_path / "motion.render.json"
    output.write_bytes(b"existing")

    with pytest.raises(VideoRenderError, match="refusing to overwrite video"):
        _validate_video_targets(output, manifest)

    output.unlink()
    manifest.write_text("{}\n", encoding="utf-8")
    with pytest.raises(VideoRenderError, match="render manifest"):
        _validate_video_targets(output, manifest)


def test_atomic_promotion_does_not_replace_existing_output(tmp_path):
    source = tmp_path / "partial.mp4"
    destination = tmp_path / "motion.mp4"
    source.write_bytes(b"new")
    destination.write_bytes(b"old")

    with pytest.raises(VideoRenderError, match="refusing to overwrite"):
        _promote_no_replace(source, destination)

    assert source.read_bytes() == b"new"
    assert destination.read_bytes() == b"old"


def test_fixed_camera_contains_all_vertices_and_preserves_aspect():
    human = np.asarray(
        [[[-1.0, 0.0, -0.5], [0.5, 2.0, 1.5]]], dtype=np.float32
    )
    object_vertices = np.asarray(
        [[[-2.0, 0.2, 1.0], [2.0, 1.0, -1.0]]], dtype=np.float32
    )
    camera = _fixed_orthographic_camera(
        human,
        object_vertices,
        width=640,
        height=480,
        camera_elev=18.0,
        camera_azim=-58.0,
        padding=1.10,
    )
    rotation = np.asarray(camera["rotation"])
    translation = np.asarray(camera["translation"])
    points = np.concatenate(
        [human.reshape(-1, 3), object_vertices.reshape(-1, 3)], axis=0
    )
    view = points @ rotation + translation

    assert view[:, 0].min() > camera["min_x"]
    assert view[:, 0].max() < camera["max_x"]
    assert view[:, 1].min() > camera["min_y"]
    assert view[:, 1].max() < camera["max_y"]
    assert camera["znear"] < view[:, 2].min()
    assert camera["zfar"] > view[:, 2].max()
    assert (camera["max_x"] - camera["min_x"]) / (
        camera["max_y"] - camera["min_y"]
    ) == pytest.approx(640 / 480)


def test_ffmpeg_command_uses_raw_rgb_and_h264(tmp_path):
    command = _ffmpeg_command(
        "/opt/ffmpeg",
        Path(tmp_path / "motion.mp4"),
        width=640,
        height=480,
        fps=30.0,
        crf=18,
        preset="medium",
    )

    assert command[0] == "/opt/ffmpeg"
    assert command[command.index("-f") + 1] == "rawvideo"
    assert command[command.index("-s:v") + 1] == "640x480"
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt", command.index("-i")) + 1] == "yuv420p"
    assert command[-1] == str(tmp_path / "motion.mp4")
