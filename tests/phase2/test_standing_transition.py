"""Handoff eligibility, temporal padding and metric coordinate conventions."""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from mixer.standing_transition import observed_motion, select_sources, scene_measures, tail_measures


def test_interpolation_padding_cannot_turn_motion_into_a_stopped_tail():
    pose = torch.zeros(8, 22, 3)
    points = torch.zeros(8, 28, 3)
    points[:6, :, 0] = torch.arange(6.)[:, None] * .1
    points[6:] = points[5]
    motion = observed_motion(dict(pose=pose, joints=points, betas=torch.ones(16)))
    assert len(motion['joints']) == 6
    assert torch.allclose((motion['joints'][-1] - motion['joints'][-2])[:, 0] * 30, torch.full((28,), 3.))
    assert len(motion['betas']) == 16


def test_selector_preserves_eligibility_and_scene_diversity():
    rows = [dict(task=0, object='box', scene='a', standing=True, eligible=False),
            dict(task=1, object='box', scene='a', standing=True, eligible=True),
            dict(task=2, object='lamp', scene='a', standing=True, eligible=True),
            dict(task=3, object='lamp', scene='b', standing=True, eligible=True),
            dict(task=4, object='lamp', scene='c', standing=False, eligible=False)]
    selected = select_sources(rows, 4)
    assert [r['task'] for r in selected] == [1, 3, 2, 0]
    assert [r['eligible'] for r in selected] == [True, True, True, False]


def test_scene_distances_are_in_metres_and_outside_is_reported():
    sdf = torch.full((1, 1, 4, 4, 4), -.1)
    vertices = torch.tensor([[[0., 0., 0.], [3., 0., 0.]]])
    values = scene_measures(vertices, sdf, dict(centroid=[0, 0, 0], extents=[4, 2, 1]))
    assert values['scene_penetration_mean_m'] == pytest.approx(.2)
    assert values['scene_outside_fraction'] == .5


def test_hand_release_is_independent_of_an_upright_body():
    joints = torch.zeros(10, 28, 3)
    joints[:, 0, 1] = 1.
    joints[:, 12, 1] = 1.5
    joints[:, [24, 26], 0] = 1.
    motion = dict(joints=joints, pose=torch.zeros(10, 22, 3),
                  object_translation=torch.tensor([[1., 0., 0.]]).repeat(10, 1),
                  object_rotation=torch.eye(3).repeat(10, 1, 1))
    limits = SimpleNamespace(tail_frames=10, tilt_deg=25, root_height_m=.65, foot_height_m=.08,
        hand_distance_m=.08, object_floor_m=.05, object_speed_m_s=.1, object_angular_speed_rad_s=.5)
    values = tail_measures(motion, torch.zeros(1, 3), limits)
    assert values['standing']
    assert not values['eligible']
    assert values['release_conditions']['grounded']
    assert not values['release_conditions']['hands']
