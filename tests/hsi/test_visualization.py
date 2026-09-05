"""The paired review retains physical time and absolute spatial differences."""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
from priors.hsi.visualization import joint_comparison


def test_translation_is_reported_instead_of_aligned_away():
    truth = torch.zeros(4, 28, 3)
    pred = truth + torch.tensor([.03, .04, 0.])
    result = joint_comparison(pred, truth)
    assert abs(result['joint_error_cm'] - 5.) < 1e-5
    assert abs(result['root_error_cm'] - 5.) < 1e-5
    assert abs(result['root_height_error_cm'] - 4.) < 1e-5


def test_explicit_sources_keep_training_filter_and_terminal_goal():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools'))
    from make_lingo_hsi_episodes import build_episodes
    language = {'ori_sequence_idx': np.array([0, 1, 2]),
                'start_idx': np.array([0, 100, 200]), 'end_idx': np.array([48, 148, 248]),
                'left_hand_inter_frame': np.array([-1, -1, 220]),
                'right_hand_inter_frame': np.array([-1, -1, -1])}
    joints = np.zeros((300, 28, 3))
    joints[199, 0] = [1., .5, 2.]
    result = build_episodes(
        {'train': {'scenes': ['room']}}, language, np.array([0, 100, 200]),
        np.array([100, 200, 300]), joints, np.array(['room']*3),
        'train', 20, source_sequence_ids=[1, 2],
    )
    assert [row['source_sequence_idx'] for row in result['room']] == [1]
    assert result['room'][0]['pelvis_goal'] == [1., 0., 2.]
