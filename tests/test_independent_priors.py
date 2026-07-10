import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from models.infbagel import Sampler  # noqa: E402
from models.priors import HOIPrior, HOI_PRIOR_SPEC, HSI_PRIOR_SPEC  # noqa: E402
from prior_utils import (  # noqa: E402
    balanced_subset_indices,
    build_motion_state,
    load_training_checkpoint,
    save_checkpoint,
    split_dataset_indices,
)


def make_batch(batch_size=2, frames=16):
    return {
        "joints": torch.zeros(batch_size, frames, 84),
        "global_rot_6d": torch.zeros(batch_size, frames, 22, 6),
        "object_trans": torch.zeros(batch_size, frames, 3),
        "object_rot_mat": torch.zeros(batch_size, frames, 3, 3),
        "contact_label": torch.zeros(batch_size, frames, 4),
    }


class DummySplitDataset:
    ori_sequence_idx = np.asarray([0, 0, 1, 1, 2, 2, 3, 3])
    ori_sequence_start_idx = np.asarray([0, 2, 4, 6])
    scene_name = ["a", "a", "b", "b", "c", "c", "d", "d"]


class DummyDataset:
    load_scene = False
    load_object_goal = False
    use_object_keypoints = False
    max_window_size = 16
    nb_voxels = [4, 4, 4]


class ZeroDenoiser(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.channels = channels

    def forward(self, x, *args, **kwargs):
        return torch.zeros_like(x) + self.anchor


class IndependentPriorTests(unittest.TestCase):
    def test_state_contracts_are_distinct(self):
        batch = make_batch()
        self.assertEqual(build_motion_state(batch, "hoi").shape[-1], 232)
        self.assertEqual(build_motion_state(batch, "hsi").shape[-1], 216)
        self.assertTrue(HOI_PRIOR_SPEC.uses_object)
        self.assertFalse(HOI_PRIOR_SPEC.uses_scene)
        self.assertTrue(HSI_PRIOR_SPEC.uses_scene)
        self.assertFalse(HSI_PRIOR_SPEC.uses_object)

    def test_group_split_has_no_sequence_leakage(self):
        train, validation = split_dataset_indices(
            DummySplitDataset(), 0.25, "sequence", seed=2027
        )
        train_groups = set(DummySplitDataset.ori_sequence_idx[train].tolist())
        validation_groups = set(DummySplitDataset.ori_sequence_idx[validation].tolist())
        self.assertFalse(train_groups & validation_groups)
        self.assertTrue(train)
        self.assertTrue(validation)

    def test_balanced_subset_covers_distinct_groups(self):
        selected = balanced_subset_indices(
            DummySplitDataset(), list(range(8)), "sequence", max_items=4, seed=7
        )
        groups = DummySplitDataset.ori_sequence_idx[selected]
        self.assertEqual(len(set(groups.tolist())), 4)

    def test_hsi_loss_has_no_dummy_object_terms(self):
        batch_size, frames = 2, 16
        sampler = Sampler(
            device="cpu", mask_ind=0, emb_f=1, batch_size=batch_size,
            channel=216, auto_regre_num=2, timesteps=10,
            ddim_timesteps=5, cm_timesteps=2, scene_type=None,
            temp_voxel_num=0, is_mix=True,
        )
        model = ZeroDenoiser(216)
        sampler.set_dataset_and_model(DummyDataset(), model)
        state = torch.zeros(batch_size, frames, 216)
        mask = torch.zeros_like(state, dtype=torch.bool)
        common = {
            "joints": state[..., :84],
            "mat": torch.eye(4).repeat(batch_size, 1, 1),
            "scene_flag": torch.zeros(batch_size, dtype=torch.long),
            "text_emb": torch.zeros(batch_size, 1, 768),
            "goal": torch.zeros(batch_size, 3),
            "flag": torch.zeros(batch_size, dtype=torch.bool),
            "pi": torch.zeros(batch_size, dtype=torch.long),
            "seq_len": torch.full((batch_size,), 48, dtype=torch.long),
        }
        losses = sampler.p_losses(
            state, common["joints"], common["mat"], common["scene_flag"],
            mask, torch.ones(batch_size, dtype=torch.long), common["text_emb"],
            common["goal"], common["goal"], common["goal"], common["flag"],
            common["flag"], common["pi"], common["pi"], common["seq_len"],
            common["flag"], common["flag"], common["flag"],
            torch.zeros(batch_size, 1, 1024, 3),
            torch.zeros(batch_size, 3, 3),
            torch.zeros(batch_size, 100, 3),
            torch.zeros(batch_size, frames, 100, 3),
            torch.zeros(batch_size, 24, 3),
            torch.zeros(batch_size, 1024, 3),
        )
        self.assertIsNone(losses["loss_otrans"])
        self.assertIsNone(losses["loss_orot"])
        self.assertIsNone(losses["loss_contact"])
        self.assertTrue(torch.isfinite(losses["loss"]))

    def test_hoi_architecture_disables_scene(self):
        model = HOIPrior(
            dim_model=32, num_heads=4, num_layers=1, dropout_p=0.0,
            nb_voxels=[8, 8, 8], free_p=0.0, load_language=True,
            load_pelvis_goal=True, language_feature_dim=768,
            temp_voxel_num=0,
        )
        self.assertFalse(model.load_scene)
        self.assertTrue(model.load_object_goal)
        self.assertEqual(model.embedding_input.in_features, 232)
        self.assertEqual(model.out.out_features, 232)

    def test_structured_checkpoint_roundtrip(self):
        model = torch.nn.Linear(3, 2)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "prior.pth"
            save_checkpoint(
                path, model, optimizer, None, epoch=4, global_step=17,
                cfg={"test": True}, prior_type="hsi",
                data_contract={"split_unit": "scene"}, metrics={"loss": 1.0},
            )
            restored = torch.nn.Linear(3, 2)
            checkpoint = load_training_checkpoint(
                path, restored, expected_prior="hsi"
            )
            self.assertEqual(checkpoint["schema_version"], 1)
            self.assertEqual(checkpoint["epoch"], 4)
            for expected, actual in zip(model.parameters(), restored.parameters()):
                self.assertTrue(torch.equal(expected, actual))


if __name__ == "__main__":
    unittest.main()
