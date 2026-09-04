"""Structural contracts for local-rotation/FK expert composition."""

import os
import sys
import unittest
from pathlib import Path

import torch
from pytorch3d import transforms

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code'))

from mixer.kinematic_composition import KinematicBodyComposer  # noqa: E402
from mixer.composition import ExpertOutputs, OBJECT_CHANNEL_START  # noqa: E402
from priors.core.window_codec import rotation_geodesic  # noqa: E402


PARENTS_22 = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9,
              12, 13, 14, 16, 17, 18, 19)
PARENTS_24 = PARENTS_22 + (20, 21)
LOWER = (1, 2, 4, 5, 7, 8, 10, 11)


class StubKinematicDataset:
    parents_22 = PARENTS_22
    parents_24 = PARENTS_24

    @staticmethod
    def normalize_torch(value, is_object=False):
        del is_object
        return value

    @staticmethod
    def denormalize_torch(value, is_object=False):
        del is_object
        return value

    @staticmethod
    def quat_ik_torch(global_rotation):
        local = global_rotation.clone()
        for joint in range(1, 22):
            parent = PARENTS_22[joint]
            local[..., joint, :, :] = (
                global_rotation[..., parent, :, :].transpose(-1, -2)
                @ global_rotation[..., joint, :, :]
            )
        return local

    @staticmethod
    def quat_fk_torch(local_rotation, local_position, use_joints24=True):
        parents = PARENTS_24 if use_joints24 else PARENTS_22
        global_rotation = [local_rotation[..., :1, :, :]]
        global_position = [local_position[..., :1, :]]
        for joint in range(1, len(parents)):
            parent = parents[joint]
            parent_rotation = global_rotation[parent]
            parent_position = global_position[parent]
            offset = local_position[..., joint:joint + 1, :]
            global_position.append(
                (parent_rotation @ offset.unsqueeze(-1)).squeeze(-1)
                + parent_position
            )
            if joint < local_rotation.shape[-3]:
                global_rotation.append(
                    parent_rotation @ local_rotation[..., joint:joint + 1, :, :]
                )
        return (
            transforms.matrix_to_quaternion(torch.cat(global_rotation, dim=-3)),
            torch.cat(global_position, dim=-2),
        )


def _local_rotations(batch, frames, scale):
    b = torch.arange(batch, dtype=torch.float32).reshape(batch, 1, 1)
    t = torch.arange(frames, dtype=torch.float32).reshape(1, frames, 1)
    j = torch.arange(22, dtype=torch.float32).reshape(1, 1, 22)
    angles = torch.stack((
        scale * (j + 1) + 0.001 * t + 0.003 * b,
        (scale * 0.7 * (j + 1) - 0.002 * t).expand(batch, frames, 22),
        (scale * 0.3 * (j + 1) + 0.001 * b).expand(batch, frames, 22),
    ), dim=-1)
    return transforms.euler_angles_to_matrix(angles, 'XYZ')


def _global_from_local(local):
    result = local.clone()
    for joint in range(1, 22):
        result[..., joint, :, :] = (
            result[..., PARENTS_22[joint], :, :] @ local[..., joint, :, :]
        )
    return result


def _fixture(batch=2, frames=16):
    dataset = StubKinematicDataset()
    offsets = torch.zeros(batch, frames, 24, 3)
    for joint in range(1, 24):
        offsets[..., joint, :] = torch.tensor([
            0.007 * joint, 0.025 + 0.001 * joint, -0.004 * joint,
        ])
    root = torch.zeros(batch, frames, 3)
    root[..., 0] = torch.linspace(-0.4, 0.5, frames)
    root[..., 2] = torch.linspace(0.2, 0.8, frames)

    hoi_local = _local_rotations(batch, frames, 0.006)
    hsi_local = _local_rotations(batch, frames, -0.009)
    hoi_global = _global_from_local(hoi_local)
    hsi_global = _global_from_local(hsi_local)
    lpos = offsets.clone()
    lpos[..., 0, :] = root
    _, fk = dataset.quat_fk_torch(
        hoi_local.reshape(-1, 22, 3, 3), lpos.reshape(-1, 24, 3),
    )
    fk = fk.reshape(batch, frames, 24, 3)

    hoi_position = torch.zeros(batch, frames, 28, 3)
    hoi_position[..., :22, :] = fk[..., :22, :]
    hoi_position[..., 25, :] = fk[..., 22, :]
    hoi_position[..., 27, :] = fk[..., 23, :]
    marker_spec = {
        22: (15, torch.tensor([0.02, 0.03, 0.01])),
        23: (15, torch.tensor([-0.02, 0.03, 0.01])),
        24: (20, torch.tensor([0.03, 0.00, 0.01])),
        26: (21, torch.tensor([-0.03, 0.00, 0.01])),
    }
    for marker, (parent, local_delta) in marker_spec.items():
        moved = (hoi_global[..., parent, :, :] @ local_delta.reshape(3, 1)).squeeze(-1)
        hoi_position[..., marker, :] = hoi_position[..., parent, :] + moved

    hoi = torch.randn(batch, frames, 232)
    hsi = torch.randn(batch, frames, 232)
    hoi[..., :84] = hoi_position.reshape(batch, frames, 84)
    hoi[..., 84:216] = transforms.matrix_to_rotation_6d(hoi_global).reshape(
        batch, frames, 132,
    )
    hsi[..., 84:216] = transforms.matrix_to_rotation_6d(hsi_global).reshape(
        batch, frames, 132,
    )
    fixed = hoi[:, :2].clone()
    return dataset, offsets, hoi, hsi, fixed, hoi_local, hsi_local


class KinematicBodyComposerTests(unittest.TestCase):
    def setUp(self):
        (self.dataset, self.offsets, self.hoi, self.hsi, self.fixed,
         self.hoi_local, self.hsi_local) = _fixture()
        self.composer = KinematicBodyComposer()
        self.output = self.composer(
            ExpertOutputs(hoi=self.hoi, hsi=self.hsi), dataset=self.dataset,
            rest_human_offsets=self.offsets, fixed_points=self.fixed,
        )

    def test_shape_finiteness_history_root_and_object_are_exact(self):
        self.assertEqual(self.output.shape, self.hoi.shape)
        self.assertTrue(torch.isfinite(self.output).all().item())
        self.assertTrue(torch.equal(self.output[:, :2], self.fixed))
        self.assertTrue(torch.equal(self.output[:, 2:, :3], self.hoi[:, 2:, :3]))
        self.assertTrue(torch.equal(
            self.output[..., OBJECT_CHANNEL_START:],
            self.hoi[..., OBJECT_CHANNEL_START:],
        ))

    def test_local_rotation_ownership_is_exact_to_the_registered_tolerance(self):
        global_rotation = transforms.rotation_6d_to_matrix(
            self.output[..., 84:216].reshape(2, 16, 22, 6)
        )
        actual = self.dataset.quat_ik_torch(global_rotation.reshape(-1, 22, 3, 3))
        actual = actual.reshape(2, 16, 22, 3, 3)
        # History is pinned wholesale, so ownership is asserted on predicted frames.
        for joint in range(22):
            expected = self.hsi_local if joint in LOWER else self.hoi_local
            error = rotation_geodesic(
                actual[:, 2:, joint], expected[:, 2:, joint],
            )
            self.assertLess(float(error.max()), 1e-5, joint)

    def test_fk_bone_lengths_equal_rest_offsets(self):
        position = self.output[..., :84].reshape(2, 16, 28, 3)
        for joint in range(1, 22):
            parent = PARENTS_22[joint]
            actual = torch.linalg.vector_norm(
                position[:, 2:, joint] - position[:, 2:, parent], dim=-1,
            )
            expected = torch.linalg.vector_norm(
                self.offsets[:, 2:, joint], dim=-1,
            )
            self.assertLess(float((actual - expected).abs().max()), 1e-5, joint)
        for offset_index, position_index, parent in ((22, 25, 20), (23, 27, 21)):
            actual = torch.linalg.vector_norm(
                position[:, 2:, position_index] - position[:, 2:, parent], dim=-1,
            )
            expected = torch.linalg.vector_norm(
                self.offsets[:, 2:, offset_index], dim=-1,
            )
            self.assertLess(float((actual - expected).abs().max()), 1e-5)

    def test_non_fk_markers_preserve_their_hoi_parent_local_vector(self):
        output_position = self.output[..., :84].reshape(2, 16, 28, 3)
        hoi_position = self.hoi[..., :84].reshape(2, 16, 28, 3)
        output_global = transforms.rotation_6d_to_matrix(
            self.output[..., 84:216].reshape(2, 16, 22, 6)
        )
        hoi_global = transforms.rotation_6d_to_matrix(
            self.hoi[..., 84:216].reshape(2, 16, 22, 6)
        )
        for marker, parent in {22: 15, 23: 15, 24: 20, 26: 21}.items():
            before = (
                hoi_global[:, 2:, parent].transpose(-1, -2)
                @ (hoi_position[:, 2:, marker] - hoi_position[:, 2:, parent]).unsqueeze(-1)
            ).squeeze(-1)
            after = (
                output_global[:, 2:, parent].transpose(-1, -2)
                @ (output_position[:, 2:, marker] - output_position[:, 2:, parent]).unsqueeze(-1)
            ).squeeze(-1)
            self.assertLess(float((before - after).abs().max()), 1e-5, marker)

    def test_description_contains_no_tunable_parameter(self):
        description = self.composer.describe()
        self.assertEqual(description['learned_parameters'], 0)
        self.assertEqual(description['root_owner'], 'hoi')
        self.assertEqual(description['scene_query_pelvis'],
                         'shared_current_and_previous_composed_x0')
        self.assertEqual(description['compose_calls'], 1)

    def test_missing_expert_and_bad_offsets_fail_closed(self):
        with self.assertRaises(ValueError):
            self.composer(
                ExpertOutputs(hoi=self.hoi), dataset=self.dataset,
                rest_human_offsets=self.offsets, fixed_points=self.fixed,
            )
        with self.assertRaises(ValueError):
            self.composer(
                ExpertOutputs(hoi=self.hoi, hsi=self.hsi), dataset=self.dataset,
                rest_human_offsets=torch.zeros(23, 3), fixed_points=self.fixed,
            )

    def test_vectorized_kinematics_match_the_production_dataset(self):
        from datasets.infbagel import InfBaGelDataset
        from datasets.utils import get_smpl_parents

        dataset = object.__new__(InfBaGelDataset)
        dataset.parents_22 = get_smpl_parents(use_joints24=False)
        dataset.parents_24 = get_smpl_parents(use_joints24=True)
        # This makes the production normalizer an identity, matching the fixture.
        dataset.min_torch = -torch.ones(3)
        dataset.max_torch = torch.ones(3)
        output = KinematicBodyComposer()(
            ExpertOutputs(hoi=self.hoi, hsi=self.hsi), dataset=dataset,
            rest_human_offsets=self.offsets, fixed_points=self.fixed,
        )
        self.assertEqual(output.shape, self.hoi.shape)
        self.assertTrue(torch.isfinite(output).all().item())

        hoi_global = transforms.rotation_6d_to_matrix(
            self.hoi[..., 84:216].reshape(2, 16, 22, 6)
        )
        hsi_global = transforms.rotation_6d_to_matrix(
            self.hsi[..., 84:216].reshape(2, 16, 22, 6)
        )
        local = dataset.quat_ik_torch(hoi_global.reshape(-1, 22, 3, 3))
        hsi_local = dataset.quat_ik_torch(hsi_global.reshape(-1, 22, 3, 3))
        local[:, LOWER] = hsi_local[:, LOWER]
        offsets = self.offsets.reshape(-1, 24, 3).clone()
        offsets[:, 0] = self.hoi[..., :3].reshape(-1, 3)
        reference_quaternion, reference_position = dataset.quat_fk_torch(
            local, offsets,
        )
        reference_global = transforms.quaternion_to_matrix(
            reference_quaternion
        ).reshape(2, 16, 22, 3, 3)
        actual_global = transforms.rotation_6d_to_matrix(
            output[..., 84:216].reshape(2, 16, 22, 6)
        )
        rotation_error = rotation_geodesic(
            actual_global[:, 2:], reference_global[:, 2:],
        )
        self.assertLess(float(rotation_error.max()), 1e-5)

        actual_position = output[..., :84].reshape(2, 16, 28, 3)
        reference_position = reference_position.reshape(2, 16, 24, 3)
        self.assertLess(float((
            actual_position[:, 2:, :22] - reference_position[:, 2:, :22]
        ).abs().max()), 1e-5)
        for offset_index, position_index in ((22, 25), (23, 27)):
            self.assertLess(float((
                actual_position[:, 2:, position_index]
                - reference_position[:, 2:, offset_index]
            ).abs().max()), 1e-5)


class KinematicConfigTests(unittest.TestCase):
    def test_override_fragment_instantiates_the_parameter_free_composer(self):
        import hydra
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver(
            'times', lambda x, y: int(x) * int(y), replace=True,
        )
        os.environ.setdefault('ROOT_DIR', str(REPO))
        with initialize_config_dir(
            version_base=None, config_dir=str(REPO / 'code' / 'config'),
        ):
            cfg = compose(config_name='config_sample_hosi_kinematic')
        composer = hydra.utils.instantiate(cfg.sampler.pelvis.body_composer)
        self.assertIsInstance(composer, KinematicBodyComposer)
        self.assertEqual(cfg.mixer_gate, 0)
        self.assertEqual(composer.describe()['learned_parameters'], 0)

    def test_r2cg_fragments_freeze_one_pair_and_one_operator_difference(self):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        OmegaConf.register_new_resolver(
            'times', lambda x, y: int(x) * int(y), replace=True,
        )
        os.environ.setdefault('ROOT_DIR', str(REPO))
        with initialize_config_dir(
            version_base=None, config_dir=str(REPO / 'code' / 'config'),
        ):
            raw = compose(config_name='config_sample_hosi_rootsplit_r2cg')
            kinematic = compose(config_name='config_sample_hosi_kinematic_r2cg')

        expected_hash = (
            '7a81a0a2627967a396e54aa08c0bad4612e294a4df33aac9ada4b063058740fe'
        )
        for cfg in (raw, kinematic):
            self.assertEqual(cfg.hsi_checkpoint_sha256, expected_hash)
            self.assertTrue(cfg.hsi_guidance_posterior_coef1)
            self.assertEqual(cfg.hsi_guidance_scale, 1.0)
            self.assertFalse(cfg.use_guidance)
            self.assertEqual(cfg.mixer_hsi_w, 1)
            self.assertEqual(cfg.mixer_hsi_object_voxel_mode, 'occupied')
            self.assertTrue(cfg.occ_list_layout_repaired)
            self.assertFalse(cfg.sampler.pelvis.inference_engineering)

        self.assertIsNone(raw.sampler.pelvis.body_composer)
        self.assertEqual(
            raw.sampler.pelvis.gate._target_, 'mixer.gates.BodyGroupGate',
        )
        self.assertEqual(dict(raw.sampler.pelvis.gate.weights), {
            'root': 0.0, 'lower_body': 1.0, 'torso': 0.0,
            'arms': 0.0, 'hands': 0.0,
        })
        self.assertEqual(
            kinematic.sampler.pelvis.body_composer._target_,
            'mixer.kinematic_composition.KinematicBodyComposer',
        )
        self.assertEqual(kinematic.mixer_gate, 0)


if __name__ == '__main__':
    unittest.main()
