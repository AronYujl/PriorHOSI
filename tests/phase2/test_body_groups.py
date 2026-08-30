"""Per-joint body-group gating: the split ChannelBlockGate cannot express.

The claims under test, in order of what would actually break a row:

1.  The groups PARTITION both blocks.  A missing joint would silently stay at
    HOI and a duplicated one would take whichever group wrote last, and neither
    shows up in a metric -- it shows up as a limb that did not move.
2.  The gate is per JOINT inside both blocks, so one weight moves a joint's
    position channels and its rotation channels together.  Gating a knee's
    position toward HSI and its rotation toward HOI is the defect this design
    exists to make unrepresentable.
3.  216:232 is not a group and cannot be reached.
4.  The layout matches the repository's own joint indices, not an assumption
    about SMPL ordering.
"""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from mixer.body_groups import (  # noqa: E402
    DEFAULT_BODY_GROUP_GATE,
    GROUP_NAMES,
    POSITION_GROUPS,
    ROTATION_GROUPS,
    body_group_channel_gate,
    describe_body_groups,
)
from mixer.composition import (  # noqa: E402
    OBJECT_CHANNEL_START,
    ExpertOutputs,
    compose_x0,
)
from mixer.gates import BodyGroupGate  # noqa: E402
from priors.core.representation import REPRESENTATION  # noqa: E402

ROTATION_START = REPRESENTATION.field('joint_rotations_6d').start


def _position_channels(joint):
    return list(range(joint * 3, (joint + 1) * 3))


def _rotation_channels(joint):
    start = ROTATION_START + joint * 6
    return list(range(start, start + 6))


class PartitionTests(unittest.TestCase):
    def test_rotation_groups_partition_22_joints(self):
        seen = [j for group in ROTATION_GROUPS.values() for j in group]
        self.assertEqual(sorted(seen), list(range(22)))
        self.assertEqual(len(seen), len(set(seen)))

    def test_position_groups_partition_28_joints(self):
        seen = [j for group in POSITION_GROUPS.values() for j in group]
        self.assertEqual(sorted(seen), list(range(28)))
        self.assertEqual(len(seen), len(set(seen)))

    def test_the_two_blocks_use_the_same_group_names(self):
        self.assertEqual(tuple(POSITION_GROUPS), GROUP_NAMES)
        self.assertEqual(tuple(ROTATION_GROUPS), GROUP_NAMES)

    def test_there_are_no_hand_rotations(self):
        """The representation stops at 22 joints, so hands are positions only.

        A gate weight on `hands` must not silently do nothing, and must not
        pretend to articulate fingers this representation cannot express.
        """
        self.assertEqual(tuple(ROTATION_GROUPS['hands']), ())
        self.assertEqual(len(POSITION_GROUPS['hands']), 4)

    def test_the_layout_matches_the_repositorys_own_joint_indices(self):
        """Cross-checked against code that is not this module.

        `eval_metrics.py:107-119` reads ankles at 7/8 and feet at 10/11;
        `priors/hoi/losses.py:335-336` reads wrists at 20/21 and the same four
        foot joints.  So those six must be lower_body and the wrists must be arms.
        """
        for joint in (7, 8, 10, 11):
            self.assertIn(joint, ROTATION_GROUPS['lower_body'])
            self.assertIn(joint, POSITION_GROUPS['lower_body'])
        for joint in (20, 21):
            self.assertIn(joint, ROTATION_GROUPS['arms'])
        # `test_infbagel_hosi.py:379-380` reads hands at position slots 24/26 and
        # `losses.py:335` at 25/27; all four are hands.
        for joint in (24, 25, 26, 27):
            self.assertIn(joint, POSITION_GROUPS['hands'])
        # The pelvis is its own group: it is the ONLY position channel that
        # reaches the metrics, via root_trans at test_infbagel_hosi.py:885.
        self.assertEqual(tuple(POSITION_GROUPS['root']), (0,))
        self.assertEqual(tuple(ROTATION_GROUPS['root']), (0,))


class ChannelGateTests(unittest.TestCase):
    def test_default_puts_root_and_legs_on_hsi_and_arms_on_hoi(self):
        gate = body_group_channel_gate()
        for joint in (0,) + tuple(ROTATION_GROUPS['lower_body']):
            for channel in _position_channels(joint) + _rotation_channels(joint):
                self.assertEqual(float(gate[channel]), 1.0, f'channel {channel}')
        for joint in ROTATION_GROUPS['arms']:
            for channel in _position_channels(joint) + _rotation_channels(joint):
                self.assertEqual(float(gate[channel]), 0.0, f'channel {channel}')
        for joint in POSITION_GROUPS['hands']:
            for channel in _position_channels(joint):
                self.assertEqual(float(gate[channel]), 0.0, f'channel {channel}')

    def test_one_weight_moves_a_joints_positions_and_rotations_together(self):
        """The defect this design makes unrepresentable.

        A knee whose position follows HSI while its rotation follows HOI is not a
        blend of two opinions; the evaluator FKs the rotations and reads the
        position only as history and occupancy, so the two would disagree about
        where the same joint is.
        """
        gate = body_group_channel_gate({'lower_body': 0.25})
        for joint in ROTATION_GROUPS['lower_body']:
            for channel in _position_channels(joint) + _rotation_channels(joint):
                self.assertAlmostEqual(float(gate[channel]), 0.25, places=6)

    def test_the_object_and_contact_channels_are_never_written(self):
        for weights in (
            {name: 1.0 for name in GROUP_NAMES},
            {'root': 1.0},
            None,
        ):
            gate = body_group_channel_gate(weights)
            with self.subTest(weights=weights):
                self.assertTrue(torch.equal(
                    gate[OBJECT_CHANNEL_START:],
                    torch.zeros(REPRESENTATION.dimension - OBJECT_CHANNEL_START),
                ))

    def test_every_human_channel_is_covered(self):
        """All 216 human channels get a value, so none defaults to HOI by omission."""
        gate = body_group_channel_gate({name: 0.5 for name in GROUP_NAMES})
        self.assertTrue(torch.equal(
            gate[:OBJECT_CHANNEL_START],
            torch.full((OBJECT_CHANNEL_START,), 0.5),
        ))

    def test_an_unknown_group_raises(self):
        with self.assertRaisesRegex(ValueError, 'unknown body group'):
            body_group_channel_gate({'left_pinky': 1.0})

    def test_an_out_of_range_weight_raises(self):
        for value in (-0.1, 1.1):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    body_group_channel_gate({'arms': value})

    def test_the_gate_honours_device_and_dtype(self):
        gate = body_group_channel_gate(dtype=torch.float64)
        self.assertEqual(gate.dtype, torch.float64)


class BodyGroupGateTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(3)
        self.hoi = torch.randn(2, 16, REPRESENTATION.dimension)
        self.hsi = torch.randn(2, 16, REPRESENTATION.dimension)

    def test_the_gate_signature_is_the_mixer_contract(self):
        gate = BodyGroupGate()
        value = gate(step=250, current=self.hoi, hoi=self.hoi, hsi=self.hsi)
        self.assertEqual(tuple(value.shape), (REPRESENTATION.dimension,))

    def test_composing_under_it_routes_each_group_to_its_expert(self):
        gate = BodyGroupGate()
        composed = compose_x0(
            ExpertOutputs(hoi=self.hoi, hsi=self.hsi),
            gate(step=1, current=self.hoi, hoi=self.hoi, hsi=self.hsi),
        )
        # Root and legs came from HSI.
        for joint in (0,) + tuple(ROTATION_GROUPS['lower_body']):
            for channel in _position_channels(joint) + _rotation_channels(joint):
                self.assertAlmostEqual(
                    float(composed[0, 5, channel]),
                    float(self.hsi[0, 5, channel]), places=5,
                )
        # Arms and hands came from HOI, bitwise.
        for joint in ROTATION_GROUPS['arms']:
            for channel in _rotation_channels(joint):
                self.assertEqual(
                    float(composed[0, 5, channel]), float(self.hoi[0, 5, channel]),
                )
        for joint in POSITION_GROUPS['hands']:
            for channel in _position_channels(joint):
                self.assertEqual(
                    float(composed[0, 5, channel]), float(self.hoi[0, 5, channel]),
                )
        # And the object block came from HOI, as it must at every gate value.
        self.assertTrue(torch.equal(
            composed[..., OBJECT_CHANNEL_START:],
            self.hoi[..., OBJECT_CHANNEL_START:],
        ))

    def test_all_groups_at_one_is_not_an_hsi_alone_row(self):
        """Even a fully-open body gate keeps the object channels at HOI."""
        gate = BodyGroupGate({name: 1.0 for name in GROUP_NAMES})
        composed = compose_x0(
            ExpertOutputs(hoi=self.hoi, hsi=self.hsi),
            gate(step=1, current=self.hoi, hoi=self.hoi, hsi=self.hsi),
        )
        self.assertTrue(torch.allclose(
            composed[..., :OBJECT_CHANNEL_START],
            self.hsi[..., :OBJECT_CHANNEL_START],
        ))
        self.assertTrue(torch.equal(
            composed[..., OBJECT_CHANNEL_START:],
            self.hoi[..., OBJECT_CHANNEL_START:],
        ))

    def test_all_groups_at_zero_lets_the_sampler_skip_the_hsi_expert(self):
        self.assertTrue(
            BodyGroupGate({name: 0.0 for name in GROUP_NAMES})
            .is_identically_zero_at(250)
        )
        self.assertFalse(BodyGroupGate().is_identically_zero_at(250))

    def test_describe_records_the_weights_and_the_membership(self):
        described = BodyGroupGate({'torso': 0.5}).describe()
        self.assertEqual(described['kind'], 'body_group')
        self.assertEqual(described['weights']['torso'], 0.5)
        self.assertEqual(described['weights']['root'], DEFAULT_BODY_GROUP_GATE['root'])
        self.assertEqual(set(described['groups']), set(GROUP_NAMES))
        # Channel counts sum to the human block, so a reader of the payload can
        # verify the partition without this module.
        total = sum(item['channels'] for item in described['groups'].values())
        self.assertEqual(total, OBJECT_CHANNEL_START)

    def test_an_unknown_group_raises(self):
        with self.assertRaisesRegex(ValueError, 'unknown body group'):
            BodyGroupGate({'tail': 1.0})

    def test_describe_is_json_serializable(self):
        import json

        json.dumps(describe_body_groups())


if __name__ == '__main__':
    unittest.main()
