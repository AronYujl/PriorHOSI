"""Regression tests for strict LINGO HSI evaluator checkpoint loading and for the
SMPL-X coordinate frame both of its arms feed forward kinematics in."""

import ast
import os
import random
import sys
import unittest
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import pytorch3d.transforms as transforms

import test_infbagel_lingo_hsi as evaluator
from datasets.infbagel import InfBaGelDataset
from datasets.utils import get_smpl_parents
from utils import SMPLX_JOINTS_28, create_smplx_model, run_smplx_model, yup_to_zup, zup_to_yup

SMPL_MODEL_DIR = REPO / "smpl_models"
EVALUATOR_SOURCE = REPO / "code" / "test_infbagel_lingo_hsi.py"


HAND_KEYS = tuple("embedding_hand_goal.layer%d" % index for index in range(4))
SCENE_KEYS = tuple("embedding_scene_goal.layer%d" % index for index in range(4))
RELEASED_CHECKPOINT = REPO / "checkpoint" / "checkpoint.pth"
RUN_B_CHECKPOINT = (
    REPO
    / "results"
    / "hsi_b_lingo_full"
    / "checkpoints"
    / "hsi_b_lingo_full_epoch222.pth"
)


def _synthetic_checkpoint(goal_keys):
    state = OrderedDict((key, torch.tensor([index])) for index, key in enumerate(goal_keys))
    state["denoiser.weight"] = torch.ones(2, 3)
    return state


TIMING_INPUT = (
    {"gen_seconds": 1.0, "frames": 30, "episode_seconds": 2.0},
    {"gen_seconds": 2.0, "frames": 60, "episode_seconds": 3.0},
    {"gen_seconds": 1.0, "frames": 30, "episode_seconds": 2.0},
    {"gen_seconds": 2.0, "frames": 30, "episode_seconds": 4.0},
    {"gen_seconds": 4.0, "frames": 60, "episode_seconds": 5.0},
)


class TimingAggregationTests(unittest.TestCase):
    def test_registered_warmup_and_aggregates_have_exact_values(self):
        result = evaluator._aggregate_timing(
            TIMING_INPUT, warmup_sequences=2, fps=30.0
        )

        self.assertAlmostEqual(result["aits"], 7.0 / 3.0)
        self.assertAlmostEqual(result["avg_fps"], 20.0)
        self.assertAlmostEqual(result["aggregate_fps"], 120.0 / 7.0)
        self.assertAlmostEqual(result["rtf"], 4.0 / 7.0)
        self.assertAlmostEqual(result["total_generation_seconds"], 7.0)
        self.assertEqual(result["timed_sequence_count"], 3)
        self.assertAlmostEqual(result["avg_frames_per_seq"], 40.0)
        self.assertAlmostEqual(result["avg_end_to_end_episode_seconds"], 11.0 / 3.0)
        self.assertEqual(result["warmup_sequences_excluded"], 2)
        self.assertIs(result["protocol_complete"], True)

    def test_macro_and_aggregate_fps_are_distinct(self):
        result = evaluator._aggregate_timing(
            TIMING_INPUT, warmup_sequences=2, fps=30.0
        )
        self.assertAlmostEqual(result["avg_fps"], 20.0)
        self.assertAlmostEqual(result["aggregate_fps"], 120.0 / 7.0)
        self.assertNotEqual(result["avg_fps"], result["aggregate_fps"])

    def test_warmup_at_or_above_sequence_count_leaves_no_timed_data(self):
        aggregate_keys = (
            "aits",
            "avg_fps",
            "aggregate_fps",
            "rtf",
            "total_generation_seconds",
            "avg_frames_per_seq",
            "avg_end_to_end_episode_seconds",
        )
        for requested in (5, 9):
            with self.subTest(requested=requested):
                result = evaluator._aggregate_timing(
                    TIMING_INPUT, warmup_sequences=requested, fps=30.0
                )
                self.assertIs(result["protocol_complete"], False)
                self.assertEqual(result["timed_sequence_count"], 0)
                self.assertEqual(result["warmup_sequences_excluded"], 5)
                for key in aggregate_keys:
                    self.assertIsNone(result[key], key)

    def test_zero_warmup_drops_nothing(self):
        result = evaluator._aggregate_timing(
            TIMING_INPUT, warmup_sequences=0, fps=30.0
        )
        self.assertEqual(result["warmup_sequences_excluded"], 0)
        self.assertEqual(result["timed_sequence_count"], len(TIMING_INPUT))
        self.assertAlmostEqual(result["total_generation_seconds"], 10.0)


class RngRewindTests(unittest.TestCase):
    def test_paired_draws_match_and_global_stream_advances_once(self):
        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        expected_first = (
            torch.randn(4),
            np.random.randn(4),
            random.random(),
        )
        expected_second = (
            torch.randn(4),
            np.random.randn(4),
            random.random(),
        )

        torch.manual_seed(1234)
        np.random.seed(1234)
        random.seed(1234)
        pre_state = evaluator._capture_rng_state()
        scene_draw = (torch.randn(4), np.random.randn(4), random.random())
        with evaluator._rng_rewound(pre_state):
            null_draw = (torch.randn(4), np.random.randn(4), random.random())
        next_draw = (torch.randn(4), np.random.randn(4), random.random())

        self.assertTrue(torch.equal(scene_draw[0], expected_first[0]))
        self.assertTrue(torch.equal(null_draw[0], scene_draw[0]))
        self.assertTrue(torch.equal(next_draw[0], expected_second[0]))
        np.testing.assert_array_equal(scene_draw[1], expected_first[1])
        np.testing.assert_array_equal(null_draw[1], scene_draw[1])
        np.testing.assert_array_equal(next_draw[1], expected_second[1])
        self.assertEqual(scene_draw[2], expected_first[2])
        self.assertEqual(null_draw[2], scene_draw[2])
        self.assertEqual(next_draw[2], expected_second[2])


class CheckpointKeyRemapTests(unittest.TestCase):
    def test_hand_goal_keys_are_remapped_in_order(self):
        state = _synthetic_checkpoint(HAND_KEYS)
        remapped, remap_count = evaluator._remap_checkpoint_keys(state)

        self.assertEqual(remap_count, 4)
        self.assertEqual(list(remapped), list(SCENE_KEYS) + ["denoiser.weight"])
        self.assertIs(remapped["denoiser.weight"], state["denoiser.weight"])

    def test_scene_goal_keys_are_an_identity_mapping(self):
        state = _synthetic_checkpoint(SCENE_KEYS)
        remapped, remap_count = evaluator._remap_checkpoint_keys(state)

        self.assertEqual(remap_count, 0)
        self.assertEqual(list(remapped), list(state))
        for key in state:
            self.assertIs(remapped[key], state[key])

    def test_mirror_orientations_produce_identical_key_sets(self):
        hand_remapped, _ = evaluator._remap_checkpoint_keys(
            _synthetic_checkpoint(HAND_KEYS)
        )
        scene_remapped, _ = evaluator._remap_checkpoint_keys(
            _synthetic_checkpoint(SCENE_KEYS)
        )
        self.assertEqual(set(hand_remapped), set(scene_remapped))

    def test_module_prefix_is_stripped_from_every_key(self):
        state = OrderedDict(
            ("module.%s" % key, value)
            for key, value in _synthetic_checkpoint(SCENE_KEYS).items()
        )
        remapped, remap_count = evaluator._remap_checkpoint_keys(state)

        self.assertEqual(remap_count, 0)
        self.assertEqual(
            list(remapped), list(SCENE_KEYS) + ["denoiser.weight"]
        )

    def test_both_embedding_orientations_collide(self):
        state = OrderedDict(
            (
                ("embedding_hand_goal.x", torch.tensor([1.0])),
                ("embedding_scene_goal.x", torch.tensor([2.0])),
            )
        )
        with self.assertRaises(KeyError):
            evaluator._remap_checkpoint_keys(state)


class StrictKeySetTests(unittest.TestCase):
    def setUp(self):
        self.model = torch.nn.Linear(3, 2)
        self.expected = list(self.model.state_dict())

    def test_missing_key_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"missing=1"):
            evaluator._assert_key_sets_match(self.expected[:-1], self.expected)

    def test_extra_key_is_rejected(self):
        with self.assertRaisesRegex(RuntimeError, r"unexpected=1"):
            evaluator._assert_key_sets_match(
                self.expected + ["bogus.weight"], self.expected
            )


@unittest.skipUnless(
    RELEASED_CHECKPOINT.is_file() and RUN_B_CHECKPOINT.is_file(),
    "both real evaluator checkpoints are required",
)
class RealCheckpointTests(unittest.TestCase):
    def test_both_checkpoint_orientations_have_the_same_shapes(self):
        released = torch.load(RELEASED_CHECKPOINT, map_location="cpu")
        run_b = torch.load(RUN_B_CHECKPOINT, map_location="cpu")

        released_remapped, released_count = evaluator._remap_checkpoint_keys(released)
        run_b_remapped, run_b_count = evaluator._remap_checkpoint_keys(run_b)

        self.assertEqual(released_count, 4)
        self.assertEqual(run_b_count, 0)
        self.assertEqual(set(released_remapped), set(run_b_remapped))
        for key in released_remapped:
            self.assertEqual(
                tuple(released_remapped[key].shape), tuple(run_b_remapped[key].shape), key
            )

    def test_strict_loader_accepts_both_real_checkpoint_orientations(self):
        os.environ["ROOT_DIR"] = str(REPO)
        cases = (
            (RELEASED_CHECKPOINT, "hand_goal", 4),
            (RUN_B_CHECKPOINT, "scene_goal", 0),
        )
        for checkpoint_path, orientation, remap_count in cases:
            with self.subTest(checkpoint_path=checkpoint_path):
                with initialize_config_dir(
                    version_base=None, config_dir=str(REPO / "code" / "config")
                ):
                    cfg = compose(
                        config_name="config_sample_infbagel_lingo_hsi",
                        overrides=[
                            "device=cpu",
                            "ckpt_path=%s" % checkpoint_path,
                        ],
                    )
                model, provenance = evaluator._load_strict_checkpoint(cfg)
                self.assertEqual(provenance["embedding_orientation"], orientation)
                self.assertEqual(provenance["remapped_tensor_count"], remap_count)
                del model


class _RotationChain:
    """Binds the dataset's real IK/FK rotation helpers without loading any data.

    ``quat_ik_torch`` and ``local2global_pose`` read nothing but ``parents_22``,
    so the evaluator's actual decode arithmetic can be exercised here rather than
    reimplemented -- a reimplementation would pass while the real chain broke.
    """

    def __init__(self):
        self.parents_22 = get_smpl_parents(use_joints24=False)

    quat_ik_torch = InfBaGelDataset.quat_ik_torch
    local2global_pose = InfBaGelDataset.local2global_pose


def _fixture_local_axis_angle(frames=5, seed=0):
    """A small, fully determined set of local axis-angles in SMPL-X template frame.

    Deliberately not near-zero and not near pi: the magnitudes below keep every
    joint away from the axis-angle branch cut so the round trip under test is the
    frame algebra alone, with no wrap ambiguity to mask a sign error.
    """
    generator = torch.Generator().manual_seed(seed)
    axis = torch.randn(frames, 22, 3, generator=generator)
    axis = axis / axis.norm(dim=-1, keepdim=True)
    angle = 0.3 + 0.9 * torch.rand(frames, 22, 1, generator=generator)
    return (axis * angle).to(torch.float64).to(torch.float32)


class SmplxFrameSourceTests(unittest.TestCase):
    """Pin the coordinate path of both arms by reading the evaluator's own AST.

    The FK block only runs inside a GPU rollout, so the property that both arms
    reach SMPL-X through one frame cannot be observed from a unit test by calling
    it.  Parsing the source is the honest alternative to not testing it: it fails
    loudly if anyone reintroduces a transform on the translation or on the FK
    output, which is exactly the 2026-08-18 defect.
    """

    @classmethod
    def setUpClass(cls):
        cls.tree = ast.parse(EVALUATOR_SOURCE.read_text(encoding="utf-8"))
        cls.functions = {
            node.name: node
            for node in ast.walk(cls.tree)
            if isinstance(node, ast.FunctionDef)
        }

    @staticmethod
    def _declared_transforms(function):
        """Every ``smplx_output_transform`` string literal set inside ``function``."""
        found = []
        for node in ast.walk(function):
            if isinstance(node, ast.Dict):
                for key, value in zip(node.keys, node.values):
                    if (
                        isinstance(key, ast.Constant)
                        and key.value == "smplx_output_transform"
                        and isinstance(value, ast.Constant)
                    ):
                        found.append(value.value)
        return found

    @staticmethod
    def _assigned_value(function, target_name):
        for node in ast.walk(function):
            if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == target_name for t in node.targets
            ):
                return node.value
        raise AssertionError("no assignment to %r in %s" % (target_name, function.name))

    @staticmethod
    def _called_names(node):
        return {
            child.func.id
            for child in ast.walk(node)
            if isinstance(child, ast.Call) and isinstance(child.func, ast.Name)
        }

    def test_both_arms_declare_the_identity_output_transform(self):
        generated = self._declared_transforms(self.functions["sampled_motion"])
        real = self._declared_transforms(self.functions["ground_truth_motion"])

        self.assertEqual(generated, ["identity"])
        self.assertEqual(real, ["identity"])
        self.assertEqual(generated, real)

    def test_generated_arm_fk_translation_takes_no_coordinate_transform(self):
        value = self._assigned_value(self.functions["sampled_motion"], "smpl_translation")

        # The pelvis must be handed to SMPL-X in the frame it is already in.  A
        # yup_to_zup here is the 2026-08-18 defect: it displaces the pelvis by
        # zup_to_yup(J_rest0) - J_rest0 = (0, +0.3795, +0.3542) m.
        self.assertEqual(self._called_names(value) & {"yup_to_zup", "zup_to_yup"}, set())

    def test_generated_arm_pose_takes_no_coordinate_transform(self):
        value = self._assigned_value(self.functions["sampled_motion"], "smpl_pose")

        # Superseded spec.  Until 2026-08-18 this assertion *required* yup_to_zup
        # here, because datasets/infbagel.py conjugated human_orient/human_pose
        # with zup_to_yup and the decoded locals carried M^T R M.  That
        # conjugation is gone -- it rotated the SMPL template, not the world, and
        # cancelled only for OMOMO -- so the decoded locals are already in the
        # template frame and any transform here is now the error.  The
        # independent-reference form of this property (FK from the composed
        # channel must reproduce human_joints_aligned.npy) lives in
        # tests/hsi/test_representation_frame.py; this AST check only pins the
        # source, because the FK block runs solely inside a GPU rollout.
        self.assertEqual(self._called_names(value) & {"yup_to_zup", "zup_to_yup"}, set())

    def test_generated_arm_reaches_smplx_through_a_single_frame(self):
        # ast.unparse needs 3.9; this environment is older, so walk the calls.
        called = self._called_names(self.functions["sampled_motion"])

        # Neither direction appears anywhere in the generated arm: pose,
        # translation and FK output all live in the one y-up world the dataset
        # serves, so any occurrence is a reintroduced frame hop.
        self.assertNotIn("zup_to_yup", called)
        self.assertNotIn("yup_to_zup", called)

    def test_schema_versions_record_the_frame_correction(self):
        self.assertEqual(evaluator.METRICS_SCHEMA_VERSION, 4)
        self.assertEqual(evaluator.MOTION_EXPORT_SCHEMA_VERSION, 3)


class SmplxFrameRoundTripTests(unittest.TestCase):
    """The frame algebra behind the fix, on a small fixture and with no assets.

    ``interp_jrot`` is deliberately excluded: ``utils.quaternion_slerp`` swaps its
    LERP weights (``q1 * step + q2 * (1 - step)``) and takes that branch whenever
    ``dot > 1 - 1e-6``, so the interpolator is not an identity even at scale 1 and
    would mask the property under test.  That is a separate, pre-existing defect.
    """

    def test_identity_decode_recovers_the_template_frame_locals(self):
        chain = _RotationChain()
        local_raw = _fixture_local_axis_angle()

        # what datasets/infbagel.py writes into the rotation channel after the
        # 2026-08-18 frame correction: the composed globals of the *untouched*
        # template-frame locals, with a world change of basis on the root only
        # (identity for LINGO, whose human_orient is already y-up)
        global_6d = transforms.matrix_to_rotation_6d(
            chain.local2global_pose(transforms.axis_angle_to_matrix(local_raw))
        )

        # the generated arm's decode, verbatim minus the interpolation
        decoded = chain.quat_ik_torch(
            transforms.rotation_6d_to_matrix(global_6d.reshape(-1, 22, 6))
        )
        recovered = transforms.matrix_to_axis_angle(decoded).reshape(-1, 22, 3)

        self.assertTrue(
            torch.allclose(recovered, local_raw, atol=2e-6),
            "max deviation %.3e" % float((recovered - local_raw).abs().max()),
        )

    def test_a_yup_to_zup_on_the_decode_is_measurably_wrong(self):
        """The rejected alternative must be measurably wrong, not merely different.

        Without this the round-trip test above would also pass if yup_to_zup were
        a no-op on this fixture.  It is the transform the pre-2026-08-18 chain
        applied, so this is the regression that would resurface first.
        """
        chain = _RotationChain()
        local_raw = _fixture_local_axis_angle()
        global_6d = transforms.matrix_to_rotation_6d(
            chain.local2global_pose(transforms.axis_angle_to_matrix(local_raw))
        )
        local_axis = transforms.matrix_to_axis_angle(
            chain.quat_ik_torch(transforms.rotation_6d_to_matrix(global_6d.reshape(-1, 22, 6)))
        ).reshape(-1, 22, 3)

        self.assertGreater(float((yup_to_zup(local_axis) - local_raw).abs().max()), 0.1)


@unittest.skipUnless(SMPL_MODEL_DIR.is_dir(), "smpl_models is required for SMPL-X FK")
class SmplxPelvisInvariantTests(unittest.TestCase):
    """FK under the corrected path must put the pelvis exactly where asked.

    ``translation_offset`` is ``source.transl[start] - source.joints[start, 0]``,
    which is ``-J_rest0``, so ``transl = pelvis - J_rest0`` and SMPL-X's root joint
    comes back at ``pelvis``.  This invariant is what the pre-fix path broke, and
    it holds only when neither the translation nor the FK output is transformed.
    """

    @classmethod
    def setUpClass(cls):
        # Scope grad suppression to each call.  torch.set_grad_enabled(False) here
        # would leak out of this module and break every backward() in tests/core.
        cls.gender = "male"
        cls.model = create_smplx_model(cls.gender, torch.device("cpu"), batch_size=1)
        cls.betas = torch.zeros(16)
        with torch.no_grad():
            _vertices, joints = run_smplx_model(
                torch.zeros(1, 22, 3),
                torch.zeros(1, 3),
                cls.betas[None],
                cls.gender,
                joints_ind=SMPLX_JOINTS_28,
                smpl_model=cls.model,
            )
        cls.rest_pelvis = joints[0, 0].clone()

    def _fk_pelvis(self, pose, translation):
        with torch.no_grad():
            _vertices, joints = run_smplx_model(
                pose,
                translation,
                self.betas[None].repeat(pose.shape[0], 1),
                self.gender,
                joints_ind=SMPLX_JOINTS_28,
                smpl_model=self.model,
            )
        return joints[:, 0]

    def test_untransformed_translation_lands_the_pelvis_on_the_requested_point(self):
        pose = _fixture_local_axis_angle(frames=4, seed=1)
        pelvis = torch.tensor(
            [[0.4, 0.95, -1.3], [0.5, 0.9, -1.1], [0.6, 1.0, -0.9], [0.7, 0.92, -0.7]]
        )
        offset = -self.rest_pelvis

        got = self._fk_pelvis(pose, pelvis + offset)

        self.assertTrue(
            torch.allclose(got, pelvis, atol=1e-6),
            "max deviation %.3e" % float((got - pelvis).abs().max()),
        )

    def test_the_prefix_transform_pair_displaces_the_pelvis_by_the_measured_vector(self):
        """Nails the defect's signature so a regression is recognisable, not just red."""
        pose = _fixture_local_axis_angle(frames=4, seed=1)
        pelvis = torch.tensor(
            [[0.4, 0.95, -1.3], [0.5, 0.9, -1.1], [0.6, 1.0, -0.9], [0.7, 0.92, -0.7]]
        )
        offset = -self.rest_pelvis

        broken = zup_to_yup(
            self._fk_pelvis(pose, yup_to_zup(pelvis + offset)).clone()
        )
        expected = pelvis + (
            zup_to_yup(self.rest_pelvis.clone()) - self.rest_pelvis
        )[None]

        self.assertTrue(
            torch.allclose(broken, expected, atol=1e-6),
            "max deviation %.3e" % float((broken - expected).abs().max()),
        )
        # and it is a real displacement, not a rounding difference
        self.assertGreater(float((broken - pelvis).norm(dim=-1).min()), 0.5)


if __name__ == "__main__":
    unittest.main()
