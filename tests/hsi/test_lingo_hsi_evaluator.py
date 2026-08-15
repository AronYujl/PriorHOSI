"""Regression tests for strict LINGO HSI evaluator checkpoint loading."""

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

import test_infbagel_lingo_hsi as evaluator


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


if __name__ == "__main__":
    unittest.main()
