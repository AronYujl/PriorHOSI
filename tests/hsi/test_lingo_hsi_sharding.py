"""Unit tests for LINGO HSI episode sharding, latency subset and shard merging.

The merge guards are the load-bearing part: a silently short merge reads as a
complete result, so every guard is exercised against a deliberately broken
input rather than asserted to exist.
"""

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import test_infbagel_lingo_hsi as evaluator


EPISODE_DIR = REPO / "data" / "lingo_hsi_test" / "data"
FULL_EPISODES = 375
FULL_WINDOWS = 2271


def _episode(scene_name, window_count):
    return (scene_name, 0, {"episode_num": window_count})


def _record(ordinal, scene_name, sequence_index, windows):
    return {
        "canonical_ordinal": ordinal,
        "scene_name": scene_name,
        "source_sequence_index": sequence_index,
        "window_count": float(windows),
        "pen_ratio": 0.01 * ordinal,
        "sampling_seconds": None,
        "per_window_wall_seconds": None,
        "excluded_as_warmup": ordinal < 2,
    }


def _shard_payload(shard_index, shard_count, ordinals, window_counts):
    windows = sum(window_counts[ordinal] for ordinal in ordinals)
    metrics = {
        "scene%d:%06d" % (ordinal % 3, ordinal): _record(
            ordinal, "scene%d" % (ordinal % 3), ordinal, window_counts[ordinal]
        )
        for ordinal in ordinals
    }
    return {
        "schema_version": 1,
        "model_name": "unit",
        "checkpoint": {"checkpoint_sha256": "a" * 64},
        "output_dir": "/tmp/shard%d" % shard_index,
        "seed": 42,
        "sample_type": "diffusion",
        "guided": True,
        "rds": {"available": False},
        "sampling_body": "smplx_vertices_10475",
        "fps": 30.0,
        "sequence_count": len(ordinals),
        "scene_count": len({record["scene_name"] for record in metrics.values()}),
        "scene_summary": {},
        "timing": {
            "per_window_wall_seconds": None,
            "total_sampling_seconds": None,
            "window_count": windows,
            "denoiser_calls_per_window": 1000,
            "sampler_steps_per_window": 500,
            "cuda_synchronized": True,
            "batch_size": 1,
            "aits": None,
            "avg_fps": None,
            "aggregate_fps": None,
            "rtf": None,
            "total_generation_seconds": None,
            "timed_sequence_count": max(0, len(ordinals) - sum(1 for o in ordinals if o < 2)),
            "avg_frames_per_seq": None,
            "avg_end_to_end_episode_seconds": None,
            "warmup_sequences_required": 2,
            "warmup_sequences_excluded": sum(1 for ordinal in ordinals if ordinal < 2),
            "protocol_complete": False,
            "timing_valid": False,
            "timing_invalid_reason": evaluator.SHARD_TIMING_INVALID_REASON,
        },
        "sharding": {
            "shard_index": shard_index,
            "shard_count": shard_count,
            "canonical_episode_total": len(window_counts),
            "canonical_window_total": sum(window_counts),
            "shard_episode_ordinals": list(ordinals),
            "shard_window_total": windows,
            "partition_rule": "greedy_longest_first_bin_packing_by_window_count",
            "per_episode_seeding": "seed_everything(seed + canonical_ordinal)",
            "timing_valid": False,
        },
        "latency_subset": {"enabled": False},
        "metrics": metrics,
    }


WINDOW_COUNTS = (5, 2, 9, 3, 4, 2, 7, 1)


def _payload_set(shard_count=2):
    bins = evaluator.plan_episode_shards(WINDOW_COUNTS, shard_count)
    return [
        _shard_payload(index, shard_count, bins[index], WINDOW_COUNTS)
        for index in range(shard_count)
    ]


class ShardPlanningTests(unittest.TestCase):
    def test_single_shard_is_the_full_enumeration(self):
        self.assertEqual(
            evaluator.plan_episode_shards(WINDOW_COUNTS, 1), (tuple(range(8)),)
        )

    def test_partition_is_exact_and_ordinals_ascend(self):
        for shard_count in (1, 2, 3, 8):
            with self.subTest(shard_count=shard_count):
                bins = evaluator.plan_episode_shards(WINDOW_COUNTS, shard_count)
                self.assertEqual(len(bins), shard_count)
                flat = [ordinal for shard in bins for ordinal in shard]
                self.assertEqual(sorted(flat), list(range(len(WINDOW_COUNTS))))
                self.assertEqual(len(flat), len(set(flat)))
                for shard in bins:
                    self.assertEqual(list(shard), sorted(shard))

    def test_balance_is_by_window_count_not_episode_count(self):
        # One 9-window episode and three 3-window ones: an episode-balanced 2-way
        # split would give 2 episodes each and loads of 12/6.
        counts = [9, 3, 3, 3]
        bins = evaluator.plan_episode_shards(counts, 2)
        loads = [sum(counts[ordinal] for ordinal in shard) for shard in bins]
        self.assertEqual(loads, [9, 9])
        self.assertEqual(sorted(len(shard) for shard in bins), [1, 3])

    def test_determinism_across_repeated_calls(self):
        first = evaluator.plan_episode_shards(WINDOW_COUNTS, 3)
        self.assertEqual(first, evaluator.plan_episode_shards(WINDOW_COUNTS, 3))

    def test_more_shards_than_episodes_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "exceeds episode count"):
            evaluator.plan_episode_shards((1, 2), 3)

    def test_zero_shards_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shard_count must be"):
            evaluator.plan_episode_shards((1, 2), 0)

    @unittest.skipUnless(EPISODE_DIR.is_dir(), "real LINGO HSI episode dir required")
    def test_real_split_eight_way_imbalance_is_under_one_percent(self):
        episodes = evaluator._load_episodes(EPISODE_DIR, None)
        counts = [int(episode["episode_num"]) for _, _, episode in episodes]
        self.assertEqual(len(counts), FULL_EPISODES)
        self.assertEqual(sum(counts), FULL_WINDOWS)
        bins = evaluator.plan_episode_shards(counts, 8)
        loads = [sum(counts[ordinal] for ordinal in shard) for shard in bins]
        self.assertEqual(sum(loads), FULL_WINDOWS)
        self.assertLess(max(loads) / (FULL_WINDOWS / 8.0), 1.01)


class EpisodeSubsetTests(unittest.TestCase):
    def test_subset_maps_sequence_and_windows_by_canonical_ordinal(self):
        episodes = [
            ("010", 0, {"source_sequence_idx": 100, "episode_num": 2}),
            ("015", 0, {"source_sequence_idx": 200, "episode_num": 3}),
            ("024", 0, {"source_sequence_idx": 300, "episode_num": 4}),
            ("031", 0, {"source_sequence_idx": 400, "episode_num": 5}),
        ]
        payload = {
            "design": "unit",
            "total_windows": 8,
            "episodes": [
                {"canonical_ordinal": 3, "sequence_id": "031:000400", "window_count": 5},
                {"canonical_ordinal": 1, "sequence_id": "015:000200", "window_count": 3},
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "subset.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            subset = evaluator._load_episode_subset(path, episodes, [2, 3, 4, 5])

        self.assertEqual(subset["canonical_ordinals"], [3, 1])
        self.assertEqual(subset["sequence_ids"], ["031:000400", "015:000200"])
        self.assertEqual(subset["episode_count"], 2)
        self.assertEqual(subset["window_count"], 8)


class LatencySubsetTests(unittest.TestCase):
    def test_subset_spans_scenes_before_taking_a_second_episode(self):
        episodes = [
            _episode("a", 3), _episode("a", 3), _episode("a", 3),
            _episode("b", 3), _episode("b", 3),
            _episode("c", 3),
        ]
        select = evaluator.select_latency_subset(episodes, 9)
        self.assertEqual(select, (0, 3, 5))
        self.assertEqual(len({episodes[i][0] for i in select}), 3)

    def test_scene_coverage_is_monotone_in_the_budget(self):
        # The rejected plain round-robin regressed here: a larger budget let one
        # 40-window episode in early and covered fewer scenes.
        episodes = [
            _episode("a", 4), _episode("a", 40),
            _episode("b", 5), _episode("b", 36),
            _episode("c", 4), _episode("d", 3), _episode("e", 4), _episode("f", 3),
        ]
        previous = 0
        for target in range(4, 61):
            covered = len(
                {episodes[i][0] for i in evaluator.select_latency_subset(episodes, target)}
            )
            self.assertGreaterEqual(covered, previous, "budget %d" % target)
            previous = covered

    def test_cheapest_scene_is_covered_first(self):
        episodes = [_episode("a", 9), _episode("b", 2), _episode("c", 3)]
        self.assertEqual(evaluator.select_latency_subset(episodes, 5), (1, 2))

    def test_budget_is_never_exceeded_and_selection_is_deterministic(self):
        episodes = [_episode("a", 4), _episode("b", 5), _episode("c", 2), _episode("a", 2)]
        selected = evaluator.select_latency_subset(episodes, 7)
        self.assertEqual(selected, evaluator.select_latency_subset(episodes, 7))
        self.assertLessEqual(sum(episodes[i][2]["episode_num"] for i in selected), 7)

    def test_budget_below_every_episode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "smaller than every episode"):
            evaluator.select_latency_subset([_episode("a", 4)], 3)

    @unittest.skipUnless(EPISODE_DIR.is_dir(), "real LINGO HSI episode dir required")
    def test_real_split_default_budget_spans_many_scenes(self):
        episodes = evaluator._load_episodes(EPISODE_DIR, None)
        selected = evaluator.select_latency_subset(episodes, 50)
        windows = sum(int(episodes[i][2]["episode_num"]) for i in selected)
        scenes = {episodes[i][0] for i in selected}
        self.assertLessEqual(windows, 50)
        self.assertGreaterEqual(windows, 45)
        self.assertGreaterEqual(len(scenes), 13)
        self.assertEqual(selected, tuple(sorted(selected)))
        self.assertEqual(len(scenes), len(set(scenes)))

    @unittest.skipUnless(EPISODE_DIR.is_dir(), "real LINGO HSI episode dir required")
    def test_real_split_coverage_is_monotone_in_the_budget(self):
        episodes = evaluator._load_episodes(EPISODE_DIR, None)
        previous = 0
        for target in (20, 30, 40, 50, 60, 70, 80):
            covered = len(
                {episodes[i][0] for i in evaluator.select_latency_subset(episodes, target)}
            )
            self.assertGreaterEqual(covered, previous, "budget %d" % target)
            previous = covered


class TimingInvalidationTests(unittest.TestCase):
    def test_every_wall_clock_aggregate_is_nulled_and_counts_survive(self):
        timing = {
            "per_window_wall_seconds": 1.0,
            "total_sampling_seconds": 2.0,
            "aits": 3.0,
            "avg_fps": 4.0,
            "aggregate_fps": 5.0,
            "rtf": 6.0,
            "total_generation_seconds": 7.0,
            "avg_end_to_end_episode_seconds": 8.0,
            "window_count": 22,
            "denoiser_calls_per_window": 1000,
            "sampler_steps_per_window": 500,
            "protocol_complete": True,
        }
        evaluator._invalidate_timing(timing)
        for key in evaluator.SHARD_INVALID_TIMING_KEYS:
            self.assertIsNone(timing[key], key)
        self.assertIs(timing["timing_valid"], False)
        self.assertIn("contend", timing["timing_invalid_reason"])
        self.assertIs(timing["protocol_complete"], False)
        self.assertEqual(timing["window_count"], 22)
        self.assertEqual(timing["denoiser_calls_per_window"], 1000)
        self.assertEqual(timing["sampler_steps_per_window"], 500)

    def test_scene_summary_timing_means_become_explicit_nulls(self):
        summary = {"a": {"metrics_mean": {"pen_ratio": 0.1}}}
        evaluator._invalidate_scene_summary_timing(summary)
        self.assertEqual(
            summary["a"]["metrics_mean"],
            {"pen_ratio": 0.1, "sampling_seconds": None, "per_window_wall_seconds": None},
        )


class MergeTests(unittest.TestCase):
    def setUp(self):
        self.payloads = _payload_set(2)
        self.episodes = len(WINDOW_COUNTS)
        self.windows = sum(WINDOW_COUNTS)

    def merge(self, payloads=None, episodes=None, windows=None):
        return evaluator.merge_shard_payloads(
            self.payloads if payloads is None else payloads,
            expected_episodes=self.episodes if episodes is None else episodes,
            expected_windows=self.windows if windows is None else windows,
        )

    def test_merge_reproduces_the_full_protocol_counts(self):
        merged = self.merge()
        self.assertEqual(merged["sequence_count"], self.episodes)
        self.assertEqual(len(merged["metrics"]), self.episodes)
        self.assertEqual(merged["timing"]["window_count"], self.windows)
        self.assertEqual(
            [record["canonical_ordinal"] for record in merged["metrics"].values()],
            list(range(self.episodes)),
        )
        self.assertNotIn("output_dir", merged)

    def test_merged_timing_is_invalid_and_call_counts_survive(self):
        merged = self.merge()
        for key in evaluator.SHARD_INVALID_TIMING_KEYS:
            self.assertIsNone(merged["timing"][key], key)
        self.assertIs(merged["timing"]["timing_valid"], False)
        self.assertIs(merged["sharding"]["timing_valid"], False)
        self.assertEqual(merged["timing"]["denoiser_calls_per_window"], 1000)
        self.assertEqual(merged["timing"]["sampler_steps_per_window"], 500)

    def test_warmup_is_canonical_not_per_shard(self):
        merged = self.merge()
        flagged = [
            record["canonical_ordinal"]
            for record in merged["metrics"].values()
            if record["excluded_as_warmup"]
        ]
        self.assertEqual(flagged, [0, 1])
        self.assertEqual(merged["timing"]["warmup_sequences_excluded"], 2)

    def test_scene_aggregation_is_recomputed_over_the_union(self):
        merged = self.merge()
        counted = sum(
            summary["sequence_count"] for summary in merged["scene_summary"].values()
        )
        self.assertEqual(counted, self.episodes)
        for summary in merged["scene_summary"].values():
            self.assertIsNone(summary["metrics_mean"]["sampling_seconds"])
            self.assertIsNone(summary["metrics_mean"]["per_window_wall_seconds"])

    def test_merged_payload_is_json_serializable_without_nan(self):
        text = json.dumps(
            evaluator._sanitize_json(self.merge()), allow_nan=False, default=evaluator._json_value
        )
        self.assertIn('"timing_valid": false', text)

    def test_subset_merge_accepts_non_contiguous_canonical_ordinals(self):
        ordinals = (1, 4, 7)
        subset = {
            "enabled": True,
            "path": "/tmp/unit-subset.json",
            "sha256": "b" * 64,
            "design": "unit",
            "canonical_ordinals": list(ordinals),
            "sequence_ids": ["scene1:000001", "scene1:000004", "scene1:000007"],
            "episode_count": len(ordinals),
            "window_count": sum(WINDOW_COUNTS[index] for index in ordinals),
        }
        payloads = [
            _shard_payload(0, 2, (1, 7), WINDOW_COUNTS),
            _shard_payload(1, 2, (4,), WINDOW_COUNTS),
        ]
        for payload in payloads:
            payload["episode_subset"] = subset
            payload["future_occ_diagnostic"] = {
                "mode": "predicted",
                "offsets": [5, 10, 15],
            }

        merged = evaluator.merge_shard_payloads(
            payloads,
            expected_episodes=len(ordinals),
            expected_windows=subset["window_count"],
        )

        self.assertEqual(
            [record["canonical_ordinal"] for record in merged["metrics"].values()],
            list(ordinals),
        )
        self.assertEqual(merged["sharding"]["canonical_episode_total"], len(WINDOW_COUNTS))
        self.assertEqual(merged["sharding"]["eligible_episode_count"], len(ordinals))
        self.assertEqual(merged["sharding"]["eligible_window_total"], subset["window_count"])

    # --- guards, each fired on a deliberately broken input -----------------
    def test_no_payloads_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "no shard payloads"):
            self.merge(payloads=[])

    def test_missing_shard_is_rejected_rather_than_merged_short(self):
        with self.assertRaisesRegex(ValueError, r"expected 2 shard payloads, received 1"):
            self.merge(payloads=self.payloads[:1])

    def test_missing_index_with_right_count_is_rejected(self):
        payloads = [self.payloads[0], copy.deepcopy(self.payloads[0])]
        payloads[1]["metrics"] = {}
        with self.assertRaisesRegex(ValueError, "both claim shard_index=0"):
            self.merge(payloads=payloads)

    def test_index_gap_names_the_missing_shard(self):
        payloads = _payload_set(3)
        payloads[2]["sharding"]["shard_index"] = 5
        with self.assertRaisesRegex(ValueError, r"shard indices \[2\] are missing"):
            evaluator.merge_shard_payloads(
                payloads, expected_episodes=self.episodes, expected_windows=self.windows
            )

    def test_episode_count_shortfall_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "protocol expects 99"):
            self.merge(episodes=99)

    def test_window_count_shortfall_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "canonical windows, protocol expects 999"):
            self.merge(windows=999)

    def test_duplicate_sequence_key_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        stolen = sorted(payloads[0]["metrics"])[0]
        payloads[1]["metrics"][stolen] = payloads[0]["metrics"][stolen]
        with self.assertRaisesRegex(ValueError, "duplicate sequence key"):
            self.merge(payloads=payloads)

    def test_missing_canonical_ordinal_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        dropped = sorted(payloads[1]["metrics"])[0]
        record = payloads[1]["metrics"].pop(dropped)
        payloads[1]["timing"]["window_count"] -= int(record["window_count"])
        with self.assertRaisesRegex(ValueError, "protocol expects 8"):
            self.merge(payloads=payloads)

    def test_ordinal_hole_with_matching_totals_is_still_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        shard0_ordinals = [
            record["canonical_ordinal"] for record in payloads[0]["metrics"].values()
        ]
        victim = max(
            payloads[1]["metrics"],
            key=lambda key: payloads[1]["metrics"][key]["canonical_ordinal"],
        )
        self.assertNotIn(
            payloads[1]["metrics"][victim]["canonical_ordinal"], shard0_ordinals
        )
        payloads[1]["metrics"][victim]["canonical_ordinal"] = shard0_ordinals[0]
        with self.assertRaisesRegex(ValueError, "appear in more than one shard"):
            self.merge(payloads=payloads)

    def test_mismatched_checkpoint_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        payloads[1]["checkpoint"]["checkpoint_sha256"] = "b" * 64
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            self.merge(payloads=payloads)

    def test_mismatched_seed_or_guidance_is_rejected(self):
        for key, value in (("seed", 43), ("guided", False), ("sample_type", "consistency")):
            with self.subTest(key=key):
                payloads = copy.deepcopy(self.payloads)
                payloads[1][key] = value
                with self.assertRaisesRegex(ValueError, "disagrees with shard 0 on %r" % key):
                    self.merge(payloads=payloads)

    def test_mismatched_denoiser_call_count_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        payloads[1]["timing"]["denoiser_calls_per_window"] = 32
        with self.assertRaisesRegex(ValueError, "denoiser_calls_per_window"):
            self.merge(payloads=payloads)

    def test_mismatched_canonical_totals_are_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        payloads[1]["sharding"]["canonical_window_total"] = 999
        with self.assertRaisesRegex(ValueError, "canonical_window_total"):
            self.merge(payloads=payloads)

    def test_unsharded_payload_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        payloads[1].pop("sharding")
        with self.assertRaisesRegex(ValueError, "no 'sharding' block"):
            self.merge(payloads=payloads)

    def test_shard_count_disagreement_is_rejected(self):
        payloads = copy.deepcopy(self.payloads)
        payloads[1]["sharding"]["shard_count"] = 3
        with self.assertRaisesRegex(ValueError, "disagree on shard_count"):
            self.merge(payloads=payloads)

    def test_requesting_more_shards_than_the_payloads_declare_is_rejected(self):
        # The defect this pins: merge_shards with shard_count=8 over a directory
        # holding a self-consistent 2-shard pair used to succeed.
        with self.assertRaisesRegex(
            ValueError, r"declare shard_count=2, the merge was asked for 8"
        ):
            evaluator.merge_shard_payloads(
                self.payloads,
                expected_episodes=self.episodes,
                expected_windows=self.windows,
                expected_shard_count=8,
            )

    def test_matching_requested_shard_count_is_accepted(self):
        merged = evaluator.merge_shard_payloads(
            self.payloads,
            expected_episodes=self.episodes,
            expected_windows=self.windows,
            expected_shard_count=2,
        )
        self.assertEqual(merged["sharding"]["merged_shard_count"], 2)


if __name__ == "__main__":
    unittest.main()
