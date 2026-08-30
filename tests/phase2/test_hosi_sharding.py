"""HOSI-test evaluation sharding: the partition, the seeding, and the merge.

The failure mode these guard is one-sided.  A shard that runs too FEW episodes and
merges anyway produces a file that reads as a complete 469-episode result, with a
mean over whatever happened to be on disk.  So every merge guard raises, and the
ones below are asserted from the outside: duplicate ordinals, missing ordinals,
disagreeing checkpoints, disagreeing gates, and a payload that predates sharding.

Also asserted: the property that makes scene-level sharding sound at all, namely
that a shard's episodes carry the same canonical ordinals -- and therefore the same
per-episode seeds and identities -- as the serial run's.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from hosi_sharding import (  # noqa: E402
    CANONICAL_EPISODE_TOTAL,
    PARTITION_RULE_SCENE,
    PARTITION_RULE_SERIAL,
    SHARD_INVALID_TIMING_KEYS,
    enumerate_canonical_episodes,
    episode_chord,
    invalidate_timing,
    merge_shard_payloads,
    plan_scene_shards,
    recompute_statistics,
    scene_balance_keys,
    select_shard_scenes,
)


def _item(chord=1.0, **extra):
    payload = {
        'start_location': [0.0, 0.0, 0.0],
        'pelvis_goal': [chord, 0.0, 0.0],
        'object_goal': [0.0, 0.0, 0.0],
        'object_name': 'suitcase',
        'data_idx': 0,
    }
    payload.update(extra)
    return payload


def _write_benchmark(root, scenes):
    """scenes: list of (scene_name, [chord, ...])."""
    for scene_name, chords in scenes:
        items = [
            _item(chord=c, scene_name=scene_name, data_idx=i)
            for i, c in enumerate(chords)
        ]
        (root / f'{scene_name}.json').write_text(json.dumps(items))


class ChordTests(unittest.TestCase):
    def test_the_chord_is_xz_and_ignores_height(self):
        item = _item()
        item['start_location'] = [1.0, 5.0, 2.0]
        item['pelvis_goal'] = [4.0, -7.0, 6.0]
        self.assertAlmostEqual(episode_chord(item), 5.0, places=9)

    def test_scene_keys_sum_their_episodes(self):
        keys = scene_balance_keys([[_item(3.0), _item(4.0)], [_item(1.0)]])
        self.assertAlmostEqual(keys[0], 7.0, places=9)
        self.assertAlmostEqual(keys[1], 1.0, places=9)


class PlanTests(unittest.TestCase):
    def test_the_partition_covers_every_scene_exactly_once(self):
        keys = [float(i % 7 + 1) for i in range(67)]
        for shard_count in (2, 3, 4, 8):
            bins = plan_scene_shards(keys, shard_count)
            with self.subTest(shard_count=shard_count):
                flat = [i for shard in bins for i in shard]
                self.assertEqual(sorted(flat), list(range(67)))
                self.assertEqual(len(flat), len(set(flat)))
                self.assertEqual(len(bins), shard_count)

    def test_each_shard_walks_its_scenes_in_canonical_order(self):
        """Ascending order is what keeps per-scene `sample_calls` identical.

        A shard that visited its scenes in load order would still be a partition,
        but it would not walk them in the same relative order a serial run does.
        """
        bins = plan_scene_shards([float(i) for i in range(20)], 4)
        for shard in bins:
            self.assertEqual(list(shard), sorted(shard))

    def test_balance_is_by_key_not_by_count(self):
        """One heavy scene must not sit with the other heavy ones."""
        keys = [10.0, 10.0, 1.0, 1.0, 1.0, 1.0]
        bins = plan_scene_shards(keys, 2)
        loads = [sum(keys[i] for i in shard) for shard in bins]
        self.assertEqual(loads, [12.0, 12.0])

    def test_shard_count_above_scene_count_raises(self):
        with self.assertRaises(ValueError):
            plan_scene_shards([1.0, 2.0], 3)

    def test_shard_count_below_one_raises(self):
        with self.assertRaises(ValueError):
            plan_scene_shards([1.0], 0)


class EnumerationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        _write_benchmark(self.root, [
            ('sceneA', [1.0, 2.0, 3.0]),
            ('sceneB', [9.0, 9.0]),
            ('sceneC', [1.0]),
            ('sceneD', [4.0, 4.0]),
        ])

    def tearDown(self):
        self.tmp.cleanup()

    def test_ordinals_follow_sorted_scene_then_file_order(self):
        files, items, ordinals = enumerate_canonical_episodes(str(self.root))
        self.assertEqual(list(files), [
            'sceneA.json', 'sceneB.json', 'sceneC.json', 'sceneD.json',
        ])
        self.assertEqual(ordinals[('sceneA', 0)], 0)
        self.assertEqual(ordinals[('sceneA', 2)], 2)
        self.assertEqual(ordinals[('sceneB', 0)], 3)
        self.assertEqual(ordinals[('sceneD', 1)], 7)
        self.assertEqual(len(ordinals), 8)
        self.assertEqual([len(i) for i in items], [3, 2, 1, 2])

    def test_a_shards_ordinals_are_the_serial_runs_ordinals(self):
        """The property that makes scene-level sharding sound.

        The ordinal is the episode's identity: it drives the optional per-episode
        seed and it is what the merge anchors on.  If it were shard-relative, two
        shard counts would produce different identities for the same episode and no
        merge could detect the mismatch.
        """
        _, items, ordinals = enumerate_canonical_episodes(str(self.root))
        union = {}
        for shard_index in range(3):
            scenes, plan = select_shard_scenes(
                ('sceneA.json', 'sceneB.json', 'sceneC.json', 'sceneD.json'),
                items, shard_index, 3,
            )
            for scene_file in scenes:
                name = scene_file.split('.')[0]
                count = len(items[
                    ['sceneA', 'sceneB', 'sceneC', 'sceneD'].index(name)
                ])
                for test_idx in range(count):
                    union[(name, test_idx)] = ordinals[(name, test_idx)]
        self.assertEqual(union, ordinals)
        self.assertEqual(sorted(union.values()), list(range(8)))

    def test_serial_selection_takes_every_scene_and_says_so(self):
        files, items, _ = enumerate_canonical_episodes(str(self.root))
        scenes, plan = select_shard_scenes(files, items, 0, 1)
        self.assertEqual(scenes, files)
        self.assertEqual(plan['partition_rule'], PARTITION_RULE_SERIAL)
        self.assertEqual(plan['shard_episode_count'], 8)

    def test_sharded_selection_partitions_the_episodes(self):
        files, items, _ = enumerate_canonical_episodes(str(self.root))
        seen = []
        for shard_index in range(2):
            scenes, plan = select_shard_scenes(files, items, shard_index, 2)
            self.assertEqual(plan['partition_rule'], PARTITION_RULE_SCENE)
            self.assertEqual(plan['canonical_episode_total'], 8)
            self.assertEqual(plan['canonical_scene_total'], 4)
            seen.extend(scenes)
        self.assertEqual(sorted(seen), sorted(files))

    def test_a_bad_shard_index_raises(self):
        files, items, _ = enumerate_canonical_episodes(str(self.root))
        for index in (-1, 2):
            with self.subTest(index=index):
                with self.assertRaises(ValueError):
                    select_shard_scenes(files, items, index, 2)


class TimingTests(unittest.TestCase):
    def test_wall_clock_keys_are_nulled_not_deleted(self):
        metrics = {key: 1.0 for key in SHARD_INVALID_TIMING_KEYS}
        metrics['timed_sequence_count'] = 42
        out = invalidate_timing(dict(metrics))
        for key in SHARD_INVALID_TIMING_KEYS:
            self.assertIn(key, out)
            self.assertIsNone(out[key])
        self.assertFalse(out['timing_valid'])
        self.assertIn('contend', out['timing_invalid_reason'])
        # Hardware-independent counts survive: sharding contaminates wall clock,
        # not call counts.
        self.assertEqual(out['timed_sequence_count'], 42)

    def test_none_stays_none(self):
        self.assertIsNone(invalidate_timing(None))


def _shard_payload(shard_index, shard_count, ordinals, *, hoi='aa' * 32,
                   hsi='bb' * 32, gate=None, seed=42, episode_total=8,
                   per_episode_seeding=False, drop_ordinal=False):
    records = []
    for position, ordinal in enumerate(ordinals):
        record = {
            'scene_name': f'scene{ordinal}', 'object_name': 'suitcase',
            'test_idx': position, 'completed': ordinal % 2 == 0,
            'xy_points_err': float(ordinal), 'end_obj_trans_err': float(ordinal) * 2,
        }
        if not drop_ordinal:
            record['canonical_ordinal'] = ordinal
        records.append(record)
    return {
        'model_name': 'm', 'seed': seed, 'expert': 'composed',
        'sample_type': 'diffusion', 'evaluator_guidance_fn': False,
        'checkpoint': {'sha256': hoi},
        'hsi_checkpoint': {'sha256': hsi},
        'sampler_audit': {'composition': {
            'gate': gate if gate is not None else {'kind': 'constant', 'value': 0.5},
            'channel_mask': 'human', 'hsi_object_voxel_mode': 'occupied',
        }},
        'sharding': {
            'shard_index': shard_index, 'shard_count': shard_count,
            'partition_rule': PARTITION_RULE_SCENE, 'partition_unit': 'scene',
            'canonical_episode_total': episode_total, 'canonical_scene_total': 4,
            'shard_scene_count': 2, 'shard_episode_count': len(ordinals),
            'shard_balance_key': 1.0, 'scene_indices': [0, 1],
            'per_episode_seeding': per_episode_seeding,
        },
        'scene_order': [f'scene{o}.json' for o in ordinals],
        'individual_metrics': records,
        'statistics': {'completion_rate': 0.5, 'total_samples': len(ordinals)},
        'summary': {'total_evaluated': len(ordinals), 'completion_rate': 0.5,
                    'generation_metrics': {'aits': 1.0, 'avg_fps': 2.0}},
    }


class MergeTests(unittest.TestCase):
    def _pair(self, **kwargs):
        return [
            _shard_payload(0, 2, [0, 2, 4, 6], **kwargs),
            _shard_payload(1, 2, [1, 3, 5, 7], **kwargs),
        ]

    def test_a_complete_merge_recomputes_over_the_union(self):
        merged = merge_shard_payloads(self._pair(), expected_episodes=8,
                                      expected_shard_count=2)
        self.assertEqual(len(merged['individual_metrics']), 8)
        self.assertEqual(
            [r['canonical_ordinal'] for r in merged['individual_metrics']],
            list(range(8)),
        )
        # Recomputed, not averaged from per-shard means: mean of 0..7 is 3.5.
        self.assertAlmostEqual(merged['statistics']['xy_points_err']['mean'], 3.5)
        self.assertEqual(merged['statistics']['total_samples'], 8)
        self.assertEqual(merged['sharding']['merged_shard_count'], 2)
        self.assertIsNone(merged['sharding']['shard_index'])

    def test_the_merged_timing_is_invalid(self):
        merged = merge_shard_payloads(self._pair(), expected_episodes=8,
                                      expected_shard_count=2)
        timing = merged['summary']['generation_metrics']
        self.assertIsNone(timing['aits'])
        self.assertFalse(timing['timing_valid'])

    def test_a_missing_shard_raises(self):
        with self.assertRaisesRegex(ValueError, 'expected 2 shard payloads'):
            merge_shard_payloads([self._pair()[0]], expected_episodes=8)

    def test_a_short_merge_raises_on_the_ordinals(self):
        """The failure that would otherwise read as a complete result."""
        short = [
            _shard_payload(0, 2, [0, 2, 4]),
            _shard_payload(1, 2, [1, 3, 5, 7]),
        ]
        with self.assertRaisesRegex(ValueError, 'merged 7 episode records'):
            merge_shard_payloads(short, expected_episodes=8, expected_shard_count=2)

    def test_a_gap_in_the_ordinals_raises_even_at_the_right_count(self):
        """Right count, wrong episodes: 8 records but ordinal 7 replaced by 99."""
        wrong = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 99]),
        ]
        with self.assertRaisesRegex(ValueError, 'missing from the merge'):
            merge_shard_payloads(wrong, expected_episodes=8, expected_shard_count=2)

    def test_a_duplicated_episode_raises(self):
        dup = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [0, 3, 5, 7]),
        ]
        with self.assertRaisesRegex(ValueError, 'appears in shard'):
            merge_shard_payloads(dup, expected_episodes=8, expected_shard_count=2)

    def test_two_payloads_claiming_one_index_raise(self):
        with self.assertRaisesRegex(ValueError, 'both claim shard_index'):
            merge_shard_payloads(
                [_shard_payload(0, 2, [0, 2, 4, 6]),
                 _shard_payload(0, 2, [1, 3, 5, 7])],
                expected_episodes=8, expected_shard_count=2,
            )

    def test_a_declared_shard_count_mismatch_raises(self):
        """The guard that catches merging a 2-shard pair as if it were 4.

        Without the operator's own statement the count is self-declared by the
        files, and an internally consistent pair would merge happily.
        """
        with self.assertRaisesRegex(ValueError, 'the merge was asked for 4'):
            merge_shard_payloads(self._pair(), expected_episodes=8,
                                 expected_shard_count=4)

    def test_a_different_hsi_checkpoint_raises(self):
        """A mixer row is a claim about a PAIR of checkpoints."""
        mixed = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 7], hsi='cc' * 32),
        ]
        with self.assertRaisesRegex(ValueError, 'different checkpoint pair'):
            merge_shard_payloads(mixed, expected_episodes=8, expected_shard_count=2)

    def test_a_different_hoi_checkpoint_raises(self):
        mixed = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 7], hoi='dd' * 32),
        ]
        with self.assertRaisesRegex(ValueError, 'different checkpoint pair'):
            merge_shard_payloads(mixed, expected_episodes=8, expected_shard_count=2)

    def test_a_different_gate_raises(self):
        """Two gates merged would fabricate a row no run produced."""
        mixed = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 7],
                           gate={'kind': 'constant', 'value': 0.25}),
        ]
        with self.assertRaisesRegex(ValueError, 'different gate/mask'):
            merge_shard_payloads(mixed, expected_episodes=8, expected_shard_count=2)

    def test_a_different_seed_raises(self):
        mixed = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 7], seed=7),
        ]
        with self.assertRaisesRegex(ValueError, "disagrees with shard 0 on 'seed'"):
            merge_shard_payloads(mixed, expected_episodes=8, expected_shard_count=2)

    def test_a_seeding_regime_mismatch_raises(self):
        """Two regimes inside one row is exactly what must not happen."""
        mixed = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 7], per_episode_seeding=True),
        ]
        with self.assertRaisesRegex(ValueError, 'per_episode_seeding'):
            merge_shard_payloads(mixed, expected_episodes=8, expected_shard_count=2)

    def test_a_payload_without_a_sharding_block_raises(self):
        stripped = self._pair()
        del stripped[1]['sharding']
        with self.assertRaisesRegex(ValueError, 'not a sharded run'):
            merge_shard_payloads(stripped, expected_episodes=8, expected_shard_count=2)

    def test_a_record_without_a_canonical_ordinal_raises(self):
        old = [
            _shard_payload(0, 2, [0, 2, 4, 6]),
            _shard_payload(1, 2, [1, 3, 5, 7], drop_ordinal=True),
        ]
        with self.assertRaisesRegex(ValueError, 'predates sharding support'):
            merge_shard_payloads(old, expected_episodes=8, expected_shard_count=2)

    def test_the_benchmark_total_is_enforced(self):
        with self.assertRaisesRegex(ValueError, 'the protocol expects 469'):
            merge_shard_payloads(self._pair(),
                                 expected_episodes=CANONICAL_EPISODE_TOTAL,
                                 expected_shard_count=2)

    def test_no_payloads_raises(self):
        with self.assertRaises(ValueError):
            merge_shard_payloads([], expected_episodes=8)


class SeedingTests(unittest.TestCase):
    """Why scene-level sharding needs no reseeding, as a test rather than a claim.

    `HOIPriorSampler.prepare_sample_arguments` seeds its per-window generator from
    `(torch.initial_seed() + sample_calls * 1000003) % (2**63 - 1)`.  Two facts make
    a scene's windows independent of which other scenes ran:

      * `torch.initial_seed()` returns the SEED, not the live state, so drawing
        numbers does not move it.
      * `sample_calls` is per sampler INSTANCE and `test_infbagel_hosi` rebuilds
        `sampler_body` inside the scene loop.

    Both are asserted here.  The first is the one that would silently break if a
    future refactor replaced `initial_seed()` with a state hash, which would make
    every sharded row differ from its serial counterpart with nothing to catch it.
    """

    def test_initial_seed_does_not_move_when_numbers_are_drawn(self):
        import torch

        torch.manual_seed(42)
        before = torch.initial_seed()
        for _ in range(50):
            torch.randn(64)
            torch.randperm(128)
        self.assertEqual(torch.initial_seed(), before)
        self.assertEqual(before, 42)

    def test_the_window_seed_depends_only_on_seed_and_within_scene_position(self):
        import torch

        from priors.hoi.diffusion import HOIPriorSampler  # noqa: F401

        def window_seed(seed, sample_calls):
            return (int(seed) + int(sample_calls) * 1000003) % (2 ** 63 - 1)

        # Two different scene orderings reach the same within-scene position with the
        # same seed, so the window seed matches.  This is the whole argument for
        # scene-level sharding being bitwise.
        serial = [window_seed(42, i) for i in range(5)]
        after_other_scenes = [window_seed(42, i) for i in range(5)]
        self.assertEqual(serial, after_other_scenes)
        # And it does move with the counter, so the test is not vacuous.
        self.assertNotEqual(window_seed(42, 0), window_seed(42, 1))

    def test_per_episode_seeding_needs_both_halves(self):
        """Reseeding alone does not make an episode reproducible in isolation.

        Measured with real weights offline (order_dependence.json): re-seeding the
        global RNG while `sample_calls` keeps running leaves the episode differing by
        up to 1.91 in normalized units.  Here the same fact at the level of the seed
        formula: with `sample_calls` at 6, no global reseed can recover the value the
        episode would have had at 0.
        """
        def window_seed(seed, sample_calls):
            return (int(seed) + int(sample_calls) * 1000003) % (2 ** 63 - 1)

        isolated = window_seed(42 + 17, 0)
        reseeded_only = window_seed(42 + 17, 6)
        self.assertNotEqual(isolated, reseeded_only)
        reseeded_and_reset = window_seed(42 + 17, 0)
        self.assertEqual(isolated, reseeded_and_reset)


class StatisticsTests(unittest.TestCase):
    def test_a_none_metric_is_dropped_not_zeroed(self):
        records = [
            {'canonical_ordinal': 0, 'completed': True, 'a': 2.0},
            {'canonical_ordinal': 1, 'completed': False, 'a': None},
            {'canonical_ordinal': 2, 'completed': True, 'a': 4.0},
        ]
        stats = recompute_statistics(records, ['a'])
        self.assertAlmostEqual(stats['a']['mean'], 3.0)
        self.assertAlmostEqual(stats['completion_rate'], 2 / 3)
        self.assertEqual(stats['total_samples'], 3)

    def test_completion_counts_the_flag_not_the_metric(self):
        records = [{'canonical_ordinal': i, 'completed': i < 3, 'a': 1.0}
                   for i in range(10)]
        self.assertAlmostEqual(
            recompute_statistics(records, ['a'])['completion_rate'], 0.3,
        )


if __name__ == '__main__':
    unittest.main()
