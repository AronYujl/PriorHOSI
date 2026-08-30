"""Projecting a HOSI payload into the shape `paired_bootstrap` pairs by name.

Every mixer row is compared against the sealed G=0 anchor, so the pairing is on the
critical path of every Phase 2 verdict.  Two failure modes are worth a test each:
pairing by POSITION instead of by name (which would silently compare episode i of
one row against episode i of another when the shard order differs), and keying on a
field the sealed anchor does not carry (which would make the anchor unpairable and
invite an intersection instead).
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

from hosi_per_sequence import (  # noqa: E402
    HosiPerSequenceError,
    episode_key,
    project,
)


def _record(scene, obj, idx, **metrics):
    record = {'scene_name': scene, 'object_name': obj, 'test_idx': idx,
              'completed': True, 'xy_points_err': 1.0}
    record.update(metrics)
    return record


def _payload(records, **extra):
    payload = {'individual_metrics': records, 'seed': 42, 'model_name': 'm'}
    payload.update(extra)
    return payload


class KeyTests(unittest.TestCase):
    def test_the_key_is_scene_object_index(self):
        self.assertEqual(
            episode_key(_record('Scene3', 'clothesstand', 4)),
            'Scene3/clothesstand/4',
        )

    def test_a_record_missing_an_identity_field_raises(self):
        record = _record('Scene3', 'tripod', 0)
        del record['object_name']
        with self.assertRaisesRegex(HosiPerSequenceError, 'object_name'):
            episode_key(record)

    def test_a_duplicate_key_raises_rather_than_overwriting(self):
        records = [_record('S', 'o', 0), _record('S', 'o', 0)]
        with self.assertRaisesRegex(HosiPerSequenceError, 'duplicate episode key'):
            project(_payload(records))


class ProjectionTests(unittest.TestCase):
    def test_metrics_are_keyed_by_name_not_by_position(self):
        out = project(_payload([
            _record('S', 'a', 0, xy_points_err=1.0),
            _record('S', 'b', 1, xy_points_err=2.0),
        ]))
        self.assertEqual(set(out['metrics']), {'S/a/0', 'S/b/1'})
        self.assertEqual(out['metrics']['S/b/1']['xy_points_err'], 2.0)
        self.assertEqual(out['sequence_count'], 2)

    def test_order_does_not_change_the_projection(self):
        """The property that makes a sharded row pairable against a serial one."""
        first = project(_payload([_record('S', 'a', 0), _record('S', 'b', 1)]))
        second = project(_payload([_record('S', 'b', 1), _record('S', 'a', 0)]))
        self.assertEqual(first['metrics'], second['metrics'])

    def test_completed_becomes_a_number_so_it_can_be_bootstrapped(self):
        out = project(_payload([
            _record('S', 'a', 0, completed=True),
            _record('S', 'b', 1, completed=False),
        ]))
        self.assertEqual(out['metrics']['S/a/0']['completed'], 1.0)
        self.assertEqual(out['metrics']['S/b/1']['completed'], 0.0)

    def test_identity_fields_are_not_projected_as_metrics(self):
        out = project(_payload([_record('S', 'a', 3)]))
        for field in ('scene_name', 'object_name', 'test_idx'):
            self.assertNotIn(field, out['metrics']['S/a/3'])

    def test_canonical_ordinal_is_not_the_key_and_is_not_a_metric(self):
        """The sealed anchor predates it; keying on it would unpair the anchor."""
        record = _record('S', 'a', 0)
        record['canonical_ordinal'] = 17
        out = project(_payload([record]))
        self.assertEqual(list(out['metrics']), ['S/a/0'])
        self.assertNotIn('canonical_ordinal', out['metrics']['S/a/0'])

    def test_a_string_metric_is_dropped_not_coerced(self):
        out = project(_payload([_record('S', 'a', 0, note='fell over')]))
        self.assertNotIn('note', out['metrics']['S/a/0'])

    def test_provenance_carries_both_checkpoints_and_the_composition(self):
        out = project(_payload(
            [_record('S', 'a', 0)],
            checkpoint={'sha256': 'aa'},
            hsi_checkpoint={'sha256': 'bb'},
            sampler_audit={'composition': {'gate': {'kind': 'body_group'}}},
        ))
        self.assertEqual(out['checkpoint']['sha256'], 'aa')
        self.assertEqual(out['hsi_checkpoint']['sha256'], 'bb')
        self.assertEqual(out['composition']['gate']['kind'], 'body_group')

    def test_a_non_hosi_payload_raises(self):
        with self.assertRaisesRegex(HosiPerSequenceError, 'individual_metrics'):
            project({'statistics': {}})

    def test_an_empty_episode_list_raises(self):
        with self.assertRaisesRegex(HosiPerSequenceError, 'non-empty list'):
            project(_payload([]))


class SealedAnchorTests(unittest.TestCase):
    """Against the real sealed G=0 payload, if it is present in this checkout."""

    SEALED = (REPO / 'results' / 'experiments'
              / 'p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829'
              / 'evaluation' / 'overall_evaluation_summary.json')

    def setUp(self):
        if not self.SEALED.is_file():
            self.skipTest('sealed G=0 payload is not in this checkout')

    def test_the_anchor_projects_to_469_unique_sequences(self):
        out = project(json.loads(self.SEALED.read_text()))
        self.assertEqual(out['sequence_count'], 469)
        self.assertEqual(len(out['metrics']), 469)

    def test_the_projection_preserves_the_aggregate_mean(self):
        """A projection that dropped or duplicated episodes would move this."""
        payload = json.loads(self.SEALED.read_text())
        out = project(payload)
        values = [record['xy_points_err'] for record in out['metrics'].values()]
        mean = sum(values) / len(values)
        self.assertAlmostEqual(
            mean, payload['statistics']['xy_points_err']['mean'], places=9,
        )


if __name__ == '__main__':
    unittest.main()
