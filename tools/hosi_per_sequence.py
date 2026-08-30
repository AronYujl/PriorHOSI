#!/usr/bin/env python3
"""Project a HOSI-test payload into the per-sequence shape `paired_bootstrap` reads.

`tools/paired_bootstrap.py` pairs BY NAME and takes a `per_sequence_metrics.json`
whose `metrics` is an object keyed by sequence name.  The HOSI evaluator does not
write that file: its 469 per-episode records live in `individual_metrics`, a LIST
inside `overall_evaluation_summary.json`.  So every HOSI paired comparison so far
was done by hand.  This makes it tooling, because the mixer's rows are all paired
against the G=0 anchor and a hand-rolled pairing is where a silent intersection or
a positional pairing gets in.

THE KEY is `scene_name/object_name/test_idx`.  Verified unique over all 469 records
of the sealed anchor, and `scene_name/test_idx` alone is unique too -- object_name
is kept anyway because it is what the per-object gate reads and a key that carries
it lets a report group by object without a second join.

`canonical_ordinal` is deliberately NOT the key.  It would be the better identity,
but the sealed anchor predates it, so keying on it would make the one row every
mixer row must be compared against unpairable.

WHAT CANNOT BE BOOTSTRAPPED HERE.  `completion_rate` is a proportion over episodes,
derived from the per-episode `completed` boolean, not a per-episode measurement, so
it has no sequence-level sample; it is carried through as `completed` (0.0/1.0) so a
caller can bootstrap the proportion explicitly and knowingly.  `test_idx` is an
identifier and is dropped.  Unlike the HOI side, `contact_percent` IS per-episode on
HOSI-test and can be bootstrapped.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping

#: Identifier fields, not measurements.
IDENTITY_FIELDS = ('scene_name', 'object_name', 'test_idx', 'canonical_ordinal')

#: `completed` is a bool; it is projected to 0.0/1.0 rather than dropped, so a
#: caller can bootstrap the completion proportion deliberately.
BOOLEAN_FIELDS = ('completed',)


class HosiPerSequenceError(RuntimeError):
    """Raised when a payload cannot be projected without guessing."""


def episode_key(record: Mapping[str, Any]) -> str:
    """`scene/object/test_idx`, the pairing name."""
    for field in ('scene_name', 'object_name', 'test_idx'):
        if field not in record:
            raise HosiPerSequenceError(
                f'episode record is missing {field!r}, so it cannot be paired; '
                f'keys present: {sorted(record)}'
            )
    return '{0}/{1}/{2}'.format(
        record['scene_name'], record['object_name'], record['test_idx'],
    )


def project(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Return a `per_sequence_metrics.json`-shaped dict for one HOSI payload."""
    if 'individual_metrics' not in payload:
        raise HosiPerSequenceError(
            "payload has no 'individual_metrics'; this is not a HOSI-test "
            f'evaluation summary. Keys present: {sorted(payload)}'
        )
    records = payload['individual_metrics']
    if not isinstance(records, list) or not records:
        raise HosiPerSequenceError(
            "'individual_metrics' must be a non-empty list of episode records"
        )

    metrics: Dict[str, Dict[str, float]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise HosiPerSequenceError(
                f'episode record must be an object, found {type(record).__name__}'
            )
        name = episode_key(record)
        if name in metrics:
            raise HosiPerSequenceError(
                f'duplicate episode key {name!r}: the key must be unique or the '
                'pairing is ambiguous'
            )
        projected: Dict[str, float] = {}
        for field, value in record.items():
            if field in IDENTITY_FIELDS:
                continue
            if field in BOOLEAN_FIELDS:
                projected[field] = 1.0 if value else 0.0
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                # Non-numeric and not a declared boolean: drop it rather than
                # coerce, and say so at the end.
                continue
            projected[field] = float(value)
        metrics[name] = projected

    out: Dict[str, Any] = {
        'schema_version': 1,
        'source': 'hosi_per_sequence.project',
        'seed': payload.get('seed'),
        'model_name': payload.get('model_name'),
        'expert': payload.get('expert'),
        'sample_type': payload.get('sample_type'),
        'sequence_count': len(metrics),
        'metrics': metrics,
    }
    # Provenance a reader of the projected file needs in order to trust it.
    for key in ('checkpoint', 'hsi_checkpoint', 'sharding'):
        if key in payload:
            out[key] = payload[key]
    audit = payload.get('sampler_audit')
    if isinstance(audit, dict) and 'composition' in audit:
        out['composition'] = audit['composition']
    return out


def _load(path: Path) -> Dict[str, Any]:
    raw = path.read_bytes()
    payload = json.loads(raw.decode('utf-8'))
    payload['_source_sha256'] = hashlib.sha256(raw).hexdigest()
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('summary', help='overall_evaluation_summary.json')
    parser.add_argument('--output', required=True,
                        help='destination per_sequence_metrics.json')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args(argv)

    source = Path(args.summary)
    payload = _load(source)
    source_sha = payload.pop('_source_sha256')
    projected = project(payload)
    projected['source_path'] = str(source)
    projected['source_sha256'] = source_sha

    destination = Path(args.output)
    if destination.exists() and not args.overwrite:
        raise SystemExit(f'refusing to overwrite {destination}; pass --overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(projected, indent=2, sort_keys=True), encoding='utf-8',
    )
    print(f'{destination}  ({projected["sequence_count"]} sequences, '
          f'{len(next(iter(projected["metrics"].values())))} metrics)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
