"""Sharding the HOSI-test evaluation across GPUs, and merging the shards back.

A composed row over the 469-episode benchmark costs 7.65 h of sampling on one GPU
(measured: 58.69 s/episode at G > 0, 2086 windows).  Sharding it is what makes the
worker's four GPUs usable.  This module holds the partition rule, the timing
invalidation and the merge, so all three are testable without importing the
evaluator's heavy dependencies.

THE UNIT IS THE SCENE, not the episode, and that is measured rather than assumed.
`set_test_scene` reloads the scene mesh and rebuilds the occupancy: 4.96 s per
scene, 67 scenes.  An episode-level shard's episodes scatter across nearly every
scene, so each of four shards pays 60-63 switches; a scene-level shard pays 17.
Predicted slowest shard at four shards, switches included:

    scene-level   1.966 h   3.94x speedup   17 switches
    episode-level 2.023 h   3.83x speedup   61 switches

Evidence: .claude/scratch/phase2-sharding/shard_unit.json.

BALANCE IS BY WINDOW COUNT, via a proxy, and the proxy is measured too.  The true
per-episode window count is not in the data files: `test_item['episode_num']` looks
like it but matches the true count only 164 of 461 times (Spearman 0.853), and the
evaluator never reads it -- `seg_len = ceil(A* arc / 0.8) + 1`, and
`cond['is_loco']` is true on all 469 episodes so that branch always fires.
Computing the true counts needs a 332 s pre-pass (build the dataset, A* 469 times).
The straight-line chord ||pelvis_goal - start_location|| in xz is free, comes from
the JSON alone, and has Spearman 0.971 against the true count; the A* arc is 1.13x
the chord on average.  At four shards the chord's packing costs 1.3 min more wall
clock than the exact packing -- and the pre-pass that would buy that back costs
5.5 min.  So the exact plan is a net loss, and the chord is the rule.
Evidence: .claude/scratch/phase2-sharding/{window_counts,packing_quality}.json.

SEEDING.  Scene-level sharding needs NO reseeding, which is measured, not assumed:

  * `HOIPriorSampler.prepare_sample_arguments` seeds its per-window generator from
    `(torch.initial_seed() + sample_calls * 1000003)`.  `initial_seed()` returns the
    SEED, not the live state, so it is constant for a run.
  * `sample_calls` is per sampler INSTANCE, and `test_infbagel_hosi` rebuilds
    `sampler_body` inside the scene loop, so it already resets at every scene.
  * `__getitem__` consumes no global RNG at test time.  Its four global draws are
    all gated off by the eval config (`train=False`, `use_random_frame_bps=False`,
    `use_object_keypoints=False`) or never fire (`np.random.randint` at
    datasets/infbagel.py:534 needs `not need_pi`, and `need_pi` is true on every
    HOSI-test episode).  Verified by hashing numpy, python-random and torch state
    before and after every `__getitem__` over three scenes: zero changes, and
    `torch.initial_seed()` constant throughout.
    Evidence: .claude/scratch/phase2-sharding/getitem_rng.json.
  * The three `randperm` draws already use dedicated generators keyed on
    `(seed, scene_name, test_idx)` (commit 1c2d99b).

So a scene's episodes are seeded identically however many other scenes ran, and a
scene-level shard reproduces the serial row BITWISE.  That is why the sealed
`p2-hosi-hoi-alone-g0-p15-guided-armb-s42-20260829` anchor stays valid and pairable
and needs no re-run.

`per_episode_seeding` is offered anyway, defaulting OFF, because it buys a property
scene sharding does not: an episode reproducible in ISOLATION.  Today episode 3 of a
scene can only be reproduced by running episodes 0-2 first, since `sample_calls`
counts windows within the scene.  Turning it on re-seeds from the canonical ordinal
AND resets `sample_calls`, which is what episode-level sharding would require.
Measured offline with a stub model: the same episode at scene position 0 vs 2
differs by up to 1.91 in normalized units today; reseeding alone does NOT fix it
(`sample_calls` is the other seed input) and resetting alone does; both together
reproduce the isolated episode exactly.
Evidence: .claude/scratch/phase2-sharding/order_dependence.json.

The cost of turning it on is that every existing HOSI row was produced without it,
so a campaign that enables it must re-run its own anchor.  Never mix the two
regimes inside one comparison.
"""

import json
import math
import os
from collections import OrderedDict
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

#: 67 scene files x 7 objects.
CANONICAL_EPISODE_TOTAL = 469
#: Measured over all 469 episodes; informational, and the merge does not gate on it
#: because the evaluator does not record per-episode window counts.
CANONICAL_WINDOW_TOTAL = 2086

PARTITION_RULE_SERIAL = 'serial_full_enumeration'
PARTITION_RULE_SCENE = 'greedy_longest_first_bin_packing_by_scene_chord_sum'


def episode_chord(item: Mapping[str, Any]) -> float:
    """Straight-line xz distance from start to pelvis goal, the balance proxy.

    y is dropped because the evaluator's A* runs on the xz occupancy projection and
    `pelvis_goal[1]` is zeroed on this benchmark anyway.
    """
    start = item['start_location']
    goal = item['pelvis_goal']
    return math.hypot(float(goal[0]) - float(start[0]), float(goal[2]) - float(start[2]))


def scene_balance_keys(
    scene_items: Sequence[Sequence[Mapping[str, Any]]]
) -> Tuple[float, ...]:
    """One balance key per scene: the sum of its episodes' chords."""
    return tuple(float(sum(episode_chord(item) for item in items))
                 for items in scene_items)


def plan_scene_shards(
    keys: Sequence[float], shard_count: int
) -> Tuple[Tuple[int, ...], ...]:
    """Partition scene indices into ``shard_count`` bins balanced on ``keys``.

    Deterministic greedy longest-first bin packing: scenes sorted by descending key
    with the canonical scene index as tie-break, each placed into the currently
    least-loaded bin with the lowest bin index as tie-break.  A sharded run costs
    the SLOWEST shard, so the objective is the maximum per-shard key total.

    Each bin's indices come back in ascending canonical order, so a shard walks its
    scenes in the same relative order a serial run does -- which is what keeps the
    per-scene `sample_calls` sequence, and therefore the numbers, identical.
    """
    shard_count = int(shard_count)
    if shard_count < 1:
        raise ValueError(f'shard_count must be >= 1, got {shard_count}')
    values = [float(key) for key in keys]
    if shard_count > len(values):
        raise ValueError(
            f'shard_count {shard_count} exceeds scene count {len(values)}'
        )
    loads = [0.0] * shard_count
    bins: List[List[int]] = [[] for _ in range(shard_count)]
    for index in sorted(range(len(values)), key=lambda i: (-values[i], i)):
        target = min(range(shard_count), key=lambda shard: (loads[shard], shard))
        bins[target].append(index)
        loads[target] += values[index]
    return tuple(tuple(sorted(shard)) for shard in bins)


def enumerate_canonical_episodes(
    json_data_dir: str, scene_files: Optional[Sequence[str]] = None
) -> Tuple[Tuple[str, ...], Tuple[Tuple[Mapping[str, Any], ...], ...], Dict[Tuple[str, int], int]]:
    """Read the benchmark and assign every episode its canonical ordinal.

    The ordinal is the episode's index in the FULL enumeration -- scene files in
    sorted order, then test items in file order -- never its index within a shard.
    It is what a merge anchors on and what identifies an episode across runs.

    Returns ``(scene_files, per_scene_items, ordinal_by_scene_and_index)``.
    """
    if scene_files is None:
        scene_files = sorted(
            name for name in os.listdir(json_data_dir) if name.endswith('.json')
        )
    else:
        scene_files = tuple(scene_files)
    per_scene: List[Tuple[Mapping[str, Any], ...]] = []
    ordinals: Dict[Tuple[str, int], int] = {}
    ordinal = 0
    for scene_file in scene_files:
        with open(os.path.join(json_data_dir, scene_file)) as handle:
            items = tuple(json.load(handle))
        per_scene.append(items)
        scene_name = scene_file.split('.')[0]
        for test_idx in range(len(items)):
            ordinals[(scene_name, test_idx)] = ordinal
            ordinal += 1
    return tuple(scene_files), tuple(per_scene), ordinals


def select_shard_scenes(
    scene_files: Sequence[str],
    per_scene_items: Sequence[Sequence[Mapping[str, Any]]],
    shard_index: int,
    shard_count: int,
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    """The scene files this shard evaluates, plus the plan for the payload."""
    shard_count = int(shard_count)
    shard_index = int(shard_index)
    if not 0 <= shard_index < shard_count:
        raise ValueError(
            f'shard_index {shard_index} out of range for shard_count {shard_count}'
        )
    keys = scene_balance_keys(per_scene_items)
    if shard_count == 1:
        selected = tuple(range(len(scene_files)))
        rule = PARTITION_RULE_SERIAL
    else:
        selected = plan_scene_shards(keys, shard_count)[shard_index]
        rule = PARTITION_RULE_SCENE
    episodes = int(sum(len(per_scene_items[index]) for index in selected))
    plan = {
        'shard_index': shard_index,
        'shard_count': shard_count,
        'partition_rule': rule,
        'partition_unit': 'scene',
        'balance_key': 'sum_of_episode_xz_chord_metres',
        'canonical_episode_total': int(sum(len(items) for items in per_scene_items)),
        'canonical_scene_total': len(scene_files),
        'shard_scene_count': len(selected),
        'shard_episode_count': episodes,
        'shard_balance_key': float(sum(keys[index] for index in selected)),
        'scene_indices': list(selected),
    }
    return tuple(scene_files[index] for index in selected), plan


# Wall-clock aggregates a contended sharded run cannot measure.  Deliberately
# separate from anything hardware-independent: nothing about the metrics is
# invalidated by sharding, only the timing.
SHARD_INVALID_TIMING_KEYS = (
    'aits',
    'avg_fps',
    'aggregate_fps',
    'total_generation_seconds',
    'avg_frames_per_seq',
    'avg_end_to_end_episode_seconds',
)
SHARD_TIMING_INVALID_REASON = (
    'shard_count>1: concurrent shards contend for host CPU, PCIe and the scene '
    'occupancy/SDF caches, so every wall-clock aggregate is contaminated. Latency '
    'for this benchmark comes only from a serial shard_count=1 pass.'
)


def invalidate_timing(generation_metrics: Optional[MutableMapping[str, Any]]) -> Optional[MutableMapping[str, Any]]:
    """Null every wall-clock aggregate and say so in the payload.

    Nulling rather than deleting: a reader diffing a sharded payload against a
    serial one must see the same keys with explicit nulls, not a structurally
    different object that quietly lacks them.
    """
    if generation_metrics is None:
        return None
    for key in SHARD_INVALID_TIMING_KEYS:
        if key in generation_metrics:
            generation_metrics[key] = None
    generation_metrics['timing_valid'] = False
    generation_metrics['timing_invalid_reason'] = SHARD_TIMING_INVALID_REASON
    return generation_metrics


#: Payload fields that must agree across shards.  Each one, if it differed, would
#: make the merged row a claim no single run ever produced.  `hsi_checkpoint` is
#: checked separately because it is a dict and is null on single-expert rows.
MERGE_AGREEMENT_KEYS = (
    'seed',
    'expert',
    'sample_type',
    'evaluator_guidance_fn',
)


def _checkpoint_identity(payload: Mapping[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    """(HOI sha256, HSI sha256).  Both are required to match across shards.

    The HSI hash is what makes a mixer row re-runnable rather than rewritable when
    HSIPrior is superseded: the row states which pair of checkpoints produced it,
    so a new HSIPrior means re-running these rows, not editing them.
    """
    checkpoint = payload.get('checkpoint') or {}
    hoi = checkpoint.get('sha256') or checkpoint.get('checkpoint_sha256')
    hsi_block = payload.get('hsi_checkpoint') or {}
    return hoi, hsi_block.get('sha256')


def _gate_identity(payload: Mapping[str, Any]) -> Any:
    """The gate description, which is as much a part of the row as the weights."""
    audit = payload.get('sampler_audit') or {}
    composition = audit.get('composition') or {}
    return json.dumps(
        {
            'gate': composition.get('gate'),
            'channel_mask': composition.get('channel_mask'),
            'hsi_object_voxel_mode': composition.get('hsi_object_voxel_mode'),
        },
        sort_keys=True,
    )


def merge_shard_payloads(
    payloads: Sequence[Mapping[str, Any]],
    expected_episodes: int = CANONICAL_EPISODE_TOTAL,
    expected_shard_count: Optional[int] = None,
    metric_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Combine shard payloads into one structurally serial-identical payload.

    Every guard raises.  A silently short merge is the worst possible outcome
    because it reads as a complete result: the aggregate would be a mean over
    whatever survived, with nothing in the file saying so.  Statistics are
    recomputed over the UNION of episode records, never averaged from per-shard
    means -- shards hold unequal episode counts, so averaging averages would weight
    them wrongly.

    ``expected_shard_count`` is the operator's own statement of how many shards the
    campaign had.  Without it the count is self-declared by the files on disk, so
    "merge 4 shards" over a directory holding a 2-shard pair would succeed: the pair
    is internally consistent.
    """
    if not payloads:
        raise ValueError('merge_shard_payloads received no shard payloads')
    expected_episodes = int(expected_episodes)

    by_index: Dict[int, Mapping[str, Any]] = {}
    for payload in payloads:
        block = payload.get('sharding')
        if not isinstance(block, Mapping):
            raise ValueError(
                "shard payload has no 'sharding' block; it is not a sharded run"
            )
        index = int(block['shard_index'])
        if index in by_index:
            raise ValueError(f'two payloads both claim shard_index={index}')
        by_index[index] = payload

    declared = {int(item['sharding']['shard_count']) for item in payloads}
    if len(declared) != 1:
        raise ValueError(f'shard payloads disagree on shard_count: {sorted(declared)}')
    shard_count = declared.pop()
    if expected_shard_count is not None and shard_count != int(expected_shard_count):
        raise ValueError(
            f'payloads on disk declare shard_count={shard_count}, the merge was '
            f'asked for {int(expected_shard_count)}'
        )
    if len(payloads) != shard_count:
        raise ValueError(
            f'expected {shard_count} shard payloads, received {len(payloads)}'
        )
    missing = sorted(set(range(shard_count)) - set(by_index))
    if missing:
        raise ValueError(
            f'shard indices {missing} are missing; refusing to merge '
            f'{len(by_index)} of {shard_count} shards'
        )

    reference = by_index[0]
    reference_checkpoints = _checkpoint_identity(reference)
    reference_gate = _gate_identity(reference)
    for index in range(1, shard_count):
        candidate = by_index[index]
        for key in MERGE_AGREEMENT_KEYS:
            if candidate.get(key) != reference.get(key):
                raise ValueError(
                    f'shard {index} disagrees with shard 0 on {key!r}: '
                    f'{candidate.get(key)!r} vs {reference.get(key)!r}'
                )
        if _checkpoint_identity(candidate) != reference_checkpoints:
            raise ValueError(
                f'shard {index} evaluated a different checkpoint pair than shard 0: '
                f'{_checkpoint_identity(candidate)} vs {reference_checkpoints}'
            )
        if _gate_identity(candidate) != reference_gate:
            raise ValueError(
                f'shard {index} used a different gate/mask than shard 0: '
                f'{_gate_identity(candidate)} vs {reference_gate}'
            )
        for key in ('canonical_episode_total', 'canonical_scene_total',
                    'partition_rule', 'per_episode_seeding'):
            if candidate['sharding'].get(key) != reference['sharding'].get(key):
                raise ValueError(
                    f'shard {index} declares {key}='
                    f'{candidate["sharding"].get(key)!r}, shard 0 declares '
                    f'{reference["sharding"].get(key)!r}'
                )

    canonical_episodes = int(reference['sharding']['canonical_episode_total'])
    if canonical_episodes != expected_episodes:
        raise ValueError(
            f'shards enumerate {canonical_episodes} canonical episodes, the '
            f'protocol expects {expected_episodes}'
        )

    records: List[Mapping[str, Any]] = []
    seen_ordinals: Dict[int, int] = {}
    scene_order: List[str] = []
    for index in range(shard_count):
        payload = by_index[index]
        for record in payload['individual_metrics']:
            ordinal = record.get('canonical_ordinal')
            if ordinal is None:
                raise ValueError(
                    f'shard {index} has an episode record with no '
                    'canonical_ordinal; it predates sharding support'
                )
            ordinal = int(ordinal)
            if ordinal in seen_ordinals:
                raise ValueError(
                    f'canonical ordinal {ordinal} appears in shard {index} and '
                    f'shard {seen_ordinals[ordinal]}'
                )
            seen_ordinals[ordinal] = index
            records.append(record)
        scene_order.extend(payload.get('scene_order') or [])

    if len(records) != expected_episodes:
        raise ValueError(
            f'merged {len(records)} episode records, the protocol expects '
            f'{expected_episodes}'
        )
    absent = sorted(set(range(expected_episodes)) - set(seen_ordinals))
    if absent:
        raise ValueError(
            f'canonical ordinals missing from the merge: {len(absent)} of '
            f'{expected_episodes}, first {absent[:10]}'
        )

    records.sort(key=lambda item: int(item['canonical_ordinal']))
    statistics = recompute_statistics(records, metric_names)

    merged = dict(reference)
    merged['individual_metrics'] = records
    merged['scene_order'] = sorted(set(scene_order))
    merged['statistics'] = statistics
    summary = dict(reference.get('summary') or {})
    summary['total_evaluated'] = len(records)
    summary['completion_rate'] = statistics['completion_rate']
    summary['key_metrics'] = {
        f'avg_{name}': statistics.get(name, {}).get('mean')
        for name in statistics
        if isinstance(statistics.get(name), Mapping)
    }
    if 'generation_metrics' in summary:
        summary['generation_metrics'] = invalidate_timing(
            dict(summary['generation_metrics'])
        )
    merged['summary'] = summary
    merged['sharding'] = {
        **{k: v for k, v in reference['sharding'].items()
           if k not in ('shard_index', 'scene_indices', 'shard_scene_count',
                        'shard_episode_count', 'shard_balance_key')},
        'shard_index': None,
        'merged_shard_count': shard_count,
        'merged_episodes_per_shard': [
            len(by_index[index]['individual_metrics']) for index in range(shard_count)
        ],
        'merged_scenes_per_shard': [
            int(by_index[index]['sharding']['shard_scene_count'])
            for index in range(shard_count)
        ],
    }
    return merged


def recompute_statistics(
    records: Sequence[Mapping[str, Any]],
    metric_names: Optional[Sequence[str]] = None,
) -> "OrderedDict[str, Any]":
    """Mean/std/min/max/median per metric over the union, plus completion.

    Matches `test_infbagel_hosi.main`'s aggregation exactly, including that a metric
    absent or None on an episode is dropped from that metric's sample rather than
    treated as zero.
    """
    import numpy as np

    if metric_names is None:
        metric_names = []
        for record in records:
            for key, value in record.items():
                if key in ('scene_name', 'object_name', 'test_idx', 'completed',
                           'canonical_ordinal'):
                    continue
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    continue
                if key not in metric_names:
                    metric_names.append(key)
    statistics: "OrderedDict[str, Any]" = OrderedDict()
    for name in metric_names:
        values = [
            float(record[name]) for record in records
            if record.get(name) is not None
        ]
        if values:
            statistics[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
                'median': float(np.median(values)),
            }
    completed = sum(1 for record in records if record.get('completed', False))
    statistics['completion_rate'] = completed / len(records) if records else 0.0
    statistics['total_samples'] = len(records)
    statistics['completed_samples'] = completed
    return statistics
