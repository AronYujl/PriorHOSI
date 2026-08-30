"""Emit the launch plan for a sharded HOSI-test row, with the return armed up front.

Prints a shell script; it does not run anything and touches no GPU.  `--execute` is
deliberately absent: launching a row is a GPU workload and needs the user's explicit
approval of one concrete experiment, so this tool's whole job is to produce a plan a
human reads and then runs.

Three properties are built in because each has cost real time before.

RETURN ARMED AT LAUNCH.  Each shard gets two detached tmux sessions: one running the
evaluation and writing `<name>.exitcode` when python exits, one blocking on that file
and then rsyncing the shard directory to the authority.  The watcher outlives the
session that started the run, survives the tunnel dropping, and transfers on FAILURE
too -- a failed shard's log is the thing you most need and the thing a
success-only return loses.  Direction is not a style choice: inbound TCP/22 to the
worker is blocked, so the worker initiates every transfer (AGENTS.md).

OMP CAPPED.  `OMP_NUM_THREADS=4` on every shard.  Uncapped, the HOI evaluation took
23 min against a capped control's 195 s on the same protocol and host -- the cost is
oversubscribed BLAS in the per-episode Python preprocessing, not the sampling loop.
Capping is bitwise identical (established on HOI eval: 18 of 18 aggregate metrics,
every delta exactly 0.0).  Four concurrent shards on one host make this worse, not
better, since they contend for the same cores.

BOTH CHECKPOINT HASHES RECORDED.  Every shard payload carries `checkpoint.sha256`
and `hsi_checkpoint.sha256`, and the merge refuses shards that disagree on either.
A mixer row is a claim about a PAIR of models, so when HSIPrior is superseded these
rows are re-run, not rewritten -- the hash is what makes that a mechanical check
rather than an act of memory.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), 'code'))

from hosi_sharding import (  # noqa: E402
    enumerate_canonical_episodes,
    scene_balance_keys,
    select_shard_scenes,
)

#: Measured 2026-08-30 on the composed path: 58.69 s/episode at G>0 over 7 episodes,
#: 2086 windows across 469 episodes, and 4.96 s per `set_test_scene`.
SECONDS_PER_WINDOW = 13.20
SECONDS_PER_SCENE_SWITCH = 4.96
#: The G==0 anchor path skips the HSI expert entirely on every step, measured 10x
#: cheaper per episode (5.89 s vs 58.69 s).
ANCHOR_SPEEDUP = 9.96


def _parse_args(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config-name', default='config_sample_hosi_composed')
    parser.add_argument('--exp-name', required=True,
                        help='run id / experiment name; one per campaign')
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--gpus', default=None,
                        help='comma-separated CUDA device ids, default 0..shards-1')
    parser.add_argument('--host', default='worker',
                        choices=('worker', 'authority'),
                        help='worker arms an rsync return; authority does not need one')
    parser.add_argument('--overrides', nargs='*', default=[],
                        help='extra Hydra overrides applied to every shard')
    parser.add_argument('--per-episode-seeding', action='store_true',
                        help='enable hosi_per_episode_seeding; requires re-running '
                             'this campaign\'s own anchor, see hosi_sharding.py')
    parser.add_argument('--anchor', action='store_true',
                        help='cost the G==0 anchor path (HSI expert skipped)')
    parser.add_argument('--json', action='store_true', help='emit the plan as JSON')
    return parser.parse_args(argv)


def _plan(args):
    root = os.environ.get('ROOT_DIR') or os.path.dirname(
        os.path.dirname(os.path.abspath(__file__)))
    files, items, ordinals = enumerate_canonical_episodes(
        os.path.join(root, 'data', 'hosi_test', 'data'))
    keys = scene_balance_keys(items)
    del keys
    shards = []
    for index in range(args.shards):
        chosen, plan = select_shard_scenes(files, items, index, args.shards)
        # Window counts are not known without the 332 s A* pre-pass, which is a net
        # loss (see hosi_sharding.py).  The ESTIMATE below therefore prorates the
        # measured 2086 total by the shard's episode share, which is accurate to the
        # ~1% the chord proxy costs -- it is a schedule, not a measurement.
        share = plan['shard_episode_count'] / len(ordinals)
        windows = 2086 * share
        seconds = windows * SECONDS_PER_WINDOW + \
            plan['shard_scene_count'] * SECONDS_PER_SCENE_SWITCH
        if args.anchor:
            seconds /= ANCHOR_SPEEDUP
        shards.append({
            'shard_index': index,
            'scenes': plan['shard_scene_count'],
            'episodes': plan['shard_episode_count'],
            'balance_key': round(plan['shard_balance_key'], 2),
            'estimated_windows': round(windows, 1),
            'estimated_hours': round(seconds / 3600.0, 3),
        })
    return {
        'exp_name': args.exp_name,
        'config_name': args.config_name,
        'shards': args.shards,
        'total_episodes': len(ordinals),
        'slowest_shard_hours': max(s['estimated_hours'] for s in shards),
        'serial_hours_estimate': round(
            (2086 * SECONDS_PER_WINDOW + 67 * SECONDS_PER_SCENE_SWITCH)
            / (ANCHOR_SPEEDUP if args.anchor else 1.0) / 3600.0, 3),
        'per_shard': shards,
    }


def _script(args, plan):
    gpus = (args.gpus.split(',') if args.gpus
            else [str(i) for i in range(args.shards)])
    if len(gpus) != args.shards:
        raise SystemExit(f'{len(gpus)} gpus for {args.shards} shards')
    overrides = list(args.overrides)
    if args.per_episode_seeding:
        overrides.append('hosi_per_episode_seeding=true')
    override_text = ' '.join(f'  {o} \\\n' for o in overrides)

    lines = [
        '#!/usr/bin/env bash',
        '# Generated by tools/launch_hosi_sharded.py -- review before running.',
        '# This is a GPU workload: it needs the user\'s explicit approval of one',
        '# concrete experiment, and a run id allocated through tools/experiment.py.',
        'set -euo pipefail',
        '',
        f'EXP_NAME={args.exp_name}',
        f'SHARDS={args.shards}',
        'CODE_DIR="$ROOT_DIR/code"',
        'RESULTS="$ROOT_DIR/results/experiments/$EXP_NAME"',
        '',
        '# Uncapped, the same protocol took 23 min against a capped 195 s: the cost',
        '# is oversubscribed BLAS in per-episode preprocessing, and capping is',
        '# bitwise identical.  Four concurrent shards make contention worse.',
        'export OMP_NUM_THREADS=4',
        'export MKL_NUM_THREADS=4',
        'export OPENBLAS_NUM_THREADS=4',
        '',
    ]
    if args.host == 'worker':
        lines += [
            '# The return is armed BEFORE the work starts and must outlive this',
            '# session.  The worker initiates the transfer: inbound TCP/22 to the',
            '# worker is blocked (AGENTS.md).',
            'AUTHORITY=10.184.17.253',
            'AUTH_KEY="$HOME/.ssh/id_ed25519_infbagel_8gpu"',
            'AUTH_STAGING=/data/yujinlun/InfBaGel-mixer/results/incoming',
            'mkdir -p "$RESULTS"',
            '',
            '# Prove the return path NOW, with one tiny file, rather than finding a',
            '# key or permission problem after the last shard finishes.',
            'echo "$EXP_NAME armed $(date -Is)" > "$RESULTS/.return-probe"',
            'rsync -aH -e "ssh -i $AUTH_KEY -o IdentitiesOnly=yes" \\',
            '  "$RESULTS/.return-probe" \\',
            '  "yujinlun@$AUTHORITY:$AUTH_STAGING/$EXP_NAME.return-probe"',
            'echo "return path verified"',
            '',
        ]
    for index, gpu in enumerate(gpus):
        name = f'$EXP_NAME-shard{index:02d}'
        lines += [
            f'# ---- shard {index} on GPU {gpu} '
            f'({plan["per_shard"][index]["episodes"]} episodes, '
            f'~{plan["per_shard"][index]["estimated_hours"]} h)',
            f'tmux new-session -d -s "{name}" "\\',
            f'  cd $CODE_DIR && CUDA_VISIBLE_DEVICES={gpu} \\',
            '  $INFBAGEL_PYTHON test_infbagel_hosi.py \\',
            f'    --config-name {args.config_name} \\',
            f'    exp_name=$EXP_NAME \\',
            f'    hosi_shard_index={index} hosi_shard_count=$SHARDS \\',
        ]
        if override_text:
            lines.append(override_text.rstrip('\\\n'))
        lines += [
            f'    > $RESULTS/shard{index:02d}.log 2>&1; \\',
            f'  echo \\$? > $RESULTS/shard{index:02d}.exitcode"',
            '',
        ]
    if args.host == 'worker':
        lines += [
            '# ---- the watcher.  Blocks on all N exitcode files, then transfers',
            '# whatever exists -- including a failed shard\'s log, which is the',
            '# artifact you most need and the one a success-only return drops.',
            'tmux new-session -d -s "$EXP_NAME-return" "\\',
            '  while [ \\$(ls $RESULTS/shard*.exitcode 2>/dev/null | wc -l) '
            '-lt $SHARDS ]; do sleep 60; done; \\',
            '  rsync -aH --partial -e \\"ssh -i $AUTH_KEY -o IdentitiesOnly=yes\\" \\',
            '    $RESULTS/ yujinlun@$AUTHORITY:$AUTH_STAGING/$EXP_NAME/"',
            '',
            'echo "armed: $SHARDS shards + return watcher"',
            'echo "watch:  tmux ls"',
            'echo "merge on the authority once all shards land:"',
        ]
    else:
        lines += [
            'echo "launched: $SHARDS shards"',
            'echo "merge once all shards finish:"',
        ]
    lines += [
        f'echo "  cd code && $INFBAGEL_PYTHON test_infbagel_hosi.py '
        f'--config-name {args.config_name} \\\\"',
        'echo "    exp_name=$EXP_NAME hosi_mode=merge_shards '
        'hosi_shard_count=$SHARDS"',
    ]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    args = _parse_args(sys.argv[1:] if argv is None else argv)
    plan = _plan(args)
    if args.json:
        print(json.dumps(plan, indent=2))
        return 0
    print('# PLAN')
    print(f'#   {plan["total_episodes"]} episodes over {plan["shards"]} shards')
    for shard in plan['per_shard']:
        print('#   shard %d: %2d scenes %3d episodes ~%.2f h'
              % (shard['shard_index'], shard['scenes'], shard['episodes'],
                 shard['estimated_hours']))
    print('#   slowest shard ~%.2f h against ~%.2f h serial (%.2fx)'
          % (plan['slowest_shard_hours'], plan['serial_hours_estimate'],
             plan['serial_hours_estimate'] / plan['slowest_shard_hours']))
    print('#   estimates prorate the measured 2086 windows by episode share; they')
    print('#   are a schedule, not a measurement.')
    print()
    print(_script(args, plan), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
