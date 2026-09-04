"""The sharded-launch plan: the three properties that have cost real time before.

Asserted on the emitted script rather than on the docstring, because a comment that
says "return armed at launch" and a script that arms it after the work are
indistinguishable from the outside.

1.  The return watcher is armed BEFORE the shards start, and it transfers on
    failure too.
2.  OMP is capped on every shard.
3.  The plan is a plan: no tool switch executes it, since a row is a GPU workload
    needing explicit approval and an allocated run id.
"""

import os
import subprocess
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
TOOL = REPO / 'tools' / 'launch_hosi_sharded.py'
sys.path.insert(0, str(REPO / 'code'))
sys.path.insert(0, str(REPO / 'tools'))

INTERPRETER = os.environ.get('INFBAGEL_PYTHON') or sys.executable


def _emit(*extra):
    environment = dict(os.environ, ROOT_DIR=str(REPO), OMP_NUM_THREADS='4')
    result = subprocess.run(
        [INTERPRETER, str(TOOL), '--exp-name', 'test-campaign', *extra],
        capture_output=True, text=True, env=environment, cwd=str(REPO), timeout=300,
    )
    if result.returncode != 0:
        raise AssertionError(f'launcher failed: {result.stderr[-2000:]}')
    return result.stdout


class PlanTests(unittest.TestCase):
    def setUp(self):
        self.script = _emit('--shards', '4')

    def test_the_return_is_armed_before_the_first_shard(self):
        """Ordering is the property, not the presence of an rsync somewhere.

        The probe transfer must precede the first `tmux new-session` that starts
        work, so a key or permission problem surfaces in seconds rather than after
        the last shard finishes.
        """
        probe = self.script.index('return path verified')
        first_shard = self.script.index('shard00"')
        self.assertLess(probe, first_shard)

    def test_the_watcher_waits_for_every_shard_and_transfers_on_failure(self):
        self.assertIn('shard*.exitcode', self.script)
        self.assertIn('-lt $SHARDS', self.script)
        # `echo $? > ...exitcode` after the python call, so a non-zero exit still
        # writes the file and still releases the watcher: the failure case is what
        # the return matters most for, since the log is the artifact you need.
        for index in range(4):
            self.assertIn(f'echo \\$? > $RESULTS/shard{index:02d}.exitcode',
                          self.script)
        # One waiter counting all four, not one waiter per shard.
        self.assertEqual(self.script.count('shard*.exitcode'), 1)

    def test_the_watcher_is_a_separate_detached_session(self):
        """It must outlive the session that launched the run."""
        self.assertIn('tmux new-session -d -s "$EXP_NAME-return"', self.script)

    def test_omp_is_capped(self):
        self.assertIn('export OMP_NUM_THREADS=4', self.script)
        self.assertIn('export MKL_NUM_THREADS=4', self.script)
        self.assertIn('export OPENBLAS_NUM_THREADS=4', self.script)

    def test_every_shard_gets_its_own_gpu_and_index(self):
        for index in range(4):
            self.assertIn(f'CUDA_VISIBLE_DEVICES={index}', self.script)
            self.assertIn(f'hosi_shard_index={index}', self.script)

    def test_the_merge_command_is_printed(self):
        self.assertIn('hosi_mode=merge_shards', self.script)

    def test_the_last_override_stays_inside_the_shard_command(self):
        script = _emit(
            '--shards', '1', '--gpus', '3', '--host', 'authority',
            '--overrides', 'first=value', 'last=value',
        )
        self.assertIn(
            '  first=value \\\n  last=value \\\n'
            '    > $RESULTS/shard00.log 2>&1;',
            script,
        )
        self.assertNotIn('  last=value \n', script)

    def test_the_script_is_valid_bash(self):
        result = subprocess.run(['bash', '-n'], input=self.script, text=True,
                                capture_output=True, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_it_says_a_row_needs_approval(self):
        self.assertIn('approval', self.script)

    def test_the_tool_has_no_execute_switch(self):
        """A plan, not a launcher: emitting is the whole job."""
        source = TOOL.read_text()
        self.assertNotIn("'--execute'", source)
        self.assertNotIn('subprocess.run', source)
        self.assertNotIn('os.system', source)


class HostTests(unittest.TestCase):
    def test_the_authority_variant_arms_no_return(self):
        script = _emit('--shards', '4', '--host', 'authority')
        self.assertNotIn('rsync', script)
        self.assertNotIn('EXP_NAME-return', script)
        self.assertIn('hosi_mode=merge_shards', script)

    def test_per_episode_seeding_is_opt_in_and_explicit(self):
        default = _emit('--shards', '2')
        self.assertNotIn('hosi_per_episode_seeding', default)
        opted = _emit('--shards', '2', '--per-episode-seeding')
        self.assertIn('hosi_per_episode_seeding=true', opted)


class EstimateTests(unittest.TestCase):
    def test_the_plan_covers_all_469_episodes(self):
        import json

        plan = json.loads(_emit('--shards', '4', '--json'))
        self.assertEqual(plan['total_episodes'], 469)
        self.assertEqual(sum(s['episodes'] for s in plan['per_shard']), 469)
        self.assertEqual(sum(s['scenes'] for s in plan['per_shard']), 67)

    def test_the_anchor_path_is_costed_lower(self):
        import json

        composed = json.loads(_emit('--shards', '4', '--json'))
        anchor = json.loads(_emit('--shards', '4', '--anchor', '--json'))
        self.assertLess(anchor['slowest_shard_hours'],
                        composed['slowest_shard_hours'] / 5)

    def test_a_gpu_count_mismatch_raises(self):
        environment = dict(os.environ, ROOT_DIR=str(REPO))
        result = subprocess.run(
            [INTERPRETER, str(TOOL), '--exp-name', 'x', '--shards', '4',
             '--gpus', '0,1'],
            capture_output=True, text=True, env=environment, cwd=str(REPO),
            timeout=300,
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('for 4 shards', result.stderr)


if __name__ == '__main__':
    unittest.main()
