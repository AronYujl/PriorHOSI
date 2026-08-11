"""Run-start provenance preflight for the HOIPrior trainer.

Covers the two defects closed on 2026-08-11:

* **R1** -- ``AGENTS.md`` has required a clean worktree for reportable runs since
  Phase 0, but the check only ever lived in ``tools/experiment.py start``.  The
  last fourteen reportable runs did not take that step: every P8/P9/P10 registry
  row records ``"no tools/experiment.py start manifest exists for this arm"``.
  The gate now lives on the path that actually executes.
* **R2** -- ``metrics.json`` is written once, at completion, so resolving
  ``git rev-parse HEAD`` there recorded whatever commit was checked out hours
  after the run began.  P10's A10/A01 arms started at ``91232ad`` and were
  recorded as ``5d39ac3``.  The commit is now resolved once, at run start.
"""

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

import train_hoi_prior as trainer


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _make_repo(directory: Path) -> str:
    """A one-commit Git repository, returning its HEAD."""
    _git(directory, "init", "--quiet")
    _git(directory, "config", "user.email", "test@example.invalid")
    _git(directory, "config", "user.name", "test")
    (directory / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(directory, "add", "tracked.txt")
    _git(directory, "commit", "--quiet", "-m", "initial")
    return _git(directory, "rev-parse", "HEAD")


def _cfg(repo: Path, **overrides: object):
    value = {
        "repo_root": str(repo),
        "run_id": None,
        "require_clean_worktree": None,
        "run_start_git_commit": None,
        "worktree_preflight": None,
    }
    value.update(overrides)
    return OmegaConf.create(value)


class ReportableRunTests(unittest.TestCase):
    def test_a_run_id_marks_the_run_reportable(self):
        self.assertFalse(trainer._reportable_run(OmegaConf.create({"run_id": None})))
        self.assertFalse(trainer._reportable_run(OmegaConf.create({"run_id": ""})))
        self.assertFalse(trainer._reportable_run(OmegaConf.create({})))
        self.assertTrue(
            trainer._reportable_run(
                OmegaConf.create({"run_id": "p1-hoi-p11-probe-s42-20260812"})
            )
        )


class CleanWorktreeGateTests(unittest.TestCase):
    def test_clean_repository_passes_and_reports_clean(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            record = trainer._validate_clean_worktree(_cfg(repo, run_id="p1-hoi-x-s42-20260812"))
            self.assertTrue(record["enforced"])
            self.assertTrue(record["clean"])
            self.assertNotIn("dirty_entries", record)

    def test_reportable_run_refuses_a_modified_tracked_file(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
            with self.assertRaises(RuntimeError) as raised:
                trainer._validate_clean_worktree(_cfg(repo, run_id="p1-hoi-x-s42-20260812"))
            self.assertIn("requires a clean worktree", str(raised.exception))
            self.assertIn("tracked.txt", str(raised.exception))

    def test_reportable_run_refuses_an_untracked_file(self):
        """Matches tools/experiment.py: an untracked file is a dirty worktree."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
            with self.assertRaises(RuntimeError) as raised:
                trainer._validate_clean_worktree(_cfg(repo, run_id="p1-hoi-x-s42-20260812"))
            self.assertIn("untracked files count", str(raised.exception))
            self.assertIn("scratch.txt", str(raised.exception))

    def test_untracked_file_inside_a_directory_is_still_dirty(self):
        """--untracked-files=all, so a nested file is not hidden behind its dir."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            nested = repo / "nested"
            nested.mkdir()
            (nested / "scratch.txt").write_text("untracked\n", encoding="utf-8")
            status = trainer._git_status_porcelain(repo)
            self.assertEqual(status, ["?? nested/scratch.txt"])

    def test_smoke_run_without_a_run_id_is_not_gated(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
            record = trainer._validate_clean_worktree(_cfg(repo))
            self.assertFalse(record["enforced"])
            self.assertFalse(record["clean"])
            self.assertEqual(record["dirty_entries"], 1)

    def test_explicit_false_overrides_the_gate_for_a_reportable_run(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
            record = trainer._validate_clean_worktree(
                _cfg(repo, run_id="p1-hoi-x-s42-20260812", require_clean_worktree=False)
            )
            self.assertFalse(record["enforced"])
            self.assertTrue(record["reportable_run"])
            self.assertIs(record["configured"], False)

    def test_explicit_true_enforces_the_gate_without_a_run_id(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            (repo / "tracked.txt").write_text("modified\n", encoding="utf-8")
            with self.assertRaises(RuntimeError):
                trainer._validate_clean_worktree(_cfg(repo, require_clean_worktree=True))

    def test_a_reportable_run_outside_a_repository_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with self.assertRaises(RuntimeError) as raised:
                trainer._validate_clean_worktree(_cfg(repo, run_id="p1-hoi-x-s42-20260812"))
            self.assertIn("committed Git object", str(raised.exception))

    def test_a_smoke_run_outside_a_repository_records_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            record = trainer._validate_clean_worktree(_cfg(Path(directory)))
            self.assertIsNone(record["clean"])
            self.assertIn("unavailable", record)


class RunStartCommitTests(unittest.TestCase):
    def test_the_commit_is_resolved_once_and_survives_a_later_commit(self):
        """The R2 regression, reproduced: HEAD moves while the run is going."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            start = _make_repo(repo)
            cfg = _cfg(repo, run_id="p1-hoi-x-s42-20260812")
            resolved = trainer._resolve_run_start_commit(cfg)
            self.assertEqual(resolved, start)

            # A commit lands mid-run, exactly as 981b9b0/b51c3b3/5d39ac3 did
            # while P10's A10 and A01 arms were training.
            (repo / "tracked.txt").write_text("a later commit\n", encoding="utf-8")
            _git(repo, "commit", "--quiet", "-am", "mid-run")
            moved = _git(repo, "rev-parse", "HEAD")
            self.assertNotEqual(moved, start)

            # The record still names the commit the run actually started from.
            self.assertEqual(trainer._run_commit(cfg), start)
            # ... and live HEAD is available separately, for the completion field.
            self.assertEqual(trainer._git_commit_or_none(cfg), moved)

    def test_resolution_writes_the_commit_into_the_config(self):
        """It must reach every rank through torch.multiprocessing.spawn."""
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            start = _make_repo(repo)
            cfg = _cfg(repo)
            trainer._resolve_run_start_commit(cfg)
            self.assertEqual(cfg.run_start_git_commit, start)
            container = OmegaConf.to_container(cfg, resolve=True)
            self.assertEqual(container["run_start_git_commit"], start)

    def test_run_commit_refuses_to_guess_when_the_preflight_did_not_run(self):
        cfg = _cfg(Path("/nonexistent"))
        with self.assertRaises(RuntimeError) as raised:
            trainer._run_commit(cfg)
        self.assertIn("preflight did not run", str(raised.exception))

    def test_worktree_preflight_record_round_trips(self):
        cfg = _cfg(Path("/nonexistent"))
        self.assertIsNone(trainer._worktree_preflight_record(cfg))
        OmegaConf.set_struct(cfg, False)
        cfg.worktree_preflight = {"enforced": True, "clean": True}
        record = trainer._worktree_preflight_record(cfg)
        self.assertEqual(record, {"enforced": True, "clean": True})


class NoLiveHeadAtWriteTimeTests(unittest.TestCase):
    """Guard the fix itself: no record may resolve HEAD when it is written."""

    def test_the_record_sites_do_not_call_git_commit_directly(self):
        source = (REPO / "code" / "train_hoi_prior.py").read_text(encoding="utf-8")
        self.assertNotIn('"git_commit": _git_commit(', source)
        self.assertIn('"git_commit": _run_commit(cfg)', source)
        self.assertIn('"git_commit_at_completion": _git_commit_or_none(cfg)', source)

    def test_the_preflight_runs_before_the_workers_are_spawned(self):
        source = (REPO / "code" / "train_hoi_prior.py").read_text(encoding="utf-8")
        body = source.split("def main(cfg: DictConfig) -> None:", 1)[1]
        preflight = body.index("_validate_clean_worktree(cfg)")
        resolve = body.index("_resolve_run_start_commit(cfg)")
        spawn = body.index("torch.multiprocessing.spawn")
        self.assertLess(preflight, spawn)
        self.assertLess(resolve, spawn)

    def test_the_configuration_declares_the_preflight_keys(self):
        config = (
            REPO / "code" / "config" / "config_train_hoi_prior.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("require_clean_worktree: null", config)
        self.assertIn("run_start_git_commit: null", config)
        self.assertIn("worktree_preflight: null", config)


if __name__ == "__main__":
    unittest.main()
