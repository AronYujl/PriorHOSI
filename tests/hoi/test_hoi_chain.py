"""The worker-side train -> evaluate -> bootstrap chain.

The 8-GPU authority host runs HSIPrior and is expected to be busy with it, so a
HOIPrior arm must complete without it: no checkpoint is transferred and no
evaluation runs on the authority host. These tests cover the orchestration --
identity derivation, the refusals, checkpoint discovery, command construction
and stage idempotency -- none of which need a GPU.
"""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "tools"))

import hoi_chain


def _git(repo: Path, *args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()


def _make_repo(directory: Path) -> None:
    _git(directory, "init", "--quiet")
    _git(directory, "config", "user.email", "test@example.invalid")
    _git(directory, "config", "user.name", "test")
    (directory / "tracked.txt").write_text("committed\n", encoding="utf-8")
    _git(directory, "add", "tracked.txt")
    _git(directory, "commit", "--quiet", "-m", "initial")


class IdentityTests(unittest.TestCase):
    def test_arm_name_expands_to_a_config_name(self):
        self.assertEqual(
            hoi_chain.arm_config_name("p9w3"), "config_train_hoi_prior_p9w3"
        )
        self.assertEqual(
            hoi_chain.arm_config_name("config_train_hoi_prior_p9w3"),
            "config_train_hoi_prior_p9w3",
        )

    def test_the_run_id_comes_from_the_arm_config_not_from_this_tool(self):
        """Orchestration must never invent a training identity."""
        self.assertEqual(
            hoi_chain.read_run_id(REPO, "config_train_hoi_prior_p9w3"),
            "p1-hoi-p8-hand-object-geom-w3-s42-20260807",
        )

    def test_an_arm_without_a_run_id_is_refused(self):
        with self.assertRaises(hoi_chain.ChainError) as raised:
            hoi_chain.read_run_id(REPO, "config_train_hoi_prior")
        self.assertIn("must name its run", str(raised.exception))

    def test_a_missing_arm_config_is_refused(self):
        with self.assertRaises(hoi_chain.ChainError):
            hoi_chain.read_run_id(REPO, "config_train_hoi_prior_does_not_exist")

    def test_the_evaluation_id_carries_its_own_date_and_sampler_condition(self):
        derived = hoi_chain.derive_eval_run_id(
            "p1-hoi-p10-geom-hinge-s42-20260809", "20260812"
        )
        self.assertEqual(derived, "p1-hoi-p10-geom-hinge-eval-guided-s42-20260812")
        hoi_chain.validate_run_id(derived)

    def test_the_sampler_tag_is_part_of_the_identity(self):
        self.assertEqual(
            hoi_chain.derive_eval_run_id(
                "p1-hoi-p9-w3-s42-20260807", "20260812", tag="unguided"
            ),
            "p1-hoi-p9-w3-eval-unguided-s42-20260812",
        )

    def test_every_shipped_arm_that_names_a_run_derives_a_valid_pair_of_run_ids(self):
        arms = sorted(
            path.stem
            for path in (REPO / "code" / "config").glob("config_train_hoi_prior_*.yaml")
        )
        self.assertGreaterEqual(len(arms), 11)
        named = 0
        for name in arms:
            with self.subTest(arm=name):
                try:
                    train = hoi_chain.read_run_id(REPO, name)
                except hoi_chain.ChainError:
                    continue
                named += 1
                hoi_chain.validate_run_id(train)
                hoi_chain.validate_run_id(
                    hoi_chain.derive_eval_run_id(train, "20260812")
                )
        self.assertGreaterEqual(named, 11)

    def test_the_arms_that_never_named_a_run_are_refused_not_guessed(self):
        """p8h1 and d2ag resolve run_id: null; their ids were supplied at launch.

        Predates the recipe refactor -- the rewrite preserved every resolved
        value exactly. The chain refuses them rather than inventing an identity;
        rerunning either one means giving it a run id first.
        """
        for arm in ("config_train_hoi_prior_p8h1", "config_train_hoi_prior_d2ag"):
            with self.subTest(arm=arm):
                with self.assertRaises(hoi_chain.ChainError):
                    hoi_chain.read_run_id(REPO, arm)

    def test_a_malformed_run_id_is_refused(self):
        for bad in ("p1-hoi-w3", "hoi-w3-s42-20260807", "p1-hoi-w3-s42-2026080"):
            with self.subTest(run_id=bad):
                with self.assertRaises(hoi_chain.ChainError):
                    hoi_chain.validate_run_id(bad)


class PreflightTests(unittest.TestCase):
    def test_a_dirty_worktree_is_refused_before_anything_is_consumed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            (repo / "scratch.txt").write_text("untracked\n", encoding="utf-8")
            with self.assertRaises(hoi_chain.ChainError) as raised:
                hoi_chain.preflight(repo, None, 0)
            self.assertIn("clean worktree", str(raised.exception))

    def test_the_wrong_host_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            with self.assertRaises(hoi_chain.ChainError) as raised:
                hoi_chain.preflight(repo, "not-this-host", 0)
            self.assertIn("meant for not-this-host", str(raised.exception))

    def test_a_clean_repository_on_the_right_host_passes(self):
        import socket

        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            _make_repo(repo)
            state = hoi_chain.preflight(repo, socket.gethostname(), 0)
            self.assertTrue(state["worktree_clean"])
            self.assertEqual(len(state["git_head"]), 40)

    def test_a_busy_host_is_refused(self):
        original = hoi_chain.free_gpu_count
        hoi_chain.free_gpu_count = lambda: 1
        try:
            with tempfile.TemporaryDirectory() as directory:
                repo = Path(directory)
                _make_repo(repo)
                with self.assertRaises(hoi_chain.ChainError) as raised:
                    hoi_chain.preflight(repo, None, 4)
                self.assertIn("4 idle GPUs required, 1 idle", str(raised.exception))
        finally:
            hoi_chain.free_gpu_count = original


class CheckpointDiscoveryTests(unittest.TestCase):
    def _run_dir(self, root: Path, run_id: str) -> Path:
        directory = root / "results" / "experiments" / run_id / "checkpoints"
        directory.mkdir(parents=True)
        return directory

    def test_the_highest_window_checkpoint_is_the_final_identity(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = self._run_dir(root, run_id)
            for windows in (3072000, 299520000, 150000000):
                (checkpoints / f"{run_id}_windows{windows:09d}.pth").write_bytes(b"x")
            self.assertEqual(
                hoi_chain.final_checkpoint(root, run_id).name,
                f"{run_id}_windows299520000.pth",
            )

    def test_zero_padding_does_not_make_it_a_string_comparison(self):
        """windows009000000 sorts above windows299520000 as text."""
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = self._run_dir(root, run_id)
            (checkpoints / f"{run_id}_windows009000000.pth").write_bytes(b"x")
            (checkpoints / f"{run_id}_windows299520000.pth").write_bytes(b"x")
            self.assertEqual(
                hoi_chain.final_checkpoint(root, run_id).name,
                f"{run_id}_windows299520000.pth",
            )

    def test_another_runs_checkpoints_are_never_picked_up(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoints = self._run_dir(root, run_id)
            (checkpoints / f"{run_id}_windows003072000.pth").write_bytes(b"x")
            (checkpoints / "p1-hoi-other-s42-20260812_windows299520000.pth").write_bytes(b"x")
            self.assertEqual(
                hoi_chain.final_checkpoint(root, run_id).name,
                f"{run_id}_windows003072000.pth",
            )

    def test_a_missing_checkpoint_tree_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(hoi_chain.ChainError):
                hoi_chain.final_checkpoint(Path(directory), "p1-hoi-x-s42-20260812")


class CommandTests(unittest.TestCase):
    def test_training_uses_the_generic_trainer_under_its_hydra_config(self):
        self.assertEqual(
            hoi_chain.train_command("py", "config_train_hoi_prior_p9w3"),
            ["py", "code/train_hoi_prior.py", "--config-name=config_train_hoi_prior_p9w3"],
        )

    def test_evaluation_uses_the_fixed_native_evaluator(self):
        command = hoi_chain.evaluate_command(
            "py", REPO, "p1-hoi-x-eval-guided-s42-20260812", Path("/tmp/final.pth")
        )
        self.assertEqual(
            command[1], str((REPO / "code" / "test_infbagel_hoi.py").resolve())
        )
        self.assertIn("--config-name=config_eval_hoi_prior", command)
        self.assertIn("exp_name=p1-hoi-x-eval-guided-s42-20260812", command)
        self.assertIn("ckpt_path=/tmp/final.pth", command)

    def test_evaluation_overrides_are_appended_verbatim(self):
        overrides = [
            "sampler.pelvis.guidance.enabled=true",
            "sampler.pelvis.guidance.guidance_scale=1000.0",
        ]
        command = hoi_chain.evaluate_command(
            "py", REPO, "p1-hoi-x-eval-guided-s42-20260812", Path("/tmp/final.pth"),
            overrides,
        )
        self.assertEqual(command[-2:], overrides)

    def test_eval_override_is_repeatable(self):
        args = hoi_chain.build_parser().parse_args([
            "--arm", "p9w3",
            "--eval-override", "a=1",
            "--eval-override", "b=x y",
        ])
        self.assertEqual(args.eval_override, ["a=1", "b=x y"])

    def test_no_stage_invokes_a_per_experiment_wrapper(self):
        commands = [
            hoi_chain.train_command("py", "config_train_hoi_prior_p9w3"),
            hoi_chain.evaluate_command(
                "py", REPO, "p1-hoi-x-eval-guided-s42-20260812", Path("c.pth")
            ),
            hoi_chain.bootstrap_command("py", "a", "b", Path("o.json")),
        ]
        entry_points = {command[1] for command in commands}
        self.assertEqual(
            entry_points,
            {
                "code/train_hoi_prior.py",
                str((REPO / "code" / "test_infbagel_hoi.py").resolve()),
                "tools/paired_bootstrap.py",
            },
        )


class StageBookkeepingTests(unittest.TestCase):
    def test_evaluate_runs_from_code_but_train_and_bootstrap_run_from_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(
                hoi_chain.stage_working_directory(root, "evaluate"),
                (root / "code").resolve(),
            )
            for stage in ("train", "bootstrap"):
                with self.subTest(stage=stage):
                    self.assertEqual(
                        hoi_chain.stage_working_directory(root, stage), root.resolve()
                    )

    def test_a_completed_stage_is_not_rerun(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertFalse(hoi_chain.stage_completed(root, run_id, "train"))
            hoi_chain.write_stage_status(root, run_id, "train", {"status": "completed"})
            self.assertTrue(hoi_chain.stage_completed(root, run_id, "train"))

    def test_a_failed_stage_is_not_treated_as_completed(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            hoi_chain.write_stage_status(root, run_id, "train", {"status": "failed"})
            self.assertFalse(hoi_chain.stage_completed(root, run_id, "train"))

    def test_status_lives_under_the_git_ignored_run_directory(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = hoi_chain.write_stage_status(root, run_id, "train", {"status": "running"})
            self.assertEqual(
                path.relative_to(root).parts[:3],
                ("results", "experiments", run_id),
            )

    def test_a_failing_stage_raises_and_records_the_failure(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaises(hoi_chain.ChainError) as raised:
                hoi_chain.run_stage(root, run_id, "train",
                                    [sys.executable, "-c", "raise SystemExit(3)"], root)
            self.assertIn("do not reuse this run id", str(raised.exception))
            record = json.loads(
                hoi_chain.stage_status_path(root, run_id, "train").read_text(encoding="utf-8")
            )
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["returncode"], 3)

    def test_a_succeeding_stage_is_recorded_as_completed(self):
        run_id = "p1-hoi-x-s42-20260812"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            record = hoi_chain.run_stage(root, run_id, "train",
                                         [sys.executable, "-c", "print('ok')"], root)
            self.assertEqual(record["status"], "completed")
            self.assertEqual(record["cwd"], str(root.resolve()))
            self.assertTrue(hoi_chain.stage_completed(root, run_id, "train"))

    def test_evaluation_status_records_overrides(self):
        run_id = "p1-hoi-x-s42-20260812"
        overrides = ["sampler.pelvis.guidance.enabled=true", "x=y"]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "code").mkdir()
            record = hoi_chain.run_stage(
                root,
                run_id,
                "evaluate",
                [sys.executable, "-c", "pass", *overrides],
                root / "code",
                extra={"eval_overrides": overrides},
            )
            persisted = json.loads(
                hoi_chain.stage_status_path(root, run_id, "evaluate").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(record["eval_overrides"], overrides)
            self.assertEqual(persisted["eval_overrides"], overrides)


class GovernanceTests(unittest.TestCase):
    def test_the_chain_never_commits_tags_or_writes_the_registry(self):
        source = (REPO / "tools" / "hoi_chain.py").read_text(encoding="utf-8")
        body = source.split('"""', 2)[2]
        for forbidden in ("registry.jsonl", '"commit"', '"tag"', "experiment.py"):
            self.assertNotIn(forbidden, body, f"the chain must not touch {forbidden}")

    def test_the_chain_is_documented_as_dispatchable_from_the_authority_host(self):
        source = (REPO / "tools" / "hoi_chain.py").read_text(encoding="utf-8")
        self.assertIn("no checkpoint leaves the worker", source)


if __name__ == "__main__":
    unittest.main()
