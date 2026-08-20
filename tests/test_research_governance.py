import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

try:
    import numpy as np
except ImportError:  # Minimal governance checks run before ML dependencies are installed.
    np = None

from tools import (
    chois_evaluator, experiment, make_lingo_split, measure_hoi_repr_ceiling,
    run_chois_evaluator,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class SplitTests(unittest.TestCase):
    def test_scene_variants_stay_together_and_split_is_deterministic(self):
        records = [
            {"sequence_id": "0", "scene_name": "room_a"},
            {"sequence_id": "1", "scene_name": "room_a_mirror"},
            {"sequence_id": "2", "scene_name": "room_b_new-loco"},
            {"sequence_id": "3", "scene_name": "room_b"},
            {"sequence_id": "4", "scene_name": "room_c_action_variant_2"},
            {"sequence_id": "5", "scene_name": "room_d"},
            {"sequence_id": "6", "scene_name": "room_e_aug_1"},
            {"sequence_id": "7", "scene_name": "088-take_shower"},
            {"sequence_id": "8", "scene_name": "088-wash"},
        ]
        first = make_lingo_split.build_split(records, 42, 0.2, {})
        second = make_lingo_split.build_split(records, 42, 0.2, {})
        self.assertEqual(first, second)
        self.assertEqual(first["scene_to_family"]["room_a_mirror"], "room_a")
        self.assertEqual(first["scene_to_family"]["room_b_new-loco"], "room_b")
        self.assertEqual(first["scene_to_family"]["088-take_shower"], "088")
        self.assertEqual(first["scene_to_family"]["088-wash"], "088")
        train = set(first["train"]["scene_families"])
        validation = set(first["validation"]["scene_families"])
        self.assertFalse(train & validation)

    def test_duplicate_sequence_is_rejected(self):
        records = [
            {"sequence_id": "0", "scene_name": "a"},
            {"sequence_id": "0", "scene_name": "b"},
        ]
        with self.assertRaises(make_lingo_split.SplitError):
            make_lingo_split.build_split(records, 42, 0.2, {})

class ManifestTests(unittest.TestCase):
    def test_hash_bound_completion_transition_is_exact_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "docs").mkdir()
            tracked = repo / "docs" / "EXPERIMENT_PLAN.md"
            tracked.write_text("initial\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Codex Test",
                    "-c", "user.email=codex@example.invalid",
                    "commit", "-q", "-m", "initial",
                ],
                cwd=repo,
                check=True,
            )
            source = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            ).strip()
            tracked.write_text("amended\n", encoding="utf-8")
            subprocess.run(["git", "add", "."], cwd=repo, check=True)
            subprocess.run(
                [
                    "git", "-c", "user.name=Codex Test",
                    "-c", "user.email=codex@example.invalid",
                    "commit", "-q", "-m", "governance",
                ],
                cwd=repo,
                check=True,
            )
            target = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=repo, text=True,
            ).strip()
            diff = subprocess.check_output(
                ["git", "diff", "--binary", source, target], cwd=repo,
            )
            diff_sha256 = hashlib.sha256(diff).hexdigest()
            manifest = {"git": {"commit": source}}
            current = experiment.git_state(repo)
            metrics = {
                "git_commit": target,
                "resume_commit_provenance": {
                    "mode": "explicit_bound_transition",
                    "checkpoint_git_commit": source,
                    "current_git_commit": target,
                    "diff_sha256": diff_sha256,
                    "changed_paths": ["docs/EXPERIMENT_PLAN.md"],
                },
            }
            args = SimpleNamespace(
                commit_transition_source=source,
                commit_transition_target=target,
                commit_transition_diff_sha256=diff_sha256,
                commit_transition_allow_path=["docs/EXPERIMENT_PLAN.md"],
            )
            transition = experiment._finish_commit_transition(
                repo, manifest, current, metrics, args,
            )
            self.assertEqual(transition["target_commit"], target)
            args.commit_transition_allow_path = ["docs/EXPERIMENT_PLAN.md", "unexpected"]
            with self.assertRaises(experiment.ManifestError):
                experiment._finish_commit_transition(
                    repo, manifest, current, metrics, args,
                )

    def test_phase_handoff_contract_is_tracked(self):
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        plan = (REPO_ROOT / "docs" / "plan" / "OVERVIEW.md").read_text(encoding="utf-8")
        summary_contract = (
            REPO_ROOT / "docs" / "phase_summaries" / "README.md"
        ).read_text(encoding="utf-8")
        self.assertIn("Scope one working session to one phase", agents)
        self.assertIn("docs/phase_summaries/PHASE_<N>.md", agents)
        self.assertIn("阶段粒度与交接约定", plan)
        self.assertIn("exact prerequisites and first action", summary_contract)
        phase_zero = (
            REPO_ROOT / "docs" / "phase_summaries" / "PHASE_0.md"
        ).read_text(encoding="utf-8")
        for required_section in (
            "Scope and gate decision",
            "Failed and negative runs retained",
            "Verification",
            "Artifacts and hashes",
            "Exact next-session entry point",
        ):
            self.assertIn(required_section, phase_zero)

    def test_hydra_guidance_has_no_missing_dataset_interpolation(self):
        guidance = (REPO_ROOT / "code" / "config" / "guidance" / "pelvis.yaml").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("${dataset.seq_len}", guidance)

    def test_training_effective_batch_contract_is_configured(self):
        config = (REPO_ROOT / "code" / "config" / "config_train_infbagel.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("batch_size: 128", config)
        self.assertIn("effective_batch_size: 1024", config)
        self.assertIn("gradient_accumulation_steps: 1", config)
        self.assertIn("num_gpus: 8", config)
        trainer = (REPO_ROOT / "code" / "train_infbagel.py").read_text(encoding="utf-8")
        self.assertIn("effective batch mismatch", trainer)
        self.assertIn("loss / int(cfg.gradient_accumulation_steps)", trainer)

        protocol_path = REPO_ROOT / "experiments" / "training_resource_protocol.json"
        protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        self.assertEqual(protocol["effective_batch"]["default_candidates"], [512, 1024, 2048, 3072])
        self.assertEqual(protocol["selection_scope"], "independent_per_expert")
        self.assertEqual(protocol["primary_budget_units"], ["processed_windows", "processed_frames"])
        for value in (512, 1024, 2048, 3072):
            experiment.validate_effective_batch(value, protocol)
        with self.assertRaises(experiment.ManifestError):
            experiment.validate_effective_batch(1536, protocol)
        experiment.validate_effective_batch(4096, protocol, allow_extended=True)

        rules = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        self.assertIn("`{512, 1024, 2048, 3072}`", rules)
        self.assertIn("`1536` are forbidden", rules)
        self.assertIn("processed windows or", rules)
        self.assertEqual(protocol["hardware_assignment"]["hoi"]["gpu_count"], 4)
        self.assertEqual(protocol["hardware_assignment"]["hsi"]["gpu_count"], 8)
        self.assertEqual(protocol["memory_audit_phase"], {"hoi": "1B", "hsi": "1C"})
        experiment.validate_training_resource_protocol(protocol_path)

        phase_1b = protocol["phase_1b_preregistration"]
        self.assertEqual(phase_1b["model"]["dimension"], 232)
        self.assertFalse(phase_1b["model"]["scene_condition"])
        self.assertEqual(phase_1b["memory_audit"]["micro_batch_candidates"], [128, 256, 512, 768])
        self.assertEqual(phase_1b["memory_audit"]["selected_micro_batch_per_gpu"], 768)
        self.assertEqual(phase_1b["memory_audit"]["selected_effective_batch_size"], 3072)
        self.assertEqual(phase_1b["screening"]["optimizer_updates_per_candidate"], 1024)
        self.assertEqual(phase_1b["screening"]["validation_cadence_updates"], 1024)
        self.assertEqual(phase_1b["screening"]["checkpoint_cadence_updates"], 1024)
        self.assertEqual(phase_1b["formal_training"]["seeds"], [42])
        self.assertEqual(phase_1b["formal_training"]["processed_windows_per_seed"], 61440000)
        self.assertEqual(phase_1b["formal_training"]["optimizer_updates_per_seed"], 20000)
        self.assertEqual(phase_1b["formal_training"]["learning_rate"], 0.0003)
        self.assertEqual(phase_1b["formal_training"]["warmup_windows"], 1572864)

        remediation = protocol["phase_1b_remediation_preregistration"]
        self.assertEqual(remediation["seed"], 42)
        self.assertEqual(remediation["planning_evidence"]["hoi_train_windows"], 597868)
        self.assertEqual(
            remediation["planning_evidence"]["hoi_train_need_pelvis_windows"],
            597868,
        )
        self.assertFalse(remediation["diagnostics"]["official_test_used_for_selection"])
        self.assertEqual(remediation["diagnostics"]["rollout_sequences"], 128)
        self.assertEqual(
            remediation["diagnostics"]["existing_checkpoint_weight_variants"],
            ["online", "ema_0.9999"],
        )
        self.assertEqual(
            remediation["representation_repairs"]["object_goal_loss_mask"],
            "end_pi == seq_length",
        )
        self.assertEqual(
            remediation["representation_repairs"]["pelvis_goal_legacy_replay_max_abs"],
            0.00001,
        )
        self.assertEqual(
            remediation["representation_repairs"]["loss_weights"]["object_surface"],
            50.0,
        )
        candidates = remediation["screening"]["candidates"]
        self.assertEqual([item["effective_batch_size"] for item in candidates], [1024, 3072])
        self.assertEqual([item["optimizer_updates"] for item in candidates], [6000, 2000])
        self.assertEqual(
            remediation["screening"]["checkpoint_weight_variants"],
            ["online", "ema_0.999", "ema_0.9999"],
        )
        self.assertEqual(remediation["conditional_geometry_fallback"]["maximum_candidates"], 1)
        self.assertTrue(
            remediation["conditional_geometry_fallback"]["must_pass_full_d2_eligibility"]
        )
        self.assertEqual(
            remediation["conditional_geometry_fallback"]["contact_geometry_channels"],
            [0, 1],
        )
        self.assertFalse(remediation["formal_training"]["screening_checkpoint_initialization"])
        self.assertEqual(
            remediation["formal_training"]["official_test_runs_after_lock"],
            {"native": 1, "chois": 1},
        )

    def test_diffusion_training_loss_has_no_stale_cleanup_variables(self):
        model = (REPO_ROOT / "code" / "models" / "infbagel.py").read_text(
            encoding="utf-8"
        )
        loss_body = model.split("    def p_losses(", 1)[1].split(
            "    def p_sample_loop(", 1
        )[0]
        self.assertNotIn("occ_goal", loss_body)
        self.assertNotIn("occ_temp", loss_body)

    def test_run_id_binds_phase_and_seed(self):
        experiment.validate_run_id("p1-hoi-smoke-s42-20260711", "p1", 42)
        with self.assertRaises(experiment.ManifestError):
            experiment.validate_run_id("p1-hoi-smoke-s7-20260711", "p1", 42)

    def test_manifest_cli_records_overrides_and_resolved_config(self):
        parser = experiment.build_parser()
        start = parser.parse_args([
            "start", "--id", "p0-test-smoke-s42-20260712", "--phase", "p0",
            "--seed", "42", "--config", "config.yaml", "--override", "device=cuda:0",
            "--command", "python evaluate.py", "--workdir", "code",
        ])
        self.assertEqual(start.override, ["device=cuda:0"])
        self.assertEqual(start.run_command, "python evaluate.py")
        self.assertEqual(start.workdir, "code")
        finish = parser.parse_args([
            "finish", "--manifest", "manifest.json", "--metrics", "metrics.json",
            "--status", "completed", "--resolved-config", "resolved.yaml",
        ])
        self.assertEqual(finish.resolved_config, "resolved.yaml")

    def test_directory_hash_depends_on_names_and_content(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "a").write_text("same", encoding="utf-8")
            before = experiment.sha256_path(root)["sha256"]
            (root / "a").rename(root / "b")
            after = experiment.sha256_path(root)["sha256"]
            self.assertNotEqual(before, after)

    def test_registry_duplicate_is_rejected(self):
        record = {
            "schema_version": 1,
            "experiment_id": "p0-test-s42-20260711",
            "phase": "p0",
            "status": "failed",
            "hypothesis": "h",
            "config": {},
            "results": {},
            "conclusion": "c",
            "next_action": "n",
            "created_at": "2026-07-11T00:00:00Z",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "registry.jsonl"
            line = json.dumps(record) + "\n"
            path.write_text(line + line, encoding="utf-8")
            with self.assertRaises(experiment.ManifestError):
                experiment.validate_registry(path)

    def test_split_validation_rejects_family_leakage(self):
        value = {
            "algorithm": "scene-family-disjoint-v1",
            "seed": 42,
            "validation_ratio": 0.2,
            "scene_to_family": {"a": "room"},
            "train": {"scene_families": ["room"], "scenes": ["a"]},
            "validation": {"scene_families": ["room"], "scenes": []},
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "split.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(experiment.ManifestError):
                experiment.validate_split(path)


@unittest.skipIf(np is None, "numpy is not installed")
class ChoisEvaluatorTests(unittest.TestCase):
    def _write_npz(self, folder, name, frames=8):
        folder.mkdir(parents=True, exist_ok=True)
        np.savez(
            folder / f"{name}.npz",
            seq_name=np.asarray(name),
            global_jpos=np.zeros((frames, 24, 3), dtype=np.float32),
        )

    def test_input_pair_contract(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "pred"
            truth = root / "gt"
            self._write_npz(predictions, "sub1_chair_000")
            self._write_npz(truth, "sub1_chair_000")
            result = chois_evaluator.validate_pair(predictions, truth)
            self.assertEqual(result["sequence_count"], 1)

    def test_input_pair_rejects_mismatched_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            predictions = root / "pred"
            truth = root / "gt"
            self._write_npz(predictions, "a")
            self._write_npz(truth, "b")
            with self.assertRaises(chois_evaluator.EvaluatorError):
                chois_evaluator.validate_pair(predictions, truth)

    def test_pinned_checkout_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "t2m_eval").mkdir(parents=True)
            evaluator = root / "t2m_eval" / "final_evaluations.py"
            evaluator.write_text("# evaluator\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "init"],
                cwd=root,
                check=True,
            )
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            config = {
                "upstream_commit": commit,
                "files": {"t2m_eval/final_evaluations.py": chois_evaluator.sha256_file(evaluator)},
            }
            result = chois_evaluator.verify_upstream(root, config)
            self.assertEqual(result["commit"], commit)

    def test_text_to_motion_checkout_rejects_unpinned_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "options").mkdir()
            module = root / "options" / "train_options.py"
            module.write_text("# parser\n", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "."], cwd=root, check=True)
            subprocess.run(
                ["git", "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "init"],
                cwd=root,
                check=True,
            )
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
            config = {
                "upstream_commit": commit,
                "files": {"options/train_options.py": chois_evaluator.sha256_file(module)},
            }
            result = run_chois_evaluator.verify_text_to_motion(root, config)
            self.assertEqual(result["commit"], commit)
            module.write_text("changed\n", encoding="utf-8")
            with self.assertRaises(run_chois_evaluator.AdapterError):
                run_chois_evaluator.verify_text_to_motion(root, config)

    def test_default_path_schema_is_stable_and_has_no_comparison_block(self):
        parser = run_chois_evaluator.build_parser()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / "metrics.json"
            parsed = parser.parse_args([
                "--chois-root", str(root / "chois"),
                "--text-to-motion-root", str(root / "text-to-motion"),
                "--predictions", str(root / "predictions"),
                "--ground-truth", str(root / "ground-truth"),
                "--data-root", str(root / "data"),
                "--glove-root", str(root / "glove"),
                "--checkpoints-dir", str(root / "checkpoints"),
                "--checkpoint", str(root / "checkpoints" / "omomo" / "text_motion_features" / "model" / "finest.tar"),
                "--device", "cpu",
                "--output", str(output),
            ])
            self.assertEqual(parsed.compare_predictions, [])
            self.assertFalse(parsed.emit_offset_corrected_fid)
            info = {"seq": {"frames": 126}}
            embeddings = np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
            fake_truth = {"motion_embeddings": embeddings, "sequence_ids": ["seq"]}
            fake_prediction = {
                "motion_embeddings": embeddings + 1.0,
                "sequence_ids": ["seq"],
                "matching_score": 1.0,
                "r_precision": [0.1, 0.2, 0.3],
            }
            fake_dataset = SimpleNamespace(sequence_ids=["seq"])

            def activation_statistics(value):
                return np.mean(value, axis=0), np.eye(value.shape[1])

            metrics = {
                "activation_statistics": activation_statistics,
                "frechet": lambda *unused: 1.25,
                "diversity": lambda *unused: 2.5,
            }
            with mock.patch.object(run_chois_evaluator.chois_evaluator, "load_config", return_value={}), \
                    mock.patch.object(run_chois_evaluator.chois_evaluator, "verify_upstream", return_value={}), \
                    mock.patch.object(run_chois_evaluator, "verify_text_to_motion", return_value={}), \
                    mock.patch.object(run_chois_evaluator.chois_evaluator, "require_assets", return_value={}), \
                    mock.patch.object(run_chois_evaluator.chois_evaluator, "read_npz_directory", side_effect=[(info, "prediction-tree"), (info, "truth-tree")]), \
                    mock.patch.object(run_chois_evaluator, "_load_components", return_value=(object(), object(), metrics)), \
                    mock.patch.object(run_chois_evaluator, "PathConfiguredCHOISEvaluationDataset", return_value=fake_dataset), \
                    mock.patch.object(run_chois_evaluator, "_loader", return_value=object()), \
                    mock.patch.object(run_chois_evaluator, "_embeddings", side_effect=[fake_truth, fake_prediction]), \
                    mock.patch.object(run_chois_evaluator.chois_evaluator, "atomic_output") as atomic_output:
                result = run_chois_evaluator.evaluate(parsed)

            self.assertEqual(
                set(result),
                {
                    "schema_version", "created_at", "adapter", "upstream",
                    "text_to_motion_dependency", "assets", "inputs",
                    "embedding_protocol", "runtime", "metrics", "uncertainty",
                },
            )
            self.assertNotIn("comparison", result)
            self.assertNotIn("offset_corrected_fid", result)
            self.assertNotIn("paired_differences", json.dumps(result, sort_keys=True))
            atomic_output.assert_called_once()

    def test_offset_diagnostic_has_no_per_sequence_variant(self):
        parser = run_chois_evaluator.build_parser()
        option_names = {
            option
            for action in parser._actions
            for option in action.option_strings
        }
        self.assertIn("--emit-offset-corrected-fid", option_names)
        self.assertNotIn("--emit-offset-corrected-fid-per-sequence", option_names)
        self.assertNotIn("--offset-corrected-fid-per-sequence", option_names)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            required = [
                "--chois-root", str(root / "chois"),
                "--text-to-motion-root", str(root / "text-to-motion"),
                "--predictions", str(root / "predictions"),
                "--ground-truth", str(root / "ground-truth"),
                "--data-root", str(root / "data"),
                "--glove-root", str(root / "glove"),
                "--checkpoints-dir", str(root / "checkpoints"),
                "--checkpoint", str(root / "checkpoints" / "omomo" / "text_motion_features" / "model" / "finest.tar"),
                "--output", str(root / "metrics.json"),
            ]
            with self.assertRaises(SystemExit):
                parser.parse_args(required + ["--emit-offset-corrected-fid-per-sequence"])

    def test_g3_frame_count_guard_fails_closed(self):
        with self.assertRaisesRegex(run_chois_evaluator.AdapterError, "G3 frame-count gate"):
            run_chois_evaluator._require_aligned_frames(
                {
                    "ground_truth": {"a": 126, "b": 126},
                    "predictions": {"a": 126, "b": 125},
                },
                expected_frames=126,
            )

    def test_shared_resample_prefix_is_bitwise_reproducible(self):
        long_stream = run_chois_evaluator._shared_resample_indices(16, 2000, 42)
        short_stream = run_chois_evaluator._shared_resample_indices(16, 200, 42)
        np.testing.assert_array_equal(long_stream[:200], short_stream)


class RepresentationCeilingProbeTests(unittest.TestCase):
    """Governance contract of tools/measure_hoi_repr_ceiling.py.

    The probe produces a permanent reference row that later phases cite, so what
    is pinned here is the envelope, not the numbers: CPU only, no checkpoint, an
    output that is never overwritten, an exclusion list copied from the evaluator
    rather than re-decided, and a classification that cannot seal a partial run.
    """

    SOURCE = REPO_ROOT / "tools" / "measure_hoi_repr_ceiling.py"

    def test_execution_contract_declares_no_gpu_and_no_checkpoint(self):
        contract = measure_hoi_repr_ceiling.EXECUTION_CONTRACT
        self.assertEqual(contract["device"], "cpu")
        for key in (
            "requires_gpu", "requires_checkpoint", "requires_model_inference",
            "requires_training", "writes_inside_run_directory",
        ):
            self.assertIs(contract[key], False, key)
        self.assertIs(contract["read_only_inputs"], True)

    def test_cli_offers_no_device_checkpoint_or_model_option(self):
        options = {
            option
            for action in measure_hoi_repr_ceiling.build_parser()._actions
            for option in action.option_strings
        }
        self.assertIn("--output", options)
        for forbidden in ("--device", "--gpu", "--checkpoint", "--ckpt-path", "--ckpt"):
            self.assertNotIn(forbidden, options)

    def test_source_carries_no_gpu_or_checkpoint_dependency(self):
        source = self.SOURCE.read_text(encoding="utf-8")
        for forbidden in (
            "torch.load", "torch.cuda", ".cuda(", "cuda:",
            "load_trained_hoi_prior", "init_model",
        ):
            self.assertNotIn(forbidden, source, forbidden)

    def test_output_path_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as temporary:
            existing = Path(temporary) / "reference_row.json"
            existing.write_text("{}\n", encoding="utf-8")
            args = measure_hoi_repr_ceiling.build_parser().parse_args(
                ["--output", str(existing), "--sequences", "1"]
            )
            with self.assertRaisesRegex(measure_hoi_repr_ceiling.CeilingError, "refusing to overwrite"):
                measure_hoi_repr_ceiling.run(args)
            self.assertEqual(measure_hoi_repr_ceiling.main(["--output", str(existing)]), 2)
            self.assertEqual(existing.read_text(encoding="utf-8"), "{}\n")

    def test_penetration_exclusion_is_copied_from_the_evaluator(self):
        evaluator = (REPO_ROOT / "code" / "test_infbagel_hoi.py").read_text(encoding="utf-8")
        marker = "if obj_name not in ["
        self.assertIn(marker, evaluator)
        literal = evaluator[evaluator.index(marker) + len(marker):]
        literal = literal[:literal.index("]")]
        excluded = {piece.strip().strip("'\"") for piece in literal.split(",")}
        self.assertEqual(excluded, set(measure_hoi_repr_ceiling.PENETRATION_EXCLUDED_OBJECTS))

    def test_classification_cannot_seal_a_partial_run(self):
        passing = [
            {"gate": gate, "status": "pass"}
            for gate in (
                "A1_gt_foot_sliding", "A2_gt_feet_height", "A3_gt_contact_percent",
                "A4_zero_gt_contact_sequences", "A4b_analytic_contact_cap",
                "A5_penetration_covered_sequences", "A6_human_pen_scaling",
                "A7_sdf_out_of_box_guard", "E1_agrees_with_2026_08_20_exploration",
            )
        ]
        classify = measure_hoi_repr_ceiling.classify
        self.assertEqual(
            classify(passing, penetration=True, full_protocol=True),
            "repr-ceiling-row-established",
        )
        self.assertEqual(
            classify(passing, penetration=True, full_protocol=False),
            "repr-ceiling-subset-smoke",
        )
        self.assertEqual(
            classify(passing, penetration=False, full_protocol=True),
            "repr-ceiling-penetration-partial",
        )
        # A7 warns when a ground-truth penetration ratio is exactly 0.0 while its
        # loss is non-zero; that is an observation, not a failed gate.
        warned = [
            dict(gate, status="warn") if gate["gate"] == "A7_sdf_out_of_box_guard" else gate
            for gate in passing
        ]
        self.assertEqual(
            classify(warned, penetration=True, full_protocol=True),
            "repr-ceiling-row-established",
        )
        for gate_id, expected in (
            ("A1_gt_foot_sliding", "repr-ceiling-anchor-fail-stop"),
            ("A4_zero_gt_contact_sequences", "repr-ceiling-anchor-fail-stop"),
            ("A5_penetration_covered_sequences", "repr-ceiling-penetration-partial"),
            ("A7_sdf_out_of_box_guard", "repr-ceiling-penetration-partial"),
            ("E1_agrees_with_2026_08_20_exploration", "repr-ceiling-contradicts-exploration"),
        ):
            failed = [
                dict(gate, status="fail") if gate["gate"] == gate_id else gate
                for gate in passing
            ]
            self.assertEqual(
                classify(failed, penetration=True, full_protocol=True), expected, gate_id
            )
        self.assertIn(
            "repr-ceiling-subset-smoke", measure_hoi_repr_ceiling.STOP_CLASSIFICATIONS
        )

    def test_analytic_contact_cap_is_397_over_438(self):
        self.assertAlmostEqual(
            measure_hoi_repr_ceiling.ANALYTIC_CONTACT_CAP, 0.906392694063927, places=15
        )
        self.assertEqual(measure_hoi_repr_ceiling.HUMAN_PEN_SCALE, 10475 / 100)

if __name__ == "__main__":
    unittest.main()
