import json
import subprocess
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # Minimal governance checks run before ML dependencies are installed.
    np = None

from tools import (
    chois_evaluator, experiment, make_hoi_split, make_lingo_split, run_chois_evaluator,
    summarize_hoi_phase1b,
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

    def test_hoi_split_generator_reproduces_locked_manifest(self):
        generated = make_hoi_split.build_split(REPO_ROOT)
        tracked = json.loads((
            REPO_ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(generated, tracked)
        experiment.validate_split(
            REPO_ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )


class ManifestTests(unittest.TestCase):
    def test_phase_handoff_contract_is_tracked(self):
        agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        plan = (REPO_ROOT / "docs" / "EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
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

    def test_hoi_evaluation_contract_is_configured(self):
        config = (REPO_ROOT / "code" / "config" / "config_sample_infbagel.yaml").read_text(
            encoding="utf-8"
        )
        self.assertIn("hoi_expected_sequences: 438", config)
        self.assertIn("hoi_sequence_limit: null", config)
        self.assertIn("hoi_timing_warmup: true", config)
        self.assertIn("chois_eval_ground_truth_dir:", config)
        evaluator = (REPO_ROOT / "code" / "test_infbagel_hoi.py").read_text(encoding="utf-8")
        self.assertIn("for seg_id_true in range(len(seq_name_dict)):", evaluator)

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

    def test_phase_1b_training_and_evaluation_paths_are_scene_free(self):
        train_config = (REPO_ROOT / "code/config/config_train_hoi_prior.yaml").read_text(encoding="utf-8")
        eval_config = (REPO_ROOT / "code/config/config_eval_hoi_prior.yaml").read_text(encoding="utf-8")
        sampler_config = (REPO_ROOT / "code/config/sampler/hoi_prior.yaml").read_text(encoding="utf-8")
        trainer = (REPO_ROOT / "code/train_hoi_prior.py").read_text(encoding="utf-8")
        model = (REPO_ROOT / "code/priors/models.py").read_text(encoding="utf-8")
        self.assertIn("dim_model: 512", train_config)
        self.assertIn("num_heads: 16", train_config)
        self.assertIn("num_layers: 8", train_config)
        self.assertIn("diffusion_steps: 500", train_config)
        self.assertIn("load_scene: false", eval_config)
        self.assertIn("_target_: priors.diffusion.HOIPriorSampler", sampler_config)
        hoi_body = model.split("class HOIPrior", 1)[1].split("class HSIPrior", 1)[0]
        self.assertNotIn("scene_condition:", hoi_body)
        self.assertIn("init_checkpoint is forbidden", trainer)
        self.assertIn("resume checkpoint training contract mismatch", trainer)
        self.assertIn("resume checkpoint Git commit mismatch", trainer)
        self.assertIn("max_consecutive_amp_overflows: 16", train_config)
        self.assertIn("amp_overflow_skips_by_rank", trainer)
        self.assertIn("scaler.update(new_scale=float(scaler.get_scale()) * 0.5)", trainer)

    def test_phase_1b_capacity_audit_keeps_all_preregistered_candidates(self):
        source = (REPO_ROOT / "tools/audit_hoi_capacity.py").read_text(encoding="utf-8")
        self.assertIn("CANDIDATES = ((128, 512), (256, 1024), (512, 2048), (768, 3072))", source)
        self.assertIn("PROCESSED_WINDOWS = 24576", source)
        self.assertIn("all_failures_and_ooms_retained", source)

    def test_phase_1b_statistics_protocol_is_locked(self):
        summary = summarize_hoi_phase1b.seed_summary([2.0])
        self.assertEqual(summary["mean"], 2.0)
        self.assertEqual(summary["point_estimate"], 2.0)
        self.assertNotIn("student_t_95_ci", summary)
        self.assertEqual(summarize_hoi_phase1b.SEEDS, (42,))
        self.assertEqual(summarize_hoi_phase1b.BOOTSTRAP_REPLICATES, 10000)
        self.assertEqual(summarize_hoi_phase1b.BOOTSTRAP_SEED, 42)

    def test_phase_1b_live_worker_preflight_is_non_overwritable_and_idle_aware(self):
        source = (REPO_ROOT / "tools/capture_hoi_worker_preflight.py").read_text(encoding="utf-8")
        self.assertIn("reportable_workload_started", source)
        self.assertIn("four_gpu_idle", source)
        self.assertIn("forbidden_snapshot_entries", source)
        self.assertIn("refusing to overwrite", source)
        self.assertIn("EXPECTED_CHOIS_CHECKPOINT_SHA256", source)

    def test_multi_server_worker_contract_is_documented(self):
        rules = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
        guide = (REPO_ROOT / "docs" / "MULTI_SERVER_TRAINING.md").read_text(encoding="utf-8")
        handoff = (
            REPO_ROOT / "docs" / "phase_summaries" / "PHASE_1A.md"
        ).read_text(encoding="utf-8")
        self.assertIn("10.184.17.253", rules)
        self.assertIn("10.181.9.214", rules)
        self.assertIn("Never bidirectionally `rsync`", rules)
        self.assertIn("All screening, training, main-table, and evaluation experiments use seed 42", rules)
        self.assertIn("OMOMO-only immutable snapshot", guide)
        self.assertIn("data/dataset", guide)
        self.assertIn("conda-pack", guide)
        self.assertIn("/home/yujinlun/data", guide)
        self.assertIn("worker initiates all server-to-server", rules)
        self.assertIn("id_ed25519_infbagel_8gpu", guide)
        self.assertIn("127.0.0.1:22214", rules)
        self.assertIn("id_ed25519_infbagel_reverse_tunnel", guide)
        self.assertIn("infbagel-reverse-ssh.service", guide)
        self.assertIn("Tkd/zHVWRLW8twkFGpOqhUEm8HmLvvWhu7nOXH+mbhg", guide)
        self.assertIn("GatewayPorts", guide)
        self.assertIn("INFBAGEL_WORKER_EXPERT=hoi", guide)
        self.assertIn("smpl_models", rules)
        self.assertIn("current research HEAD", handoff)
        self.assertIn("4-GPU worker", handoff)

        preflight = json.loads((
            REPO_ROOT / "experiments" / "results" /
            "p1_hoi_worker_preflight_s42_20260713.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(preflight["status"], "passed")
        self.assertFalse(preflight["reportable_experiment"])
        self.assertFalse(preflight["phase_1b_started"])
        self.assertFalse(preflight["capacity_or_batch_decision"])
        self.assertTrue(preflight["data"]["full_rsync_checksum_passed"])
        self.assertFalse(preflight["data"]["scene_supervision"]["dataset_loaded"])
        self.assertEqual(preflight["tests"]["skipped"], 2)
        self.assertTrue(preflight["single_gpu_smoke"]["loss_finite"])
        self.assertEqual(preflight["single_gpu_smoke"]["initialization"], "random")
        self.assertEqual(
            preflight["returned_artifacts"]["sha256"],
            "1afab7ce2383d820ef16f481f4dae7bd18f94d9fb5675d3fb5ca00dab3f56d38",
        )

        remote_control = json.loads((
            REPO_ROOT / "experiments" / "results" /
            "p1_hoi_worker_remote_control_s42_20260713.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(remote_control["status"], "passed")
        self.assertFalse(remote_control["reportable_experiment"])
        self.assertFalse(remote_control["phase_1b_started"])
        self.assertEqual(
            remote_control["connectivity"]["authority_listen_scope"],
            "loopback_only",
        )
        self.assertFalse(remote_control["connectivity"]["gateway_ports_enabled"])
        self.assertTrue(remote_control["persistence"]["enabled"])
        self.assertTrue(remote_control["persistence"]["linger"])
        self.assertTrue(remote_control["remote_execution"]["cuda_available"])
        self.assertEqual(remote_control["worker"]["git_head"],
                         "55cf5f3b0342e8016818ba39e78d62e3f206bc2a")

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


if __name__ == "__main__":
    unittest.main()
