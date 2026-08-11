"""HOIPrior governance tests.

Split out of ``tests/test_research_governance.py`` at commit 0ddc0ef with every
assertion carried over verbatim.  These seven methods read HOI-only files
(``code/train_hoi_prior.py``, ``code/priors/hoi/*``, the
``config_train_hoi_prior*`` / ``config_eval_hoi_prior`` / ``sampler/hoi_prior``
configs, ``tools/audit_hoi_capacity.py``, ``tools/capture_hoi_worker_preflight.py``,
``tools/make_hoi_split.py``, ``tools/summarize_hoi_phase1b.py``), so they cannot
live in the project-wide governance module once ``phase/01c-hsi`` deletes those
paths.  The 16 project-wide methods stay in ``tests/test_research_governance.py``
and run on both expert branches.
"""

import json
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
except ImportError:  # Minimal governance checks run before ML dependencies are installed.
    np = None

from tools import experiment, make_hoi_split, summarize_hoi_phase1b


REPO_ROOT = Path(__file__).resolve().parents[2]


class HOIGovernanceTests(unittest.TestCase):
    def test_hoi_split_generator_reproduces_locked_manifest(self):
        generated = make_hoi_split.build_split(REPO_ROOT)
        tracked = json.loads((
            REPO_ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(generated, tracked)
        experiment.validate_split(
            REPO_ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )

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

    def test_phase_1b_training_and_evaluation_paths_are_scene_free(self):
        train_config = (REPO_ROOT / "code/config/config_train_hoi_prior.yaml").read_text(encoding="utf-8")
        eval_config = (REPO_ROOT / "code/config/config_eval_hoi_prior.yaml").read_text(encoding="utf-8")
        sampler_config = (REPO_ROOT / "code/config/sampler/hoi_prior.yaml").read_text(encoding="utf-8")
        trainer = (REPO_ROOT / "code/train_hoi_prior.py").read_text(encoding="utf-8")
        model = (REPO_ROOT / "code/priors/hoi/models.py").read_text(encoding="utf-8")
        hsi_model = (REPO_ROOT / "code/priors/hsi/models.py").read_text(encoding="utf-8")
        dataset = (REPO_ROOT / "code/priors/hoi/data.py").read_text(encoding="utf-8")
        sampler = (REPO_ROOT / "code/priors/hoi/diffusion.py").read_text(encoding="utf-8")
        evaluator = (REPO_ROOT / "code/test_infbagel_hoi.py").read_text(encoding="utf-8")
        self.assertIn("dim_model: 512", train_config)
        self.assertIn("num_heads: 16", train_config)
        self.assertIn("num_layers: 8", train_config)
        self.assertIn("diffusion_steps: 500", train_config)
        self.assertIn("load_scene: false", eval_config)
        self.assertIn("_target_: priors.hoi.diffusion.HOIPriorSampler", sampler_config)
        # The two experts now live in separate packages, so the old
        # "HOIPrior body = text between class HOIPrior and class HSIPrior"
        # slice no longer exists.  The scene-freedom guarantee is asserted more
        # strongly instead: the whole HOI expert module defines HOIPrior and
        # mentions no scene condition anywhere, and HSIPrior is somewhere else.
        self.assertIn("class HOIPrior(nn.Module):", model)
        self.assertNotIn("class HSIPrior", model)
        self.assertNotIn("scene_condition:", model)
        self.assertNotIn("scene_condition:", model.split("class HOIPrior", 1)[1])
        self.assertIn("class HSIPrior(nn.Module):", hsi_model)
        self.assertIn("scene_condition:", hsi_model)
        self.assertIn("init_checkpoint is forbidden", trainer)
        self.assertIn("resume checkpoint training contract mismatch", trainer)
        self.assertIn("resume checkpoint Git commit mismatch", trainer)
        self.assertIn("max_consecutive_amp_overflows: 16", train_config)
        self.assertIn("amp_overflow_skips_by_rank", trainer)
        self.assertIn("scaler.update(new_scale=float(scaler.get_scale()) * 0.5)", trainer)
        self.assertIn("ema_decays: [0.999, 0.9999]", train_config)
        self.assertIn("object_surface_weight: 50.0", train_config)
        self.assertIn("WindowStateCodec", dataset)
        self.assertIn("terminal_window", dataset)
        self.assertIn("goals[:, :3] = pelvis_goal", sampler)
        self.assertIn("window_codec.encode", evaluator)
        self.assertIn("recompute_rollout_bps", evaluator)
        self.assertIn("current_frame.object_reference", evaluator)

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
        self.assertIn("DISPLAY_ONLY_UTILIZATION_PERCENT_MAX = 1", source)
        self.assertIn("IDLE_MEMORY_USED_MIB_MAX = 128", source)
        self.assertIn("compute_processes_must_be_empty", source)

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


if __name__ == "__main__":
    unittest.main()
