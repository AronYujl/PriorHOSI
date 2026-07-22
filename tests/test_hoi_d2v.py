import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from train_hoi_prior import (
    _build_optimizer,
    _build_scheduler,
    _optimization_contract,
    _validate_d2t_contract,
    _validate_d2u_contract,
    _validate_d2v_contract,
    _validate_d2v_execution_host,
)
from tools.run_hoi_d2v_evaluation import (
    BASELINE_KEYS,
    CONTROL_AGGREGATE_SHA256,
    CONTROL_CHECKPOINT_SHA256,
    CONTROL_PER_SEQUENCE_SHA256,
    RUN_ID as EVAL_RUN_ID,
    classify as classify_evaluation,
    compare_records,
    validate_training_result,
)


def merged_config(name):
    base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
    intervention = OmegaConf.load(ROOT / f"code/config/{name}.yaml")
    cfg = OmegaConf.merge(base, intervention)
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(
        ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    return cfg


def d2v_config():
    return merged_config("config_train_hoi_prior_d2v")


class D2VLongBudgetTests(unittest.TestCase):
    def test_exact_config_changes_only_budget_and_mode_from_d2u(self):
        d2v = d2v_config()
        d2u = merged_config("config_train_hoi_prior_d2u")
        _validate_d2v_contract(d2v, 4)
        _validate_d2u_contract(d2u, 4)
        ignored = {
            "mode", "subphase", "run_id", "d2u_balanced_author_update",
            "d2v_balanced_long_budget", "max_processed_windows",
        }
        d2v_plain = OmegaConf.to_container(d2v, resolve=False)
        d2u_plain = OmegaConf.to_container(d2u, resolve=False)
        self.assertEqual(
            {key: value for key, value in d2v_plain.items() if key not in ignored},
            {key: value for key, value in d2u_plain.items() if key not in ignored},
        )
        self.assertEqual(d2v.max_processed_windows, 61440000)
        self.assertEqual(d2v.max_processed_windows // d2v.effective_batch_size, 30000)

    def test_author_update_optimizer_contract_is_identical_to_d2u(self):
        cfg = d2v_config()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = _build_optimizer(cfg, [parameter])
        scheduler = _build_scheduler(cfg, optimizer, 30000, 0)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertIsNone(scheduler)
        self.assertEqual(
            _optimization_contract(cfg),
            {
                "optimizer": "Adam", "betas": [0.9, 0.999],
                "weight_decay": 0.0, "learning_rate": 0.0001,
                "scheduler": "none", "warmup_windows": 0,
                "gradient_clipping": False, "gradient_clip_norm": None,
                "amp": False, "ema_decays": [], "primary_weight_variant": "online",
            },
        )

    def test_contract_fails_closed_on_any_non_budget_contract_mutation(self):
        mutations = {
            "max_processed_windows": 6144000,
            "fk_weight": 0.4,
            "learning_rate": 0.0003,
            "checkpoint_interval_windows": 6144000,
            "resume_checkpoint": "/tmp/forbidden.pth",
            "weight_init_checkpoint": "/tmp/d2u.pth",
            "d2m_candidate": "balanced",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                cfg = d2v_config()
                setattr(cfg, field, value)
                with self.assertRaisesRegex(ValueError, "D2-V"):
                    _validate_d2v_contract(cfg, 4)

    def test_modes_remain_mutually_exclusive(self):
        d2t = merged_config("config_train_hoi_prior_d2t")
        d2t.d2v_balanced_long_budget = True
        with self.assertRaisesRegex(ValueError, "d2v_mode_off"):
            _validate_d2t_contract(d2t, 4)
        d2u = merged_config("config_train_hoi_prior_d2u")
        d2u.d2v_balanced_long_budget = True
        with self.assertRaisesRegex(ValueError, "d2v_mode_off"):
            _validate_d2u_contract(d2u, 4)

    def test_worker_host_and_python_are_fail_closed(self):
        cfg = d2v_config()
        with mock.patch("socket.gethostname", return_value="ubuntu"):
            with self.assertRaisesRegex(RuntimeError, "infbagel-4gpu/node01"):
                _validate_d2v_execution_host(cfg)
        environment = {
            "INFBAGEL_WORKER_EXPERT": "hoi",
            "INFBAGEL_PYTHON": "/home/yujinlun/data/envs/infbagel/bin/python",
        }
        with mock.patch("socket.gethostname", return_value="node01"), mock.patch.dict(
            os.environ, environment, clear=False,
        ), mock.patch("train_hoi_prior.sys.executable", environment["INFBAGEL_PYTHON"]):
            _validate_d2v_execution_host(cfg)


class D2VEvaluationAndGovernanceTests(unittest.TestCase):
    @staticmethod
    def _native_records(offset=0.0, contact=0.7, foot=0.3):
        return {
            f"sequence-{index:03d}": {
                "mpjpe": 10.0 + offset,
                "end_obj_trans_err": 4.0 + offset,
                "pelvis_goal_error_cm": 3.0 + offset,
                "obj_trans_dist": 12.0 + offset,
                "foot_sliding": foot,
                "contact_f1": contact,
            }
            for index in range(32)
        }

    def test_native_gate_and_classifications_match_preregistration(self):
        d2u = self._native_records(offset=1.0, contact=0.70, foot=0.30)
        d2v = self._native_records(offset=0.0, contact=0.69, foot=0.30)
        target_metrics = {key: 1.0 for key in BASELINE_KEYS}
        target_metrics["contact_f1"] = 0.69
        ratios = {
            "mpjpe": 1.0, "end_obj_trans_err": 1.0,
            "xy_points_err": 1.0, "obj_trans_dist": 1.0,
            "foot_sliding": 1.0,
        }
        result = classify_evaluation(
            compare_records(d2u, d2v), target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(result["classification"], "effective-diffusion-hoi-prior-stop")
        self.assertTrue(result["mechanism_passed"])
        result = classify_evaluation(
            compare_records(d2u, self._native_records(offset=2.0)),
            target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(result["classification"], "long-budget-negative-stop")
        ratios["mpjpe"] = 1.31
        result = classify_evaluation(
            compare_records(d2u, d2v), target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(
            result["classification"], "long-budget-positive-but-not-effective-stop",
        )

    def test_training_metrics_bind_exact_final_random_checkpoint(self):
        target_sha = "a" * 64
        checkpoint = Path(
            "/tmp/p1-hoi-d2v-balanced-long-budget-s42-20260722_windows061440000.pth"
        )
        metrics = {
            "status": "stable",
            "run_id": "p1-hoi-d2v-balanced-long-budget-s42-20260722",
            "seed": 42,
            "initialization": "random",
            "training_start": "random",
            "released_checkpoint_used": False,
            "processed_windows": 61440000,
            "optimizer_updates": 30000,
            "world_size": 4,
            "effective_batch_size": 2048,
            "optimization_contract": {
                "optimizer": "Adam", "betas": [0.9, 0.999],
                "weight_decay": 0.0, "learning_rate": 0.0001,
                "scheduler": "none", "warmup_windows": 0,
                "gradient_clipping": False, "gradient_clip_norm": None,
                "amp": False, "ema_decays": [], "primary_weight_variant": "online",
            },
            "loss_weights": {
                "fk": 0.3569973401779424,
                "object_surface": 0.4772322188400037,
                "velocity": 0.1,
                "terminal_object_goal": 1.0,
            },
            "ema_decays": [],
            "primary_weight_variant": "online",
            "weight_initialization": {
                "mode": "random", "source_checkpoint": None,
                "restored_components": [],
            },
            "checkpoint_hashes": [{
                "path": str(checkpoint), "sha256": target_sha,
                "processed_windows": 61440000,
            }],
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.json"
            path.write_text(json.dumps(metrics), encoding="utf-8")
            args = SimpleNamespace(
                target_checkpoint=checkpoint,
                target_sha256=target_sha,
                training_metrics=path,
            )
            result = validate_training_result(args)
            self.assertTrue(all(result["checks"].values()))
            metrics["processed_windows"] = 6144000
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "processed_windows"):
                validate_training_result(args)

    def test_registry_plan_and_control_hashes_are_locked(self):
        self.assertEqual(EVAL_RUN_ID, "p1-hoi-d2v-native-eval-s42-20260722")
        self.assertEqual(
            CONTROL_CHECKPOINT_SHA256,
            "7cb379263f8a72e7f9017e4ada9d521a9e25f7c160c061305a92b9822bda2cad",
        )
        self.assertEqual(
            CONTROL_AGGREGATE_SHA256,
            "1f898f0f8127dbf7f93b0cc376e575e89e05e11212a6a846dc821dee85f9e5ba",
        )
        self.assertEqual(
            CONTROL_PER_SEQUENCE_SHA256,
            "5cf7a88923f4d4c0daa2b3c5e7adac46920cbd51d820dcc6e8003139feeb654b",
        )
        records = [
            json.loads(line) for line in
            (ROOT / "experiments/registry.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        training = next(item for item in records if item["experiment_id"] ==
                        "p1-hoi-d2v-balanced-long-budget-preregister-s42-20260722")
        evaluation = next(item for item in records if item["experiment_id"] ==
                          "p1-hoi-d2v-native-eval-preregister-s42-20260722")
        self.assertEqual(training["config"]["manipulated_factor"]["ratio"], 10)
        self.assertFalse(training["config"]["manipulated_factor"]["resume_control"])
        self.assertTrue(training["config"]["fixed"]["random_initialization"])
        self.assertFalse(training["config"]["consistency_authorized"])
        self.assertTrue(evaluation["config"]["control"]["reused_without_regeneration"])
        self.assertFalse(evaluation["config"]["consistency_authorized"])
        completed_training = next(item for item in records if item["experiment_id"] ==
                                  "p1-hoi-d2v-balanced-long-budget-s42-20260722")
        completed_evaluation = next(item for item in records if item["experiment_id"] ==
                                    "p1-hoi-d2v-native-eval-s42-20260722")
        self.assertEqual(completed_training["results"]["final_checkpoint_sha256"],
                         "e0705681bbaeed40d353494852494d8b7bdaf4d32da92368c0d2ceedea4c01a4")
        self.assertTrue(completed_training["config"]["d2u_initial_model_state_identical"])
        self.assertEqual(completed_evaluation["results"]["classification"],
                         "long-budget-negative-stop")
        self.assertFalse(completed_evaluation["results"]["mechanism_passed"])
        self.assertFalse(completed_evaluation["results"]["effective_diffusion_passed"])
        self.assertFalse(completed_evaluation["results"]["checkpoint_selected"])
        self.assertGreater(completed_evaluation["results"]["target_metrics"]["contact_f1"], 0.60)
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-V from-random balanced long-budget screen", plan)
        self.assertIn("p1-hoi-d2v-native-eval-s42-20260722", plan)
        self.assertIn("本 subphase 不授权 CM", plan)
        self.assertIn("D2-V balanced long-budget completion", plan)
        compact = json.loads((
            ROOT / "experiments/results/"
            "p1_hoi_phase1b_d2v_balanced_long_budget_s42_20260722.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(compact["classification"], "long-budget-negative-stop")
        self.assertFalse(compact["checkpoint_selected"])
        self.assertFalse(compact["consistency_authorized"])
        self.assertTrue(compact["training"]["d2u_initial_model_state_identical"])
        self.assertEqual(compact["training"]["processed_windows"], 61440000)
        self.assertEqual(compact["evaluation"]["target_metrics"]["contact_f1"],
                         0.628590477397954)
        self.assertFalse(compact["evaluation"]["mechanism_checks"]["foot_sliding_preserved"])


if __name__ == "__main__":
    unittest.main()
