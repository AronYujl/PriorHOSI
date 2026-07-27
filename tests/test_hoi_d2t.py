import hashlib
import json
import os
import sys
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
    _checkpoint_value,
    _gradient_l2_norm,
    _optimization_contract,
    _primary_validation_model,
    _validate_d2t_contract,
    _validate_d2t_execution_host,
)
from tools.capture_hoi_worker_preflight import four_gpu_evaluation_idle, four_gpu_idle
from tools.run_hoi_d2t_evaluation import (
    BASELINE_KEYS,
    CONTROL_AGGREGATE_SHA256,
    CONTROL_PER_SEQUENCE_SHA256,
    RUN_ID as EVAL_RUN_ID,
    TARGET_SHA256,
    classify as classify_evaluation,
    compare_records,
)


EXPECTED_FIXED_SOURCE_SHA256 = {
    "code/priors/representation.py": "a510b4ddfb4f6b60e3917219a87898a717d91cbfb858d0993f3e054b5a1abf74",
    "code/priors/window_codec.py": "74ed335330425bbc0941d99f9f816c8b81d0eebbc59d2d680770f964a3312b53",
    "code/priors/data.py": "62132421b973b1d77c273f80ce48b81507966c0fe75563acd8c1e2158cb54cc5",
    # D2-AC/D2-AD extend the shared model module behind explicit architecture
    # variants; targeted tests lock exact base-path and parameter parity.
    "code/priors/models.py": "0c783d50e023ae43ff42e2bebf79c756c24589baf98dfc0a9de353906fb38559",
    "code/priors/losses.py": "e14cee19e59e9ac698d4d412ccd388f9d0bf903f22e6774b13cc736087d9d1be",
    "code/priors/diffusion.py": "e264ca65aacaf944e447ec41c56c23105888f87c35bc45490ef9a1a3f2006406",
}


def d2t_config():
    base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
    intervention = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2t.yaml")
    cfg = OmegaConf.merge(base, intervention)
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(
        ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    return cfg


class D2TUpdateRuleTests(unittest.TestCase):
    def test_exact_config_and_optimizer_contract(self):
        cfg = d2t_config()
        _validate_d2t_contract(cfg, 4)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = _build_optimizer(cfg, [parameter])
        scheduler = _build_scheduler(cfg, optimizer, 3000, 0)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertIsNone(scheduler)
        self.assertEqual(optimizer.defaults["lr"], 0.0001)
        self.assertEqual(optimizer.defaults["betas"], (0.9, 0.999))
        self.assertEqual(optimizer.defaults["weight_decay"], 0.0)
        self.assertEqual(cfg.max_processed_windows // cfg.effective_batch_size, 3000)
        self.assertEqual(
            _optimization_contract(cfg),
            {
                "optimizer": "Adam",
                "betas": [0.9, 0.999],
                "weight_decay": 0.0,
                "learning_rate": 0.0001,
                "scheduler": "none",
                "warmup_windows": 0,
                "gradient_clipping": False,
                "gradient_clip_norm": None,
                "amp": False,
                "ema_decays": [],
                "primary_weight_variant": "online",
            },
        )

    def test_contract_fails_closed_on_single_field_change(self):
        cfg = d2t_config()
        cfg.learning_rate = 0.0003
        with self.assertRaisesRegex(ValueError, "learning_rate"):
            _validate_d2t_contract(cfg, 4)
        cfg = d2t_config()
        cfg.resume_checkpoint = "/tmp/forbidden.pth"
        with self.assertRaisesRegex(ValueError, "random_initialization"):
            _validate_d2t_contract(cfg, 4)

    def test_unclipped_gradient_norm_is_observational(self):
        first = torch.nn.Parameter(torch.tensor([3.0]))
        second = torch.nn.Parameter(torch.tensor([4.0]))
        first.grad = torch.tensor([3.0])
        second.grad = torch.tensor([4.0])
        before = (first.grad.clone(), second.grad.clone())
        self.assertEqual(float(_gradient_l2_norm([first, second])), 5.0)
        torch.testing.assert_close(first.grad, before[0])
        torch.testing.assert_close(second.grad, before[1])

    def test_validation_and_checkpoint_use_online_only(self):
        cfg = d2t_config()
        module = torch.nn.Linear(2, 2)
        wrapper = SimpleNamespace(module=module)
        self.assertIs(_primary_validation_model(cfg, wrapper, {}), module)
        optimizer = _build_optimizer(cfg, module.parameters())
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        value = _checkpoint_value(
            cfg,
            wrapper,
            {},
            optimizer,
            None,
            scaler,
            world_size=4,
            processed_windows=6144000,
            optimizer_updates=3000,
            amp_overflow_skips=0,
            epoch=10,
            batches_consumed_in_epoch=20,
            rng_pattern="checkpoint.rank{rank}.rng.pth",
            weight_initialization={"mode": "random", "restored_components": []},
        )
        self.assertEqual(value["ema_models"], {})
        self.assertNotIn("ema_model", value)
        self.assertIsNone(value["scheduler"])
        self.assertIsNone(value["scaler"])
        self.assertEqual(value["primary_weight_variant"], "online")
        self.assertEqual(value["optimization_contract"]["optimizer"], "Adam")

    def test_worker_host_and_environment_are_fail_closed(self):
        cfg = d2t_config()
        with mock.patch("socket.gethostname", return_value="ubuntu"):
            with self.assertRaisesRegex(RuntimeError, "infbagel-4gpu/node01"):
                _validate_d2t_execution_host(cfg)
        environment = {
            "INFBAGEL_WORKER_EXPERT": "hoi",
            "INFBAGEL_PYTHON": "/home/yujinlun/data/envs/infbagel/bin/python",
        }
        with mock.patch("socket.gethostname", return_value="node01"), mock.patch.dict(
            os.environ, environment, clear=False,
        ), mock.patch("train_hoi_prior.sys.executable", environment["INFBAGEL_PYTHON"]):
            _validate_d2t_execution_host(cfg)


class D2TScientificAndGovernanceTests(unittest.TestCase):
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

    def test_native_evaluation_gate_matches_preregistration(self):
        control = self._native_records(offset=1.0, contact=0.70, foot=0.30)
        target = self._native_records(offset=0.0, contact=0.69, foot=0.30)
        comparison = compare_records(control, target)
        target_metrics = {key: 1.0 for key in BASELINE_KEYS}
        target_metrics["contact_f1"] = 0.69
        ratios = {
            "mpjpe": 1.0,
            "end_obj_trans_err": 1.0,
            "xy_points_err": 1.0,
            "obj_trans_dist": 1.0,
            "foot_sliding": 1.0,
        }
        result = classify_evaluation(
            comparison, target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(result["classification"], "effective-diffusion-hoi-prior-stop")
        self.assertTrue(result["mechanism_passed"])
        target = self._native_records(offset=0.0, contact=0.60, foot=0.30)
        result = classify_evaluation(
            compare_records(control, target), target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(result["classification"], "author-update-rule-negative-stop")

    def test_native_evaluation_registry_and_hashes_are_locked(self):
        self.assertEqual(EVAL_RUN_ID, "p1-hoi-d2t-native-eval-s42-20260721")
        self.assertEqual(TARGET_SHA256, "1543af304acf76f385dbd3656a1ca82ea25dcd504ee120f7f63e821d71483647")
        self.assertEqual(CONTROL_AGGREGATE_SHA256, "d95d3090455e763159a4cac793301f9f4744837bf60b4ed21eaef4a141c9ad2b")
        self.assertEqual(CONTROL_PER_SEQUENCE_SHA256, "11c11fcd90c0ce2e67d705bb64c3a78bbe2b0e9f84aff7fcb57cab25087e2a1f")
        records = [json.loads(line) for line in (ROOT / "experiments/registry.jsonl").read_text(encoding="utf-8").splitlines()]
        record = next(item for item in records if item["experiment_id"] == "p1-hoi-d2t-native-eval-preregister-s42-20260721")
        self.assertEqual(record["config"]["run_id"], EVAL_RUN_ID)
        self.assertTrue(record["config"]["control"]["reused_without_regeneration"])
        self.assertFalse(record["config"]["consistency_authorized"])

    def test_display_only_idle_tolerance_is_tight(self):
        gpus = [
            {
                "memory_used_mib": 100,
                "utilization_percent": 1 if index == 0 else 0,
                "pstate": "P8",
            }
            for index in range(4)
        ]
        self.assertTrue(four_gpu_idle(gpus, []))
        with_process = ["GPU-0, 123, python, 1"]
        self.assertFalse(four_gpu_idle(gpus, with_process))
        gpus[0]["utilization_percent"] = 2
        self.assertFalse(four_gpu_idle(gpus, []))
        gpus[0]["utilization_percent"] = 1
        gpus[0]["memory_used_mib"] = 129
        self.assertFalse(four_gpu_idle(gpus, []))
        gpus[0]["memory_used_mib"] = 100
        gpus[0]["pstate"] = "P2"
        self.assertFalse(four_gpu_idle(gpus, []))

    def test_evaluation_only_idle_ignores_gpu0_display_utilization(self):
        gpus = [
            {
                "index": index,
                "memory_used_mib": 100 if index == 0 else 15,
                "utilization_percent": 10 if index == 0 else 0,
                "pstate": "P8",
            }
            for index in range(4)
        ]
        self.assertTrue(four_gpu_evaluation_idle(gpus, []))
        self.assertFalse(four_gpu_evaluation_idle(gpus, ["GPU-0, 123, python, 1"]))
        gpus[0]["memory_used_mib"] = 129
        self.assertFalse(four_gpu_evaluation_idle(gpus, []))
        gpus[0]["memory_used_mib"] = 100
        gpus[1]["utilization_percent"] = 2
        self.assertFalse(four_gpu_evaluation_idle(gpus, []))

    def test_model_data_diffusion_and_loss_sources_are_unchanged(self):
        for relative, expected in EXPECTED_FIXED_SOURCE_SHA256.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)

    def test_registry_locks_worker_lifecycle_and_no_checkpoint_load(self):
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
        ]
        record = next(
            item for item in records
            if item["experiment_id"]
            == "p1-hoi-d2t-author-update-rule-preregister-s42-20260721"
        )
        config = record["config"]
        self.assertEqual(config["execution_host"], "infbagel-4gpu")
        self.assertEqual(config["worker_address"], "10.181.9.214")
        self.assertTrue(config["authority_hoi_cuda_forbidden"])
        self.assertEqual(config["manipulated_factor"]["effective_batch_size"], 2048)
        self.assertEqual(config["fixed"]["optimizer_updates"], 3000)
        self.assertEqual(config["forbidden_load"], [
            "released_checkpoint",
            "author_diffusion_checkpoint",
            "author_consistency_checkpoint",
            "prior_checkpoint",
            "resume_checkpoint",
            "ema_model",
            "ema_models",
        ])
        self.assertFalse(config["consistency_authorized"])

    def test_plan_names_exact_lifecycle_and_stop_contract(self):
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-T author-DDPM update-rule parity screen", plan)
        self.assertIn("tools/experiment.py start", plan)
        self.assertIn("10.184.17.253", plan)
        self.assertIn("明确禁止运行任何 HOIPrior CUDA workload", plan)
        self.assertIn("只有 effective-diffusion gate 通过后", plan)


if __name__ == "__main__":
    unittest.main()
