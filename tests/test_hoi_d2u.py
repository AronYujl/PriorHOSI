import hashlib
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
    _validate_d2u_execution_host,
)
from tools.run_hoi_d2u_evaluation import (
    BASELINE_KEYS,
    CONTROL_AGGREGATE_SHA256,
    CONTROL_CHECKPOINT_SHA256,
    CONTROL_PER_SEQUENCE_SHA256,
    RUN_ID as EVAL_RUN_ID,
    classify as classify_evaluation,
    compare_records,
    validate_training_result,
)


EXPECTED_FIXED_SOURCE_SHA256 = {
    "code/priors/representation.py": "a510b4ddfb4f6b60e3917219a87898a717d91cbfb858d0993f3e054b5a1abf74",
    "code/priors/window_codec.py": "74ed335330425bbc0941d99f9f816c8b81d0eebbc59d2d680770f964a3312b53",
    "code/priors/data.py": "62132421b973b1d77c273f80ce48b81507966c0fe75563acd8c1e2158cb54cc5",
    # D2-AC/D2-AD/D2-AE/D2-AF extend the shared model module behind explicit
    # architecture variants; targeted tests lock exact base-path parity.
    "code/priors/models.py": "f7d464e48629a5e6420ea6a21f1ff8130980223cb6f59944a6226a83a952dd12",
    "code/priors/losses.py": "e14cee19e59e9ac698d4d412ccd388f9d0bf903f22e6774b13cc736087d9d1be",
    "code/priors/diffusion.py": "fd8d05c34689cf4697920097bd330e6a25e3424c7460eb3a4e7ef12f45ed17a2",
}


def merged_config(name):
    base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
    intervention = OmegaConf.load(ROOT / f"code/config/{name}.yaml")
    cfg = OmegaConf.merge(base, intervention)
    cfg.repo_root = str(ROOT)
    cfg.split_manifest = str(
        ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
    )
    return cfg


def d2u_config():
    return merged_config("config_train_hoi_prior_d2u")


class D2UBalancedObjectiveTests(unittest.TestCase):
    def test_exact_config_changes_only_locked_loss_geometry(self):
        d2u = d2u_config()
        d2t = merged_config("config_train_hoi_prior_d2t")
        _validate_d2u_contract(d2u, 4)
        _validate_d2t_contract(d2t, 4)
        ignored = {
            "mode", "subphase", "run_id", "d2t_author_update_rule",
            "d2u_balanced_author_update", "fk_weight", "object_surface_weight",
        }
        d2u_plain = OmegaConf.to_container(d2u, resolve=False)
        d2t_plain = OmegaConf.to_container(d2t, resolve=False)
        self.assertEqual(
            {key: value for key, value in d2u_plain.items() if key not in ignored},
            {key: value for key, value in d2t_plain.items() if key not in ignored},
        )
        self.assertEqual(d2u.fk_weight, 0.3569973401779424)
        self.assertEqual(d2u.object_surface_weight, 0.4772322188400037)

    def test_author_update_optimizer_contract_is_identical_to_d2t(self):
        cfg = d2u_config()
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = _build_optimizer(cfg, [parameter])
        scheduler = _build_scheduler(cfg, optimizer, 3000, 0)
        self.assertIs(type(optimizer), torch.optim.Adam)
        self.assertIsNone(scheduler)
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

    def test_contract_fails_closed_on_loss_or_checkpoint_mutation(self):
        mutations = {
            "fk_weight": 0.4,
            "object_surface_weight": 0.5,
            "velocity_weight": 0.2,
            "goal_weight": 2.0,
            "learning_rate": 0.0003,
            "resume_checkpoint": "/tmp/forbidden.pth",
            "weight_init_checkpoint": "/tmp/balanced.pth",
            "d2m_candidate": "balanced",
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                cfg = d2u_config()
                setattr(cfg, field, value)
                with self.assertRaisesRegex(ValueError, "D2-U"):
                    _validate_d2u_contract(cfg, 4)

    def test_d2t_mode_remains_exact_and_mutually_exclusive(self):
        cfg = merged_config("config_train_hoi_prior_d2t")
        _validate_d2t_contract(cfg, 4)
        cfg.d2u_balanced_author_update = True
        with self.assertRaisesRegex(ValueError, "d2u_mode_off"):
            _validate_d2t_contract(cfg, 4)

    def test_worker_host_and_python_are_fail_closed(self):
        cfg = d2u_config()
        with mock.patch("socket.gethostname", return_value="ubuntu"):
            with self.assertRaisesRegex(RuntimeError, "infbagel-4gpu/node01"):
                _validate_d2u_execution_host(cfg)
        environment = {
            "INFBAGEL_WORKER_EXPERT": "hoi",
            "INFBAGEL_PYTHON": "/home/yujinlun/data/envs/infbagel/bin/python",
        }
        with mock.patch("socket.gethostname", return_value="node01"), mock.patch.dict(
            os.environ, environment, clear=False,
        ), mock.patch("train_hoi_prior.sys.executable", environment["INFBAGEL_PYTHON"]):
            _validate_d2u_execution_host(cfg)


class D2UEvaluationAndGovernanceTests(unittest.TestCase):
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
        d2t = self._native_records(offset=1.0, contact=0.70, foot=0.30)
        d2u = self._native_records(offset=0.0, contact=0.69, foot=0.30)
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
            compare_records(d2t, d2u), target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(result["classification"], "effective-diffusion-hoi-prior-stop")
        self.assertTrue(result["mechanism_passed"])
        result = classify_evaluation(
            compare_records(d2t, self._native_records(offset=2.0)),
            target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(result["classification"], "balanced-objective-negative-stop")
        ratios["mpjpe"] = 1.31
        result = classify_evaluation(
            compare_records(d2t, d2u), target_metrics, ratios, contract_passed=True,
        )
        self.assertEqual(
            result["classification"], "balanced-objective-positive-but-not-effective-stop",
        )

    def test_training_metrics_bind_the_exact_final_random_checkpoint(self):
        target_sha = "a" * 64
        checkpoint = Path(
            "/tmp/p1-hoi-d2u-balanced-author-update-s42-20260721_windows006144000.pth"
        )
        metrics = {
            "status": "stable",
            "run_id": "p1-hoi-d2u-balanced-author-update-s42-20260721",
            "seed": 42,
            "initialization": "random",
            "training_start": "random",
            "released_checkpoint_used": False,
            "processed_windows": 6144000,
            "optimizer_updates": 3000,
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
                "processed_windows": 6144000,
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
            metrics["loss_weights"]["fk"] = 50.0
            path.write_text(json.dumps(metrics), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "loss_weights"):
                validate_training_result(args)

    def test_registry_plan_and_control_hashes_are_locked(self):
        self.assertEqual(EVAL_RUN_ID, "p1-hoi-d2u-native-eval-s42-20260721")
        self.assertEqual(
            CONTROL_CHECKPOINT_SHA256,
            "1543af304acf76f385dbd3656a1ca82ea25dcd504ee120f7f63e821d71483647",
        )
        self.assertEqual(
            CONTROL_AGGREGATE_SHA256,
            "8862cfdc013482bc4fb3810bfbcaf3131010b3ee9bee13324dccd27df85e2702",
        )
        self.assertEqual(
            CONTROL_PER_SEQUENCE_SHA256,
            "422dd0e8bc87cc896186ebdaea9fcd22868ae4ba40d2d61d5bb16a1000500f9e",
        )
        records = [
            json.loads(line) for line in
            (ROOT / "experiments/registry.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        training = next(item for item in records if item["experiment_id"] ==
                        "p1-hoi-d2u-balanced-author-update-preregister-s42-20260721")
        evaluation = next(item for item in records if item["experiment_id"] ==
                          "p1-hoi-d2u-native-eval-preregister-s42-20260721")
        self.assertEqual(training["config"]["manipulated_factor"]["intervention"], {
            "fk": 0.3569973401779424,
            "object_surface": 0.4772322188400037,
            "velocity": 0.1,
            "terminal_goal": 1.0,
        })
        self.assertTrue(training["config"]["fixed"]["random_initialization"])
        self.assertFalse(training["config"]["consistency_authorized"])
        self.assertTrue(evaluation["config"]["control"]["reused_without_regeneration"])
        self.assertFalse(evaluation["config"]["consistency_authorized"])
        completed = next(item for item in records if item["experiment_id"] ==
                         "p1-hoi-d2u-native-eval-s42-20260721")
        self.assertEqual(
            completed["results"]["classification"],
            "balanced-objective-negative-stop",
        )
        self.assertFalse(completed["results"]["mechanism_passed"])
        self.assertFalse(completed["results"]["effective_diffusion_passed"])
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn("D2-U from-random balanced-objective screen", plan)
        self.assertIn("p1-hoi-d2u-native-eval-s42-20260721", plan)
        self.assertIn("自动延长预算", plan)
        compact = json.loads((
            ROOT / "experiments/results/"
            "p1_hoi_phase1b_d2u_balanced_author_update_s42_20260721.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(compact["classification"], "balanced-objective-negative-stop")
        self.assertFalse(compact["checkpoint_selected"])
        self.assertFalse(compact["consistency_authorized"])
        self.assertEqual(
            compact["training"]["final_checkpoint_sha256"],
            "7cb379263f8a72e7f9017e4ada9d521a9e25f7c160c061305a92b9822bda2cad",
        )

    def test_model_data_loss_and_diffusion_sources_are_unchanged(self):
        for relative, expected in EXPECTED_FIXED_SOURCE_SHA256.items():
            actual = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
            self.assertEqual(actual, expected, relative)


if __name__ == "__main__":
    unittest.main()
