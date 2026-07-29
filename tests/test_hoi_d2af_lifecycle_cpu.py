from __future__ import annotations

import ast
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SHA256,
    diffusion_schedule_contract_metadata,
)
from priors.sparse_relation import (  # noqa: E402
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
)
from tools.diagnose_hoi_d2af import (  # noqa: E402
    archive_or_validate_resolved_config as archive_cpu_config,
    config_contract,
    resolved_config,
    validate_actual_run_id as validate_cpu_run_id,
)
from tools.run_hoi_d2af_eligibility import (  # noqa: E402
    EXPECTED_GLOBAL_INDICES_SHA256,
    EXPECTED_SEQUENCE_NAMES_SHA256,
    _validated_prerequisite,
    archive_or_validate_resolved_config as archive_eligibility_config,
    mutable_anchor_corruption,
    newline_sha256,
    paired_bootstrap,
    resolved_workload_config,
    selection_contract,
    validate_actual_run_id as validate_eligibility_run_id,
)
from train_hoi_prior import (  # noqa: E402
    _d2af_formal_source_contract,
    _validate_d2af_contract,
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class D2AFLifecycleRunIdTests(unittest.TestCase):
    def test_cpu_and_eligibility_ids_require_actual_date(self):
        date = datetime.now().astimezone().strftime("%Y%m%d")
        self.assertEqual(
            validate_cpu_run_id(
                f"p1-hoi-d2af-cpu-contract-s42-{date}"
            ),
            date,
        )
        self.assertEqual(
            validate_cpu_run_id(
                f"p1-hoi-d2af-cpu-contract-r2-s42-{date}"
            ),
            date,
        )
        self.assertEqual(
            validate_eligibility_run_id(
                "p1-hoi-d2af-clean-signal-eligibility-"
                f"r1-s42-{date}"
            ),
            date,
        )
        with self.assertRaises(ValueError):
            validate_cpu_run_id(
                "p1-hoi-d2af-cpu-contract-s42-20260728"
            )
        with self.assertRaises(ValueError):
            validate_eligibility_run_id(
                "p1-hoi-d2af-clean-signal-eligibility-s42-20260728"
            )

    def test_resolve_only_archive_then_actual_byte_equality(self):
        for helper in (archive_cpu_config, archive_eligibility_config):
            with self.subTest(helper=helper.__module__):
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "resolved.yaml"
                    expected = helper(
                        path,
                        "schema_version: 1\n",
                        resolve_only=True,
                    )
                    self.assertEqual(expected, sha256_file(path))
                    self.assertEqual(
                        helper(
                            path,
                            "schema_version: 1\n",
                            resolve_only=False,
                        ),
                        expected,
                    )
                    with self.assertRaises(RuntimeError):
                        helper(
                            path,
                            "schema_version: 2\n",
                            resolve_only=False,
                        )
                    with self.assertRaises(FileExistsError):
                        helper(
                            path,
                            "schema_version: 1\n",
                            resolve_only=True,
                        )


class D2AFEligibilityPureFunctionTests(unittest.TestCase):
    def test_selection_hashes_are_the_preregistered_newline_contract(self):
        dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest=str(
                ROOT
                / "experiments/splits/"
                "omomo_hoi_train_validation_seed42.json"
            ),
        )
        contract = selection_contract(dataset)
        self.assertEqual(
            contract["global_indices_sha256"],
            EXPECTED_GLOBAL_INDICES_SHA256,
        )
        self.assertEqual(
            contract["sequence_names_sha256"],
            EXPECTED_SEQUENCE_NAMES_SHA256,
        )
        self.assertEqual(contract["windows"], 29382)
        self.assertEqual(contract["sequences"], 216)

    def test_newline_hash_is_not_json_or_raw_integer_bytes(self):
        values = [1, 20, 300]
        expected = hashlib.sha256(b"1\n20\n300\n").hexdigest()
        self.assertEqual(newline_sha256(values), expected)
        self.assertNotEqual(
            newline_sha256(values),
            hashlib.sha256(json.dumps(values).encode("utf-8")).hexdigest(),
        )
        self.assertNotEqual(
            newline_sha256(values),
            hashlib.sha256(
                np.asarray(values, dtype=np.int64).tobytes()
            ).hexdigest(),
        )

    def test_mutable_corruption_excludes_anchor_zero(self):
        clean = torch.zeros(2, 4, 3, 100, 4)
        noisy = clean.clone()
        noisy[:, 0] = 1000.0
        self.assertTrue(torch.equal(
            mutable_anchor_corruption(noisy, clean),
            torch.zeros(2),
        ))
        noisy[:, 1:] = 2.0
        self.assertTrue(torch.equal(
            mutable_anchor_corruption(noisy, clean),
            torch.full((2,), 2.0),
        ))

    def test_paired_bootstrap_is_sequence_unit_and_deterministic(self):
        values = np.linspace(0.1, 1.0, 216, dtype=np.float64)
        first = paired_bootstrap(values)
        second = paired_bootstrap(values)
        self.assertEqual(first, second)
        self.assertEqual(first["sequence_count"], 216)
        self.assertEqual(first["bootstrap_replicates"], 10000)
        self.assertTrue(first["ci_lower_gt_zero"])


class D2AFPrerequisiteAndSchemaTests(unittest.TestCase):
    def setUp(self):
        self.commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
        ).strip()
        self.source = _d2af_formal_source_contract(ROOT)

    def _summary(
        self,
        *,
        run_id: str,
        status: str,
        classification: str,
        smoke: bool,
        resolved_path: Path,
    ):
        value = {
            "schema_version": 1,
            "status": status,
            "classification": classification,
            "run_id": run_id,
            "identity": {
                "git_commit": self.commit,
                "worktree_clean": True,
            },
            "formal_source_contract": self.source,
            "resolved_config_path": str(resolved_path.resolve()),
            "resolved_config_sha256": sha256_file(resolved_path),
            "optimizer_created": False,
            "optimizer_updates": 0,
            "checkpoint_loads": 0,
            "scientific_checkpoint_loads": 0,
            "checkpoint_writes": 0,
        }
        if smoke:
            value.update({
                "initialization": "random",
                "initial_model_state_sha256":
                    "b549358a847205ca7cf6376fd5125a60f"
                    "87295c455a95fb72d245a4249b7bc8c",
                "schedule": diffusion_schedule_contract_metadata(),
            })
        return value

    def test_prerequisite_binding_is_absolute_hashed_and_source_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            resolved = directory / "resolved.yaml"
            resolved.write_text("schema_version: 1\n", encoding="utf-8")
            cpu_path = directory / "cpu.json"
            smoke_path = directory / "smoke.json"
            cpu_path.write_text(json.dumps(self._summary(
                run_id="p1-hoi-d2af-cpu-contract-s42-20260729",
                status="passed",
                classification="cpu-contract-passed",
                smoke=False,
                resolved_path=resolved,
            )), encoding="utf-8")
            smoke_path.write_text(json.dumps(self._summary(
                run_id=(
                    "p1-hoi-d2af-gpu-functional-smoke-"
                    "s42-20260729"
                ),
                status="stable",
                classification="functional-smoke-passed",
                smoke=True,
                resolved_path=resolved,
            )), encoding="utf-8")
            cpu = _validated_prerequisite(
                repo=ROOT,
                label="authority_cpu_contract",
                path=cpu_path,
                expected_sha256=sha256_file(cpu_path),
                expected_classification="cpu-contract-passed",
                expected_status="passed",
                current_commit=self.commit,
                formal_source_contract=self.source,
            )
            smoke = _validated_prerequisite(
                repo=ROOT,
                label="functional_smoke",
                path=smoke_path,
                expected_sha256=sha256_file(smoke_path),
                expected_classification="functional-smoke-passed",
                expected_status="stable",
                current_commit=self.commit,
                formal_source_contract=self.source,
            )
            self.assertEqual(cpu["formal_source_contract"], self.source)
            self.assertEqual(smoke["formal_source_contract"], self.source)
            self.assertTrue(cpu["git_commit_is_current_ancestor"])
            self.assertTrue(smoke["git_commit_is_current_ancestor"])

    def test_resolved_eligibility_config_contains_no_interpolation(self):
        date = datetime.now().astimezone().strftime("%Y%m%d")
        run_id = (
            "p1-hoi-d2af-clean-signal-eligibility-"
            f"s42-{date}"
        )
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            resolved = resolved_workload_config(
                repo=ROOT,
                run_id=run_id,
                output=directory / "metrics.json",
                resolved_config_output=directory / "resolved.yaml",
                identity={"git_commit": self.commit},
                formal_source_contract=self.source,
                cpu_path=directory / "cpu.json",
                cpu_sha256="0" * 64,
                smoke_path=directory / "smoke.json",
                smoke_sha256="1" * 64,
            )
        self.assertNotIn("${", resolved)
        self.assertIn("model_created: false", resolved)
        self.assertIn("checkpoint_loads: 0", resolved)
        self.assertIn("batch_size: 128", resolved)
        self.assertIn("num_workers: 0", resolved)

    def test_minimal_passing_artifact_satisfies_trainer_validator(self):
        date = datetime.now().astimezone().strftime("%Y%m%d")
        cfg = resolved_config(ROOT, date)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            path = directory / "eligibility.json"
            resolved_path = directory / "resolved.yaml"
            resolved_path.write_text(
                "schema_version: 1\n",
                encoding="utf-8",
            )
            binding_common = {
                "git_commit": self.commit,
                "formal_source_contract": self.source,
                "resolved_config_path": str(resolved_path.resolve()),
                "resolved_config_sha256": sha256_file(resolved_path),
                "checks": {"all_required": True},
            }
            value = {
                "schema_version": 1,
                "status": "passed",
                "classification": "clean-signal-premise-passed",
                "run_id": (
                    "p1-hoi-d2af-clean-signal-eligibility-"
                    f"s42-{date}"
                ),
                "seed": 42,
                "checkpoint_loads": 0,
                "model_created": False,
                "optimizer_created": False,
                "official_test_used": False,
                "selection": {
                    "partition": "internal_validation",
                    "sequences": 216,
                    "windows": 29382,
                    "global_indices_sha256":
                        EXPECTED_GLOBAL_INDICES_SHA256,
                    "sequence_names_sha256":
                        EXPECTED_SEQUENCE_NAMES_SHA256,
                },
                "schedule": diffusion_schedule_contract_metadata(),
                "gates": {
                    "c249_minus_c0_ci_lower_gt_zero": True,
                    "c499_minus_c249_ci_lower_gt_zero": True,
                    "anchor0_prescaling_max_abs_le_1e_minus_6": True,
                },
                "formal_source_contract": self.source,
                "identity": {
                    "git_commit": self.commit,
                    "worktree_clean": True,
                },
                "authority_cpu_contract": {
                    "path": str((Path(directory) / "cpu.json").resolve()),
                    "sha256": "0" * 64,
                    "run_id": (
                        "p1-hoi-d2af-cpu-contract-"
                        f"s42-{date}"
                    ),
                    "status": "passed",
                    "classification": "cpu-contract-passed",
                    **binding_common,
                },
                "functional_smoke": {
                    "path": str((Path(directory) / "smoke.json").resolve()),
                    "sha256": "1" * 64,
                    "run_id": (
                        "p1-hoi-d2af-gpu-functional-smoke-"
                        f"s42-{date}"
                    ),
                    "status": "stable",
                    "classification": "functional-smoke-passed",
                    **binding_common,
                },
                "prerequisite_source_contract_match": True,
                "noise_streams": {
                    str(timestep): {
                        "seed": 42 + 1_000_003 * timestep,
                        "device": "cpu",
                        "dtype": "torch.float32",
                        "shape_per_window": [16, 232],
                        "values": 29382 * 16 * 232,
                        "sha256": str(timestep).zfill(64),
                    }
                    for timestep in (0, 249, 499)
                },
                "sparse_assets": {
                    "mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
                    "manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
                    "stacked_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
                },
                "resolved_config_path": str(resolved_path.resolve()),
                "resolved_config_sha256": sha256_file(resolved_path),
                "resolved_config_has_unresolved_interpolation": False,
                "formal_training_authorized": True,
                "performance_benchmark_authorized": True,
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            cfg.d2af_clean_signal_eligibility_path = str(path.resolve())
            cfg.d2af_clean_signal_eligibility_sha256 = sha256_file(path)
            contract = _validate_d2af_contract(
                cfg,
                4,
                require_eligibility_gate=True,
                require_performance_gate=False,
            )
            value["functional_smoke"]["formal_source_contract"] = {
                "sha256": "f" * 64,
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            cfg.d2af_clean_signal_eligibility_sha256 = sha256_file(path)
            with self.assertRaisesRegex(
                ValueError,
                "functional_smoke",
            ):
                _validate_d2af_contract(
                    cfg,
                    4,
                    require_eligibility_gate=True,
                    require_performance_gate=False,
                )
        self.assertTrue(contract["eligibility_gate_required"])
        self.assertIsNotNone(contract["eligibility_gate"])


class D2AFNoTrainingEligibilityStaticTests(unittest.TestCase):
    def test_eligibility_source_has_no_model_checkpoint_or_optimizer_calls(self):
        path = ROOT / "tools/run_hoi_d2af_eligibility.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        called_names = set()
        called_attributes = set()
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                called_names.add(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                parts = []
                value = node.func
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    parts.append(value.id)
                called_attributes.add(".".join(reversed(parts)))
        self.assertNotIn("build_expert", called_names)
        self.assertNotIn("load_trained_hoi_prior", called_names)
        self.assertNotIn("torch.load", called_attributes)
        self.assertFalse(any(
            value.startswith("torch.optim")
            for value in called_attributes
        ))


if __name__ == "__main__":
    unittest.main()
