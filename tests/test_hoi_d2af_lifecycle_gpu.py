from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys
import unittest

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.diffusion_schedule import SQRT_ALPHA_BAR_SHA256  # noqa: E402
from tools import benchmark_hoi_d2af as benchmark  # noqa: E402
from tools import smoke_hoi_d2af as smoke  # noqa: E402
from train_hoi_prior import D2AF_FORMAL_SOURCE_SCOPES  # noqa: E402


class D2AFFunctionalSmokeLifecycleTests(unittest.TestCase):
    def test_registered_identity_and_budget_are_exact(self):
        actual_date = datetime.now().astimezone().strftime("%Y%m%d")
        run_id = f"p1-hoi-d2af-gpu-functional-smoke-s42-{actual_date}"
        self.assertEqual(smoke._validate_actual_run_id(run_id), actual_date)
        self.assertEqual(smoke.EXPECTED_BATCH_SIZE, 8)
        self.assertEqual(smoke.DISTINCT_TIMESTEPS, (0, 249, 499))
        self.assertEqual(
            smoke.REGISTERED_TIMESTEPS,
            (0, 249, 499, 0, 249, 499, 0, 499),
        )
        self.assertEqual(smoke.SCALING_MAX_ABS_TOLERANCE, 1.0e-6)
        self.assertEqual(
            smoke.FAILURE_CLASSIFICATION,
            "diffusion-reliability-contract-failure-stop",
        )

    def test_resolved_config_binds_source_schedule_and_no_update_contract(self):
        actual_date = datetime.now().astimezone().strftime("%Y%m%d")
        formal_run_id = smoke._formal_run_id_for_date(actual_date)
        cfg = smoke._resolved_config(ROOT, formal_run_id)
        source_contract = {
            "algorithm": "test",
            "scopes": list(D2AF_FORMAL_SOURCE_SCOPES),
            "tracked_file_count": 1,
            "sha256": "a" * 64,
        }
        resolved = OmegaConf.create(smoke._resolved_workload_config(
            cfg,
            repo=ROOT,
            run_id=(
                "p1-hoi-d2af-gpu-functional-smoke-"
                f"s42-{actual_date}"
            ),
            expected_commit="b" * 40,
            formal_source_contract=source_contract,
            output=ROOT / "results/test-smoke.json",
            resolved_config_output=ROOT / "results/test-smoke.yaml",
        ))
        self.assertEqual(
            OmegaConf.to_container(resolved.formal_source_contract),
            source_contract,
        )
        self.assertEqual(
            resolved.contracts.schedule.sqrt_alpha_bar_sha256,
            SQRT_ALPHA_BAR_SHA256,
        )
        self.assertFalse(resolved.workload.optimizer_created)
        self.assertEqual(resolved.workload.optimizer_updates, 0)
        self.assertEqual(resolved.workload.checkpoint_loads, 0)
        self.assertEqual(resolved.workload.checkpoint_writes, 0)
        self.assertEqual(
            list(resolved.workload.timesteps),
            [0, 249, 499],
        )


class D2AFPerformanceBenchmarkLifecycleTests(unittest.TestCase):
    def test_registered_full_micro_batch_gate_is_exact(self):
        self.assertEqual(benchmark.WORLD_SIZE, 4)
        self.assertEqual(benchmark.MICRO_BATCH_PER_GPU, 512)
        self.assertEqual(benchmark.EFFECTIVE_BATCH, 2048)
        self.assertEqual(benchmark.WARMUP_UPDATES, 64)
        self.assertEqual(benchmark.MEASURED_UPDATES, 256)
        self.assertEqual(benchmark.TOTAL_UPDATES, 320)
        self.assertEqual(benchmark.MEASURED_WINDOWS, 524_288)
        self.assertAlmostEqual(
            benchmark.MINIMUM_THROUGHPUT,
            3179.689863044761,
        )
        self.assertAlmostEqual(
            benchmark.MAXIMUM_ETA_HOURS,
            5.367399778519349,
        )
        self.assertEqual(
            benchmark.FAILURE_CLASSIFICATION,
            "diffusion-reliability-performance-negative-stop",
        )

    def test_resolved_config_binds_eligibility_formal_id_and_hashes(self):
        actual_date = datetime.now().astimezone().strftime("%Y%m%d")
        formal_run_id = smoke._formal_run_id_for_date(actual_date)
        cfg = smoke._resolved_config(ROOT, formal_run_id)
        eligibility_path = ROOT / "results/eligibility.json"
        cfg.d2af_clean_signal_eligibility_path = str(eligibility_path)
        cfg.d2af_clean_signal_eligibility_sha256 = "c" * 64
        source_contract = {
            "algorithm": "test",
            "scopes": list(D2AF_FORMAL_SOURCE_SCOPES),
            "tracked_file_count": 2,
            "sha256": "d" * 64,
        }
        eligibility_contract = {
            "path": str(eligibility_path),
            "sha256": "c" * 64,
            "run_id": (
                "p1-hoi-d2af-clean-signal-eligibility-"
                f"s42-{actual_date}"
            ),
            "formal_source_contract": source_contract,
        }
        resolved = OmegaConf.create(benchmark._resolved_workload_config(
            cfg,
            repo=ROOT,
            run_id=(
                "p1-hoi-d2af-performance-benchmark-"
                f"s42-{actual_date}"
            ),
            expected_commit="e" * 40,
            formal_source_contract=source_contract,
            eligibility_contract=eligibility_contract,
            output_dir=ROOT / "results/test-benchmark",
            resolved_config_output=ROOT / "results/test-benchmark.yaml",
        ))
        self.assertEqual(resolved.formal_run_id, formal_run_id)
        self.assertEqual(
            resolved.identity_contracts.eligibility_sha256,
            "c" * 64,
        )
        self.assertEqual(
            resolved.identity_contracts.sqrt_alpha_bar_sha256,
            SQRT_ALPHA_BAR_SHA256,
        )
        self.assertEqual(
            resolved.performance_gate.minimum_throughput_windows_per_second,
            benchmark.MINIMUM_THROUGHPUT,
        )
        self.assertEqual(
            resolved.performance_gate.maximum_full_budget_eta_hours,
            benchmark.MAXIMUM_ETA_HOURS,
        )
        self.assertIn(
            "gpu_pool_route_rho_writeback_derived",
            list(resolved.timing.categories),
        )
        launcher = list(resolved.launcher)
        self.assertIn("--eligibility-path", launcher)
        self.assertIn("--eligibility-sha256", launcher)

    def test_runtime_source_scope_and_gpu_only_fields_are_registered(self):
        self.assertIn("tools/smoke_hoi_d2af.py", D2AF_FORMAL_SOURCE_SCOPES)
        self.assertIn("tools/benchmark_hoi_d2af.py", D2AF_FORMAL_SOURCE_SCOPES)
        source = (
            ROOT / "tools/benchmark_hoi_d2af.py"
        ).read_text(encoding="utf-8")
        for required in (
            '"cpu_dynamic_geometry": False',
            '"relation_build_device": "cuda"',
            '"cuda_timing_synchronized": True',
            '"four_rank_schedule_hashes"',
            '"eligibility_sha256"',
        ):
            self.assertIn(required, source)


if __name__ == "__main__":
    unittest.main()
