from __future__ import annotations

import sys
import unittest
from datetime import datetime
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.diffusion_schedule import SQRT_ALPHA_BAR_SHA256  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    SparseCurrentStateRelationField,
    build_d2ag_relation_source,
)
from tools import benchmark_hoi_d2ag as benchmark  # noqa: E402
from tools import smoke_hoi_d2ag as smoke  # noqa: E402
from tools.diagnose_hoi_d2ae import synthetic_inputs  # noqa: E402
from train_hoi_prior import (  # noqa: E402
    D2AF_MAXIMUM_ETA_HOURS,
    D2AF_MINIMUM_THROUGHPUT,
    D2AG_FORMAL_SOURCE_SCOPES,
    D2AG_MAXIMUM_ETA_HOURS,
    D2AG_MINIMUM_THROUGHPUT,
)


ACTUAL_DATE = datetime.now().astimezone().strftime("%Y%m%d")


def relation_arguments(values):
    return {
        "rest_object_points": values["rest_object_points"],
        "world_to_local_rotation": values["world_to_local_rotation"],
        "object_rotation_reference": values["object_rotation_reference"],
        "position_minimum": values["position_minimum"],
        "position_maximum": values["position_maximum"],
        "object_minimum": values["object_minimum"],
        "object_maximum": values["object_maximum"],
    }


class StubPair:
    __slots__ = ("name",)

    def __init__(self, name: str) -> None:
        self.name = name

    def elapsed_time(self, other) -> float:
        return 1.0


class StubProfiler:
    """Duck type of tools.benchmark_hoi_d2ae.Profiler without CUDA events."""

    def __init__(self) -> None:
        self.active = True
        self.cuda_pairs: dict = {}
        self.loader_wait: list = []
        self.relation_shapes: dict = {}
        self.relation_runtime: dict = {}

    def pair(self, name: str):
        pair = StubPair(name)
        if self.active:
            self.cuda_pairs.setdefault(name, []).append((pair, pair))
        return pair

    def finish(self, end) -> None:
        return None


class D2AGFunctionalSmokeLifecycleTests(unittest.TestCase):
    def test_registered_identity_and_budget_are_exact(self):
        run_id = f"p1-hoi-d2ag-gpu-functional-smoke-s42-{ACTUAL_DATE}"
        self.assertEqual(smoke._validate_actual_run_id(run_id), ACTUAL_DATE)
        self.assertEqual(smoke.EXPECTED_BATCH_SIZE, 8)
        self.assertEqual(smoke.DISTINCT_TIMESTEPS, (0, 249, 499))
        self.assertEqual(
            smoke.REGISTERED_TIMESTEPS, (0, 249, 499, 0, 249, 499, 0, 499),
        )
        self.assertEqual(smoke.PARITY_MAX_ABS_TOLERANCE, 1.0e-6)
        self.assertEqual(
            smoke.FAILURE_CLASSIFICATION,
            "selfcond-relation-source-contract-failure-stop",
        )
        self.assertEqual(
            smoke._formal_run_id_for_date(ACTUAL_DATE),
            f"p1-hoi-d2ag-selfcond-relation-source-s42-{ACTUAL_DATE}",
        )
        for invalid in (
            f"p1-hoi-d2af-gpu-functional-smoke-s42-{ACTUAL_DATE}",
            "p1-hoi-d2ag-gpu-functional-smoke-s42-20200101",
        ):
            with self.assertRaises(ValueError):
                smoke._validate_actual_run_id(invalid)

    def test_resolved_config_binds_the_selfcond_no_update_contract(self):
        formal_run_id = smoke._formal_run_id_for_date(ACTUAL_DATE)
        cfg = smoke._resolved_config(ROOT, formal_run_id)
        source_contract = {
            "algorithm": "test",
            "scopes": list(D2AG_FORMAL_SOURCE_SCOPES),
            "tracked_file_count": 1,
            "sha256": "a" * 64,
        }
        resolved = OmegaConf.create(smoke._resolved_workload_config(
            cfg,
            repo=ROOT,
            run_id=f"p1-hoi-d2ag-gpu-functional-smoke-s42-{ACTUAL_DATE}",
            expected_commit="b" * 40,
            formal_source_contract=source_contract,
            output=ROOT / "results/test-smoke.json",
            resolved_config_output=ROOT / "results/test-smoke.yaml",
        ))
        self.assertEqual(
            resolved.lifecycle, "d2ag_single_gpu_functional_smoke",
        )
        workload = resolved.workload
        self.assertEqual(workload.batch_size, 8)
        self.assertEqual(workload.optimizer_updates, 0)
        self.assertEqual(workload.checkpoint_loads, 0)
        self.assertEqual(workload.checkpoint_writes, 0)
        self.assertTrue(workload.random_initialization)
        self.assertEqual(
            workload.variable_anchor_source, "detached_model_x0_hat",
        )
        self.assertEqual(
            workload.unselected_variable_anchor_source, "current_noisy_state",
        )
        self.assertEqual(workload.history_anchor_source, "current_noisy_state")
        self.assertEqual(
            workload.self_condition_probability,
            D2AG_SELF_CONDITION_PROBABILITY,
        )
        self.assertTrue(workload.relation_field_always_active)
        self.assertFalse(workload.sqrt_alpha_bar_attenuation)

    def test_static_source_contract_holds_and_detects_a_real_dereference(self):
        value = smoke._static_source_contract()
        self.assertTrue(all(value["checks"].values()), value["checks"])
        self.assertEqual(value["schedule_dereference_sites"], [])
        token = "field." + "sqrt_alpha" + "_bar"
        self.assertEqual(
            len(smoke.schedule_dereference_sites(f"x = {token}.gather(0, t)")), 1,
        )
        self.assertEqual(
            len(smoke.schedule_dereference_sites(f"if {token} is None: pass")), 0,
        )
        self.assertEqual(
            len(
                smoke.schedule_dereference_sites(f"if {token} is not None: pass")
            ),
            0,
        )

    def test_probe_selection_mask_is_a_partial_subset(self):
        mask = smoke.PROBE_SELECTION_MASK
        self.assertEqual(len(mask), smoke.EXPECTED_BATCH_SIZE)
        self.assertGreater(sum(mask), 0)
        self.assertLess(sum(mask), len(mask))


class D2AGBenchmarkLifecycleTests(unittest.TestCase):
    def test_registered_identity_and_budget_are_exact(self):
        run_id = f"p1-hoi-d2ag-performance-benchmark-s42-{ACTUAL_DATE}"
        self.assertTrue(benchmark.RUN_ID_RE.fullmatch(run_id))
        self.assertIsNone(
            benchmark.RUN_ID_RE.fullmatch(
                f"p1-hoi-d2af-performance-benchmark-s42-{ACTUAL_DATE}"
            )
        )
        self.assertEqual(benchmark.WORLD_SIZE, 4)
        self.assertEqual(benchmark.MICRO_BATCH_PER_GPU, 512)
        self.assertEqual(benchmark.EFFECTIVE_BATCH, 2048)
        self.assertEqual(benchmark.WARMUP_UPDATES, 64)
        self.assertEqual(benchmark.MEASURED_UPDATES, 256)
        self.assertEqual(benchmark.TOTAL_UPDATES, 320)
        self.assertEqual(benchmark.MEASURED_WINDOWS, 524_288)
        self.assertEqual(benchmark.FORMAL_WINDOWS, 61_440_000)
        self.assertEqual(
            benchmark.FAILURE_CLASSIFICATION,
            "selfcond-relation-source-performance-negative-stop",
        )

    def test_gate_is_bound_to_the_d2ag_floor_only(self):
        self.assertEqual(benchmark.MINIMUM_THROUGHPUT, D2AG_MINIMUM_THROUGHPUT)
        self.assertEqual(benchmark.MAXIMUM_ETA_HOURS, D2AG_MAXIMUM_ETA_HOURS)
        self.assertNotEqual(
            benchmark.MINIMUM_THROUGHPUT, D2AF_MINIMUM_THROUGHPUT,
        )
        self.assertNotEqual(
            benchmark.MAXIMUM_ETA_HOURS, D2AF_MAXIMUM_ETA_HOURS,
        )
        self.assertAlmostEqual(
            benchmark.MINIMUM_THROUGHPUT,
            0.85 * benchmark.SEALED_D2X_THROUGHPUT,
            places=6,
        )

    def test_resolved_config_declares_both_instrumentation_fixes(self):
        formal_run_id = smoke._formal_run_id_for_date(ACTUAL_DATE)
        cfg = smoke._resolved_config(ROOT, formal_run_id)
        source_contract = {
            "algorithm": "test",
            "scopes": list(D2AG_FORMAL_SOURCE_SCOPES),
            "tracked_file_count": 1,
            "sha256": "a" * 64,
        }
        resolved = OmegaConf.create(benchmark._resolved_workload_config(
            cfg,
            repo=ROOT,
            run_id=f"p1-hoi-d2ag-performance-benchmark-s42-{ACTUAL_DATE}",
            expected_commit="b" * 40,
            formal_source_contract=source_contract,
            output_dir=ROOT / "results/test-benchmark",
            resolved_config_output=ROOT / "results/test-benchmark.yaml",
        ))
        categories = list(resolved.timing.categories)
        self.assertIn("estimate_trunk_forward", categories)
        self.assertIn("gpu_relation_geometry_estimate", categories)
        self.assertIn("gpu_pool_route_writeback_derived", categories)
        self.assertNotIn("gpu_pool_route_rho_writeback_derived", categories)
        self.assertEqual(resolved.workload.estimate_forward_per_update, 1)
        self.assertTrue(
            resolved.workload.estimate_forward_selected_subset_only
        )
        self.assertFalse(resolved.workload.sqrt_alpha_bar_attenuation)
        self.assertFalse(
            resolved.identity_contracts.field_schedule_buffer_registered
        )
        gate = resolved.performance_gate
        self.assertEqual(
            gate.minimum_throughput_windows_per_second, D2AG_MINIMUM_THROUGHPUT,
        )
        self.assertEqual(
            gate.maximum_full_budget_eta_hours, D2AG_MAXIMUM_ETA_HOURS,
        )
        self.assertTrue(gate.d2af_predecessor_gate_inapplicable)
        self.assertFalse(gate.d2af_waiver_inherited)
        self.assertFalse(gate.sweep_on_failure)

    def test_benchmark_never_dereferences_the_field_schedule(self):
        source = (ROOT / "tools/benchmark_hoi_d2ag.py").read_text(
            encoding="utf-8",
        )
        self.assertEqual(smoke.schedule_dereference_sites(source), [])
        self.assertNotIn("set_rho_override(", source)


class D2AGBenchmarkInstrumentationTests(unittest.TestCase):
    """CPU proof of the two mandatory benchmark instrumentation fixes.

    A two-pass update is driven through the real relation field: first the
    subset-shaped estimate pass with ``relation_source_estimate`` set exactly as
    ``models.py`` sets it, then the full-batch graph pass.
    """

    FULL_BATCH = 6
    SELECTED = 2
    UPDATES = 3
    GRAPH_NAMES = (
        "gpu_relation_geometry",
        "gpu_relation_module",
        "gpu_point_encoder",
        "gpu_relation_projection",
        "gpu_relation_norm",
    )

    def drive(self, *, set_estimate_flag: bool) -> dict:
        field = SparseCurrentStateRelationField(
            512, selfcond_relation_source=True,
        ).eval()
        profiler = StubProfiler()
        state: dict = {"calls": 0, "selected_count": None}
        observer = benchmark.estimate_observer(profiler, state)
        instrumentation = benchmark.install_relation_instrumentation(
            profiler, field,
        )
        full = synthetic_inputs(batch=self.FULL_BATCH, seed=7)
        subset = synthetic_inputs(batch=self.SELECTED, seed=7)
        motion_full = torch.randn(self.FULL_BATCH, 16, 512)
        motion_subset = torch.randn(self.SELECTED, 16, 512)
        try:
            for _ in range(self.UPDATES):
                end = profiler.pair("forward_and_loss")
                observer("begin", self.SELECTED)
                field.relation_source_estimate = bool(set_estimate_flag)
                with torch.no_grad():
                    field(
                        motion_subset,
                        subset["current"],
                        **relation_arguments(subset),
                        timesteps=subset["timesteps"],
                    )
                field.relation_source_estimate = False
                observer("end", self.SELECTED)
                with torch.no_grad():
                    field(
                        motion_full,
                        full["current"],
                        **relation_arguments(full),
                        timesteps=full["timesteps"],
                    )
                profiler.finish(end)
        finally:
            instrumentation["remove"]()
        residual = benchmark._derived_pool_route_summary(profiler)
        return {
            "latched_batch": profiler.relation_shapes.get(
                "current", [None]
            )[0],
            "graph_counts": {
                name: len(profiler.cuda_pairs.get(name, []))
                for name in self.GRAPH_NAMES
            },
            "estimate_geometry": len(
                profiler.cuda_pairs.get("gpu_relation_geometry_estimate", [])
            ),
            "estimate_trunk": len(
                profiler.cuda_pairs.get("estimate_trunk_forward", [])
            ),
            "derived_count": residual["count"],
            "observer_calls": state["calls"],
            "relation_runtime": profiler.relation_runtime,
        }

    def test_shapes_latch_on_the_graph_pass_not_the_estimate_pass(self):
        value = self.drive(set_estimate_flag=True)
        self.assertEqual(value["latched_batch"], self.FULL_BATCH)
        self.assertFalse(
            value["relation_runtime"]["field_schedule_buffer_registered"]
        )

    def test_one_graph_pair_and_one_estimate_pair_per_update(self):
        value = self.drive(set_estimate_flag=True)
        for name, count in value["graph_counts"].items():
            self.assertEqual(count, self.UPDATES, name)
        self.assertEqual(value["estimate_geometry"], self.UPDATES)
        self.assertEqual(value["estimate_trunk"], self.UPDATES)
        self.assertEqual(value["derived_count"], self.UPDATES)
        self.assertEqual(value["observer_calls"], self.UPDATES)

    def test_both_defects_reproduce_without_the_estimate_flag(self):
        broken = self.drive(set_estimate_flag=False)
        # Fix 1: shapes would latch the subset batch and fail the rank-0 check.
        self.assertEqual(broken["latched_batch"], self.SELECTED)
        # Fix 2: graph names would collect two pairs per update, so the derived
        # residual count no longer equals MEASURED_UPDATES.
        self.assertEqual(
            broken["graph_counts"]["gpu_relation_module"], 2 * self.UPDATES,
        )
        self.assertNotEqual(broken["derived_count"], self.UPDATES)

    def test_observer_rejects_an_unknown_stage(self):
        profiler = StubProfiler()
        observer = benchmark.estimate_observer(profiler, {})
        with self.assertRaises(ValueError):
            observer("middle", 1)


class D2AGSmokeSourceSemanticsTests(unittest.TestCase):
    def test_selfcond_source_is_partial_and_pins_history_on_cpu(self):
        values = synthetic_inputs(batch=smoke.EXPECTED_BATCH_SIZE, seed=31)
        current = values["current"]
        estimate = torch.randn(
            current.shape, generator=torch.Generator().manual_seed(77),
        )
        mask = torch.tensor(smoke.PROBE_SELECTION_MASK, dtype=torch.bool)
        index = mask.nonzero(as_tuple=True)[0]
        source = build_d2ag_relation_source(
            current, estimate.index_select(0, index), index=index,
        )
        self.assertEqual(
            float((source[:, :2] - current[:, :2]).abs().amax()), 0.0,
        )
        unselected = (~mask).nonzero(as_tuple=True)[0]
        self.assertEqual(
            float(
                (
                    source.index_select(0, unselected)
                    - current.index_select(0, unselected)
                ).abs().amax()
            ),
            0.0,
        )
        variable = list(D2AG_VARIABLE_ANCHORS)
        self.assertEqual(
            float(
                (
                    source.index_select(0, index)[:, variable]
                    - estimate.index_select(0, index)[:, variable]
                ).abs().amax()
            ),
            0.0,
        )

    def test_schedule_hash_reference_is_the_diffusion_module_not_the_field(self):
        source = (ROOT / "tools/smoke_hoi_d2ag.py").read_text(encoding="utf-8")
        self.assertIn("tensor_sha256(diffusion.sqrt_alpha_bar)", source)
        self.assertEqual(smoke.schedule_dereference_sites(source), [])
        # The smoke must compare against the canonical constant by name, so the
        # literal digest never appears inline where it could drift.
        self.assertIn("SQRT_ALPHA_BAR_SHA256", source)
        self.assertNotIn(SQRT_ALPHA_BAR_SHA256, source)


if __name__ == "__main__":
    unittest.main()
