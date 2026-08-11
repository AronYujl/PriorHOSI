import hashlib
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh
from pytorch3d import transforms

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from datasets.utils import get_smpl_parents, zup_to_yup
from priors.hoi.data import PriorWindowDataset
from priors.hoi.diffusion import GaussianDiffusion, prepare_clean_x0
from priors.hoi.losses import hoi_training_losses
from priors.hoi.remediation import (
    bps_replay_equivalence_gate,
    deterministic_derangement,
    field_squared_error,
    select_internal_triples,
    select_teacher_windows,
    selection_sha256,
)
from priors.core.window_codec import (
    WindowFrame,
    project_to_so3,
    rotation_geodesic,
    zup_to_yup_tensor,
)
from tools.diagnose_hoi_bps_equivalence import (
    EXPECTED_GLOBAL_SELECTION_SHA256,
    EXPECTED_SEQUENCE_WINDOW_SHA256,
    PLY_SHA256,
    replay_device as d2c_replay_device,
    resolved_config as d2c_resolved_config,
    select_d2c_windows,
    verify_assets as verify_d2c_assets,
)
from tools.diagnose_hoi_bps_tolerance import (
    EXPECTED_HASHES as D2D_EXPECTED_HASHES,
    NEAREST_LINEAR_DISTANCE_GAP_M_MAX as D2D_LINEAR_GAP,
    NEAREST_SQUARED_DISTANCE_GAP_M2_MAX as D2D_SQUARED_GAP,
    resolved_config as d2d_resolved_config,
    select_d2d_windows,
)
from tools.diagnose_hoi_bps_linear_equivalence import (
    EXPECTED_HASHES as D2E_EXPECTED_HASHES,
    NEAREST_LINEAR_DISTANCE_GAP_M_MAX as D2E_LINEAR_GAP,
    resolved_config as d2e_resolved_config,
    select_d2e_windows,
)
from tools.diagnose_hoi_d2p import (
    D0_T499,
    D2_THRESHOLDS,
    classify_mechanism,
    field_error_per_sample,
    resolved_config as d2p_resolved_config,
)
from tools.diagnose_hoi_d2f import (
    classify as classify_d2f,
    object_rotation_manifold_error,
    resolved_config as d2f_resolved_config,
)
from tools.diagnose_hoi_bps_backend import (
    AUTHOR_BASELINE,
    AUTHOR_BPS_BLOB,
    BPS_TOLERANCE,
    classify_backend,
    resolved_config as d2b_resolved_config,
    yup_to_zup_tensor,
)
from tools.evaluate_hoi_remediation import paired_bootstrap
from tools.select_hoi_remediation import select

class RemediationDiagnosticTest(unittest.TestCase):
    def _dataset(self):
        names = np.asarray(("seq_a", "seq_b"), dtype=object)
        sequence_ids = np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int64)
        return SimpleNamespace(
            partition="internal_validation",
            indices=np.arange(6),
            sequence_ids=sequence_ids,
            scene_names=names,
            language={"pi": np.asarray((0, 42, 84, 0, 42, 84), dtype=np.int64)},
        )

    def test_selection_is_internal_only_and_deterministic(self):
        dataset = self._dataset()
        self.assertEqual(select_internal_triples(dataset, 2), select_internal_triples(dataset, 2))
        self.assertEqual(len(select_teacher_windows(dataset, 4)), 4)
        dataset.partition = "test"
        with self.assertRaisesRegex(ValueError, "internal-validation only"):
            select_internal_triples(dataset, 1)

    def test_derangement(self):
        permutation = deterministic_derangement(8)
        self.assertTrue(torch.all(permutation != torch.arange(8)))
        self.assertEqual(sorted(permutation.tolist()), list(range(8)))

    def test_fieldwise_error_excludes_history(self):
        target = torch.zeros(2, 16, 232)
        prediction = target.clone()
        prediction[:, :2] = 100.0
        errors = field_squared_error(prediction, target)
        self.assertTrue(all(float(value) == 0.0 for value in errors.values()))

        per_sample = field_error_per_sample(prediction, target)
        self.assertTrue(all(value.shape == (2,) for value in per_sample.values()))
        self.assertTrue(all(float(value.max()) == 0.0 for value in per_sample.values()))

    def test_paired_bootstrap_requires_positive_lower_bound(self):
        matched = np.zeros(128)
        permuted = np.linspace(0.5, 1.5, 128)
        result = paired_bootstrap(matched, permuted)
        self.assertTrue(result["matched_significantly_better"])
        self.assertGreater(result["bootstrap_95_ci"][0], 0.0)

    def test_d2_g_is_allowed_only_for_contact_only_failure(self):
        checks = {
            "object_goal_ratio_le_0.70": True,
            "pelvis_goal_ratio_le_0.70": True,
            "contact_f1_increase_ge_0.10": False,
            "mpjpe_ratio_le_1.10": True,
            "foot_sliding_ratio_le_1.10": True,
            "finite": True,
            "all_conditions_significant": True,
        }
        record = {
            "eligible": False, "checks": checks, "geometry_score": 0.5,
            "physical_contact_f1": 0.2, "effective_batch_size": 1024,
        }
        self.assertEqual(select([record])["decision"], "D2-G-contact-only-fallback")
        record["checks"] = dict(checks, finite=False)
        self.assertEqual(select([record])["decision"], "stop-no-eligible-candidate")

    @staticmethod
    def _mechanism_candidate(joint_ratio, object_ratio, text_significant, bps_significant, trace_passed):
        return {
            "teacher_x0": {
                "499": {
                    "matched_fieldwise_mse": {
                        "joint_positions": D0_T499["joint_positions"] * joint_ratio,
                        "object_translation": D0_T499["object_translation"] * object_ratio,
                    },
                    "sensitivity": {
                        "text_permuted": {"bootstrap": {"matched_significantly_better": text_significant}},
                        "bps_permuted": {"bootstrap": {"matched_significantly_better": bps_significant}},
                    },
                },
            },
            "reverse_trace": {
                "final": {
                    "d2_threshold_checks": {
                        "object_goal_error_cm": trace_passed,
                        "pelvis_goal_error_cm": trace_passed,
                        "mpjpe_cm": trace_passed,
                    },
                },
            },
        }

    def test_d2p_mechanism_classification_is_deterministic(self):
        failed_contract = classify_mechanism({"passed": False}, {})
        self.assertEqual(failed_contract["category"], "coordinate-contract-defect")
        underfit = self._mechanism_candidate(2.0, 2.0, False, False, False)
        result = classify_mechanism({"passed": True}, {"R-1024": underfit, "R-3072": underfit})
        self.assertEqual(result["category"], "high-noise-condition-underfit")
        exposure = self._mechanism_candidate(1.0, 1.0, True, True, False)
        result = classify_mechanism({"passed": True}, {"R-1024": exposure, "R-3072": exposure})
        self.assertEqual(result["category"], "reverse-process-exposure-gap")
        mixed = self._mechanism_candidate(2.0, 1.0, False, False, False)
        result = classify_mechanism({"passed": True}, {"R-1024": mixed, "R-3072": mixed})
        self.assertEqual(result["category"], "mixed-mechanism")

    def test_d2p_resolved_config_forbids_selection_and_official_data(self):
        args = SimpleNamespace(
            checkpoint_r1024="/tmp/r1024.pth",
            sha256_r1024="1" * 64,
            checkpoint_r3072="/tmp/r3072.pth",
            sha256_r3072="2" * 64,
            run_id="p1-hoi-d2p5-mechanism-s42-20260715",
            teacher_batch_size=64,
            device="cuda:0",
            output="/tmp/d2p.json",
        )
        config = d2p_resolved_config(args)
        self.assertFalse(config["official_test_used"])
        self.assertFalse(config["chois_used"])
        self.assertFalse(config["checkpoint_selection"])
        self.assertFalse(config["sampler_stored_per_frame_bps"])
        self.assertFalse(config["sampler_future_gt"])
        self.assertEqual(config["training_updates"], 0)
        self.assertEqual(config["teacher"]["windows"], 512)
        self.assertEqual(config["reverse_trace"]["sequences"], 32)
        self.assertEqual(config["subphase"], "1B-D2-P5")
        self.assertEqual(
            config["contract_replay"]["nearest_linear_distance_gap_m_max"], D2E_LINEAR_GAP,
        )
        self.assertEqual(
            config["contract_replay"]["nearest_squared_distance_gap_m2"], "report_only",
        )

    def test_d2f_config_locks_paired_so3_scope_and_gates(self):
        args = SimpleNamespace(
            checkpoint_r1024="/tmp/r1024.pth",
            sha256_r1024="1" * 64,
            checkpoint_r3072="/tmp/r3072.pth",
            sha256_r3072="2" * 64,
            run_id="p1-hoi-d2f-so3-reverse-s42-20260715",
            device="cuda:0",
            output="/tmp/d2f.json",
        )
        config = d2f_resolved_config(args)
        self.assertEqual(config["subphase"], "1B-D2-F0")
        self.assertEqual(config["paired_variants"], ["control", "object_so3_x0"])
        self.assertTrue(config["paired_initial_and_posterior_noise"])
        self.assertEqual(config["projection_channels"], [219, 228])
        self.assertTrue(config["projection_before_each_posterior_mean"])
        self.assertFalse(config["other_channel_clamp"])
        self.assertFalse(config["posterior_change"])
        self.assertFalse(config["checkpoint_selection"])
        self.assertEqual(config["training_updates"], 0)
        self.assertFalse(config["official_test_used"])
        self.assertFalse(config["chois_used"])
        self.assertFalse(config["sampler_stored_per_frame_bps"])
        self.assertFalse(config["sampler_future_gt"])
        self.assertEqual(config["absolute_gate"]["object_goal_error_cm_max"], D2_THRESHOLDS["object_goal_error_cm"])

    @staticmethod
    def _d2f_variant(object_goal, pelvis_goal, mpjpe, *, manifold=True):
        metrics = {
            "object_goal_error_cm": object_goal,
            "pelvis_goal_error_cm": pelvis_goal,
            "mpjpe_cm": mpjpe,
        }
        return {
            "paired_seed": 42,
            "final": {
                **metrics,
                "finite": True,
                "history_check": True,
                "manifold_checks": {
                    "orthogonality_frobenius": manifold,
                    "determinant_abs_error": manifold,
                },
                "d2_threshold_checks": {
                    name: value <= D2_THRESHOLDS[name] for name, value in metrics.items()
                },
            },
        }

    def test_d2f_classification_has_fixed_absolute_and_training_triggers(self):
        control = self._d2f_variant(100.0, 20.0, 50.0)
        projected = self._d2f_variant(40.0, 20.5, 50.5)
        candidates = {
            "R-1024": {"control": control, "object_so3_x0": projected},
            "R-3072": {"control": control, "object_so3_x0": control},
        }
        decision = classify_d2f(candidates)
        self.assertEqual(decision["category"], "sampler-mechanism-positive-training-insufficient")
        self.assertFalse(decision["d2f1_authorized"])
        self.assertTrue(decision["d2f2_authorized"])
        candidates["R-1024"]["object_so3_x0"] = self._d2f_variant(7.0, 20.0, 30.0)
        decision = classify_d2f(candidates)
        self.assertEqual(decision["absolute_gate_passes"], ["R-1024"])
        self.assertTrue(decision["d2f1_authorized"])
        self.assertFalse(decision["d2f2_authorized"])

    def test_d2b_config_is_backend_only_and_fixed_to_author_assets(self):
        args = SimpleNamespace(
            run_id="p1-hoi-d2b-bps-replay-s42-20260715",
            output="/tmp/d2b.json",
            cuda_device="cuda:0",
        )
        config = d2b_resolved_config(args)
        self.assertEqual(config["author_baseline_commit"], AUTHOR_BASELINE)
        self.assertEqual(config["author_bps_blob"], AUTHOR_BPS_BLOB)
        self.assertEqual(config["bps_max_abs_tolerance"], BPS_TOLERANCE)
        self.assertEqual(config["devices"], ["cpu", "cuda:0"])
        self.assertEqual(config["checkpoint_count_loaded"], 0)
        self.assertEqual(config["model_forward_calls"], 0)
        self.assertEqual(config["training_updates"], 0)
        self.assertFalse(config["official_test_used"])
        self.assertFalse(config["chois_used"])
        self.assertFalse(config["checkpoint_selection"])

    def test_d2b_coordinate_conversion_and_gate_are_not_relaxed(self):
        raw = torch.tensor(((1.0, 2.0, 3.0), (-4.0, 5.0, -6.0)))
        converted = zup_to_yup_tensor(raw)
        torch.testing.assert_close(yup_to_zup_tensor(converted), raw)
        cpu = {"passed": False}
        self.assertEqual(
            classify_backend(cpu, {"passed": True}), "cpu-knn-tie-backend-artifact",
        )
        self.assertEqual(
            classify_backend(cpu, {"passed": False}), "backend-replay-unresolved",
        )

    def test_d2c_gate_accepts_only_strict_or_provable_mesh_ties(self):
        basis = torch.zeros(2, 3)
        vertices = torch.tensor(((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (2.0, 0.0, 0.0)))
        selected = torch.tensor((0, 0), dtype=torch.long)
        recomputed = zup_to_yup_tensor(vertices[selected] - basis)
        stored = recomputed.clone()
        stored[1] = zup_to_yup_tensor(vertices[1] - basis[1])
        gate = bps_replay_equivalence_gate(recomputed, stored, basis, vertices, selected)
        self.assertTrue(gate["passed"])
        self.assertTrue(bool(gate["strict"][0]))
        self.assertTrue(bool(gate["tie"][1]))
        self.assertLessEqual(float(gate["stored_mesh_residual_m"][1]), 1e-6)
        self.assertLessEqual(float(gate["nearest_squared_distance_gap_m2"][1]), 1e-7)
        self.assertEqual(gate["nearest_squared_distance_gap_m2"].dtype, torch.float64)

        non_tie = stored.clone()
        non_tie[1] = zup_to_yup_tensor(vertices[2] - basis[1])
        rejected = bps_replay_equivalence_gate(recomputed, non_tie, basis, vertices, selected)
        self.assertFalse(rejected["passed"])
        self.assertTrue(bool(rejected["failure"][1]))

        off_mesh = stored.clone()
        off_mesh[1] += torch.tensor((0.0, 0.0, 2e-6))
        rejected = bps_replay_equivalence_gate(recomputed, off_mesh, basis, vertices, selected)
        self.assertFalse(rejected["passed"])
        self.assertTrue(bool(rejected["failure"][1]))

    def test_d2c_config_forbids_sampler_gt_and_checkpoint_work(self):
        args = SimpleNamespace(
            run_id="p1-hoi-d2c-bps-equivalence-s42-20260715",
            output="/tmp/d2c.json",
            cuda_device="cuda:0",
        )
        config = d2c_resolved_config(args)
        self.assertFalse(config["sampler_stored_per_frame_bps"])
        self.assertFalse(config["sampler_future_gt"])
        self.assertEqual(config["checkpoint_count_loaded"], 0)
        self.assertEqual(config["model_forward_calls"], 0)
        self.assertEqual(config["training_updates"], 0)
        self.assertEqual(config["selection"]["windows"], 832)
        self.assertEqual(config["gate"]["strict_component_max_abs"], 1e-4)
        self.assertIn("stop before CUDA", config["execution_order"])

    def test_d2d_dual_tolerance_accepts_float32_tie_but_retains_linear_cap(self):
        basis = torch.zeros(1, 3)
        vertices = torch.tensor(((1.0, 0.0, 0.0), (-1.0000001, 0.0, 0.0)))
        selected = torch.tensor((0,), dtype=torch.long)
        recomputed = zup_to_yup_tensor(vertices[selected] - basis)
        stored = zup_to_yup_tensor(vertices[1:] - basis)
        old_gate = bps_replay_equivalence_gate(
            recomputed, stored, basis, vertices, selected,
        )
        self.assertFalse(old_gate["passed"])
        adjusted = bps_replay_equivalence_gate(
            recomputed, stored, basis, vertices, selected,
            nearest_squared_distance_gap_m2_max=D2D_SQUARED_GAP,
            nearest_linear_distance_gap_m_max=D2D_LINEAR_GAP,
        )
        self.assertTrue(adjusted["passed"])
        self.assertGreater(float(adjusted["nearest_squared_distance_gap_m2"][0]), 1e-7)
        self.assertLessEqual(
            float(adjusted["nearest_squared_distance_gap_m2"][0]), D2D_SQUARED_GAP,
        )
        self.assertLessEqual(
            float(adjusted["nearest_linear_distance_gap_m"][0]), D2D_LINEAR_GAP,
        )

        near_origin = torch.tensor(((0.001, 0.0, 0.0), (-0.0010003, 0.0, 0.0)))
        recomputed = zup_to_yup_tensor(near_origin[:1] - basis)
        stored = zup_to_yup_tensor(near_origin[1:] - basis)
        rejected = bps_replay_equivalence_gate(
            recomputed, stored, basis, near_origin, selected,
            nearest_squared_distance_gap_m2_max=D2D_SQUARED_GAP,
            nearest_linear_distance_gap_m_max=D2D_LINEAR_GAP,
        )
        self.assertLess(float(rejected["nearest_squared_distance_gap_m2"][0]), D2D_SQUARED_GAP)
        self.assertGreater(float(rejected["nearest_linear_distance_gap_m"][0]), D2D_LINEAR_GAP)
        self.assertFalse(rejected["passed"])

    def test_d2d_config_and_disjoint_holdout_are_locked(self):
        args = SimpleNamespace(
            run_id="p1-hoi-d2d-bps-tolerance-s42-20260715",
            output="/tmp/d2d.json",
            cuda_device="cuda:2",
        )
        config = d2d_resolved_config(args)
        self.assertEqual(config["selection"]["combined_windows"], 1664)
        self.assertEqual(config["selection"]["hashes"], D2D_EXPECTED_HASHES)
        self.assertEqual(config["gate"]["nearest_squared_distance_gap_m2_max"], 2.5e-7)
        self.assertEqual(config["gate"]["nearest_linear_distance_gap_m_max"], 2.5e-7)
        self.assertFalse(config["sampler_stored_per_frame_bps"])
        self.assertFalse(config["sampler_future_gt"])
        self.assertEqual(config["checkpoint_count_loaded"], 0)
        self.assertEqual(config["training_updates"], 0)
        dataset = PriorWindowDataset(
            str(REPO), "hoi", partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )
        subsets, coverage = select_d2d_windows(dataset)
        self.assertEqual(set(subsets["calibration"]) & set(subsets["holdout"]), set())
        self.assertTrue(all(len(positions) == 832 for positions in subsets.values()))
        self.assertTrue(all(set(values) == set(PLY_SHA256) for values in coverage.values()))
        self.assertTrue(all(count == 64 for values in coverage.values() for count in values.values()))

    def test_d2e_linear_gate_reports_but_does_not_gate_squared_gap(self):
        basis = torch.zeros(1, 3)
        vertices = torch.tensor(((1.0, 0.0, 0.0), (-1.0000002, 0.0, 0.0)))
        selected = torch.tensor((0,), dtype=torch.long)
        recomputed = zup_to_yup_tensor(vertices[:1] - basis)
        stored = zup_to_yup_tensor(vertices[1:] - basis)
        d2d = bps_replay_equivalence_gate(
            recomputed, stored, basis, vertices, selected,
            nearest_squared_distance_gap_m2_max=D2D_SQUARED_GAP,
            nearest_linear_distance_gap_m_max=D2D_LINEAR_GAP,
        )
        self.assertGreater(float(d2d["nearest_squared_distance_gap_m2"][0]), D2D_SQUARED_GAP)
        self.assertLessEqual(float(d2d["nearest_linear_distance_gap_m"][0]), D2E_LINEAR_GAP)
        self.assertFalse(d2d["passed"])
        d2e = bps_replay_equivalence_gate(
            recomputed, stored, basis, vertices, selected,
            nearest_squared_distance_gap_m2_max=float("inf"),
            nearest_linear_distance_gap_m_max=D2E_LINEAR_GAP,
        )
        self.assertTrue(d2e["passed"])
        self.assertTrue(bool(d2e["tie"][0]))

    def test_d2e_config_and_fresh_holdout_are_locked(self):
        args = SimpleNamespace(
            run_id="p1-hoi-d2e-bps-linear-equivalence-r1-s42-20260715",
            output="/tmp/d2e.json",
            cuda_device="cuda:2",
        )
        config = d2e_resolved_config(args)
        self.assertEqual(config["selection"]["combined_windows"], 2496)
        self.assertEqual(config["selection"]["hashes"], D2E_EXPECTED_HASHES)
        self.assertEqual(config["gate"]["nearest_linear_distance_gap_m_max"], 2.5e-7)
        self.assertEqual(config["gate"]["nearest_squared_distance_gap_m2"], "report_only")
        self.assertFalse(config["sampler_stored_per_frame_bps"])
        self.assertFalse(config["sampler_future_gt"])
        self.assertEqual(config["checkpoint_count_loaded"], 0)
        self.assertEqual(config["training_updates"], 0)
        dataset = PriorWindowDataset(
            str(REPO), "hoi", partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )
        subsets, coverage = select_d2e_windows(dataset)
        self.assertEqual(
            set(subsets["disclosed_calibration"]) & set(subsets["fresh_holdout"]), set(),
        )
        self.assertEqual(len(subsets["disclosed_calibration"]), 1664)
        self.assertEqual(len(subsets["fresh_holdout"]), 832)
        self.assertTrue(all(set(values) == set(PLY_SHA256) for values in coverage.values()))
        self.assertTrue(all(value == 128 for value in coverage["disclosed_calibration"].values()))
        self.assertTrue(all(value == 64 for value in coverage["fresh_holdout"].values()))

    def test_d2c_selection_and_immutable_ply_hashes(self):
        dataset = PriorWindowDataset(
            str(REPO), "hoi", partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )
        positions, coverage = select_d2c_windows(dataset)
        self.assertEqual(len(positions), 832)
        self.assertEqual(set(coverage), set(PLY_SHA256))
        self.assertTrue(all(value == 64 for value in coverage.values()))
        self.assertEqual(
            selection_sha256(int(dataset.indices[position]) for position in positions),
            EXPECTED_GLOBAL_SELECTION_SHA256,
        )
        payload = "\n".join(
            f"{dataset.scene_names[int(dataset.sequence_ids[int(dataset.indices[position])])]}:"
            f"{int(dataset.indices[position])}"
            for position in positions
        )
        self.assertEqual(hashlib.sha256(payload.encode()).hexdigest(), EXPECTED_SEQUENCE_WINDOW_SHA256)
        assets = verify_d2c_assets()
        self.assertEqual(set(assets["rest_object_ply"]), set(PLY_SHA256))

    def test_d2c_known_cpu_mismatches_use_the_explicit_gate(self):
        dataset = PriorWindowDataset(
            str(REPO), "hoi", partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )
        global_indices = (426713, 511231, 25839, 182367, 186967)
        positions = [int(np.flatnonzero(dataset.indices == value)[0]) for value in global_indices]
        result = d2c_replay_device(dataset, positions, torch.device("cpu"))
        self.assertGreaterEqual(result["tie_basis_points"], 1)
        self.assertEqual(
            result["strict_basis_points"] + result["tie_basis_points"]
            + result["unexplained_basis_points"],
            5 * 1024,
        )
        for failure in result["failures"]:
            self.assertGreater(failure["nearest_squared_distance_gap_m2"], 1e-7)
        linear = d2c_replay_device(
            dataset,
            positions,
            torch.device("cpu"),
            nearest_squared_distance_gap_m2_max=float("inf"),
            nearest_linear_distance_gap_m_max=D2E_LINEAR_GAP,
            expected_windows_per_class=None,
            expected_object_classes=None,
        )
        self.assertTrue(linear["passed"])
        self.assertEqual(linear["unexplained_basis_points"], 0)


class WindowStateCodecTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(REPO), "hoi", partition="train", limit=3,
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )

    def test_real_round_trip_and_rotation_geodesic(self):
        item = self.dataset[0]
        frame = WindowFrame(
            item["window_origin"], item["world_to_local_rotation"],
            item["object_rotation_reference"],
        )
        decoded = self.dataset.codec.decode(item["x"], frame)
        index = int(item["window_index"])
        frames = np.arange(int(self.dataset.starts[index]), int(self.dataset.ends[index]), 3)
        expected_joints = torch.from_numpy(
            np.array(self.dataset.joints[frames], dtype=np.float32, copy=True)
        )
        expected_object = torch.from_numpy(
            np.array(self.dataset.object_trans[frames], dtype=np.float32, copy=True)
        )
        self.assertLessEqual(float((decoded["joints"] - expected_joints).abs().max()), 1e-5)
        self.assertLessEqual(float((decoded["object_translation"] - expected_object).abs().max()), 1e-5)
        expected_rotation = torch.from_numpy(
            np.array(self.dataset.object_rot[frames], dtype=np.float32, copy=True)
        )
        self.assertLessEqual(float(rotation_geodesic(decoded["object_rotation"], expected_rotation).max()), 1e-4)
        rebased, new_frame = self.dataset.codec.rebase(item["x"], frame)
        rebased_decoded = self.dataset.codec.decode(rebased, new_frame)
        self.assertLessEqual(float((rebased_decoded["joints"] - expected_joints).abs().max()), 1e-5)
        self.assertLessEqual(
            float(rotation_geodesic(rebased_decoded["object_rotation"], expected_rotation).max()), 1e-4,
        )

    def test_pelvis_endpoint_replays_legacy_omomo_semantics(self):
        item = self.dataset[0]
        index = int(item["window_index"])
        start, end = int(self.dataset.starts[index]), int(self.dataset.ends[index])
        frames = np.arange(start, end, 3)
        pelvis = np.array(self.dataset.joints[frames[-1], 0], dtype=np.float32, copy=True)
        origin = item["window_origin"].numpy()
        rotation = item["world_to_local_rotation"].numpy()
        legacy = (pelvis - origin) @ rotation.T
        legacy[1] = 0.0
        self.assertLessEqual(float(np.max(np.abs(item["goals"][:3].numpy() - legacy))), 1e-5)
        self.assertEqual(float(item["goals"][1]), 0.0)
        self.assertGreater(float(item["goals"][[0, 2]].abs().max()), 0.0)

    def test_terminal_mask_selects_one_final_window_per_training_sequence(self):
        final_dataset = PriorWindowDataset(
            str(REPO), "hoi", partition="train",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )
        indices = final_dataset.indices
        sequences = final_dataset.sequence_ids[indices]
        ends = final_dataset.ends[indices]
        terminal = ends == final_dataset.seq_ends[sequences] - 1
        self.assertEqual(int(terminal.sum()), len(set(sequences.tolist())))
        final_position = int(np.flatnonzero(terminal)[0])
        item = final_dataset[final_position]
        self.assertTrue(bool(item["terminal_window"]))
        self.assertEqual(float(item["progress"][1]), float(item["progress"][2]))

    def test_bps_replay_uses_fixed_basis_and_current_pose(self):
        item = self.dataset[0]
        sequence = str(self.dataset.scene_names[int(item["sequence_index"])])
        object_name = sequence.split("_")[1]
        mesh = trimesh.load_mesh(REPO / "data/object/rest_object_geo" / f"{object_name}.ply", process=False)
        rest = torch.from_numpy(zup_to_yup(np.asarray(mesh.vertices, dtype=np.float32).copy()))
        replay = self.dataset.codec.recompute_bps(rest, item["object_rotation_reference"])
        self.assertLessEqual(float((replay - item["object_bps"]).abs().max()), 1e-4)

    def test_projection_is_proper_so3(self):
        matrices = torch.randn(32, 3, 3)
        projected = project_to_so3(matrices)
        identity = torch.eye(3).expand(32, -1, -1)
        torch.testing.assert_close(projected @ projected.transpose(-1, -2), identity, atol=1e-5, rtol=1e-5)
        torch.testing.assert_close(torch.det(projected), torch.ones(32), atol=1e-5, rtol=1e-5)

    def test_reverse_x0_projection_restores_history_and_only_changes_object_rotation(self):
        clean = torch.randn(2, 16, 232)
        clean[:, 2:, 219:228] *= 4.0
        fixed = torch.randn(2, 2, 232)
        control = prepare_clean_x0(clean, fixed, object_so3_x0=False)
        projected = prepare_clean_x0(clean, fixed, object_so3_x0=True)
        torch.testing.assert_close(control[:, :2], fixed)
        torch.testing.assert_close(projected[:, :2], fixed)
        active = torch.ones(232, dtype=torch.bool)
        active[219:228] = False
        torch.testing.assert_close(projected[:, 2:, active], control[:, 2:, active])
        errors = object_rotation_manifold_error(projected)
        self.assertLessEqual(errors["orthogonality_frobenius_max"], 1e-5)
        self.assertLessEqual(errors["determinant_abs_error_max"], 1e-5)

    def test_gaussian_sampler_applies_projection_before_final_posterior_mean(self):
        class InvalidObjectModel(torch.nn.Module):
            def forward(self, noisy, timesteps, text, bps, goals, progress):
                del timesteps, text, bps, goals, progress
                value = torch.zeros_like(noisy)
                invalid = (torch.eye(3) * 2.0).reshape(1, 1, 9)
                value[..., 219:228] = invalid
                return value

        diffusion = GaussianDiffusion(500)
        model = InvalidObjectModel()
        fixed = torch.zeros(1, 2, 232)
        text = torch.zeros(1, 768)
        bps = torch.zeros(1, 1024, 3)
        goals = torch.zeros(1, 9)
        progress = torch.zeros(1, 3)
        control_generator = torch.Generator().manual_seed(42)
        projected_generator = torch.Generator().manual_seed(42)
        control = diffusion.sample(
            model, fixed, text, bps, goals, progress,
            generator=control_generator, object_so3_x0=False,
        )
        projected = diffusion.sample(
            model, fixed, text, bps, goals, progress,
            generator=projected_generator, object_so3_x0=True,
        )
        control_error = object_rotation_manifold_error(control)["orthogonality_frobenius_max"]
        projected_error = object_rotation_manifold_error(projected)["orthogonality_frobenius_max"]
        self.assertGreater(control_error, 1.0)
        # The float32 t=0 posterior coefficient is close to, but not exactly, one;
        # the evaluator's existing final handoff projection closes this residual.
        self.assertLess(projected_error, 1e-3)
        self.assertLess(projected_error, control_error)
        torch.testing.assert_close(projected[:, :2], fixed)

    def test_nonterminal_goal_loss_is_differentiable_zero(self):
        prediction = torch.zeros(2, 16, 232, requires_grad=True)
        target = prediction.detach().clone()
        # Keep valid identity object matrices for stable SVD gradients.
        identity = torch.eye(3).reshape(1, 1, 9).repeat(2, 16, 1)
        prediction.data[..., 219:228] = identity
        target[..., 219:228] = identity
        goals = torch.zeros(2, 9)
        goals[:, 6:9] = 1.0
        parents = torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long)
        losses = hoi_training_losses(
            prediction, target, goals, torch.zeros(2, 24, 3), parents,
            torch.full((3,), -2.0), torch.full((3,), 2.0),
            torch.full((3,), -2.0), torch.full((3,), 2.0),
            torch.zeros(2, dtype=torch.bool), torch.randn(2, 100, 3),
            torch.eye(3).repeat(2, 1, 1), torch.eye(3).repeat(2, 1, 1),
        )
        self.assertEqual(float(losses["object_goal"]), 0.0)
        gradient, = torch.autograd.grad(losses["object_goal"], prediction, retain_graph=True)
        self.assertEqual(float(gradient.abs().max()), 0.0)
        self.assertTrue(torch.isfinite(losses["object_surface"]))


if __name__ == "__main__":
    unittest.main()
