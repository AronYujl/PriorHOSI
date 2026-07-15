import hashlib
import unittest
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import trimesh
from pytorch3d import transforms

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from datasets.utils import get_smpl_parents, zup_to_yup
from priors.data import PriorWindowDataset
from priors.losses import hoi_training_losses
from priors.remediation import (
    bps_replay_equivalence_gate,
    deterministic_derangement,
    field_squared_error,
    select_internal_triples,
    select_teacher_windows,
    selection_sha256,
)
from priors.window_codec import (
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
from tools.diagnose_hoi_d2p import (
    D0_T499,
    classify_mechanism,
    field_error_per_sample,
    resolved_config as d2p_resolved_config,
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
            run_id="p1-hoi-d2p-mechanism-s42-20260715",
            teacher_batch_size=64,
            device="cuda:0",
            output="/tmp/d2p.json",
        )
        config = d2p_resolved_config(args)
        self.assertFalse(config["official_test_used"])
        self.assertFalse(config["chois_used"])
        self.assertFalse(config["checkpoint_selection"])
        self.assertEqual(config["training_updates"], 0)
        self.assertEqual(config["teacher"]["windows"], 512)
        self.assertEqual(config["reverse_trace"]["sequences"], 32)

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
