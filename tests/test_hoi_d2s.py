import inspect
import json
import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))
sys.path.insert(0, str(ROOT))

from priors.contact_guidance import author_hand_object_components
from priors.data import PriorWindowDataset
from priors.denoiser_response import (
    CHECKPOINT_SHA256,
    DIRECTIONS,
    GATE_RATIO_METRICS,
    PARENT_TIMESTEPS,
    PHASE_OFFSETS,
    PRIOR_ROLLOUT_OFFSETS,
    PROTECTED_GROUPS,
    RUN_ID,
    SCALES,
    SELECTION_SHA256,
    TARGET_TIMESTEPS,
    UPPER_ROTATION_JOINTS,
    author_components_per_sample,
    direction_update,
    fixed_mask_from_contact,
    mechanism_gate,
    scaled_candidate_batch,
    select_largest_eligible_scale,
    select_response_holdout,
    unpack_scale_major,
)
from priors.representation import REPRESENTATION
from priors.routed_guidance import upper_rotation_mask
from tools.diagnose_hoi_d2s import probe_parent_response, run_unguided_chunk
from tools.summarize_hoi_d2s import compact, validate_identity
from datasets.utils import get_smpl_parents
from tools.evaluate_hoi_remediation import global_goals, stack_frames


class _IdentityModel(torch.nn.Module):
    def forward(self, noisy, timesteps, text, bps, goals, progress):
        del timesteps, text, bps, goals, progress
        return noisy


class D2SSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest=(
                "experiments/splits/"
                "omomo_hoi_train_validation_seed42.json"
            ),
        )

    def test_selection_is_locked_deterministic_and_fresh(self):
        first = select_response_holdout(self.dataset)
        second = select_response_holdout(self.dataset)
        self.assertEqual(first["global_indices"], second["global_indices"])
        self.assertEqual(first["sha256"], SELECTION_SHA256)
        self.assertEqual(first["phase_offsets"], list(PHASE_OFFSETS))
        self.assertEqual(first["sequences"], 64)
        self.assertEqual(first["windows"], 192)
        self.assertTrue(set(PHASE_OFFSETS).isdisjoint(PRIOR_ROLLOUT_OFFSETS))
        for triple in first["triples"]:
            pi = [
                int(self.dataset.language["pi"][int(self.dataset.indices[position])])
                for position in triple
            ]
            self.assertEqual(pi, list(PHASE_OFFSETS))

    def test_parent_target_boundaries_include_both_endpoints(self):
        self.assertEqual(PARENT_TIMESTEPS, tuple(value + 1 for value in TARGET_TIMESTEPS))
        self.assertIn((0, 1), tuple(zip(TARGET_TIMESTEPS, PARENT_TIMESTEPS)))
        self.assertIn((498, 499), tuple(zip(TARGET_TIMESTEPS, PARENT_TIMESTEPS)))


class D2SFormulaAndRoutingTests(unittest.TestCase):
    def test_fixed_mask_per_sample_sum_matches_author_formula(self):
        torch.manual_seed(42)
        batch, frames, vertices = 3, 16, 32
        joints = torch.randn(batch, frames, 24, 3) + 2.0
        surface = torch.randn(batch, frames, vertices, 3) - 2.0
        translation = torch.randn(batch, frames, 3)
        rotation = torch.eye(3).reshape(1, 1, 3, 3).repeat(batch, frames, 1, 1)
        contact = torch.zeros(batch, frames, 4)
        contact[:, :, :2] = 1.0
        mask = fixed_mask_from_contact(contact)
        per_sample = author_components_per_sample(
            joints, surface, translation, rotation, mask,
        )
        aggregate = author_hand_object_components(
            joints, surface, translation, rotation, contact,
        )
        self.assertTrue(torch.allclose(
            per_sample["raw_total"].sum(), aggregate["total"],
            rtol=1e-6, atol=1e-6,
        ))
        self.assertTrue(torch.equal(mask, fixed_mask_from_contact(contact * 2.0)))

    def test_upper_direction_has_exact_support_and_history_zero(self):
        gradient = torch.ones(2, 16, REPRESENTATION.dimension)
        author = direction_update(gradient, "author_all")
        upper = direction_update(gradient, "upper_raw")
        mask = upper_rotation_mask()
        self.assertTrue(torch.equal(author[:, :2], torch.zeros_like(author[:, :2])))
        self.assertTrue(torch.equal(upper[:, 2:, mask], gradient[:, 2:, mask]))
        self.assertTrue(torch.equal(upper[:, 2:, ~mask], torch.zeros_like(upper[:, 2:, ~mask])))
        self.assertEqual(int(mask.sum()), len(UPPER_ROTATION_JOINTS) * 6)
        with self.assertRaises(ValueError):
            direction_update(gradient, "observed_best")

    def test_scale_major_candidate_batch_is_exact_and_restores_history(self):
        posterior = torch.randn(2, 16, REPRESENTATION.dimension)
        update = torch.randn_like(posterior)
        fixed = torch.randn(2, 2, REPRESENTATION.dimension)
        packed = scaled_candidate_batch(posterior, update, fixed)
        values = unpack_scale_major(packed, 2)
        self.assertEqual(values.shape[0], len(SCALES))
        for index, scale in enumerate(SCALES):
            self.assertTrue(torch.equal(values[index, :, :2], fixed))
            self.assertTrue(torch.allclose(
                values[index, :, 2:],
                posterior[:, 2:] + float(scale) * update[:, 2:],
            ))

    def test_largest_eligible_scale_and_zero_denominator_rule(self):
        batch = 3
        baseline_loss = torch.full((batch,), 10.0)
        losses = torch.full((len(SCALES), batch), 11.0)
        losses[-1] = baseline_loss
        losses[0, 0] = 9.0
        losses[0, 1] = 9.0
        losses[1, 1] = 9.0
        natural = torch.full((batch, 16, REPRESENTATION.dimension), 4.0)
        natural[2] = 0.0
        responses = torch.full(
            (len(SCALES), batch, 16, REPRESENTATION.dimension), 2.0,
        )
        responses[-1] = 0.0
        responses[0, 0] = 0.5
        responses[1, 1] = 0.5
        selected = select_largest_eligible_scale(
            baseline_loss, losses, natural, responses,
        )
        self.assertEqual(selected["selected_index"].tolist(), [0, 1, len(SCALES) - 1])
        self.assertEqual(selected["selected_scale"].tolist(), [1.0, 0.5, 0.0])
        self.assertTrue(selected["largest_eligible_replay"].all())
        self.assertTrue(selected["selected_is_eligible"].all())
        self.assertFalse(selected["eligible"][0, 2])
        self.assertEqual(set(selected["group_passes"]), set(PROTECTED_GROUPS))

    def test_probe_reports_all_directions_scales_and_restores_history(self):
        dataset = PriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
        )
        position = 0
        item = dataset[position]
        target = item["x"][None]
        frame = stack_frames([item], torch.device("cpu"))
        pelvis_goal, object_goal = global_goals(
            dataset, [item], frame, torch.device("cpu"),
        )
        result = probe_parent_response(
            _IdentityModel(),
            dataset,
            target.clone(),
            target.clone(),
            target[:, :2].clone(),
            item["text_embedding"][None],
            item["object_bps"][None],
            item["goals"][None],
            item["progress"][None],
            target,
            frame,
            item["rest_human_offsets"][None],
            torch.from_numpy(get_smpl_parents(use_joints24=True).copy()).long(),
            item["rest_object_points"][None],
            pelvis_goal,
            object_goal,
            ["test_sequence"],
            [position],
            parent_timestep=1,
        )
        self.assertEqual(result["target_timestep"], 0)
        self.assertEqual(set(result["directions"]), set(DIRECTIONS))
        for direction in DIRECTIONS:
            self.assertEqual(
                set(result["directions"][direction]["scale_records"]),
                {f"{scale:g}" for scale in SCALES},
            )
            self.assertEqual(len(result["directions"][direction]["selected_records"]), 1)
        self.assertLessEqual(result["history_max_abs"], 1e-7)
        self.assertTrue(result["finite"])


class D2SGateAndGovernanceTests(unittest.TestCase):
    @staticmethod
    def timestep(passing=True):
        lower = 0.01 if passing else -0.01
        upper = 1.04 if passing else 1.06
        return {
            "nonzero_selected_fraction": 0.75 if passing else 0.25,
            "controller_comparison": {
                "fixed_mask_author_loss": {"bootstrap_95_ci": [lower, 0.2]},
                "fk_union_5cm": {
                    "recall": {"bootstrap_95_ci": [lower, 0.2]},
                    "f1": {"bootstrap_95_ci": [lower, 0.2]},
                },
                "ratios": {
                    metric: {"bootstrap_95_ci": [0.95, upper]}
                    for metric in GATE_RATIO_METRICS
                },
            },
        }

    def test_gate_requires_four_of_five_low_timesteps(self):
        four = {
            str(timestep): self.timestep(index < 4)
            for index, timestep in enumerate((0, 1, 10, 50, 100))
        }
        positive = mechanism_gate({"complete": True}, four)
        self.assertEqual(
            positive["classification"],
            "denoiser-response-frontier-positive-stop",
        )
        three = {
            str(timestep): self.timestep(index < 3)
            for index, timestep in enumerate((0, 1, 10, 50, 100))
        }
        negative = mechanism_gate({"complete": True}, three)
        self.assertEqual(
            negative["classification"],
            "denoiser-response-frontier-negative-stop",
        )
        failure = mechanism_gate({"complete": False}, four)
        self.assertEqual(
            failure["classification"],
            "denoiser-response-frontier-contract-failure-stop",
        )
        for value in (positive, negative, failure):
            self.assertFalse(value["full_trajectory_controller_authorized"])
            self.assertFalse(value["training_started"])

    def test_selection_signature_and_runner_keep_gt_out_of_controller(self):
        selection_source = inspect.getsource(select_largest_eligible_scale)
        self.assertNotIn("target", inspect.signature(select_largest_eligible_scale).parameters)
        self.assertNotIn("target", selection_source)
        probe_source = inspect.getsource(probe_parent_response)
        self.assertLess(
            probe_source.index("select_largest_eligible_scale("),
            probe_source.index("expanded_target ="),
        )
        sampler_source = inspect.getsource(run_unguided_chunk)
        self.assertIn("diffusion.posterior_sample(", sampler_source)
        self.assertIn("current = posterior", sampler_source)
        self.assertNotIn('item["object_bps"]', sampler_source)
        self.assertIn("current_bps(", sampler_source)

    def test_plan_and_registry_lock_run_inputs_and_stop(self):
        plan = (ROOT / "docs/EXPERIMENT_PLAN.md").read_text(encoding="utf-8")
        self.assertIn(RUN_ID, plan)
        self.assertIn(CHECKPOINT_SHA256, plan)
        self.assertIn("denoiser-response-frontier-positive-stop", plan)
        records = [
            json.loads(line)
            for line in (ROOT / "experiments/registry.jsonl").read_text(
                encoding="utf-8",
            ).splitlines()
            if line.strip()
        ]
        prereg = [
            record for record in records
            if record["experiment_id"]
            == "p1-hoi-d2s-denoiser-response-frontier-preregister-s42-20260717"
        ]
        self.assertEqual(len(prereg), 1)
        self.assertEqual(prereg[0]["config"]["run_id"], RUN_ID)
        self.assertEqual(prereg[0]["config"]["checkpoint"]["sha256"], CHECKPOINT_SHA256)
        self.assertFalse(prereg[0]["config"]["training_authorized"])
        self.assertFalse(prereg[0]["config"]["full_trajectory_controller_authorized"])

    def test_summary_requires_identity_and_removes_raw_records(self):
        metrics = {"run_id": RUN_ID, "git_commit": "abc", "status": "completed"}
        manifest = {"experiment_id": RUN_ID, "git": {"commit": "abc"}}
        validate_identity(metrics, manifest)
        with self.assertRaises(ValueError):
            validate_identity({**metrics, "status": "failed"}, manifest)
        value = compact({"timesteps": {"0": {"raw": {"records": [1]}}}, "keep": 2})
        self.assertEqual(value, {"timesteps": {"0": {}}, "keep": 2})


if __name__ == "__main__":
    unittest.main()
