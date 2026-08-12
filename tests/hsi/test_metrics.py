"""HSI native-domain metric tests.

Every case here has a closed-form expected value.  That is deliberate: the
survey behind ``docs/plan/PHASE_1C_HSI.md`` (2026-08-12 同日修订, subsection B)
found that same-named metrics in this literature are mutually incompatible
quantities -- the same LINGO baseline is published as ``Pene_mean`` 0.402,
0.421, 0.392 and 1397 -- so a test that merely checks shapes would not tell us
which quantity we compute.  Geometry is supplied by analytic stand-ins rather
than a real scene field, because a half-space has an exact signed distance and a
LINGO scene takes two minutes to voxelize.
"""

import math
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi import metrics as M


JOINTS = 28


class HalfSpaceBelowZero:
    """Occupied where y <= 0, so the signed distance of a point is simply its y."""

    def signed_distance(self, points):
        return points[..., 1].clone()


def _static_body(frames: int, height: float = 1.0) -> torch.Tensor:
    joints = torch.zeros(frames, JOINTS, 3)
    joints[:, :, 1] = height
    return joints


def _sliding_feet(frames: int, foot_height: float, per_frame_dx: float) -> torch.Tensor:
    """Body held high, all four foot joints at one height, sliding along +x."""
    joints = _static_body(frames, height=foot_height + 1.0)
    for frame in range(frames):
        for joint in M.FOOT_JOINTS:
            joints[frame, joint, 1] = foot_height
            joints[frame, joint, 0] = per_frame_dx * frame
    return joints


def _windows_from(sequence: torch.Tensor, window: int = 16, history: int = 2):
    """Cut one continuous sequence into windows that share `history` frames."""
    stride = window - history
    windows = []
    start = 0
    while start + window <= sequence.shape[0]:
        windows.append(sequence[start : start + window].clone())
        start += stride
    return windows


class StitchTests(unittest.TestCase):
    def test_stitch_drops_exactly_the_overlapping_history_frames(self):
        sequence = torch.arange(44, dtype=torch.float32).view(44, 1, 1).repeat(1, JOINTS, 3)
        stitched = M.stitch_windows(_windows_from(sequence))
        self.assertEqual(stitched.history_frames, 2)
        self.assertEqual(stitched.frames.shape[0], 44)
        self.assertEqual(stitched.window_lengths, (16, 14, 14))
        self.assertEqual(stitched.seams, (16, 30))
        # Frame identity, not merely the right length: a wrong drop would still
        # produce 44 frames while silently duplicating two of them.
        torch.testing.assert_close(stitched.frames, sequence)

    def test_stitch_rejects_a_discontinuous_overlap_when_asked(self):
        sequence = torch.arange(44, dtype=torch.float32).view(44, 1, 1).repeat(1, JOINTS, 3)
        windows = _windows_from(sequence)
        windows[1] = windows[1] + 5.0  # its history no longer matches window 0's tail
        with self.assertRaises(ValueError):
            M.stitch_windows(windows, overlap_atol=1e-6)

    def test_overlap_checking_is_opt_in(self):
        """Documented on purpose: a sampler may regenerate its conditioning frames."""
        sequence = torch.arange(44, dtype=torch.float32).view(44, 1, 1).repeat(1, JOINTS, 3)
        windows = _windows_from(sequence)
        windows[1] = windows[1] + 5.0
        stitched = M.stitch_windows(windows)  # no atol -> accepted
        self.assertEqual(stitched.frames.shape[0], 44)
        # A rollout driver that must guarantee frame identity should pass an atol.
        M.stitch_windows(_windows_from(sequence), overlap_atol=1e-6)



class PenetrationTests(unittest.TestCase):
    def _samples(self, frames, per_frame_depths):
        points = torch.zeros(frames, len(per_frame_depths[0]), 3)
        for frame, depths in enumerate(per_frame_depths):
            points[frame, :, 1] = torch.tensor(depths)
        return points

    def test_exact_values_against_a_half_space(self):
        points = torch.zeros(10, 20, 3)
        points[:, :, 1] = -0.05  # every sample 5 cm inside
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        self.assertAlmostEqual(result["pen_ratio"], 1.0, places=9)
        self.assertAlmostEqual(result["pen_depth_mean"], 0.05, places=7)
        self.assertAlmostEqual(result["pen_depth_max"], 0.05, places=7)

    def test_threshold_excludes_penetration_shallower_than_three_centimetres(self):
        points = torch.zeros(10, 20, 3)
        points[:, :, 1] = -0.02  # 2 cm inside: real, but below the reporting threshold
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        self.assertEqual(result["pen_ratio"], 0.0)

    def test_depth_mean_averages_over_penetrating_samples_only(self):
        points = torch.zeros(1, 4, 3)
        points[0, :, 1] = torch.tensor([-0.05, -0.05, 0.10, 0.20])
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        self.assertAlmostEqual(result["pen_ratio"], 0.5, places=9)
        # 0.05, not 0.025 (all samples) and not a signed mixture with the exterior.
        self.assertAlmostEqual(result["pen_depth_mean"], 0.05, places=7)

    def test_burst_is_the_mean_of_squares_and_is_superlinear(self):
        # Frame 0 fully penetrating, frame 1 clean: mean fraction 0.5.
        points = torch.zeros(2, 4, 3)
        points[0, :, 1] = -0.05
        points[1, :, 1] = 0.10
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        self.assertAlmostEqual(result["pen_burst"], 100.0 * (1.0 ** 2 + 0.0 ** 2) / 2, places=9)
        # The squaring is the reason this metric exists: one catastrophic frame
        # must not be diluted the way a plain mean would dilute it.
        self.assertGreater(result["pen_burst"], 100.0 * (0.5 ** 2))


class EngagementTests(unittest.TestCase):
    def test_contact_count_scales_with_how_much_of_the_body_touches(self):
        geometry = HalfSpaceBelowZero()
        few = torch.zeros(1, 10, 3)
        few[0, :2, 1] = 0.01           # 2 samples inside the +5 cm band
        few[0, 2:, 1] = 0.50
        many = torch.zeros(1, 10, 3)
        many[0, :8, 1] = 0.01          # 8 samples in the band
        many[0, 8:, 1] = 0.50
        self.assertAlmostEqual(M.engagement_metrics(few, geometry)["contact_count"], 2.0, places=9)
        self.assertAlmostEqual(M.engagement_metrics(many, geometry)["contact_count"], 8.0, places=9)

    def test_the_binary_form_is_labelled_saturated(self):
        points = torch.zeros(1, 4, 3)
        points[0, :, 1] = 0.01
        keys = M.engagement_metrics(points, HalfSpaceBelowZero())
        # GT saturates the binary form at 0.996-0.9996, so it may exist as a
        # diagnostic but its name must stop anyone reporting it as engagement.
        binary = [key for key in keys if "frame_ratio" in key]
        self.assertTrue(binary)
        for key in binary:
            self.assertIn("saturated", key)


class FootTests(unittest.TestCase):
    def test_nemf_exact_value_for_a_known_slide(self):
        frames, dx = 11, 0.01
        result = M.fs_nemf(_sliding_feet(frames, foot_height=0.0, per_frame_dx=dx))
        # 4 foot joints x 10 transitions x 1 cm, weight 1 at h=0, divided by T.
        expected = len(M.FOOT_JOINTS) * (frames - 1) * dx * 100.0 / frames
        self.assertAlmostEqual(result["fs_nemf"], expected, places=6)
        self.assertAlmostEqual(result["fs_nemf_ankle"], expected / 2, places=6)
        self.assertAlmostEqual(result["fs_nemf_toe"], expected / 2, places=6)

    def test_nemf_is_linear_in_slide_distance(self):
        slow = M.fs_nemf(_sliding_feet(11, 0.0, 0.01))["fs_nemf"]
        fast = M.fs_nemf(_sliding_feet(11, 0.0, 0.02))["fs_nemf"]
        self.assertAlmostEqual(fast, 2 * slow, places=6)

    def test_nemf_height_weight_follows_the_published_ramp(self):
        # Ankles at 4 cm with H = 8 cm give weight 2 - 2**0.5; toes sit on the
        # floor so the min-foot-height translation is a no-op.
        frames, dx = 11, 0.01
        joints = _static_body(frames, height=1.0)
        for frame in range(frames):
            for joint in M.TOE_JOINTS:
                joints[frame, joint, 1] = 0.0
            for joint in M.ANKLE_JOINTS:
                joints[frame, joint, 1] = 0.04
                joints[frame, joint, 0] = dx * frame
        weight = 2.0 - 2.0 ** (0.04 / M.NEMF_ANKLE_HEIGHT_M)
        expected = len(M.ANKLE_JOINTS) * (frames - 1) * dx * 100.0 * weight / frames
        self.assertAlmostEqual(M.fs_nemf(joints)["fs_nemf_ankle"], expected, places=6)
        self.assertAlmostEqual(weight, 2.0 - math.sqrt(2.0), places=9)

    def test_translation_is_what_stops_a_floating_slide_scoring_zero(self):
        floating = _sliding_feet(11, foot_height=0.9, per_frame_dx=0.01)
        grounded = _sliding_feet(11, foot_height=0.0, per_frame_dx=0.01)
        self.assertAlmostEqual(
            M.fs_nemf(floating)["fs_nemf"], M.fs_nemf(grounded)["fs_nemf"], places=6
        )
        # Without it, a sequence that slides 90 cm in the air is scored perfect.
        self.assertEqual(M.fs_nemf(floating, translate_to_min_foot_height=False)["fs_nemf"], 0.0)

    def test_skate_ratio_gates_on_height_and_speed(self):
        fast = _sliding_feet(11, foot_height=0.02, per_frame_dx=0.03)  # 0.9 m/s at 30 fps
        slow = _sliding_feet(11, foot_height=0.02, per_frame_dx=0.02)  # 0.6 m/s at 30 fps
        airborne = _sliding_feet(11, foot_height=0.30, per_frame_dx=0.03)
        self.assertAlmostEqual(M.skate_ratio(fast, fps=30.0)["skate_ratio"], 1.0, places=9)
        self.assertAlmostEqual(M.skate_ratio(slow, fps=30.0)["skate_ratio"], 0.0, places=9)
        # Above the contact height no foot is planted, so nothing counts as skating.
        self.assertAlmostEqual(M.skate_ratio(airborne, fps=30.0)["skate_ratio"], 0.0, places=9)

    def test_skate_ratio_is_frame_rate_invariant(self):
        # The same physical motion: 0.9 m/s along +x with feet 2 cm off the floor.
        at_30 = _sliding_feet(11, foot_height=0.02, per_frame_dx=0.9 / 30.0)
        at_60 = _sliding_feet(21, foot_height=0.02, per_frame_dx=0.9 / 60.0)
        self.assertAlmostEqual(
            M.skate_ratio(at_30, fps=30.0)["skate_ratio"],
            M.skate_ratio(at_60, fps=60.0)["skate_ratio"],
            places=9,
        )
        # And the threshold really is a speed: halving fps at fixed per-frame
        # displacement halves the speed and must cross back under the gate.
        borderline = _sliding_feet(11, foot_height=0.02, per_frame_dx=0.03)
        self.assertAlmostEqual(M.skate_ratio(borderline, fps=30.0)["skate_ratio"], 1.0, places=9)
        self.assertAlmostEqual(M.skate_ratio(borderline, fps=15.0)["skate_ratio"], 0.0, places=9)


class GoalTests(unittest.TestCase):
    def _walk_then_drift(self):
        frames = 21
        joints = _static_body(frames, height=1.0)
        for frame in range(frames):
            # Reach x = 1.0 at frame 10, then continue past it to x = 2.0.
            joints[frame, :, 0] = 0.1 * frame
        return joints

    def test_min_and_last_diverge_when_the_model_drifts_away(self):
        result = M.goal_metrics(self._walk_then_drift(), torch.tensor([1.0, 0.0, 0.0]), fps=30.0)
        self.assertLess(result["min_dist"], 1e-6)
        self.assertAlmostEqual(result["last_dist"], 1.0, places=6)
        # Reporting only one of them would hide this failure mode entirely.
        self.assertGreater(result["last_dist"], result["min_dist"])

    def test_success_is_reported_at_both_thresholds(self):
        result = M.goal_metrics(self._walk_then_drift(), torch.tensor([1.0, 0.0, 0.0]), fps=30.0)
        for threshold in ("10cm", "20cm"):
            self.assertIn(f"success_min_{threshold}", result)
            self.assertIn(f"time_to_goal_{threshold}_s", result)
        self.assertEqual(result["success_min_10cm"], 1.0)
        self.assertEqual(result["success_last_10cm"], 0.0)

    def test_distance_is_horizontal_only(self):
        joints = _static_body(5, height=1.0)
        raised = _static_body(5, height=9.0)
        goal = torch.tensor([0.0, 0.0, 0.0])
        self.assertAlmostEqual(
            M.goal_metrics(joints, goal, fps=30.0)["last_dist"],
            M.goal_metrics(raised, goal, fps=30.0)["last_dist"],
            places=9,
        )

    def test_decomposition_separates_planar_from_height(self):
        roots = torch.zeros(5, 3)
        roots[:, 0] = 3.0
        roots[:, 1] = 0.5
        result = M.goal_error_decomposition(roots, torch.tensor([0.0, 0.0, 0.0]))
        self.assertAlmostEqual(result["goal_planar_err_m"], 3.0, places=6)
        self.assertAlmostEqual(result["goal_height_err_m"], 0.5, places=6)


class SeamTests(unittest.TestCase):
    def _smooth(self, length=44):
        time = torch.arange(length, dtype=torch.float32)
        base = torch.sin(time * 0.35)
        return base.view(length, 1, 1).repeat(1, JOINTS, 3)

    def _kinked(self, length=44, seam=16, kink=0.5):
        sequence = self._smooth(length).clone()
        ramp = torch.zeros(length)
        ramp[seam:] = kink * torch.arange(1, length - seam + 1, dtype=torch.float32)
        sequence = sequence + ramp.view(length, 1, 1)
        return sequence

    def test_ratio_is_about_one_when_the_seam_is_smooth(self):
        stitched = M.stitch_windows(_windows_from(self._smooth()))
        ratio = M.jerk_metrics(stitched, fps=30.0)["jerk_ratio"]
        self.assertLess(abs(ratio - 1.0), 0.6, f"smooth seam should not look irregular, got {ratio}")

    def test_ratio_rises_when_a_discontinuity_is_injected_at_the_seam(self):
        smooth = M.jerk_metrics(M.stitch_windows(_windows_from(self._smooth())), fps=30.0)
        kinked = M.jerk_metrics(M.stitch_windows(_windows_from(self._kinked())), fps=30.0)
        self.assertGreater(kinked["jerk_ratio"], smooth["jerk_ratio"])
        self.assertGreater(kinked["jerk_ratio"], 1.0)

    def test_ratio_cannot_be_improved_by_smoothing_the_whole_motion(self):
        """Its self-normalising form is the reason it was chosen over raw jerk."""
        kinked = self._kinked()
        full = M.jerk_metrics(M.stitch_windows(_windows_from(kinked)), fps=30.0)
        damped = M.jerk_metrics(M.stitch_windows(_windows_from(kinked * 0.25)), fps=30.0)
        self.assertLess(damped["boundary_jerk"], full["boundary_jerk"])
        self.assertAlmostEqual(damped["jerk_ratio"], full["jerk_ratio"], places=4)

    def test_transition_distance_alignment_removes_a_constant_offset(self):
        stitched = M.stitch_windows(_windows_from(self._kinked()))
        result = M.transition_distance(stitched)
        self.assertGreater(result["transition_distance_unaligned"], 0.0)
        self.assertLessEqual(result["transition_distance_aligned"], result["transition_distance_unaligned"])
        self.assertEqual(result["transition_seams"], 2)


class ReactionDivergenceTests(unittest.TestCase):
    def test_rds_is_the_mean_per_joint_distance_between_the_paired_runs(self):
        with_scene = _static_body(7)
        without_scene = with_scene.clone()
        without_scene[:, :, 0] += 0.3  # a pure 30 cm shift on every joint
        result = M.reaction_divergence_score(with_scene, without_scene)
        self.assertAlmostEqual(result["rds"], 0.3, places=6)

    def test_rds_is_zero_when_the_scene_condition_changes_nothing(self):
        body = _static_body(7)
        # A model that ignores the scene scores 0 no matter how clean its
        # penetration is; that is why this sits beside the penetration columns.
        self.assertAlmostEqual(M.reaction_divergence_score(body, body.clone())["rds"], 0.0, places=9)


if __name__ == "__main__":
    unittest.main()
