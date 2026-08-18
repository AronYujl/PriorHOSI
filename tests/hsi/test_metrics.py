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


class FixedSdf:
    """Geometry returning a prescribed ``[T, S]`` table, decoupled from height.

    A half-space is the cleanest closed form, but it pins ``sdf == y``, so
    penetration depth and height cannot be varied independently and *every*
    penetrating sample necessarily sits below the 2 cm floor-exclusion band.
    That makes ``pene_sum_*_floorexcl`` identically zero against it -- correct,
    but untestable.  This stand-in decouples the two, which is what lets the
    floor-excluded summed form be pinned in closed form at all.  It asserts its
    table matches the queried points, so a fixture that drifts out of shape
    fails loudly instead of silently broadcasting.
    """

    def __init__(self, table):
        self.sdf = torch.tensor(table, dtype=torch.float64)

    def signed_distance(self, points):
        queried = tuple(points.shape[:-1])
        if tuple(self.sdf.shape) != queried:
            raise AssertionError(
                "fixture SDF table %s does not match the points %s it was queried "
                "with" % (tuple(self.sdf.shape), queried)
            )
        return self.sdf.clone()


def _heights(rows) -> torch.Tensor:
    """``[T, S, 3]`` points whose only non-zero coordinate is the given height."""
    table = torch.tensor(rows, dtype=torch.float32)
    points = torch.zeros(table.shape[0], table.shape[1], 3)
    points[..., 1] = table
    return points


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


def _sliding_feet_xz(
    frames: int, foot_height: float, per_frame_dx: float, per_frame_dz: float
) -> torch.Tensor:
    """As :func:`_sliding_feet`, but the slide has both an x and a z component.

    This is the only fixture that can tell L1 from L2: on a pure +x slide the two
    horizontal magnitudes coincide, so an axis-aligned test cannot see which one
    the metric uses.
    """
    joints = _static_body(frames, height=foot_height + 1.0)
    for frame in range(frames):
        for joint in M.FOOT_JOINTS:
            joints[frame, joint, 1] = foot_height
            joints[frame, joint, 0] = per_frame_dx * frame
            joints[frame, joint, 2] = per_frame_dz * frame
    return joints


def _sliding_feet_at_heights(
    frames: int, heights, per_frame_dx: float
) -> torch.Tensor:
    """One height per foot joint (order ``M.FOOT_JOINTS``), all sliding along +x.

    Lets a single foot be put below ``y = 0`` while the others are parked out of
    the contact band, which is what isolates the height clamp.
    """
    joints = _static_body(frames, height=1.0)
    for frame in range(frames):
        for joint, height in zip(M.FOOT_JOINTS, heights):
            joints[frame, joint, 1] = float(height)
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

    # ------------------------------------------------------------------
    # The 2026-08-17 surface-threshold family: pene_pct_scene, pen_value,
    # pene_samples and the floor-excluded summed form.  Threshold 0, not
    # TeSMo's -3 cm, so every case below states which family it pins.
    # ------------------------------------------------------------------

    #: SDF and height for a [2, 4] body, deliberately decoupled.  Frame 1's
    #: sample 1 penetrates 20 cm while sitting 5 mm above the floor -- inside
    #: the 2 cm exclusion band -- which is the only configuration that separates
    #: the floor-excluded summed form from the unexcluded one.
    DECOUPLED_SDF = [[-0.3, 0.5, 0.5, 0.5],
                     [-0.1, -0.2, 0.5, 0.5]]
    DECOUPLED_Y = [[0.60, 1.0, 1.0, 1.0],
                   [0.40, 0.005, 1.0, 1.0]]

    def test_surface_threshold_family_against_a_half_space(self):
        points = torch.zeros(10, 20, 3)
        points[:, :, 1] = -0.05  # every sample 5 cm inside
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        self.assertAlmostEqual(result["pene_pct_scene"], 1.0, places=12)
        self.assertAlmostEqual(result["pen_value"], 0.05, places=7)
        # A count, not a fraction: pen_value's denominator, over sample-frames.
        self.assertEqual(result["pene_samples"], 200.0)
        self.assertEqual(result["pene_samples"], result["pen_sample_frames"])
        # Against a half-space at y = 0 every penetrating sample is by
        # construction below the 2 cm band, so the summed form is identically
        # zero.  That is the intended behaviour, not a defect: 67.4% of the
        # sealed GT penetration mass sits below y = 0 and is feet resting on the
        # ground plane rather than scene penetration.  pene_pct_scene above
        # still sees all of it, which is why the exclusion is applied to the
        # summed form only.
        self.assertEqual(result["pene_sum_mean_floorexcl"], 0.0)
        self.assertEqual(result["pene_sum_max_floorexcl"], 0.0)

    def test_the_two_penetration_thresholds_disagree_by_construction(self):
        points = _heights([[-0.05, -0.01, 0.10, 0.20]])
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        # -3 cm keeps one sample, the surface keeps two.  Reporting one family
        # only would force the choice that plan section B records as the
        # four-incompatible-quantities problem, so both are returned and each
        # carries its own count.
        self.assertEqual(result["pen_samples"], 1.0)
        self.assertEqual(result["pene_samples"], 2.0)
        self.assertAlmostEqual(result["pen_ratio"], 0.25, places=12)
        self.assertAlmostEqual(result["pene_pct_scene"], 0.50, places=12)
        self.assertAlmostEqual(result["pen_depth_mean"], 0.05, places=7)   # (0.05)/1
        self.assertAlmostEqual(result["pen_value"], 0.03, places=7)        # (0.05+0.01)/2

    def test_exact_surface_threshold_values_on_a_decoupled_geometry(self):
        points = _heights(self.DECOUPLED_Y)
        result = M.penetration_metrics(points, FixedSdf(self.DECOUPLED_SDF))
        # 3 of 8 sample-frames have sdf < 0.  A fraction in [0, 1] despite
        # LINGO's "%" in the column name.
        self.assertAlmostEqual(result["pene_pct_scene"], 3.0 / 8.0, places=12)
        self.assertEqual(result["pene_samples"], 3.0)
        self.assertEqual(result["pen_sample_frames"], 8.0)
        # mean |sdf| over those three only: (0.3 + 0.1 + 0.2) / 3.  Not / 8,
        # which would be 0.075, and not a signed mixture with the exterior.
        self.assertAlmostEqual(result["pen_value"], 0.2, places=12)
        # Per frame, sum of |sdf| over penetrating samples at y >= 2 cm, then
        # mean and max over frames.  Frame 0 contributes 0.3; frame 1
        # contributes 0.1 because its 0.2-deep sample sits in the floor band.
        self.assertAlmostEqual(result["pene_sum_mean_floorexcl"], 0.2, places=12)
        # The worst frame *total*, never the deepest single sample: the deepest
        # sample here is 0.3 in frame 0 and also 0.2 in frame 1, and the max of
        # the frame totals is 0.3, not pen_depth_max's 0.3-by-coincidence.
        self.assertAlmostEqual(result["pene_sum_max_floorexcl"], 0.3, places=12)

    def test_the_floor_exclusion_is_what_moves_the_summed_form(self):
        """Same fixture with the band at the floor itself: the 0.2-deep sample
        5 mm up is no longer excluded, so the mean rises 0.2 -> 0.3.  Without
        this the previous test could not tell the exclusion was applied."""
        points = _heights(self.DECOUPLED_Y)
        unexcluded = M.penetration_metrics(
            points, FixedSdf(self.DECOUPLED_SDF), floor_exclusion_height_m=0.0
        )
        self.assertAlmostEqual(unexcluded["pene_sum_mean_floorexcl"], 0.3, places=12)
        # The threshold-0 counting columns are indifferent to the band, by design.
        self.assertAlmostEqual(unexcluded["pene_pct_scene"], 3.0 / 8.0, places=12)
        self.assertEqual(unexcluded["pene_samples"], 3.0)

    def test_pen_value_is_nan_not_zero_when_nothing_penetrates(self):
        points = _heights([[0.40] * 5] * 3)
        result = M.penetration_metrics(points, HalfSpaceBelowZero())
        self.assertEqual(result["pene_samples"], 0.0)
        self.assertEqual(result["pene_pct_scene"], 0.0)
        # nan, so "no penetration" cannot be read off the table as "perfectly
        # shallow penetration"; the evaluator drops non-finite values from its
        # aggregates, and pene_samples = 0 is what tells a reader why.
        self.assertTrue(math.isnan(result["pen_value"]))
        # Deliberately asymmetric: the pen_* family's zero-fill is a sealed
        # semantic and is left alone.  This assertion exists so that the
        # asymmetry is a decision on the record rather than an accident.
        self.assertEqual(result["pen_depth_mean"], 0.0)
        self.assertEqual(result["pen_depth_max"], 0.0)

    def test_a_nonfinite_sdf_is_excluded_rather_than_poisoning_the_frame_sum(self):
        """A diverged rollout must cost visibility, not the whole column.

        The DIMOS summed form written literally as ``|min(sdf, 0)|`` propagates
        nan through ``torch.minimum``, which would turn one bad vertex into a
        nan frame total and a nan sequence mean.  Guarding on ``isfinite``
        instead keeps the rest of the frame measurable, so this pins the guard.
        """
        heights = [row + [1.0] for row in self.DECOUPLED_Y]
        control_table = [row + [0.5] for row in self.DECOUPLED_SDF]
        poisoned_table = [row + [0.5] for row in self.DECOUPLED_SDF]
        poisoned_table[0][4] = float("nan")

        points = _heights(heights)
        control = M.penetration_metrics(points, FixedSdf(control_table))
        poisoned = M.penetration_metrics(points, FixedSdf(poisoned_table))

        self.assertEqual(poisoned["nonfinite_ratio"], 1.0 / 10.0)
        self.assertEqual(control["nonfinite_ratio"], 0.0)
        for key in ("pene_sum_mean_floorexcl", "pene_sum_max_floorexcl",
                    "pene_samples", "pene_pct_scene", "pen_value",
                    "pen_ratio", "pen_samples"):
            self.assertEqual(poisoned[key], control[key], key)
        # And it is genuinely the same numbers as the 4-sample fixture: the
        # extra exterior column changes only the denominator-bearing keys.
        self.assertAlmostEqual(poisoned["pene_sum_mean_floorexcl"], 0.2, places=12)
        self.assertEqual(poisoned["pene_samples"], 3.0)


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
    # The 2026-08-18 revision made fs_nemf faithful to NeMF: L2 horizontal
    # magnitude, mean over the four foot joints, T-1 denominator, no
    # pre-translation, and the weight's height clamped to max(h, 0).  Every case
    # below pins one of those, because all four are silent changes -- the key
    # names and the unit are unchanged and only the numbers move.
    def test_nemf_exact_value_for_a_known_slide(self):
        frames, dx = 11, 0.01
        result = M.fs_nemf(_sliding_feet(frames, foot_height=0.0, per_frame_dx=dx))
        # Weight 1 at h = 0, all four feet sliding identically, mean over the four
        # joints and division by the T-1 transitions: the value collapses to the
        # per-frame slide in centimetres, independent of joint count and length.
        expected = dx * 100.0
        # places=7, not 12: the fixture is built in float32 (the dtype the real
        # pipeline hands over), so 0.01 m per frame arrives as 0.010000000149,
        # a 1.5e-8 relative offset that belongs to the fixture, not the metric.
        self.assertAlmostEqual(result["fs_nemf"], expected, places=7)
        # And it is the mean, not the sum: the summed form was 4x this.
        self.assertNotAlmostEqual(result["fs_nemf"], 4.0 * expected, places=6)
        # The parts attribute the total, so they must add up to it.
        self.assertAlmostEqual(result["fs_nemf_ankle"], expected / 2, places=7)
        self.assertAlmostEqual(result["fs_nemf_toe"], expected / 2, places=7)
        self.assertAlmostEqual(
            result["fs_nemf_ankle"] + result["fs_nemf_toe"], result["fs_nemf"], places=12
        )

    def test_nemf_averages_over_however_many_foot_joints_it_is_given(self):
        # The divisor is the actual foot-joint count, not a hard-coded 4: driving
        # the same fixture with one ankle and one toe must leave the mean of an
        # all-identical slide unchanged.
        frames, dx = 11, 0.01
        joints = _sliding_feet(frames, foot_height=0.0, per_frame_dx=dx)
        two_feet = M.fs_nemf(joints, ankle_joints=(7,), toe_joints=(10,))
        self.assertAlmostEqual(two_feet["fs_nemf"], dx * 100.0, places=7)
        self.assertAlmostEqual(two_feet["fs_nemf"], M.fs_nemf(joints)["fs_nemf"], places=12)

    def test_nemf_uses_the_l2_horizontal_magnitude_not_l1(self):
        # A 45-degree slide of the same per-axis step: L2 gives sqrt(2) x the
        # axis-aligned value, L1 would give exactly 2x.  This is the one deviation
        # a pure +x fixture cannot see.
        frames, step = 11, 0.01
        straight = M.fs_nemf(_sliding_feet(frames, 0.0, step))["fs_nemf"]
        diagonal = M.fs_nemf(_sliding_feet_xz(frames, 0.0, step, step))["fs_nemf"]
        self.assertAlmostEqual(diagonal, straight * math.sqrt(2.0), places=12)
        self.assertNotAlmostEqual(diagonal, straight * 2.0, places=6)

    def test_nemf_divides_by_transitions_so_length_cancels(self):
        # T-1, not T: a constant-velocity slide is the same skate whether it lasts
        # 11 frames or 41, and only the transition count makes that exact.
        short = M.fs_nemf(_sliding_feet(11, 0.0, 0.01))["fs_nemf"]
        long = M.fs_nemf(_sliding_feet(41, 0.0, 0.01))["fs_nemf"]
        self.assertEqual(short, long)

    def test_nemf_is_linear_in_slide_distance(self):
        slow = M.fs_nemf(_sliding_feet(11, 0.0, 0.01))["fs_nemf"]
        fast = M.fs_nemf(_sliding_feet(11, 0.0, 0.02))["fs_nemf"]
        self.assertAlmostEqual(fast, 2 * slow, places=6)

    def test_nemf_height_weight_follows_the_published_ramp(self):
        # Ankles at 4 cm with H = 8 cm give weight 2 - 2**0.5; the toes sit on the
        # floor and do not move, so the toe group contributes nothing.
        frames, dx = 11, 0.01
        joints = _static_body(frames, height=1.0)
        for frame in range(frames):
            for joint in M.TOE_JOINTS:
                joints[frame, joint, 1] = 0.0
            for joint in M.ANKLE_JOINTS:
                joints[frame, joint, 1] = 0.04
                joints[frame, joint, 0] = dx * frame
        weight = 2.0 - 2.0 ** (0.04 / M.NEMF_ANKLE_HEIGHT_M)
        expected = len(M.ANKLE_JOINTS) * dx * 100.0 * weight / len(M.FOOT_JOINTS)
        self.assertAlmostEqual(M.fs_nemf(joints)["fs_nemf_ankle"], expected, places=7)
        self.assertAlmostEqual(M.fs_nemf(joints)["fs_nemf_toe"], 0.0, places=12)
        self.assertAlmostEqual(weight, 2.0 - math.sqrt(2.0), places=9)

    def test_the_height_clamp_is_an_identity_when_no_foot_is_below_the_floor(self):
        # Every foot at or above y = 0, so max(h, 0) == h everywhere and the two
        # computations must agree *exactly*, not merely closely.  This is the
        # property that makes the clamp a safe deviation from published NeMF:
        # it can only touch sub-floor data.
        joints = _sliding_feet_at_heights(11, (0.0, 0.02, 0.01, 0.0), 0.01)
        clamped = M.fs_nemf(joints)
        raw = M.fs_nemf(joints, clamp_height_in_weight=False)
        self.assertGreater(clamped["fs_nemf"], 0.0)
        for key in ("fs_nemf", "fs_nemf_ankle", "fs_nemf_toe"):
            self.assertEqual(clamped[key], raw[key], key)

    def test_the_height_clamp_bites_on_a_sub_floor_foot(self):
        # One toe 5 cm below the floor, everything else parked at 1 m and so out
        # of every band.  Unclamped, the weight 2 - 2**(-1.25) > 1 scores the
        # penetrating foot as *more* skate at identical displacement, which is the
        # penetration/foot-skate coupling the clamp exists to remove.
        frames, dx = 11, 0.01
        joints = _sliding_feet_at_heights(frames, (1.0, 1.0, -0.05, 1.0), dx)
        clamped = M.fs_nemf(joints)["fs_nemf"]
        raw = M.fs_nemf(joints, clamp_height_in_weight=False)["fs_nemf"]
        one_foot = dx * 100.0 / len(M.FOOT_JOINTS)
        self.assertAlmostEqual(clamped, one_foot * 1.0, places=7)
        self.assertAlmostEqual(
            raw, one_foot * (2.0 - 2.0 ** (-0.05 / M.NEMF_TOE_HEIGHT_M)), places=7
        )
        self.assertLess(clamped, raw)

    def test_a_sub_floor_foot_is_still_counted_as_a_contact_frame(self):
        # Band membership uses the raw h, so h < 0 stays in the band and is scored
        # at the clamped weight 1 -- identical to a foot exactly on the floor.
        # Dropping it from the band instead would make a penetrating foot's slide
        # free, which is the opposite of the intent.
        frames, dx = 11, 0.01
        below = M.fs_nemf(_sliding_feet_at_heights(frames, (1.0, 1.0, -0.05, 1.0), dx))
        on_floor = M.fs_nemf(_sliding_feet_at_heights(frames, (1.0, 1.0, 0.0, 1.0), dx))
        self.assertGreater(below["fs_nemf"], 0.0)
        self.assertEqual(below["fs_nemf"], on_floor["fs_nemf"])

    def test_a_floating_slide_scores_zero_now_that_nothing_is_pre_translated(self):
        # The pre-translation removed in the 2026-08-18 revision was the only
        # thing that made a hovering rollout visible here, and it cost a 2.458x
        # deflation plus a reordering of sequences.  The blind spot is real and is
        # covered by other columns rather than by this one.
        floating = _sliding_feet(11, foot_height=0.9, per_frame_dx=0.01)
        grounded = _sliding_feet(11, foot_height=0.0, per_frame_dx=0.01)
        self.assertEqual(M.fs_nemf(floating)["fs_nemf"], 0.0)
        self.assertGreater(M.fs_nemf(grounded)["fs_nemf"], 0.0)
        # skate_ratio shares the blind spot by construction (absolute 5 cm gate),
        # so the foot columns alone cannot catch a floating rollout ...
        self.assertAlmostEqual(M.skate_ratio(floating, fps=30.0)["skate_ratio"], 0.0, places=12)
        # ... engagement is what does: contact collapses to zero.
        geometry = HalfSpaceBelowZero()
        self.assertAlmostEqual(
            M.engagement_metrics(floating, geometry)["contact_count"], 0.0, places=12
        )
        self.assertAlmostEqual(
            M.engagement_metrics(grounded, geometry)["contact_count"],
            float(len(M.FOOT_JOINTS)),
            places=12,
        )

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
