"""Phase 1C HSI evaluation metrics: pure functions over arrays.

Every formula, threshold, unit and aggregation order here is pinned by
``docs/plan/PHASE_1C_HSI.md``, section **C** of the 2026-08-12 同日修订
("几何口径纠正与指标公式定版").  That section exists because of the survey
finding recorded in its section B: in this literature ``Pene_mean`` and friends
are *at least four mutually incompatible quantities sharing one symbol* -- the
same LINGO baseline is published as 0.402 / 0.421 / 0.392 / 1397.  So this
module never says "following DIMOS/LINGO"; it writes the expression, the
aggregation order, the unit and the sign convention, and any deviation from a
cited paper is called out in the docstring of the function that deviates.

Scope
=====
Pure functions over arrays.  No file I/O, no SMPL-X forward pass, no rollout
driver, no config loading -- those belong to a later, not-yet-approved stage.
Scene geometry arrives through the :class:`SceneGeometry` protocol, which is
duck-typed on purpose: nothing here imports ``priors.hsi.scene_field``, so the
tests can drive every geometric metric with a closed-form half-space or box.

Per-sequence, never aggregated
==============================
Every function scores **one** sequence and returns a flat ``Dict[str, float]``,
which is exactly the record shape ``tools/paired_bootstrap.py`` consumes
(``{sequence_name: {metric: number}}``, see ``load_per_sequence``).  Nothing
here averages across sequences or scenes, because section C requires ``scene``
to remain available as a blocking factor: GT per-frame penetration measures
30.8% / 94.7% / 76.1% across three LINGO scenes, so an aggregate is driven more
by which scenes the split contains than by the model.

Frames, floor and units
=======================
* Positions are metres in the LINGO world frame, y-up, ``[T, J, 3]``.
* The floor is **exactly** ``y = 0`` (plan section C: "地面 y = 0（LINGO 世界系中
  精确，无需估计）").  Nothing in this module estimates a floor height, and
  ``code/eval_metrics.py:determine_floor_height_and_contacts`` (DBSCAN over
  static foot heights) is deliberately not used.  The single exception is
  :func:`fs_nemf`, whose pinned NeMF definition pre-translates the sequence so
  that the *minimum foot height* is 0; that is part of the metric's definition,
  not a floor estimate, and it is an explicit keyword argument.
* Frame rate is **never** a module-level constant on a hot path.  Every metric
  whose value depends on it takes ``fps`` as a required keyword argument, because
  the GMD/TeSMo 2.5 cm-per-frame skating threshold is *not* frame-rate invariant
  and must be applied as 0.75 m/s (plan section C, 足部 item 2).

Rollout hygiene
===============
From the second window onward the ``history_frames`` overlapping frames are
dropped **before any metric is computed** (DIMOS's ``start_frame = 2``), and
every temporal metric is computed on the **stitched** sequence, never per window
and then averaged.  :func:`stitch_windows` is the only way to produce a
:class:`StitchedSequence`, and the temporal metrics reject a raw window stack by
shape with a message pointing back here, so the mistake is structurally hard to
make rather than merely discouraged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

try:  # Protocol is stdlib from 3.8; the fallback keeps the import cheap and total.
    from typing import Protocol
except ImportError:  # pragma: no cover - Python < 3.8 is not supported anyway
    Protocol = object  # type: ignore[assignment,misc]

import torch

__all__ = [
    "SceneGeometry",
    "StitchedSequence",
    "stitch_windows",
    "penetration_metrics",
    "engagement_metrics",
    "reaction_divergence_score",
    "reachability_diagnostic",
    "fs_nemf",
    "skate_ratio",
    "goal_metrics",
    "goal_error_decomposition",
    "jerk_metrics",
    "transition_distance",
]


# --------------------------------------------------------------------------
# Pinned constants.  Each carries the plan clause that fixed it.
# --------------------------------------------------------------------------

#: DIMOS's ``start_frame = 2`` for a 2-frame motion primitive (plan C, 接缝 item 3).
DEFAULT_HISTORY_FRAMES = 2

#: TeSMo's penetration threshold: SDF < -3 cm (plan C, 穿透 item 1).
PENETRATION_THRESHOLD_M = -0.03

#: Engagement band: samples within +5 cm of the surface (plan C, engagement item 1).
CONTACT_BAND_M = 0.05

#: NeMF/HuMoR height thresholds (plan C, 足部 item 1).
NEMF_ANKLE_HEIGHT_M = 0.08
NEMF_TOE_HEIGHT_M = 0.04

#: GMD/TeSMo skating thresholds, with the slide restated as a speed (plan C, 足部 item 2).
SKATE_CONTACT_HEIGHT_M = 0.05
SKATE_SPEED_MPS = 0.75

#: InfBaGel's 10 cm and the DIMOS/LINGO lineage's 20 cm (plan C, 目标到达 item 2).
GOAL_THRESHOLDS_M: Tuple[float, ...] = (0.10, 0.20)

#: SMPL body joint ids, matching ``code/eval_metrics.py:186-190`` so the HSI foot
#: numbers index the same joints as the HOI table does.
ANKLE_JOINTS: Tuple[int, ...] = (7, 8)
TOE_JOINTS: Tuple[int, ...] = (10, 11)
FOOT_JOINTS: Tuple[int, ...] = (7, 8, 10, 11)
PELVIS_JOINT = 0

_EPS = 1e-12


class SceneGeometry(Protocol):
    """The geometry interface the Tier-1 metrics query.

    Structural, not nominal: any object with these members works, and this module
    never imports a concrete implementation.  ``priors.hsi.scene_field.SceneField``
    is the production one; the tests use analytic half-spaces and boxes whose
    signed distance is known in closed form, which is strictly better evidence
    than agreeing with another implementation.

    ``signed_distance(points_world)``
        ``[..., 3]`` metres -> ``[...]`` metres.  **NEGATIVE = inside** occupied
        geometry.  Per plan section A the primary source is the scene mesh, and
        per section A's out-of-bbox rule the value outside the world bbox is
        positive and never clipped.
    ``out_of_bounds(points_world)``
        ``[...]`` bool.  Optional: :func:`penetration_metrics` falls back to
        ``in_bounds`` and then to ``bounds`` when it is absent, so this module
        stays decoupled from whichever spelling the field lands on.
    ``reachability_violation(points_world)``
        ``[...]`` bool.  **Secondary diagnostic only** -- the occupancy grid is a
        reachability/free-space volume, not solid geometry (plan section A), so
        this is never reported as penetration.
    ``voxel_size`` / ``bounds`` / ``is_watertight`` / ``scene_name``
        Provenance the caller records beside the numbers.
    """

    def signed_distance(self, points_world: torch.Tensor) -> torch.Tensor:
        ...  # pragma: no cover - protocol declaration


# --------------------------------------------------------------------------
# Rollout stitching
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StitchedSequence:
    """One rollout, history frames already dropped, with its seam indices.

    ``frames``
        ``[T, ...]`` the concatenated sequence.
    ``seams``
        Indices into ``frames`` at which a new window's first *kept* frame sits,
        i.e. the discontinuity lies between ``frames[s - 1]`` and ``frames[s]``.
        Empty for a single-window sequence.
    ``window_lengths``
        Kept length contributed by each window, in rollout order.
    ``history_frames``
        How many leading frames were dropped from windows 2..n.
    """

    frames: torch.Tensor
    seams: Tuple[int, ...]
    window_lengths: Tuple[int, ...]
    history_frames: int

    def __len__(self) -> int:
        return int(self.frames.shape[0])


def stitch_windows(
    windows: Iterable[Union[torch.Tensor, "Sequence[float]"]],
    *,
    history_frames: int = DEFAULT_HISTORY_FRAMES,
    overlap_atol: Optional[float] = None,
) -> StitchedSequence:
    """Concatenate rollout windows, dropping the overlapping history frames.

    Plan section C, 接缝 item 3: "从第二个窗口起，重叠的 2 个 history frame 必须在
    计算任何指标前丢弃 (DIMOS 的 ``start_frame = 2``); 所有时序指标在拼接后的整条
    序列上计算，不得逐窗口计算后再平均."

    Window 0 is kept whole -- its first frames are the given initial state, not a
    re-generation of anything.  Windows 1..n contribute ``w[history_frames:]``.

    ``overlap_atol``
        When set, assert that each dropped block equals the previous window's
        last ``history_frames`` frames to within this absolute tolerance.  Off by
        default because a sampler is free to regenerate its conditioning frames;
        turn it on to catch an off-by-one in a rollout driver, where the two
        blocks are supposed to be the same frames.
    """
    if history_frames < 0:
        raise ValueError("history_frames must be non-negative, got %d" % history_frames)

    tensors = [torch.as_tensor(w) for w in windows]
    if not tensors:
        raise ValueError("stitch_windows needs at least one window")

    trailing = tuple(tensors[0].shape[1:])
    for index, window in enumerate(tensors):
        if window.ndim < 1:
            raise ValueError("window %d is a scalar; expected [T, ...]" % index)
        if tuple(window.shape[1:]) != trailing:
            raise ValueError(
                "window %d has trailing shape %s, window 0 has %s; windows must "
                "describe the same quantity" % (index, tuple(window.shape[1:]), trailing)
            )
        if index and int(window.shape[0]) <= history_frames:
            raise ValueError(
                "window %d has %d frames but %d are history; it would contribute "
                "nothing after the drop" % (index, int(window.shape[0]), history_frames)
            )

    kept = [tensors[0]]
    for index in range(1, len(tensors)):
        current = tensors[index]
        if overlap_atol is not None:
            previous_tail = tensors[index - 1][tensors[index - 1].shape[0] - history_frames :]
            claimed = current[:history_frames]
            if previous_tail.shape != claimed.shape or not torch.allclose(
                previous_tail.to(torch.float64), claimed.to(torch.float64), atol=overlap_atol, rtol=0.0
            ):
                raise ValueError(
                    "window %d's first %d frames are not the previous window's last "
                    "%d within atol=%g; the history overlap is misaligned"
                    % (index, history_frames, history_frames, overlap_atol)
                )
        kept.append(current[history_frames:])

    lengths = tuple(int(part.shape[0]) for part in kept)
    seams = []
    offset = lengths[0]
    for length in lengths[1:]:
        seams.append(offset)
        offset += length
    return StitchedSequence(
        frames=torch.cat(kept, dim=0),
        seams=tuple(seams),
        window_lengths=lengths,
        history_frames=int(history_frames),
    )


def _frames_and_seams(
    motion: Union[StitchedSequence, torch.Tensor],
    *,
    expect_ndim: int,
    what: str,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """Unwrap a stitched sequence, and refuse an un-stitched window stack by shape."""
    if isinstance(motion, StitchedSequence):
        frames, seams = motion.frames, motion.seams
    else:
        frames, seams = torch.as_tensor(motion), ()
    if frames.ndim == expect_ndim + 1:
        raise ValueError(
            "%s has %d axes, expected %d ([T, ...]).  This looks like a stack of "
            "rollout windows; call stitch_windows(...) first so the overlapping "
            "history frames are dropped and the metric is computed on the whole "
            "sequence rather than per window." % (what, frames.ndim, expect_ndim)
        )
    if frames.ndim != expect_ndim:
        raise ValueError("%s has %d axes, expected %d" % (what, frames.ndim, expect_ndim))
    return frames, seams


def _positions(
    motion: Union[StitchedSequence, torch.Tensor],
    *,
    what: str,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """``[T, J, 3]`` float64 positions plus seams.  float64 keeps the analytic
    tests exact and costs nothing at evaluation scale."""
    frames, seams = _frames_and_seams(motion, expect_ndim=3, what=what)
    if int(frames.shape[-1]) != 3:
        raise ValueError("%s must end in an xyz axis, got shape %s" % (what, tuple(frames.shape)))
    return frames.to(torch.float64), seams


def _float(value: torch.Tensor) -> float:
    return float(value.item())


def _nan() -> float:
    return float("nan")


# --------------------------------------------------------------------------
# Tier 1: penetration
# --------------------------------------------------------------------------


def _out_of_bounds(geometry: SceneGeometry, points: torch.Tensor) -> Optional[torch.Tensor]:
    """Best-effort out-of-bbox mask, tolerant of which spelling the field exposes."""
    hook = getattr(geometry, "out_of_bounds", None)
    if callable(hook):
        return torch.as_tensor(hook(points)).to(torch.bool)
    inside = getattr(geometry, "in_bounds", None)
    if callable(inside):
        return ~torch.as_tensor(inside(points)).to(torch.bool)
    bounds = getattr(geometry, "bounds", None)
    if bounds is not None:
        low, high = bounds
        low = torch.as_tensor(low).to(points.dtype)
        high = torch.as_tensor(high).to(points.dtype)
        return ~((points >= low) & (points <= high)).all(dim=-1)
    return None


def penetration_metrics(
    points: Union[StitchedSequence, torch.Tensor],
    geometry: SceneGeometry,
    *,
    threshold_m: float = PENETRATION_THRESHOLD_M,
) -> Dict[str, float]:
    """Human-scene penetration for one sequence.  Plan section C, 穿透 items 1-3.

    ``points``
        ``[T, S, 3]`` metres.  ``S`` is the sampling body: SMPL-X's 10475
        vertices are the 主口径 and the 28 joints are the fast diagnostic; plan
        section C states the two are **not interchangeable**, so which one was
        used must be registered beside the number.  This function does not know
        or care which it was given.

    Returned, all for this one sequence:

    ``pen_ratio``
        Fraction of **sample x frame** pairs with ``sdf < threshold_m``.  TeSMo's
        threshold; TeSMo's own ratio is over *frames*, and the plan deliberately
        restates it over sample-frames, so ``pen_frame_ratio`` below carries the
        literal TeSMo form as a separate, separately-named number.
    ``pen_depth_mean`` / ``pen_depth_max``
        Mean and max of ``|sdf|`` over the penetrating samples only, metres.
    ``pen_burst``
        ``100 * mean_t[(per-frame penetrating fraction)^2]`` -- Dyn-HSI Eq. 9.
        The square is deliberately superlinear so one catastrophic frame is not
        diluted by a long clean sequence, which is the bursty failure shape of an
        autoregressive rollout.
    ``pen_frame_ratio``
        Fraction of frames containing at least one penetrating sample.
    ``oob_ratio``
        Fraction of sample-frames outside the geometry's world bbox.  Plan
        section A pins the rule as "不裁剪、记为正距离" and requires this to be
        reported separately -- 4.3% of GT joints leave the bbox, so it is not a
        corner case.  ``nan`` if the geometry exposes no bounds query.
    ``nonfinite_ratio``
        Fraction of sample-frames whose SDF is not finite (a diverged rollout).
        These never count as penetrating, so they must be visible.

    Deliberately **not** implemented: DIMOS's ``penetration_mean``/``max``
    (``evaluation/cal_metric_interaction.py``), which is
    ``negative_values.abs().sum(axis=-1)`` -- a **sum over vertices** of depth,
    i.e. a penetration-volume proxy, then mean/max over frames.  Three reasons it
    is excluded rather than added: (1) it scales with mesh resolution and body
    model, so it is not comparable to any published number computed on a
    different sampling body -- and we score SMPL-X vertices against a mesh SDF,
    not DIMOS's object SDF; (2) its ``Pene_max`` is the max over frames of the
    *summed* depth, not the deepest vertex, so every reader misreads it; (3) it
    is the single largest contributor to the four-incompatible-quantities problem
    in plan section B.  ``pen_depth_mean`` / ``pen_depth_max`` answer the question
    readers think DIMOS's columns answer.
    """
    frames, _ = _frames_and_seams(points, expect_ndim=3, what="points")
    if int(frames.shape[-1]) != 3:
        raise ValueError("points must end in an xyz axis, got %s" % (tuple(frames.shape),))
    n_frames, n_samples = int(frames.shape[0]), int(frames.shape[1])
    if n_frames == 0 or n_samples == 0:
        raise ValueError("points must be non-empty, got shape %s" % (tuple(frames.shape),))

    sdf = torch.as_tensor(geometry.signed_distance(frames))
    if tuple(sdf.shape) != (n_frames, n_samples):
        raise ValueError(
            "geometry.signed_distance returned shape %s for points %s; expected [T, S]"
            % (tuple(sdf.shape), tuple(frames.shape))
        )
    sdf = sdf.to(torch.float64)

    finite = torch.isfinite(sdf)
    # NaN < threshold is False in IEEE, so a diverged sample never counts as
    # penetrating; nonfinite_ratio is what makes that visible instead of silent.
    penetrating = finite & (sdf < float(threshold_m))
    total = float(n_frames * n_samples)
    per_frame_fraction = penetrating.to(torch.float64).mean(dim=1)

    depths = sdf[penetrating].abs()
    out = {
        "pen_ratio": _float(penetrating.to(torch.float64).mean()),
        "pen_depth_mean": _float(depths.mean()) if depths.numel() else 0.0,
        "pen_depth_max": _float(depths.max()) if depths.numel() else 0.0,
        "pen_burst": 100.0 * _float((per_frame_fraction ** 2).mean()),
        "pen_frame_ratio": _float(penetrating.any(dim=1).to(torch.float64).mean()),
        "nonfinite_ratio": float((~finite).sum().item()) / total,
        "pen_samples": float(int(penetrating.sum().item())),
        "pen_sample_frames": total,
    }
    oob = _out_of_bounds(geometry, frames)
    out["oob_ratio"] = _float(oob.to(torch.float64).mean()) if oob is not None else _nan()
    return out


# --------------------------------------------------------------------------
# Tier 1: engagement
# --------------------------------------------------------------------------


def engagement_metrics(
    points: Union[StitchedSequence, torch.Tensor],
    geometry: SceneGeometry,
    *,
    band_m: float = CONTACT_BAND_M,
) -> Dict[str, float]:
    """Human-scene engagement for one sequence.  Plan section C, engagement item 1.

    ``HSIPRIOR_DESIGN_PRIORS.md`` #7 makes this mandatory: report it in the same
    table as every penetration and foot-sliding number, and never claim a
    penetration win without it.  The measured mechanism is not hypothetical --
    SUMMON's *w/o contact loss* ablation takes the best non-collision in its table
    (0.995) with contact collapsed to 0.194, and HOI's own D2-AH looked
    best-in-class on penetration purely because engagement had collapsed.

    ``contact_count``
        Mean number of samples per frame within the band, i.e. ``sdf <= band_m``.
        A **count**, not a ratio.  Plan section C forbids the binary form: GT
        "at least one sample near a surface" measures 0.746 / 0.996 / 0.9996 over
        three scenes -- saturated and signal-free -- while the count spans
        1.64 / 3.46 / 2.72 on the 28-joint body and does discriminate.
    ``contact_count_exterior``
        Same count restricted to ``0 <= sdf <= band_m``.  Its gap from
        ``contact_count`` is exactly how much of the engagement is penetration
        rather than contact, which is worth seeing.
    ``contact_frame_ratio_saturated_diagnostic``
        The binary "at least one sample in the band" per frame.  **Not a
        reportable metric** -- the name says so.  Kept only so that a future
        comparison against a paper that reports the binary form can be made
        without re-deriving it, and so nobody re-invents it unnamed.

    The band is ``sdf <= +band_m`` including penetrating samples, matching the
    dilated-shell measurement that produced the GT reference values quoted above
    (plan section C cites 9b's "joint inside a 4 cm dilated shell around occupied
    voxels", and a dilation contains the set it dilates).
    """
    frames, _ = _frames_and_seams(points, expect_ndim=3, what="points")
    if int(frames.shape[-1]) != 3:
        raise ValueError("points must end in an xyz axis, got %s" % (tuple(frames.shape),))
    if int(frames.shape[0]) == 0 or int(frames.shape[1]) == 0:
        raise ValueError("points must be non-empty, got shape %s" % (tuple(frames.shape),))

    sdf = torch.as_tensor(geometry.signed_distance(frames)).to(torch.float64)
    finite = torch.isfinite(sdf)
    in_band = finite & (sdf <= float(band_m))
    exterior = in_band & (sdf >= 0.0)
    return {
        "contact_count": _float(in_band.to(torch.float64).sum(dim=1).mean()),
        "contact_count_exterior": _float(exterior.to(torch.float64).sum(dim=1).mean()),
        "contact_frame_ratio_saturated_diagnostic": _float(
            in_band.any(dim=1).to(torch.float64).mean()
        ),
    }


def reaction_divergence_score(
    joints_with_scene: Union[StitchedSequence, torch.Tensor],
    joints_without_scene: Union[StitchedSequence, torch.Tensor],
) -> Dict[str, float]:
    """FantasyHSI RDS: mean per-joint distance between a paired scene-on /
    scene-off generation.  Plan section C, engagement item 2.

    Higher means the model actually reacted to the geometry.  It is the one
    quantity in the survey that is *structurally* immune to the
    "score low penetration by avoiding the scene" failure mode: a model that
    ignores the scene condition scores RDS ~ 0 however clean its penetration is.

    This function is pure -- producing the two paired rollouts under a shared
    initial latent, posterior noise, conditions and ordering (design prior #6) is
    the job of the not-yet-approved rollout stage.  Both arguments must already be
    stitched and frame-aligned.
    """
    with_scene, _ = _positions(joints_with_scene, what="joints_with_scene")
    without_scene, _ = _positions(joints_without_scene, what="joints_without_scene")
    if with_scene.shape != without_scene.shape:
        raise ValueError(
            "RDS needs frame-aligned paired rollouts; got %s and %s"
            % (tuple(with_scene.shape), tuple(without_scene.shape))
        )
    per_joint = torch.linalg.vector_norm(with_scene - without_scene, dim=-1)
    return {"rds": _float(per_joint.mean()), "rds_max": _float(per_joint.max())}


def reachability_diagnostic(
    points: Union[StitchedSequence, torch.Tensor],
    geometry: SceneGeometry,
) -> Dict[str, float]:
    """SECONDARY diagnostic: fraction of sample-frames in an unreachable voxel.

    Plan section A: ``Scene/<scene>.npy`` marks cells "occupied by scene objects
    **or unreachable**", scene 004 reads 0.512 occupied with its densest layer at
    the y~1.98 m ceiling, and GT joints land in "occupied" voxels 7.1% of the
    time with 4.3% outside the bbox entirely.  It is a reachability/free-space
    volume, not solid geometry, so this is explicitly **not** penetration and the
    key name says so.  Never put this in a penetration column.
    """
    frames, _ = _frames_and_seams(points, expect_ndim=3, what="points")
    hook = getattr(geometry, "reachability_violation", None)
    if not callable(hook):
        raise TypeError(
            "geometry %r exposes no reachability_violation(); this diagnostic needs "
            "the occupancy grid, which the mesh-derived field may not carry"
            % (getattr(geometry, "scene_name", geometry),)
        )
    violation = torch.as_tensor(hook(frames)).to(torch.bool)
    return {"reachability_violation_ratio": _float(violation.to(torch.float64).mean())}


# --------------------------------------------------------------------------
# Tier 1: foot
# --------------------------------------------------------------------------


def fs_nemf(
    joints: Union[StitchedSequence, torch.Tensor],
    *,
    ankle_joints: Sequence[int] = ANKLE_JOINTS,
    toe_joints: Sequence[int] = TOE_JOINTS,
    ankle_height_m: float = NEMF_ANKLE_HEIGHT_M,
    toe_height_m: float = NEMF_TOE_HEIGHT_M,
    translate_to_min_foot_height: bool = True,
) -> Dict[str, float]:
    """NeMF foot skate, the FS variant LINGO cites.  Plan section C, 足部 item 1.

    ``s = v * (2 - 2^(h/H))``, accumulated over the ankle and toe joints while
    their height ``h`` is below ``H``; ``H`` = 4 cm for toes and 8 cm for ankles
    (NeMF takes these from HuMoR).  The weight is 1 at ``h = 0`` and ramps to 0 at
    ``h = H``, so there is no hard contact decision.

    Three implementation details that the NeMF paper body omits and that plan
    section C pins, each of which changes the number:

    * ``v`` is the **L1** horizontal displacement ``|dx| + |dz|``, not the L2 norm.
      This is where this function differs from
      ``code/eval_metrics.py:compute_foot_sliding_for_smpl``, which uses L2.
    * the four joints are **summed**, not averaged -- again unlike
      ``compute_foot_sliding_for_smpl``, which divides by 4.  That legacy function
      is retained by plan section C 足部 item 3 purely so the HSI table can carry a
      column comparable with the HOI table; it is a *different quantity* and this
      module does not reimplement or replace it.
    * the sequence is first translated so the **minimum foot height is 0**.
      Without it a sequence floating above the floor scores 0 by construction.

    Note the resulting tension, which is real and is left visible rather than
    silently resolved: the pre-translation makes ``fs_nemf`` measure drift
    relative to the lowest foot the sequence ever reaches, whereas
    :func:`skate_ratio` uses the exact ``y = 0`` LINGO floor.  A sequence hovering
    30 cm up is invisible to the first and fully visible to the second.  That is
    why plan section C reports both and calls them non-monotone in each other.
    Pass ``translate_to_min_foot_height=False`` to score against the true floor;
    the default is the pinned NeMF definition.

    Unit: **cm per frame**.  NeMF's own GT calibration is 0.512 cm/frame, which is
    the reference any value here should be read against.

    Returns ``fs_nemf`` plus its ``fs_nemf_ankle`` / ``fs_nemf_toe`` parts, all
    cm/frame, for this one sequence.
    """
    positions, _ = _positions(joints, what="joints")
    n_frames = int(positions.shape[0])
    if n_frames < 2:
        return {"fs_nemf": 0.0, "fs_nemf_ankle": 0.0, "fs_nemf_toe": 0.0}

    ankle_joints = tuple(int(j) for j in ankle_joints)
    toe_joints = tuple(int(j) for j in toe_joints)
    every_foot = ankle_joints + toe_joints
    _check_joints(every_foot, int(positions.shape[1]), "foot joints")

    if translate_to_min_foot_height:
        offset = positions[:, every_foot, 1].min()
        positions = positions.clone()
        positions[:, :, 1] = positions[:, :, 1] - offset

    def group(indices: Tuple[int, ...], height_m: float) -> torch.Tensor:
        pos = positions[:, indices, :]                              # [T, G, 3]
        step = pos[1:] - pos[:-1]                                   # [T-1, G, 3]
        # L1 horizontal displacement, per plan section C.
        displacement = step[..., 0].abs() + step[..., 2].abs()      # [T-1, G]
        height = pos[:-1, :, 1]                                     # height at the earlier frame
        weight = 2.0 - torch.pow(torch.tensor(2.0, dtype=positions.dtype), height / height_m)
        contributing = height < height_m
        return (displacement * weight)[contributing].sum()

    ankle = group(ankle_joints, float(ankle_height_m))
    toe = group(toe_joints, float(toe_height_m))
    scale = 100.0 / float(n_frames)  # metres -> cm, then per frame
    return {
        "fs_nemf": _float((ankle + toe)) * scale,
        "fs_nemf_ankle": _float(ankle) * scale,
        "fs_nemf_toe": _float(toe) * scale,
    }


def skate_ratio(
    joints: Union[StitchedSequence, torch.Tensor],
    *,
    fps: float,
    foot_joints: Sequence[int] = FOOT_JOINTS,
    contact_height_m: float = SKATE_CONTACT_HEIGHT_M,
    slide_speed_mps: float = SKATE_SPEED_MPS,
) -> Dict[str, float]:
    """GMD/TeSMo foot skating ratio.  Plan section C, 足部 item 2.

    "The proportion of frames where either foot slides a distance greater than
    2.5 cm while in contact with the ground (foot height < 5 cm)."  The 2.5 cm is
    a **per-frame displacement** and is therefore *not* frame-rate invariant:
    plan section C requires it restated as **0.75 m/s**, which is 2.5 cm/frame at
    the LINGO rate of 30 fps and 1.25 cm/frame at 60.  ``fps`` is a required
    argument for exactly that reason -- as a hidden constant this metric silently
    changes meaning whenever the rollout rate changes.

    Floor is the exact LINGO ``y = 0``; heights are absolute and nothing is
    pre-translated here (contrast :func:`fs_nemf`).

    Horizontal (xz) L2 speed, evaluated over the ``T-1`` displacement frames, with
    the height taken at the earlier frame of each pair, and a frame counted when
    **any** of the foot joints satisfies both conditions.
    """
    if not fps > 0:
        raise ValueError("fps must be positive, got %r" % (fps,))
    positions, _ = _positions(joints, what="joints")
    n_frames = int(positions.shape[0])
    if n_frames < 2:
        return {"skate_ratio": 0.0, "skate_frames": 0.0, "skate_denominator_frames": 0.0}

    indices = tuple(int(j) for j in foot_joints)
    _check_joints(indices, int(positions.shape[1]), "foot joints")

    pos = positions[:, indices, :]
    step = pos[1:] - pos[:-1]
    horizontal = torch.stack((step[..., 0], step[..., 2]), dim=-1)
    speed = torch.linalg.vector_norm(horizontal, dim=-1) * float(fps)     # m/s
    height = pos[:-1, :, 1]
    skating = ((height < float(contact_height_m)) & (speed > float(slide_speed_mps))).any(dim=1)
    denominator = float(n_frames - 1)
    return {
        "skate_ratio": _float(skating.to(torch.float64).mean()),
        "skate_frames": float(int(skating.sum().item())),
        "skate_denominator_frames": denominator,
    }


def _check_joints(indices: Tuple[int, ...], n_joints: int, what: str) -> None:
    bad = [j for j in indices if j < 0 or j >= n_joints]
    if bad:
        raise IndexError("%s %s out of range for a %d-joint body" % (what, bad, n_joints))


# --------------------------------------------------------------------------
# Tier 2: goal reaching
# --------------------------------------------------------------------------


def goal_metrics(
    joints: Union[StitchedSequence, torch.Tensor],
    goal_position: Union[torch.Tensor, Sequence[float]],
    *,
    fps: float,
    joint_indices: Optional[Sequence[int]] = None,
    thresholds_m: Sequence[float] = GOAL_THRESHOLDS_M,
) -> Dict[str, float]:
    """Goal reaching for one sequence.  Plan section C, 目标到达 items 1, 2, 4.

    Distance is DIMOS's: **min over joints**, **horizontal (xz) only**.

    ``last_dist`` / ``min_dist``
        Terminal and best-over-rollout distance, metres.  Both are reported
        because they separate exactly when the model reaches the goal and then
        drifts away, which is the real autoregressive failure mode.
    ``success_min_{k}cm`` / ``success_last_{k}cm``
        Plan section C pins the two thresholds -- 10 cm (InfBaGel, our baseline's
        protocol) and 20 cm (the DIMOS/LINGO lineage) -- but the two source
        protocols disagree on the *basis*: DIMOS tests ``dists_xy.min() < 0.2``
        (ever reached) while InfBaGel's ``S%`` tests the final distance.  Rather
        than guess, both bases are returned for both thresholds and the reporting
        table picks one explicitly.
    ``time_to_goal_{k}cm_s``
        First frame index satisfying the threshold, divided by ``fps``, seconds;
        ``nan`` if never satisfied.  ``fps`` is required rather than defaulted to
        30 so the seconds cannot silently become wrong.
    """
    if not fps > 0:
        raise ValueError("fps must be positive, got %r" % (fps,))
    positions, _ = _positions(joints, what="joints")
    goal = torch.as_tensor(goal_position).to(torch.float64).reshape(-1)
    if goal.numel() != 3:
        raise ValueError("goal_position must be a single xyz point, got %d values" % goal.numel())

    if joint_indices is not None:
        indices = tuple(int(j) for j in joint_indices)
        _check_joints(indices, int(positions.shape[1]), "goal joints")
        positions = positions[:, indices, :]

    planar = torch.stack(
        (positions[..., 0] - goal[0], positions[..., 2] - goal[2]), dim=-1
    )
    per_frame = torch.linalg.vector_norm(planar, dim=-1).min(dim=1).values     # [T]

    out = {
        "last_dist": _float(per_frame[-1]),
        "min_dist": _float(per_frame.min()),
    }
    for threshold in thresholds_m:
        threshold = float(threshold)
        label = "%dcm" % int(round(threshold * 100.0))
        reached = per_frame < threshold
        out["success_min_%s" % label] = 1.0 if bool(reached.any()) else 0.0
        out["success_last_%s" % label] = 1.0 if bool(reached[-1]) else 0.0
        if bool(reached.any()):
            first = int(torch.nonzero(reached, as_tuple=False)[0, 0].item())
            out["time_to_goal_%s_s" % label] = first / float(fps)
        else:
            out["time_to_goal_%s_s" % label] = _nan()
    return out


def goal_error_decomposition(
    root_positions: Union[StitchedSequence, torch.Tensor],
    goal_position: Union[torch.Tensor, Sequence[float]],
    *,
    root_forward: Optional[Union[torch.Tensor, Sequence[float]]] = None,
    goal_forward: Optional[Union[torch.Tensor, Sequence[float]]] = None,
) -> Dict[str, float]:
    """TeSMo's three-way goal decomposition.  Plan section C, 目标到达 item 3.

    Planar position (m, xz), orientation (rad) and root height (m), reported
    separately at the terminal frame, so "arrived at the wrong place" is
    distinguishable from "arrived facing the wrong way".

    ``root_positions`` is ``[T, 3]``.  ``root_forward`` is ``[T, 3]`` or ``[3]``;
    when either forward vector is missing, ``goal_orientation_err_rad`` is ``nan``
    rather than silently 0 -- a missing orientation must not read as a perfect one.
    Orientation is the unsigned angle between the xz projections, in [0, pi].
    """
    frames, _ = _frames_and_seams(root_positions, expect_ndim=2, what="root_positions")
    frames = frames.to(torch.float64)
    if int(frames.shape[-1]) != 3:
        raise ValueError("root_positions must be [T, 3], got %s" % (tuple(frames.shape),))
    goal = torch.as_tensor(goal_position).to(torch.float64).reshape(-1)
    if goal.numel() != 3:
        raise ValueError("goal_position must be a single xyz point, got %d values" % goal.numel())

    final = frames[-1]
    planar = torch.stack((final[0] - goal[0], final[2] - goal[2]))
    out = {
        "goal_planar_err_m": _float(torch.linalg.vector_norm(planar)),
        "goal_height_err_m": _float((final[1] - goal[1]).abs()),
        "goal_orientation_err_rad": _nan(),
    }
    if root_forward is None or goal_forward is None:
        return out

    def yaw(vector: Union[torch.Tensor, Sequence[float]]) -> Optional[torch.Tensor]:
        tensor = torch.as_tensor(vector).to(torch.float64)
        if tensor.ndim == 2:
            tensor = tensor[-1]
        if tensor.numel() != 3:
            raise ValueError("forward vectors must be xyz, got %d values" % tensor.numel())
        flat = torch.stack((tensor[0], tensor[2]))
        norm = torch.linalg.vector_norm(flat)
        if float(norm.item()) < _EPS:
            return None
        return flat / norm

    a, b = yaw(root_forward), yaw(goal_forward)
    if a is None or b is None:
        return out
    cosine = torch.clamp((a * b).sum(), -1.0, 1.0)
    out["goal_orientation_err_rad"] = _float(torch.arccos(cosine))
    return out


# --------------------------------------------------------------------------
# Tier 3: window-seam continuity
# --------------------------------------------------------------------------


def jerk_metrics(
    joints: StitchedSequence,
    *,
    fps: float,
    boundary_radius_frames: Optional[int] = None,
) -> Dict[str, float]:
    """SEAM-form ``jerk_ratio`` = boundary jerk / interior jerk.  Plan section C,
    接缝 item 1.

    This is **our own metric**, not an adopted one: plan section C says so
    outright ("本项目自定义：HSI 文献中无此指标，不得声称沿用").  No HSI paper
    measures the window seam; SEAM (robotics) is where the ratio form comes from,
    and it reported 0.195 / 0.094 = 2.07 for its own chunked policy.

    The ratio is the point.  It is self-normalising, needs no GT reference, and
    cannot be gamed by smoothing the whole motion, because that lowers numerator
    and denominator together -- which is exactly the failure of the absolute jerk
    numbers (FlowMDM marks its PJ "->", not "down", and adds AUJ specifically
    because TEACH's Slerp drives jerk to ~0).

    Jerk is the third finite difference over a 4-frame stencil,
    ``J[t] = (p[t+3] - 3 p[t+2] + 3 p[t+1] - p[t]) / dt^3``, magnitude taken per
    joint and then averaged over joints, in m/s^3.  A stencil is a **boundary**
    sample when its 4-frame span straddles a seam, i.e. it contains both
    ``s - 1`` and ``s``; everything else is interior.  Setting
    ``boundary_radius_frames = r`` instead widens that to every stencil whose
    centre lies within ``r`` frames of the seam (the FlowMDM transition-window
    shape).

    Requires a :class:`StitchedSequence`: the metric is undefined without seam
    indices, and computing it per window would measure the opposite of what it is
    for.  Returns ``nan`` for the ratio when either set is empty (a single-window
    sequence has no boundary; a very short one may have no interior).
    """
    if not isinstance(joints, StitchedSequence):
        raise TypeError(
            "jerk_metrics needs a StitchedSequence so it knows where the seams "
            "are; build it with stitch_windows(...)"
        )
    if not fps > 0:
        raise ValueError("fps must be positive, got %r" % (fps,))
    positions, seams = _positions(joints, what="joints")
    n_frames = int(positions.shape[0])
    if n_frames < 4:
        return {
            "jerk_ratio": _nan(),
            "boundary_jerk": _nan(),
            "interior_jerk": _nan(),
            "boundary_jerk_samples": 0.0,
            "interior_jerk_samples": 0.0,
        }

    dt = 1.0 / float(fps)
    third = (
        positions[3:] - 3.0 * positions[2:-1] + 3.0 * positions[1:-2] - positions[:-3]
    ) / (dt ** 3)
    magnitude = torch.linalg.vector_norm(third, dim=-1).mean(dim=-1)      # [T-3]

    n_stencils = int(magnitude.shape[0])
    boundary = torch.zeros(n_stencils, dtype=torch.bool, device=magnitude.device)
    # Stencil t0 spans frames [t0, t0+3]; its centre sits at t0 + 1.5 and a seam
    # at s is the cut between frames s-1 and s, i.e. at s - 0.5.
    centre = torch.arange(n_stencils, dtype=torch.float64, device=magnitude.device) + 1.5
    for seam in seams:
        if boundary_radius_frames is None:
            # span contains both seam-1 and seam  <=>  seam-3 <= t0 <= seam-1
            lo, hi = max(0, seam - 3), min(n_stencils - 1, seam - 1)
            if lo <= hi:
                boundary[lo : hi + 1] = True
        else:
            boundary |= (centre - (float(seam) - 0.5)).abs() <= float(boundary_radius_frames)

    interior = ~boundary
    boundary_jerk = _float(magnitude[boundary].mean()) if bool(boundary.any()) else _nan()
    interior_jerk = _float(magnitude[interior].mean()) if bool(interior.any()) else _nan()
    if interior_jerk != interior_jerk or boundary_jerk != boundary_jerk:
        ratio = _nan()
    elif interior_jerk == 0.0:
        ratio = _nan() if boundary_jerk == 0.0 else float("inf")
    else:
        ratio = boundary_jerk / interior_jerk
    return {
        "jerk_ratio": ratio,
        "boundary_jerk": boundary_jerk,
        "interior_jerk": interior_jerk,
        "boundary_jerk_samples": float(int(boundary.sum().item())),
        "interior_jerk_samples": float(int(interior.sum().item())),
    }


def transition_distance(
    joints: StitchedSequence,
    *,
    root_joint: int = PELVIS_JOINT,
) -> Dict[str, float]:
    """TEACH transition distance at every seam, aligned and unaligned.  Plan
    section C, 接缝 item 2.

    Unaligned: mean over joints of ``||p[s] - p[s-1]||``, metres -- the raw jump
    across the seam, which includes any root translation jump.  Aligned: the same
    after subtracting each frame's root joint, i.e. the pose-only jump.  TEACH
    reports both (0.107 vs 0.122 m) and so do we, because the two separate a
    teleporting root from a snapping pose.

    Blind to velocity and acceleration discontinuity by construction; that is
    what :func:`jerk_metrics` is for.

    Returns the mean over seams plus the worst seam.  ``nan`` when the sequence
    has no seam.
    """
    if not isinstance(joints, StitchedSequence):
        raise TypeError(
            "transition_distance needs a StitchedSequence so it knows where the "
            "seams are; build it with stitch_windows(...)"
        )
    positions, seams = _positions(joints, what="joints")
    if not seams:
        return {
            "transition_distance_unaligned": _nan(),
            "transition_distance_aligned": _nan(),
            "transition_distance_unaligned_max": _nan(),
            "transition_distance_aligned_max": _nan(),
            "transition_seams": 0.0,
        }
    _check_joints((int(root_joint),), int(positions.shape[1]), "root joint")

    before = positions[[s - 1 for s in seams]]                # [S, J, 3]
    after = positions[list(seams)]
    unaligned = torch.linalg.vector_norm(after - before, dim=-1).mean(dim=-1)
    before_local = before - before[:, root_joint : root_joint + 1, :]
    after_local = after - after[:, root_joint : root_joint + 1, :]
    aligned = torch.linalg.vector_norm(after_local - before_local, dim=-1).mean(dim=-1)
    return {
        "transition_distance_unaligned": _float(unaligned.mean()),
        "transition_distance_aligned": _float(aligned.mean()),
        "transition_distance_unaligned_max": _float(unaligned.max()),
        "transition_distance_aligned_max": _float(aligned.max()),
        "transition_seams": float(len(seams)),
    }
