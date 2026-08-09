"""Preregistered Phase 1B P2 inference-time contact guidance for HOIPrior.

Preregistration: ``docs/EXPERIMENT_PLAN.md`` section "2026-08-01 Phase 1B
推理期接触引导 P2（协议对齐，用户批准）" and registry row
``p1-hoi-p2-inference-contact-guidance-preregister-s42-20260801``.

The module is protocol alignment, not a model change.  It is inert unless a
caller explicitly enables it: ``GaussianDiffusion.sample`` only consults a
guidance object when one is passed, so the fixed 500-step unguided native
protocol stays bit-identical by construction.

Two preregistered arms:

* Arm A - InfBaGel-faithful.  The author's complete
  :func:`guidance_loss.apply_hoi_guidance_loss` (hand-object x10 plus
  feet-floor x500), gradient with respect to ``x0_hat``, scaled by
  ``guidance_scale`` and added raw to ``x_{t-1}`` on every reverse step except
  the last.  This mirrors ``code/models/infbagel.py`` where the released
  baseline adds ``torch.autograd.grad(-loss, x_start)[0] * guidance_scale``
  directly to ``x_prev``.
* Arm B - CHOIS-style DDPM analogue.  The same loss, but applied only on the
  last ``last_steps`` reverse steps, with the gradient additionally scaled by
  ``posterior_variance[t]`` and clamped, following
  ``chois_release/manip/model/transformer_object_motion_cond_diffusion.py``
  lines 419 (``classifier_scale``), 438-446 (variance scaling and clip) and
  520 (``guidance_fn is not None and i > 0 and i < 10``).

Determinism: the object surface comes from the D2-Q0 deterministic uniform
index subset, never ``torch.randperm``.  Nothing here consumes global or
generator RNG, so a fixed configuration reproduces bitwise.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F

from guidance_loss import (
    apply_feet_floor_contact_guidance,
    apply_hoi_guidance_loss,
)

from .contact_guidance import (
    decoded_fk_positions,
    transformed_object_vertices,
)
from .representation import REPRESENTATION


ARM_A = "a"
ARM_B = "b"
ARMS: Tuple[str, ...] = (ARM_A, ARM_B)

# Arm A reproduces the released InfBaGel inference configuration, whose
# ``guidance_weight`` is 1 (results/experiments/p0-hoi-table5-baseline-...).
DEFAULT_GUIDANCE_SCALE = 1.0
# Arm B defaults follow CHOIS: ``classifier_scale = 1e3`` (:419), the guided
# window ``0 < i < 10`` (:520) and the ``clip_denoised`` bound 1.0 (:438-446).
CHOIS_CLASSIFIER_SCALE = 1000.0
DEFAULT_LAST_STEPS = 10
DEFAULT_CLAMP = 1.0
CLAMP_UPDATE = "update"
CLAMP_STATE = "state"
CLAMP_TARGETS: Tuple[str, ...] = (CLAMP_UPDATE, CLAMP_STATE)
# An element is counted as clamp-saturated when its pre-clamp magnitude already
# reaches the bound; the tolerance absorbs float32 round-off only.
CLAMP_SATURATION_TOLERANCE = 1e-6

# The author's feet-floor and hand-object weights, kept only so the audit can
# recover the two components from one authoritative loss call.
AUTHOR_FEET_WEIGHT = 500.0
AUTHOR_HAND_WEIGHT = 10.0
FK_JOINTS = 24

# ``guidance_loss.apply_hoi_guidance_loss`` masks its contact term to frames the
# model itself already believes are in contact: it reads ``contact[..., -4:-2]``
# and thresholds at ``> 0.95`` (code/guidance_loss.py:29-31).  Guidance is
# therefore an amplifier of committed engagement, not a generator of it.  P5
# measures the dose-response of that gate, so the threshold and the mask source
# are configurable here.  We never edit the author's file: instead the two
# consumed channels are rewritten to hard 1.0/0.0 before the call, so the
# author's own ``> 0.95`` reproduces exactly the intended mask.
MASK_SOURCE_PREDICTED = "predicted"
MASK_SOURCE_GROUND_TRUTH = "ground_truth"
MASK_SOURCES: Tuple[str, ...] = (MASK_SOURCE_PREDICTED, MASK_SOURCE_GROUND_TRUTH)
DEFAULT_CONTACT_MASK_SOURCE = MASK_SOURCE_PREDICTED
# The author's own threshold; the default keeps the guided path bit-identical.
DEFAULT_CONTACT_MASK_THRESHOLD = 0.95
# Values the author's downstream ``> 0.95`` maps to True / False exactly.
_MASK_IN = 1.0
_MASK_OUT = 0.0

# ``apply_hand_object_interaction_guidance_loss`` collapses two sub-terms into
# one scalar, ``bs * (loss_contact + loss_consistency)``
# (code/guidance_loss.py:38-69), and the sealed mask sweep came out flat because
# they move in opposite directions.  Widening the mask admits more frames, so
# the hinge ``loss_contact`` can only rise, while ``loss_consistency`` falls:
# ``1 - torch.mean(similarity * contact_mask)`` normalises by every one of the
# T*T pairs, including the masked-off zeros, so admitting a frame mechanically
# raises that mean whether or not the motion actually became more consistent.
# These settings reweight the two halves independently and optionally replace
# the author's normaliser.  ``code/guidance_loss.py`` is still never edited: the
# author's arithmetic is reimplemented verbatim in :func:`author_hand_subterms`,
# and the default configuration does not route through it at all.
CONSISTENCY_NORMALIZATION_AUTHOR = "author"
CONSISTENCY_NORMALIZATION_MASKED_PAIRS = "masked_pairs"
CONSISTENCY_NORMALIZATIONS: Tuple[str, ...] = (
    CONSISTENCY_NORMALIZATION_AUTHOR,
    CONSISTENCY_NORMALIZATION_MASKED_PAIRS,
)
# The author's own multipliers and normaliser; the defaults keep the guided path
# bit-identical.
DEFAULT_CONTACT_WEIGHT = 1.0
DEFAULT_CONSISTENCY_WEIGHT = 1.0
DEFAULT_CONSISTENCY_NORMALIZATION = CONSISTENCY_NORMALIZATION_AUTHOR

# Preregistered P7 object-goal terminal term.  The author's guidance knows only
# "the palm should touch the object"; it has no notion of where the object is
# supposed to END UP.  The object channels are NOT detached in the author's
# contact hinge (only the consistency term detaches them, guidance_loss.py:42-47),
# so guidance already moves the object -- it simply moves it toward the hand,
# which is why stronger contact guidance measurably WORSENS end_obj_trans_err
# (3.83750 unguided -> 3.91086 at contact_weight=3) while improving the full
# trajectory obj_trans_dist (14.81981 -> 14.80752).
#
# ``object_goal`` is a task INPUT available at inference time, not ground truth,
# so unlike the cell-U mask this term is deployable.  It is nonetheless disabled
# by default: end_obj_trans_err scores the single frame whose target the model
# was told, so optimizing it directly is close to metric-gaming and must always
# be reported beside obj_trans_dist to show whether the full trajectory was
# sacrificed for the endpoint.
DEFAULT_OBJECT_GOAL_WEIGHT = 0.0

GUIDANCE_KEYS: Tuple[str, ...] = (
    "enabled",
    "arm",
    "guidance_scale",
    "last_steps",
    "clamp",
    "clamp_target",
    "contact_mask_source",
    "contact_mask_threshold",
    "contact_weight",
    "consistency_weight",
    "consistency_normalization",
    "object_goal_weight",
)


@dataclass(frozen=True)
class GuidanceSettings:
    """Resolved, validated inference-guidance configuration."""

    enabled: bool = False
    arm: str = ARM_A
    guidance_scale: float = DEFAULT_GUIDANCE_SCALE
    last_steps: int = DEFAULT_LAST_STEPS
    clamp: Optional[float] = DEFAULT_CLAMP
    clamp_target: str = CLAMP_UPDATE
    contact_mask_source: str = DEFAULT_CONTACT_MASK_SOURCE
    contact_mask_threshold: float = DEFAULT_CONTACT_MASK_THRESHOLD
    contact_weight: float = DEFAULT_CONTACT_WEIGHT
    consistency_weight: float = DEFAULT_CONSISTENCY_WEIGHT
    consistency_normalization: str = DEFAULT_CONSISTENCY_NORMALIZATION
    object_goal_weight: float = DEFAULT_OBJECT_GOAL_WEIGHT

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown HOIPrior guidance arm: {self.arm!r}")
        scale = float(self.guidance_scale)
        if not (scale == scale) or scale in (float("inf"), float("-inf")):
            raise ValueError("guidance_scale must be finite")
        if int(self.last_steps) < 1:
            raise ValueError("guidance last_steps must be at least one")
        if self.clamp is not None:
            bound = float(self.clamp)
            if not (bound == bound) or bound <= 0.0 or bound == float("inf"):
                raise ValueError("guidance clamp must be a positive finite bound")
        if self.clamp_target not in CLAMP_TARGETS:
            raise ValueError(f"unknown guidance clamp target: {self.clamp_target!r}")
        if self.contact_mask_source not in MASK_SOURCES:
            raise ValueError(
                f"unknown guidance contact mask source: {self.contact_mask_source!r}"
            )
        threshold = float(self.contact_mask_threshold)
        if not (threshold == threshold) or not (0.0 <= threshold < 1.0):
            raise ValueError(
                "guidance contact_mask_threshold must lie in [0.0, 1.0)"
            )
        for name in ("contact_weight", "consistency_weight", "object_goal_weight"):
            weight = float(getattr(self, name))
            if not (weight == weight) or weight in (float("inf"), float("-inf")):
                raise ValueError(f"guidance {name} must be finite")
            if weight < 0.0:
                raise ValueError(f"guidance {name} must be non-negative")
        if self.consistency_normalization not in CONSISTENCY_NORMALIZATIONS:
            raise ValueError(
                "unknown guidance consistency normalization: "
                f"{self.consistency_normalization!r}"
            )

    @property
    def uses_default_contact_mask(self) -> bool:
        """True when the mask is bit-identical to the author's own gate."""
        return (
            self.contact_mask_source == MASK_SOURCE_PREDICTED
            and float(self.contact_mask_threshold)
            == float(DEFAULT_CONTACT_MASK_THRESHOLD)
        )

    @property
    def uses_default_hand_decomposition(self) -> bool:
        """True when the hand term is bit-identical to the author's own scalar.

        When this holds, :func:`author_full_hoi_loss` calls the author's
        ``apply_hoi_guidance_loss`` directly rather than reassembling it from
        :func:`author_hand_subterms`, so the sealed P2/P3/P5/D2-AI path is the
        same call it has always been.

        ``object_goal_weight`` participates deliberately.  If it did not, a
        non-zero terminal weight would take the author's untouched branch and be
        silently discarded -- exactly the no-op that voided the first P6 round,
        where the settings never reached the loss that produced the gradient.
        """
        return (
            float(self.contact_weight) == float(DEFAULT_CONTACT_WEIGHT)
            and float(self.consistency_weight) == float(DEFAULT_CONSISTENCY_WEIGHT)
            and self.consistency_normalization == DEFAULT_CONSISTENCY_NORMALIZATION
            and float(self.object_goal_weight) == float(DEFAULT_OBJECT_GOAL_WEIGHT)
        )

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, object]]) -> "GuidanceSettings":
        """Build settings from a plain mapping or an OmegaConf ``DictConfig``."""
        if config is None:
            return cls(enabled=False)
        try:
            keys = [str(key) for key in config]
        except TypeError as error:  # pragma: no cover - defensive
            raise ValueError(f"invalid guidance configuration: {config!r}") from error
        unknown = sorted(set(keys) - set(GUIDANCE_KEYS))
        if unknown:
            raise ValueError(f"unknown HOIPrior guidance keys: {unknown}")
        values: Dict[str, object] = {key: config[key] for key in keys}
        clamp = values.get("clamp", DEFAULT_CLAMP)
        return cls(
            enabled=bool(values.get("enabled", False)),
            arm=str(values.get("arm", ARM_A)),
            guidance_scale=float(values.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)),
            last_steps=int(values.get("last_steps", DEFAULT_LAST_STEPS)),
            clamp=None if clamp is None else float(clamp),
            clamp_target=str(values.get("clamp_target", CLAMP_UPDATE)),
            contact_mask_source=str(
                values.get("contact_mask_source", DEFAULT_CONTACT_MASK_SOURCE)
            ),
            contact_mask_threshold=float(
                values.get("contact_mask_threshold", DEFAULT_CONTACT_MASK_THRESHOLD)
            ),
            contact_weight=float(
                values.get("contact_weight", DEFAULT_CONTACT_WEIGHT)
            ),
            consistency_weight=float(
                values.get("consistency_weight", DEFAULT_CONSISTENCY_WEIGHT)
            ),
            consistency_normalization=str(
                values.get(
                    "consistency_normalization", DEFAULT_CONSISTENCY_NORMALIZATION,
                )
            ),
            object_goal_weight=float(
                values.get("object_goal_weight", DEFAULT_OBJECT_GOAL_WEIGHT)
            ),
        )

    def applies_at(self, reverse_step: int) -> bool:
        """Guidance is never applied on the final reverse step (``step == 0``)."""
        if not self.enabled or reverse_step <= 0:
            return False
        if self.arm == ARM_B:
            return reverse_step < int(self.last_steps)
        return True

    def as_dict(self) -> Dict[str, object]:
        return {
            "enabled": bool(self.enabled),
            "arm": self.arm,
            "guidance_scale": float(self.guidance_scale),
            "last_steps": int(self.last_steps),
            "clamp": None if self.clamp is None else float(self.clamp),
            "clamp_target": self.clamp_target,
            "contact_weight": float(self.contact_weight),
            "consistency_weight": float(self.consistency_weight),
            "consistency_normalization": self.consistency_normalization,
            "loss": "guidance_loss.apply_hoi_guidance_loss",
        }


def resolve_contact_mask(
    contact: torch.Tensor,
    *,
    settings: Optional["GuidanceSettings"] = None,
    ground_truth_contact: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Rewrite the two contact channels the author's loss consumes.

    ``guidance_loss.apply_hoi_guidance_loss`` reads ``contact[..., -4:-2]`` and
    thresholds at ``> 0.95``.  Rather than edit the author's file we rewrite
    those two channels to hard ``1.0`` / ``0.0``, so the author's own comparison
    reproduces the mask selected here exactly.

    With the default settings the resulting booleans are identical to the
    author's, so the guided path stays bit-identical -- the default returns the
    input tensor untouched rather than a rewritten copy.

    ``ground_truth`` is a deliberately NON-DEPLOYABLE diagnostic upper bound:
    ground-truth contact does not exist at inference time.  It answers only "how
    much of the contact gap could guidance recover if the engagement decision
    were perfect", and must never be used as a deployed configuration.
    """
    if settings is None or settings.uses_default_contact_mask:
        return contact
    if settings.contact_mask_source == MASK_SOURCE_GROUND_TRUTH:
        if ground_truth_contact is None:
            raise ValueError(
                "the ground-truth guidance contact mask requires GT contact "
                "labels; this probe is diagnostic only and is not deployable"
            )
        if ground_truth_contact.shape != contact.shape:
            raise ValueError(
                "ground-truth contact must match the predicted contact shape: "
                f"{tuple(ground_truth_contact.shape)} != {tuple(contact.shape)}"
            )
        source = ground_truth_contact
    else:
        source = contact
    threshold = float(settings.contact_mask_threshold)
    masked = source[..., -4:-2] > threshold
    rewritten = contact.clone()
    rewritten[..., -4:-2] = torch.where(
        masked,
        torch.full_like(rewritten[..., -4:-2], _MASK_IN),
        torch.full_like(rewritten[..., -4:-2], _MASK_OUT),
    )
    return rewritten


def author_hand_subterms(
    fk_joints: torch.Tensor,
    object_vertices: torch.Tensor,
    object_translation: torch.Tensor,
    object_rotation: torch.Tensor,
    contact: torch.Tensor,
    *,
    consistency_normalization: str = DEFAULT_CONSISTENCY_NORMALIZATION,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the author's two hand-object sub-terms separately.

    ``apply_hand_object_interaction_guidance_loss`` (code/guidance_loss.py:4-71)
    returns only ``bs * (loss_contact + loss_consistency)``, so the two halves
    cannot be reweighted through it.  The arithmetic below is copied verbatim
    from that function -- the same palm indices 22/23, the same ``torch.cdist``
    followed by ``min(dim=2)``, the same ``> 0.95`` comparison on the (possibly
    already mask-rewritten) channels, the same ``contact_threshold = 0.02``, the
    same ``F.l1_loss`` against zeros, and the same normalize-then-matmul cosine
    construction.  Nothing is "improved": a divergence here would silently
    invalidate every comparison against the sealed results, so
    ``tests/test_hoi_guidance_subterms.py`` reconciles this function against the
    author's own scalar bitwise.

    ``consistency_normalization`` selects the denominator of the temporal term:

    * ``"author"`` -- ``1 - mean(sim * mask)`` over all ``T*T`` entries, i.e. the
      author's own form, kept exactly as written.
    * ``"masked_pairs"`` -- ``1 - sum(sim * mask) / clamp_min(sum(mask), 1)``,
      the mean taken over only the mask-on pairs.  This is a deliberate
      DEVIATION from the author's arithmetic, not a bug fix to author code: the
      author's mean divides by every pair including the masked-off zeros, so
      admitting more frames mechanically raises it regardless of whether the
      motion became more consistent, which confounds any mask dose-response.
    """
    if consistency_normalization not in CONSISTENCY_NORMALIZATIONS:
        raise ValueError(
            "unknown guidance consistency normalization: "
            f"{consistency_normalization!r}"
        )
    human_jnts = fk_joints
    obj_verts = object_vertices
    pred_seq_com_pos = object_translation
    pred_obj_rot_mat = object_rotation
    contact_labels = contact

    num_seq = human_jnts.shape[0]
    num_steps = human_jnts.shape[1]

    l_palm_idx = 22
    r_palm_idx = 23

    left_palm_jpos = human_jnts[:, :, l_palm_idx, :]
    right_palm_jpos = human_jnts[:, :, r_palm_idx, :]

    contact_points = torch.cat(
        (left_palm_jpos[:, :, None, :], right_palm_jpos[:, :, None, :]), dim=2,
    )
    bs, seq_len, _, _ = contact_points.shape

    dists = torch.cdist(
        contact_points.reshape(bs * seq_len, 2, 3)[:, :, :],
        obj_verts.reshape(bs * seq_len, -1, 3),
    )
    dists, _ = torch.min(dists, 2)

    pred_contact_semantic = contact_labels[:, :, -4:-2]

    contact_labels = pred_contact_semantic > 0.95

    contact_labels = (
        contact_labels.reshape(bs * seq_len, -1)[:, :2].detach().to(dists.device)
    )

    zero_target = torch.zeros_like(dists).to(dists.device)
    contact_threshold = 0.02

    loss_contact = F.l1_loss(
        torch.maximum(dists * contact_labels[:, :2] - contact_threshold, zero_target),
        zero_target,
    )

    left_palm_to_obj_com = left_palm_jpos - pred_seq_com_pos.detach()
    right_palm_to_obj_com = right_palm_jpos - pred_seq_com_pos.detach()
    relative_left_palm_jpos = torch.matmul(
        pred_obj_rot_mat.detach().transpose(2, 3), left_palm_to_obj_com[:, :, :, None],
    ).squeeze(-1)
    relative_right_palm_jpos = torch.matmul(
        pred_obj_rot_mat.detach().transpose(2, 3), right_palm_to_obj_com[:, :, :, None],
    ).squeeze(-1)

    contact_labels = contact_labels.reshape(num_seq, num_steps, -1)

    left_contact_labels_expanded = contact_labels[:, :, 0:1]
    left_contact_mask = (
        left_contact_labels_expanded * left_contact_labels_expanded.transpose(-1, -2)
    )

    right_contact_labels_expanded = contact_labels[:, :, 1:2]
    right_contact_mask = (
        right_contact_labels_expanded * right_contact_labels_expanded.transpose(-1, -2)
    )

    left_norms = torch.norm(relative_left_palm_jpos, dim=-1, keepdim=True)
    left_normalized = relative_left_palm_jpos / left_norms
    left_similarity = torch.matmul(left_normalized, left_normalized.transpose(-1, -2))

    right_norms = torch.norm(relative_right_palm_jpos, dim=-1, keepdim=True)
    right_normalized = relative_right_palm_jpos / right_norms
    right_similarity = torch.matmul(right_normalized, right_normalized.transpose(-1, -2))

    if consistency_normalization == CONSISTENCY_NORMALIZATION_AUTHOR:
        loss_consistency = (
            1 - torch.mean(left_similarity * left_contact_mask)
            + 1 - torch.mean(right_similarity * right_contact_mask)
        )
    else:
        # DEVIATION from the author, deliberately: divide by the mask-on pairs
        # only.  ``clamp_min(1)`` keeps an all-off hand at exactly the author's
        # ``1 - 0``, so a fully masked-off window scores 2.0 either way.
        left_pairs = left_contact_mask.to(left_similarity.dtype).sum().clamp_min(1.0)
        right_pairs = right_contact_mask.to(right_similarity.dtype).sum().clamp_min(1.0)
        loss_consistency = (
            1 - (left_similarity * left_contact_mask).sum() / left_pairs
            + 1 - (right_similarity * right_contact_mask).sum() / right_pairs
        )

    return loss_contact, loss_consistency


def object_goal_terminal_loss(
    object_translation: torch.Tensor,
    object_goal: torch.Tensor,
) -> torch.Tensor:
    """Squared distance from the LAST frame's object position to its goal.

    ``end_obj_trans_err`` scores exactly one frame: the final frame of the final
    rollout window (``obj_trans_pred_seg[-1:, -1, :]``,
    ``test_infbagel_hoi.py:158``).  This term therefore acts on that frame only,
    rather than pulling the whole trajectory toward the endpoint, which would
    trade away ``obj_trans_dist``.

    ``object_goal`` is a task INPUT, available at inference time, so this is a
    deployable term -- unlike the cell-U ground-truth mask.  It is still off by
    default: the metric it targets scores the model's ability to reproduce a
    number it was handed, so any gain here must be reported beside
    ``obj_trans_dist`` to show whether the full trajectory paid for it.
    """
    if object_translation.ndim != 3 or object_translation.shape[-1] != 3:
        raise ValueError(
            "object-goal guidance expects [B,T,3] object translation, got "
            f"{tuple(object_translation.shape)}"
        )
    batch = object_translation.shape[0]
    goal = object_goal.reshape(batch, 3).to(object_translation)
    terminal = object_translation[:, -1, :]
    return ((terminal - goal) ** 2).sum(dim=-1).mean() * batch


def author_full_hoi_loss(
    fk_joints: torch.Tensor,
    object_vertices: torch.Tensor,
    object_translation: torch.Tensor,
    object_rotation: torch.Tensor,
    contact: torch.Tensor,
    settings: Optional["GuidanceSettings"] = None,
    object_goal: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Call the author's unmodified complete HOI guidance loss.

    ``code/guidance_loss.py`` is never edited: this wrapper only validates that
    the tensors produced by the HOIPrior codec match the conventions the author
    assumes.  ``apply_hoi_guidance_loss`` indexes joints 10/11 (toes) and 22/23
    (palms) of a ``[B,T,24,3]`` world-frame, y-up, metre-scale FK tensor, which
    is exactly what :func:`priors.losses._fk_positions` returns for the decoded
    HOIPrior state.  ``scene_flag`` and ``get_nearest_free_voxel`` are unused in
    the HOI branch and are therefore passed as ``None``.

    ``settings`` is optional and inert by default.  With ``settings is None`` or
    ``settings.uses_default_hand_decomposition``, this is exactly the call it
    has always been -- no reimplementation, no ``0.0 * anything`` -- so every
    sealed P2/P3/P5/D2-AI result stays reproducible bitwise.  Only a non-default
    weight or normalization reassembles the total from
    :func:`author_hand_subterms` and
    :func:`guidance_loss.apply_feet_floor_contact_guidance`.
    """
    if fk_joints.ndim != 4 or fk_joints.shape[2:] != (FK_JOINTS, 3):
        raise ValueError(
            f"HOI guidance expects [B,T,{FK_JOINTS},3] FK joints, got {tuple(fk_joints.shape)}"
        )
    batch, frames = fk_joints.shape[:2]
    if object_vertices.ndim != 4 or object_vertices.shape[:2] != (batch, frames):
        raise ValueError("HOI guidance object vertices differ from FK joints")
    if object_vertices.shape[-1] != 3:
        raise ValueError("HOI guidance object vertices must be [B,T,V,3]")
    if object_translation.shape != (batch, frames, 3):
        raise ValueError("HOI guidance object translation must be [B,T,3]")
    if object_rotation.shape != (batch, frames, 3, 3):
        raise ValueError("HOI guidance object rotation must be [B,T,3,3]")
    if contact.shape != (batch, frames, 4):
        raise ValueError("HOI guidance contact must be [B,T,4]")
    if settings is None or settings.uses_default_hand_decomposition:
        return apply_hoi_guidance_loss(
            fk_joints,
            object_vertices,
            object_translation,
            object_rotation,
            contact,
            None,
            None,
        )
    loss_contact, loss_consistency = author_hand_subterms(
        fk_joints,
        object_vertices,
        object_translation,
        object_rotation,
        contact,
        consistency_normalization=settings.consistency_normalization,
    )
    hand = batch * (
        float(settings.contact_weight) * loss_contact
        + float(settings.consistency_weight) * loss_consistency
    )
    total = (
        AUTHOR_HAND_WEIGHT * hand
        + AUTHOR_FEET_WEIGHT * apply_feet_floor_contact_guidance(fk_joints)
    )
    goal_weight = float(settings.object_goal_weight)
    if goal_weight:
        if object_goal is None:
            raise ValueError(
                "a non-zero guidance object_goal_weight requires the object goal; "
                "without it the terminal term would be silently dropped"
            )
        total = total + goal_weight * object_goal_terminal_loss(
            object_translation, object_goal,
        )
    return total


def guidance_gradient_with_subterms(
    clean: torch.Tensor,
    *,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
    settings: Optional["GuidanceSettings"] = None,
    ground_truth_contact: Optional[torch.Tensor] = None,
    object_goal: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``d(-loss)/d(x0_hat)``, the total and feet loss, and both hand halves.

    The last two values are the UNWEIGHTED ``loss_contact`` and
    ``loss_consistency`` of :func:`author_hand_subterms`, i.e. before the
    ``bs``, ``contact_weight``/``consistency_weight`` and ``x10`` factors that
    :func:`author_full_hoi_loss` applies.  They are an audit-only replay in the
    same sense as the feet term below: they are computed under ``no_grad`` from
    the tensors that produced the gradient, so the sampled trajectory stays
    bit-identical while the manifest can report which half of the hand term the
    guided update was actually chasing.
    """
    with torch.enable_grad():
        differentiable_clean = clean.detach().requires_grad_(True)
        decoded = codec.decode(differentiable_clean, frame)
        fk_joints = decoded_fk_positions(decoded, rest_human_offsets, parents_24)
        vertices = transformed_object_vertices(
            rest_vertices,
            decoded["object_rotation"],
            decoded["object_translation"],
        )
        # Default settings return the tensor untouched, so the guided path stays
        # bit-identical to the sealed P2/P3 results.
        contact = resolve_contact_mask(
            decoded["contact"],
            settings=settings,
            ground_truth_contact=ground_truth_contact,
        )
        # ``settings`` IS forwarded: the reweighted hand term is the manipulated
        # factor of the P6 sweep, so it must be the loss the guided update is
        # taken from.  ``author_full_hoi_loss`` branches internally on
        # ``uses_default_hand_decomposition``, so at the default weights this
        # still calls the author's own function and the sealed P2/P3/P5 path
        # stays bit-identical.
        loss = author_full_hoi_loss(
            fk_joints,
            vertices,
            decoded["object_translation"],
            decoded["object_rotation"],
            contact,
            settings=settings,
            object_goal=object_goal,
        )
        # Audit-only replay of the author's feet term.  It costs one min and one
        # mse over [B,T,1] and lets the manifest prove the x500 component was
        # actually present in the loss that produced the gradient.
        feet = apply_feet_floor_contact_guidance(fk_joints)
        # Audit-only replay of the two hand halves.  ``no_grad`` keeps it out of
        # the autograd graph that the guided update is taken from.
        with torch.no_grad():
            loss_contact, loss_consistency = author_hand_subterms(
                fk_joints,
                vertices,
                decoded["object_translation"],
                decoded["object_rotation"],
                contact,
                consistency_normalization=(
                    DEFAULT_CONSISTENCY_NORMALIZATION
                    if settings is None
                    else settings.consistency_normalization
                ),
            )
        gradient = torch.autograd.grad(-loss, differentiable_clean)[0]
    return (
        gradient.detach(),
        loss.detach(),
        feet.detach(),
        loss_contact.detach(),
        loss_consistency.detach(),
    )


def guidance_gradient(
    clean: torch.Tensor,
    *,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
    settings: Optional["GuidanceSettings"] = None,
    ground_truth_contact: Optional[torch.Tensor] = None,
    object_goal: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``d(-loss)/d(x0_hat)`` plus the total and feet-floor loss terms.

    The historical three-value contract, kept for every existing caller.  Use
    :func:`guidance_gradient_with_subterms` when the two hand halves are needed.
    """
    gradient, loss, feet, _, _ = guidance_gradient_with_subterms(
        clean,
        codec=codec,
        frame=frame,
        rest_human_offsets=rest_human_offsets,
        parents_24=parents_24,
        rest_vertices=rest_vertices,
        settings=settings,
        ground_truth_contact=ground_truth_contact,
        object_goal=object_goal,
    )
    return gradient, loss, feet


class GuidanceAudit:
    """Device-resident accumulators, converted to Python only at report time."""

    def __init__(self, settings: Optional["GuidanceSettings"] = None) -> None:
        self.applied_steps = 0
        self.sample_calls = 0
        # The configuration the recorded steps were produced under, so a sweep
        # cell can be attributed to its decomposition without re-reading config.
        self.settings = settings
        self._loss_sum: Optional[torch.Tensor] = None
        self._feet_loss_sum: Optional[torch.Tensor] = None
        self._gradient_square_sum: Optional[torch.Tensor] = None
        self._gradient_absolute_maximum: Optional[torch.Tensor] = None
        self._update_absolute_maximum: Optional[torch.Tensor] = None
        self._nonfinite_steps: Optional[torch.Tensor] = None
        self._gradient_elements = 0
        self._loss_contact_sum: Optional[torch.Tensor] = None
        self._loss_consistency_sum: Optional[torch.Tensor] = None
        self._subterm_steps = 0
        self._clamp_saturated_sum: Optional[torch.Tensor] = None
        self._clamp_elements = 0
        self._clamp_active_steps = 0
        # Cell-U only.  How engaged the ground-truth windows handed to the probe
        # actually were, recorded so the artifact carries its own evidence.  It
        # is NOT asserted against the evaluator's ``gt_contact_percent``: that is
        # a geometric judgement on interpolated evaluation frames, while this is
        # the dataset's annotation channel at >0.95.  Over the 482 test
        # annotation files those two are 0.66188 and 0.62794 -- different
        # statistics, so equating them would fire on correct alignment.
        self.ground_truth_windows = 0
        self._ground_truth_engaged = 0
        self._ground_truth_frames = 0

    def record_ground_truth_window(self, engaged: int, frames: int) -> None:
        """Record one validated cell-U ground-truth window."""
        self.ground_truth_windows += 1
        self._ground_truth_engaged += int(engaged)
        self._ground_truth_frames += int(frames)

    def bind_settings(self, settings: "GuidanceSettings") -> None:
        """Attach the configuration these accumulators are recorded under.

        One audit is shared by every window of a sampling pass, so this is
        called once per :class:`HOIContactGuidance`.  It fails closed rather
        than silently mixing two decompositions into one reported cell.
        """
        if self.settings is not None and self.settings != settings:
            raise ValueError(
                "one GuidanceAudit cannot span two guidance configurations"
            )
        self.settings = settings

    def record(
        self,
        gradient: torch.Tensor,
        update: torch.Tensor,
        loss: torch.Tensor,
        feet_loss: torch.Tensor,
        loss_contact: Optional[torch.Tensor] = None,
        loss_consistency: Optional[torch.Tensor] = None,
        *,
        clamp_saturated: Optional[torch.Tensor] = None,
        clamp_elements: Optional[int] = None,
    ) -> None:
        """Accumulate one guided reverse step.

        ``loss_contact`` / ``loss_consistency`` are the UNWEIGHTED hand halves.
        They are optional so that a caller without them keeps working; their
        reported means then stay ``None`` rather than silently becoming zero.

        ``clamp_saturated`` is the count of elements already at the clamp bound
        BEFORE clamping, and is ``None`` when no clamp was in force on this
        step.  ``clamp_elements`` is the size of the clamped tensor, recorded
        whether or not the clamp is active, so the denominator is always known.
        """
        if self._loss_sum is None:
            zero = gradient.new_zeros(())
            self._loss_sum = zero.clone()
            self._feet_loss_sum = zero.clone()
            self._gradient_square_sum = zero.clone()
            self._gradient_absolute_maximum = zero.clone()
            self._update_absolute_maximum = zero.clone()
            self._nonfinite_steps = zero.clone()
            self._loss_contact_sum = zero.clone()
            self._loss_consistency_sum = zero.clone()
            self._clamp_saturated_sum = torch.zeros(
                (), dtype=torch.long, device=gradient.device,
            )
        self.applied_steps += 1
        self._gradient_elements += gradient.numel()
        self._loss_sum += loss.detach().reshape(())
        self._feet_loss_sum += feet_loss.detach().reshape(())
        self._gradient_square_sum += gradient.square().sum()
        self._gradient_absolute_maximum = torch.maximum(
            self._gradient_absolute_maximum, gradient.abs().amax(),
        )
        self._update_absolute_maximum = torch.maximum(
            self._update_absolute_maximum, update.abs().amax(),
        )
        self._nonfinite_steps += (
            ~torch.isfinite(gradient).all()
        ).to(self._nonfinite_steps)
        if loss_contact is not None and loss_consistency is not None:
            self._loss_contact_sum += loss_contact.detach().reshape(())
            self._loss_consistency_sum += loss_consistency.detach().reshape(())
            self._subterm_steps += 1
        if clamp_elements is not None:
            self._clamp_elements += int(clamp_elements)
        if clamp_saturated is not None:
            self._clamp_saturated_sum += clamp_saturated.detach().reshape(()).long()
            self._clamp_active_steps += 1

    def as_dict(self) -> Dict[str, object]:
        settings = self.settings
        value: Dict[str, object] = {
            "guidance_applied_steps": self.applied_steps,
            "guidance_sample_calls": self.sample_calls,
            # The decomposition actually in force, threaded from the settings
            # object the guidance held; never re-read from config.
            "guidance_contact_weight": (
                None if settings is None else float(settings.contact_weight)
            ),
            "guidance_consistency_weight": (
                None if settings is None else float(settings.consistency_weight)
            ),
            "guidance_consistency_normalization": (
                None if settings is None else settings.consistency_normalization
            ),
        }
        if self._loss_sum is None or not self.applied_steps:
            value.update({
                "guidance_loss_mean": None,
                "guidance_feet_loss_mean": None,
                "guidance_hand_loss_mean": None,
                "guidance_loss_contact_mean": None,
                "guidance_loss_consistency_mean": None,
                "guidance_gradient_rms": None,
                "guidance_gradient_max_abs": None,
                "guidance_update_max_abs": None,
                "guidance_nonfinite_steps": 0,
                "guidance_clamp_saturation_fraction": None,
                "guidance_clamp_saturated_elements": 0,
                "guidance_clamp_total_elements": 0,
            })
            value.update(self._ground_truth_report())
            return value
        loss_mean = float(self._loss_sum.detach().cpu()) / self.applied_steps
        feet_mean = float(self._feet_loss_sum.detach().cpu()) / self.applied_steps
        saturated = int(self._clamp_saturated_sum.detach().cpu())
        value.update({
            "guidance_loss_mean": loss_mean,
            "guidance_feet_loss_mean": feet_mean,
            "guidance_feet_weighted_mean": AUTHOR_FEET_WEIGHT * feet_mean,
            # Unchanged definition: the hand half recovered by subtracting the
            # weighted feet term from the total.  Every sealed artifact was
            # parsed with this meaning, so it is NOT redefined in terms of the
            # sub-terms below even though those are now available directly.
            "guidance_hand_loss_mean": (
                loss_mean - AUTHOR_FEET_WEIGHT * feet_mean
            ) / AUTHOR_HAND_WEIGHT,
            "guidance_loss_contact_mean": (
                None if not self._subterm_steps
                else float(self._loss_contact_sum.detach().cpu())
                / self._subterm_steps
            ),
            "guidance_loss_consistency_mean": (
                None if not self._subterm_steps
                else float(self._loss_consistency_sum.detach().cpu())
                / self._subterm_steps
            ),
            "guidance_gradient_rms": (
                float(self._gradient_square_sum.detach().cpu())
                / self._gradient_elements
            ) ** 0.5,
            "guidance_gradient_max_abs": float(
                self._gradient_absolute_maximum.detach().cpu()
            ),
            "guidance_update_max_abs": float(
                self._update_absolute_maximum.detach().cpu()
            ),
            "guidance_nonfinite_steps": int(self._nonfinite_steps.detach().cpu()),
            # ``None`` rather than 0.0 when no clamp was in force: a real zero
            # means "the clamp destroyed nothing", which is a different claim.
            "guidance_clamp_saturation_fraction": (
                saturated / self._clamp_elements
                if self._clamp_active_steps and self._clamp_elements
                else None
            ),
            "guidance_clamp_saturated_elements": saturated,
            "guidance_clamp_total_elements": self._clamp_elements,
        })
        value.update(self._ground_truth_report())
        return value

    def _ground_truth_report(self) -> Dict[str, object]:
        """Cell-U evidence: how engaged the validated GT windows actually were."""
        frames = self._ground_truth_frames
        return {
            "guidance_ground_truth_windows": self.ground_truth_windows,
            "guidance_ground_truth_engaged_frames": self._ground_truth_engaged,
            "guidance_ground_truth_total_frames": frames,
            "guidance_ground_truth_engagement_fraction": (
                self._ground_truth_engaged / frames if frames else None
            ),
        }


class HOIContactGuidance:
    """One window's guidance context handed to ``GaussianDiffusion.sample``."""

    def __init__(
        self,
        settings: GuidanceSettings,
        *,
        posterior_variance: torch.Tensor,
        codec,
        frame,
        rest_human_offsets: torch.Tensor,
        parents_24: torch.Tensor,
        rest_vertices: torch.Tensor,
        audit: Optional[GuidanceAudit] = None,
        ground_truth_contact: Optional[torch.Tensor] = None,
        object_goal: Optional[torch.Tensor] = None,
    ) -> None:
        if not settings.enabled:
            raise ValueError("HOIContactGuidance requires an enabled configuration")
        if rest_human_offsets.ndim != 3 or rest_human_offsets.shape[1:] != (FK_JOINTS, 3):
            raise ValueError(
                f"guidance rest offsets must be [B,{FK_JOINTS},3], "
                f"got {tuple(rest_human_offsets.shape)}"
            )
        if rest_vertices.ndim != 3 or rest_vertices.shape[-1] != 3:
            raise ValueError("guidance rest vertices must be [B,V,3]")
        if rest_vertices.shape[0] != rest_human_offsets.shape[0]:
            raise ValueError("guidance rest vertices and rest offsets differ in batch")
        self.settings = settings
        self.posterior_variance = posterior_variance
        self.codec = codec
        self.frame = frame
        self.rest_human_offsets = rest_human_offsets
        self.parents_24 = parents_24
        self.rest_vertices = rest_vertices
        self.audit = audit
        if audit is not None:
            audit.bind_settings(settings)
        # Only consumed when contact_mask_source is the non-default
        # ground_truth probe; see resolve_contact_mask.
        self.ground_truth_contact = ground_truth_contact
        # Preregistered P7: the object goal is a task input, so this term is
        # deployable; it is inert unless object_goal_weight is non-zero.
        self.object_goal = object_goal
        if (
            settings.contact_mask_source == MASK_SOURCE_GROUND_TRUTH
            and ground_truth_contact is None
        ):
            raise ValueError(
                "the ground-truth guidance contact mask probe requires GT "
                "contact labels to be supplied; it is diagnostic only"
            )

    def applies_at(self, reverse_step: int) -> bool:
        return self.settings.applies_at(reverse_step)

    def apply(
        self,
        posterior: torch.Tensor,
        clean: torch.Tensor,
        fixed_history: torch.Tensor,
        reverse_step: int,
    ) -> torch.Tensor:
        """Return the guided ``x_{t-1}`` with the immutable history re-pinned."""
        if not self.applies_at(reverse_step):
            return posterior
        (
            gradient,
            loss,
            feet_loss,
            loss_contact,
            loss_consistency,
        ) = guidance_gradient_with_subterms(
            clean,
            codec=self.codec,
            frame=self.frame,
            rest_human_offsets=self.rest_human_offsets,
            parents_24=self.parents_24,
            rest_vertices=self.rest_vertices,
            settings=self.settings,
            ground_truth_contact=self.ground_truth_contact,
            object_goal=self.object_goal,
        )
        update = gradient * self.settings.guidance_scale
        clamp_saturated: Optional[torch.Tensor] = None
        clamp_elements: Optional[int] = None
        if self.settings.arm == ARM_B:
            update = update * self.posterior_variance[reverse_step]
            if self.settings.clamp is not None and self.settings.clamp_target == CLAMP_UPDATE:
                bound = float(self.settings.clamp)
                # Counted BEFORE clamping: how much of the Arm-B update the
                # clamp is about to destroy.  A sweep cell that looks null could
                # otherwise just be saturating.
                clamp_saturated = (
                    update.abs() >= bound - CLAMP_SATURATION_TOLERANCE
                ).sum()
                clamp_elements = update.numel()
                update = update.clamp(-bound, bound)
            else:
                # The denominator is still known when the clamp is off; the
                # audit reports the fraction as None, not a misleading 0.0.
                clamp_elements = update.numel()
        result = posterior + update
        if (
            self.settings.arm == ARM_B
            and self.settings.clamp is not None
            and self.settings.clamp_target == CLAMP_STATE
        ):
            result = result.clamp(-self.settings.clamp, self.settings.clamp)
        result[:, :REPRESENTATION.history_frames] = fixed_history
        if self.audit is not None:
            self.audit.record(
                gradient,
                update,
                loss,
                feet_loss,
                loss_contact,
                loss_consistency,
                clamp_saturated=clamp_saturated,
                clamp_elements=clamp_elements,
            )
        return result
