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

# The author's feet-floor and hand-object weights, kept only so the audit can
# recover the two components from one authoritative loss call.
AUTHOR_FEET_WEIGHT = 500.0
AUTHOR_HAND_WEIGHT = 10.0
FK_JOINTS = 24

GUIDANCE_KEYS: Tuple[str, ...] = (
    "enabled",
    "arm",
    "guidance_scale",
    "last_steps",
    "clamp",
    "clamp_target",
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
            "loss": "guidance_loss.apply_hoi_guidance_loss",
        }


def author_full_hoi_loss(
    fk_joints: torch.Tensor,
    object_vertices: torch.Tensor,
    object_translation: torch.Tensor,
    object_rotation: torch.Tensor,
    contact: torch.Tensor,
) -> torch.Tensor:
    """Call the author's unmodified complete HOI guidance loss.

    ``code/guidance_loss.py`` is never edited: this wrapper only validates that
    the tensors produced by the HOIPrior codec match the conventions the author
    assumes.  ``apply_hoi_guidance_loss`` indexes joints 10/11 (toes) and 22/23
    (palms) of a ``[B,T,24,3]`` world-frame, y-up, metre-scale FK tensor, which
    is exactly what :func:`priors.losses._fk_positions` returns for the decoded
    HOIPrior state.  ``scene_flag`` and ``get_nearest_free_voxel`` are unused in
    the HOI branch and are therefore passed as ``None``.
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
    return apply_hoi_guidance_loss(
        fk_joints,
        object_vertices,
        object_translation,
        object_rotation,
        contact,
        None,
        None,
    )


def guidance_gradient(
    clean: torch.Tensor,
    *,
    codec,
    frame,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    rest_vertices: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``d(-loss)/d(x0_hat)`` plus the total and feet-floor loss terms."""
    with torch.enable_grad():
        differentiable_clean = clean.detach().requires_grad_(True)
        decoded = codec.decode(differentiable_clean, frame)
        fk_joints = decoded_fk_positions(decoded, rest_human_offsets, parents_24)
        vertices = transformed_object_vertices(
            rest_vertices,
            decoded["object_rotation"],
            decoded["object_translation"],
        )
        loss = author_full_hoi_loss(
            fk_joints,
            vertices,
            decoded["object_translation"],
            decoded["object_rotation"],
            decoded["contact"],
        )
        # Audit-only replay of the author's feet term.  It costs one min and one
        # mse over [B,T,1] and lets the manifest prove the x500 component was
        # actually present in the loss that produced the gradient.
        feet = apply_feet_floor_contact_guidance(fk_joints)
        gradient = torch.autograd.grad(-loss, differentiable_clean)[0]
    return gradient.detach(), loss.detach(), feet.detach()


class GuidanceAudit:
    """Device-resident accumulators, converted to Python only at report time."""

    def __init__(self) -> None:
        self.applied_steps = 0
        self.sample_calls = 0
        self._loss_sum: Optional[torch.Tensor] = None
        self._feet_loss_sum: Optional[torch.Tensor] = None
        self._gradient_square_sum: Optional[torch.Tensor] = None
        self._gradient_absolute_maximum: Optional[torch.Tensor] = None
        self._update_absolute_maximum: Optional[torch.Tensor] = None
        self._nonfinite_steps: Optional[torch.Tensor] = None
        self._gradient_elements = 0

    def record(
        self,
        gradient: torch.Tensor,
        update: torch.Tensor,
        loss: torch.Tensor,
        feet_loss: torch.Tensor,
    ) -> None:
        if self._loss_sum is None:
            zero = gradient.new_zeros(())
            self._loss_sum = zero.clone()
            self._feet_loss_sum = zero.clone()
            self._gradient_square_sum = zero.clone()
            self._gradient_absolute_maximum = zero.clone()
            self._update_absolute_maximum = zero.clone()
            self._nonfinite_steps = zero.clone()
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

    def as_dict(self) -> Dict[str, object]:
        value: Dict[str, object] = {
            "guidance_applied_steps": self.applied_steps,
            "guidance_sample_calls": self.sample_calls,
        }
        if self._loss_sum is None or not self.applied_steps:
            value.update({
                "guidance_loss_mean": None,
                "guidance_feet_loss_mean": None,
                "guidance_hand_loss_mean": None,
                "guidance_gradient_rms": None,
                "guidance_gradient_max_abs": None,
                "guidance_update_max_abs": None,
                "guidance_nonfinite_steps": 0,
            })
            return value
        loss_mean = float(self._loss_sum.detach().cpu()) / self.applied_steps
        feet_mean = float(self._feet_loss_sum.detach().cpu()) / self.applied_steps
        value.update({
            "guidance_loss_mean": loss_mean,
            "guidance_feet_loss_mean": feet_mean,
            "guidance_feet_weighted_mean": AUTHOR_FEET_WEIGHT * feet_mean,
            "guidance_hand_loss_mean": (
                loss_mean - AUTHOR_FEET_WEIGHT * feet_mean
            ) / AUTHOR_HAND_WEIGHT,
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
        })
        return value


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
        gradient, loss, feet_loss = guidance_gradient(
            clean,
            codec=self.codec,
            frame=self.frame,
            rest_human_offsets=self.rest_human_offsets,
            parents_24=self.parents_24,
            rest_vertices=self.rest_vertices,
        )
        update = gradient * self.settings.guidance_scale
        if self.settings.arm == ARM_B:
            update = update * self.posterior_variance[reverse_step]
            if self.settings.clamp is not None and self.settings.clamp_target == CLAMP_UPDATE:
                update = update.clamp(-self.settings.clamp, self.settings.clamp)
        result = posterior + update
        if (
            self.settings.arm == ARM_B
            and self.settings.clamp is not None
            and self.settings.clamp_target == CLAMP_STATE
        ):
            result = result.clamp(-self.settings.clamp, self.settings.clamp)
        result[:, :REPRESENTATION.history_frames] = fixed_history
        if self.audit is not None:
            self.audit.record(gradient, update, loss, feet_loss)
        return result
