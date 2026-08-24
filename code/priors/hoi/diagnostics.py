"""Registered causal probes for the HOI expert.

Geometry-weight records intentionally contain raw per-checkpoint, per-timestep
measurements only.  Cross-record geometric means, derived weights, calibration
bands and no-go verdicts belong to the later reduction step, not this module.
"""

from __future__ import annotations

import contextlib
import hashlib
import inspect
import json
import math
import os
import random
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ..core.representation import REPRESENTATION
from . import losses as hoi_loss_module
from .losses import (
    P8_CONTACT_HAND_CHANNELS,
    P8_CONTACT_THRESHOLD,
    P8_PALM_JOINTS,
    _reduce_per_hand_global,
    _reduce_per_hand_per_frame,
    hoi_training_losses,
)


_CHANNEL_GROUPS = {
    "root_translation": slice(0, 3),
    "other_joint_positions": slice(3, 84),
    "rotations": slice(84, 216),
    "object": slice(216, 228),
}

DERIVATION_TIMESTEPS = (0, 125, 250, 375, 499)
DERIVATION_SHARDS = (0, 1, 2, 3)
DERIVATION_MODES = ("sealed", "per_hand_per_frame")
DERIVATION_MANIFEST_TOOL_SHA256 = "423163b773d1dad544e6023ba0bd1ac8e15d395c42e8524bf5ceea7378c989e0"
DERIVATION_SPEC_SHA256 = "3e7c2e3730435a12a49cfd12b5ff91b5512925b0ecd74a9078b78e6eb737412a"
DERIVATION_NON_GEOMETRY = ("joint_position", "joint_rotation", "fk", "object_translation",
    "object_rotation", "object_surface", "contact",)
DERIVATION_CANDIDATE_SOURCE = "modes.per_hand_per_frame.aggregate.w_geom_star"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _slice_batch(batch: Mapping[str, object], count: int) -> Dict[str, object]:
    batch_size = int(batch["x"].shape[0])
    return {
        key: (
            value[:count]
            if torch.is_tensor(value) and value.ndim and value.shape[0] == batch_size
            else value
        )
        for key, value in batch.items()
    }


def _geometry_losses(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
    *,
    weight: float,
    mask_mode: str,
    detach_object: bool,
    detach_root: bool,
) -> Dict[str, torch.Tensor]:
    """Evaluate the configured geometry loss with explicit probe controls."""
    return hoi_training_losses(
        prediction,
        batch["x"],
        batch["goals"],
        batch["rest_human_offsets"],
        parents,
        position_minimum,
        position_maximum,
        object_minimum,
        object_maximum,
        batch["terminal_window"],
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
        fk_weight=float(cfg.fk_weight),
        object_surface_weight=float(cfg.object_surface_weight),
        velocity_weight=float(cfg.velocity_weight),
        goal_weight=float(cfg.goal_weight),
        hand_object_contact_weight=float(weight),
        hand_object_contact_hinge=float(cfg.get("hand_object_contact_hinge", 0.0)),
        hand_object_contact_mask_mode=str(mask_mode),
        hand_object_contact_detach_object=bool(detach_object),
        hand_object_contact_detach_root=bool(detach_root),
        fk_foot_temporal_routing=bool(cfg.get("fk_foot_temporal_routing", False)),
        routed_foot_residual_multiplier=float(
            cfg.get("routed_foot_residual_multiplier", 1.0)
        ),
    )


def root_gradient_share_probe(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    training_loader: Iterable[Mapping[str, torch.Tensor]],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
    *,
    checkpoint_path: Path,
    output_path: Path,
    window_count: int = 64,
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Measure the sealed W3 geometry term's share of root supervision.

    The trainer's own ``_move_batch`` and ``_forward_losses`` helpers perform
    the real noising, model forward and objective construction.  Windows are
    consumed in loader order, without sampling or selection.  The probe changes
    no training or evaluation protocol and writes only its caller-named JSON.
    """
    # This import direction is deliberate: the trainer never imports this
    # diagnostics module, so the probe cannot enter the training hot path.
    from train_hoi_prior import _forward_losses, _move_batch

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if window_count <= 0:
        raise ValueError("root-gradient probe window_count must be positive")
    if not checkpoint_path.is_file():
        raise ValueError(f"probe checkpoint does not exist: {checkpoint_path}")
    if (
        float(cfg.get("hand_object_contact_weight", 0.0)) != 3.0
        or float(cfg.get("hand_object_contact_hinge", 0.0)) != 0.0
        or str(cfg.get("hand_object_contact_mask_mode", "sealed")) != "sealed"
        or bool(cfg.get("hand_object_contact_detach_object", False))
        or bool(cfg.get("hand_object_contact_detach_root", False))
    ):
        raise ValueError("root-gradient probe requires the sealed W3 objective settings")

    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)

    group_squared = {name: 0.0 for name in _CHANNEL_GROUPS}
    total_root_squared = 0.0
    non_geometry_root_squared = 0.0
    root_dot = 0.0
    consumed = 0
    batches = 0
    previous_training = model.training
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [] if device.type != "cuda" else [device.index or 0]

    try:
        random.seed(42)
        np.random.seed(42)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(42)
            if device.type == "cuda":
                torch.cuda.manual_seed(42)
            model.train()
            for raw_batch in training_loader:
                if consumed >= window_count:
                    break
                take = min(window_count - consumed, int(raw_batch["x"].shape[0]))
                batch = _move_batch(_slice_batch(raw_batch, take), device)
                captured = []
                hook = model.register_forward_hook(
                    lambda _module, _arguments, output: captured.append(output)
                )
                try:
                    losses = _forward_losses(
                        model,
                        diffusion,
                        batch,
                        parents,
                        position_minimum,
                        position_maximum,
                        object_minimum,
                        object_maximum,
                        cfg,
                        generator=generator,
                    )
                finally:
                    hook.remove()
                if len(captured) != 1:
                    raise AssertionError(
                        f"W3 probe expected one training forward, observed {len(captured)}"
                    )
                prediction = captured[0]
                weight = float(cfg.hand_object_contact_weight)
                g_geom, = torch.autograd.grad(
                    weight * losses["hand_object_contact_geometry"],
                    prediction,
                    retain_graph=True,
                )
                g_total, = torch.autograd.grad(
                    losses["total"], prediction, retain_graph=True,
                )

                detached_losses = _geometry_losses(
                    prediction,
                    batch,
                    parents,
                    position_minimum,
                    position_maximum,
                    object_minimum,
                    object_maximum,
                    cfg,
                    weight=float(cfg.hand_object_contact_weight),
                    mask_mode="sealed",
                    detach_object=bool(
                        cfg.get("hand_object_contact_detach_object", False)
                    ),
                    detach_root=True,
                )
                g_geom_detached, = torch.autograd.grad(
                    weight * detached_losses["hand_object_contact_geometry"],
                    prediction,
                    retain_graph=True,
                )
                if int(torch.count_nonzero(g_geom_detached[..., 0:3])) != 0:
                    raise AssertionError("root-detached geometry gradient is not exactly zero")
                if not torch.equal(
                    g_geom_detached[..., 84:216], g_geom[..., 84:216]
                ):
                    raise AssertionError(
                        "root detach changed a geometry rotation gradient bit"
                    )

                for name, channels in _CHANNEL_GROUPS.items():
                    group_squared[name] += float(
                        g_geom[..., channels].double().square().sum().item()
                    )
                geometry_root = g_geom[..., 0:3].double()
                total_root = g_total[..., 0:3].double()
                non_geometry_root = total_root - geometry_root
                total_root_squared += float(total_root.square().sum().item())
                non_geometry_root_squared += float(
                    non_geometry_root.square().sum().item()
                )
                root_dot += float((geometry_root * total_root).sum().item())
                consumed += take
                batches += 1
    finally:
        model.train(previous_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    if consumed != window_count:
        raise ValueError(
            f"training loader ended after {consumed} windows; {window_count} required"
        )
    geometry_root_norm = group_squared["root_translation"] ** 0.5
    total_root_norm = total_root_squared ** 0.5
    if total_root_norm == 0.0:
        raise ValueError("total objective has zero root gradient; share is undefined")
    cosine_denominator = geometry_root_norm * total_root_norm
    result: Dict[str, object] = {
        "probe": "root_gradient_share_probe",
        "seed": 42,
        "window_count": consumed,
        "batch_count": batches,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "geometry_gradient_l2": {
            name: squared ** 0.5 for name, squared in group_squared.items()
        },
        "root_gradient_share": geometry_root_norm / total_root_norm,
        "non_geometry_root_gradient_l2": non_geometry_root_squared ** 0.5,
        "geometry_total_root_cosine_similarity": (
            root_dot / cosine_denominator if cosine_denominator else 0.0
        ),
        "self_check": {
            "detached_root_gradient_exactly_zero": True,
            "detached_rotation_gradient_bitwise_equal": True,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return result


def _write_probe_json(output_path: Path, result: Mapping[str, object]) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def geometry_term_forward_scale_probe(
    training_loader: Iterable[Mapping[str, torch.Tensor]],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
    *,
    output_path: Path,
    window_count: int = 256,
) -> Dict[str, object]:
    """Measure the unit-weight geometry term at and around its GT target."""
    if window_count <= 0:
        raise ValueError("geometry forward-scale probe window_count must be positive")

    sigma_items = (("0.02", 0.02), ("0.05", 0.05), ("0.10", 0.10))
    floor_sum = 0.0
    perturbed_sums = {key: 0.0 for key, _ in sigma_items}
    consumed = 0
    batches = 0
    total_active_frames = 0
    total_engaged_frames = 0
    engaged_windows = 0
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)
    loss_key_present = True

    try:
        random.seed(42)
        np.random.seed(42)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for raw_batch in training_loader:
                if consumed >= window_count:
                    break
                take = min(window_count - consumed, int(raw_batch["x"].shape[0]))
                batch = _slice_batch(raw_batch, take)
                target = batch["x"]
                floor_losses = _geometry_losses(
                    target,
                    batch,
                    parents,
                    position_minimum,
                    position_maximum,
                    object_minimum,
                    object_maximum,
                    cfg,
                    weight=1.0,
                    mask_mode="sealed",
                    detach_object=bool(
                        cfg.get("hand_object_contact_detach_object", False)
                    ),
                    detach_root=bool(
                        cfg.get("hand_object_contact_detach_root", False)
                    ),
                )
                loss_key_present = loss_key_present and (
                    "hand_object_contact_geometry" in floor_losses
                )
                if not loss_key_present:
                    raise AssertionError(
                        "unit-weight geometry call omitted hand_object_contact_geometry"
                    )
                floor_sum += float(
                    floor_losses["hand_object_contact_geometry"].item()
                ) * take

                for key, sigma in sigma_items:
                    epsilon = torch.randn(
                        target.shape,
                        dtype=target.dtype,
                        device=target.device,
                        generator=generator,
                    )
                    perturbed_losses = _geometry_losses(
                        target + sigma * epsilon,
                        batch,
                        parents,
                        position_minimum,
                        position_maximum,
                        object_minimum,
                        object_maximum,
                        cfg,
                        weight=1.0,
                        mask_mode="sealed",
                        detach_object=bool(
                            cfg.get("hand_object_contact_detach_object", False)
                        ),
                        detach_root=bool(
                            cfg.get("hand_object_contact_detach_root", False)
                        ),
                    )
                    if "hand_object_contact_geometry" not in perturbed_losses:
                        raise AssertionError(
                            "unit-weight geometry call omitted hand_object_contact_geometry"
                        )
                    perturbed_sums[key] += float(
                        perturbed_losses["hand_object_contact_geometry"].item()
                    ) * take

                active = target[:, REPRESENTATION.history_frames:, 228:232]
                engaged = (
                    active[..., P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD
                ).any(dim=-1)
                total_active_frames += int(engaged.numel())
                total_engaged_frames += int(engaged.sum().item())
                engaged_windows += int(engaged.any(dim=-1).sum().item())
                consumed += take
                batches += 1
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    if consumed != window_count:
        raise ValueError(
            f"training loader ended after {consumed} windows; {window_count} required"
        )
    floor = floor_sum / consumed
    perturbed = {key: value / consumed for key, value in perturbed_sums.items()}
    if not math.isfinite(floor):
        raise ValueError("geometry forward-scale floor is non-finite")
    if not all(math.isfinite(value) for value in perturbed.values()):
        raise ValueError("geometry forward-scale perturbation is non-finite")
    sensitivity = {
        key: (perturbed[key] - floor) / (sigma * sigma)
        for key, sigma in sigma_items
    }
    floor_ratio = {
        key: floor / perturbed[key] if perturbed[key] != 0.0 else 0.0
        for key, _ in sigma_items
    }
    active_frames_per_window = total_active_frames // consumed
    coverage = {
        "engaged_frame_fraction": total_engaged_frames / total_active_frames,
        "engaged_window_fraction": engaged_windows / consumed,
        "mean_engaged_frames_per_engaged_window": (
            total_engaged_frames / engaged_windows if engaged_windows else 0.0
        ),
        "active_frames_per_window": active_frames_per_window,
        "total_active_frames": total_active_frames,
        "total_engaged_frames": total_engaged_frames,
        "total_windows": consumed,
    }
    monotone = all(
        perturbed[left] <= perturbed[right]
        for left, right in (("0.02", "0.05"), ("0.05", "0.10"))
    )
    result: Dict[str, object] = {
        "probe": "geometry_term_forward_scale_probe",
        "seed": 42,
        "window_count": consumed,
        "batch_count": batches,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "term_weight_used": 1.0,
        "floor": floor,
        "sigmas": [sigma for _, sigma in sigma_items],
        "perturbed": perturbed,
        "sensitivity": sensitivity,
        "floor_ratio": floor_ratio,
        "coverage": coverage,
        "self_check": {
            "loss_key_present": True,
            "floor_is_finite": True,
            "perturbed_monotone_in_sigma": monotone,
            "coverage_in_unit_interval": (
                0.0 <= coverage["engaged_frame_fraction"] <= 1.0
                and 0.0 <= coverage["engaged_window_fraction"] <= 1.0
            ),
        },
    }
    _write_probe_json(output_path, result)
    return result


def geometry_mask_fix_floor_probe(
    training_loader: Iterable[Mapping[str, torch.Tensor]],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
    *,
    output_path: Path,
    window_count: int = 256,
    batch_size=None,
) -> Dict[str, object]:
    """Compare the sealed geometry floor with both per-hand mask reducers."""
    if window_count <= 0:
        raise ValueError("geometry mask-fix floor probe window_count must be positive")
    if float(cfg.get("hand_object_contact_hinge", 0.0)) != 0.0:
        raise ValueError("geometry mask-fix floor probe requires zero hinge")
    if P8_PALM_JOINTS != (22, 23) or P8_CONTACT_HAND_CHANNELS != slice(0, 2):
        raise AssertionError(
            "P8 palm/contact alignment changed: expected left/right joints 22/23 "
            "aligned with contact channels 0/1"
        )
    for name, tensor in (
        ("parents", parents),
        ("position_minimum", position_minimum),
        ("position_maximum", position_maximum),
        ("object_minimum", object_minimum),
        ("object_maximum", object_maximum),
    ):
        if tensor.device.type != "cpu":
            raise ValueError(f"geometry mask-fix floor probe is CPU-only; {name} is on {tensor.device}")

    sigma_items = (("0.02", 0.02), ("0.05", 0.05), ("0.10", 0.10))
    variant_names = ("sealed", "per_hand_global", "per_hand_per_frame")
    condition_sums = {
        name: {"floor": 0.0, **{key: 0.0 for key, _ in sigma_items}}
        for name in variant_names
    }
    consumed = batches = 0
    total_active_frames = total_engaged_frames = engaged_windows = 0
    total_contacting_entries = total_both_contact_frames = 0
    observed_batch_size = batch_size
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(42)

    def measure_condition(prediction, batch, take, condition):
        with _captured_geometry_loss_call() as capture:
            losses = _geometry_losses(
                prediction, batch, parents, position_minimum, position_maximum,
                object_minimum, object_maximum, cfg, weight=1.0,
                mask_mode="sealed",
                detach_object=bool(cfg.get("hand_object_contact_detach_object", False)),
                detach_root=bool(cfg.get("hand_object_contact_detach_root", False)),
            )
        fk, surface, labels, authoritative = capture.records[0]
        if losses.get("hand_object_contact_geometry") is not authoritative:
            raise AssertionError("training losses did not return the captured geometry-loss scalar")
        for name, tensor in (("geometry_fk", fk), ("predicted_surface", surface), ("contact_ground_truth", labels)):
            if tensor.device.type != "cpu":
                raise ValueError(
                    f"geometry mask-fix floor probe is CPU-only; {name} is on {tensor.device}"
                )
        active = slice(REPRESENTATION.history_frames, None)
        palms = fk[:, active][:, :, P8_PALM_JOINTS, :]
        predicted_surface = surface[:, active]
        batch_count, frames = palms.shape[:2]
        nearest = torch.cdist(
            palms.reshape(batch_count * frames, 2, 3),
            predicted_surface.reshape(batch_count * frames, -1, 3),
        ).min(dim=-1)[0].reshape(batch_count, frames, 2)
        per_hand = labels[:, active, P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD
        engaged = per_hand.any(dim=-1)
        per_frame = nearest.square().mean(dim=-1)
        weight = engaged.to(per_frame)
        recomputed = (per_frame * weight).sum() / weight.sum().clamp_min(1.0)
        authoritative_value = float(authoritative.double().item())
        recomputed_value = float(recomputed.double().item())
        if not math.isclose(
            recomputed_value, authoritative_value, rel_tol=1e-6, abs_tol=1e-12
        ):
            raise AssertionError(
                "sealed geometry mismatch: recomputed "
                f"{recomputed_value}, authoritative {authoritative_value}"
            )
        scalars = {
            "sealed": authoritative,
            "per_hand_global": _reduce_per_hand_global(nearest, per_hand),
            "per_hand_per_frame": _reduce_per_hand_per_frame(nearest, per_hand),
        }
        for name, scalar in scalars.items():
            condition_sums[name][condition] += float(scalar.double().item()) * take
        return per_hand

    try:
        random.seed(42)
        np.random.seed(42)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            for raw_batch in training_loader:
                if consumed >= window_count:
                    break
                for name, tensor in raw_batch.items():
                    if torch.is_tensor(tensor) and tensor.device.type != "cpu":
                        raise ValueError(
                            f"geometry mask-fix floor probe is CPU-only; batch {name} is on {tensor.device}"
                        )
                raw_size = int(raw_batch["x"].shape[0])
                take = min(window_count - consumed, raw_size)
                if observed_batch_size is None:
                    observed_batch_size = raw_size
                batch = _slice_batch(raw_batch, take)
                target = batch["x"]
                per_hand = measure_condition(target, batch, take, "floor")
                for key, sigma in sigma_items:
                    epsilon = torch.randn(
                        target.shape, dtype=target.dtype, device=target.device,
                        generator=generator,
                    )
                    measure_condition(target + sigma * epsilon, batch, take, key)
                engaged = per_hand.any(dim=-1)
                total_active_frames += int(engaged.numel())
                total_engaged_frames += int(engaged.sum().item())
                engaged_windows += int(engaged.any(dim=-1).sum().item())
                total_contacting_entries += int(per_hand.sum().item())
                total_both_contact_frames += int(per_hand.all(dim=-1).sum().item())
                consumed += take
                batches += 1
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    if consumed != window_count:
        raise ValueError(
            f"training loader ended after {consumed} windows; {window_count} required"
        )
    variants = {}
    for name in variant_names:
        floor = condition_sums[name]["floor"] / consumed
        perturbed = {key: condition_sums[name][key] / consumed for key, _ in sigma_items}
        variants[name] = {
            "floor": floor,
            "perturbed": perturbed,
            "floor_ratio": {
                key: floor / perturbed[key] if perturbed[key] != 0.0 else 0.0
                for key, _ in sigma_items
            },
            "sensitivity": {
                key: (perturbed[key] - floor) / (sigma * sigma)
                for key, sigma in sigma_items
            },
            "denominator_name": (
                "contacting_hand_entries" if name == "per_hand_global" else "engaged_frames"
            ),
            "denominator_total": (
                total_contacting_entries if name == "per_hand_global" else total_engaged_frames
            ),
        }

    coverage = {
        "engaged_frame_fraction": total_engaged_frames / total_active_frames,
        "engaged_window_fraction": engaged_windows / consumed,
        "mean_engaged_frames_per_engaged_window": (
            total_engaged_frames / engaged_windows if engaged_windows else 0.0
        ),
        "contacting_entry_fraction": total_contacting_entries / (2 * total_active_frames),
        "both_frame_fraction_of_engaged": (
            total_both_contact_frames / total_engaged_frames if total_engaged_frames else 0.0
        ),
        "active_frames_per_window": total_active_frames // consumed,
        "total_active_frames": total_active_frames,
        "total_engaged_frames": total_engaged_frames,
        "total_engaged_windows": engaged_windows,
        "total_contacting_hand_entries": total_contacting_entries,
        "total_both_contact_frames": total_both_contact_frames,
        "total_windows": consumed,
    }
    violations = []
    monotone_by_variant = {}
    for name, variant in variants.items():
        monotone = True
        for left, right in (("0.02", "0.05"), ("0.05", "0.10")):
            if variant["perturbed"][left] > variant["perturbed"][right]:
                monotone = False
                violations.append({
                    "variant": name, "left_sigma": left, "right_sigma": right,
                    "left_value": variant["perturbed"][left],
                    "right_value": variant["perturbed"][right],
                })
        monotone_by_variant[name] = monotone

    sealed_floor = variants["sealed"]["floor"]
    references = {256: 0.21783776953816414, 568486: 0.06627798145844942}
    reference = references.get(window_count)
    relative_error = (
        abs(sealed_floor - reference) / abs(reference) if reference is not None else None
    )
    sealed_floor_matches_anchor = (
        math.isclose(sealed_floor, reference, rel_tol=1e-6, abs_tol=1e-12)
        if reference is not None
        else None
    )
    if reference is not None and not sealed_floor_matches_anchor:
        raise AssertionError(
            f"sealed floor anchor mismatch: observed {sealed_floor}, reference {reference}"
        )
    denominator_check = (
        variants["sealed"]["denominator_name"] == "engaged_frames"
        and variants["per_hand_per_frame"]["denominator_name"] == "engaged_frames"
        and variants["per_hand_global"]["denominator_name"] == "contacting_hand_entries"
    )
    default_is_sealed = (
        inspect.signature(hoi_loss_module.masked_hand_object_distance_loss)
        .parameters["hand_object_contact_mask_mode"].default == "sealed"
    )
    numeric_values = [
        value for variant in variants.values()
        for value in (
            variant["floor"], *variant["perturbed"].values(),
            *variant["floor_ratio"].values(), *variant["sensitivity"].values(),
        )
    ] + [value for value in coverage.values() if isinstance(value, float)]
    all_finite = all(math.isfinite(value) for value in numeric_values)
    self_check = {
        "all_values_finite": all_finite,
        "variant_denominators_as_specified": denominator_check,
        "default_mode_is_sealed": default_is_sealed,
        "accumulation_dtype": "float64",
    }
    if not all_finite or not denominator_check or not default_is_sealed:
        raise AssertionError(f"geometry mask-fix floor self-check failed: {self_check}")
    floor_a = variants["per_hand_global"]["floor"]
    floor_b = variants["per_hand_per_frame"]["floor"]
    result: Dict[str, object] = {
        "probe": "geometry_mask_fix_floor_probe",
        "seed": 42,
        "window_count": consumed,
        "batch_count": batches,
        "batch_size": observed_batch_size,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[3], text=True
        ).strip(),
        "sigmas": [sigma for _, sigma in sigma_items],
        "variants": variants,
        "coverage": coverage,
        "ratios": {
            "A_over_B_per_batch": floor_a / floor_b if floor_b else None,
            "A_over_B_global_estimate_from_M1": 1.0015483487511598,
            "sealed_over_B_per_batch": sealed_floor / floor_b if floor_b else None,
        },
        "diagnostics_report_only": {
            "perturbed_monotone_in_sigma": monotone_by_variant,
            "perturbed_monotonicity_violations": violations,
        },
        "invariance": {
            # Named for what this probe actually compares: the sealed floor against a
            # pinned anchor, within ``tolerance``.  It is ``null`` when no anchor exists
            # for this window count.  The probe does NOT verify a bitwise property; that
            # is proven by ``tests/hoi/test_losses.py`` (``torch.equal`` on the sealed
            # path) and by the sealed numbers reproducing the sealed R0 artifact exactly.
            "sealed_floor_matches_anchor": sealed_floor_matches_anchor,
            "sealed_floor_observed": sealed_floor,
            "sealed_floor_reference": reference,
            "sealed_floor_relative_error": relative_error,
            "tolerance": 1e-6,
        },
        "self_check": self_check,
    }
    _write_probe_json(output_path, result)
    return result


@contextlib.contextmanager
def _captured_geometry_loss_call():
    """Capture the exact tensors routed through the real geometry loss once."""
    real_loss = hoi_loss_module.masked_hand_object_distance_loss
    capture = SimpleNamespace(invocations=0, records=[])

    def recording_loss(*args, **kwargs):
        capture.invocations += 1
        if len(args) < 3:
            raise AssertionError(
                "geometry palm-decomposition capture requires three positional inputs"
            )
        result = real_loss(*args, **kwargs)
        capture.records.append((args[0], args[1], args[2], result))
        return result

    hoi_loss_module.masked_hand_object_distance_loss = recording_loss
    try:
        yield capture
    finally:
        hoi_loss_module.masked_hand_object_distance_loss = real_loss
        if capture.invocations != 1:
            raise AssertionError(
                "geometry palm-decomposition probe expected exactly one geometry-loss "
                f"invocation per forward, observed {capture.invocations}"
            )


class _PalmRoleAccumulator:
    """Float64 streaming moments for palm distances, with exact integer count."""

    def __init__(self) -> None:
        self.count = 0
        self.sum = 0.0
        self.sumsq = 0.0

    def update(self, values: torch.Tensor) -> float:
        values = values.double()
        batch_sumsq = float(values.square().sum().item())
        self.count += int(values.numel())
        self.sum += float(values.sum().item())
        self.sumsq += batch_sumsq
        return batch_sumsq

    def merge(self, other: "_PalmRoleAccumulator") -> None:
        self.count += other.count
        self.sum += other.sum
        self.sumsq += other.sumsq

    def metrics(self) -> Dict[str, object]:
        if self.count == 0:
            return {
                "count": 0,
                "mean_distance_m": None,
                "mean_squared_m2": None,
                "rms_m": None,
            }
        mean_squared = float(self.sumsq / self.count)
        return {
            "count": self.count,
            "mean_distance_m": float(self.sum / self.count),
            "mean_squared_m2": mean_squared,
            "rms_m": math.sqrt(mean_squared),
        }


def _decompose_palm_geometry_batch(
    fk: torch.Tensor,
    predicted_surface: torch.Tensor,
    contact_ground_truth: torch.Tensor,
) -> Dict[str, object]:
    """Split one real-loss input batch by semantic palm-contact identity."""
    if P8_PALM_JOINTS != (22, 23) or P8_CONTACT_HAND_CHANNELS != slice(0, 2):
        raise AssertionError(
            "P8 palm/contact alignment changed: expected left/right joints 22/23 "
            "aligned with contact channels 0/1"
        )
    for name, tensor in (
        ("geometry_fk", fk),
        ("predicted_surface", predicted_surface),
        ("contact_ground_truth", contact_ground_truth),
    ):
        if tensor.device.type != "cpu":
            raise ValueError(
                f"geometry palm-decomposition probe is CPU-only; {name} is on "
                f"{tensor.device}"
            )

    active = slice(REPRESENTATION.history_frames, None)
    palms = fk[:, active][:, :, P8_PALM_JOINTS, :]
    surface = predicted_surface[:, active]
    batch, frames = palms.shape[:2]
    nearest = torch.cdist(
        palms.reshape(batch * frames, len(P8_PALM_JOINTS), 3),
        surface.reshape(batch * frames, -1, 3),
    ).min(dim=-1)[0].reshape(batch, frames, len(P8_PALM_JOINTS))
    per_hand_contact = (
        contact_ground_truth[:, active, P8_CONTACT_HAND_CHANNELS]
        > P8_CONTACT_THRESHOLD
    )
    left = per_hand_contact[..., 0]
    right = per_hand_contact[..., 1]
    masks = {
        "left_only": left & ~right,
        "right_only": right & ~left,
        "both": left & right,
        "neither": ~left & ~right,
    }
    engaged = left | right

    role_values = {
        "contacting": torch.cat(
            (
                nearest[..., 0][masks["left_only"]],
                nearest[..., 1][masks["right_only"]],
                nearest[masks["both"]].reshape(-1),
            )
        ),
        "free": torch.cat(
            (
                nearest[..., 1][masks["left_only"]],
                nearest[..., 0][masks["right_only"]],
            )
        ),
    }
    category_values = {
        "left_only": {
            "contacting": nearest[..., 0][masks["left_only"]],
            "free": nearest[..., 1][masks["left_only"]],
        },
        "right_only": {
            "contacting": nearest[..., 1][masks["right_only"]],
            "free": nearest[..., 0][masks["right_only"]],
        },
        "both": {
            "contacting": nearest[masks["both"]].reshape(-1),
            "free": nearest.new_empty((0,)),
        },
    }
    nearest_palm_vs_label = {}
    for category, contacting_index, free_index in (
        ("left_only", 0, 1),
        ("right_only", 1, 0),
    ):
        contacting = nearest[..., contacting_index][masks[category]]
        free = nearest[..., free_index][masks[category]]
        agree = contacting < free
        tie = contacting == free
        disagree = contacting > free
        nearest_palm_vs_label[category] = {
            "single_contact_frames": int(contacting.numel()),
            "agree": int(agree.sum().item()),
            "disagree": int(disagree.sum().item()),
            "tie": int(tie.sum().item()),
            "disagree_contacting": contacting[disagree],
            "disagree_free": free[disagree],
        }
    frame_counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
    frame_counts["engaged"] = int(engaged.sum().item())
    frame_counts["active"] = int(engaged.numel())
    return {
        "frame_counts": frame_counts,
        "role_values": role_values,
        "category_values": category_values,
        "nearest_palm_vs_label": nearest_palm_vs_label,
    }


def geometry_term_palm_decomposition_probe(
    training_loader: Iterable[Mapping[str, torch.Tensor]],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
    *,
    output_path: Path,
    window_count: int = 256,
) -> Dict[str, object]:
    """Decompose the R0 geometry floor by semantic contacting/free palm roles."""
    if window_count <= 0:
        raise ValueError("geometry palm-decomposition probe window_count must be positive")
    if float(cfg.get("hand_object_contact_hinge", 0.0)) != 0.0:
        raise ValueError("geometry palm-decomposition probe requires the R0 zero hinge")
    if P8_PALM_JOINTS != (22, 23) or P8_CONTACT_HAND_CHANNELS != slice(0, 2):
        raise AssertionError(
            "P8 palm/contact alignment changed: expected left/right joints 22/23 "
            "aligned with contact channels 0/1"
        )

    category_names = ("left_only", "right_only", "both", "neither")
    role_names = ("contacting", "free")
    frame_counts = {name: 0 for name in category_names}
    pooled = {role: _PalmRoleAccumulator() for role in role_names}
    by_category_values = {
        category: {role: _PalmRoleAccumulator() for role in role_names}
        for category in ("left_only", "right_only", "both")
    }
    nearest_palm_vs_label_counts = {
        category: {
            "single_contact_frames": 0,
            "agree": 0,
            "disagree": 0,
            "tie": 0,
        }
        for category in ("left_only", "right_only")
    }
    disagree_distances = {
        category: {
            "contacting": _PalmRoleAccumulator(),
            "free": _PalmRoleAccumulator(),
        }
        for category in ("left_only", "right_only")
    }
    contribution_sums = {role: 0.0 for role in role_names}
    category_contribution_sums = {
        category: {role: 0.0 for role in role_names}
        for category in ("left_only", "right_only", "both")
    }
    reference_sum = 0.0
    consumed = 0
    batches = 0

    for raw_batch in training_loader:
        if consumed >= window_count:
            break
        take = min(window_count - consumed, int(raw_batch["x"].shape[0]))
        batch = _slice_batch(raw_batch, take)
        if batch["x"].device.type != "cpu":
            raise ValueError("geometry palm-decomposition probe is CPU-only")
        with _captured_geometry_loss_call() as capture:
            losses = _geometry_losses(
                batch["x"],
                batch,
                parents,
                position_minimum,
                position_maximum,
                object_minimum,
                object_maximum,
                cfg,
                weight=1.0,
                mask_mode="sealed",
                detach_object=bool(
                    cfg.get("hand_object_contact_detach_object", False)
                ),
                detach_root=bool(cfg.get("hand_object_contact_detach_root", False)),
            )
        if len(capture.records) != 1:
            raise AssertionError(
                "geometry palm-decomposition capture did not retain exactly one record"
            )
        fk, predicted_surface, contact_ground_truth, captured_scalar = capture.records[0]
        if losses.get("hand_object_contact_geometry") is not captured_scalar:
            raise AssertionError(
                "training losses did not return the captured geometry-loss scalar"
            )
        decomposed = _decompose_palm_geometry_batch(
            fk, predicted_surface, contact_ground_truth
        )
        counts = decomposed["frame_counts"]
        for category in category_names:
            frame_counts[category] += counts[category]
        batch_engaged = counts["engaged"]
        denominator = float(2 * max(batch_engaged, 1))
        for role in role_names:
            values = decomposed["role_values"][role]
            batch_sumsq = pooled[role].update(values)
            contribution_sums[role] += batch_sumsq / denominator * take
        for category in ("left_only", "right_only", "both"):
            for role in role_names:
                values = decomposed["category_values"][category][role]
                batch_sumsq = by_category_values[category][role].update(values)
                category_contribution_sums[category][role] += (
                    batch_sumsq / denominator * take
                )
        for category in ("left_only", "right_only"):
            batch_nearest = decomposed["nearest_palm_vs_label"][category]
            for name in ("single_contact_frames", "agree", "disagree", "tie"):
                nearest_palm_vs_label_counts[category][name] += batch_nearest[name]
            disagree_distances[category]["contacting"].update(
                batch_nearest["disagree_contacting"]
            )
            disagree_distances[category]["free"].update(
                batch_nearest["disagree_free"]
            )
        reference_sum += float(captured_scalar.item()) * take
        consumed += take
        batches += 1

    if consumed != window_count:
        raise ValueError(
            f"training loader ended after {consumed} windows; {window_count} required"
        )
    frame_counts["engaged"] = (
        frame_counts["left_only"] + frame_counts["right_only"] + frame_counts["both"]
    )
    frame_counts["active"] = frame_counts["engaged"] + frame_counts["neither"]
    partition_ok = sum(frame_counts[name] for name in category_names) == frame_counts[
        "active"
    ]
    engaged_ok = frame_counts["engaged"] == (
        frame_counts["left_only"] + frame_counts["right_only"] + frame_counts["both"]
    )
    if not partition_ok or not engaged_ok:
        raise AssertionError("palm contact categories do not reproduce the loss mask")
    if window_count == 256:
        expected_counts = {
            "active": 3584,
            "engaged": 3353,
            "left_only": 1129,
            "right_only": 2224,
            "both": 0,
            "neither": 231,
        }
        mismatches = {
            name: {"expected": expected, "observed": frame_counts[name]}
            for name, expected in expected_counts.items()
            if frame_counts[name] != expected
        }
        if mismatches:
            raise AssertionError(
                "R0 palm-decomposition frame counts changed; mismatches: "
                f"{mismatches}"
            )

    pooled_metrics = {
        role: pooled[role].metrics() for role in role_names
    }
    by_category = {
        category: {
            role: by_category_values[category][role].metrics()
            for role in role_names
        }
        for category in ("left_only", "right_only", "both")
    }
    nearest_palm_vs_label = {}
    nearest_partition_ok = True
    for category in ("left_only", "right_only"):
        counts = nearest_palm_vs_label_counts[category]
        category_partition_ok = (
            counts["agree"] + counts["disagree"] + counts["tie"]
            == counts["single_contact_frames"]
        )
        if counts["single_contact_frames"] != frame_counts[category]:
            raise AssertionError(
                "nearest-palm-vs-label single-contact count does not match "
                f"{category} frame count: {counts['single_contact_frames']} vs "
                f"{frame_counts[category]}"
            )
        nearest_partition_ok = nearest_partition_ok and category_partition_ok
        if not category_partition_ok:
            raise AssertionError(
                "nearest-palm-vs-label categories do not partition "
                f"{category} single-contact frames: {counts}"
            )
        denominator = counts["single_contact_frames"]
        nearest_palm_vs_label[category] = {
            "single_contact_frames": denominator,
            "proportion_denominator": denominator,
            "proportion_denominator_name": "single_contact_frames",
            "agree": {
                "count": counts["agree"],
                "fraction": counts["agree"] / denominator if denominator else None,
            },
            "disagree": {
                "count": counts["disagree"],
                "fraction": counts["disagree"] / denominator if denominator else None,
            },
            "tie": {
                "count": counts["tie"],
                "fraction": counts["tie"] / denominator if denominator else None,
            },
            "disagree_subset": {
                "contacting": disagree_distances[category]["contacting"].metrics(),
                "free": disagree_distances[category]["free"].metrics(),
            },
        }
    total_counts = {
        name: sum(nearest_palm_vs_label_counts[category][name]
                  for category in ("left_only", "right_only"))
        for name in ("single_contact_frames", "agree", "disagree", "tie")
    }
    total_partition_ok = (
        total_counts["agree"] + total_counts["disagree"] + total_counts["tie"]
        == total_counts["single_contact_frames"]
    )
    nearest_partition_ok = nearest_partition_ok and total_partition_ok
    if not total_partition_ok:
        raise AssertionError(
            "nearest-palm-vs-label categories do not partition total "
            f"single-contact frames: {total_counts}"
        )
    total_disagree_distances = {
        role: _PalmRoleAccumulator() for role in role_names
    }
    for role in role_names:
        for category in ("left_only", "right_only"):
            total_disagree_distances[role].merge(disagree_distances[category][role])
    total_denominator = total_counts["single_contact_frames"]
    nearest_palm_vs_label["total"] = {
        "single_contact_frames": total_denominator,
        "proportion_denominator": total_denominator,
        "proportion_denominator_name": "single_contact_frames",
        "agree": {
            "count": total_counts["agree"],
            "fraction": total_counts["agree"] / total_denominator
            if total_denominator else None,
        },
        "disagree": {
            "count": total_counts["disagree"],
            "fraction": total_counts["disagree"] / total_denominator
            if total_denominator else None,
        },
        "tie": {
            "count": total_counts["tie"],
            "fraction": total_counts["tie"] / total_denominator
            if total_denominator else None,
        },
        "disagree_subset": {
            role: total_disagree_distances[role].metrics() for role in role_names
        },
    }
    contributions = {
        role: contribution_sums[role] / consumed for role in role_names
    }
    reconstructed_floor = sum(contributions.values())
    reference_floor = reference_sum / consumed
    absolute_error = abs(reconstructed_floor - reference_floor)
    relative_error = (
        (reconstructed_floor - reference_floor) / reference_floor
        if reference_floor != 0.0
        else reconstructed_floor
    )
    tolerance = 1e-6
    if abs(relative_error) > tolerance:
        raise AssertionError(
            "geometry palm-decomposition failed to reproduce the real loss: "
            f"relative error {relative_error} exceeds {tolerance}"
        )
    shares = {
        role: contributions[role] / reconstructed_floor
        if reconstructed_floor != 0.0
        else 0.0
        for role in role_names
    }
    category_contributions = {}
    for category in ("left_only", "right_only", "both"):
        contact_value = category_contribution_sums[category]["contacting"] / consumed
        free_value = category_contribution_sums[category]["free"] / consumed
        category_contributions[category] = {
            "contacting_contribution": contact_value,
            "free_contribution": free_value,
            "contacting_share": (
                contact_value / reconstructed_floor if reconstructed_floor else 0.0
            ),
            "free_share": free_value / reconstructed_floor if reconstructed_floor else 0.0,
        }

    result: Dict[str, object] = {
        "probe": "geometry_term_palm_decomposition_probe",
        "seed": 42,
        "window_count": consumed,
        "batch_count": batches,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "palm_contact_alignment": {
            "palm_0": {"joint": 22, "side": "left", "contact_channel": 0},
            "palm_1": {"joint": 23, "side": "right", "contact_channel": 1},
            "index_aligned": True,
        },
        "frame_counts": frame_counts,
        "contacting": pooled_metrics["contacting"],
        "free": pooled_metrics["free"],
        "by_category": by_category,
        "nearest_palm_vs_label": nearest_palm_vs_label,
        "contributions": {
            "contacting_contribution": contributions["contacting"],
            "free_contribution": contributions["free"],
            "contacting_share": shares["contacting"],
            "free_share": shares["free"],
            "by_category": category_contributions,
        },
        "reaggregation": {
            "reference_floor": reference_floor,
            "reconstructed_floor": reconstructed_floor,
            "absolute_error": absolute_error,
            "relative_error": relative_error,
            "tolerance": tolerance,
        },
        "self_check": {
            "capture_invocations_exactly_one": True,
            "category_counts_partition_active_frames": partition_ok,
            "engaged_matches_loss_mask": engaged_ok,
            "nearest_palm_vs_label_partitions_single_contact_frames": (
                nearest_partition_ok
            ),
            "reaggregation_within_tolerance": True,
            "accumulation_dtype": "float64",
        },
    }
    _write_probe_json(output_path, result)
    return result


_GEOMETRY_GRADIENT_COMPONENTS = (
    "joint_position",
    "joint_rotation",
    "fk",
    "object_translation",
    "object_rotation",
    "object_surface",
    "hand_object_contact_geometry",
    "total",
)


def _require_plain_forward_path(cfg) -> None:
    from train_hoi_prior import (
        _is_d2ab,
        _is_d2ad,
        _is_d2ag,
        _is_d2z,
        _is_sparse_relation,
    )

    for predicate_name, predicate in (
        ("_is_d2ag", _is_d2ag),
        ("_is_d2ad", _is_d2ad),
        ("_is_sparse_relation", _is_sparse_relation),
        ("_is_d2z", _is_d2z),
        ("_is_d2ab", _is_d2ab),
    ):
        if predicate(cfg):
            raise ValueError(
                "geometry weight-derivation probe requires the plain "
                f"_forward_losses path; {predicate_name}(cfg) is true"
            )


@contextlib.contextmanager
def _pinned_timestep(timestep):
    """Force train_hoi_prior._forward_losses to use one fixed timestep.

    The replacement still calls the real torch.randint first and discards its
    result, so the generator is advanced EXACTLY as the unpatched trainer path
    advances it and the subsequent noise draw is bit-identical.
    """
    import train_hoi_prior

    real_randint = train_hoi_prior.torch.randint
    pin = SimpleNamespace(substitutions=0)

    def pinned_randint(*args, **kwargs):
        drawn = real_randint(*args, **kwargs)
        high = kwargs.get("high", args[1] if len(args) > 1 else None)
        size = kwargs.get("size", args[2] if len(args) > 2 else None)
        caller = inspect.currentframe().f_back
        batch_size = None
        if (
            caller is not None
            and caller.f_globals.get("__name__") == "train_hoi_prior"
            and caller.f_code.co_name == "_forward_losses"
        ):
            batch_size = int(caller.f_locals["clean"].shape[0])
        if (
            high == REPRESENTATION.diffusion_steps
            and size == (batch_size,)
            and tuple(drawn.shape) == (batch_size,)
        ):
            pin.substitutions += 1
            return torch.full_like(drawn, int(timestep))
        return drawn

    train_hoi_prior.torch.randint = pinned_randint
    try:
        yield pin
    finally:
        train_hoi_prior.torch.randint = real_randint


def _per_cell_seed(shard_id: int, timestep: int) -> int:
    return int.from_bytes(hashlib.sha256(f"42:{int(shard_id)}:{int(timestep)}".encode()).digest()[:8], "big"
    ) % (2 ** 63)
def _paired_identity_guard(prediction_sealed, prediction_variant, total_sealed, total_variant) -> bool:
    if prediction_sealed is not prediction_variant:
        raise AssertionError("paired geometry modes did not use the same prediction object")
    if total_sealed is not total_variant:
        raise AssertionError("paired geometry modes did not use the same total gradient object")
    return True
def _require_paired_zero_weight(measurement_mode: str, configured_weight: float) -> None:
    if measurement_mode == "paired_joint" and configured_weight != 0.0:
        _contract_error(
            "E_PAIRED_REQUIRES_ZERO_WEIGHT",
            f"configured_hand_object_contact_weight={configured_weight!r}")
def _rc2_ratio(parameter_space_weight: float, output_space_weight: float) -> Tuple[float, bool]:
    ratio = max(parameter_space_weight / output_space_weight, output_space_weight / parameter_space_weight,)
    return ratio, ratio < 3.0
def _candidate_is_allowed(
    measurement_mode, gates_shared, b_gate, sealed_gate_divergence, rc2_ratios):
    return (measurement_mode == "paired_joint" and all(gates_shared.values())
        and b_gate and not sealed_gate_divergence and all(ratio < 3.0 for ratio in rc2_ratios)
    )
def _geomean(values: Sequence[float]) -> float:
    from .auxiliary_balancing import _geometric_mean
    return _geometric_mean(values)
def _dispersion(values: Sequence[float], ddof: int) -> float:
    values = np.asarray(tuple(values), dtype=np.float64)
    return (float("inf") if len(values) <= ddof or not np.isfinite(values).all()
        or bool((values <= 0).any())
        else float(math.exp(float(np.std(np.log(values), ddof=ddof))))
    )
def _side_dispersion_gates(w_by_cell, shard_ids, timesteps, w_ratio, fail_on_masking=True):
    sides = ("human", "object", "combined")
    def cell(s, t):
        values = w_by_cell.get(s, w_by_cell.get(str(s)))
        return values.get(t, values.get(str(t)))
    ng3s, csds, ng30s, csd0s = ({side: {} for side in sides} for _ in range(4))
    for side in sides:
        for s in shard_ids:
            values = [cell(s, t)[side] for t in timesteps]
            ng3s[side][str(s)], ng30s[side][str(s)] = _dispersion(values, 1), _dispersion(values, 0)
        for t in timesteps:
            values = [cell(s, t)[side] for s in shard_ids]
            csds[side][str(t)], csd0s[side][str(t)] = _dispersion(values, 1), _dispersion(values, 0)
    ng3 = {side: max(ng3s[side].values()) for side in sides}
    csd = {side: max(csds[side].values()) for side in sides}
    ng30 = {side: max(ng30s[side].values()) for side in sides}
    csd0 = {side: max(csd0s[side].values()) for side in sides}
    g4 = {side: ng3[side] < 10.0 for side in sides}
    g5 = {side: csd[side] < 1.5 for side in sides}
    combined = g4["combined"] and g5["combined"]
    sides_pass = all(g4[s] and g5[s] for s in sides[:2])
    if combined and not sides_pass and fail_on_masking:
        _contract_error(
            "E_DERIVATION_SIDE_DISPERSION_MASKED", "combined passes while side dispersion fails"
        )
    verdict = (
        "imbalanced_stop" if not 0.1 <= w_ratio <= 10.0
        else "acceptable" if 1.0 / 3.0 <= w_ratio <= 3.0
        else "imbalanced_review")
    return {"ng3_max_over_shards": ng3, "ng3_pooled": {
            s: _dispersion([cell(a, b)[s] for a in shard_ids for b in timesteps], 1) for s in sides
        },
        "csd_max_over_timesteps": csd, "csd_from_shard_means": {s: _dispersion(
                [_geomean([cell(a, b)[s] for b in timesteps]) for a in shard_ids], 1) for s in sides
        },
        "ddof0_variants": {
            "ng3_max_over_shards": ng30, "ng3_pooled": {
                s: _dispersion([cell(a, b)[s] for a in shard_ids for b in timesteps], 0)
                for s in sides
            },
            "csd_max_over_timesteps": csd0, "csd_from_shard_means": {s: _dispersion(
                    [_geomean([cell(a, b)[s] for b in timesteps]) for a in shard_ids], 0)
                for s in sides
            },
        },
        "G4_ng3": g4, "G5_csd": g5,
        "G6_no_side_masking": not (combined and not sides_pass), "side_balance_verdict": verdict,
        "ng3_by_shard": ng3s, "csd_by_timestep": csds,
    }
def _contract_error(code: str, detail: str) -> None:
    raise ValueError(f"{code}: {detail}")
def _dataset_for_loader(loader):
    dataset = getattr(loader, "dataset", loader)
    while hasattr(dataset, "dataset"):
        dataset = dataset.dataset
    return dataset
def _validate_derivation_manifest(manifest, training_loader, manifest_path=None) -> None:
    if manifest.get("provenance", {}).get("tool_sha256") != DERIVATION_MANIFEST_TOOL_SHA256:
        _contract_error("E_MANIFEST_TOOL_SHA_MISMATCH", "manifest build-tool SHA mismatch")
    dataset = _dataset_for_loader(
        next(iter(training_loader.values()))
        if isinstance(training_loader, Mapping) else training_loader
    )
    try:
        repo, fp = Path(dataset.repo), manifest["dataset_config_fingerprint"]
        actual = {"split_manifest_sha256": _sha256(
                repo / "experiments/splits/omomo_hoi_train_validation_seed42.json"),
            "norm_sha256": _sha256(repo / "data/train/norm.npy"), "total_windows": len(dataset),
            "total_sequences": len(set(dataset.sequence_ids[dataset.indices].tolist())),
            "history_frames": REPRESENTATION.history_frames, "contact_channels": [228, 229],
            "contact_threshold": P8_CONTACT_THRESHOLD}
        expected = {key: fp[key] for key in ("split_manifest_sha256", "norm_sha256", "total_windows",
            "total_sequences", "history_frames", "contact_channels",
            "contact_threshold",)}
    except (AttributeError, KeyError, OSError, TypeError, ValueError):
        _contract_error("E_MANIFEST_DATASET_FINGERPRINT_MISMATCH",
            "dataset metadata or fingerprint files could not be established",
        )
    if actual != expected:
        _contract_error(
            "E_MANIFEST_DATASET_FINGERPRINT_MISMATCH", f"expected {expected!r}, actual {actual!r}"
        )
def _source_provenance() -> Dict[str, object]:
    root = Path(__file__).resolve().parents[3]
    source_paths = {"diagnostics.py": Path(__file__).resolve(),
        "losses.py": root / "code/priors/hoi/losses.py",
        "auxiliary_balancing.py": root / "code/priors/hoi/auxiliary_balancing.py",
        "train_hoi_prior.py": root / "code/train_hoi_prior.py",
        "measure_hoi_geometry_gradient.py": root / "tools/measure_hoi_geometry_gradient.py",
        "build_hoi_gradient_manifest.py": root / "tools/build_hoi_gradient_manifest.py",
        "config_train_hoi_prior_p12.yaml": root / "code/config/config_train_hoi_prior_p12.yaml",
        "d2ai.yaml": root / "code/config/recipe/d2ai.yaml",
    }
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "git_commit": commit, "git_dirty": dirty,
        "tool_sha256": _sha256(root / "tools/build_hoi_gradient_manifest.py"),
        "probe_sha256": _sha256(root / "tools/measure_hoi_geometry_gradient.py"),
        "source_sha256": {name: _sha256(path) for name, path in source_paths.items() if path.is_file()},
    }
def geometry_weight_derivation_probe(
    model: torch.nn.Module,
    diffusion: torch.nn.Module,
    training_loader: Iterable[Mapping[str, torch.Tensor]],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
    *,
    checkpoint_path: Path,
    output_path: Path,
    window_count: int = 256,
    timesteps: tuple[int, ...] = DERIVATION_TIMESTEPS,
    device: Optional[torch.device] = None,
    manifest_path: Optional[Path] = None,
    manifest: Optional[Mapping[str, object]] = None,
    shard_ids: Sequence[int] = DERIVATION_SHARDS,
    measurement_mode: str = "paired_joint",
    mask_mode: Optional[str] = None,
    rc1_loader=None,
    l3_crosscheck: Optional[Mapping[str, object]] = None,
    timing: Optional[Mapping[str, Optional[float]]] = None,
) -> Dict[str, object]:
    """Measure the paired, output-space geometry-weight derivation contract."""
    # This import direction is deliberate: the trainer never imports this
    # diagnostics module, so the probe cannot enter the training hot path.
    from train_hoi_prior import _forward_losses, _move_batch
    checkpoint_path, output_path = Path(checkpoint_path), Path(output_path)
    if not checkpoint_path.is_file():
        raise ValueError(f"probe checkpoint does not exist: {checkpoint_path}")
    _require_plain_forward_path(cfg)
    cfg_mask_mode = str(cfg.get("hand_object_contact_mask_mode", "sealed"))
    if (float(cfg.get("hand_object_contact_hinge", 0.0)) != 0.0
        or cfg_mask_mode not in {"sealed", "per_hand_global", "per_hand_per_frame"}
        or bool(cfg.get("hand_object_contact_detach_object", False))
        or bool(cfg.get("hand_object_contact_detach_root", False))
    ):
        raise ValueError(
            "geometry weight-derivation probe requires zero hinge and attached gradients"
        )
    configured_weight = float(cfg.get("hand_object_contact_weight", 0.0))
    if measurement_mode not in {"paired_joint", "single_mode_l3"}:
        raise ValueError(f"unknown measurement_mode: {measurement_mode}")
    _require_paired_zero_weight(measurement_mode, configured_weight)
    if measurement_mode == "paired_joint" and tuple(timesteps) != DERIVATION_TIMESTEPS:
        raise ValueError(f"paired_joint requires timesteps {DERIVATION_TIMESTEPS!r}")
    if measurement_mode == "single_mode_l3":
        mask_mode = str(mask_mode or cfg_mask_mode)
        if mask_mode not in DERIVATION_MODES:
            raise ValueError("single_mode_l3 requires sealed or per_hand_per_frame")
        measured_modes = (mask_mode,)
    else:
        measured_modes = DERIVATION_MODES
    if manifest is None:
        if manifest_path is None:
            _contract_error("E_MANIFEST_REQUIRED", "weight derivation requires a manifest")
        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    _validate_derivation_manifest(manifest, training_loader, manifest_path)
    sampling = manifest.get("sampling", {})
    expected_windows = int(sampling.get("windows_per_shard", 0))
    if expected_windows != 256 or int(window_count) != expected_windows:
        _contract_error(
            "E_WINDOW_COUNT_MISMATCH",
            f"window_count={window_count}, manifest windows_per_shard={expected_windows}",
        )
    shard_ids = tuple(int(shard) for shard in shard_ids)
    if not shard_ids or len(set(shard_ids)) != len(shard_ids):
        raise ValueError("derivation shard ids must be non-empty and unique")
    shard_records = {
        int(record["shard_id"]): record for record in manifest.get("shards", [])
    }
    if any(shard not in shard_records for shard in shard_ids):
        _contract_error(
            "E_MANIFEST_INDEX_NOT_IN_DATASET", "requested shard is absent from manifest")
    if measurement_mode == "paired_joint" and tuple(shard_ids) != DERIVATION_SHARDS:
        raise ValueError(f"paired_joint requires shards {DERIVATION_SHARDS!r}")
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    if device.type != "cpu":
        raise ValueError("geometry weight-derivation probe is CPU-only")
    if isinstance(training_loader, Mapping):
        loaders = {int(shard): training_loader[int(shard)] for shard in shard_ids}
    elif len(shard_ids) == 1:
        loaders = {shard_ids[0]: training_loader}
    else:
        raise ValueError("paired shard measurement requires one loader per shard")
    for shard, loader in loaders.items():
        if getattr(loader, "batch_size", 16) != 16:
            raise ValueError(f"shard {shard} must use batch_size=16")
        if getattr(loader, "drop_last", False):
            raise ValueError("derivation loaders must not drop partial batches")
    geometry_from_separate_call = True
    timesteps = tuple(int(timestep) for timestep in timesteps)
    generator = torch.Generator(device="cpu")
    shard_cells: Dict[int, Dict[int, Dict[str, object]]] = {shard: {} for shard in shard_ids}
    input_sha256: Dict[str, Dict[str, str]] = {str(shard): {} for shard in shard_ids}
    cell_seed: Dict[str, Dict[str, int]] = {str(shard): {} for shard in shard_ids}
    previous_training = model.training
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    first_cell_seconds = None
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    def empty_accumulator():
        return {
            "nong": {name: 0.0 for name in DERIVATION_NON_GEOMETRY},
            "total": 0.0, "H": 0.0, "O": 0.0, "Hq": 0.0, "Oq": 0.0,
            "counts": {name: 0 for name in ("engaged_frames", "engaged_windows",
                "contacting_hand_entries", "both_contact_frames")},
            "modes": {mode: {
                "root": 0.0, "human_rotation": 0.0,
                "object_translation": 0.0, "object_rotation": 0.0,
                "target": {name: 0.0 for name in ("geometry", "nongeometry", "dot",
                    "geometry_root", "nongeometry_root", "dot_root")},
                "all": 0.0} for mode in measured_modes},
            "rc2": {mode: {name: 0.0 for name in ("H", "O", "G")} for mode in measured_modes},
            "rc2_seconds": 0.0}
    def square_sum(value):
        return float(value.double().square().sum().item())
    def parameter_square(values):
        return sum(square_sum(value) for value in values if value is not None)
    def geometry_terms(prediction, batch):
        common = (prediction, batch, parents, position_minimum, position_maximum,
            object_minimum, object_maximum, cfg)
        kwargs = {
            "weight": 1.0,
            "detach_object": bool(cfg.get("hand_object_contact_detach_object", False)),
            "detach_root": bool(cfg.get("hand_object_contact_detach_root", False)),
        }
        if measurement_mode == "paired_joint":
            return {
                "sealed": _geometry_losses(*common, mask_mode="sealed", **kwargs)["hand_object_contact_geometry"],
                "per_hand_per_frame": _geometry_losses(*common,
                    mask_mode="per_hand_per_frame", **kwargs)["hand_object_contact_geometry"],
            }
        return ({"sealed": _geometry_losses(*common,
                mask_mode="sealed", **kwargs)["hand_object_contact_geometry"]}
            if mask_mode == "sealed" else {"per_hand_per_frame": _geometry_losses(
                *common, mask_mode="per_hand_per_frame", **kwargs)["hand_object_contact_geometry"]}
        )
    def measure_cell(loader, shard, timestep, expected_batch_size=16, parameter_check=False):
        accumulator = empty_accumulator()
        observed_indices = []
        input_digest = hashlib.sha256()
        seed = _per_cell_seed(shard, timestep)
        generator.manual_seed(seed)
        consumed = batches = 0
        captured = []
        def capture_input(_module, args):
            noisy, drawn_timesteps = args[:2]
            input_digest.update(noisy.detach().double().contiguous().cpu().numpy().tobytes())
            input_digest.update(drawn_timesteps.detach().contiguous().cpu().numpy().tobytes())
        pre_hook = model.register_forward_pre_hook(capture_input)
        forward_hook = model.register_forward_hook(
            lambda _module, _arguments, output: captured.append(output)
        )
        try:
            for raw_batch in loader:
                raw_size = int(raw_batch["x"].shape[0])
                if raw_size != expected_batch_size:
                    raise ValueError(
                        f"shard {shard} timestep {timestep} has batch size {raw_size}, "
                        f"expected {expected_batch_size}"
                    )
                if "window_index" not in raw_batch:
                    _contract_error(
                        "E_MANIFEST_INDEX_NOT_IN_DATASET",
                        "manifest-backed batches must expose window_index",
                    )
                observed_indices.extend(int(value) for value in raw_batch["window_index"].tolist())
                batch = _move_batch(raw_batch, device)
                with _pinned_timestep(timestep) as pin:
                    losses = _forward_losses(
                        model, diffusion, batch, parents, position_minimum,
                        position_maximum, object_minimum, object_maximum, cfg,
                        generator=generator,
                    )
                if pin.substitutions != 1:
                    raise AssertionError(
                        "geometry weight probe expected exactly one pinned timestep substitution "
                        f"per forward, observed {pin.substitutions}"
                    )
                if len(captured) != batches + 1:
                    raise AssertionError(
                        "geometry weight probe expected one training forward per batch, "
                        f"observed {len(captured)} after batch {batches}"
                    )
                prediction = captured[-1]
                prediction_sealed = prediction
                prediction_variant = prediction
                mode_losses = geometry_terms(prediction, batch)
                g_total, = torch.autograd.grad(losses["total"], prediction, retain_graph=True)
                g_total_sealed = g_total
                g_total_variant = g_total
                if measurement_mode == "paired_joint":
                    _paired_identity_guard(prediction_sealed, prediction_variant, g_total_sealed, g_total_variant
                    )
                gradients = {}
                for name in DERIVATION_NON_GEOMETRY:
                    gradients[name], = torch.autograd.grad(
                        losses[name], prediction, retain_graph=True
                    )
                    accumulator["nong"][name] += square_sum(gradients[name])
                accumulator["total"] += square_sum(g_total)
                h_gradient = gradients["joint_position"] + gradients["joint_rotation"]
                o_gradient = gradients["object_translation"] + gradients["object_rotation"]
                accumulator["H"] += square_sum(h_gradient)
                accumulator["O"] += square_sum(o_gradient)
                accumulator["Hq"] += square_sum(gradients["joint_position"]) + square_sum(
                    gradients["joint_rotation"])
                accumulator["Oq"] += square_sum(gradients["object_translation"]) + square_sum(
                    gradients["object_rotation"])
                active = batch["x"][:, REPRESENTATION.history_frames:, 228:230] \
                    > P8_CONTACT_THRESHOLD
                accumulator["counts"]["engaged_frames"] += int(active.any(dim=-1).sum().item())
                accumulator["counts"]["engaged_windows"] += int(
                    active.any(dim=-1).any(dim=-1).sum().item())
                accumulator["counts"]["contacting_hand_entries"] += int(active.sum().item())
                accumulator["counts"]["both_contact_frames"] += int(active.all(dim=-1).sum().item())
                if parameter_check:
                    rc2_started = time.perf_counter()
                    if not parameters:
                        _contract_error(
                            "E_RC2_PARAMETER_SPACE_UNAVAILABLE", "model has no trainable parameters"
                        )
                    for name, value in (
                        ("H", losses["joint_position"] + losses["joint_rotation"]),
                        ("O", losses["object_translation"] + losses["object_rotation"]),
                    ):
                        value_square = parameter_square(torch.autograd.grad(
                                value, parameters, retain_graph=True, allow_unused=True))
                        for mode in mode_losses:
                            accumulator["rc2"][mode][name] += value_square
                    for mode, loss in mode_losses.items():
                        accumulator["rc2"][mode]["G"] += parameter_square(torch.autograd.grad(
                                loss, parameters, retain_graph=True, allow_unused=True))
                    accumulator["rc2_seconds"] += time.perf_counter() - rc2_started
                for mode, loss in mode_losses.items():
                    gradient, = torch.autograd.grad(loss, prediction, retain_graph=True)
                    if int(torch.count_nonzero(gradient[..., 3:84])) != 0:
                        _contract_error(
                            "E_DERIVATION_SUPPORT_ASSERTION", "A1 failed: channels 3:84")
                    if int(torch.count_nonzero(gradient[..., 228:232])) != 0:
                        _contract_error(
                            "E_DERIVATION_SUPPORT_ASSERTION", "A2 failed: channels 228:232")
                    if int(torch.count_nonzero(gradient[:, 0:2, :])) != 0:
                        _contract_error(
                            "E_DERIVATION_SUPPORT_ASSERTION", "A3 failed: history frames")
                    nongeometry = g_total - configured_weight * gradient
                    stats = accumulator["modes"][mode]
                    stats["root"] += square_sum(gradient[..., 0:3])
                    stats["human_rotation"] += square_sum(gradient[..., 84:216])
                    stats["object_translation"] += square_sum(gradient[..., 216:219])
                    stats["object_rotation"] += square_sum(gradient[..., 219:228])
                    stats["all"] += square_sum(gradient)
                    target = gradient[..., 84:228].double()
                    target_nong = nongeometry[..., 84:228].double()
                    target_root = gradient[..., 0:228].double()
                    target_nong_root = nongeometry[..., 0:228].double()
                    stats["target"]["geometry"] += float(target.square().sum().item())
                    stats["target"]["nongeometry"] += float(target_nong.square().sum().item())
                    stats["target"]["dot"] += float((target * target_nong).sum().item())
                    stats["target"]["geometry_root"] += float(target_root.square().sum().item())
                    stats["target"]["nongeometry_root"] += float(
                        target_nong_root.square().sum().item())
                    stats["target"]["dot_root"] += float(
                        (target_root * target_nong_root).sum().item())
                consumed += raw_size
                batches += 1
        finally:
            pre_hook.remove()
            forward_hook.remove()
        expected_indices = sorted(
            int(value) for value in shard_records[int(shard)]["window_indices"])
        if consumed != window_count or observed_indices != expected_indices:
            _contract_error(
                "E_MANIFEST_INDEX_NOT_IN_DATASET",
                f"shard {shard} consumed {consumed} windows in unexpected order",
            )
        return accumulator, input_digest.hexdigest(), seed, batches
    try:
        random.seed(42)
        np.random.seed(42)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            model.train()
            for shard in shard_ids:
                for timestep in timesteps:
                    started = time.perf_counter()
                    accumulator, digest, seed, batches = measure_cell(
                        loaders[shard], shard, timestep, parameter_check=(
                            measurement_mode == "paired_joint" and shard == 0 and timestep == 250
                        )
                    )
                    if first_cell_seconds is None:
                        first_cell_seconds = time.perf_counter() - started
                    shard_cells[shard][timestep] = accumulator
                    input_sha256[str(shard)][str(timestep)] = digest
                    cell_seed[str(shard)][str(timestep)] = seed
    finally:
        model.train(previous_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    rc1_ratios = {mode: None for mode in DERIVATION_MODES}
    rc1_delta_seconds = None
    if measurement_mode == "paired_joint" and rc1_loader is not None:
        model.train()
        rc1_started = time.perf_counter()
        rc1_accumulator, _rc1_digest, _rc1_seed, _rc1_batches = measure_cell(
            rc1_loader, 0, 250, expected_batch_size=32, parameter_check=False
        )
        rc1_delta_seconds = time.perf_counter() - rc1_started
        model.train(previous_training)
        base = shard_cells[0][250]
        for mode in DERIVATION_MODES:
            base_w = math.sqrt(base["H"] * base["O"]) / math.sqrt(base["modes"][mode]["all"])
            rc1_w = math.sqrt(rc1_accumulator["H"] * rc1_accumulator["O"]
            ) / math.sqrt(rc1_accumulator["modes"][mode]["all"])
            rc1_ratios[mode] = rc1_w / base_w
    mode_cells: Dict[str, Dict[int, Dict[int, Dict[str, float]]]] = {
        mode: {shard: {} for shard in shard_ids} for mode in measured_modes
    }
    shared_by_shard = {}
    modes_by_name = {mode: {"per_shard": {}} for mode in measured_modes}
    report_ng2 = {"target_channel": {}, "target_channel_with_root": {}}
    for shard in shard_ids:
        shared_by_shard[str(shard)] = {name: {} for name in (
            "gradient_l2_nongeometry", "reference_norms", "engaged_counts",
            "human_side_l2", "object_side_l2", "p0_calibration")}
        for timestep in timesteps:
            key = str(timestep)
            acc = shard_cells[shard][timestep]
            H, O = math.sqrt(acc["H"]), math.sqrt(acc["O"])
            Hq, Oq = math.sqrt(acc["Hq"]), math.sqrt(acc["Oq"])
            counts = acc["counts"]
            for name, value in (("H", H), ("O", O)):
                if not math.isfinite(value) or value <= 0:
                    _contract_error(
                        "E_DERIVATION_NONPOSITIVE_NORM",
                        f"shard {shard} timestep {timestep} {name}={value!r}; "
                        f"engaged_frames={counts['engaged_frames']} "
                        f"engaged_windows={counts['engaged_windows']} "
                        f"contacting_hand_entries={counts['contacting_hand_entries']}",
                    )
            H_error, O_error = abs(H - Hq) / H, abs(O - Oq) / O
            if H_error > 1e-12 or O_error > 1e-12:
                _contract_error(
                    "E_DERIVATION_SUPPORT_ASSERTION", "reference tensor sum differs from quadrature"
                )
            gradient_l2_nongeometry = {
                name: math.sqrt(acc["nong"][name]) for name in DERIVATION_NON_GEOMETRY}
            gradient_l2_nongeometry["total"] = math.sqrt(acc["total"])
            shared_by_shard[str(shard)]["gradient_l2_nongeometry"][key] = gradient_l2_nongeometry
            shared_by_shard[str(shard)]["reference_norms"][key] = {
                "H": H, "O": O, "H_quadrature": Hq, "O_quadrature": Oq,
                "H_quadrature_relative_error": H_error, "O_quadrature_relative_error": O_error}
            shared_by_shard[str(shard)]["engaged_counts"][key] = acc["counts"]
            nong = shared_by_shard[str(shard)]["gradient_l2_nongeometry"][key]
            shared_by_shard[str(shard)]["human_side_l2"][key] = math.sqrt(sum(
                acc["nong"][name] for name in ("joint_position", "joint_rotation", "fk")))
            shared_by_shard[str(shard)]["object_side_l2"][key] = math.sqrt(sum(
                acc["nong"][name] for name in (
                    "object_translation", "object_rotation", "object_surface")))
            shared_by_shard[str(shard)]["p0_calibration"][key] = {
                "fk_over_object_surface": nong["fk"] / nong["object_surface"]}
            for mode in measured_modes:
                stats = acc["modes"][mode]
                Gh = math.sqrt(stats["root"] + stats["human_rotation"])
                Go = math.sqrt(stats["object_translation"] + stats["object_rotation"])
                G = math.sqrt(stats["all"])
                for name, value in (("G_human", Gh), ("G_object", Go), ("G", G)):
                    if not math.isfinite(value) or value <= 0:
                        counts = acc["counts"]
                        _contract_error(
                            "E_DERIVATION_NONPOSITIVE_NORM",
                            f"shard {shard} timestep {timestep} {name}={value!r}; "
                            f"engaged_frames={counts['engaged_frames']} "
                            f"engaged_windows={counts['engaged_windows']} "
                            f"contacting_hand_entries={counts['contacting_hand_entries']}",
                        )
                pythagoras_error = abs(G * G - Gh * Gh - Go * Go) / (G * G)
                if pythagoras_error > 1e-12:
                    _contract_error(
                        "E_DERIVATION_SUPPORT_ASSERTION", "A4 pythagoras identity failed")
                mode_cells[mode][shard][timestep] = {
                    "H": H, "O": O, "G_human": Gh, "G_object": Go, "G": G,
                    "root": math.sqrt(stats["root"]),
                    "human_rotation": math.sqrt(stats["human_rotation"]),
                    "object_translation": math.sqrt(stats["object_translation"]),
                    "object_rotation": math.sqrt(stats["object_rotation"]),
                    "pythagoras_relative_error": pythagoras_error}
                target = stats["target"]
                geom_l2, nong_l2 = math.sqrt(target["geometry"]), math.sqrt(target["nongeometry"])
                geom_root, nong_root = (
                    math.sqrt(target["geometry_root"]), math.sqrt(target["nongeometry_root"]))
                modes_by_name[mode]["per_shard"].setdefault(
                    str(shard), {"geometry_by_channel": {}, "gradient_l2": {},
                                 "root_channel_geometry_l2": {}, "target_channel": {},
                                 "target_channel_with_root": {}})
                entry = modes_by_name[mode]["per_shard"][str(shard)]
                entry["geometry_by_channel"][key] = {"root_translation": math.sqrt(stats["root"]),
                    "human_rotation": math.sqrt(stats["human_rotation"]),
                    "object_translation": math.sqrt(stats["object_translation"]),
                    "object_rotation": math.sqrt(stats["object_rotation"]),
                    "G_human": Gh, "G_object": Go, "G_all": G,
                    "pythagoras_relative_error": pythagoras_error}
                entry["gradient_l2"][key] = {"hand_object_contact_geometry": G}
                entry["root_channel_geometry_l2"][key] = math.sqrt(stats["root"])
                target_denominator = geom_l2 * nong_l2
                root_denominator = geom_root * nong_root
                entry["target_channel"][key] = {"geometry_l2": geom_l2, "nongeometry_l2": nong_l2,
                    "cosine_similarity": stats["target"]["dot"] / target_denominator
                    if target_denominator else 0.0}
                entry["target_channel_with_root"][key] = {
                    "geometry_l2": geom_root, "nongeometry_l2": nong_root,
                    "cosine_similarity": stats["target"]["dot_root"] / root_denominator
                    if root_denominator else 0.0}
                report_ng2["target_channel"].setdefault(
                    mode, {}).setdefault(str(shard), {})[key] = geom_l2 / nong_l2 \
                    if nong_l2 else 0.0
                report_ng2["target_channel_with_root"].setdefault(
                    mode, {}).setdefault(str(shard), {})[key] = geom_root / nong_root \
                    if nong_root else 0.0
    gates = {}
    aggregates = {}
    for mode in measured_modes:
        cells = mode_cells[mode]
        w_by_cell = {
            str(shard): {str(timestep): {"human": cells[shard][timestep]["H"]
                / cells[shard][timestep]["G_human"], "object": cells[shard][timestep]["O"]
                / cells[shard][timestep]["G_object"], "combined": math.sqrt(
                    cells[shard][timestep]["H"] * cells[shard][timestep]["O"]
                ) / cells[shard][timestep]["G"]
            } for timestep in timesteps}
            for shard in shard_ids}
        shard_derivation = {}
        for shard in shard_ids:
            by_t = {side: {str(t): w_by_cell[str(shard)][str(t)][side] for t in timesteps}
                for side in ("human", "object", "combined")}
            shard_derivation[str(shard)] = {
                "w_human": _geomean(tuple(by_t["human"].values())),
                "w_object": _geomean(tuple(by_t["object"].values())),
                "w_geom_star": _geomean(tuple(by_t["combined"].values())),
                "w_ratio": _geomean(tuple(by_t["human"].values())) / _geomean(
                    tuple(by_t["object"].values())),
                "w_by_timestep": by_t}
        aggregate = {"w_human": _geomean([shard_derivation[str(s)]["w_human"] for s in shard_ids]),
            "w_object": _geomean([shard_derivation[str(s)]["w_object"] for s in shard_ids]),
            "w_geom_star": _geomean([shard_derivation[str(s)]["w_geom_star"] for s in shard_ids]),
            "w_ratio": _geomean(
                [shard_derivation[str(s)]["w_human"] for s in shard_ids]) / _geomean(
                [shard_derivation[str(s)]["w_object"] for s in shard_ids]),
            "w_by_shard": {side: {str(s): shard_derivation[str(s)][
                f"w_{side}" if side != "combined" else "w_geom_star"]
                for s in shard_ids} for side in ("human", "object", "combined")},
            "w_by_cell": w_by_cell}
        gate_values = _side_dispersion_gates(
            w_by_cell, shard_ids, timesteps, aggregate["w_ratio"],
            mode == "per_hand_per_frame")
        numerator = {side: [ math.sqrt(cells[s][t]["H"] * cells[s][t]["O"])
            if side == "combined" else (cells[s][t]["H"]
            if side == "human" else cells[s][t]["O"]) for s in shard_ids for t in timesteps]
            for side in ("human", "object", "combined")}
        denominator = {side: [ cells[s][t]["G"] if side == "combined" else (cells[s][t]["G_human"]
            if side == "human" else cells[s][t]["G_object"]) for s in shard_ids for t in timesteps]
            for side in ("human", "object", "combined")}
        reduction_errors = [ abs(aggregate[key] - _geomean([w_by_cell[str(s)][str(t)][side]
                for s in shard_ids for t in timesteps])) / aggregate[key]
            for side, key in (("human", "w_human"), ("object", "w_object"),
                              ("combined", "w_geom_star"))]
        reduction_errors += [ abs(aggregate[key] - _geomean(numerator[side])
                / _geomean(denominator[side])) / aggregate[key]
            for side, key in (("human", "w_human"), ("object", "w_object"),
                              ("combined", "w_geom_star"))]
        aggregate.update({"ng3_max_over_shards": gate_values["ng3_max_over_shards"],
            "ng3_pooled": gate_values["ng3_pooled"],
            "csd_max_over_timesteps": gate_values["csd_max_over_timesteps"],
            "csd_from_shard_means": gate_values["csd_from_shard_means"],
            "ddof0_variants": gate_values["ddof0_variants"],
            "reduction_identity_relative_error": max(reduction_errors),
            "side_balance_verdict": gate_values["side_balance_verdict"]})
        rc2_ok = None
        if measurement_mode == "paired_joint" and 0 in shard_ids and 250 in timesteps:
            rc2_values = {}
            for rc_mode in measured_modes:
                rc = shard_cells[0][250]["rc2"][rc_mode]
                parameter_w = math.sqrt(rc["H"] * rc["O"]) ** 0.5 / math.sqrt(rc["G"])
                output_w = w_by_cell["0"]["250"]["combined"]
                ratio, _ = _rc2_ratio(parameter_w, output_w)
                rc2_values[rc_mode] = (parameter_w, ratio)
            rc2_ok = all(value[1] < 3.0 for value in rc2_values.values())
        gates[mode] = {"G2_finite_positive": True,
            "G4_ng3": gate_values["G4_ng3"],
            "G5_csd": gate_values["G5_csd"],
            "G6_no_side_masking": gate_values["G6_no_side_masking"],
            "G7_side_ratio": gate_values["side_balance_verdict"] == "acceptable",
            "G8_parameter_space_crosscheck": rc2_ok,
            "all_passed": all(gate_values["G4_ng3"].values()) \
                and all(gate_values["G5_csd"].values()) \
                and gate_values["G6_no_side_masking"] \
                and gate_values["side_balance_verdict"] == "acceptable" and rc2_ok is not False,
            "report_only": mode == "sealed",
        }
        aggregates[mode] = aggregate
        modes_by_name[mode]["mask_mode"] = mode
        modes_by_name[mode]["is_candidate_source"] = mode == "per_hand_per_frame"
        modes_by_name[mode]["aggregate"] = aggregate
        modes_by_name[mode]["gates"] = gates[mode]
        for shard in shard_ids:
            modes_by_name[mode]["per_shard"][str(shard)]["derivation"] = {
                **shard_derivation[str(shard)],
                "ng3": {
                    side: _dispersion(
                        list(shard_derivation[str(shard)]["w_by_timestep"][side].values()), 1)
                    for side in ("human", "object", "combined")
                },
                "ng3_ddof0": {
                    side: _dispersion(
                        list(shard_derivation[str(shard)]["w_by_timestep"][side].values()), 0)
                    for side in ("human", "object", "combined")
                },
            }
    rc2_report = {mode: {"parameter": None, "ratio": None} for mode in measured_modes}
    if measurement_mode == "paired_joint" and 0 in shard_ids and 250 in timesteps:
        for mode in measured_modes:
            rc = shard_cells[0][250]["rc2"][mode]
            parameter_w = math.sqrt(rc["H"] * rc["O"]) ** 0.5 / math.sqrt(rc["G"])
            output_w = aggregates[mode]["w_by_cell"]["0"]["250"]["combined"]
            ratio, _ = _rc2_ratio(parameter_w, output_w)
            rc2_report[mode] = {"parameter": parameter_w, "ratio": ratio}
    l3 = dict(l3_crosscheck or {})
    l3.setdefault("performed", False)
    l3.setdefault("artifacts", [])
    l3.setdefault("input_sha256_equal", False)
    l3.setdefault("cell_seed_equal", False)
    l3.setdefault("nongeometry_norms_bitwise_equal", False)
    l3.setdefault("geometry_matches_paired_joint", {mode: False for mode in DERIVATION_MODES})
    l3_passed = bool( l3["performed"] and l3["input_sha256_equal"] and l3["cell_seed_equal"]
        and l3["nongeometry_norms_bitwise_equal"]
        and all(l3["geometry_matches_paired_joint"].get(mode, False)
                for mode in DERIVATION_MODES)) if measurement_mode == "paired_joint" else False
    manifest_sha = _sha256(Path(manifest_path)) if manifest_path is not None else str(
        manifest.get("manifest_sha256", ""))
    gates_shared = {
        "G1_manifest": bool(manifest.get("coverage", {}).get("accepted")) \
            and bool(manifest.get("allocation_quantization_check", {}).get("accepted")) \
            and all(bool(record.get("coverage", {}).get("accepted"))
                    and bool(record.get("allocation_quantization_check", {}).get("accepted"))
                    for record in shard_records.values()),
        "G3_support_assertions": True,
        "G9_pairing": l3_passed if measurement_mode == "paired_joint" else False,
        "G10_provenance": _source_provenance().get("git_dirty") is False,
    }
    sealed_gate_divergence = (
        measurement_mode == "paired_joint" and gates["sealed"]["all_passed"] is False
        and gates["per_hand_per_frame"]["all_passed"] is True)
    blocked = []
    if not all(gates_shared.values()):
        blocked.extend(name for name, value in gates_shared.items() if not value)
    if measurement_mode == "paired_joint" and not gates["per_hand_per_frame"]["all_passed"]:
        blocked.extend(name for name, value in gates["per_hand_per_frame"].items()
                       if name.startswith("G") and value is False)
    candidate_produced = _candidate_is_allowed(
        measurement_mode,
        gates_shared,
        gates["per_hand_per_frame"]["all_passed"] if measurement_mode == "paired_joint" else False,
        sealed_gate_divergence,
        [value["ratio"] for value in rc2_report.values() if value["ratio"] is not None],
    ) and not blocked
    candidate = {"source": DERIVATION_CANDIDATE_SOURCE,
        "hand_object_contact_weight": (
            aggregates.get("per_hand_per_frame", {}).get("w_geom_star")
            if candidate_produced else None),
        "mask_mode": "per_hand_per_frame", "produced": candidate_produced,
        "blocked_by": blocked, "sealed_gate_divergence": sealed_gate_divergence,
        "parameter_space_unverified_cells": 19, "training_authorized": False}
    if measurement_mode == "paired_joint" and any(
        value["ratio"] >= 3.0 for value in rc2_report.values()):
        candidate["blocked_by"].append("E_RC2_PARAMETER_SPACE_DIVERGENCE")
        candidate["produced"] = False
        candidate["hand_object_contact_weight"] = None
    mode_names = list(measured_modes)
    bp2 = False
    if len(mode_names) == 2:
        bp2 = _geomean([mode_cells["per_hand_per_frame"][s][t]["G"]
            for s in shard_ids for t in timesteps]) < _geomean([
            mode_cells["sealed"][s][t]["G"]
            for s in shard_ids for t in timesteps])
    timing_values = dict(timing or {})
    t_setup = float(timing_values.get("t_setup_seconds") or 0.0)
    t_cell = float(timing_values.get("t_cell_plain_seconds") or first_cell_seconds or 0.0)
    t_rc1 = timing_values.get("t_rc1_delta_seconds")
    if t_rc1 is None:
        t_rc1 = rc1_delta_seconds
    t_rc2 = timing_values.get("t_rc2_delta_seconds")
    if t_rc2 is None and measurement_mode == "paired_joint":
        t_rc2 = (
            shard_cells[0][250]["rc2_seconds"]
            if 0 in shard_cells and 250 in shard_cells[0] else None)
    t_l3 = timing_values.get("t_l3_total_seconds")
    projected = (
        t_setup + 20.0 * t_cell + float(t_rc1 or 0.0)
        + float(t_rc2 or 0.0) + float(t_l3 or 0.0))
    result: Dict[str, object] = {
        "probe": "geometry_weight_derivation_probe", "seed": 42,
        "window_count": window_count, "batch_count": 16,
        "timesteps": list(timesteps), "timestep_seam": "pinned_real_forward_losses",
        "measurement_mode": measurement_mode, "measured_mask_modes": mode_names,
        "hand_object_contact_mask_mode": cfg_mask_mode, "cfg_mask_mode_selects_measurement": False,
        "configured_hand_object_contact_weight": configured_weight,
        "geometry_from_separate_call": geometry_from_separate_call,
        "spec_sha256": DERIVATION_SPEC_SHA256, "probe_sha256": _source_provenance()["probe_sha256"],
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "sampling": {
            "manifest_path": str(Path(manifest_path).resolve()) if manifest_path else None,
            "manifest_sha256": manifest_sha, "shards": list(shard_ids),
            "windows_per_shard": expected_windows,
            "windows_total": expected_windows * len(shard_ids),
            "batch_size": 16, "batches_per_cell": 16, "partial_batches": 0,
            "window_index_space": "language_motion_dict_window_index",
            "consumption_order": "ascending_manifest_window_index",
            "dataset_fingerprint_verified": True,
            "manifest_coverage_revalidated": {
                "both_frame_fraction_of_engaged_union": manifest.get(
                    "coverage", {}).get("both_frame_fraction_of_engaged"),
                "corpus_reference": manifest.get("coverage", {}).get("corpus_reference"),
                "per_shard": {str(s): shard_records[s].get("coverage", {}).get(
                    "both_frame_fraction_of_engaged") for s in shard_ids},
                "allocation_quantization_engaged_window_fraction": manifest.get(
                    "allocation_quantization_check", {}).get("engaged_window_fraction"),
                "accepted": gates_shared["G1_manifest"],
            },
        },
        "pairing": {
            "input_sha256": input_sha256, "cell_seed": cell_seed,
            "timesteps": list(timesteps), "L1_share_manifest": True,
            "L1_share_noise": True, "L1_share_timesteps": True,
            "L1_note": "tautological under the single-forward design; kept as a refactor guard",
            "L2_prediction_is_identical_object": True,
            "L2_total_gradient_is_identical_object": True,
            "L3_independent_invocation_crosscheck": {
                "performed": bool(l3.get("performed")), "cell": {"shard": 0, "timestep": 250},
                "artifacts": l3.get("artifacts", []),
                "input_sha256_equal": bool(l3.get("input_sha256_equal")),
                "cell_seed_equal": bool(l3.get("cell_seed_equal")),
                "nongeometry_norms_bitwise_equal": bool(
                    l3.get("nongeometry_norms_bitwise_equal")),
                "geometry_matches_paired_joint": l3.get(
                    "geometry_matches_paired_joint", {}),
                "note": (
                    "L3 artifacts are reproducibility evidence only; their derivation/aggregate/candidate "
                    "blocks must never be cited"
                ),
            },
        },
        "shared": {"per_shard": shared_by_shard},
        "modes": modes_by_name,
        "gates_shared": gates_shared,
        "candidate": candidate,
        "report_only": {
            "BP1_fk_over_object_surface_reference": 1.3367948864888777,
            "BP1_reference_interval": [1.069435909191102, 1.6709936081110972],
            "BP1_in_interval": None,
            "BP1_note": (
                "report-only for two reasons: no measured legacy anchor exists, and "
                "the reference is a parameter-space ratio while the probe measures output space"
            ),
            "BP2_variant_geometry_below_sealed": bp2,
            "BP3_w_geom_star_B_over_sealed": (aggregates["per_hand_per_frame"]["w_geom_star"]
                / aggregates["sealed"]["w_geom_star"]
                if len(mode_names) == 2 else None),
            "BP3_confidence": "low", "NG2": report_ng2["target_channel"],
            "NG2_with_root": report_ng2["target_channel_with_root"],
            "RC1_batch_size_invariance": {
                "cell": {"shard": 0, "timestep": 250}, "w_geom_star_ratio_32_over_16": rc1_ratios},
            "RC2_parameter_space": {
                "cell": {"shard": 0, "timestep": 250},
                "w_geom_star_parameter_space": {
                    mode: rc2_report[mode]["parameter"] for mode in measured_modes},
                "ratio_to_output_space": {
                    mode: rc2_report[mode]["ratio"] for mode in measured_modes},
                "stop_threshold": 3.0,
                "scope_note": (
                    "one cell of twenty; passing does not establish agreement on the "
                    "other 19 cells and is not evidence of "
                    "training validity"
                ),
            },
        },
        "timing": {
            "t_setup_seconds": t_setup, "t_cell_plain_seconds": t_cell,
            "t_rc1_delta_seconds": t_rc1, "t_rc2_delta_seconds": t_rc2,
            "t_l3_total_seconds": t_l3, "projected_total_seconds": projected,
            "projection_formula": "t_setup + 20*t_cell_plain + t_rc1 + t_rc2 + t_l3",
            "forbidden_projection_note": "never 20 x (a wall clock that includes RC1/RC2/L3 or setup)",
            "peak_rss_kb": int(
                __import__("resource").getrusage(__import__("resource").RUSAGE_SELF).ru_maxrss),
            "omp_num_threads": int(os.environ.get("OMP_NUM_THREADS", "0") or 0),
            "device": "cpu",
        },
        "self_check": {
            "plain_forward_path_verified": True,
            "geometry_gradient_on_joint_positions_3_84_exactly_zero": True,
            "geometry_gradient_on_contact_228_232_exactly_zero": True,
            "geometry_gradient_on_history_frames_exactly_zero": True,
            "geometry_pythagoras_holds": True,
            "reference_norm_tensor_sum_equals_quadrature": True,
            "reduction_orders_agree": all(
                aggregates[mode]["reduction_identity_relative_error"] <= 1e-12
                for mode in measured_modes),
            "single_forward_per_batch": True,
            "both_modes_from_one_prediction": measurement_mode == "paired_joint",
            "configured_weight_is_zero": configured_weight == 0.0,
            "mask_mode_recorded": True, "no_silent_sealed_fallback": True,
            "all_values_finite": True, "accumulation_dtype": "float64",
            "pinned_timestep_substitutions_per_forward": 1},
        "provenance": _source_provenance(),
    }
    _write_probe_json(output_path, result)
    return result
