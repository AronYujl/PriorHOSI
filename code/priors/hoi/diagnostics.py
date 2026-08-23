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
import random
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from ..core.representation import REPRESENTATION
from . import losses as hoi_loss_module
from .losses import (
    P8_CONTACT_HAND_CHANNELS,
    P8_CONTACT_THRESHOLD,
    P8_PALM_JOINTS,
    hoi_training_losses,
)


_CHANNEL_GROUPS = {
    "root_translation": slice(0, 3),
    "other_joint_positions": slice(3, 84),
    "rotations": slice(84, 216),
    "object": slice(216, 228),
}


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


def _palm_role_metrics(values: torch.Tensor) -> Dict[str, object]:
    """Return float64 pooled metrics for one collection of palm-frames."""
    values = values.double()
    count = int(values.numel())
    if count == 0:
        return {
            "count": 0,
            "mean_distance_m": None,
            "mean_squared_m2": None,
            "rms_m": None,
        }
    mean_distance = float(values.sum().item() / count)
    mean_squared = float(values.square().sum().item() / count)
    return {
        "count": count,
        "mean_distance_m": mean_distance,
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
    frame_counts = {name: int(mask.sum().item()) for name, mask in masks.items()}
    frame_counts["engaged"] = int(engaged.sum().item())
    frame_counts["active"] = int(engaged.numel())
    return {
        "frame_counts": frame_counts,
        "role_values": role_values,
        "category_values": category_values,
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
    pooled = {role: [] for role in role_names}
    by_category_values = {
        category: {role: [] for role in role_names}
        for category in ("left_only", "right_only", "both")
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
            pooled[role].append(values)
            contribution_sums[role] += (
                float(values.double().square().sum().item()) / denominator * take
            )
        for category in ("left_only", "right_only", "both"):
            for role in role_names:
                values = decomposed["category_values"][category][role]
                by_category_values[category][role].append(values)
                category_contribution_sums[category][role] += (
                    float(values.double().square().sum().item()) / denominator * take
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
        if frame_counts["engaged"] != 3353 or frame_counts["active"] != 3584:
            raise AssertionError(
                "R0 palm-decomposition frame counts changed: expected 3353 engaged "
                f"and 3584 active, observed {frame_counts['engaged']} engaged and "
                f"{frame_counts['active']} active"
            )

    pooled_metrics = {
        role: _palm_role_metrics(torch.cat(pooled[role])) for role in role_names
    }
    by_category = {
        category: {
            role: _palm_role_metrics(torch.cat(by_category_values[category][role]))
            for role in role_names
        }
        for category in ("left_only", "right_only", "both")
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
    window_count: int = 64,
    timesteps: tuple[int, ...] = (250, 499),
    device: Optional[torch.device] = None,
) -> Dict[str, object]:
    """Measure raw component gradients for later geometry-weight reduction."""
    # This import direction is deliberate: the trainer never imports this
    # diagnostics module, so the probe cannot enter the training hot path.
    from train_hoi_prior import _forward_losses, _move_batch

    checkpoint_path = Path(checkpoint_path)
    output_path = Path(output_path)
    if window_count <= 0:
        raise ValueError("geometry weight-derivation window_count must be positive")
    if not timesteps:
        raise ValueError("geometry weight-derivation timesteps must not be empty")
    if not checkpoint_path.is_file():
        raise ValueError(f"probe checkpoint does not exist: {checkpoint_path}")
    _require_plain_forward_path(cfg)
    if (
        float(cfg.get("hand_object_contact_hinge", 0.0)) != 0.0
        or bool(cfg.get("hand_object_contact_detach_object", False))
        or bool(cfg.get("hand_object_contact_detach_root", False))
    ):
        raise ValueError(
            "geometry weight-derivation probe requires zero hinge and attached gradients"
        )

    configured_weight = float(cfg.get("hand_object_contact_weight", 0.0))
    geometry_from_separate_call = configured_weight == 0.0
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")
    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(42)
    component_names = _GEOMETRY_GRADIENT_COMPONENTS
    component_squared = {
        str(timestep): {name: 0.0 for name in component_names}
        for timestep in timesteps
    }
    target_squared = {
        str(timestep): {"geometry": 0.0, "nongeometry": 0.0, "dot": 0.0}
        for timestep in timesteps
    }
    root_squared = {str(timestep): 0.0 for timestep in timesteps}
    previous_training = model.training
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    cuda_devices = [] if device.type != "cuda" else [device.index or 0]
    batch_count: Optional[int] = None

    try:
        random.seed(42)
        np.random.seed(42)
        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(42)
            if device.type == "cuda":
                torch.cuda.manual_seed(42)
            model.train()
            for timestep in timesteps:
                consumed = 0
                batches = 0
                key = str(timestep)
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
                        with _pinned_timestep(timestep) as pin:
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
                    if pin.substitutions != 1:
                        raise AssertionError(
                            "geometry weight probe expected exactly one pinned "
                            "timestep substitution per forward, observed "
                            f"{pin.substitutions}"
                        )
                    if len(captured) != 1:
                        raise AssertionError(
                            "geometry weight probe expected one training forward, "
                            f"observed {len(captured)}"
                        )
                    prediction = captured[0]
                    if geometry_from_separate_call:
                        geometry_losses = _geometry_losses(
                            prediction,
                            batch,
                            parents,
                            position_minimum,
                            position_maximum,
                            object_minimum,
                            object_maximum,
                            cfg,
                            weight=1.0,
                            detach_object=bool(
                                cfg.get("hand_object_contact_detach_object", False)
                            ),
                            detach_root=bool(
                                cfg.get("hand_object_contact_detach_root", False)
                            ),
                        )
                        geometry_loss = geometry_losses[
                            "hand_object_contact_geometry"
                        ]
                    else:
                        geometry_loss = losses["hand_object_contact_geometry"]

                    component_losses = {
                        name: losses[name]
                        for name in component_names
                        if name != "hand_object_contact_geometry"
                    }
                    component_losses["hand_object_contact_geometry"] = geometry_loss
                    gradients = {}
                    for name in component_names:
                        gradient, = torch.autograd.grad(
                            component_losses[name], prediction, retain_graph=True
                        )
                        gradients[name] = gradient
                        component_squared[key][name] += float(
                            gradient.double().square().sum().item()
                        )

                    geometry_gradient = gradients["hand_object_contact_geometry"]
                    if int(torch.count_nonzero(geometry_gradient[..., 3:84])) != 0:
                        raise AssertionError(
                            "geometry gradient on joint-position channels 3:84 is not exactly zero"
                        )
                    nongeometry_gradient = (
                        gradients["total"] - configured_weight * geometry_gradient
                    )
                    geometry_target = geometry_gradient[..., 84:228].double()
                    nongeometry_target = nongeometry_gradient[..., 84:228].double()
                    target_squared[key]["geometry"] += float(
                        geometry_target.square().sum().item()
                    )
                    target_squared[key]["nongeometry"] += float(
                        nongeometry_target.square().sum().item()
                    )
                    target_squared[key]["dot"] += float(
                        (geometry_target * nongeometry_target).sum().item()
                    )
                    root_squared[key] += float(
                        geometry_gradient[..., 0:3].double().square().sum().item()
                    )
                    consumed += take
                    batches += 1
                if consumed != window_count:
                    raise ValueError(
                        f"training loader ended after {consumed} windows; "
                        f"{window_count} required"
                    )
                if batch_count is None:
                    batch_count = batches
                elif batch_count != batches:
                    raise AssertionError("training loader batch count changed across timesteps")
    finally:
        model.train(previous_training)
        random.setstate(python_state)
        np.random.set_state(numpy_state)

    gradient_l2 = {
        key: {name: value ** 0.5 for name, value in values.items()}
        for key, values in component_squared.items()
    }
    human_side_l2 = {
        key: sum(
            component_squared[key][name]
            for name in ("joint_position", "joint_rotation", "fk")
        ) ** 0.5
        for key in component_squared
    }
    object_side_l2 = {
        key: sum(
            component_squared[key][name]
            for name in ("object_translation", "object_rotation", "object_surface")
        ) ** 0.5
        for key in component_squared
    }
    target_channel = {}
    for key, values in target_squared.items():
        geometry_l2 = values["geometry"] ** 0.5
        nongeometry_l2 = values["nongeometry"] ** 0.5
        denominator = geometry_l2 * nongeometry_l2
        target_channel[key] = {
            "geometry_l2": geometry_l2,
            "nongeometry_l2": nongeometry_l2,
            "cosine_similarity": values["dot"] / denominator if denominator else 0.0,
        }
    root_channel_geometry_l2 = {
        key: value ** 0.5 for key, value in root_squared.items()
    }
    p0_calibration = {
        key: {
            "fk_over_object_surface": (
                gradient_l2[key]["fk"] / gradient_l2[key]["object_surface"]
                if gradient_l2[key]["object_surface"] else 0.0
            )
        }
        for key in gradient_l2
    }
    finite_values = []
    for values in gradient_l2.values():
        finite_values.extend(values.values())
    finite_values.extend(human_side_l2.values())
    finite_values.extend(object_side_l2.values())
    finite_values.extend(root_channel_geometry_l2.values())
    for values in target_channel.values():
        finite_values.extend(values.values())
    for values in p0_calibration.values():
        finite_values.extend(values.values())
    if not all(math.isfinite(value) for value in finite_values):
        raise ValueError("geometry weight-derivation probe produced a non-finite value")

    result: Dict[str, object] = {
        "probe": "geometry_weight_derivation_probe",
        "seed": 42,
        "window_count": window_count,
        "batch_count": int(batch_count or 0),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
        ).strip(),
        "configured_hand_object_contact_weight": configured_weight,
        "geometry_from_separate_call": geometry_from_separate_call,
        "timesteps": [int(timestep) for timestep in timesteps],
        "timestep_seam": "pinned_real_forward_losses",
        "gradient_l2": gradient_l2,
        "human_side_l2": human_side_l2,
        "object_side_l2": object_side_l2,
        "target_channel": target_channel,
        "root_channel_geometry_l2": root_channel_geometry_l2,
        "p0_calibration": p0_calibration,
        "self_check": {
            "plain_forward_path_verified": True,
            "geometry_gradient_on_joint_positions_exactly_zero": True,
            "all_values_finite": True,
        },
    }
    _write_probe_json(output_path, result)
    return result
