"""Registered causal probes for the HOI expert."""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional

import numpy as np
import torch

from .losses import hoi_training_losses


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


def _detached_geometry_losses(
    prediction: torch.Tensor,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    cfg,
) -> Dict[str, torch.Tensor]:
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
        hand_object_contact_weight=float(cfg.hand_object_contact_weight),
        hand_object_contact_hinge=float(cfg.get("hand_object_contact_hinge", 0.0)),
        hand_object_contact_detach_object=bool(
            cfg.get("hand_object_contact_detach_object", False)
        ),
        hand_object_contact_detach_root=True,
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

                detached_losses = _detached_geometry_losses(
                    prediction,
                    batch,
                    parents,
                    position_minimum,
                    position_maximum,
                    object_minimum,
                    object_maximum,
                    cfg,
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
