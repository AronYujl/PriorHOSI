"""Isolated D2-Z dataset metadata and immutable-GT gated loss."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from pytorch3d import transforms

from .data import PriorWindowDataset
from .losses import (
    D2X_FOOT_XZ_VELOCITY_SLOTS,
    _fk_positions,
    _velocity_residuals,
    hoi_training_losses,
)
from .near_ground import (
    immutable_gt_near_ground_gate,
    load_gate_audit_floors,
    sha256_file,
)


class D2ZPriorWindowDataset(PriorWindowDataset):
    """Add only the preregistered loss metadata to the unchanged HOI dataset."""

    def __init__(
        self,
        repo_root: str,
        expert: str,
        partition: str = "train",
        limit: int = 0,
        split_manifest: Optional[str] = None,
        *,
        gate_audit_path: str,
        gate_audit_sha256: str,
    ) -> None:
        if expert != "hoi":
            raise ValueError("D2-Z dataset is restricted to HOI")
        if split_manifest in (None, "", False):
            raise ValueError("D2-Z dataset requires the locked HOI split manifest")
        repo = Path(repo_root).resolve()
        split_path = Path(str(split_manifest))
        if not split_path.is_absolute():
            split_path = repo / split_path
        split = json.loads(split_path.read_text(encoding="utf-8"))
        if split.get("algorithm") != "omomo-sequence-sha256-seed42-v1":
            raise ValueError("D2-Z dataset split algorithm mismatch")
        expected_sequences = sorted(
            int(value) for value in split[partition]["sequence_indices"]
        )
        super().__init__(
            repo_root,
            expert,
            partition=partition,
            limit=limit,
            split_manifest=str(split_path),
        )
        self.d2z_floor_by_sequence = load_gate_audit_floors(
            gate_audit_path,
            gate_audit_sha256,
            partition=partition,
            expected_sequence_indices=expected_sequences,
            expected_split_sha256=sha256_file(split_path),
        )

    def __getitem__(self, item: int) -> Dict[str, torch.Tensor]:
        result = super().__getitem__(item)
        index = int(self.indices[item])
        start, end = int(self.starts[index]), int(self.ends[index])
        frames = np.arange(start, end, 3)
        sequence = int(self.sequence_ids[index])
        joints = np.asarray(self.joints[frames], dtype=np.float32)
        result["d2z_near_ground_gate"] = torch.from_numpy(
            immutable_gt_near_ground_gate(
                joints,
                self.d2z_floor_by_sequence[sequence],
            )
        )
        return result


def d2z_velocity_loss(
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    near_ground_gate: torch.Tensor,
) -> torch.Tensor:
    """Apply 1024 only to active routed joint/component residual entries."""
    if (
        predicted_residual.shape != target_residual.shape
        or predicted_residual.shape[-1] != 87
    ):
        raise ValueError("D2-Z near-ground gating expects matching [...,87] residual tensors")
    expected_shape = predicted_residual.shape[:-1] + (4,)
    if near_ground_gate is None or near_ground_gate.shape != expected_shape:
        shape = None if near_ground_gate is None else tuple(near_ground_gate.shape)
        raise ValueError(f"D2-Z near-ground gate shape {shape} != {tuple(expected_shape)}")
    if near_ground_gate.dtype != torch.bool:
        raise ValueError("D2-Z near-ground gate must have bool dtype")
    if near_ground_gate.device != predicted_residual.device:
        raise ValueError("D2-Z near-ground gate must be on the residual device")
    if near_ground_gate.requires_grad:
        raise ValueError("D2-Z near-ground gate must be immutable")
    slot_weights = torch.ones_like(predicted_residual)
    routed_weights = torch.where(
        near_ground_gate,
        predicted_residual.new_tensor(1024.0),
        predicted_residual.new_tensor(1.0),
    ).repeat_interleave(2, dim=-1)
    slot_weights[..., list(D2X_FOOT_XZ_VELOCITY_SLOTS)] = routed_weights
    return ((predicted_residual - target_residual).square() * slot_weights).mean()


def d2z_hoi_training_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    goals: torch.Tensor,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    terminal_window: torch.Tensor,
    rest_object_points: torch.Tensor,
    world_to_local_rotation: torch.Tensor,
    object_rotation_reference: torch.Tensor,
    near_ground_gate: torch.Tensor,
    *,
    fk_weight: float,
    object_surface_weight: float,
    velocity_weight: float,
    goal_weight: float,
) -> Dict[str, torch.Tensor]:
    """Replace only D2-X velocity reduction with the preregistered D2-Z gate."""
    losses = hoi_training_losses(
        prediction,
        target,
        goals,
        rest_human_offsets,
        parents_24,
        position_minimum,
        position_maximum,
        object_minimum,
        object_maximum,
        terminal_window,
        rest_object_points,
        world_to_local_rotation,
        object_rotation_reference,
        fk_weight=fk_weight,
        object_surface_weight=object_surface_weight,
        velocity_weight=velocity_weight,
        goal_weight=goal_weight,
        fk_foot_temporal_routing=True,
        routed_foot_residual_multiplier=1.0,
    )
    scale = (position_maximum - position_minimum).reshape(1, 1, 1, 3)
    minimum = position_minimum.reshape(1, 1, 1, 3)
    predicted_positions = (
        (prediction[..., :84].reshape(*prediction.shape[:2], 28, 3) + 1.0)
        * scale / 2.0
        + minimum
    )
    predicted_rotation = transforms.rotation_6d_to_matrix(
        prediction[..., 84:216].reshape(*prediction.shape[:2], 22, 6)
    )
    predicted_fk = _fk_positions(
        predicted_positions[..., 0, :],
        predicted_rotation,
        rest_human_offsets,
        parents_24,
    )
    predicted_velocity, target_velocity = _velocity_residuals(
        prediction,
        target,
        predicted_fk,
        position_minimum,
        position_maximum,
        fk_foot_temporal_routing=True,
    )
    losses["velocity"] = d2z_velocity_loss(
        predicted_velocity,
        target_velocity,
        near_ground_gate,
    )
    losses["total"] = (
        losses["reconstruction"]
        + float(fk_weight) * losses["fk"]
        + float(object_surface_weight) * losses["object_surface"]
        + float(velocity_weight) * losses["velocity"]
        + float(goal_weight) * losses["object_goal"]
    )
    return losses
