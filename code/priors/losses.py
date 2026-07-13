"""Preregistered HOIPrior training and validation objectives."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F
from pytorch3d import transforms

from .representation import REPRESENTATION


def _fk_positions(
    root: torch.Tensor,
    global_rotation: torch.Tensor,
    rest_offsets: torch.Tensor,
    parents: torch.Tensor,
) -> torch.Tensor:
    """Recover the 24 SMPL joint positions from global rotations and offsets."""
    positions = [root]
    for joint in range(1, 24):
        parent = int(parents[joint])
        if parent < 0 or parent >= global_rotation.shape[2]:
            raise ValueError(f"invalid FK parent {parent} for joint {joint}")
        offset = rest_offsets[:, None, joint, :, None]
        rotated = torch.matmul(global_rotation[:, :, parent], offset).squeeze(-1)
        positions.append(positions[parent] + rotated)
    return torch.stack(positions, dim=2)


def hoi_training_losses(
    prediction: torch.Tensor,
    target: torch.Tensor,
    goals: torch.Tensor,
    rest_human_offsets: torch.Tensor,
    parents_24: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    *,
    fk_weight: float = 50.0,
    velocity_weight: float = 0.1,
    goal_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    if prediction.shape != target.shape or prediction.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(f"expected matching [B,16,232], got {prediction.shape}/{target.shape}")
    predicted = prediction[:, REPRESENTATION.history_frames:]
    truth = target[:, REPRESENTATION.history_frames:]
    losses: Dict[str, torch.Tensor] = {
        "joint_position": F.mse_loss(predicted[..., :84], truth[..., :84]),
        "joint_rotation": F.smooth_l1_loss(predicted[..., 84:216], truth[..., 84:216]),
        "object_translation": F.mse_loss(predicted[..., 216:219], truth[..., 216:219]),
        "object_rotation": F.smooth_l1_loss(predicted[..., 219:228], truth[..., 219:228]),
        "contact": F.smooth_l1_loss(predicted[..., 228:232], truth[..., 228:232]),
    }

    scale = (position_maximum - position_minimum).reshape(1, 1, 1, 3)
    minimum = position_minimum.reshape(1, 1, 1, 3)
    predicted_positions = (
        (prediction[..., :84].reshape(*prediction.shape[:2], 28, 3) + 1.0) * scale / 2.0
        + minimum
    )
    target_positions = (
        (target[..., :84].reshape(*target.shape[:2], 28, 3) + 1.0) * scale / 2.0
        + minimum
    )
    predicted_rotation = transforms.rotation_6d_to_matrix(
        prediction[..., 84:216].reshape(*prediction.shape[:2], 22, 6)
    )
    fk = _fk_positions(
        predicted_positions[..., 0, :], predicted_rotation, rest_human_offsets, parents_24,
    )
    active = slice(REPRESENTATION.history_frames, None)
    hand_fk = F.mse_loss(fk[:, active, [20, 21, 22, 23]], target_positions[:, active, [20, 21, 25, 27]])
    foot_fk = F.mse_loss(fk[:, active, [7, 8, 10, 11]], target_positions[:, active, [7, 8, 10, 11]])
    losses["fk"] = hand_fk + foot_fk

    velocity_channels = torch.cat((prediction[..., :84], prediction[..., 216:219]), dim=-1)
    target_velocity_channels = torch.cat((target[..., :84], target[..., 216:219]), dim=-1)
    predicted_previous = torch.cat((
        target_velocity_channels[:, REPRESENTATION.history_frames - 1:REPRESENTATION.history_frames],
        velocity_channels[:, REPRESENTATION.history_frames:-1],
    ), dim=1)
    losses["velocity"] = F.mse_loss(
        velocity_channels[:, REPRESENTATION.history_frames:] - predicted_previous,
        target_velocity_channels[:, REPRESENTATION.history_frames:] - target_velocity_channels[:, REPRESENTATION.history_frames - 1:-1],
    )
    losses["object_goal"] = F.mse_loss(prediction[:, -1, 216:219], goals[:, 6:9])
    reconstruction = sum(losses[name] for name in (
        "joint_position", "joint_rotation", "object_translation", "object_rotation", "contact",
    ))
    losses["reconstruction"] = reconstruction
    losses["total"] = (
        reconstruction
        + float(fk_weight) * losses["fk"]
        + float(velocity_weight) * losses["velocity"]
        + float(goal_weight) * losses["object_goal"]
    )
    with torch.no_grad():
        predicted_contact = prediction[:, REPRESENTATION.history_frames:, 228:232] >= 0.5
        target_contact = target[:, REPRESENTATION.history_frames:, 228:232] >= 0.5
        losses["contact_accuracy"] = (predicted_contact == target_contact).float().mean()
    return losses
