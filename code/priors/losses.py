"""Preregistered HOIPrior training and validation objectives."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F
from pytorch3d import transforms

from .representation import REPRESENTATION
from .window_codec import project_to_so3


D2X_FOOT_JOINTS = (7, 8, 10, 11)
D2X_FOOT_XZ_VELOCITY_SLOTS = tuple(
    joint * 3 + component for joint in D2X_FOOT_JOINTS for component in (0, 2)
)

# Preregistered P8 hand-object relative geometry term.
#
# The palms in the 24-joint SMPL FK tensor, matching the author's inference
# guidance (``code/guidance_loss.py:15-16`` uses ``l_palm_idx = 22`` and
# ``r_palm_idx = 23``).  The existing ``hand_fk`` term reads FK joints
# ``[20, 21, 22, 23]`` against target slots ``[20, 21, 25, 27]``, where 20/21 are
# the wrists; this term deliberately uses the palms only.
P8_PALM_JOINTS = (22, 23)
# The ground-truth contact annotation carries the two hand-semantic channels
# FIRST.  Measured over all 482 test annotation files: channel means are
# 0.5810 / 0.5895 for channels 0-1 and 0.0083 / 0.0118 for channels 2-3, so
# masking on the LAST two would engage only ~1.9% of frames and silently reduce
# the whole term to a no-op.  The channels are strictly binary (each has exactly
# two unique values, 0.0 and 1.0), so the threshold is not a free parameter:
# ``> 0.5`` and ``> 0.95`` select identical frames.
P8_CONTACT_HAND_CHANNELS = slice(0, 2)
P8_CONTACT_THRESHOLD = 0.5


def masked_hand_object_distance_loss(
    fk: torch.Tensor,
    object_surface: torch.Tensor,
    contact_ground_truth: torch.Tensor,
    *,
    hinge: float = 0.0,
    detach_object: bool = False,
) -> torch.Tensor:
    """Palm-to-object-surface squared distance, on ground-truth contact frames.

    The existing ``hand_fk`` term is minimised when the palm reaches the GT palm
    position, *regardless of where the model put the object*.  If the object
    prediction is off, satisfying ``hand_fk`` therefore places the palm at a
    point that is NOT on the predicted object surface -- which is exactly the
    "committed to contact but the hand is far away" failure.  This term is
    minimised only when the palm and the predicted object are mutually
    consistent, so the two are complementary rather than redundant.

    Restricted to GT contact frames: on non-contact frames there is no correct
    palm-to-object distance to pull toward, and applying one would distort
    locomotion and non-HOI arm poses.

    Returns a mean over engaged frames, so the value does not scale with how
    much of a batch happens to be in contact.

    ``hinge`` (metres, preregistered P10) replaces the per-palm distance with the
    hinged residual ``(nearest - hinge).clamp_min(0.0)`` *before* squaring, so the
    term stops pulling once a palm is already within ``hinge`` of the predicted
    surface.  The author's inference guidance uses exactly this shape with a 2 cm
    threshold (``code/guidance_loss.py:36-39``: ``contact_threshold = 0.02`` and
    ``torch.maximum(dists - contact_threshold, 0)``).  Without it the squared
    term keeps rewarding ever-smaller distances and therefore pays for
    interpenetration.  ``hinge == 0.0`` executes the literal pre-P10 expression
    ``nearest.square()``, bit for bit, which is what keeps the sealed weight-3
    run a valid cell of the P10 comparison; a negative or non-finite hinge is
    rejected rather than silently ignored by the branch.

    ``detach_object`` (preregistered P10) detaches the *local* view of the object
    surface, so this term's gradient reaches the hand only.  With it false, the
    cheapest way to reduce the term is to drag the predicted object toward the
    hand, which corrupts object placement.  The caller hands the same
    ``predicted_surface`` tensor to the ``object_surface`` term; ``Tensor.detach``
    returns a new view and never mutates the caller's tensor or its graph, so
    object supervision through that term is unaffected.  Never use the in-place
    ``detach_()`` here.
    """
    hinge = float(hinge)
    if not math.isfinite(hinge) or hinge < 0.0:
        raise ValueError(
            f"P8 geometry hinge must be a finite non-negative distance in metres, got {hinge}"
        )
    if fk.ndim != 4 or fk.shape[2] < max(P8_PALM_JOINTS) + 1:
        raise ValueError(
            f"P8 geometry expects [B,T,>=24,3] FK joints, got {tuple(fk.shape)}"
        )
    if object_surface.ndim != 4 or object_surface.shape[-1] != 3:
        raise ValueError(
            f"P8 geometry expects [B,T,V,3] object surface, got {tuple(object_surface.shape)}"
        )
    if contact_ground_truth.shape[:2] != fk.shape[:2]:
        raise ValueError("P8 geometry contact labels differ from FK joints in [B,T]")
    active = slice(REPRESENTATION.history_frames, None)
    batch = fk.shape[0]
    palms = fk[:, active][:, :, P8_PALM_JOINTS, :]
    surface = object_surface[:, active]
    if detach_object:
        # Local view only: the caller's ``predicted_surface`` keeps its graph, so
        # the separate ``object_surface`` reconstruction term is untouched.
        surface = surface.detach()
    frames = palms.shape[1]
    if surface.shape[1] != frames:
        raise ValueError("P8 geometry object surface and FK differ in frame count")
    engaged = (
        contact_ground_truth[:, active, P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD
    ).any(dim=-1)
    distances = torch.cdist(
        palms.reshape(batch * frames, len(P8_PALM_JOINTS), 3),
        surface.reshape(batch * frames, -1, 3),
    )
    nearest = distances.min(dim=-1)[0].reshape(batch, frames, len(P8_PALM_JOINTS))
    if hinge > 0.0:
        per_frame = (nearest - hinge).clamp_min(0.0).square().mean(dim=-1)
    else:
        # Literal sealed pre-P10 expression.  Do not fold this into the hinged
        # branch with hinge=0: the weight-3 cell of the P10 comparison is reused
        # from a sealed run and only stays valid while this path is bit-identical.
        per_frame = nearest.square().mean(dim=-1)
    weight = engaged.to(per_frame)
    # clamp_min keeps the gradient defined when a batch holds no contact frames.
    return (per_frame * weight).sum() / weight.sum().clamp_min(1.0)



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


def _velocity_residuals(
    prediction: torch.Tensor,
    target: torch.Tensor,
    predicted_fk_positions: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    *,
    fk_foot_temporal_routing: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed-size velocity residual pair, optionally routing foot x/z through FK."""
    velocity_channels = torch.cat((prediction[..., :84], prediction[..., 216:219]), dim=-1)
    if fk_foot_temporal_routing:
        scale = (position_maximum - position_minimum).reshape(1, 1, 1, 3)
        minimum = position_minimum.reshape(1, 1, 1, 3)
        normalized_fk = 2.0 * (predicted_fk_positions - minimum) / scale - 1.0
        direct_positions = prediction[..., :84].reshape(*prediction.shape[:2], 28, 3)
        replacement_positions = torch.cat(
            (normalized_fk, direct_positions[..., 24:, :]), dim=2,
        )
        route_mask = torch.zeros_like(direct_positions, dtype=torch.bool)
        route_mask[..., D2X_FOOT_JOINTS, 0] = True
        route_mask[..., D2X_FOOT_JOINTS, 2] = True
        routed_positions = torch.where(route_mask, replacement_positions, direct_positions)
        velocity_channels = torch.cat(
            (routed_positions.flatten(start_dim=-2), prediction[..., 216:219]), dim=-1,
        )
    target_velocity_channels = torch.cat((target[..., :84], target[..., 216:219]), dim=-1)
    predicted_previous = torch.cat((
        target_velocity_channels[:, REPRESENTATION.history_frames - 1:REPRESENTATION.history_frames],
        velocity_channels[:, REPRESENTATION.history_frames:-1],
    ), dim=1)
    predicted_residual = (
        velocity_channels[:, REPRESENTATION.history_frames:] - predicted_previous
    )
    target_residual = (
        target_velocity_channels[:, REPRESENTATION.history_frames:]
        - target_velocity_channels[:, REPRESENTATION.history_frames - 1:-1]
    )
    return predicted_residual, target_residual


def _velocity_loss(
    predicted_residual: torch.Tensor,
    target_residual: torch.Tensor,
    *,
    fk_foot_temporal_routing: bool,
    routed_foot_residual_multiplier: float,
) -> torch.Tensor:
    """Apply the preregistered D2-Y per-slot weighting without changing D2-X."""
    multiplier = float(routed_foot_residual_multiplier)
    if not math.isfinite(multiplier) or multiplier < 1.0:
        raise ValueError("routed-foot residual multiplier must be finite and at least one")
    if multiplier == 1.0:
        return F.mse_loss(predicted_residual, target_residual)
    if not fk_foot_temporal_routing:
        raise ValueError("routed-foot residual amplification requires FK-foot temporal routing")
    if predicted_residual.shape != target_residual.shape or predicted_residual.shape[-1] != 87:
        raise ValueError(
            "routed-foot residual amplification expects matching [...,87] residual tensors"
        )
    slot_weights = predicted_residual.new_ones(87)
    slot_weights[list(D2X_FOOT_XZ_VELOCITY_SLOTS)] = multiplier
    return ((predicted_residual - target_residual).square() * slot_weights).mean()


def hoi_training_losses(
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
    *,
    fk_weight: float = 50.0,
    object_surface_weight: float = 50.0,
    velocity_weight: float = 0.1,
    goal_weight: float = 1.0,
    hand_object_contact_weight: float = 0.0,
    hand_object_contact_hinge: float = 0.0,
    hand_object_contact_detach_object: bool = False,
    fk_foot_temporal_routing: bool = False,
    routed_foot_residual_multiplier: float = 1.0,
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

    predicted_velocity, target_velocity = _velocity_residuals(
        prediction,
        target,
        fk,
        position_minimum,
        position_maximum,
        fk_foot_temporal_routing=bool(fk_foot_temporal_routing),
    )
    losses["velocity"] = _velocity_loss(
        predicted_velocity,
        target_velocity,
        fk_foot_temporal_routing=bool(fk_foot_temporal_routing),
        routed_foot_residual_multiplier=float(routed_foot_residual_multiplier),
    )
    terminal = terminal_window.reshape(-1).to(device=prediction.device, dtype=prediction.dtype)
    goal_per_window = (prediction[:, -1, 216:219] - goals[:, 6:9]).square().mean(dim=-1)
    # Keep a differentiable zero when a batch contains no terminal windows.
    losses["object_goal"] = (goal_per_window * terminal).sum() / terminal.sum().clamp_min(1.0)

    object_scale = (object_maximum - object_minimum).reshape(1, 1, 3)
    object_base = object_minimum.reshape(1, 1, 3)
    predicted_object_translation = (prediction[..., 216:219] + 1.0) * object_scale / 2.0 + object_base
    target_object_translation = (target[..., 216:219] + 1.0) * object_scale / 2.0 + object_base
    predicted_relative = project_to_so3(prediction[..., 219:228].reshape(*prediction.shape[:2], 3, 3))
    target_relative = project_to_so3(target[..., 219:228].reshape(*target.shape[:2], 3, 3))
    reference = object_rotation_reference[:, None]
    predicted_global_rotation = project_to_so3(predicted_relative @ reference)
    target_global_rotation = project_to_so3(target_relative @ reference)
    world_to_local = world_to_local_rotation[:, None]
    predicted_local_rotation = world_to_local @ predicted_global_rotation
    target_local_rotation = world_to_local @ target_global_rotation
    points = rest_object_points.to(prediction)
    predicted_surface = torch.einsum("bpc,btdc->btpd", points, predicted_local_rotation)
    target_surface = torch.einsum("bpc,btdc->btpd", points, target_local_rotation)
    predicted_surface = predicted_surface + predicted_object_translation[:, :, None]
    target_surface = target_surface + target_object_translation[:, :, None]
    losses["object_surface"] = F.mse_loss(
        predicted_surface[:, REPRESENTATION.history_frames:],
        target_surface[:, REPRESENTATION.history_frames:],
    )
    if hand_object_contact_weight != 0.0:
        losses["hand_object_contact_geometry"] = masked_hand_object_distance_loss(
            fk, predicted_surface, target[:, :, 228:232],
            # ``predicted_surface`` is the same tensor the ``object_surface`` term
            # above consumes; the detach below is applied to a local view inside
            # the callee and therefore cannot weaken that supervision.
            hinge=float(hand_object_contact_hinge),
            detach_object=bool(hand_object_contact_detach_object),
        )
    reconstruction = sum(losses[name] for name in (
        "joint_position", "joint_rotation", "object_translation", "object_rotation", "contact",
    ))
    losses["reconstruction"] = reconstruction
    weighted_geometry = (
        float(fk_weight) * losses["fk"]
        + float(object_surface_weight) * losses["object_surface"]
    )
    if hand_object_contact_weight != 0.0:
        weighted_geometry = (
            weighted_geometry
            + float(hand_object_contact_weight) * losses["hand_object_contact_geometry"]
        )
    losses["total"] = (
        reconstruction
        + weighted_geometry
        + float(velocity_weight) * losses["velocity"]
        + float(goal_weight) * losses["object_goal"]
    )
    with torch.no_grad():
        predicted_contact = prediction[:, REPRESENTATION.history_frames:, 228:232] >= 0.5
        target_contact = target[:, REPRESENTATION.history_frames:, 228:232] >= 0.5
        losses["contact_accuracy"] = (predicted_contact == target_contact).float().mean()
    return losses
