"""Body-aligned mesh geometry for HSI clean-state motion refinement."""

import torch
from torch import nn
from torch.nn import functional as F


BODY_GEOMETRY_RANGE_M = 0.25


def query_body_geometry(joints_world, scene_flag, window_rotation, sdf_bank):
    """Query ordered body joints, retaining frame and joint correspondence.

    ``window_rotation`` maps window coordinates to world coordinates. The six
    finite-difference neighbours lie on world axes; their SDF gradient is then
    expressed in the window frame. All seven positions use the existing batched,
    mirror-aware mesh field lookup and retain their coordinate gradients.

    An out-of-bounds stencil carries no usable local geometry. Its feature is
    zero with validity zero, while raw distances and bounds flags remain visible
    in the returned diagnostics. Positive exterior distances retain the bank's
    meaning of distance beyond the scanned field.
    """
    stencil = joints_world.new_tensor([
        [0, 0, 0], [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
    ]) * sdf_bank.voxel_size
    points = joints_world.unsqueeze(-2) + stencil
    distances, out_of_bounds = sdf_bank.signed_distance(points, scene_flag)
    signed_distance = distances[..., 0]
    gradient_world = torch.stack([
        distances[..., 1] - distances[..., 2],
        distances[..., 3] - distances[..., 4],
        distances[..., 5] - distances[..., 6],
    ], dim=-1) / (2.0 * sdf_bank.voxel_size)
    gradient_local = torch.einsum(
        "btjc,bcd->btjd", gradient_world, window_rotation.float()
    )
    valid = ~out_of_bounds.any(dim=-1)
    geometry = torch.cat([
        signed_distance.unsqueeze(-1) / BODY_GEOMETRY_RANGE_M, gradient_local,
    ], dim=-1)
    features = torch.cat([
        torch.where(valid.unsqueeze(-1), geometry, torch.zeros_like(geometry)),
        valid.to(geometry.dtype).unsqueeze(-1),
    ], dim=-1)
    return {
        "features": features,
        "signed_distance": signed_distance,
        "gradient_world": gradient_world,
        "valid": valid,
        "center_out_of_bounds": out_of_bounds[..., 0],
        "stencil_out_of_bounds": out_of_bounds,
    }


class BodyGeometryRefiner(nn.Module):
    """Predict a human-state residual from motion and ordered local geometry."""

    def __init__(self, hidden_dim=512, hidden_width=256):
        super().__init__()
        self.input = nn.Linear(hidden_dim + 24 * 5, hidden_width)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_width, 216)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, motion_hidden, body_features):
        joined = torch.cat([motion_hidden, body_features.flatten(-2)], dim=-1)
        return self.output(self.activation(self.input(joined)))


def body_geometry_relation_loss(
    pred_joints, target_joints, scene_flag, bank, history_frames=2, range_m=0.25,
):
    """Match near-surface GT relations on jointly observed future body points.

    The field's centre validity defines supervision, independently of the
    seven-point stencil used for conditioning. Empty supervision contributes a
    differentiable zero. Bounds masks and future-point exclusion rates expose
    any motion outside the scanned geometry.
    """
    prediction_sdf, prediction_oob = bank.signed_distance(pred_joints, scene_flag)
    target_sdf, target_oob = bank.signed_distance(target_joints, scene_flag)
    future = torch.ones_like(prediction_oob)
    future[:, :history_frames] = False
    invalid = prediction_oob | target_oob
    valid = future & ~invalid & (target_sdf.abs() <= range_m)
    errors = F.smooth_l1_loss(
        prediction_sdf / range_m, target_sdf / range_m, reduction="none"
    )
    loss = torch.where(valid, errors, torch.zeros_like(errors)).sum()
    loss = loss / valid.sum().clamp_min(1)
    future_count = future.sum().clamp_min(1)
    return {
        "loss": loss,
        "valid": valid,
        "prediction_signed_distance": prediction_sdf,
        "target_signed_distance": target_sdf,
        "prediction_out_of_bounds": prediction_oob,
        "target_out_of_bounds": target_oob,
        "invalid_fraction": (future & invalid).sum() / future_count,
        "active_fraction": valid.sum() / future_count,
    }
