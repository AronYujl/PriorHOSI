"""Body-aligned mesh geometry for HSI clean-state motion refinement."""

import torch
from torch import nn
from torch.nn import functional as F


BODY_GEOMETRY_RANGE_M = 0.25

# The FK ordering is the 24-joint ordering used by the HSI denoiser.  Keeping
# this partition explicit makes a left/right permutation a meaningful causal
# diagnostic: the same geometry is then presented to the wrong body group.
BODY_GROUPS = (
    (0, 3, 6, 9, 12, 15),       # pelvis, spine, neck, head
    (1, 4, 7, 10),               # left leg
    (2, 5, 8, 11),               # right leg
    (16, 18, 20, 22),             # left arm
    (17, 19, 21, 23),             # right arm
)


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
    """Predict a human-state residual from body-group and temporal geometry.

    A flattened ``24x5`` vector lets the correction behave like an unstructured
    scene residual.  We first encode each FK joint, add a joint identity, pool
    into five fixed kinematic groups, and append the one-step temporal change
    of those group tokens.  The output remains per-frame, so the surrounding
    diffusion contract and zero-initialized warm start are unchanged.
    """

    def __init__(self, hidden_dim=512, hidden_width=256, geometry_dim=48,
                 temporal_dim=16):
        super().__init__()
        self.geometry_dim = int(geometry_dim)
        self.temporal_dim = int(temporal_dim)
        self.body_groups = BODY_GROUPS
        self.joint_encoder = nn.Linear(5, self.geometry_dim)
        self.joint_identity = nn.Parameter(torch.zeros(24, self.geometry_dim))
        self.group_encoder = nn.Sequential(
            nn.LayerNorm(self.geometry_dim),
            nn.Linear(self.geometry_dim, self.geometry_dim),
            nn.SiLU(),
        )
        nn.init.zeros_(self.group_encoder[1].bias)
        group_count = len(self.body_groups)
        # Keep the first geometry coordinate equal to the mean signed distance
        # so the existing branch diagnostic remains an interpretable probe.
        geometry_width = 1 + group_count * self.geometry_dim * 2 + self.temporal_dim
        self.input = nn.Linear(hidden_dim + geometry_width, hidden_width)
        self.activation = nn.SiLU()
        self.output = nn.Linear(hidden_width, 216)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def _group_tokens(self, encoded):
        groups = []
        for indices in self.body_groups:
            group = encoded[..., list(indices), :].mean(dim=-2)
            groups.append(self.group_encoder(group))
        return torch.stack(groups, dim=-2)

    def _temporal_encoding(self, batch, frames, device, dtype):
        position = torch.linspace(0.0, 1.0, frames, device=device, dtype=dtype)
        half = self.temporal_dim // 2
        frequencies = torch.arange(half, device=device, dtype=dtype)
        frequencies = torch.pow(10000.0, -frequencies / max(half - 1, 1))
        angles = position[:, None] * frequencies[None, :]
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if encoding.shape[-1] < self.temporal_dim:
            encoding = F.pad(encoding, (0, self.temporal_dim - encoding.shape[-1]))
        return encoding[:, :self.temporal_dim].unsqueeze(0).expand(batch, -1, -1)

    def forward(self, motion_hidden, body_features):
        if body_features.ndim != 4 or body_features.shape[-2:] != (24, 5):
            raise ValueError(
                f"body geometry must have shape [B,T,24,5], got {tuple(body_features.shape)}"
            )
        batch, frames = body_features.shape[:2]
        encoded = self.joint_encoder(body_features) + self.joint_identity
        groups = self._group_tokens(encoded)
        delta = torch.cat([torch.zeros_like(groups[:, :1]), groups[:, 1:] - groups[:, :-1]], dim=1)
        temporal = self._temporal_encoding(batch, frames, groups.device, groups.dtype)
        distance_anchor = body_features[..., 0, 0:1]
        joined = torch.cat([
            motion_hidden, distance_anchor, groups.flatten(-2),
            delta.flatten(-2), temporal,
        ], dim=-1)
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
