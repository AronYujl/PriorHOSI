"""Geometric and residual-learning contracts for body-aligned scene queries."""

import sys
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch.nn import functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.body_geometry import (
    BODY_GROUPS,
    BodyGeometryRefiner,
    body_geometry_relation_loss,
    query_body_geometry,
)
from priors.hsi.penetration import SceneSDFBank
from priors.hsi.scene_field import SceneGeometry


def _plane(name="plane", normal=(0.0, 1.0, 0.0), offset=0.0):
    shape, origin, pitch = (40, 40, 40), (-0.4, -0.4, -0.4), 0.02
    axes = [origin[i] + (torch.arange(n) + 0.5) * pitch
            for i, n in enumerate(shape)]
    points = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    field = (points * torch.tensor(normal)).sum(-1) + offset
    return SceneGeometry(
        name, field.numpy(), np.asarray(origin), pitch, is_watertight=True,
    )


def _bank(geometry=None):
    return SceneSDFBank.from_geometries(
        {0: geometry if geometry is not None else _plane()}, dtype=torch.float32,
    )


def test_plane_query_keeps_all_joint_and_frame_correspondence_in_one_call():
    bank = _bank()
    joints = torch.zeros(1, 3, 24, 3)
    joints[..., 1] = torch.linspace(-0.2, 0.2, 72).reshape(1, 3, 24)
    with mock.patch.object(bank, "signed_distance", wraps=bank.signed_distance) as lookup:
        result = query_body_geometry(joints, torch.tensor([0]), torch.eye(3)[None], bank)
    assert lookup.call_count == 1
    assert lookup.call_args.args[0].shape == (1, 3, 24, 7, 3)
    assert result["features"].shape == (1, 3, 24, 5)
    assert result["valid"].all()
    torch.testing.assert_close(result["signed_distance"], joints[..., 1], atol=1e-6, rtol=0)
    torch.testing.assert_close(result["features"][..., 0], joints[..., 1] / 0.25)
    expected_gradient = torch.zeros_like(joints)
    expected_gradient[..., 1] = 1.0
    torch.testing.assert_close(result["gradient_world"], expected_gradient, atol=2e-6, rtol=0)
    torch.testing.assert_close(result["features"][..., 1:4], expected_gradient, atol=2e-6, rtol=0)
    assert torch.equal(result["features"][..., 4], torch.ones(1, 3, 24))


def test_mixed_scenes_and_mirror_gradient_use_each_batch_flag():
    source = _plane("source", normal=(1.0, 0.0, 0.0))
    other = _plane("other", normal=(0.0, 0.0, 1.0), offset=0.1)
    bank = SceneSDFBank.from_geometries(
        {3: source, 7: source, 12: other},
        flag_to_name={3: "source", 7: "source_mirror", 12: "other"},
        dtype=torch.float32,
    )
    joints = torch.tensor([0.1, 0.05, -0.1]).expand(3, 2, 24, 3).clone()
    result = query_body_geometry(joints, torch.tensor([3, 7, 12]), torch.eye(3).repeat(3, 1, 1), bank)
    expected_sdf = torch.tensor([0.1, -0.1, 0.0])[:, None, None].expand(3, 2, 24)
    expected_gradient = torch.tensor([[1., 0., 0.], [-1., 0., 0.], [0., 0., 1.]])
    torch.testing.assert_close(result["signed_distance"], expected_sdf, atol=1e-6, rtol=0)
    torch.testing.assert_close(
        result["gradient_world"], expected_gradient[:, None, None].expand_as(joints),
        atol=2e-6, rtol=0,
    )


def test_world_gradient_is_rotated_once_into_the_window_frame():
    bank = _bank(_plane(normal=(1.0, 0.0, 0.0)))
    # A +90 degree rotation around world y maps local +z into world +x.
    rotation = torch.tensor([[[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]]])
    joints = torch.zeros(1, 2, 24, 3)
    result = query_body_geometry(joints, torch.tensor([0]), rotation, bank)
    expected = torch.tensor([0., 0., 1.]).expand_as(joints)
    torch.testing.assert_close(result["features"][..., 1:4], expected, atol=2e-6, rtol=0)


def test_query_preserves_position_gradients():
    bank = _bank()
    joints = torch.zeros(1, 2, 24, 3, requires_grad=True)
    result = query_body_geometry(joints, torch.tensor([0]), torch.eye(3)[None], bank)
    gradient = torch.autograd.grad(result["features"][..., 0].sum(), joints)[0]
    expected = torch.zeros_like(joints)
    expected[..., 1] = 4.0
    torch.testing.assert_close(gradient, expected, atol=5e-6, rtol=0)


def test_local_direction_features_retain_coordinate_gradients():
    pitch = 0.02
    axis = -0.4 + (torch.arange(40) + 0.5) * pitch
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    geometry = SceneGeometry(
        "bilinear", (x * y).numpy(), np.asarray([-0.4] * 3), pitch,
        is_watertight=True,
    )
    joints = torch.full((1, 2, 24, 3), 0.1, requires_grad=True)
    result = query_body_geometry(joints, torch.tensor([0]), torch.eye(3)[None], _bank(geometry))
    gradient = torch.autograd.grad(result["features"][..., 1:4].sum(), joints)[0]
    expected = torch.tensor([1., 1., 0.]).expand_as(joints)
    torch.testing.assert_close(gradient, expected, atol=1e-5, rtol=0)


def test_any_outside_stencil_is_invalid_while_raw_distance_remains_visible():
    bank = _bank()
    joints = torch.zeros(1, 1, 24, 3)
    joints[..., 1] = 0.1
    joints[0, 0, 0, 0] = 0.39  # Centre inside; +x stencil lies outside.
    joints[0, 0, 1, 0] = 0.5   # Centre outside the scanned field.
    result = query_body_geometry(joints, torch.tensor([0]), torch.eye(3)[None], bank)
    assert not result["valid"][0, 0, :2].any()
    assert result["valid"][0, 0, 2:].all()
    assert not result["center_out_of_bounds"][0, 0, 0]
    assert result["center_out_of_bounds"][0, 0, 1]
    assert torch.equal(result["features"][0, 0, :2], torch.zeros(2, 5))
    assert (result["signed_distance"][0, 0, :2] > 0).all()


def test_zero_initialized_refiner_preserves_backbone_and_learns_last_layer():
    torch.manual_seed(42)
    refiner = BodyGeometryRefiner(hidden_dim=12, hidden_width=16)
    hidden = torch.randn(2, 3, 12)
    features = torch.randn(2, 3, 24, 5)
    residual = refiner(hidden, features)
    assert residual.shape == (2, 3, 216)
    assert torch.equal(residual, torch.zeros_like(residual))
    target = torch.randn_like(residual)
    F.mse_loss(residual, target).backward()
    assert refiner.output.weight.grad.norm() > 0
    assert refiner.output.bias.grad.norm() > 0
    assert torch.equal(refiner.input.weight.grad, torch.zeros_like(refiner.input.weight))
    assert not any(isinstance(module, torch.nn.Dropout) for module in refiner.modules())


def test_refiner_exposes_fixed_body_groups_and_temporal_difference():
    torch.manual_seed(42)
    refiner = BodyGeometryRefiner(hidden_dim=12, hidden_width=16)
    hidden = torch.zeros(1, 3, 12)
    features = torch.zeros(1, 3, 24, 5)
    features[:, 0, BODY_GROUPS[1], 0] = 1.0
    features[:, 1, BODY_GROUPS[1], 0] = 0.5
    features[:, 2, BODY_GROUPS[4], 0] = -1.0
    encoded = refiner.joint_encoder(features) + refiner.joint_identity
    groups = refiner._group_tokens(encoded)
    assert groups.shape == (1, 3, len(BODY_GROUPS), refiner.geometry_dim)
    # A change in one leg is localized to that group's temporal token; a right
    # arm change at the next frame must not be mistaken for the leg change.
    delta = torch.cat([torch.zeros_like(groups[:, :1]), groups[:, 1:] - groups[:, :-1]], dim=1)
    assert delta[:, 1, 1].abs().sum() > 0
    assert torch.equal(delta[:, 1, 2], torch.zeros_like(delta[:, 1, 2]))
    assert delta[:, 2, 4].abs().sum() > 0


def test_relation_loss_masks_history_range_and_both_out_of_bounds():
    bank = _bank()
    target = torch.zeros(1, 4, 24, 3)
    predicted = target.clone()
    predicted[..., 1] = 0.1
    target[0, 2, 0, 1] = 0.3  # GT outside the near-surface relation band.
    predicted[0, 2, 1, 0] = 0.5  # Prediction outside the field.
    target[0, 2, 2, 0] = 0.5     # GT outside the field.
    predicted.requires_grad_()
    result = body_geometry_relation_loss(predicted, target, torch.tensor([0]), bank)
    expected_valid = torch.zeros(1, 4, 24, dtype=torch.bool)
    expected_valid[:, 2:] = True
    expected_valid[0, 2, :3] = False
    assert torch.equal(result["valid"], expected_valid)
    torch.testing.assert_close(result["loss"], torch.tensor(0.08), atol=1e-6, rtol=0)
    torch.testing.assert_close(result["invalid_fraction"], torch.tensor(2 / 48))
    torch.testing.assert_close(result["active_fraction"], torch.tensor(45 / 48))
    gradient = torch.autograd.grad(result["loss"], predicted)[0]
    assert torch.equal(gradient[:, :2], torch.zeros_like(gradient[:, :2]))
    assert torch.equal(gradient[0, 2, :3], torch.zeros_like(gradient[0, 2, :3]))
    assert (gradient[..., 1][expected_valid] > 0).all()
    assert result["prediction_out_of_bounds"][0, 2, 1]
    assert result["target_out_of_bounds"][0, 2, 2]


def test_relation_loss_has_differentiable_zero_with_no_observed_surface_targets():
    bank = _bank()
    target = torch.zeros(1, 4, 24, 3)
    target[..., 1] = 0.3
    predicted = target.clone().requires_grad_()
    result = body_geometry_relation_loss(predicted, target, torch.tensor([0]), bank)
    assert not result["valid"].any()
    assert result["loss"] == 0
    assert result["active_fraction"] == 0
    assert result["invalid_fraction"] == 0
    gradient = torch.autograd.grad(result["loss"], predicted)[0]
    assert torch.equal(gradient, torch.zeros_like(gradient))
