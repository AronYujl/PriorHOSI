"""Relation invariants, frame semantics and useful gradients for residual mixing."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import numpy as np
import pytest
from pytorch3d import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))

from mixer.input_views import KnownEmptyObjectView, empty_forward_trajectory
from mixer.relational import CELL_KEYS, RelationalCorrector, RelationalGeometry, RelationalObjective, optimize_relational_cells, source_floor_height
from mixer.relational_diagnostics import (
    RelationalArmijoTrace, RelationalOptimizerTrace, RelationalPrototypeDiagnostic,
    diagnose_relational_optimizer,
)
from tests.phase2.test_kinematic_composition import _fixture


def _geometry():
    torch.manual_seed(42)
    fixture = _fixture(batch=1)
    # Existing fixture returns a coherent skeleton under both expert rotations.
    dataset, offsets, hoi, hsi, fixed, _, _ = fixture
    def rotation(x, y, z):
        return transforms.euler_angles_to_matrix(torch.tensor([x, y, z]), 'XYZ')[None]
    mat = torch.eye(4)[None]
    mat[:, :3, :3] = rotation(0.1, 0.7, -0.2)
    mat[:, :3, 3] = torch.tensor([0.3, 0.1, -0.4])
    context = {'mat': mat, 'obj_rot_mat_prefix': rotation(0.3, -0.2, 0.4),
               'obj_rot_mat_ref': rotation(-0.1, 0.4, 0.2)}
    points = torch.tensor([[[-0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0]]])
    return RelationalGeometry(hoi, dataset, offsets, context, points), fixed


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_source_floor_matches_native_rule_on_nonpadded_source(device):
    from eval_metrics import determine_floor_height_and_contacts
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    generator = torch.Generator().manual_seed(42)
    human = torch.randn(24, 16, 24, 3, generator=generator).cumsum(1) * .004
    human[..., 1] += .07
    # All static heights, and no real low-speed samples despite a final endpoint.
    human[0, :, 10, :] = torch.tensor([0., .04, 0.])
    human[0, :, 11, :] = torch.tensor([0., .08, 0.])
    human[1, :, (10, 11), 0] = torch.arange(16)[:, None] * .06
    actual, counts = source_floor_height(human.to(device))
    expected = []
    for source in human.numpy():
        time = np.arange(46) / 3
        dense = np.stack([np.interp(time, np.arange(16), source[:, j, axis])
                          for j in range(24) for axis in range(3)], axis=-1)
        dense = dense.reshape(46, 24, 3).astype(np.float32)
        expected.append(float(determine_floor_height_and_contacts(dense)))
    torch.testing.assert_close(actual.cpu(), torch.tensor(expected), atol=2e-7, rtol=1e-6)
    assert counts[1].item() == 0 and actual[1].item() == 0
    assert actual[0].item() == pytest.approx(.04)


def test_source_floor_mask_is_relative_and_frozen_during_optimization():
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    target = geometry.decode(torch.zeros(1, 16, geometry.dimension))['human'][:, 2:]
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target,
                                    source_floor=True, source_stance_velocity=True)
    original_floor, original_mask = objective.floor_height.clone(), objective.stance.clone()
    original_feet = objective.source_feet.clone()
    height = objective.anchor['human'][..., (7, 8, 10, 11), 1] - original_floor[:, None, None]
    expected = height < torch.tensor([.08, .08, .04, .04])
    assert torch.equal(original_mask, expected[:, 2:] & expected[:, 1:-1])
    optimize_relational_cells(geometry, objective, steps=3, cells=('a00',), include_floor=False)
    assert torch.equal(objective.floor_height, original_floor)
    assert torch.equal(objective.stance, original_mask)
    assert torch.equal(objective.source_feet, original_feet)
    assert not objective.source_feet.requires_grad
    default = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target)
    explicit = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target,
                                   source_floor=False, source_stance_velocity=False)
    a, _, _ = optimize_relational_cells(geometry, default, steps=3, cells=('a00',), include_floor=False)
    b, _, _ = optimize_relational_cells(geometry, explicit, steps=3, cells=('a00',), include_floor=False)
    assert torch.equal(a, b)


def test_source_stance_velocity_has_zero_energy_and_gradient_at_moving_source():
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    parameters = torch.zeros(1, 16, geometry.dimension, requires_grad=True)
    state = geometry.decode(parameters)
    target = state['human'].detach()[:, 2:]
    source = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target,
                                 source_stance_velocity=True)
    stationary = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target)
    # Isolate the target semantics with known support; height selection is tested above.
    source.stance.fill_(True)
    stationary.stance.fill_(True)
    terms, metrics = source.evaluate(state)
    old_terms, _ = stationary.evaluate(state)
    assert old_terms['stance'].item() > 0
    assert metrics['stance_displacement_cm'].item() > 0
    assert terms['stance'].item() == metrics['stance_increment_cm'].item() == 0
    gradient, = torch.autograd.grad(terms['stance'].sum(), parameters)
    assert torch.count_nonzero(gradient) == 0


def test_source_stance_velocity_penalizes_added_horizontal_motion_on_fixed_support():
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    state = geometry.decode(torch.zeros(1, 16, geometry.dimension))
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long),
                                    state['human'][:, 2:], source_stance_velocity=True)
    objective.stance.fill_(True)
    baseline = state['human'].detach()
    # A shared spatial shift preserves displacement; vertical motion also leaves
    # this horizontal-only constraint unchanged even when crossing height thresholds.
    shift = torch.zeros_like(baseline)
    shift[..., 0] = .125
    shift[..., 1] = torch.arange(16)[None, :, None] * .1
    unchanged, _ = objective.evaluate(dict(state, human=baseline + shift))
    torch.testing.assert_close(unchanged['stance'], torch.zeros(1), atol=1e-10, rtol=0)
    # One added horizontal step on a supported toe is penalized and has a gradient.
    movement = torch.zeros_like(baseline)
    movement[:, 8:, 10, 0] = .02
    moved_human = (baseline + movement).requires_grad_()
    changed, metrics = objective.evaluate(dict(state, human=moved_human))
    assert changed['stance'].item() > 0 and metrics['stance_increment_cm'].item() > 0
    changed['stance'].sum().backward()
    assert moved_human.grad[:, 7:9, 10, 0].abs().sum().item() > 0
    assert torch.count_nonzero(moved_human.grad[..., 1]) == 0
    objective.stance[:, :, 2] = False
    excluded, _ = objective.evaluate(dict(state, human=moved_human.detach()))
    assert excluded['stance'].item() == 0


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_optimizer_trace_preserves_output_rng_and_attributes_source_departure(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    geometry, _ = _geometry()
    for name, value in vars(geometry).copy().items():
        if torch.is_tensor(value):
            setattr(geometry, name, value.to(device))
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool, device=points.device), points,
    )
    zero = geometry.base.new_zeros(1, 16, geometry.dimension)
    objective = RelationalObjective(
        geometry, torch.zeros(1, dtype=torch.long, device=device),
        geometry.decode(zero)['human'][:, 2:], source_stance_velocity=True,
    )
    objective.contact.fill_(True)
    objective.stance.fill_(True)
    expected, expected_parameters, expected_metrics = optimize_relational_cells(
        geometry, objective, steps=20, cells=('a00',), include_floor=False,
    )
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    trace = RelationalOptimizerTrace()
    actual, parameters, metrics = optimize_relational_cells(
        geometry, objective, steps=20, cells=('a00',), include_floor=False, trace=trace,
    )
    assert torch.equal(actual, expected)
    assert torch.equal(parameters, expected_parameters)
    assert all(torch.equal(metrics[name], expected_metrics[name]) for name in metrics)
    snapshot = diagnose_relational_optimizer(geometry, objective, trace, 20, .05, 'a00', False)
    assert torch.equal(torch.get_rng_state(), cpu_rng)
    if device == 'cuda':
        assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    reference, shadow = snapshot['reference'], snapshot['contact_off']
    assert reference['parameters'].shape == (21, 1, 16, geometry.dimension)
    assert reference['gradients'].shape == (20, 1, 16, geometry.dimension)
    assert reference['loss'][0].item() > 0
    assert reference['loss'][-1].item() > reference['loss'][0].item()
    for name in ('residual', 'stance', 'endpoint'):
        assert torch.count_nonzero(reference['initial_term_gradients'][name]) == 0
    assert torch.equal(reference['initial_term_gradients']['contact'], reference['gradients'][0])
    first_gradient = reference['gradients'][0]
    expected_update = -.05 * first_gradient / (first_gradient.abs() + 1e-8)
    torch.testing.assert_close(reference['parameters'][1], expected_update, atol=1e-8, rtol=1e-6)
    assert torch.count_nonzero(shadow['parameters']) == 0
    assert torch.count_nonzero(shadow['gradients']) == 0
    assert torch.count_nonzero(shadow['loss']) == 0
    for name in ('residual', 'contact', 'stance', 'endpoint'):
        assert torch.equal(snapshot['contact_off_final_original_terms'][name], reference['terms'][name][0])
    mask = snapshot['contact_mask'].to(device)
    squared = snapshot['source_contact_residual'].to(device)[:, 2:].square().sum(-1)
    contact = (squared * mask).flatten(1).sum(1) / mask.flatten(1).sum(1) / (3 * .05**2)
    torch.testing.assert_close(contact.cpu(), reference['terms']['contact'][0], atol=0, rtol=0)
    # The cached tensor inputs suffice to reconstruct every saved parameter state.
    replay = RelationalGeometry.__new__(RelationalGeometry)
    for name, value in snapshot['geometry'].items():
        setattr(replay, name, value.to(device))
    torch.testing.assert_close(replay.decode(parameters)['human'], geometry.decode(parameters)['human'], atol=0, rtol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('cell', ['a00', 'a01'])
def test_armijo_minimizes_full_objective_with_identical_recorded_output(device, cell):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    geometry, _ = _geometry()
    for name, value in vars(geometry).copy().items():
        if torch.is_tensor(value):
            setattr(geometry, name, value.to(device))
    def nearest(points, flags):
        target = points.clone()
        target[..., 0] = points[..., 0].clamp(max=0)
        return points[..., 0] > 0, target
    geometry.dataset.get_nearest_free_voxel = nearest
    zero = geometry.base.new_zeros(1, 16, geometry.dimension)
    objective = RelationalObjective(
        geometry, torch.zeros(1, dtype=torch.long, device=device),
        geometry.decode(zero)['human'][:, 2:], source_floor=True, source_stance_velocity=True,
    )
    objective.contact.fill_(True)
    frozen = (objective.stance.clone(), objective.source_feet.clone(), objective.floor_height.clone())
    kwargs = dict(steps=20, learning_rate=1., cells=(cell,), include_floor=False, solver='armijo')
    expected, expected_parameters, expected_metrics = optimize_relational_cells(geometry, objective, **kwargs)
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone() if device == 'cuda' else None
    trace = RelationalArmijoTrace()
    encoded, parameters, metrics = optimize_relational_cells(geometry, objective, trace=trace, **kwargs)
    assert torch.equal(encoded, expected) and torch.equal(parameters, expected_parameters)
    assert all(torch.equal(metrics[k], expected_metrics[k]) for k in metrics)
    assert torch.equal(cpu_rng, torch.get_rng_state())
    if device == 'cuda':
        assert torch.equal(cuda_rng, torch.cuda.get_rng_state())
    for before, after in zip(frozen, (objective.stance, objective.source_feet, objective.floor_height)):
        assert torch.equal(before, after)
    payload = trace.payload()
    assert (payload['loss'][1:] <= payload['loss'][:-1]).all()
    assert metrics['energy_total'].item() <= metrics['before_energy_total'].item()
    assert metrics['gradient_evaluations'].item() <= 20
    assert metrics['line_search_trials'].item() <= 400
    assert metrics['optimizer_updates'].item() == payload['accepted'].sum().item()
    if cell == 'a00':
        assert parameters.abs().max().item() < 1e-5
    else:
        assert metrics['energy_total'].item() < metrics['before_energy_total'].item()
        assert metrics['optimizer_updates'].item() > 0
    for trial in payload['trials']:
        take = trial['accepted']
        assert (trial['loss'][take] <= trial['armijo_bound'][take]).all()
        assert (trial['loss'][take] < payload['loss'][trial['iteration']][take]).all()
    assert torch.equal(encoded[:, :2], geometry.base[:, :2])
    assert torch.equal(encoded[..., 228:], geometry.base[..., 228:])


class ScalarRelationGeometry:
    """Analytic one-coordinate problem for complete-loss and voxel-switch tests."""
    dimension = 1
    base = torch.zeros(1, 3, 232, dtype=torch.float64)

    def decode(self, parameters):
        return {'x': parameters[:, 2, 0]}

    def encode(self, state, history):
        return torch.cat((history.expand(len(state['x']), -1, -1),
                          state['x'][:, None, None].expand(-1, 1, 232)), dim=1)


def scalar_relation_objective(residual_weight, switching=False):
    class Objective:
        coordinates = []

        def evaluate(self, state):
            x = state['x']
            self.coordinates.extend(x.detach().tolist())
            target = torch.where(x.detach() >= .4, 3., 1.) if switching else torch.ones_like(x)
            self.last_scene_query = {'nearest': target[:, None, None, None],
                                     'occupied': torch.ones(len(x), 1, 1, dtype=torch.bool)}
            zero = x * 0
            terms = {name: zero for name in ('contact', 'stance', 'floor', 'endpoint', 'hsi', 'human_scene')}
            terms.update(residual=residual_weight*x.square(),
                         object_scene=(.75 if switching else 1.)*(x-target).square())
            return terms, {}
    return Objective()


def test_armijo_uses_complete_loss_and_independent_cells():
    geometry = ScalarRelationGeometry()
    objective = scalar_relation_objective(50.)
    trace = RelationalArmijoTrace()
    _, parameters, metrics = optimize_relational_cells(
        geometry, objective, cells=('a00', 'a01'), include_floor=False,
        solver='armijo', learning_rate=1., trace=trace,
    )
    assert parameters[0].count_nonzero() == 0
    # The optimum of 50*x^2+(x-1)^2 is 1/51, far from the scene-only optimum 1.
    assert parameters[1, 2, 0].item() == pytest.approx(1/51, abs=1e-5)
    assert metrics['energy_total'][1].item() < 1.
    assert any(t['terms']['object_scene'][1] < 1 and t['loss'][1] > 1
               and not t['accepted'][1] for t in trace.trials)
    assert not trace.trials[0]['accepted'].any()
    for i, cell in enumerate(('a00', 'a01')):
        _, separate, _ = optimize_relational_cells(
            geometry, scalar_relation_objective(50.), cells=(cell,), include_floor=False,
            solver='armijo', learning_rate=1.,
        )
        torch.testing.assert_close(parameters[i], separate[0], atol=0, rtol=0)


def test_armijo_requeries_voxel_switch_and_returns_last_accepted_state():
    geometry = ScalarRelationGeometry()
    objective = scalar_relation_objective(0., switching=True)
    trace = RelationalArmijoTrace(max_backtracks=4)
    _, parameters, metrics = optimize_relational_cells(
        geometry, objective, cells=('a01',), include_floor=False,
        solver='armijo', learning_rate=1., max_backtracks=4, trace=trace,
    )
    # The first proposal x=1.5 would lower a frozen (x-1)^2 surrogate. The voxel
    # switch to target=3 makes the true loss larger, and the search must reject it.
    assert objective.coordinates[2] == 1.5
    assert trace.trials[0]['loss'].item() > trace.losses[0].item()
    assert not trace.trials[0]['accepted'].item()
    assert 0 < parameters[0, 2, 0].item() < .4
    assert metrics['optimizer_stop_code'].item() == 2
    payload = trace.payload()
    assert torch.equal(parameters, payload['parameters'][-1])
    assert (payload['loss'][1:] <= payload['loss'][:-1]).all()
    assert not payload['accepted'][-1].any()
    assert torch.equal(payload['parameters'][-1], payload['parameters'][-2])
    assert metrics['energy_total'].item() < metrics['before_energy_total'].item()


def test_empty_path_matches_the_forward_recurrence():
    betas = torch.linspace(0.0001, 0.02, 500)
    generator = torch.Generator().manual_seed(42)
    noise = torch.randn(500, 1, 2, 16, generator=generator)
    path = empty_forward_trajectory(betas, noise)
    current = torch.zeros_like(noise[0])
    for step, beta in enumerate(betas):
        current = (1-beta).sqrt() * current + beta.sqrt() * noise[step]
        torch.testing.assert_close(path[step], current, atol=3e-6, rtol=3e-6)
    alpha_bar = (1-betas).cumprod(0)
    weights = (betas / alpha_bar).sqrt()
    for earlier, later in ((0, 1), (10, 100), (100, 499)):
        variance = alpha_bar[earlier] * weights[:earlier+1].square().sum()
        covariance = (alpha_bar[earlier] * alpha_bar[later]).sqrt() * weights[:earlier+1].square().sum()
        torch.testing.assert_close(variance, 1-alpha_bar[earlier], atol=1e-6, rtol=1e-4)
        torch.testing.assert_close(covariance, (alpha_bar[later]/alpha_bar[earlier]).sqrt()*(1-alpha_bar[earlier]), atol=1e-6, rtol=1e-4)


def test_empty_view_keeps_human_and_global_rng_and_repeats_by_window_seed():
    current = torch.ones(1, 16, 232)
    view = KnownEmptyObjectView()
    before = torch.get_rng_state().clone()
    view.begin_window(current, 42)
    first = view.for_step(current, 100)
    assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(first[..., :216], current[..., :216])
    assert torch.count_nonzero(first[:, :2, 216:]) == 0
    view.begin_window(current, 42)
    assert torch.equal(first, view.for_step(current, 100))
    assert not torch.equal(first[:, 2:, 216:], view.for_step(current, 101)[:, 2:, 216:])


def test_common_motion_preserves_root_object_and_hand_object_relations():
    geometry, fixed = _geometry()
    parameters = torch.zeros(1, 16, geometry.dimension)
    base = geometry.decode(parameters)
    torch.testing.assert_close(
        base['human'][..., :22, :], geometry.world_points(geometry.positions[..., :22, :]),
        atol=2e-6, rtol=2e-5,
    )
    parameters[..., :4] = torch.tensor([0.3, -0.2, 0.4, 0.5])
    moved = geometry.decode(parameters)
    for state in (base, moved):
        state['root_world_rotation'] = geometry.world_rotation[:, None] @ state['global_rotation'][..., 0, :, :]
    for joint in (0, 22):
        point_a = base['human'][..., joint, :]
        point_b = moved['human'][..., joint, :]
        relative_a = geometry.object_rotation_world.transpose(-1, -2) @ (point_a-base['object_translation_world'])[..., None]
        relative_b = moved['object_rotation_world'].transpose(-1, -2) @ (point_b-moved['object_translation_world'])[..., None]
        torch.testing.assert_close(relative_a, relative_b, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(
        base['root_world_rotation'].transpose(-1, -2) @ base['object_rotation_world'],
        moved['root_world_rotation'].transpose(-1, -2) @ moved['object_rotation_world'],
        atol=2e-6, rtol=2e-5,
    )
    encoded = geometry.encode(moved, fixed)
    assert torch.equal(encoded[:, :2], fixed)
    assert torch.equal(encoded[..., 228:], geometry.base[..., 228:])
    reconstructed_object = geometry.prefix[:, None] @ encoded[:, 2:, 219:228].reshape(1, 14, 3, 3) @ geometry.reference[:, None]
    torch.testing.assert_close(reconstructed_object, moved['object_rotation_world'][:, 2:], atol=2e-6, rtol=2e-5)


def test_zero_residual_has_finite_nonzero_root_and_leg_gradients():
    geometry, _ = _geometry()
    parameters = torch.zeros(1, 16, geometry.dimension, requires_grad=True)
    state = geometry.decode(parameters)
    target = state['human'].detach().clone()
    target[:, 2:, 10, 0] += 0.1
    loss = (state['human']-target).square().sum() + state['object_translation_world'][:, 2:, 2].sum()
    loss.backward()
    assert torch.isfinite(parameters.grad).all()
    assert parameters.grad[:, 2:, :3].abs().sum() > 0
    assert parameters.grad[:, 2:, 4:7].abs().sum() > 0
    assert torch.count_nonzero(parameters.grad[:, :2]) == 0


def test_batched_optimizer_records_each_cell_and_reduces_scene_constraint():
    geometry, _ = _geometry()
    def nearest(points, flags):
        del flags
        occupied = points[..., 0] > 0
        target = points.clone()
        target[..., 0] = torch.minimum(points[..., 0], torch.zeros_like(points[..., 0]))
        return occupied, target
    geometry.dataset.get_nearest_free_voxel = nearest
    zero = torch.zeros(1, 16, geometry.dimension)
    target = geometry.decode(zero)['human'][:, 2:].clone()
    target[..., 0] -= 0.05
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target)
    encoded, parameters, metrics = optimize_relational_cells(geometry, objective, steps=4)
    assert encoded.shape == (4, 16, 232)
    assert metrics['gradient_finite'].eq(1).all()
    assert torch.isfinite(encoded).all()
    assert torch.equal(encoded[:, :2], geometry.base[:, :2].expand(4, -1, -1))
    assert not torch.equal(parameters[0], parameters[1])
    assert metrics['energy_human_scene'][3] + metrics['energy_object_scene'][3] < metrics['before_energy_human_scene'][3] + metrics['before_energy_object_scene'][3]


def test_registered_probe_serializes_four_cells_with_enabled_gradients(tmp_path):
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    view = KnownEmptyObjectView()
    view.begin_window(geometry.base, 42)
    conditional = geometry.base.clone()
    conditional[:, 2:, 0] += 0.01
    sampler = SimpleNamespace(
        dataset=geometry.dataset, hsi_input_view=view,
        _hsi_prediction_pair=lambda *args: (conditional, geometry.base),
    )
    context = {
        'mat': geometry.mat, 'obj_rot_mat_prefix': geometry.prefix,
        'obj_rot_mat_ref': geometry.reference,
        'seq_name_dict': {0: 'sub16_clothesstand_0'},
        'obj_rest_verts': {'clothesstand': geometry.object_points[0]},
        'scene_flag': torch.zeros(1, dtype=torch.long),
    }
    probe = RelationalPrototypeDiagnostic(steps=(0,), optimizer_steps=2)
    probe.begin_episode({'canonical_ordinal': 0, 'scene_name': 'scene', 'object_name': 'clothesstand', 'test_idx': 0}, tmp_path)
    probe.begin_window(geometry.base)
    before = torch.get_rng_state().clone()
    probe.observe(sampler, geometry.base, geometry.base, torch.tensor([0]), context, geometry.offsets, geometry.base)
    assert torch.equal(before, torch.get_rng_state())
    payload = json.loads(Path(probe.finish_episode()['path']).read_text())
    assert payload['probe'] == 'relational_constrained_window'
    assert set(payload['metrics']) == {'a00', 'a10', 'a01', 'a11'}
    assert all(record['initial_t0_gradient_finite'] == 1 for record in payload['metrics'].values())


def test_independent_rollout_cell_matches_its_batched_window_cell():
    geometry, _ = _geometry()
    def nearest(points, flags):
        target = points.clone()
        target[..., 0] = points[..., 0].clamp(max=0)
        return points[..., 0] > 0, target
    geometry.dataset.get_nearest_free_voxel = nearest
    target = geometry.decode(torch.zeros(1, 16, geometry.dimension))['human'][:, 2:] - 0.02
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target)
    batch_output, batch_parameters, _ = optimize_relational_cells(geometry, objective, steps=3)
    for index, cell in enumerate(CELL_KEYS):
        output, parameters, metrics = optimize_relational_cells(geometry, objective, steps=3, cells=(cell,))
        torch.testing.assert_close(output[0], batch_output[index], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(parameters[0], batch_parameters[index], atol=2e-6, rtol=2e-5)
        assert metrics['gradient_finite'].item() == 1


@pytest.mark.parametrize('source_stance_velocity', [False, True])
@pytest.mark.parametrize('solver', ['adam', 'armijo'])
def test_corrector_applies_real_optimizer_and_preserves_history_contacts(source_stance_velocity, solver):
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    geometry.dataset.scene_grid_torch = torch.tensor([-3., 0., -4., 3., 2., 4., 300., 100., 400.])
    context = {
        'mat': geometry.mat, 'obj_rot_mat_prefix': geometry.prefix,
        'obj_rot_mat_ref': geometry.reference,
        'seq_name_dict': {0: 'sub16_clothesstand_0'},
        'obj_rest_verts': {'clothesstand': geometry.object_points[0]},
        'scene_flag': torch.zeros(1, dtype=torch.long),
    }
    sampler = SimpleNamespace(dataset=geometry.dataset, inner_hoi=SimpleNamespace(sample_calls=1),
                              _hsi_prediction_pair=lambda *args: (geometry.base, geometry.base))
    corrector = RelationalCorrector(cell='a01', optimizer_steps=2,
                                    source_stance_velocity=source_stance_velocity,
                                    solver=solver, learning_rate=1. if solver == 'armijo' else .05)
    arguments = (sampler, geometry.base, geometry.base, torch.tensor([0]), context, geometry.offsets, geometry.base)
    assert corrector.correct(*arguments, step=9) is geometry.base
    with torch.no_grad():
        result = corrector.correct(*arguments, step=0)
    assert result.shape == geometry.base.shape
    assert not torch.equal(result, geometry.base)
    assert torch.equal(result[:, :2], geometry.base[:, :2])
    assert torch.equal(result[..., 228:], geometry.base[..., 228:])
    audit = corrector.audit_dict()
    assert audit['source_stance_velocity'] is source_stance_velocity
    if source_stance_velocity:
        assert audit['records'][0]['metrics']['before_energy_stance'] == 0
        assert audit['records'][0]['metrics']['before_stance_increment_cm'] == 0
    assert audit['calls'] == 1
    assert audit['records'][0]['metrics']['gradient_finite'] == 1
    assert audit['records'][0]['history_exact'] and audit['records'][0]['contact_exact']
    recorded = RelationalCorrector(cell='a01', optimizer_steps=2, record_motion=True,
                                   source_stance_velocity=source_stance_velocity,
                                   optimizer_diagnostic=source_stance_velocity and solver == 'adam',
                                   solver=solver, learning_rate=1. if solver == 'armijo' else .05,
                                   solver_diagnostic=solver == 'armijo')
    rng_before = torch.get_rng_state().clone()
    with torch.no_grad():
        observed = recorded.correct(*arguments, step=0)
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert torch.equal(result, observed)
    assert recorded.records[0]['metrics'] == corrector.records[0]['metrics']
    snapshot = recorded.motion_records[0]
    if source_stance_velocity and solver == 'adam':
        assert snapshot['optimizer_diagnostic']['reference']['parameters'].shape[0] == 3
        assert recorded.audit_dict()['shadow_optimizer_gradient_steps'] == 2
    if solver == 'armijo':
        assert snapshot['solver_diagnostic']['armijo']['parameters'].shape[0] <= 3
        assert snapshot['solver_diagnostic']['adam']['parameters'].shape[0] == 21
        assert recorded.audit_dict()['solver_shadow_gradient_steps'] == 20
    source = snapshot['states']['source']['human']
    stance = source[..., (7, 8, 10, 11), 1] < torch.tensor([.08, .08, .04, .04])
    assert torch.equal(snapshot['stance_mask'], stance[:, 2:] & stance[:, 1:-1])
    decoded = geometry.decode(snapshot['parameters'])
    torch.testing.assert_close(snapshot['states']['corrected']['human'], decoded['human'])
    for state in snapshot['states'].values():
        assert all(value.device.type == 'cpu' and not value.requires_grad for value in state.values())


def test_zero_step_reconstruction_projects_redundant_positions_without_motion_update():
    geometry, _ = _geometry()
    # A denoiser can predict redundant positions inconsistent with its rotations.
    geometry.positions[:, 2:, 10, 0] += 0.1
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    zero = torch.zeros(1, 16, geometry.dimension)
    state = geometry.decode(zero)
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), state['human'][:, 2:])
    encoded, parameters, metrics = optimize_relational_cells(geometry, objective, steps=0, cells=('a00',))
    torch.testing.assert_close(encoded, geometry.encode(state, geometry.base[:, :2]))
    assert parameters.count_nonzero() == 0
    assert metrics['gradient_max'].item() == 0
    torch.testing.assert_close(state['global_rotation'], geometry.global_rotation, atol=2e-6, rtol=2e-5)
    torch.testing.assert_close(state['object_position'], geometry.object_position)
    assert not torch.allclose(state['local_fk'][:, 2:, 10], geometry.positions[:, 2:, 10])
    assert torch.equal(encoded[:, :2], geometry.base[:, :2])
    assert torch.equal(encoded[..., 228:], geometry.base[..., 228:])


def test_floor_exclusion_matches_independent_zero_floor_objective_and_keeps_telemetry():
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points,
    )
    target = geometry.decode(torch.zeros(1, 16, geometry.dimension))['human'][:, 2:]
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), target)

    class ZeroFloorObjective:
        def evaluate(self, state):
            terms, metrics = objective.evaluate(state)
            terms['floor'] = terms['floor'] * 0
            return terms, metrics

    default, default_parameters, _ = optimize_relational_cells(geometry, objective, steps=3, cells=('a00',))
    explicit, explicit_parameters, _ = optimize_relational_cells(geometry, objective, steps=3, cells=('a00',), include_floor=True)
    assert torch.equal(default, explicit)
    assert torch.equal(default_parameters, explicit_parameters)
    actual, parameters, metrics = optimize_relational_cells(geometry, objective, steps=3, cells=('a00',), include_floor=False)
    expected, expected_parameters, _ = optimize_relational_cells(geometry, ZeroFloorObjective(), steps=3, cells=('a00',))
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(parameters, expected_parameters)
    assert not torch.allclose(parameters, default_parameters)
    assert metrics['energy_floor'].item() > 0
    assert metrics['gradient_finite'].item() == 1


def test_corrected_clean_drives_posterior_and_next_scene_reference():
    from tests.phase2.test_composed_sampler import _evaluator_arguments, _make_composed_with_hsi
    sampler, _, _, _ = _make_composed_with_hsi(0, batch=1)
    arguments = _evaluator_arguments(batch=1)
    arguments['fixed_points'][..., 219:228] = torch.eye(3).flatten()
    arguments['human_dict'] = {'rest_human_offsets': torch.zeros(24, 3)}
    posterior_clean = {}
    original_posterior = sampler.inner_hoi.diffusion.posterior_sample

    class Correction:
        def correct(self, owner, current, previous_x0, timesteps, context, offsets, clean, step):
            if step < 499:
                assert torch.equal(previous_x0, posterior_clean[step+1])
            result = clean.clone()
            if step in (10, 1, 0):
                result[:, 2:, 0] += 0.025
            posterior_clean[step] = result.clone()
            return result

        def audit_dict(self):
            return {'calls': 3}

    def posterior(current, clean, timesteps, noise, fixed):
        assert torch.equal(clean, posterior_clean[int(timesteps[0])])
        return original_posterior(current, clean, timesteps, noise, fixed)

    sampler.relational_corrector = Correction()
    sampler.inner_hoi.diffusion.posterior_sample = posterior
    samples, _ = sampler.p_sample_loop(**arguments)
    assert len(posterior_clean) == 500
    coefficient = sampler.inner_hoi.diffusion.posterior_mean_coef1[0]
    torch.testing.assert_close(samples[-1][:, 2:, :219], coefficient * posterior_clean[0][:, 2:, :219])
    assert torch.equal(samples[-1][:, :2], arguments['fixed_points'])
    audit = sampler.audit_dict()['composition']
    assert audit['object_channels_from_hoi'] is False
    assert audit['relational_correction']['calls'] == 3
