"""DP teacher coupling, local surrogate, domain and native-chain contracts."""
from types import SimpleNamespace
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

import pytest
import torch

from mixer.scene_evidence import (SceneEvidenceEditor, SceneEvidenceTeacher,
    domain_state, domain_accepts, editable_mask, epsilon_from_x0,
    evidence_direction, local_armijo)
from priors.core.ddpm import GaussianDiffusion
from tests.phase2.test_relational import _geometry
from tests.phase2.test_composed_sampler import _make_pair, _evaluator_arguments


def test_epsilon_conversion_and_gaussian_evidence_sign():
    alpha, sigma = .8, .6
    clean, noise = torch.tensor(2.), torch.tensor(-.5)
    z = alpha * clean + sigma * noise
    torch.testing.assert_close(epsilon_from_x0(z, clean, alpha, sigma), noise)
    # Gaussian base mean0 versus conditional mean1: negative epsilon difference.
    delta = epsilon_from_x0(z, torch.tensor(1.), alpha, sigma) - epsilon_from_x0(z, torch.tensor(0.), alpha, sigma)
    assert delta < 0
    assert -alpha / sigma * delta > 0


def test_mask_and_hsi_output_authority():
    shape = (2, 16, 232)
    hoi, hsi = torch.ones(shape), torch.full(shape, 100.)
    mask = editable_mask(hoi)
    v = evidence_direction(hoi, hsi, .8, .6, 1., .1, mask)
    assert torch.count_nonzero(v[:, :2]) == 0
    assert torch.count_nonzero(v[..., 228:]) == 0
    torch.testing.assert_close(v[:, 2:, 216:228], torch.full((2, 14, 12), .8/.6))


class QuerySampler:
    def __init__(self):
        self.inner_hoi = SimpleNamespace(diffusion=GaussianDiffusion(), sample_calls=5)
        self.raw_inputs, self.worlds, self.pairs = [], [], []
        self.weight = torch.nn.Parameter(torch.tensor(.2), requires_grad=False)
    def _hoi_raw_x0(self, z, t, arguments, bps):
        self.raw_inputs.append(z.clone())
        return z * self.weight
    def _hsi_model_arguments(self, current, previous, t, context):
        self.worlds.append((current.clone(), previous.clone()))
        common = [torch.tensor(float(i)) for i in range(17)]
        common[1] = t
        common[13] = torch.ones(current.shape[0], dtype=torch.bool)
        return tuple(common)
    def _hsi_predict_pair(self, z, common):
        self.pairs.append((z.clone(), common))
        return z + self.weight, z


def test_teacher_shared_noise_clean_world_empty_view_cancellation_and_rng():
    sampler = QuerySampler()
    source = torch.randn(1, 16, 232)
    state = torch.random.get_rng_state().clone()
    teacher = SceneEvidenceTeacher(sampler, {}, None, {}, source, 42)
    direction, record = teacher.query(source, 250)
    assert torch.equal(state, torch.random.get_rng_state())
    assert torch.equal(sampler.raw_inputs[0], sampler.raw_inputs[1])
    assert record['hoi_reference_rms'] == [0.]
    assert record['hsi_evidence_rms'][0] > 0
    assert not direction.requires_grad and sampler.weight.grad is None
    assert torch.count_nonzero(direction[:, :2]) == 0
    assert torch.count_nonzero(direction[..., 216:]) == 0
    assert torch.equal(sampler.worlds[0][0], source)
    assert torch.equal(sampler.worlds[0][1], source)
    view, common = sampler.pairs[0]
    assert torch.count_nonzero(view[:, :2, 216:]) == 0
    assert torch.count_nonzero(view[:, 2:, 216:]) > 0
    assert torch.equal(view[..., :216], sampler.raw_inputs[0][..., :216])
    assert not common[13].any()
    assert sampler.inner_hoi.sample_calls == 5
    teacher.query(source + .1, 200)
    assert torch.equal(sampler.worlds[1][0], source + .1)
    again = SceneEvidenceTeacher(QuerySampler(), {}, None, {}, source, 42)
    repeated, _ = again.query(source, 250)
    assert torch.equal(direction, repeated)


def test_lambda_zero_skips_hsi_and_source_reference_stays_fixed():
    source = torch.randn(1, 16, 232)
    sampler = QuerySampler()
    teacher = SceneEvidenceTeacher(sampler, {}, None, {}, source, 42, lambda_dp=0.)
    direction, _ = teacher.query(source, 250)
    assert not direction.any() and sampler.pairs == []
    teacher.query(source + 1, 200)
    assert torch.equal(teacher.source, source)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_local_armijo_analytic_quadratic_and_independent_rejection(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    origin = torch.zeros(2, 1, 1, device=device)
    teacher = torch.tensor([-1., -100.], device=device)
    calls = []
    def energy(a):
        calls.append(a.detach().clone())
        x = a[:, 0, 0]
        return teacher * x + .5 * x.square()
    def valid(a):
        return torch.tensor([True, False], device=device)
    with torch.no_grad():
        result, trace = local_armijo(origin, energy, valid, max_backtracks=3)
    torch.testing.assert_close(result[:, 0, 0], torch.tensor([1., 0.], device=device))
    assert trace['reason'] == ['accepted', 'search_exhausted']
    assert trace['value'] == [0., 0.]
    assert trace['trials'][0]['value'][0] == -0.5


def test_zero_teacher_and_satisfied_explicit_target_stay_exact():
    p = torch.zeros(1, 1, 1)
    result, trace = local_armijo(p, lambda a: a.square().flatten(1).sum(1),
                                 lambda a: torch.ones(1, dtype=torch.bool))
    assert torch.equal(result, p)
    assert trace['reason'] == ['zero_gradient'] and trace['trials'] == []


def test_nonfinite_gradient_is_a_failure():
    with pytest.raises(FloatingPointError):
        local_armijo(torch.zeros(1, 1, 1),
                     lambda a: a.flatten(1).sum(1) * float('nan'),
                     lambda a: torch.ones(1, dtype=torch.bool))


def test_float_bounds_reject_new_outside_and_increasing_existing_distance():
    grid = torch.tensor([0., 0., 0., 1., 1., 1., 100., 100., 100.])
    human = torch.full((1, 16, 24, 3), .5)
    obj = torch.full((1, 16, 2, 3), .5)
    def state(points):
        return domain_state(dict(human=human, object_surface=points), grid)
    inside = state(obj)
    exterior = obj.clone(); exterior[:, 2:, 0, 0] = -.001
    out = state(exterior)
    assert not domain_accepts(inside, out).item()
    assert out[2].all() and not out[1].all()  # truncation is not geometry
    worse = exterior.clone(); worse[:, 2:, 0, 0] = -.002
    assert not domain_accepts(out, state(worse)).item()
    assert domain_accepts(out, inside).item()
    upper = obj.clone(); upper[:, 2:, 0, 0] = 1.
    assert not domain_accepts(inside, state(upper)).item()


def test_disabled_editor_is_bitwise_native_and_rng_compatible():
    reference, composed, _, _ = _make_pair()
    composed.scene_editor = SceneEvidenceEditor(enabled=False)
    arguments = _evaluator_arguments(batch=1)
    torch.manual_seed(42)
    expected, _ = reference.p_sample_loop(**arguments)
    state = torch.random.get_rng_state().clone()
    torch.manual_seed(42)
    actual, _ = composed.p_sample_loop(**arguments)
    assert torch.equal(expected[0], actual[-1])
    assert torch.equal(state, torch.random.get_rng_state())
    assert composed.scene_editor.records == []


def test_relational_vjp_under_no_grad_and_frozen_teacher():
    geometry, fixed = _geometry()
    with torch.no_grad():
        origin = torch.zeros(1, 16, geometry.dimension)
        source = geometry.encode(geometry.decode(origin), fixed)
        v = torch.zeros_like(source); v[:, 2:, 0] = -1.
        def energy(a):
            x = geometry.encode(geometry.decode(a), fixed)
            return (v * (x - source)).flatten(1).sum(1) + .5*a.square().flatten(1).sum(1)
        result, trace = local_armijo(origin, energy,
                                     lambda a: torch.ones(1, dtype=torch.bool))
        edited = geometry.encode(geometry.decode(result), fixed)
    assert trace['accepted'] == [True]
    assert torch.isfinite(result).all() and result.abs().max() > 0
    assert torch.equal(edited[:, :2], fixed)
    assert torch.equal(edited[..., 228:], source[..., 228:])
    assert energy(result).item() < 0


def test_real_geometry_editor_history_recording_and_rng():
    geometry, fixed = _geometry()
    sampler = QuerySampler()
    sampler.dataset = geometry.dataset
    sampler.dataset.scene_grid_torch = torch.tensor([-100., -100., -100., 100., 100., 100., 10., 10., 10.])
    sampler.dataset.get_nearest_free_voxel = lambda points, flags: (
        torch.zeros(points.shape[:-1], dtype=torch.bool), points)
    context = dict(mat=geometry.mat, obj_rot_mat_prefix=geometry.prefix,
                   obj_rot_mat_ref=geometry.reference, scene_flag=torch.zeros(1, dtype=torch.long),
                   seq_name_dict={0: 'sub16_cube_0'}, obj_rest_verts={'cube': geometry.object_points[0]})
    editor = SceneEvidenceEditor(enabled=True, noise_levels=(250, 100), record_motion=True)
    rng = torch.random.get_rng_state().clone()
    with torch.no_grad():
        result = editor.edit(sampler, geometry.base, {}, None, context, geometry.offsets, 42)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert torch.equal(result[:, :2], geometry.base[:, :2])
    assert torch.equal(result[..., 228:], geometry.base[..., 228:])
    assert torch.isfinite(result).all()
    assert len(editor.records[0]['iterations']) == 2
    assert editor.records[0]['hsi_teacher_calls'] == 4
    assert editor.records[0]['iterations'][0]['hoi_reference_rms'] == [0.]
    unrecorded = SceneEvidenceEditor(enabled=True, noise_levels=(250, 100), record_motion=False)
    other = unrecorded.edit(sampler, geometry.base, {}, None, context, geometry.offsets, 42)
    assert torch.equal(result, other)
    assert unrecorded.motion_records == []
    assert sampler.weight.grad is None


def test_actual_position_normalization_is_shared():
    import numpy as np
    root = Path(__file__).resolve().parents[2]
    arrays = [np.load(root / 'data' / part / 'norm.npy') for part in ('train', 'test', 'dataset')]
    assert all(np.array_equal(arrays[0][:2], array[:2]) for array in arrays[1:])
