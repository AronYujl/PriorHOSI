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


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_actual_native_object_query_rng_is_independent(device):
    from models.infbagel import Sampler
    from tests.phase2.test_hsi_inference_engineering import CountingSceneDataset, _arguments
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    args=list(_arguments())
    args=[a.to(device) if torch.is_tensor(a) else a for a in args]
    args[13]={'cube':torch.arange(4500,device=device,dtype=torch.float32).reshape(1500,3)/1000}
    args[14]={0:'sub10_cube_000'};args[15]=torch.eye(3,device=device)[None]
    native=Sampler(device=device,mask_ind=0,emb_f=0,batch_size=1,channel=232,
                   auto_regre_num=2,timesteps=500,ddim_timesteps=25,cm_timesteps=16,
                   scene_type='occ_temp',temp_voxel_num=3,
                   occ_list_layout_repaired=True)
    dataset=CountingSceneDataset();dataset.vis=True
    samples=[];get_occ=dataset.get_occ_for_points
    def observe(points,object_points,flag):
        if object_points is not None:samples.append(object_points.clone())
        return get_occ(points,object_points,flag)
    dataset.get_occ_for_points=observe
    native.set_dataset_and_model(dataset,torch.nn.Linear(1,1).to(device))
    class ActualQuery(QuerySampler):
        def _hsi_model_arguments(self,current,previous,t,context):
            native._compute_occ_sample(current,previous,*args[2:])
            return super()._hsi_model_arguments(current,previous,t,context)
    source=args[0];sampler=ActualQuery()
    sampler.inner_hoi.diffusion.to(device)
    cpu=torch.random.get_rng_state().clone()
    cuda=torch.cuda.get_rng_state().clone() if device=='cuda' else None
    teacher=SceneEvidenceTeacher(sampler,{},None,{},source,42)
    teacher.query(source,250)
    assert torch.equal(cpu,torch.random.get_rng_state())
    if cuda is not None:assert torch.equal(cuda,torch.cuda.get_rng_state())
    first=[p.clone() for p in samples];samples.clear()
    torch.rand(10)
    another=SceneEvidenceTeacher(sampler,{},None,{},source,42)
    another.query(source,250)
    assert len(first)==5
    assert all(torch.equal(a,b) for a,b in zip(first,samples))


def test_scene_balanced_scale_excludes_verification_and_inactive_sources():
    from mixer.scene_calibration import estimate_global_scale
    def episode(scene,ratios,energy=1.):
        return dict(episode=dict(scene_name=scene),records=[dict(mode='calibrate',
            source_terms=dict(human_scene=[energy],object_scene=[0.]),
            iterations=[dict(parameter_gradient_norms=dict(explicit=[ratio],hsi_evidence=[1.]))
                        for _ in range(8)]) for ratio in ratios])
    episodes=[episode('a',[10.,10.,1000.]),episode('b',[20.,20.]),episode('c',[30.]),
              episode('holdout',[1e9]),episode('a',[1e10],energy=0.)]
    result=estimate_global_scale(episodes,['a','b','c'])
    assert result['lambda_dp']==2. and result['active_windows']=={'a':3,'b':2,'c':1}
    assert result['inactive_windows']==1
    with pytest.raises(ValueError,match='coverage'):
        estimate_global_scale(episodes,['a','b','missing'])


def test_passive_calibration_returns_source_and_leaves_parameters_zero():
    geometry,_=_geometry();sampler=QuerySampler();sampler.dataset=geometry.dataset
    sampler.dataset.scene_grid_torch=torch.tensor([-100.,-100.,-100.,100.,100.,100.,10.,10.,10.])
    sampler.dataset.get_nearest_free_voxel=lambda p,f:(torch.zeros(p.shape[:-1],dtype=torch.bool),p)
    context=dict(mat=geometry.mat,obj_rot_mat_prefix=geometry.prefix,obj_rot_mat_ref=geometry.reference,
                 scene_flag=torch.zeros(1,dtype=torch.long),seq_name_dict={0:'sub10_cube_0'},
                 obj_rest_verts={'cube':geometry.object_points[0]})
    editor=SceneEvidenceEditor(enabled=True,mode='calibrate',lambda_dp=1.,noise_levels=(250,100))
    result=editor.edit(sampler,geometry.base,{},None,context,geometry.offsets,42)
    assert torch.equal(result,geometry.base)
    assert not editor.motion_records[0]['parameters'].any()
    assert editor.records[0]['returned_source_exact']
    assert torch.equal(editor.motion_records[0]['edited'],geometry.base)
    assert editor.records[0]['edit_rms']==0
    assert all(i['hoi_reference_rms']==[0.] for i in editor.records[0]['iterations'])
