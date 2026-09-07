"""Conditional pose-repair scheduler, authority, relation and failure contracts."""
from types import SimpleNamespace

import pytest
import torch
from pytorch3d import transforms

from tests.phase2.test_relational import _geometry
from tests.phase2.test_scene_evidence import QuerySampler
from mixer.conditional_repair import ConditionalRepairEditor, ConstrainedPoseFit, locked_encode, ddim_repair_step
from mixer.kinematic_composition import _local_from_global
from models.infbagel import DDIMSolver
from priors.core.ddpm import GaussianDiffusion


def fixture():
    geometry, _ = _geometry()
    dataset = geometry.dataset
    dataset.scene_grid_torch = torch.tensor([-100., -100., -100., 100., 100., 100., 10., 10., 10.])
    dataset.get_nearest_free_voxel = lambda p, f: (torch.zeros(p.shape[:-1], dtype=torch.bool), p)
    context = dict(mat=geometry.mat, obj_rot_mat_prefix=geometry.prefix,
        obj_rot_mat_ref=geometry.reference, scene_flag=torch.zeros(1, dtype=torch.long),
        seq_name_dict={0:'sub16_cube_0'}, obj_rest_verts={'cube':geometry.object_points[0]})
    return geometry, context


def test_partial_ddim_matches_analytic_skips_and_finishes_at_constrained_clean():
    diffusion = GaussianDiffusion()
    solver = DDIMSolver(diffusion.sqrt_alpha_bar.square().numpy(), 500, 25)
    clean = torch.randn(1, 16, 232)
    noise = torch.randn_like(clean)
    current = diffusion.q_sample(clean, torch.tensor([99]), noise)
    for index, level in zip(range(4, -1, -1), (99,79,59,39,19)):
        epsilon = (current-diffusion.sqrt_alpha_bar[level]*clean)/diffusion.sqrt_one_minus_alpha_bar[level]
        actual = ddim_repair_step(diffusion, solver, current, clean, level, index)
        alpha = solver.ddim_alpha_cumprods_prev[index].sqrt()
        expected = alpha*clean+(1-alpha.square()).sqrt()*epsilon
        expected[:, :2] = clean[:, :2]
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        current = actual
        current[:, :2] = clean[:, :2]
    assert torch.equal(current, clean)


def test_pose_fit_zero_reconstruction_and_nonzero_local_only_authority():
    geometry, context = fixture()
    fit = ConstrainedPoseFit(geometry, geometry, context['scene_flag'])
    original_rng = torch.get_rng_state().clone()
    parameters, traces = fit.fit(fit.zero, geometry.local_rotation)
    assert torch.equal(parameters, fit.zero)
    assert torch.equal(locked_encode(geometry, parameters), geometry.base)
    assert all(t['reason']==['zero_gradient'] for t in traces)
    # Hand-free head rotation is a useful local repair, even with active grasps.
    fit.objective.contact.fill_(True)
    target = geometry.local_rotation.clone()
    target[:, 2:, 15] = target[:, 2:, 15] @ transforms.axis_angle_to_matrix(torch.tensor([.08, 0., 0.]))
    before = geometry.base.clone()
    parameters, traces = fit.fit(parameters, target)
    assert parameters[...,4:].abs().max()>0
    result = locked_encode(geometry, parameters)
    assert torch.equal(result[:,:2], before[:,:2])
    for a,b in ((0,3),(84,90),(216,232)):
        assert torch.equal(result[...,a:b],before[...,a:b])
    assert all(bool(v.all()) for v in fit.guards(parameters).values())
    assert torch.equal(original_rng, torch.get_rng_state())


def test_contact_guard_measures_real_source_anchors_and_fixed_mask():
    geometry, context = fixture()
    fit = ConstrainedPoseFit(geometry, geometry, context['scene_flag'])
    fit.objective.contact.fill_(True)
    anchors = fit.objective.hand_anchor.clone()
    large = fit.zero.clone()
    large[:,2:,4+3*15:4+3*16] = 2  # articulated shoulder affects actual hand FK
    assert not fit.guards(large)['contact'].all()
    assert torch.equal(anchors,fit.objective.hand_anchor)
    # An inactive source hand imposes no new grasp obligation.
    fit.objective.contact.zero_()
    assert fit.guards(large)['contact'].all()


@pytest.mark.parametrize('enabled,nonfinite', [(False,False),(True,False),(True,True)])
def test_editor_full_condition_known_empty_private_rng_and_visible_fallback(enabled, nonfinite):
    geometry, context = fixture()
    sampler = QuerySampler()
    sampler.dataset = geometry.dataset
    inputs = []
    def model(view, *common, **kwargs):
        assert kwargs == {'is_sample':True}
        assert not common[13].any()
        assert not view[:,:2,216:].any()
        assert view[:,2:,216:].abs().max()>0
        inputs.append(view.clone())
        target = geometry.base.clone()
        if nonfinite:
            target[...,90] = float('nan')
        else:
            target[...,84:216] = transforms.matrix_to_rotation_6d(geometry.global_rotation).flatten(-2)
        return target
    sampler.hsi_sampler = SimpleNamespace(student_model=model,
        solver=DDIMSolver(sampler.inner_hoi.diffusion.sqrt_alpha_bar.square().numpy(),500,25))
    # Zero geometry step isolates repair; production uses its frozen positive step.
    editor = ConditionalRepairEditor(enabled=True, repair_enabled=enabled, lambda_dp=0,
        mode='edit', noise_levels=(99,), initial_step=0., record_motion=True)
    rng = torch.get_rng_state().clone()
    result = editor.edit(sampler, geometry.base, {}, None, context, geometry.offsets, 42)
    assert torch.equal(rng,torch.get_rng_state())
    assert len(inputs)==(5 if enabled else 0)
    record=editor.records[-1]
    assert record['history_exact'] and record['common_exact'] and record['contact_exact']
    assert all(record['final_guards'].values())
    assert torch.isfinite(result).all()
    if not enabled or nonfinite:
        assert record['fallback']
    if nonfinite:
        assert all(t['reason']=='nonfinite_prediction' for t in record['steps'])
    if enabled:
        assert len(sampler.worlds)==5
        # Clean query carries real object, independently of masked model view.
        assert torch.equal(sampler.worlds[0][0][...,216:],editor.motion_records[-1]['proposal'][...,216:])


def test_baseline_returns_actual_native_source_exactly():
    geometry, context=fixture()
    sampler=QuerySampler()
    sampler.dataset=geometry.dataset
    editor=ConditionalRepairEditor(enabled=True,baseline_hoi=True,lambda_dp=0,mode='edit')
    result=editor.edit(sampler,geometry.base,{},None,context,geometry.offsets,42)
    assert torch.equal(result,geometry.base)
    assert sampler.pairs==sampler.raw_inputs==[]
    assert editor.records[-1]['hsi_calls']==0


def test_free_coordinate_projection_keeps_exact_locks_despite_svd_roundoff(monkeypatch):
    import mixer.conditional_repair as repair
    from mixer.relation_projection import nullspace_projection
    generator=torch.Generator().manual_seed(42)
    jacobian=torch.randn(1,14,6,67,generator=generator)
    jacobian[...,:4]=0
    gradient=torch.randn(1,16,67,generator=generator)
    gradient[...,:4]=0
    # Including zero columns in finite precision does not guarantee exact zeros.
    old,_=nullspace_projection(jacobian,gradient[:,2:])
    assert torch.count_nonzero(old[...,:4])>0
    monkeypatch.setattr(repair,'contact_jacobian',lambda objective,p:(jacobian,torch.zeros(1,14,6)))
    projected=repair.project_local_pose(None,torch.zeros_like(gradient),gradient)
    assert torch.count_nonzero(projected[...,:4])==0
    assert torch.count_nonzero(projected[:,:2])==0
    assert projected[...,4:].abs().max()>0
    torch.testing.assert_close((jacobian.double()@projected[:,2:].double().unsqueeze(-1)).squeeze(-1),torch.zeros(1,14,6,dtype=torch.float64),atol=1e-6,rtol=0)
