"""Temporal teacher lattice, actual occupancy primitive and paired diagnostics."""
from types import SimpleNamespace
from pathlib import Path
import sys
import json
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))

import pytest
import torch

from datasets.infbagel import InfBaGelDataset
from mixer.scene_evidence import SceneEvidenceEditor
from mixer.scene_evidence_diagnostics import alignment, physical_probe, physical_rms
from mixer.scene_views import teacher_common, temporal_environment
from tests.phase2.test_hsi_inference_engineering import _sampler_and_dataset, _arguments
from tests.phase2.test_relational import _geometry
from mixer.relational import RelationalObjective


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_native_temporal_coordinates_axes_and_static_storage(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    sampler, dataset = _sampler_and_dataset()
    sampler.device = device
    dataset.scene_grid_torch = torch.tensor([-10., -10., -10., 10., 10., 10., 20., 20., 20.], device=device)
    args = [x.to(device) if torch.is_tensor(x) else x for x in _arguments()]
    args[1] = args[0]
    args[2] = torch.tensor([[[0., 0., 1., .4], [0., 1., 0., 1.3], [-1., 0., 0., -.2], [0., 0., 0., 1.]]], device=device)
    points_seen = []
    get_occ = dataset.get_occ_for_points
    def observe(points, objects, flags):
        points_seen.append((points.clone(), objects))
        return get_occ(points, objects, flags)
    dataset.get_occ_for_points = observe
    occ, blocks, positions = sampler._compute_occ_sample(*args)
    native_points = [x[0] for x in points_seen[-3:]]
    common = [torch.zeros(1, device=device)] * 17
    common[0], common[15], common[16] = occ, blocks, positions
    common = tuple(common)
    context = dict(mat=args[2], scene_flag=args[3])
    snapshot = blocks.clone()
    rng = torch.get_rng_state().clone()
    updated, meta = temporal_environment(common, args[0], context, sampler)
    assert all(torch.equal(x, y[0]) for x, y in zip(native_points, points_seen[-3:]))
    assert all(y[1] is None for y in points_seen[-3:])
    assert torch.equal(updated[15], common[15])  # object-independent dataset
    assert all(updated[i] is common[i] for i in range(17) if i != 15)
    assert torch.equal(snapshot, blocks)
    assert meta['frames'] == sampler._get_temp_frame_indices(3)
    assert torch.equal(rng, torch.get_rng_state())
    assert teacher_common(common, args[0], context, sampler, 'legacy_occupied') is common
    shifted, shifted_meta = temporal_environment(common, args[0], context, sampler, 2.)
    torch.testing.assert_close(shifted_meta['centers'] - meta['centers'],
                               (2 * args[2][:, :3, 0]).expand(3, -1, -1))
    assert torch.equal(shifted[15][:1], common[15][:1])
    with pytest.raises(ValueError, match='batch1'):
        temporal_environment(common, args[0].expand(2, -1, -1), context, sampler)


def test_actual_environment_primitive_preserves_wall_under_object_overlay():
    dataset = InfBaGelDataset.__new__(InfBaGelDataset)
    dataset.scene_grid_torch = torch.tensor([0., 0., 0., 3., 3., 3., 3., 3., 3.])
    dataset.scene_occ = torch.zeros(1, 3, 3, 3, dtype=torch.bool)
    dataset.scene_occ[0, 1, 1, 1] = True
    dataset.nb_voxels = [1, 1, 2]
    dataset.train, dataset.vis, dataset.load_object_goal = False, True, False
    points = torch.tensor([[[1.2, 1.2, 1.2], [2.2, 1.2, 1.2]]])
    flag = torch.zeros(1, dtype=torch.long)
    before = dataset.scene_occ.clone()
    fused = dataset.get_occ_for_points(points, points, flag)
    environment = dataset.get_occ_for_points(points, None, flag)
    assert fused.flatten().tolist() == [2, 2]
    assert environment.flatten().tolist() == [1, 0]
    assert torch.equal(before, dataset.scene_occ)


def test_physical_probe_matches_future_equal_group_rms_and_zero_ray():
    geometry, _ = _geometry()
    geometry.dataset.get_nearest_free_voxel = lambda p, f: (torch.zeros(p.shape[:-1], dtype=torch.bool), p)
    p = torch.zeros(1, 16, geometry.dimension)
    objective = RelationalObjective(geometry, torch.zeros(1, dtype=torch.long), geometry.decode(p)['human'][:, 2:])
    grid = torch.tensor([-100., -100., -100., 100., 100., 100., 10., 10., 10.])
    g = torch.zeros_like(p)
    g[:, 2:, 0] = 1
    for target in (.001, .005):
        record, proposal = physical_probe(geometry, objective, p, g, target, grid)
        assert record['reason'] == 'matched' and record['domain_admissible']
        assert record['actual_rms_m'] == pytest.approx(target, abs=2e-7)
        assert torch.equal(geometry.decode(proposal)['human'][:, :2], geometry.decode(p)['human'][:, :2])
    record, proposal = physical_probe(geometry, objective, p, g * 0, .005, grid)
    assert record['reason'] == 'zero_direction' and torch.equal(proposal, p)
    assert alignment(g, g * 0)['cosine'] is None
    assert alignment(g, -g)['cosine'] == pytest.approx(-1.)


def test_mismatched_view_is_not_a_production_method():
    with pytest.raises(ValueError, match='diagnostic-only'):
        SceneEvidenceEditor(teacher_scene_view='mismatched_environment_temporal')
    with pytest.raises(ValueError, match='passive'):
        SceneEvidenceEditor(mode='edit', diagnostics={'enabled': True})


@pytest.mark.parametrize('probe', ['views', 'relation_compatible'])
def test_fixed_source_pairing_static_base_and_passive_history(probe):
    from tests.phase2.test_scene_evidence import QuerySampler
    from mixer.scene_evidence import SceneEvidenceTeacher
    from mixer.scene_evidence_diagnostics import run_fixed_source_views
    geometry, _ = _geometry()
    native, dataset = _sampler_and_dataset()
    dataset.scene_grid_torch = torch.tensor([-100., -100., -100., 100., 100., 100., 10., 10., 10.])
    dataset.scene_occ = torch.ones(1, 2, 3, 4, dtype=torch.bool)
    geometry.dataset.scene_grid_torch = dataset.scene_grid_torch
    geometry.dataset.get_nearest_free_voxel = lambda p, f: (torch.zeros(p.shape[:-1], dtype=torch.bool), p)
    context = dict(mat=geometry.mat, scene_flag=torch.zeros(1, dtype=torch.long),
                   obj_rot_mat_prefix=geometry.prefix, obj_rot_mat_ref=geometry.reference,
                   seq_name_dict={0: 'sub10_cube_0'}, obj_rest_verts={'cube': geometry.object_points[0]},
                   pelvis_goal=torch.zeros(1, 3), object_goal=torch.ones(1, 3), scene_goal=torch.zeros(1, 3),
                   is_object=torch.ones(1, dtype=torch.bool), is_loco=torch.zeros(1, dtype=torch.bool),
                   need_pelvis_dir=torch.ones(1, dtype=torch.bool))
    class PairedSampler(QuerySampler):
        def _hsi_model_arguments(self, current, previous, t, context):
            common = list(super()._hsi_model_arguments(current, previous, t, context))
            common[0] = torch.ones(1, 3, 2, 4)
            common[15] = torch.ones(4, 3, 2, 4)
            common[16] = torch.zeros(4, 1, 2)
            return tuple(common)
        def _hsi_predict_pair(self, z, common):
            self.pairs.append((z.clone(), common))
            return z + common[15][1:].mean() * .1, z
    sampler = PairedSampler()
    sampler.dataset, sampler.hsi_sampler = geometry.dataset, native
    p = torch.zeros(1, 16, geometry.dimension)
    reference = geometry.encode(geometry.decode(p), geometry.base[:, :2])
    objective = RelationalObjective(geometry, context['scene_flag'], geometry.decode(p)['human'][:, 2:])
    editor = SceneEvidenceEditor(enabled=True, mode='calibrate', lambda_dp=26., noise_levels=(250,),
        diagnostics=OmegaConf.create(dict(enabled=True, probe=probe, minimum_keep_ratio=.001,
            noise_draws=2, mismatch_local_x_m=2., physical_probe_rms_m=[.001])))
    returned = editor.edit(sampler, geometry.base, {}, None, context, geometry.offsets, 42)
    record = editor.records[0]
    result = record['view_diagnostic']
    assert result['ambient_rng_preserved'] and result['scene_storage_unchanged']
    assert record['hsi_teacher_calls'] == (14 if probe == 'relation_compatible' else 12)
    assert record['hoi_teacher_calls'] == (12 if probe == 'relation_compatible' else 6)
    assert all(x['hoi_reference_rms'] == [0.] for x in record['iterations'])
    assert all(x['base_prediction_equal'] for x in record['iterations'])
    assert torch.equal(returned, geometry.base)
    assert editor.motion_records[0]['local_bps'] is None
    assert json.loads(json.dumps(editor.audit_dict(), allow_nan=False))['diagnostics']['physical_probe_rms_m'] == [.001]
    views = 2 if probe == 'relation_compatible' else 3
    for start in (0, views):
        assert all(torch.equal(sampler.pairs[start][0], sampler.pairs[i][0]) for i in range(start, start+views))
    assert not torch.equal(sampler.pairs[0][0], sampler.pairs[views][0])
    assert not p.any()
