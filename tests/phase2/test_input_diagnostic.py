"""HSI input semantics, physical measurement, and passive-carrier isolation."""

import tempfile
import json
from pathlib import Path
import sys

import torch
import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / 'code'))

from mixer.diagnostics import (
    HSIInputDiagnostic, aggregate_episode, decode_human, empty_motion_view,
    masked_object_arguments, prediction_difference,
    write_analysis_inputs,
)
from models.infbagel import Unet
from tests.phase2.test_composed_sampler import (
    _evaluator_arguments, _make_composed_with_hsi,
)


def test_empty_view_matches_training_noise_and_keeps_the_world_intact():
    torch.manual_seed(42)
    current = torch.randn(2, 16, 232)
    original = current.clone()
    noise = torch.randn(2, 16, 16)
    actual = empty_motion_view(current, 0.2, noise)
    assert torch.equal(current, original)
    assert torch.equal(actual[..., :216], current[..., :216])
    assert torch.count_nonzero(actual[:, :2, 216:]) == 0
    assert torch.equal(actual[:, 2:, 216:], 0.2 * noise[:, 2:])
    assert torch.equal(empty_motion_view(current, 0.7, noise)[:, 2:, 216:], 0.7 * noise[:, 2:])


def test_actual_unet_masks_object_goal_and_bps_together():
    torch.manual_seed(42)
    model = Unet(
        dim_model=32, num_heads=4, num_layers=1, dropout_p=0.0,
        dim_input=232, dim_output=232, nb_voxels=[8, 8, 8],
        load_scene=False, load_language=False, load_scene_goal=True,
        load_pelvis_goal=True, load_object_goal=True, scene_type='occ_temp',
        temp_voxel_num=3, is_mix=False,
    ).eval()
    sampler, _, _, _ = _make_composed_with_hsi(0, batch=1)
    sampler.hsi_sampler.student_model = model
    arguments = _evaluator_arguments(batch=1)
    world = torch.randn(1, 16, 232)
    context = sampler._hsi_context(
        1, *[arguments[key] for key in (
            'mat', 'scene_flag', 'text_emb', 'pelvis_goal', 'scene_goal',
            'object_goal', 'need_scene', 'need_pelvis_dir', 'pi', 'end_pi',
            'seq_length', 'need_pi', 'is_loco', 'is_object', 'obj_bps_data',
            'object_points', 'obj_rot_mat_ref', 'obj_rest_verts',
        )], None, arguments['seq_name_dict'], None, False,
    )
    common = sampler._hsi_model_arguments(world, world, torch.tensor([100]), context)
    masked = masked_object_arguments(common)
    changed = list(masked)
    changed[12] = common[12] + 100
    changed[14] = common[14] + 100
    with torch.no_grad():
        masked_result = sampler._hsi_predict(world, masked)
        assert torch.equal(masked_result, sampler._hsi_predict(world, tuple(changed)))
        assert not torch.equal(masked_result, sampler._hsi_predict(world, common))
    for index, (before, after) in enumerate(zip(common, masked)):
        if index != 13:
            assert before is after
    assert arguments['is_object'].all()


def test_centimetres_use_half_range_and_fk_uses_root_plus_rotations():
    class Dataset:
        @staticmethod
        def denormalize_torch(value):
            return (value + 1) * 2 - 1  # 4-m range: normalized 0.01 means 2 cm.

    base = torch.zeros(1, 16, 232)
    base[..., 84:216] = torch.tensor([1., 0., 0., 0., 1., 0.]).repeat(22)
    moved = base.clone()
    moved[..., 0] += 0.01
    offsets = torch.zeros(24, 3)
    baseline = decode_human(base, Dataset(), offsets)
    result = prediction_difference(decode_human(moved, Dataset(), offsets), baseline)
    assert torch.isclose(result['raw_root_cm_mean'], torch.tensor(2.), atol=1e-4)
    assert torch.isclose(result['fk_all_cm_mean'], torch.tensor(2.), atol=1e-4)
    assert result['rotation_all_deg_mean'] == 0
    moved = base.clone()
    moved[:, :2, :84] += 10
    moved[:, 2:, 3:84] += 10
    result = prediction_difference(decode_human(moved, Dataset(), offsets), baseline)
    assert result['raw_all_cm_mean'] > 0
    assert result['fk_all_cm_mean'] == 0


def test_aggregation_keeps_generated_history_separate_from_initial_prefix():
    records = [
        {'window_index': window, 'step': step,
         'contrasts': {'both_legacy': {'fk_all_cm_mean': value}}}
        for window, step, value in [(0, 100, 99.), (1, 100, 2.), (2, 100, 4.), (1, 499, 50.)]
    ]
    aggregate = aggregate_episode(records)['both_legacy']
    assert aggregate['initial_late_fk_all_cm_mean'] == 99.
    assert aggregate['generated_late_fk_all_cm_mean'] == 3.
    assert aggregate['generated_t499_fk_all_cm_mean'] == 50.


def test_analysis_pairs_by_episode_and_averages_episodes_within_scene(tmp_path):
    from mixer.diagnostics import CONTRASTS
    source = tmp_path / 'source'
    source.mkdir()
    for index, value in enumerate((1., 3., 9.)):
        item = {
            'scene_name': 'a' if index < 2 else 'b', 'object_name': str(index),
            'test_idx': index, 'canonical_ordinal': index,
        }
        records = [{'window_index': 0, 'step': 100, 'contrasts': {
            contrast: {'fk_all_cm_mean': value} for contrast in CONTRASTS
        }}]
        (source / f'episode-{index}.json').write_text(json.dumps({
            'episode': item, 'window_count': 1, 'steps': [100],
            'records': records, 'metrics': aggregate_episode(records),
        }))
    result = write_analysis_inputs([source], tmp_path / 'analysis', expected_episodes=3)
    assert result['episodes'] == 3
    scenes = json.loads(Path(result['analysis_inputs']['both_legacy-scene']).read_text())
    assert scenes['metrics']['a']['initial_late_fk_all_cm_mean'] == 2.
    assert scenes['metrics']['b']['initial_late_fk_all_cm_mean'] == 9.
    with pytest.raises(ValueError, match='duplicate'):
        write_analysis_inputs([source, source], tmp_path / 'duplicate', expected_episodes=3)
    broken = json.loads((source / 'episode-0.json').read_text())
    broken['steps'] = [100, 1]
    (source / 'episode-0.json').write_text(json.dumps(broken))
    with pytest.raises(ValueError, match='incomplete'):
        write_analysis_inputs([source], tmp_path / 'incomplete', expected_episodes=3)


def test_passive_probe_preserves_both_carrier_windows_and_global_rng():
    reference, _, _, _ = _make_composed_with_hsi(0, batch=1)
    measured, _, _, _ = _make_composed_with_hsi(0, batch=1)
    measured.input_diagnostic = HSIInputDiagnostic(steps=(499, 1), seed=42)
    original_query = measured._hsi_model_arguments

    def occupancy_with_vertex_subsampling(*args):
        torch.randperm(127)  # The real dataset.vis occupancy branch consumes CPU RNG.
        return original_query(*args)

    measured._hsi_model_arguments = occupancy_with_vertex_subsampling
    arguments = _evaluator_arguments(batch=1)
    arguments['human_dict'] = {'rest_human_offsets': torch.zeros(24, 3)}
    with tempfile.TemporaryDirectory() as temporary:
        measured.input_diagnostic.begin_episode(
            {'canonical_ordinal': 3, 'scene_name': 'scene', 'object_name': 'object', 'test_idx': 0},
            temporary,
        )
        for _ in range(2):
            state_before = torch.get_rng_state().clone()
            expected, _ = reference.p_sample_loop(**arguments)
            state_after = torch.get_rng_state().clone()
            torch.set_rng_state(state_before)
            actual, _ = measured.p_sample_loop(**arguments)
            assert torch.equal(torch.get_rng_state(), state_after)
            assert all(torch.equal(a, b) for a, b in zip(actual, expected))
            arguments['fixed_points'] = expected[-1][:, -2:].clone()
        output = measured.input_diagnostic.finish_episode()
        assert output['windows'] == 2
        assert len(measured.input_diagnostic.records) == 4
        assert all(
            value == 0
            for record in measured.input_diagnostic.records
            for value in record['contrasts']['repeat_legacy'].values()
        )
