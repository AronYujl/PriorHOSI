"""Checkpoint and run-provenance contracts for continued HSI training."""

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'code'))
import train_infbagel as trainer
from utils import init_model, load_checkpoint_parameters, load_state_dict_eval


class GeometryModel(torch.nn.Module):
    def __init__(self, mode='body_geometry_warm_start', enabled=True):
        super().__init__()
        self.body_geometry_enabled = enabled
        self.checkpoint_load_mode = mode
        self.out = torch.nn.Linear(3, 2)
        self.body_geometry_refiner = torch.nn.Module()
        self.body_geometry_refiner.output = torch.nn.Linear(3, 2)
        torch.nn.init.zeros_(self.body_geometry_refiner.output.weight)
        torch.nn.init.zeros_(self.body_geometry_refiner.output.bias)


def backbone_checkpoint(model):
    return {name: torch.full_like(value, 0.75) for name, value in model.state_dict().items()
            if not name.startswith('body_geometry_refiner.')}


def test_warm_start_loads_every_backbone_parameter_and_preserves_zero_refiner():
    model = GeometryModel()
    checkpoint = backbone_checkpoint(model)
    load_checkpoint_parameters(model, checkpoint)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name] if name in checkpoint else torch.zeros_like(value))


@pytest.mark.parametrize('defect', ['missing_backbone', 'unexpected_backbone', 'partial_refiner', 'full_new_checkpoint'])
def test_warm_start_requires_exactly_the_registered_new_keys(defect):
    model = GeometryModel()
    checkpoint = backbone_checkpoint(model)
    if defect == 'missing_backbone':
        del checkpoint['out.weight']
    elif defect == 'unexpected_backbone':
        checkpoint['other_expert.weight'] = torch.zeros(1)
    elif defect == 'partial_refiner':
        checkpoint['body_geometry_refiner.output.bias'] = torch.zeros(2)
    else:
        checkpoint = model.state_dict()
    with pytest.raises(RuntimeError, match='exactly the new body_geometry_refiner keys'):
        load_checkpoint_parameters(model, checkpoint)


def test_strict_mode_requires_the_complete_new_checkpoint():
    model = GeometryModel(mode='strict')
    with pytest.raises(RuntimeError, match='Missing key'):
        load_checkpoint_parameters(model, backbone_checkpoint(model))
    checkpoint = {name: torch.full_like(value, 0.25) for name, value in model.state_dict().items()}
    load_checkpoint_parameters(model, checkpoint)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, checkpoint[name])


def test_legacy_model_keeps_its_original_partial_loading_contract(capsys):
    model = GeometryModel(mode='strict', enabled=False)
    load_checkpoint_parameters(model, backbone_checkpoint(model))
    assert 'Missing keys in checkpoint' in capsys.readouterr().out


def test_evaluation_loads_ddp_prefix_then_enforces_new_model_keys(tmp_path):
    model = GeometryModel(mode='strict')
    checkpoint = {'module.' + name: torch.full_like(value, 0.5)
                  for name, value in model.state_dict().items()}
    path = tmp_path / 'model.pth'
    torch.save(checkpoint, path)
    load_state_dict_eval(model, path, map_location='cpu', device='cpu')
    assert not model.training
    for value in model.state_dict().values():
        torch.testing.assert_close(value, torch.full_like(value, 0.5))


def test_training_warm_start_uses_the_same_loader_without_ddp(tmp_path):
    model = GeometryModel()
    checkpoint = backbone_checkpoint(model)
    path = tmp_path / 'r2.pth'
    torch.save(checkpoint, path)
    with patch('utils.hydra.utils.instantiate', return_value=model):
        loaded = init_model(SimpleNamespace(ckpt=path), 'cpu', eval=False,
                            load_state_dict=True, need_ddp=False)
    assert loaded is model and loaded.training
    torch.testing.assert_close(loaded.out.weight, checkpoint['out.weight'])


def git(repo, *args):
    return subprocess.check_output(['git', *args], cwd=repo, text=True).strip()


@pytest.fixture
def repository(tmp_path):
    git(tmp_path, 'init', '-q')
    (tmp_path / 'source.py').write_text('version = 1\n')
    git(tmp_path, 'add', 'source.py')
    git(tmp_path, '-c', 'user.name=HSI Test', '-c', 'user.email=hsi-test@example.invalid',
        'commit', '-qm', 'initial')
    return tmp_path


@pytest.mark.parametrize('tracked', [True, False])
def test_reportable_training_refuses_tracked_and_untracked_changes(repository, tracked):
    (repository / ('source.py' if tracked else 'new.py')).write_text('changed = True\n')
    with pytest.raises(RuntimeError, match='clean worktree'):
        trainer.capture_training_start({'run_id': 'registered-training'}, repository)
    assert trainer.capture_training_start({'run_id': None}, repository)['git_commit'] == git(repository, 'rev-parse', 'HEAD')


def test_completion_preserves_the_started_commit_after_source_advances(repository):
    started = trainer.capture_training_start({'run_id': 'registered-training'}, repository)
    (repository / 'source.py').write_text('version = 2\n')
    git(repository, 'add', 'source.py')
    git(repository, '-c', 'user.name=HSI Test', '-c', 'user.email=hsi-test@example.invalid',
        'commit', '-qm', 'advance')
    completed = trainer.completed_training_provenance(started, repository)
    assert completed['git_commit'] == started['git_commit']
    assert completed['completion_head'] == git(repository, 'rev-parse', 'HEAD')
    assert completed['completion_head'] != completed['git_commit']


def test_dirty_preflight_stops_before_cuda_workers_are_spawned():
    with patch.object(trainer, 'require_clean_worktree', side_effect=RuntimeError('clean worktree')), \
            patch.object(torch.multiprocessing, 'spawn') as spawn, \
            patch.object(torch.cuda, 'device_count') as device_count:
        with pytest.raises(RuntimeError, match='clean worktree'):
            trainer.train(OmegaConf.create({'run_id': 'registered-training'}))
    spawn.assert_not_called()
    device_count.assert_not_called()


def test_training_cli_still_exposes_hydra_config_without_starting_workers():
    result = subprocess.run(
        [sys.executable, str(Path(trainer.__file__).resolve()), '--cfg', 'job'],
        text=True, capture_output=True, check=True,
    )
    assert 'sample_type: diffusion' in result.stdout


@pytest.mark.parametrize('field,value', [
    ('body_geometry_enabled', True), ('body_geometry_loss_weight', 0.5),
    ('body_geometry_permutation', 'left_right'),
    ('body_geometry_encoder', 'legacy_flatten'),
])
def test_resume_refuses_changed_body_geometry_contract(field, value):
    geometry = {key: None for key in trainer.RESUME_GEOMETRY_FIELDS}
    state = {'schema_version': trainer.RESUME_STATE_VERSION, 'epoch_completed': True,
             'geometry': dict(geometry)}
    geometry[field] = value
    with pytest.raises(ValueError, match=field):
        trainer.check_resume_compatibility(state, geometry)
