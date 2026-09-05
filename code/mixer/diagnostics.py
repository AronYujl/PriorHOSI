"""Paired HSI input probes on a passive, generated HOI carrier."""

import json
from collections import defaultdict
from pathlib import Path

import torch
from pytorch3d import transforms

from .body_groups import POSITION_GROUPS, ROTATION_GROUPS
from .input_views import empty_motion_view, masked_object_arguments
from .kinematic_composition import (
    _expand_rest_offsets, _forward_kinematics, _local_from_global,
)


CONTRASTS = {
    'tokens_legacy': ('tokens', 'legacy'),
    'motion_legacy': ('motion', 'legacy'),
    'both_legacy': ('both', 'legacy'),
    'both_tokens': ('both', 'tokens'),
    'both_motion': ('both', 'motion'),
    'repeat_legacy': ('repeat', 'legacy'),
}
FK_GROUPS = {**ROTATION_GROUPS, 'hands': (22, 23)}


def decode_human(value, dataset, rest_offsets):
    batch, frames = value.shape[:2]
    positions = dataset.denormalize_torch(value[..., :84]).reshape(
        batch, frames, 28, 3,
    )
    rotations = transforms.rotation_6d_to_matrix(
        value[..., 84:216].reshape(batch, frames, 22, 6),
    )
    offsets = _expand_rest_offsets(rest_offsets, batch, frames, positions)
    offsets[..., 0, :] = positions[..., 0, :]
    _, fk = _forward_kinematics(_local_from_global(rotations), offsets)
    return positions[:, 2:], rotations[:, 2:], fk[:, 2:]


def prediction_difference(first, second):
    """Physical changes of future predictions, including evaluator-facing FK."""
    positions_a, rotations_a, fk_a = first
    positions_b, rotations_b, fk_b = second
    raw_cm = (positions_a - positions_b).norm(dim=-1) * 100
    fk_cm = (fk_a - fk_b).norm(dim=-1) * 100
    # ||Ra-Rb||_F = 2 sqrt(2) sin(theta/2); clamp only the trig domain.
    chord = (rotations_a - rotations_b).square().sum(dim=(-1, -2)).sqrt()
    angle = 2 * torch.asin((chord / (2 * 2**0.5)).clamp(0, 1)) * (180 / torch.pi)
    result = {}
    for prefix, values, groups in (
        ('raw', raw_cm, POSITION_GROUPS), ('fk', fk_cm, FK_GROUPS),
        ('rotation', angle, ROTATION_GROUPS),
    ):
        unit = 'deg' if prefix == 'rotation' else 'cm'
        result[f'{prefix}_all_{unit}_mean'] = values.mean()
        result[f'{prefix}_all_{unit}_max'] = values.max()
        for group, joints in groups.items():
            if joints:
                result[f'{prefix}_{group}_{unit}_mean'] = values[..., joints].mean()
    return result


def aggregate_episode(records, contrast_names=CONTRASTS):
    """Average within episode, keeping prefix type and noise levels explicit."""
    collected = defaultdict(list)
    for record in records:
        history = 'initial' if record['window_index'] == 0 else 'generated'
        strata = [f'{history}_t{record["step"]}']
        if record['step'] in (100, 10, 1, 0):
            strata.append(f'{history}_late')
        for contrast, metrics in record['contrasts'].items():
            for metric, value in metrics.items():
                for stratum in strata:
                    collected[(contrast, f'{stratum}_{metric}')].append(value)
    result = {contrast: {} for contrast in contrast_names}
    for (contrast, metric), values in collected.items():
        result[contrast][metric] = sum(values) / len(values)
    return result


def write_analysis_inputs(source_dirs, output_dir, expected_episodes=28,
                          contrast_names=CONTRASTS):
    """Export paired episode/scene inputs for the existing bootstrap command."""
    episodes = {}
    for directory in source_dirs:
        for path in sorted(Path(directory).glob('episode-*.json')):
            payload = json.loads(path.read_text())
            item = payload['episode']
            key = f'{item["scene_name"]}/{item["object_name"]}/{item["test_idx"]}'
            if key in episodes:
                raise ValueError(f'duplicate diagnostic episode: {key}')
            expected = {
                (window, step) for window in range(payload['window_count'])
                for step in payload['steps']
            }
            actual = {(row['window_index'], row['step']) for row in payload['records']}
            if actual != expected or len(payload['records']) != len(expected):
                raise ValueError(f'incomplete diagnostic states: {key}')
            episodes[key] = payload
    if len(episodes) != expected_episodes:
        raise ValueError(f'expected {expected_episodes} diagnostic episodes, got {len(episodes)}')
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)
    exported = {}
    for contrast in contrast_names:
        episode_metrics = {
            name: value['metrics'][contrast] for name, value in episodes.items()
        }
        grouped = defaultdict(lambda: defaultdict(list))
        for name, metrics in episode_metrics.items():
            for metric, value in metrics.items():
                grouped[episodes[name]['episode']['scene_name']][metric].append(value)
        scene_metrics = {
            scene: {metric: sum(values) / len(values) for metric, values in metrics.items()}
            for scene, metrics in grouped.items()
        }
        for unit, values in (('episode', episode_metrics), ('scene', scene_metrics)):
            path = output_dir / f'{contrast}-{unit}.json'
            with path.open('x') as handle:
                json.dump({
                    'schema_version': 1, 'seed': 42, 'unit': unit,
                    'sequence_count': len(values), 'metrics': values,
                }, handle, indent=2, allow_nan=False)
            exported[f'{contrast}-{unit}'] = str(path)
    return {
        'episodes': len(episodes),
        'scenes': sorted({p['episode']['scene_name'] for p in episodes.values()}),
        'windows': sum(p['window_count'] for p in episodes.values()),
        'states': sum(len(p['records']) for p in episodes.values()),
        'analysis_inputs': exported,
    }


class HSIInputDiagnostic:
    """Named probe: object_input_semantics. Predictions never feed the carrier."""

    probe_name = 'object_input_semantics'
    contrast_names = CONTRASTS

    def __init__(self, steps=(499, 400, 250, 100, 10, 1, 0), seed=42):
        self.steps = tuple(steps)
        self.seed = int(seed)

    def begin_episode(self, episode, output_dir):
        self.episode = dict(episode)
        self.output_dir = Path(output_dir)
        self.records = []
        self.window_index = -1

    def begin_window(self, current):
        self.window_index += 1
        generator = torch.Generator(device=current.device)
        generator.manual_seed(
            self.seed + self.episode['canonical_ordinal'] * 1000003 + self.window_index,
        )
        self.noise = torch.randn(
            (*current.shape[:2], 16), generator=generator,
            dtype=current.dtype, device=current.device,
        )

    @torch.no_grad()
    def observe(self, sampler, current, previous_x0, timesteps, context,
                rest_offsets, hoi_clean):
        step = int(timesteps[0])
        if step not in self.steps:
            return
        # Real occupancy reconstruction subsamples object vertices with CPU
        # randperm. Keep that query and all observer forwards out of the
        # carrier's CPU/CUDA RNG streams.
        devices = [current.device.index] if current.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            self._observe(sampler, current, previous_x0, timesteps, context,
                          rest_offsets, hoi_clean, step)

    def _observe(self, sampler, current, previous_x0, timesteps, context,
                 rest_offsets, hoi_clean, step):
        common = sampler._hsi_model_arguments(current, previous_x0, timesteps, context)
        masked = masked_object_arguments(common)
        sigma = sampler.inner_hoi.diffusion.sqrt_one_minus_alpha_bar[step]
        empty = empty_motion_view(current, sigma, self.noise)
        predictions = {
            'legacy': sampler._hsi_predict(current, common),
            'tokens': sampler._hsi_predict(current, masked),
            'motion': sampler._hsi_predict(empty, common),
            'both': sampler._hsi_predict(empty, masked),
            'repeat': sampler._hsi_predict(current, common),
        }
        decoded = {
            name: decode_human(value, sampler.dataset, rest_offsets)
            for name, value in predictions.items()
        }
        contrasts = {}
        for name, (first, second) in CONTRASTS.items():
            metrics = prediction_difference(decoded[first], decoded[second])
            metrics['human_max_abs_normalized'] = (
                predictions[first][:, 2:, :216] - predictions[second][:, 2:, :216]
            ).abs().max()
            contrasts[name] = metrics
        # One device-to-host transfer for the scalar metrics of this state.
        keys = [(contrast, metric) for contrast, values in contrasts.items() for metric in values]
        scalars = torch.stack([contrasts[c][m] for c, m in keys]).cpu().tolist()
        for (contrast, metric), value in zip(keys, scalars):
            contrasts[contrast][metric] = value
        self.records.append({
            'window_index': self.window_index, 'step': step, 'contrasts': contrasts,
        })
        snapshot = {
            'current': current.cpu(), 'previous_x0': previous_x0.cpu(),
            'hoi_clean': hoi_clean.cpu(), 'empty_motion_noise': self.noise.cpu(),
            'predictions': {name: value.cpu() for name, value in predictions.items()},
            'mat': context['mat'].cpu(), 'rest_offsets': rest_offsets.cpu(),
        }
        torch.save(snapshot, self.output_dir / (
            f'state-{self.episode["canonical_ordinal"]:03d}'
            f'-w{self.window_index:03d}-t{step:03d}.pt'
        ))

    def finish_episode(self):
        payload = {
            'schema_version': 1, 'probe': self.probe_name, 'seed': self.seed,
            'episode': self.episode, 'window_count': self.window_index + 1,
            'steps': list(self.steps), 'records': self.records,
            'metrics': aggregate_episode(self.records, self.contrast_names),
        }
        path = self.output_dir / f'episode-{self.episode["canonical_ordinal"]:03d}.json'
        with path.open('x') as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
        return {'path': str(path), **self.episode, 'windows': self.window_index + 1}
