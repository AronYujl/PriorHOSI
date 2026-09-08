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


def factor_body_prediction(source, full):
    """Factor native full-body geometry into planar motion and its remainder."""
    from .hsi_motion_target import planar_yaw
    from .surface_edit import yaw_matrix, shared_object_transform
    pivot = source['joints'][:, 0]
    shift = full['joints'][:, 0]-pivot
    shift = shift.clone(); shift[:, 1] = 0
    original_rotation = transforms.axis_angle_to_matrix(source['pose'][:, 0])
    predicted_rotation = transforms.axis_angle_to_matrix(full['pose'][:, 0])
    yaw = planar_yaw(predicted_rotation@original_rotation.transpose(-1, -2))
    rotation = yaw_matrix(yaw)
    obj, obj_rot = shared_object_transform(pivot, source['object_translation'],
                                         source['object_rotation'], shift, yaw)
    planar, residual = {}, {}
    for key in ('joints', 'verts'):
        planar[key] = pivot[:, None]+(rotation[:, None]@
            (source[key]-pivot[:, None])[..., None]).squeeze(-1)+shift[:, None]
        residual[key] = pivot[:, None]+(rotation[:, None].transpose(-1, -2)@
            (full[key]-pivot[:, None]-shift[:, None])[..., None]).squeeze(-1)
    planar.update(object_translation=obj, object_rotation=obj_rot)
    residual.update(object_translation=source['object_translation'], object_rotation=source['object_rotation'])
    combined = dict(full, object_translation=obj, object_rotation=obj_rot)
    rebuilt = pivot[:, None]+(rotation[:, None]@
        (residual['verts']-pivot[:, None])[..., None]).squeeze(-1)+shift[:, None]
    return dict(planar=planar, residual=residual, full=combined), float((rebuilt-full['verts']).abs().max())


def body_readout_measures(track, source, floor, coarse_length):
    """Support uses the source floor, so lifting the body cannot redefine it."""
    from .surface_edit import FEET, object_frame_hands
    feet, reference = track['joints'][:, FEET], source['joints'][:, FEET]
    limits = feet.new_tensor((.08, .08, .04, .04))
    stance = reference[..., 1]-floor < limits
    support = feet[..., 1]-floor < limits
    displacement = (feet-reference).norm(dim=-1)
    relative = object_frame_hands(track['joints'], track['object_translation'], track['object_rotation'])
    original = object_frame_hands(source['joints'], source['object_translation'], source['object_rotation'])
    seam = torch.arange(16, coarse_length, 14, device=feet.device)*3
    speed = (track['joints'][1:]-track['joints'][:-1]).norm(dim=-1)*30
    root_y = track['joints'][:, 0, 1]-source['joints'][:, 0, 1]
    return dict(source_floor_support_fraction=float(support.float().mean()),
        source_stance_foot_displacement_cm=float((displacement*stance).sum()/stance.sum()*100),
        root_y_change_cm=float(root_y.mean()*100), root_y_abs_change_cm=float(root_y.abs().mean()*100),
        hand_object_mean_drift_cm=float((relative-original).norm(dim=-1).mean()*100),
        hand_object_max_drift_cm=float((relative-original).norm(dim=-1).max()*100),
        seam_joint_speed_cm_s=float(speed[seam-1].mean()*100),
        mean_joint_speed_cm_s=float(speed.mean()*100))


@torch.no_grad()
def run_body_readout(cfg):
    """Named cached probe: full_body_readout, with the native evaluator."""
    import subprocess
    import time
    import numpy as np
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from test_infbagel_hosi import decode_sample_window, seed_everything
    from utils import interpolate_joints
    from .continuation_outcomes import native_tracks, write_json
    from .surface_edit import load_object_sdf, native_metrics
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('reportable cached diagnostic requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    protocol_path = Path(cfg.hsi_body_readout.protocol)
    root = protocol_path.resolve().parents[2]
    protocol = json.loads(protocol_path.read_text())
    tasks = json.loads((root/protocol['task_manifest']).read_text())['tasks']
    if cfg.hsi_body_readout.task_ids is not None:
        tasks = [t for t in tasks if t['canonical_ordinal'] in cfg.hsi_body_readout.task_ids]
    out = Path(cfg.hosi_output_dir); out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), out/'resolved.yaml')
    seed_everything(42)
    device = torch.device(cfg.device)
    smpl_cache = {}; current_scene = None; records = []
    torch.cuda.synchronize(device); started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for item in tasks:
        ordinal, scene = item['canonical_ordinal'], item['scene_name']
        if scene != current_scene:
            dc = OmegaConf.merge(cfg.dataset, dict(device=str(device), vis=True,
                                 load_object_payload=False, test_scene_name=scene))
            dataset = InfBaGelDataset(**dc)
            dataset.obj_rest_verts = {k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
            native = json.loads((root/'data/hosi_test/data'/(scene+'.json')).read_text())
            key = scene+'_sdf'; sdf_root = root/'data/hosi_test/Scene_sdf'
            sdf = np.load(sdf_root/(key+'.npy'))
            info = json.loads((sdf_root/(key+'_info.json')).read_text())
            current_scene = scene
        task = dict(native[item['test_idx']], test_idx=item['test_idx'])
        path, = (root/protocol['source_run']).glob(f"{protocol['source_arm']}-shard*/episode-motion-{ordinal:03d}.pt")
        saved = torch.load(path, map_location='cpu', weights_only=False)
        world = {k:torch.as_tensor(v).reshape(-1,28,3) if k=='points_world' else torch.as_tensor(v)
                 for k,v in saved['stitched'].items()}
        source = native_tracks(cfg, dataset, world, task, True, smpl_cache, body_parameters=True)
        obj = dataset.obj_rest_verts[item['object_name']]
        obj_sdf, obj_info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', item['object_name'])
        model = smpl_cache[source['gender']]
        def evaluate(track):
            return native_metrics(track, task, obj, obj_sdf, obj_info, sdf, info, model.faces, 42)
        baseline = evaluate(source)
        previous = json.loads(path.with_name(f'episode-audit-{ordinal:03d}.json').read_text())['metrics']
        joint_error = float((source['joints'].cpu()-saved['evaluated_joints_world']).abs().max())
        metric_error = max(abs(float(baseline[k])-float(previous[k])) for k in baseline)
        assert joint_error <= 1e-5 and metric_error <= 1e-5, (joint_error, metric_error)
        teacher_path, = (root/protocol['teacher_run']).glob(f'lanes/*/task-{ordinal:03d}/teacher.pt')
        teacher = torch.load(teacher_path, map_location='cpu', weights_only=False)
        length = len(world['points_world'])
        floor = baseline['feet_height']/100
        baseline.update(body_readout_measures(source, source, floor, length))
        def raw_fk_error(points, track):
            raw = interpolate_joints(points.reshape(-1,84).to(device), cfg.interp_s).reshape(-1,28,3)
            raw = raw-raw[:, :1]
            fk = track['joints']-track['joints'][:, :1]
            return float((raw-fk).norm(dim=-1).mean()*100)
        baseline['raw_fk_relative_error_cm'] = raw_fk_error(world['points_world'], source)
        task_rows = []; full_tracks = {}; factor_errors = []
        motions = dict(source={k:v.cpu() for k,v in source.items() if torch.is_tensor(v) and k!='verts'})
        for draw in range(2):
            arms = dict(source=dict(baseline))
            for scene_view in ('correct', 'wrong'):
                predicted_world = {k:v.clone() for k,v in world.items()}
                source_decode_error = 0.
                for window in teacher['windows']:
                    idx = window['window']; dest = slice(14*idx+2,14*idx+16)
                    decoded = decode_sample_window(cfg, window['predictions'][scene_view][draw].to(device),
                                                   dataset, window['mat'].to(device))
                    original = decode_sample_window(cfg, window['source'].to(device), dataset, window['mat'].to(device))
                    for key, decoded_key, shape in (('points_world','points_orig',(-1,28,3)),
                                                   ('global_rot_6d','global_rot_6d',(-1,22,6))):
                        value = decoded[decoded_key].cpu().reshape(shape)[2:]
                        reference = original[decoded_key].cpu().reshape(shape)[2:]
                        target_shape = predicted_world[key][dest].shape
                        source_decode_error = max(source_decode_error,
                            float((reference.reshape(target_shape)-world[key][dest]).abs().max()))
                        predicted_world[key][dest] = value.reshape(target_shape)
                assert source_decode_error < 1e-5, source_decode_error
                full = native_tracks(cfg, dataset, predicted_world, task, True, smpl_cache, body_parameters=True)
                factors, error = factor_body_prediction(source, full)
                assert error < 1e-5, error
                factor_errors.append(error)
                raw_error = raw_fk_error(predicted_world['points_world'], full)
                for factor, track in factors.items():
                    arm = scene_view+'_'+factor
                    metrics = evaluate(track)
                    metrics.update(body_readout_measures(track, source, floor, length))
                    # Position-channel agreement belongs to the original full prediction.
                    if factor == 'full': metrics['raw_fk_relative_error_cm'] = raw_error
                    arms[arm] = metrics
                    motions[f'{arm}_draw{draw}'] = {k:v.cpu() for k,v in track.items()
                        if torch.is_tensor(v) and k!='verts'}
                assert arms[scene_view+'_planar']['hand_object_max_drift_cm'] < .001
                full_tracks[scene_view, draw] = full['joints'].clone()
            task_rows.append(dict(draw=draw, arms=arms, source_decode_error=source_decode_error))
        response = {}
        groups = dict(all=tuple(range(28)), legs=(1,2,4,5,7,8,10,11), torso=(3,6,9,12,15), arms=(16,17,18,19,20,21), hands=(24,26))
        for group, indices in groups.items():
            response[group] = dict(scene_difference_cm=float(torch.stack([
                (full_tracks['correct', d][:,indices]-full_tracks['wrong', d][:,indices]).norm(dim=-1).mean() for d in range(2)]).mean()*100),
                draw_difference_cm=float((full_tracks['correct',0][:,indices]-full_tracks['correct',1][:,indices]).norm(dim=-1).mean()*100))
        means = {arm:{k:sum(float(r['arms'][arm][k]) for r in task_rows)/2
                      for k in task_rows[0]['arms'][arm]} for arm in protocol['arms']}
        record = dict(task=ordinal, scene=scene, object=item['object_name'], windows=len(teacher['windows']),
            source_joint_error_m=joint_error, source_metric_error=metric_error,
            factorization_error_m=max(factor_errors), draws=task_rows, means=means, response=response)
        dest = out/f'task-{ordinal:03d}'; dest.mkdir()
        write_json(dest/'metrics.json',record)
        with (dest/'motion.pt').open('xb') as handle: torch.save(motions, handle)
        records.append(record)
        print(json.dumps(dict(task=ordinal, completed=True, source_error=metric_error,
            full_hs=means['correct_full']['scene_human_penetration_s_mean'])), flush=True)
    torch.cuda.synchronize(device)
    write_json(out/'metrics.json',dict(commit=commit,
        git_commit_at_completion=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tasks=len(records), windows=sum(r['windows'] for r in records), seconds=time.perf_counter()-started,
        peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3, expert_forwards=0))


def summarize_body_readout(run_root, task_manifest, device='cuda:7'):
    from .continuation_outcomes import write_json
    from .scene_calibration import paired_local_metrics
    run_root = Path(run_root)
    tasks = json.loads(Path(task_manifest).read_text())['tasks']
    records = {}
    for path in run_root.glob('lanes/*/task-*/metrics.json'):
        row = json.loads(path.read_text()); records[row['task']] = row
    assert set(records) == {t['canonical_ordinal'] for t in tasks}
    arms = list(next(iter(records.values()))['means'])
    by_task = {arm:{str(k):r['means'][arm] for k,r in records.items()} for arm in arms}
    scenes = sorted({r['scene'] for r in records.values()})
    by_scene = {arm:{scene:{metric:sum(r['means'][arm][metric] for r in records.values() if r['scene']==scene)/
        sum(r['scene']==scene for r in records.values()) for metric in next(iter(by_task[arm].values()))}
        for scene in scenes} for arm in arms}
    means = {arm:{metric:sum(v[metric] for v in by_task[arm].values())/len(records)
                  for metric in next(iter(by_task[arm].values()))} for arm in arms}
    pairs = [(arm,'source') for arm in arms if arm!='source']
    pairs += [(view+'_full',view+'_'+factor) for view in ('correct','wrong') for factor in ('planar','residual')]
    pairs += [('correct_'+factor,'wrong_'+factor) for factor in ('planar','residual','full')]
    contrasts = {a+'__minus__'+b:{unit:paired_local_metrics(data[b],data[a],device)
                 for unit,data in (('task',by_task),('scene',by_scene))} for a,b in pairs}
    source, full, planar, wrong = (means[k] for k in ('source','correct_full','correct_planar','wrong_full'))
    hs = 'scene_human_penetration_s_mean'; os = 'scene_obj_penetration_s_mean'
    geometry = dict(full_improves_source=full[hs]<=.99*source[hs],
                    full_improves_planar=full[hs]<=.99*planar[hs],
                    correct_scene_benefit=full[hs]<=wrong[hs]-.005*source[hs])
    protection = dict(contact=full['contact_percent']>=source['contact_percent']-.002,
        foot_sliding=full['foot_sliding']<=source['foot_sliding']+.01,
        feet_height=abs(full['feet_height']-source['feet_height'])<=1,
        source_floor_support=full['source_floor_support_fraction']>=source['source_floor_support_fraction']-.02,
        object_scene=full[os]<=1.01*source[os], completion=full['completed']>=source['completed'])
    summary = dict(tasks=len(records),windows=sum(r['windows'] for r in records.values()),draws=2,
        means=means,geometry_conditions=geometry,protection_conditions=protection,
        geometry_signal=all(geometry.values()),body_transfer_entry=all(geometry.values()) and all(protection.values()),
        audit=dict(source_joint_max_error_m=max(r['source_joint_error_m'] for r in records.values()),
                   source_metric_max_error=max(r['source_metric_error'] for r in records.values()),
                   factorization_max_error_m=max(r['factorization_error_m'] for r in records.values())),
        contrasts=contrasts, expert_forwards=0, training_allowed=False,test_set_development=True)
    out=run_root/'analysis';out.mkdir()
    for arm in arms:
        for unit,values in (('task',by_task[arm]),('scene',by_scene[arm])):
            write_json(out/f'{arm}-{unit}.json',dict(metrics=values))
    write_json(out/'summary.json',summary)
    write_json(out/'records.json',records)
    return summary


@torch.no_grad()
def run_body_projection(cfg):
    """Named cached probe: hand_foot_continuous_body_readout."""
    import subprocess
    import time
    import numpy as np
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from test_infbagel_hosi import seed_everything
    from .continuation_outcomes import native_tracks, write_json
    from .surface_edit import load_object_sdf, native_metrics, decode_body
    from .body_projection import NativeBodyProjection, native_rest_offsets, smooth_body_target, NATIVE_ANCHORS
    if cfg.get('run_id') and subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('reportable cached projection requires a clean worktree')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    protocol_path = Path(cfg.hsi_body_projection.protocol)
    root = protocol_path.resolve().parents[2]
    protocol = json.loads(protocol_path.read_text())
    tasks = json.loads((root/protocol['task_manifest']).read_text())['tasks']
    if cfg.hsi_body_projection.task_ids is not None:
        tasks = [t for t in tasks if t['canonical_ordinal'] in cfg.hsi_body_projection.task_ids]
    out = Path(cfg.hosi_output_dir); out.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'resolved.yaml')
    seed_everything(42)
    device = torch.device(cfg.device)
    smpl_cache = {}; current_scene = None; records = []
    geometry_root = cfg.hsi_body_projection.get('geometry_root')
    direction_probe = cfg.hsi_body_projection.get('direction_probe',False)
    teacher = None
    if geometry_root is not None and not direction_probe:
        from .hsi_motion_target import CurrentMotionTeacher
        teacher = CurrentMotionTeacher(cfg,protocol)
    torch.cuda.synchronize(device);started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    for item in tasks:
        ordinal,scene = item['canonical_ordinal'],item['scene_name']
        if scene!=current_scene:
            dc = OmegaConf.merge(cfg.dataset,dict(device=str(device),vis=True,load_object_payload=False,test_scene_name=scene))
            dataset = InfBaGelDataset(**dc)
            dataset.obj_rest_verts = {k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
            if teacher is not None:
                teacher.set_dataset(dataset)
            native = json.loads((root/'data/hosi_test/data'/(scene+'.json')).read_text())
            key=scene+'_sdf';sdf_root=root/'data/hosi_test/Scene_sdf'
            sdf=np.load(sdf_root/(key+'.npy'));info=json.loads((sdf_root/(key+'_info.json')).read_text())
            current_scene=scene
        task=dict(native[item['test_idx']],test_idx=item['test_idx'])
        path,=(root/protocol['source_run']).glob(f"{protocol['source_arm']}-shard*/episode-motion-{ordinal:03d}.pt")
        saved=torch.load(path,map_location='cpu',weights_only=False)
        world={k:torch.as_tensor(v).reshape(-1,28,3) if k=='points_world' else torch.as_tensor(v)
               for k,v in saved['stitched'].items()}
        source=native_tracks(cfg,dataset,world,task,True,smpl_cache,body_parameters=True)
        model=smpl_cache[source['gender']]
        raw_source=source
        geometry_payload=None
        if geometry_root is not None:
            geometry_path,=Path(geometry_root).glob(f'lanes/*/task-{ordinal:03d}/{cfg.hsi_body_projection.geometry_arm}.pt')
            geometry_payload=torch.load(geometry_path,map_location=device,weights_only=False)
            if geometry_payload['solver']['best_iteration']!=0:
                source=dict(geometry_payload['motion'],betas=raw_source['betas'],gender=raw_source['gender'])
                source['verts'],decoded_joints=decode_body(source,model)
                from .surface_edit import edit_envelope
                fixed=edit_envelope(len(source['pose']),device)==0
                source['verts'][fixed]=raw_source['verts'][fixed]
                assert (decoded_joints-source['joints']).abs().max()<=1e-5
        obj=dataset.obj_rest_verts[item['object_name']]
        obj_sdf,obj_info=load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files',item['object_name'])
        def evaluate(track):
            return native_metrics(track,task,obj,obj_sdf,obj_info,sdf,info,model.faces,42)
        baseline=evaluate(source)
        previous=(json.loads(path.with_name(f'episode-audit-{ordinal:03d}.json').read_text())['metrics']
                  if geometry_payload is None else geometry_payload['metrics'])
        reference_joints=(saved['evaluated_joints_world'].to(device) if geometry_payload is None
                          else geometry_payload['motion']['joints'])
        joint_error=float((source['joints']-reference_joints).abs().max())
        metric_error=max(abs(float(baseline[k])-float(previous[k])) for k in baseline)
        assert joint_error<=1e-5 and metric_error<=1e-5,(joint_error,metric_error)
        projector=NativeBodyProjection(source,native_rest_offsets(model,source['betas']))
        fk_error=float((projector.reference-source['joints'][:,NATIVE_ANCHORS]).norm(dim=-1).max())
        assert fk_error<=1e-5,fk_error
        floor=baseline['feet_height']/100; length=len(world['points_world'])
        baseline.update(body_readout_measures(source,source,floor,length))
        baseline.update(projector.measures(projector.rotation,projector.translation))
        baseline['native_anchor_max_error_m']=0.
        baseline['initial_final_max_error_m']=0.
        baseline['object_max_error_m']=0.
        dest=out/f'task-{ordinal:03d}';dest.mkdir()
        if direction_probe:
            cache_path,=(root/protocol['body_cache']).glob(f'lanes/*/task-{ordinal:03d}/motion.pt')
            cache=torch.load(cache_path,map_location=device,weights_only=False)
            assert torch.equal(cache['source']['joints'],source['joints'])
            record=feasible_body_direction_probe(projector,model,sdf,info,evaluate,baseline,cache,
                                                 floor,length,protocol,dest)
            record.update(task=ordinal,scene=scene,object=item['object_name'],windows=len(saved['windows']),
                source_joint_error_m=joint_error,source_metric_error=metric_error,
                source_fk_anchor_error_m=fk_error,hsi_calls=0)
            write_json(dest/'metrics.json',record)
            records.append(record)
            print(json.dumps(dict(task=ordinal,completed=True,seconds=record['seconds'],
                gradient_norm=record['gradient_audit']['norm'],hs={a:m['scene_human_penetration_s_mean']
                    for a,m in record['means'].items()})),flush=True)
            continue
        query_audit=None
        if teacher is None:
            body_path,=(root/protocol['body_cache']).glob(f'lanes/*/task-{ordinal:03d}/motion.pt')
            cache=torch.load(body_path,map_location=device,weights_only=False)
        else:
            cache,query_audit=teacher.query_current_task(saved,task,ordinal,dest,source,projector,smpl_cache)
        assert torch.equal(cache['source']['joints'],source['joints'])
        rows=[];solves=[]
        motions=dict(source={k:v.cpu() for k,v in source.items() if torch.is_tensor(v) and k!='verts'})
        def decode(rotation,translation):
            motion=dict(source,pose=transforms.matrix_to_axis_angle(rotation.double()).to(source['pose'].dtype),
                        translation=translation.clone())
            motion['pose'][projector.fixed]=source['pose'][projector.fixed]
            motion['translation'][projector.fixed]=source['translation'][projector.fixed]
            motion['verts'],motion['joints']=decode_body(motion,model)
            motion['verts'][projector.fixed]=source['verts'][projector.fixed]
            motion['joints'][projector.fixed]=source['joints'][projector.fixed]
            return motion
        for draw in range(2):
            arms=dict(source=dict(baseline))
            for view in ('correct','wrong'):
                target_rotation,target_translation=smooth_body_target(source,cache[f'{view}_full_draw{draw}'])
                smooth=decode(target_rotation,target_translation)
                rotation,translation,solve=projector.solve(target_rotation,target_translation)
                candidate_states=solve.pop('states')
                projected=source if solve['amplitude']==0 else decode(rotation,translation)
                for factor,track,rot,trans in [('smooth',smooth,target_rotation,target_translation),
                                              ('projected',projected,rotation,translation)]:
                    arm=view+'_'+factor
                    metrics=evaluate(track)
                    metrics.update(body_readout_measures(track,source,floor,length))
                    metrics.update(projector.measures(rot,trans))
                    metrics['native_anchor_max_error_m']=float((track['joints'][:,NATIVE_ANCHORS]-source['joints'][:,NATIVE_ANCHORS]).norm(dim=-1).max())
                    metrics['initial_final_max_error_m']=float((track['joints'][projector.fixed]-source['joints'][projector.fixed]).abs().max())
                    metrics['object_max_error_m']=float((track['object_translation']-source['object_translation']).abs().max())
                    if factor=='projected':
                        assert metrics['native_anchor_max_error_m']<=1e-5,metrics['native_anchor_max_error_m']
                        assert metrics['initial_final_max_error_m']==0 and metrics['object_max_error_m']==0
                    arms[arm]=metrics
                    motions[f'{arm}_draw{draw}']={k:v.cpu() for k,v in track.items() if torch.is_tensor(v) and k!='verts'}
                target_change=(smooth['joints']-source['joints']).norm(dim=-1).mean()
                actual_change=(projected['joints']-source['joints']).norm(dim=-1).mean()
                solve.update(draw=draw,view=view,target_body_displacement_cm=float(target_change*100),
                             retained_target_fraction=float(actual_change/target_change))
                solves.append(solve)
                motions[f'{view}_attempts_draw{draw}']=candidate_states
                print(json.dumps(dict(task=ordinal,draw=draw,view=view,amplitude=solve['amplitude'],
                    anchor_error=arms[view+'_projected']['native_anchor_max_error_m'],seconds=solve['seconds'])),flush=True)
            rows.append(dict(draw=draw,arms=arms))
        means={arm:{k:sum(float(row['arms'][arm][k]) for row in rows)/2 for k in rows[0]['arms'][arm]}
               for arm in protocol['arms']}
        record=dict(task=ordinal,scene=scene,object=item['object_name'],windows=len(saved['windows']),
            source_joint_error_m=joint_error,source_metric_error=metric_error,source_fk_anchor_error_m=fk_error,
            draws=rows,means=means,solves=solves,query_audit=query_audit,
            hsi_calls=0 if query_audit is None else query_audit['hsi_calls'])
        write_json(dest/'metrics.json',record)
        with (dest/'motion.pt').open('xb') as handle:torch.save(motions,handle)
        records.append(record)
        print(json.dumps(dict(task=ordinal,completed=True)),flush=True)
    torch.cuda.synchronize(device)
    write_json(out/'metrics.json',dict(commit=commit,
        git_commit_at_completion=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        tasks=len(records),windows=sum(r['windows'] for r in records),seconds=time.perf_counter()-started,
        peak_memory_gib=torch.cuda.max_memory_allocated(device)/1024**3,expert_forwards=0 if teacher is None else teacher.calls))


def summarize_body_projection(run_root,task_manifest,device='cuda:7'):
    from .continuation_outcomes import write_json
    from .scene_calibration import paired_local_metrics
    run_root=Path(run_root)
    records={}
    for path in run_root.glob('lanes/*/task-*/metrics.json'):
        row=json.loads(path.read_text());records[row['task']]=row
    tasks=json.loads(Path(task_manifest).read_text())['tasks']
    assert set(records)=={t['canonical_ordinal'] for t in tasks}
    arms=list(next(iter(records.values()))['means'])
    by_task={arm:{str(k):r['means'][arm] for k,r in records.items()} for arm in arms}
    scenes=sorted({r['scene'] for r in records.values()})
    by_scene={arm:{scene:{metric:sum(r['means'][arm][metric] for r in records.values() if r['scene']==scene)/
        sum(r['scene']==scene for r in records.values()) for metric in next(iter(by_task[arm].values()))}
        for scene in scenes} for arm in arms}
    means={arm:{metric:sum(v[metric] for v in by_task[arm].values())/len(records)
                for metric in next(iter(by_task[arm].values()))} for arm in arms}
    pairs=[(a,'source') for a in arms if a!='source']+[
        ('correct_projected','correct_smooth'),('wrong_projected','wrong_smooth'),
        ('correct_projected','wrong_projected'),('correct_smooth','wrong_smooth')]
    contrasts={a+'__minus__'+b:{unit:paired_local_metrics(data[b],data[a],device)
               for unit,data in [('task',by_task),('scene',by_scene)]} for a,b in pairs}
    source,correct,wrong=(means[a] for a in ['source','correct_projected','wrong_projected'])
    hs='scene_human_penetration_s_mean';os='scene_obj_penetration_s_mean'
    geometry=dict(improves_source=correct[hs]<=.99*source[hs],correct_scene_benefit=correct[hs]<=wrong[hs]-.005*source[hs])
    count=sum(r['means']['correct_projected']['body_displacement_cm']>=.5 for r in records.values())
    motion=dict(mean_displacement=correct['body_displacement_cm']>=.5,task_coverage=count>=14)
    max_anchor=max(r['draws'][d]['arms'][a]['native_anchor_max_error_m'] for r in records.values()
                   for d in range(2) for a in ('correct_projected','wrong_projected'))
    protection=dict(contact=correct['contact_percent']>=source['contact_percent']-.002,
        foot_sliding=correct['foot_sliding']<=source['foot_sliding']+.01,
        object_scene=correct[os]<=1.01*source[os],completion=correct['completed']>=source['completed'],
        source_floor_support=correct['source_floor_support_fraction']>=source['source_floor_support_fraction']-.002,
        native_anchors=max_anchor<=1e-5,endpoint_locks=correct['initial_final_max_error_m']==0,
        object_identity=correct['object_max_error_m']==0)
    summary=dict(tasks=len(records),windows=sum(r['windows'] for r in records.values()),draws=2,means=means,
        geometry_conditions=geometry,motion_conditions=motion,protection_conditions=protection,
        meaningful_task_count=count,entry=all(geometry.values()) and all(motion.values()) and all(protection.values()),
        audit=dict(source_joint_max_error_m=max(r['source_joint_error_m'] for r in records.values()),
            source_metric_max_error=max(r['source_metric_error'] for r in records.values()),
            source_fk_anchor_max_error_m=max(r['source_fk_anchor_error_m'] for r in records.values()),
            native_anchor_max_error_m=max_anchor),contrasts=contrasts,
        expert_forwards=sum(r.get('hsi_calls',0) for r in records.values()),
        training_allowed=False,test_set_development=True)
    if summary['expert_forwards']:
        coverage={};sensitivity={};breadth={}
        for reference in ('source','wrong_projected'):
            delta={k:row['means']['correct_projected'][hs]-row['means'][reference][hs] for k,row in records.items()}
            counts=dict(improved=sum(v<0 for v in delta.values()),worsened=sum(v>0 for v in delta.values()),equal=sum(v==0 for v in delta.values()))
            coverage[reference]=counts
            a={str(k):row['means'][reference] for k,row in records.items() if k!=375}
            b={str(k):row['means']['correct_projected'] for k,row in records.items() if k!=375}
            sensitivity[reference]=paired_local_metrics(a,b,device)
            breadth[reference+'_remaining27']=sensitivity[reference][hs]['delta']<=0
            breadth[reference+'_task_coverage']=counts['improved']>=counts['worsened']
        summary.update(task_coverage=coverage,remaining27=sensitivity,evidence_breadth_conditions=breadth,
                       incremental_evidence=summary['entry'] and all(breadth.values()))
    out=run_root/'analysis';out.mkdir()
    for arm in arms:
        for unit,values in [('task',by_task[arm]),('scene',by_scene[arm])]:
            write_json(out/f'{arm}-{unit}.json',dict(metrics=values))
    write_json(out/'summary.json',summary);write_json(out/'records.json',records)
    return summary


@torch.no_grad()
def feasible_body_direction_probe(projection,model,sdf,info,evaluate,baseline,cache,floor,length,protocol,dest):
    """Fixed native rays: correct/wrong HSI and a projected scene-gradient control."""
    import time
    from .surface_edit import decode_body
    from .body_projection import (NativeSceneDifferential,tangent_body_direction,
                                  normalize_body_directions,NATIVE_ANCHORS)
    source=projection.source;device=source['pose'].device
    torch.cuda.synchronize(device);started=time.perf_counter()
    differential=NativeSceneDifferential(projection,model,sdf,info)
    gradient,scalar=differential.gradient()
    torch.cuda.synchronize(device);gradient_seconds=time.perf_counter()-started
    hs='scene_human_penetration_s_mean'
    scalar_error=abs(scalar-baseline[hs])
    assert scalar_error<=1e-5,(scalar,baseline[hs],scalar_error)
    raw={}
    for draw in range(2):
        for view in ('correct','wrong'):
            target=cache[f'{view}_smooth_draw{draw}']
            target_rotation=transforms.axis_angle_to_matrix(target['pose'].double())
            angular=transforms.matrix_to_axis_angle(projection.rotation.double().transpose(-1,-2)@target_rotation)
            raw[f'{view}_draw{draw}']=torch.cat(((target['translation']-source['translation']).double()/projection.position_scale,
                                                angular.flatten(1)/projection.angle_scale),-1)
    raw['geometry']=-gradient
    for direction in raw.values():direction[projection.fixed]=0
    projected={}
    for name,direction in raw.items():projected[name],jacobian=tangent_body_direction(projection,direction)
    directions,normalization=normalize_body_directions(projection,projected)
    audits={};per_frame={}
    for name,direction in directions.items():
        before,after=raw[name].norm(),projected[name].norm()
        cosine_denominator=projected[name].norm()*projected['geometry'].norm()
        audits[name]=dict(raw_norm=float(before),projected_norm=float(after),
            retained_norm_fraction=float(after/before) if float(before)>0 else None,
            raw_derivative=float((gradient*raw[name]).sum()),
            projected_derivative_before_scaling=float((gradient*projected[name]).sum()),
            removed_derivative=float((gradient*(raw[name]-projected[name])).sum()),
            derivative=float((gradient*direction).sum()),
            cosine_to_feasible_geometry=float((projected[name]*projected['geometry']).sum()/cosine_denominator)
                if float(cosine_denominator)>0 else None,
            linear_anchor_max_error_m=float((jacobian@direction[...,None]).abs().max()))
        per_frame[name]=(gradient*direction).sum(-1).cpu()
    def decode(direction,step):
        if torch.count_nonzero(direction)==0:return source,projection.rotation,projection.translation
        rotation=projection.rotation@transforms.axis_angle_to_matrix(
            (direction[:,3:]*step*projection.angle_scale).reshape(-1,22,3)).to(source['pose'])
        translation=projection.translation+(direction[:,:3]*step*projection.position_scale).to(source['translation'])
        rotation,translation=projection.restore(rotation,translation,20)
        pose=source['pose']+(transforms.matrix_to_axis_angle(rotation.double())-
                            transforms.matrix_to_axis_angle(projection.rotation.double())).to(source['pose'])
        pose[projection.fixed]=source['pose'][projection.fixed]
        translation[projection.fixed]=source['translation'][projection.fixed]
        motion=dict(source,pose=pose,translation=translation)
        motion['verts'],motion['joints']=decode_body(motion,model)
        for key in ('verts','joints'):motion[key][projection.fixed]=source[key][projection.fixed]
        return motion,rotation,translation
    def protections(metrics):
        bounds=dict(anchor_max_error_m=1e-6,native_anchor_max_error_m=1e-5,root_max_change_m=.1,
            rotation_max_change_deg=20,correction_speed_max_cm_s=30,
            correction_speed_mean_cm_s=10,correction_seam_speed_mean_cm_s=10)
        failures=[key for key,limit in bounds.items() if metrics[key]>limit]
        if metrics['contact_percent']<baseline['contact_percent']-.002:failures.append('contact')
        if metrics['foot_sliding']>baseline['foot_sliding']+.01:failures.append('foot_sliding')
        if metrics['scene_obj_penetration_s_mean']>baseline['scene_obj_penetration_s_mean']*1.01:failures.append('object_scene')
        if metrics['completed']<baseline['completed']:failures.append('completion')
        if metrics['source_floor_support_fraction']<baseline['source_floor_support_fraction']-.002:failures.append('support')
        if metrics['initial_final_max_error_m']!=0:failures.append('endpoints')
        if metrics['object_max_error_m']!=0:failures.append('object_identity')
        return failures
    results={};states={};failure_rows={}
    state=lambda motion:{k:v.cpu() for k,v in motion.items() if torch.is_tensor(v) and k!='verts'}
    states['source']=state(source)
    for name,direction in directions.items():
        ray_started=time.perf_counter();negative=None
        for label,step in (('negative_quarter',protocol['steps']['central_derivative_negative']),
                           ('quarter',protocol['steps']['secondary']),('unit',protocol['steps']['primary'])):
            motion,rotation,translation=decode(direction,step)
            states[name+'_'+label]=state(motion)
            if step<0:
                negative=float(differential.frame_sums(motion['verts']).mean())
                audits[name]['negative_quarter_hs']=negative
                continue
            metrics=evaluate(motion)
            metrics.update(body_readout_measures(motion,source,floor,length))
            metrics.update(projection.measures(rotation,translation))
            metrics['native_anchor_max_error_m']=float((motion['joints'][:,NATIVE_ANCHORS]-source['joints'][:,NATIVE_ANCHORS]).norm(dim=-1).max())
            metrics['initial_final_max_error_m']=float((motion['joints'][projection.fixed]-source['joints'][projection.fixed]).abs().max())
            metrics['object_max_error_m']=float((motion['object_translation']-source['object_translation']).abs().max())
            failures=protections(metrics)
            metrics.update(protected=float(not failures),hs_directional_derivative=audits[name]['derivative'],
                hs_linear_prediction=step*audits[name]['derivative'],native_hs_change=metrics[hs]-baseline[hs],
                linear_body_rms_m=step*normalization['directions'][name]['achieved_linear_body_rms_m'])
            results[name+'_'+label]=metrics;failure_rows[name+'_'+label]=failures
            audits[name][label+'_hs_change']=metrics[hs]-baseline[hs]
            if label=='quarter':audits[name]['central_derivative']=(metrics[hs]-negative)/(2*step)
        torch.cuda.synchronize(device)
        audits[name]['ray_seconds']=time.perf_counter()-ray_started
        print(json.dumps(dict(direction=name,derivative=audits[name]['derivative'],
            finite=audits[name]['unit_hs_change'],scale=normalization['common_scale'],
            failures=failure_rows[name+'_unit'])),flush=True)
    baseline=dict(baseline,protected=1.,hs_directional_derivative=0.,hs_linear_prediction=0.,
                  native_hs_change=0.,linear_body_rms_m=0.)
    rows=[]
    for draw in range(2):
        arms=dict(source=baseline)
        for view in ('correct','wrong','geometry'):
            for label in ('quarter','unit'):
                key=view if view=='geometry' else f'{view}_draw{draw}'
                arms[view+'_'+label]=results[key+'_'+label]
        rows.append(dict(draw=draw,arms=arms))
    means={arm:{key:sum(float(row['arms'][arm][key]) for row in rows)/2 for key in rows[0]['arms'][arm]}
           for arm in protocol['arms']}
    with (dest/'motion.pt').open('xb') as f:torch.save(states,f)
    with (dest/'directions.pt').open('xb') as f:torch.save(dict(gradient=gradient.cpu(),
        raw={k:v.cpu() for k,v in raw.items()},projected={k:v.cpu() for k,v in projected.items()},
        normalized={k:v.cpu() for k,v in directions.items()},frame_derivatives=per_frame),f)
    torch.cuda.synchronize(device)
    return dict(draws=rows,means=means,direction_audits=audits,normalization=normalization,
        feasibility_failures=failure_rows,gradient_audit=dict(norm=float(gradient.norm()),
            projected_norm=float(projected['geometry'].norm()),native_scalar=scalar,
            source_scalar_max_error=scalar_error,seconds=gradient_seconds),seconds=time.perf_counter()-started)


def summarize_feasible_directions(run_root,task_manifest,device='cuda:7'):
    from .continuation_outcomes import write_json
    from .scene_calibration import paired_local_metrics
    run_root=Path(run_root)
    records={}
    for path in run_root.glob('lanes/*/task-*/metrics.json'):
        row=json.loads(path.read_text());records[row['task']]=row
    tasks=json.loads(Path(task_manifest).read_text())['tasks']
    assert set(records)=={t['canonical_ordinal'] for t in tasks}
    arms=list(next(iter(records.values()))['means'])
    by_task={arm:{str(k):r['means'][arm] for k,r in records.items()} for arm in arms}
    scenes=sorted({r['scene'] for r in records.values()})
    by_scene={arm:{scene:{metric:sum(r['means'][arm][metric] for r in records.values() if r['scene']==scene)/
        sum(r['scene']==scene for r in records.values()) for metric in next(iter(by_task[arm].values()))}
        for scene in scenes} for arm in arms}
    means={arm:{metric:sum(v[metric] for v in by_task[arm].values())/len(records)
                for metric in next(iter(by_task[arm].values()))} for arm in arms}
    pairs=[(a,'source') for a in arms if a!='source']+[
        ('correct_'+step,'wrong_'+step) for step in ('quarter','unit')]+[
        ('correct_unit','correct_quarter'),('geometry_unit','geometry_quarter'),('correct_unit','geometry_unit')]
    contrasts={a+'__minus__'+b:{unit:paired_local_metrics(data[b],data[a],device)
        for unit,data in [('task',by_task),('scene',by_scene)]} for a,b in pairs}
    hs='scene_human_penetration_s_mean'
    coverage={};remaining={};conditions={}
    for reference in ('source','wrong_unit'):
        delta=[r['means']['correct_unit'][hs]-r['means'][reference][hs] for r in records.values()]
        coverage[reference]=dict(improved=sum(x<0 for x in delta),worsened=sum(x>0 for x in delta),equal=sum(x==0 for x in delta))
        a={k:v for k,v in by_task[reference].items() if k!='375'}
        b={k:v for k,v in by_task['correct_unit'].items() if k!='375'}
        remaining[reference]=paired_local_metrics(a,b,device)
        conditions[reference+'_mean']=means['correct_unit'][hs]<means[reference][hs]
        conditions[reference+'_remaining27']=remaining[reference][hs]['delta']<0
        conditions[reference+'_coverage']=coverage[reference]['improved']>=coverage[reference]['worsened']
    conditions['protection']=all(r['means']['correct_unit']['protected']==1 for r in records.values())
    geometry_delta=[r['means']['geometry_unit'][hs]-r['means']['source'][hs] for r in records.values()]
    geometry_coverage=dict(improved=sum(x<0 for x in geometry_delta),worsened=sum(x>0 for x in geometry_delta),equal=sum(x==0 for x in geometry_delta))
    geometry_opportunity=(means['geometry_unit'][hs]<means['source'][hs] and
        geometry_coverage['improved']>geometry_coverage['worsened'] and
        all(r['means']['geometry_unit']['protected']==1 for r in records.values()))
    direction_tables={view:{} for view in ('correct','wrong','geometry')}
    for task,row in records.items():
        for view in direction_tables:
            values=[row['direction_audits']['geometry']] if view=='geometry' else [row['direction_audits'][view+f'_draw{d}'] for d in range(2)]
            keys=('raw_norm','projected_norm','raw_derivative','projected_derivative_before_scaling',
                  'removed_derivative','derivative','linear_anchor_max_error_m','central_derivative','quarter_hs_change','unit_hs_change')
            direction_tables[view][str(task)]={k:sum(v[k] for v in values)/len(values) for k in keys}
    direction_scenes={view:{scene:{metric:sum(row[metric] for key,row in table.items() if records[int(key)]['scene']==scene)/
        sum(records[int(key)]['scene']==scene for key in table) for metric in next(iter(table.values()))}
        for scene in scenes} for view,table in direction_tables.items()}
    direction_contrasts={a+'__minus__'+b:{unit:paired_local_metrics(data[b],data[a],device)
        for unit,data in [('task',direction_tables),('scene',direction_scenes)]}
        for a,b in [('correct','wrong'),('correct','geometry')]}
    evidence=all(conditions.values())
    summary=dict(tasks=len(records),windows=sum(r['windows'] for r in records.values()),draws=2,
        declared_draw_arm_rows=28*2*7,unique_native_evaluations=28*11,expert_forwards=0,
        means=means,contrasts=contrasts,task_coverage=coverage,remaining27=remaining,
        local_hsi_conditions=conditions,local_hsi_signal=evidence,geometry_coverage=geometry_coverage,
        geometry_opportunity=geometry_opportunity,
        classification='limited_step_hsi_signal' if evidence else 'hsi_direction_misalignment' if geometry_opportunity else 'local_opportunity_unestablished',
        direction_contrasts=direction_contrasts,
        audit=dict(source_joint_max_error_m=max(r['source_joint_error_m'] for r in records.values()),
            source_metric_max_error=max(r['source_metric_error'] for r in records.values()),
            source_fk_anchor_max_error_m=max(r['source_fk_anchor_error_m'] for r in records.values()),
            differentiable_source_scalar_max_error=max(r['gradient_audit']['source_scalar_max_error'] for r in records.values()),
            native_anchor_max_error_m=max(r['draws'][d]['arms'][a]['native_anchor_max_error_m'] for r in records.values() for d in range(2) for a in arms)),
        zero_gradient_tasks=[k for k,r in records.items() if r['gradient_audit']['norm']==0],
        zero_hs_tasks=[k for k,r in records.items() if r['means']['source'][hs]==0],
        training_allowed=False,test_set_development=True)
    out=run_root/'analysis';out.mkdir()
    for arm in arms:
        for unit,values in [('task',by_task[arm]),('scene',by_scene[arm])]:
            write_json(out/f'{arm}-{unit}.json',dict(metrics=values))
    write_json(out/'direction_tasks.json',direction_tables)
    write_json(out/'direction_scenes.json',direction_scenes)
    write_json(out/'summary.json',summary);write_json(out/'records.json',records)
    return summary
