"""Measure and generate standing handoffs from cached native HOI motion."""

import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .continuation_outcomes import native_tracks, write_json
from .surface_edit import FEET, decode_body, native_hand_distances, load_object_sdf


def observed_motion(motion):
    """The final two native interpolation samples are held padding."""
    length = len(motion['pose'])
    return {key: value[:-2] if torch.is_tensor(value) and value.ndim and
            value.shape[0] == length and key != 'betas' else value
            for key, value in motion.items()}


def tail_measures(motion, object_vertices, thresholds, start_root=None):
    joints = motion['joints']
    tail = joints[-thresholds.tail_frames:]
    rotation = transforms.axis_angle_to_matrix(motion['pose'][-len(tail):, 0])
    root_velocity = (tail[1:, 0] - tail[:-1, 0]) * 30
    root_omega = transforms.matrix_to_axis_angle(rotation[1:] @ rotation[:-1].transpose(-1, -2)) * 30
    spine = tail[:, 12] - tail[:, 0]
    tilt = torch.acos((spine[:, 1] / spine.norm(dim=-1)).clamp(-1, 1)) * 180 / math.pi
    foot_distance = tail[:, FEET, 1].abs().amin(-1)
    obj_position = motion['object_translation'][-len(tail):]
    obj_rotation = motion['object_rotation'][-len(tail):]
    obj_world = (obj_rotation @ object_vertices.T).transpose(-1, -2) + obj_position[:, None]
    hands = native_hand_distances(tail, obj_world)
    obj_omega = transforms.matrix_to_axis_angle(obj_rotation[1:] @ obj_rotation[:-1].transpose(-1, -2)) * 30
    result = dict(
        tilt_max_deg=float(tilt.max()), root_height_min_m=float(tail[:, 0, 1].min()),
        foot_height_max_m=float(foot_distance.max()),
        foot_speed_mean_m_s=float(((tail[1:, FEET] - tail[:-1, FEET]) * 30).norm(dim=-1).mean()),
        root_speed_max_m_s=float(root_velocity.norm(dim=-1).max()),
        root_angular_speed_max_rad_s=float(root_omega.norm(dim=-1).max()),
        hand_object_min_m=float(hands.min()),
        object_floor_max_m=float(obj_world[..., 1].amin(-1).abs().max()),
        object_speed_max_m_s=float(((obj_position[1:] - obj_position[:-1]) * 30).norm(dim=-1).max()),
        object_angular_speed_max_rad_s=float(obj_omega.norm(dim=-1).max()),
    )
    standing = dict(upright=result['tilt_max_deg'] <= thresholds.tilt_deg,
                    height=result['root_height_min_m'] >= thresholds.root_height_m,
                    support=result['foot_height_max_m'] <= thresholds.foot_height_m)
    release = dict(hands=result['hand_object_min_m'] >= thresholds.hand_distance_m,
                   grounded=result['object_floor_max_m'] <= thresholds.object_floor_m,
                   object_slow=result['object_speed_max_m_s'] <= thresholds.object_speed_m_s,
                   object_rotation_slow=result['object_angular_speed_max_rad_s'] <= thresholds.object_angular_speed_rad_s)
    result.update(standing=all(standing.values()), released_grounded=all(release.values()),
                  standing_conditions=standing, release_conditions=release)
    result['eligible'] = result['standing'] and result['released_grounded']
    if start_root is not None:
        result['root_drift_max_m'] = float((joints[:, 0, [0, 2]] - start_root[[0, 2]]).norm(dim=-1).max())
        conditions = dict(standing, hands=release['hands'],
            slow=result['root_speed_max_m_s'] <= thresholds.root_speed_m_s,
            rotation_slow=result['root_angular_speed_max_rad_s'] <= thresholds.root_angular_speed_rad_s,
            stays_near=result['root_drift_max_m'] <= thresholds.root_drift_m)
        result['standing_success_conditions'] = conditions
        result['standing_success'] = all(conditions.values())
    return result


def select_sources(rows, count):
    """Prefer valid handoffs, preserving labelled standing stress cases."""
    selected, scenes = [], set()
    for eligible in (True, False):
        pool = [r for r in rows if r['standing'] and r['eligible'] == eligible]
        while pool and len(selected) < count:
            for name in sorted(set(r['object'] for r in pool)):
                candidates = [r for r in pool if r['object'] == name]
                row = min(candidates, key=lambda r: (r['scene'] in scenes, r['task']))
                selected.append(row)
                scenes.add(row['scene'])
                pool = [r for r in pool if r['task'] != row['task']]
                if len(selected) == count:
                    break
    return selected


def scene_measures(vertices, sdf, info):
    extent = float(max(info['extents']))
    centre = vertices.new_tensor(info['centroid'])
    query = (vertices - centre) / (extent / 2)
    signed = torch.nn.functional.grid_sample(sdf, query[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3),
        padding_mode='border', align_corners=True).reshape(vertices.shape[:2]) * extent / 2
    depth = (-signed).clamp_min(0)
    return dict(scene_penetration_mean_m=float(depth.mean()), scene_penetration_max_m=float(depth.max()),
                scene_outside_fraction=float((query.abs() > 1).any(-1).float().mean()))


def object_measures(track, sdf, info):
    from eval_metrics import compute_signed_distances
    relative = (track['object_rotation'].transpose(-1, -2) @
        (track['verts'] - track['object_translation'][:, None]).transpose(-1, -2)).transpose(-1, -2)
    signed = compute_signed_distances(sdf, relative.new_tensor(info['centroid'])[None],
                                     relative.new_tensor(info['extents'])[None], relative)
    return dict(human_object_penetration_mean_m=float((-signed).clamp_min(0).mean()),
                human_object_penetration_max_m=float((-signed).clamp_min(0).max()))


def source_task(row, root):
    tasks = json.loads((root / 'data/hosi_test/data' / (row['scene'] + '.json')).read_text())
    return dict(tasks[row['test_idx']], test_idx=row['test_idx'])


def load_motion(path, arm, device):
    saved = torch.load(path, map_location='cpu', weights_only=False)[arm]
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in saved.items()}


def audit_sources(cfg, out):
    import trimesh
    from utils import zup_to_yup
    vertices = {}
    tasks = json.loads(Path(cfg.stand_wait.task_manifest).read_text())['tasks']
    paths = {int(p.parent.name[5:]): p for p in Path(cfg.stand_wait.source_root).glob('lanes/*/task-*/motion.pt')}
    rows = []
    for item in tasks:
        ordinal, name = item['canonical_ordinal'], item['object_name']
        if name not in vertices:
            mesh = trimesh.load_mesh(Path(cfg.dataset.folder) / 'rest_object_geo' / (name + '.ply'))
            vertices[name] = torch.as_tensor(zup_to_yup(np.asarray(mesh.vertices)), device=cfg.device, dtype=torch.float32)
        motion = observed_motion(load_motion(paths[ordinal], cfg.stand_wait.source_arm, cfg.device))
        row = dict(task=ordinal, scene=item['scene_name'], object=name, test_idx=item['test_idx'],
                   path=str(paths[ordinal]), **tail_measures(motion, vertices[name], cfg.stand_wait.thresholds))
        rows.append(row)
    write_json(out / 'source_audit.json', rows)
    selected = select_sources(rows, cfg.stand_wait.max_sources)
    summary = dict(total=len(rows), standing=sum(r['standing'] for r in rows),
                   eligible=sum(r['eligible'] for r in rows), selected=[r['task'] for r in selected],
                   selected_eligible=sum(r['eligible'] for r in selected),
                   release_failure_counts=dict(Counter(k for r in rows for k, v in r['release_conditions'].items() if not v)),
                   standing_failure_counts=dict(Counter(k for r in rows for k, v in r['standing_conditions'].items() if not v)))
    write_json(out / 'selection.json', summary)
    print(json.dumps(dict(source_audit=summary)), flush=True)
    return selected, summary, vertices


def make_history(cfg, dataset, motion, task, model):
    from test_infbagel_hosi import get_mat, decode_sample_window
    from .body_projection import native_rest_offsets
    from .kinematic_composition import _forward_kinematics
    from .hsi_motion_target import encode_native_window, native_coarse_pose
    sequence = dataset.ori_sequence_idx[task['data_idx']]
    translation = torch.as_tensor(dataset.transl[sequence], device=cfg.device)
    offsets = native_rest_offsets(model, motion['betas'])
    history_indices = [-4, -1]
    indices = history_indices + [-1] * 14
    pose = motion['pose'][indices]
    local_rotation = transforms.axis_angle_to_matrix(pose)
    native_offsets = offsets[None].expand(16, -1, -1).clone()
    native_offsets[:, 0] += motion['translation'][indices]
    world_rotation, _ = _forward_kinematics(local_rotation, native_offsets)
    world_rotation = world_rotation[:, :22]
    points = motion['joints'][indices] - offsets[0] - translation
    mat = get_mat(cfg, points[None].flatten(2), 0)
    reference = torch.eye(3, device=cfg.device)[None]
    object_position = motion['object_translation'][-1:].expand(16, -1)
    object_rotation = motion['object_rotation'][-1:].expand(16, -1, -1)
    clean = encode_native_window(dataset, points[None], world_rotation[None], object_position[None],
        object_rotation[None], torch.zeros(1, 16, 4, device=cfg.device), mat, reference)
    decoded = decode_sample_window(cfg, clean, dataset, mat)
    rebuilt = dict(motion, pose=native_coarse_pose(decoded['global_rot_6d'], dataset),
                   translation=decoded['points_orig'].reshape(16, 28, 3)[:, 0] + translation)
    _, joints = decode_body(rebuilt, model)
    error = float((joints[:2] - motion['joints'][history_indices]).abs().max())
    if error > 1e-4:
        raise AssertionError(('native history reconstruction', error))
    return clean, mat, reference, error


@torch.no_grad()
def sample_transition(cfg, sampler, dataset, motion, task, model, embedding):
    from test_infbagel_hosi import decode_sample_window
    from .input_views import KnownEmptyObjectView
    clean, mat, reference, history_error = make_history(cfg, dataset, motion, task, model)
    fixed = clean[:, :2].clone()
    zero = torch.zeros(1, dtype=torch.bool, device=cfg.device)
    one = ~zero
    # The learned position channel has a native root reconstruction offset.
    # Its stationary goal is expressed in that same channel frame.
    local_goal = dataset.denormalize_torch(clean[:, -1:, :3]).reshape(1, 3)
    local_goal[:, 1] = 0
    obj_vertices = dataset.obj_rest_verts[task['object_name']].to(cfg.device)
    object_world = (motion['object_rotation'][-1] @ obj_vertices.T).T + motion['object_translation'][-1]
    sequence = dataset.ori_sequence_idx[task['data_idx']]
    seq_name = dataset.scene_name[sequence]
    context = sampler._hsi_context(1, mat, torch.tensor([dataset.scene_dict['occ_' + task['scene_name']]], device=cfg.device),
        embedding, local_goal, torch.zeros(1, 3, device=cfg.device), torch.zeros(1, 3, device=cfg.device),
        one, one, torch.zeros(1, device=cfg.device, dtype=torch.long),
        torch.full((1,), 48, device=cfg.device, dtype=torch.long),
        torch.full((1,), 48, device=cfg.device, dtype=torch.long), one, zero, zero,
        torch.zeros(1, 1024, 3, device=cfg.device), object_world[None], reference,
        dataset.obj_rest_verts, {}, {0: seq_name}, torch.eye(3, device=cfg.device)[None], False)
    offsets = torch.as_tensor(dataset.rest_human_offsets[sequence], device=cfg.device, dtype=torch.float32)
    human_dict = dict(rest_human_offsets=offsets[None, None].expand(1, 16, -1, -1).clone())
    torch.manual_seed(42)
    generator = torch.Generator(device=cfg.device).manual_seed(42)
    current = torch.randn(clean.shape, device=cfg.device, generator=generator)
    current[:, :2] = fixed
    previous = current.clone()
    sampler.hsi_input_view = KnownEmptyObjectView()
    sampler.hsi_input_view.begin_window(current, 42)
    native = sampler.hsi_sampler
    c1, c2, logvar = [x.to(cfg.device) for x in
        (native.posterior_mean_coef1, native.posterior_mean_coef2, native.posterior_log_variance_clipped)]
    torch.cuda.synchronize(cfg.device)
    started = time.perf_counter()
    for step in reversed(range(500)):
        t = torch.full((1,), step, dtype=torch.long, device=cfg.device)
        query = current.clone()
        query[..., 216:] = clean[..., 216:]
        previous[..., 216:] = clean[..., 216:]
        prediction = sampler._hsi_x0(query, previous, t, context)
        prediction[:, :2] = fixed
        updated = c1[step] * prediction + c2[step] * current
        if step:
            updated += (logvar[step] * .5).exp() * torch.randn(current.shape, device=cfg.device, generator=generator)
            updated = sampler._apply_hsi_posterior_guidance(updated, prediction, t, context, human_dict, fixed)
        updated[:, :2] = fixed
        current, previous = updated, prediction
    torch.cuda.synchronize(cfg.device)
    seconds = time.perf_counter() - started
    current[..., 216:] = clean[..., 216:]
    decoded = decode_sample_window(cfg, current, dataset, mat)
    world = dict(points_world=decoded['points_orig'].reshape(-1, 28, 3).cpu(),
                 global_rot_6d=decoded['global_rot_6d'].cpu(),
                 object_translation_world=motion['object_translation'][-1:].expand(16, -1).cpu(),
                 object_rotation_world=motion['object_rotation'][-1:].expand(16, -1, -1).cpu())
    return world, dict(history_max_error_m=history_error, generation_seconds=seconds,
                       hsi_forward_calls=1000, hoi_forward_calls=0, seed=42,
                       fixed_history_max_error=float((current[:, :2, :216] - fixed[..., :216]).abs().max()),
                       object_state_max_error_m=0.0), current.cpu()


def load_texts(cfg):
    import clip
    model, _ = clip.load(cfg.stand_wait.clip_checkpoint, device=cfg.device)
    model.eval()
    with torch.no_grad():
        features = model.encode_text(clip.tokenize(list(cfg.stand_wait.prompts)).to(cfg.device)).float()
        features = features / features.norm(dim=-1, keepdim=True)
    return features


@torch.no_grad()
def run_standing_transition(cfg):
    import hydra
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from utils import create_smplx_model
    from test_infbagel_lingo_hsi import _remap_checkpoint_keys
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('reportable standing transition requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    torch.set_num_threads(4)
    out = Path(cfg.hosi_output_dir)
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), out / 'resolved.yaml')
    started = time.perf_counter()
    selected, source_summary, _ = audit_sources(cfg, out)
    records = []
    if selected:
        embeddings = load_texts(cfg)
        model_cfg = OmegaConf.merge(cfg.model.infbagel, dict(ckpt=str(cfg.hsi_ckpt_path)))
        model = hydra.utils.instantiate(model_cfg)
        state, _ = _remap_checkpoint_keys(torch.load(cfg.hsi_ckpt_path, map_location='cpu', weights_only=False))
        model.load_state_dict(state, strict=True)
        del state
        model = model.to(cfg.device).eval().requires_grad_(False)
        sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
        dataset, current_scene, smpl_cache = None, None, {}
        for row in selected:
            task = source_task(row, root)
            if current_scene != row['scene']:
                if dataset is None:
                    dc = OmegaConf.merge(cfg.dataset, dict(test_scene_name=row['scene']))
                    dataset = InfBaGelDataset(**dc)
                else:
                    dataset.set_test_scene(row['scene'])
                dataset.obj_rest_verts = {k: v.to(cfg.device) for k, v in dataset.obj_rest_verts.items()}
                sampler.dataset = dataset
                sampler.hsi_sampler.set_dataset_and_model(dataset, model)
                sdf_root = root / 'data/hosi_test/Scene_sdf'
                info = json.loads((sdf_root / (row['scene'] + '_sdf_info.json')).read_text())
                sdf = torch.as_tensor(np.load(sdf_root / (row['scene'] + '_sdf.npy')), device=cfg.device, dtype=torch.float32)[None, None]
                current_scene = row['scene']
            motion = observed_motion(load_motion(row['path'], cfg.stand_wait.source_arm, cfg.device))
            gender = dataset.gender[dataset.ori_sequence_idx[task['data_idx']]]
            motion['gender'] = gender
            if gender not in smpl_cache:
                smpl_cache[gender] = create_smplx_model(gender, torch.device(cfg.device), batch_size=1)
            object_vertices = dataset.obj_rest_verts[row['object']]
            object_sdf, object_info = load_object_sdf(root / 'data/object/rest_object_sdf_256_npy_files', row['object'])
            object_sdf = torch.as_tensor(object_sdf, device=cfg.device, dtype=torch.float32)[None]
            held = {k: v[-1:].expand(43, *v.shape[1:]) if torch.is_tensor(v) and v.ndim and
                    v.shape[0] == len(motion['pose']) and k != 'betas' else v for k, v in motion.items()}
            held['verts'], held['joints'] = decode_body(held, smpl_cache[gender])
            hold_metrics = tail_measures(held, object_vertices, cfg.stand_wait.thresholds, motion['joints'][-1, 0])
            hold_metrics.update(scene_measures(held['verts'], sdf, info))
            hold_metrics.update(object_measures(held, object_sdf, object_info))
            task_directory = out / f'task-{row["task"]:03d}'
            task_directory.mkdir()
            write_json(task_directory / 'static_reference.json', dict(kind='held source pose, not generated motion', metrics=hold_metrics))
            for prompt_index, prompt in enumerate(cfg.stand_wait.prompts):
                dest = out / f'task-{row["task"]:03d}' / f'prompt-{prompt_index}'
                dest.mkdir(parents=True)
                torch.cuda.reset_peak_memory_stats(cfg.device)
                world, audit, clean = sample_transition(cfg, sampler, dataset, motion, task,
                    smpl_cache[gender], embeddings[prompt_index:prompt_index + 1])
                track = native_tracks(cfg, dataset, world, task, False, smpl_cache, body_parameters=True)
                # Frame3 is the current handoff pose; preceding frames are history.
                track = {k: v[3:] if torch.is_tensor(v) and v.ndim and v.shape[0] == 46 and k != 'betas' else v
                         for k, v in track.items()}
                metrics = tail_measures(track, object_vertices, cfg.stand_wait.thresholds, motion['joints'][-1, 0])
                metrics.update(scene_measures(track['verts'], sdf, info))
                metrics.update(object_measures(track, object_sdf, object_info))
                metrics['seam_position_max_m'] = float((track['joints'][0] - motion['joints'][-1]).norm(dim=-1).max())
                incoming_velocity = (motion['joints'][-1] - motion['joints'][-4]) * 10
                outgoing_velocity = (track['joints'][3] - track['joints'][0]) * 10
                metrics['seam_velocity_change_mean_m_s'] = float((outgoing_velocity - incoming_velocity).norm(dim=-1).mean())
                metrics['geometry_success'] = (metrics['scene_penetration_mean_m'] <= cfg.stand_wait.thresholds.scene_mean_penetration_m
                    and metrics['scene_penetration_max_m'] <= cfg.stand_wait.thresholds.scene_max_penetration_m
                    and metrics['scene_outside_fraction'] == 0)
                metrics['handoff_success'] = bool(row['eligible'] and metrics['standing_success'] and metrics['geometry_success']
                    and audit['history_max_error_m'] <= 1e-4)
                audit['peak_allocated_bytes'] = torch.cuda.max_memory_allocated(cfg.device)
                record = dict(task=row['task'], scene=row['scene'], object=row['object'], prompt=prompt,
                              prompt_index=prompt_index, source_eligible=row['eligible'], metrics=metrics, audit=audit,
                              static_reference=hold_metrics)
                with (dest / 'motion.pt').open('xb') as handle:
                    torch.save(dict(world=world, clean=clean, source=row,
                        motion={k: v.cpu() if torch.is_tensor(v) else v for k, v in track.items() if k != 'verts'}), handle)
                write_json(dest / 'metrics.json', record)
                records.append(record)
                print(json.dumps(record), flush=True)
    summary = dict(source_audit=source_summary, tasks=len(selected), windows=len(records), records=records,
                   git_commit=commit, git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
                   seconds=time.perf_counter() - started,
                   source_manifest=str(Path(cfg.stand_wait.source_root) / 'manifest.json'),
                   model=str(cfg.hsi_ckpt_path), contention=True, gpu=str(cfg.device), seed=42,
                   timing_scope='shared-GPU exploratory diagnostic, includes source scan and data/model setup')
    summary['prompts'] = {}
    for index, prompt in enumerate(cfg.stand_wait.prompts):
        rows = [r for r in records if r['prompt_index'] == index]
        summary['prompts'][str(index)] = dict(text=prompt, count=len(rows),
            standing_success=sum(r['metrics']['standing_success'] for r in rows),
            handoff_success=sum(r['metrics']['handoff_success'] for r in rows),
            source_eligible=sum(r['source_eligible'] for r in rows),
            mean={k: float(np.mean([r['metrics'][k] for r in rows])) for k in
                ('root_drift_max_m', 'root_speed_max_m_s', 'tilt_max_deg', 'hand_object_min_m',
                 'scene_penetration_mean_m', 'scene_penetration_max_m', 'seam_velocity_change_mean_m_s')} if rows else {})
    write_json(out / 'metrics.json', summary)
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}), flush=True)
