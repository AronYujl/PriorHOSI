"""Source-only OMOMO/LINGO transition membership and target-scene checks."""

import json
import math
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .inbetween import heading
from .multitask import SourceCorpus, audit_source_boundaries, lingo_sources, original_tasks, write_json
from .multitask_geometry import geometry_measures, geometry_passes, signed_query, transformed_motion
from .surface_edit import load_object_sdf, yaw_matrix


def direction_guard(direction, boundary, hand_distance_m):
    """Membership state guard; it never reads generated-model output."""
    if direction == 'omomo_to_lingo':
        return {
            'omomo_upright': boundary['torso_tilt_max_deg'] <= 25,
            'omomo_supported': boundary['pelvis_height_min_m'] >= .65,
            'omomo_feet_supported': boundary['foot_floor_distance_max_m'] <= .08,
            'object_supported_slow': boundary['supported_slow'],
            'hands_released': boundary['hands_released'],
        }
    if direction == 'lingo_to_omomo':
        return {'omomo_initial_hands_released': boundary['hand_object_min_m'] >= hand_distance_m}
    raise ValueError(f'unknown transition direction: {direction}')


def select_lingo_pool(records, per_type):
    """Select a deterministic source-ordered pool without model scores."""
    ordered = sorted(records, key=lambda r: (r['data_idx'], r['source_id']))
    selected = [record for action_type in sorted({record['task_type'] for record in ordered})
                for record in [item for item in ordered if item['task_type'] == action_type][:per_type]]
    return sorted(selected, key=lambda r: (r['data_idx'], r['source_id']))


def source_motion(corpus, record, frames, model, device):
    """Retarget source rotations to the OMOMO subject body without scene copying."""
    from utils import SMPLX_JOINTS_28, run_smplx_model
    pose = torch.as_tensor(np.concatenate((corpus.orient[frames, None],
        corpus.pose[frames].reshape(-1, 21, 3)), axis=1), device=device, dtype=torch.float32)
    if corpus.name == 'OMOMO':
        correction = pose.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        pose[:, 0] = transforms.matrix_to_axis_angle(
            correction @ transforms.axis_angle_to_matrix(pose[:, 0]))
    betas = torch.as_tensor(np.array(record['_target_betas']), device=device, dtype=torch.float32)
    neutral = model.J_regressor @ (model.v_template + torch.einsum(
        'vci,i->vc', model.shapedirs[..., :len(betas)], betas))
    roots = torch.as_tensor(np.array(corpus.joints[frames, 0]), device=device, dtype=torch.float32)
    translation = roots-neutral[0]
    vertices, joints = run_smplx_model(pose, translation, betas, record['_target_gender'],
        joints_ind=SMPLX_JOINTS_28, smpl_model=model)
    return dict(pose=pose, translation=translation, joints=joints, verts=vertices,
        betas=betas, gender=record['_target_gender'])


def align_motion(motion, source_frame, target_root, target_heading):
    source_root = motion['joints'][source_frame, 0]
    source_heading = heading(motion['joints'][source_frame:source_frame+1])[0]
    delta = torch.atan2((target_heading-source_heading).sin(),
                        (target_heading-source_heading).cos())
    rotation = yaw_matrix(delta.reshape(1))[0]
    translation = target_root-source_root @ rotation.T
    return transformed_motion(motion, rotation, translation), dict(
        yaw_rad=float(delta), source_root=source_root.tolist(), target_root=target_root.tolist(),
        source_heading_rad=float(source_heading), target_heading_rad=float(target_heading))


def object_scene_passes(object_world, sdf, info, thresholds):
    signed, outside = signed_query(object_world, sdf, info)
    values = dict(scene_penetration_mean_m=float((-signed).clamp_min(0).mean()),
        scene_penetration_max_m=float((-signed).clamp_min(0).max()),
        scene_outside_fraction=float(outside.float().mean()),
        floor_penetration_max_m=float((-object_world[..., 1]).clamp_min(0).max()))
    values['passes'] = (values['scene_penetration_mean_m'] <= thresholds.scene_mean_penetration_m
        and values['scene_penetration_max_m'] <= thresholds.scene_max_penetration_m
        and values['scene_outside_fraction'] == 0
        and values['floor_penetration_max_m'] <= thresholds.floor_max_penetration_m)
    return values


def full_source_geometry(motion, scene, object_sdf, object_info, object_vertices,
                         object_position, object_rotation, thresholds):
    count = len(motion['joints'])
    positions = object_position.expand(count, -1)
    rotations = object_rotation.expand(count, -1, -1)
    body = geometry_measures(motion['verts'], scene[0], scene[1], object_sdf,
        object_info, positions[:, None], rotations)
    object_world = object_vertices @ object_rotation.T+object_position
    obj_scene = object_scene_passes(object_world, scene[0], scene[1], thresholds)
    return dict(body=body, object_scene=obj_scene,
        passes=geometry_passes(body) and obj_scene['passes'],
        frame_count=count)


def task_object_transform(corpus, record, task, direction, body_motion):
    frames = record['task_reference_context_frames'] if direction == 'omomo_to_lingo' else record['initial_context_frames']
    source_object = torch.as_tensor(np.array(corpus.object_translation[frames[-1 if direction == 'omomo_to_lingo' else 0]]),
        device=body_motion['joints'].device, dtype=torch.float32)
    source_rotation = torch.as_tensor(np.array(corpus.object_rotation[frames[-1 if direction == 'omomo_to_lingo' else 0]]),
        device=body_motion['joints'].device, dtype=torch.float32)
    if direction == 'omomo_to_lingo':
        rotation = terminal_goal_rotation(body_motion['joints'][-1, 0], source_object, task)
        target_root = body_motion['joints'][-1, 0].clone()
        target_root[[0, 2]] = target_root.new_tensor(task['pelvis_goal'])[[0, 2]]
        shift = target_root-body_motion['joints'][-1, 0] @ rotation.T
        object_position = target_root.new_tensor(task['object_goal'])
        object_rotation = rotation @ source_rotation
        return transformed_motion(body_motion, rotation, shift), object_position, object_rotation
    target_root = body_motion['joints'][0, 0].clone()
    target_root[[0, 2]] = target_root.new_tensor(task['start_location'])[[0, 2]]
    shift = target_root-body_motion['joints'][0, 0]
    return transformed_motion(body_motion, torch.eye(3, device=shift.device), shift), source_object+shift, source_rotation


def terminal_goal_rotation(source_pelvis, source_object, task):
    source = source_object-source_pelvis
    target = source_object.new_tensor(task['object_goal'])-source_object.new_tensor(task['pelvis_goal'])
    yaw = torch.atan2(source[2], source[0])-torch.atan2(target[2], target[0])
    return yaw_matrix(yaw.reshape(1))[0]


def _load_scene(root, scene_name, device):
    sdf_root = root/'data/hosi_test/Scene_sdf'
    return (torch.as_tensor(np.load(sdf_root/(scene_name+'_sdf.npy')), dtype=torch.float32,
            device=device)[None, None],
        json.loads((sdf_root/(scene_name+'_sdf_info.json')).read_text()))


@torch.no_grad()
def run_source_eligibility(cfg):
    from omegaconf import OmegaConf
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered source eligibility requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = Path(cfg.multitask.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    device = cfg.device
    thresholds = cfg.multitask.source_eligibility
    corpora = {name: SourceCorpus(root, name) for name in ('OMOMO', 'LINGO')}
    tasks, hoi = original_tasks(root, corpora['OMOMO'])
    split = json.loads(Path(cfg.multitask.split_manifest).read_text())
    lingo, exclusions = lingo_sources(corpora['LINGO'], split)
    all_sources = hoi+lingo
    torch.cuda.synchronize(device)
    audit_source_boundaries(corpora, all_sources, device)
    torch.cuda.synchronize(device)
    sources = {record['source_id']: record for record in all_sources}
    lingo = sorted(lingo, key=lambda r: (r['data_idx'], r['source_id']))
    per_type = int(cfg.multitask.source_eligibility.source_candidates_per_type)
    lingo_pool = select_lingo_pool(lingo, per_type)
    models, object_cache, motion_cache = {}, {}, {}
    accepted, attempts = [], []
    scene_claimed = {'omomo_to_lingo': set(), 'lingo_to_omomo': set()}
    direction_limits = {'omomo_to_lingo': int(cfg.multitask.episode_limit),
                        'lingo_to_omomo': int(cfg.multitask.episode_limit)}

    for direction in ('omomo_to_lingo', 'lingo_to_omomo'):
        for task_row in tasks:
            if len([r for r in accepted if r['direction'] == direction]) >= direction_limits[direction]:
                break
            task = task_row['original_task']; hoi_record = sources[task_row['source_id']]
            scene_name = task['scene_name']
            if scene_name in scene_claimed[direction]:
                continue
            gate_boundary = hoi_record['source_boundary_audit']['exit' if direction == 'omomo_to_lingo' else 'entry']
            state_checks = direction_guard(direction, gate_boundary, float(thresholds.hand_distance_m))
            if not all(state_checks.values()):
                attempts.append(dict(direction=direction, task_id=task_row['task_id'],
                    status='omomo_source_state_ineligible', state_checks=state_checks))
                continue
            if hoi_record['gender'] not in models:
                models[hoi_record['gender']] = create_smplx_model(hoi_record['gender'], torch.device(device)).eval().requires_grad_(False)
            model = models[hoi_record['gender']]
            target_betas = np.asarray(corpora['OMOMO'].betas[hoi_record['source_sequence_idx']])
            object_name = task['object_name']
            if object_name not in object_cache:
                mesh = trimesh.load_mesh(root/'data/test/rest_object_geo'/(object_name+'.ply'))
                object_vertices = torch.as_tensor(zup_to_yup(np.asarray(mesh.vertices)), device=device, dtype=torch.float32)
                obj_sdf, obj_info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', object_name)
                object_cache[object_name] = (object_vertices, torch.as_tensor(obj_sdf, dtype=torch.float32, device=device)[None, None], obj_info)
            object_vertices, object_sdf, object_info = object_cache[object_name]
            scene = _load_scene(root, scene_name, device)
            omomo_frames = (hoi_record['task_reference_context_frames'] if direction == 'omomo_to_lingo'
                            else hoi_record['initial_context_frames'])
            hoi_motion_record = dict(hoi_record, _target_betas=target_betas, _target_gender=hoi_record['gender'])
            hoi_motion = source_motion(corpora['OMOMO'], hoi_motion_record, omomo_frames, model, device)
            hoi_motion, object_position, object_rotation = task_object_transform(
                corpora['OMOMO'], hoi_record, task, direction, hoi_motion)
            object_world = object_vertices @ object_rotation.T+object_position
            object_geometry = object_scene_passes(object_world, scene[0], scene[1], thresholds)
            omomo_geometry = full_source_geometry(hoi_motion, scene, object_sdf, object_info,
                object_vertices, object_position, object_rotation, thresholds)
            if not omomo_geometry['passes']:
                attempts.append(dict(direction=direction, task_id=task_row['task_id'],
                    omomo_source_id=hoi_record['source_id'], scene_name=scene_name,
                    state_checks=state_checks, object_geometry=object_geometry,
                    omomo_source_geometry=omomo_geometry,
                    status='omomo_source_scene_geometry_failed', source_only=True,
                    model_output_used=False))
                continue
            for lingo_record in lingo_pool:
                if lingo_record['task_type'] not in ('locomotion', 'static_object_interaction'):
                    continue
                attempt = dict(direction=direction, task_id=task_row['task_id'],
                    omomo_source_id=hoi_record['source_id'], lingo_source_id=lingo_record['source_id'],
                    scene_name=scene_name, source_scene=lingo_record['source_scene'],
                    lingo_frame_interval=[lingo_record['source_start_frame'], lingo_record['source_stop_frame']],
                    state_checks=state_checks, object_geometry=object_geometry,
                    omomo_source_geometry=omomo_geometry)
                source_key = (lingo_record['source_id'], hoi_record['source_id'])
                if source_key not in motion_cache:
                    lingo_record = dict(lingo_record, _target_betas=target_betas, _target_gender=hoi_record['gender'])
                    frames = list(range(lingo_record['source_start_frame'], lingo_record['source_stop_frame']))
                    motion_cache[source_key] = source_motion(corpora['LINGO'], lingo_record, frames, model, device)
                lingo_motion = motion_cache[source_key]
                target_root = hoi_motion['joints'][-1 if direction == 'omomo_to_lingo' else 0, 0]
                target_heading = heading(hoi_motion['joints'][-10:] if direction == 'omomo_to_lingo' else hoi_motion['joints'][:10]).mean()
                aligned, alignment = align_motion(lingo_motion, 0 if direction == 'omomo_to_lingo' else len(lingo_motion['joints'])-1,
                    target_root, target_heading)
                geometry = full_source_geometry(aligned, scene, object_sdf, object_info,
                    object_vertices, object_position, object_rotation, thresholds)
                attempt.update(alignment=alignment, source_motion_geometry=geometry,
                    source_only=True, model_output_used=False)
                if not geometry['passes']:
                    attempt['status'] = 'lingo_source_scene_geometry_failed'
                    attempts.append(attempt)
                    continue
                attempt['status'] = 'accepted_source_transition'
                attempt['target_object_translation'] = object_position.tolist()
                attempt['target_object_rotation'] = object_rotation.tolist()
                accepted.append(attempt)
                scene_claimed[direction].add(scene_name)
                attempts.append(attempt)
                break
    write_json(output/'candidate_transitions.json', dict(schema_version=1, candidates=accepted))
    write_json(output/'transition_audit.json', dict(schema_version=1, attempts=attempts))
    write_json(output/'source_catalog.json', dict(original_tasks=tasks, sources=all_sources, exclusions=exclusions))
    pool_ids = {record['source_id'] for record in lingo_pool}
    write_json(output/'pool_selection.json', dict(rule='first source_candidates_per_type eligible LINGO records by action type and data_idx',
        source_candidates_per_type=per_type, selected_source_ids=sorted(pool_ids),
        pool_not_attempted_source_ids=sorted(record['source_id'] for record in lingo if record['source_id'] not in pool_ids)))
    summary = dict(schema_version=1, subphase='5.5.2a', seed=int(cfg.seed), git_commit=commit,
        original_hosi_tasks=len(tasks), lingo_sources=len(lingo), excluded_lingo_sources=len(exclusions),
        lingo_pool_size=len(lingo_pool), lingo_pool_not_attempted=len(lingo)-len(lingo_pool),
        accepted_by_direction=Counter(r['direction'] for r in accepted),
        attempts_by_status=Counter(r['status'] for r in attempts), candidates=len(accepted),
        model_output_used=False, generated_motion_samples=0,
        elapsed_seconds=time.perf_counter()-started, device=str(device),
        next_subphase='5.5.2b actual-history execution')
    summary['accepted_by_direction'] = dict(summary['accepted_by_direction'])
    summary['attempts_by_status'] = dict(summary['attempts_by_status'])
    write_json(output/'summary.json', summary)
    write_json(output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    print(json.dumps(summary), flush=True)
