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
from .multitask import (SourceCorpus, audit_source_boundaries, lingo_sources, original_tasks,
    transition_edge, validate_episode, write_json)
from .multitask_geometry import (geometry_measures, seating_support_mask, segment_from_source,
    signed_query, support_patches, terminal_goal_rotation, transformed_motion)
from .surface_edit import decode_body, load_object_sdf, yaw_matrix

GRASPED_ENTRY = 'lingo_to_omomo_grasped_entry'
MOTION_KEYS = ('pose', 'translation', 'joints', 'verts', 'object_translation', 'object_rotation')


def direction_guard(direction, boundary, hand_distance_m, contact_fraction=.5):
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
    if direction == GRASPED_ENTRY:
        return dict(initial_contact=boundary['first_frame_hand_contact'],
            sustained_source_contact=boundary['hand_contact_frame_fraction'] >= contact_fraction)
    raise ValueError(f'unknown transition direction: {direction}')


def select_lingo_pool(records, per_type):
    """Select a deterministic source-ordered pool without model scores."""
    ordered = sorted(records, key=lambda r: (r['data_idx'], r['source_id']))
    selected = [record for action_type in sorted({record['task_type'] for record in ordered})
                for record in [item for item in ordered if item['task_type'] == action_type][:per_type]]
    return sorted(selected, key=lambda r: (r['data_idx'], r['source_id']))


def source_motion(corpus, record, frames, model, device):
    """Retarget source rotations to the OMOMO subject body without scene copying."""
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
    motion = dict(pose=pose, translation=translation, betas=betas, gender=record['_target_gender'])
    motion['verts'], motion['joints'] = decode_body(motion, model)
    return motion


def ground_source_motion(motion):
    shift = motion['translation'].new_zeros(3)
    shift[1] = -motion['verts'][..., 1].min()
    result = transformed_motion(motion, torch.eye(3, device=shift.device), shift)
    result['ground_translation_m'] = float(shift[1])
    return result


def align_motion(motion, source_frame, target_root, target_heading):
    """Align yaw/XZ while retaining the source's grounded height profile."""
    source_root = motion['joints'][source_frame, 0]
    source_heading = heading(motion['joints'][source_frame])
    delta = torch.atan2((target_heading-source_heading).sin(),
                        (target_heading-source_heading).cos())
    rotation = yaw_matrix(delta.reshape(1))[0]
    translation = target_root-source_root @ rotation.T
    translation[1] = 0
    return transformed_motion(motion, rotation, translation), dict(
        yaw_rad=float(delta), source_root=source_root.tolist(), target_root=target_root.tolist(),
        source_heading_rad=float(source_heading), target_heading_rad=float(target_heading),
        translation_m=translation.tolist(), height_rule='grounded_source_profile')


def geometry_checks(values, thresholds):
    return dict(scene_mean=values['scene_penetration_mean_m'] <= thresholds.scene_mean_penetration_m,
        scene_max=values['scene_penetration_max_m'] <= thresholds.scene_max_penetration_m,
        bounds=values['scene_outside_fraction'] == 0,
        body_object=values['object_penetration_max_m'] <= thresholds.object_max_penetration_m,
        floor=values['floor_penetration_max_m'] <= thresholds.floor_max_penetration_m)


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
    body = geometry_measures(motion['verts'], scene[0], scene[1], object_sdf,
        object_info, object_position[..., None, :], object_rotation)
    object_world = object_vertices @ object_rotation.transpose(-1, -2)+object_position[..., None, :]
    obj_scene = object_scene_passes(object_world, scene[0], scene[1], thresholds)
    return dict(body=body, object_scene=obj_scene,
        body_checks=geometry_checks(body, thresholds),
        passes=all(geometry_checks(body, thresholds).values()) and obj_scene['passes'],
        frame_count=len(motion['joints']))


def object_support(object_world, scene):
    floor_gap = float(object_world[:, 1].min().abs())
    points = object_world[object_world[:, 1].argsort()[:32]]
    signed, outside = signed_query(points, *scene)
    above, below = points.clone(), points.clone()
    above[:, 1] += .04
    below[:, 1] -= .04
    upper, _ = signed_query(above, *scene)
    lower, _ = signed_query(below, *scene)
    on_surface = (signed.abs() <= .05) & (upper > 0) & (lower < 0) & ~outside
    return dict(floor_distance_m=floor_gap, scene_support_points=int(on_surface.sum()),
        supported=floor_gap <= .05 or bool(on_surface.any()), rule='floor_or_directional_scene_support_proxy')


def source_support(motion, record, model, scene):
    foot_gap = float(motion['joints'][:, [7, 8, 10, 11], 1].abs().amin(-1).max())
    result = dict(foot_distance_max_m=foot_gap, feet_supported=foot_gap <= .08,
        seat_required=record['task_type'] == 'static_object_interaction')
    if result['seat_required']:
        indices = range(10) if record['text'] == 'stand up from seat' else range(len(motion['pose'])-10, len(motion['pose']))
        patches = torch.stack([support_patches(motion['verts'][i:i+1], motion['joints'][i:i+1], model) for i in indices])
        result['seat_supported'] = bool(seating_support_mask(patches, *scene).all())
        result['seat_support_points'] = patches[-1].tolist()
    else:
        result['seat_supported'] = True
    result['passes'] = result['feet_supported'] and result['seat_supported']
    return result


def task_object_transform(corpus, record, task, direction, body_motion):
    forward = direction == 'omomo_to_lingo'
    frames = record['task_reference_context_frames'] if forward else record['initial_context_frames']
    positions = torch.as_tensor(np.array(corpus.object_translation[frames]),
        device=body_motion['joints'].device, dtype=torch.float32)
    rotations = torch.as_tensor(np.array(corpus.object_rotation[frames]), device=positions.device, dtype=torch.float32)
    anchor = -1 if forward else 0
    if forward:
        rotation = terminal_goal_rotation(body_motion['joints'][-1, 0], positions[-1], task)
    else:
        delta = positions.new_tensor(task['pelvis_goal'])-positions.new_tensor(task['start_location'])
        initial_rotation = transforms.axis_angle_to_matrix(body_motion['pose'][0, 0])
        initial_heading = transforms.matrix_to_euler_angles(initial_rotation, 'YXZ')[0]
        # Match the dataset's extrinsic zxy heading removal and the native
        # HOSI evaluator's atan2(-delta_z, delta_x) + pi/2 placement.
        task_heading = torch.atan2(-delta[2], delta[0])+math.pi/2
        rotation = yaw_matrix((task_heading-initial_heading).reshape(1))[0]
    target = body_motion['joints'][anchor, 0].clone()
    target[[0, 2]] = target.new_tensor(task['pelvis_goal'] if forward else task['start_location'])[[0, 2]]
    shift = target-body_motion['joints'][anchor, 0] @ rotation.T
    moved = transformed_motion(body_motion, rotation, shift)
    moved['object_translation'] = positions @ rotation.T+shift
    moved['object_rotation'] = rotation @ rotations
    return moved, moved['object_translation'][anchor], moved['object_rotation'][anchor]


def _load_scene(root, scene_name, device):
    sdf_root = root/'data/hosi_test/Scene_sdf'
    return (torch.as_tensor(np.load(sdf_root/(scene_name+'_sdf.npy')), dtype=torch.float32, device=device)[None, None],
        json.loads((sdf_root/(scene_name+'_sdf_info.json')).read_text()))


def static_first_context(motion, frames=10):
    return {k: v[:1].expand((frames,)+v.shape[1:]).clone() if k in MOTION_KEYS else v for k, v in motion.items()}


def save_candidate(output, row, hoi, lingo, direction, body, placed, position, rotation, measurements):
    forward = direction == 'omomo_to_lingo'
    candidate_id = row['task_id']+'-'+direction
    witness_path = 'witnesses/'+candidate_id+'.pt'
    target = static_first_context(body) if not forward else placed
    torch.save({name: {k: v.cpu() if torch.is_tensor(v) else v for k, v in motion.items() if k != 'verts'}
        for name, motion in dict(hoi_context=body, lingo_motion=placed, target_context=target).items()}, output/witness_path)
    scene = row['original_task']['scene_name']
    hoiseg = segment_from_source(hoi, 'hoi', scene, row['original_task']['start_location'],
        row['original_task']['pelvis_goal'], 'source_context' if forward else 'previous_generated_history')
    hoiseg.update(original_conditions=row['original_task'], object_goal=row['original_task']['object_goal'])
    start, goal = placed['joints'][0, 0].tolist(), placed['joints'][-1, 0].tolist()
    start[1] = goal[1] = 0.
    lingoseg = segment_from_source(lingo, 'lingo', scene, start, goal,
        'previous_generated_history' if forward else 'source_context')
    if lingo['task_type'] == 'static_object_interaction':
        lingoseg['contact_target'] = dict(kind='seating_support_surface',
            support_points=measurements['lingo_support']['seat_support_points'], semantic_review='pending')
    edge = transition_edge('hoi' if forward else 'lingo', 'lingo' if forward else 'hoi',
        'release_to_lingo' if forward else 'acquire_contact',
        'fixed_initial_transform' if not forward else 'hold_achieved_supported_object')
    edge.update(bridge_contract='kimodo_acquire_contact' if not forward else 'kimodo_context_bridge',
        target_context_reference=dict(artifact=witness_path, motion_key='target_context', frame_start=0, frame_stop=10,
            fps=30, role='prescribed_static_first_omomo_pose' if not forward else 'placed_lingo_entry'),
        source_context_reference=dict(artifact=witness_path, motion_key='lingo_motion' if not forward else 'hoi_context',
            frame_start=len(placed['pose'])-10 if not forward else 0, frame_stop=len(placed['pose']) if not forward else 10, fps=30))
    return dict(episode_id=candidate_id, direction=direction, original_hosi_task_id=row['task_id'], scene_name=scene,
        body_identity=dict(source_id=hoi['source_id'], gender=hoi['gender'], body_parameters=hoi['body_parameters']),
        segments=[hoiseg, lingoseg] if forward else [lingoseg, hoiseg], transitions=[edge],
        persistent_objects=[dict(object_id=row['original_task']['object_name'],
            geometry='data/test/rest_object_geo/'+row['original_task']['object_name']+'.ply',
            planned_translation=position.tolist(), planned_rotation=rotation.tolist(), support=measurements['object_support'])],
        construction=dict(source_only=True, model_output_used=False, witness=witness_path, measures=measurements),
        evaluation=dict(include_transition_frames=True, successor_state='actual_generated_history', source_motion_is_output_gt=False))


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
    (output/'witnesses').mkdir()
    device, thresholds = cfg.device, cfg.multitask.source_eligibility
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    corpora = {name: SourceCorpus(root, name) for name in ('OMOMO', 'LINGO')}
    tasks, hoi = original_tasks(root, corpora['OMOMO'])
    lingo, exclusions = lingo_sources(corpora['LINGO'], json.loads(Path(cfg.multitask.split_manifest).read_text()))
    audit_source_boundaries(corpora, hoi+lingo, device,
        hand_distance_m=float(thresholds.hand_distance_m))
    records = {r['source_id']: r for r in hoi+lingo}
    pool = select_lingo_pool(lingo, int(thresholds.source_candidates_per_type))
    directions = ['omomo_to_lingo', 'lingo_to_omomo']
    if thresholds.get('allow_grasped_omomo_initial', False):
        directions.append(GRASPED_ENTRY)
    accepted, audit, attempts = [], [], []
    models, objects, templates = {}, {}, {}
    previous_body, previous_scene = None, None
    for direction in directions:
        claimed, count = set(), 0
        for row in tasks:
            task, source = row['original_task'], records[row['source_id']]
            forward = direction == 'omomo_to_lingo'
            boundary = source['source_boundary_audit']['exit' if forward else 'entry']
            checks = direction_guard(direction, boundary, float(thresholds.hand_distance_m),
                float(thresholds.get('grasp_contact_frame_fraction', .5)))
            task_audit = dict(direction=direction, task_id=row['task_id'], source_id=source['source_id'],
                scene_name=task['scene_name'], source_boundary=boundary, state_checks=checks, attempted_pairs=0)
            audit.append(task_audit)
            if not all(checks.values()):
                task_audit['status'] = 'omomo_source_state_ineligible'
                continue
            if count >= int(cfg.multitask.episode_limit) or task['scene_name'] in claimed:
                task_audit['status'] = 'not_attempted_episode_cap' if count >= int(cfg.multitask.episode_limit) else 'not_attempted_scene_selected'
                continue
            if previous_scene != task['scene_name']:
                scene = _load_scene(root, task['scene_name'], device)
                previous_scene = task['scene_name']
            if source['gender'] not in models:
                models[source['gender']] = create_smplx_model(source['gender'], torch.device(device)).eval().requires_grad_(False)
            model = models[source['gender']]
            if previous_body != source['source_id']:
                templates.clear()
                previous_body = source['source_id']
            body_record = dict(source, _target_betas=corpora['OMOMO'].betas[source['source_sequence_idx']],
                _target_gender=source['gender'])
            frames = source['task_reference_context_frames'] if forward else source['initial_context_frames']
            body = source_motion(corpora['OMOMO'], body_record, frames, model, device)
            body, position, rotation = task_object_transform(corpora['OMOMO'], source, task, direction, body)
            name = task['object_name']
            if name not in objects:
                vertices = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(
                    root/'data/test/rest_object_geo'/(name+'.ply')).vertices)), device=device, dtype=torch.float32)
                array, info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', name)
                objects[name] = (vertices, torch.as_tensor(array, dtype=torch.float32, device=device)[None, None], info)
            vertices, obj_sdf, obj_info = objects[name]
            support = object_support(vertices @ rotation.T+position, scene)
            body_geo = full_source_geometry(body, scene, obj_sdf, obj_info, vertices,
                body['object_translation'], body['object_rotation'], thresholds)
            body_feet = float(body['joints'][:, [7, 8, 10, 11], 1].abs().amin(-1).max())
            task_audit.update(object_support=support, omomo_geometry=body_geo, omomo_foot_distance_m=body_feet)
            if not body_geo['passes'] or not support['supported'] or body_feet > .08:
                task_audit['status'] = 'omomo_source_geometry_or_support_failed'
                continue
            target_index = -1 if forward else 0
            target_root, target_heading = body['joints'][target_index, 0], heading(body['joints'][target_index])
            for other in pool:
                if other['source_id'] not in templates:
                    transfer = dict(other, _target_betas=body_record['_target_betas'], _target_gender=source['gender'])
                    interval = list(range(other['source_start_frame'], other['source_stop_frame']))
                    templates[other['source_id']] = ground_source_motion(source_motion(corpora['LINGO'], transfer, interval, model, device))
                template = templates[other['source_id']]
                placed, alignment = align_motion(template, 0 if forward else len(template['pose'])-1, target_root, target_heading)
                lingo_support = source_support(placed, other, model, scene)
                geo = full_source_geometry(placed, scene, obj_sdf, obj_info, vertices, position, rotation, thresholds)
                values = dict(omomo_geometry=body_geo, object_support=support, lingo_support=lingo_support,
                    lingo_geometry=geo, alignment=alignment, ground_translation_m=template['ground_translation_m'],
                    source_contact=boundary)
                passed = geo['passes'] and lingo_support['passes']
                attempt = dict(direction=direction, task_id=row['task_id'], omomo_source_id=source['source_id'],
                    lingo_source_id=other['source_id'], text=other['text'], source_scene=other['source_scene'],
                    frame_interval=[other['source_start_frame'], other['source_stop_frame']], measures=values,
                    status='accepted_source_transition' if passed else 'lingo_geometry_or_support_failed')
                attempts.append(attempt)
                task_audit['attempted_pairs'] += 1
                if passed:
                    episode = save_candidate(output, row, source, other, direction, body, placed, position, rotation, values)
                    validate_episode(episode, records)
                    accepted.append(episode)
                    task_audit.update(status='accepted_source_transition', episode_id=episode['episode_id'])
                    count += 1
                    claimed.add(task['scene_name'])
                    print(json.dumps(dict(candidate=episode['episode_id'], lingo=other['source_id'], text=other['text'])), flush=True)
                    break
            else:
                task_audit['status'] = 'no_feasible_lingo_source'
        print(json.dumps(dict(direction=direction, accepted=count, audited_tasks=len(tasks))), flush=True)
    write_json(output/'transition_audit.json', dict(tasks=audit, attempts=attempts))
    write_json(output/'source_catalog.json', dict(original_tasks=tasks, sources=hoi+lingo, exclusions=exclusions))
    ids = {r['source_id'] for r in pool}
    write_json(output/'pool_selection.json', dict(selected_source_ids=sorted(ids),
        pool_not_attempted_source_ids=sorted(r['source_id'] for r in lingo if r['source_id'] not in ids)))
    needed = {s['source_id'] for e in accepted for s in e['segments']}
    manifest = dict(schema_version=2, artifact_root=str(output), seed=int(cfg.seed),
        original_hosi=dict(task_count=len(tasks), tasks=tasks), sources=[records[s] for s in sorted(needed)],
        episodes=accepted, selection_uses_model_outputs=False, membership_frozen_before_bridge_generation=True, model_samples=0)
    write_json(output/'task_manifest.json', manifest)
    torch.cuda.synchronize(device)
    summary = dict(schema_version=2, subphase='5.5.2a.1', status='completed', seed=int(cfg.seed), git_commit=commit,
        original_hosi_tasks=len(tasks), lingo_sources=len(lingo), excluded_lingo_sources=len(exclusions), lingo_pool_size=len(pool),
        accepted_by_direction={d:sum(e['direction'] == d for e in accepted) for d in directions}, task_audit_rows=len(audit),
        status_counts=dict(Counter(r['status'] for r in audit)), pair_attempts=len(attempts),
        candidates=len(accepted), model_output_used=False, generated_motion_samples=0,
        elapsed_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device), device=str(device),
        git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    write_json(output/'summary.json', summary)
    write_json(output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    print(json.dumps(summary), flush=True)
