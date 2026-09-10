"""Source-only OMOMO/LINGO transition membership and target-scene checks."""

import json
import math
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .inbetween import heading
from .multitask import (SourceCorpus, audit_source_boundaries, lingo_sources, original_tasks,
    human_boundary_measures, transition_edge, validate_episode, write_json)
from .multitask_geometry import (geometry_measures, seating_support_mask, segment_from_source,
    signed_query, support_patches, terminal_goal_rotation, transformed_motion)
from .surface_edit import decode_body, load_object_sdf, yaw_matrix

GRASPED_ENTRY = 'lingo_to_omomo_grasped_entry'
MOTION_KEYS = ('pose', 'translation', 'joints', 'verts', 'object_translation', 'object_rotation')

DATASET_ACTIONS = {
    'walk': 'locomotion',
    'maintains stand posture': 'standing',
    'maintains sit posture': 'seated',
    'stand up from seat': 'seated',
    **{f'sit down on {seat}': 'seated'
       for seat in ('chair', 'office chair', 'sofa', 'couch', 'bed', 'toilet', 'bench')},
}


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


def select_lingo_pool(records, per_type, balance_families=False, max_duration_s=None):
    """Select a deterministic source-ordered pool without model scores."""
    if max_duration_s is not None:
        records = [r for r in records if r['source_duration_s'] <= max_duration_s]
    ordered = sorted(records, key=lambda r: (r['data_idx'], r['source_id']))
    if balance_families:
        selected = []
        for action_type in sorted({r['task_type'] for r in ordered}):
            groups = defaultdict(list)
            for record in ordered:
                if record['task_type'] == action_type:
                    groups[(record['source_scene_family'], record['text'])].append(record)
            count, depth = 0, 0
            while count < per_type and any(len(group) > depth for group in groups.values()):
                for key in sorted(groups):
                    if len(groups[key]) > depth and count < per_type:
                        selected.append(groups[key][depth])
                        count += 1
                depth += 1
        return selected
    selected = [record for action_type in sorted({record['task_type'] for record in ordered})
                for record in [item for item in ordered if item['task_type'] == action_type][:per_type]]
    return sorted(selected, key=lambda r: (r['data_idx'], r['source_id']))


def ordered_pair_pool(pool, source_counts, family_counts):
    """Alternate action types, preferring sources not used by earlier pairs."""
    groups = []
    for kind in sorted({r['task_type'] for r in pool}):
        groups.append(sorted((r for r in pool if r['task_type'] == kind), key=lambda r:
            (source_counts[r['source_id']], family_counts[r['source_scene_family']], r['data_idx'])))
    return [group[depth] for depth in range(max(map(len, groups), default=0))
            for group in groups if depth < len(group)]


def select_execution_pilot(episodes, records, limit):
    pool = [e for e in episodes if e['direction'] == GRASPED_ENTRY]
    selected, scenes, sources, families = [], Counter(), Counter(), Counter()
    while pool and len(selected) < limit:
        for name in sorted({e['persistent_objects'][0]['object_id'] for e in pool}):
            choices = [e for e in pool if e['persistent_objects'][0]['object_id'] == name]
            def key(episode):
                source = records[episode['segments'][0]['source_id']]
                return (scenes[episode['scene_name']], sources[source['source_id']],
                    families[source['source_scene_family']], episode['episode_id'])
            episode = min(choices, key=key)
            selected.append(episode['episode_id'])
            source = records[episode['segments'][0]['source_id']]
            scenes[episode['scene_name']] += 1
            sources[source['source_id']] += 1
            families[source['source_scene_family']] += 1
            pool.remove(episode)
            if len(selected) == limit:
                break
    return selected


def source_coverage(episodes, records):
    lingo = [records[s['source_id']] for e in episodes for s in e['segments'] if s['source_dataset'] == 'LINGO']
    hoi = [s['source_id'] for e in episodes for s in e['segments'] if s['source_dataset'] == 'OMOMO']
    durations = [r['source_duration_s'] for r in lingo]
    return dict(target_scenes=len({e['scene_name'] for e in episodes}),
        original_tasks=len({e['original_hosi_task_id'] for e in episodes}),
        unique_omomo_sources=len(set(hoi)), unique_lingo_sources=len({r['source_id'] for r in lingo}),
        lingo_scene_families=len({r['source_scene_family'] for r in lingo}),
        lingo_source_reuse=dict(Counter(r['source_id'] for r in lingo)),
        lingo_texts=dict(Counter(r['text'] for r in lingo)),
        objects=dict(Counter(e['persistent_objects'][0]['object_id'] for e in episodes)),
        lingo_duration_s=dict(min=min(durations), median=float(np.median(durations)), max=max(durations)) if durations else None)


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


def save_candidate(output, row, hoi, lingo, direction, body, placed, position, rotation, measurements, qualify_source=False):
    forward = direction == 'omomo_to_lingo'
    candidate_id = row['task_id']+'-'+direction
    if qualify_source:
        candidate_id += '-'+lingo['source_id']
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
        if lingo['text'].startswith('sit down'):
            lingoseg['scene_goal'] = placed['joints'][-1, 0].tolist()
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
    expansion = thresholds.get('expand_coverage', False)
    pool = select_lingo_pool(lingo, int(thresholds.source_candidates_per_type), expansion,
        thresholds.get('maximum_source_duration_s'))
    directions = ['omomo_to_lingo', 'lingo_to_omomo']
    if thresholds.get('allow_grasped_omomo_initial', False):
        directions.append(GRASPED_ENTRY)
    accepted, audit, attempts = [], [], []
    if expansion:
        inherited_path = Path(thresholds.inherited_manifest)
        inherited = json.loads(inherited_path.read_text())
        if inherited['original_hosi']['tasks'] != tasks:
            raise ValueError('expanded inventory changed an original HOSI task')
        for episode in inherited['episodes']:
            shutil.copyfile(Path(inherited['artifact_root'])/episode['construction']['witness'],
                output/episode['construction']['witness'])
            episode['construction']['inherited_manifest'] = str(inherited_path)
            accepted.append(episode)
    inherited_count = len(accepted)
    models, objects, templates = {}, {}, {}
    previous_body, previous_scene = None, None
    for direction in directions:
        prior = [e for e in accepted if e['direction'] == direction]
        scene_counts = Counter(e['scene_name'] for e in prior)
        source_counts = Counter(s['source_id'] for e in prior for s in e['segments'] if s['source_dataset'] == 'LINGO')
        family_counts = Counter(records[s['source_id']]['source_scene_family'] for e in prior for s in e['segments'] if s['source_dataset'] == 'LINGO')
        count = len(prior)
        for row in tasks:
            task, source = row['original_task'], records[row['source_id']]
            forward = direction == 'omomo_to_lingo'
            boundary = source['source_boundary_audit']['exit' if forward else 'entry']
            checks = direction_guard(direction, boundary, float(thresholds.hand_distance_m),
                float(thresholds.get('grasp_contact_frame_fraction', .5)))
            existing = [e for e in accepted if e['direction'] == direction and e['original_hosi_task_id'] == row['task_id']]
            selected_ids = [e['episode_id'] for e in existing]
            selected_sources = {s['source_id'] for e in existing for s in e['segments'] if s['source_dataset'] == 'LINGO'}
            task_audit = dict(direction=direction, task_id=row['task_id'], source_id=source['source_id'],
                scene_name=task['scene_name'], source_boundary=boundary, state_checks=checks, attempted_pairs=0,
                accepted_episode_ids=selected_ids, inherited_episode_ids=list(selected_ids))
            audit.append(task_audit)
            if not all(checks.values()):
                task_audit['status'] = 'omomo_source_state_ineligible'
                continue
            if count >= int(cfg.multitask.episode_limit) or scene_counts[task['scene_name']] >= int(thresholds.get('candidates_per_scene', 1)):
                task_audit['search_stop'] = 'not_attempted_episode_cap' if count >= int(cfg.multitask.episode_limit) else 'not_attempted_scene_selected'
                task_audit['status'] = 'accepted_source_transition' if selected_ids else task_audit['search_stop']
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
            search_pool = ordered_pair_pool(pool, source_counts, family_counts) if expansion else pool
            task_audit['unattempted_sources'] = []
            stop = None
            for other in search_pool:
                reason = stop or ('already_selected' if other['source_id'] in selected_sources else
                    'source_reuse_limit' if expansion and source_counts[other['source_id']] >= int(thresholds.source_reuse_limit) else None)
                if reason:
                    task_audit['unattempted_sources'].append(dict(source_id=other['source_id'], reason=reason))
                    continue
                if other['source_id'] not in templates:
                    transfer = dict(other, _target_betas=body_record['_target_betas'], _target_gender=source['gender'])
                    interval = list(range(other['source_start_frame'], other['source_stop_frame']))
                    templates[other['source_id']] = ground_source_motion(source_motion(corpora['LINGO'], transfer, interval, model, device))
                template = templates[other['source_id']]
                placed, alignment = align_motion(template, 0 if forward else len(template['pose'])-1, target_root, target_heading)
                lingo_support = source_support(placed, other, model, scene)
                geo = (full_source_geometry(placed, scene, obj_sdf, obj_info, vertices, position, rotation, thresholds)
                    if lingo_support['passes'] or not expansion else None)
                values = dict(omomo_geometry=body_geo, object_support=support, lingo_support=lingo_support,
                    lingo_geometry=geo, alignment=alignment, ground_translation_m=template['ground_translation_m'],
                    source_contact=boundary)
                passed = lingo_support['passes'] and geo['passes']
                attempt = dict(direction=direction, task_id=row['task_id'], omomo_source_id=source['source_id'],
                    lingo_source_id=other['source_id'], text=other['text'], source_scene=other['source_scene'],
                    frame_interval=[other['source_start_frame'], other['source_stop_frame']], measures=values,
                    status='accepted_source_transition' if passed else 'lingo_geometry_or_support_failed',
                    full_geometry_evaluated=geo is not None)
                attempts.append(attempt)
                task_audit['attempted_pairs'] += 1
                if passed:
                    episode = save_candidate(output, row, source, other, direction, body, placed, position, rotation, values, expansion)
                    validate_episode(episode, records)
                    accepted.append(episode)
                    selected_ids.append(episode['episode_id'])
                    selected_sources.add(other['source_id'])
                    task_audit.update(status='accepted_source_transition', episode_id=episode['episode_id'])
                    count += 1
                    scene_counts[task['scene_name']] += 1
                    source_counts[other['source_id']] += 1
                    family_counts[other['source_scene_family']] += 1
                    print(json.dumps(dict(candidate=episode['episode_id'], lingo=other['source_id'], text=other['text'])), flush=True)
                if len(selected_ids) >= int(thresholds.get('candidates_per_task', 1)):
                    stop = 'task_candidate_limit'
                elif count >= int(cfg.multitask.episode_limit):
                    stop = 'direction_candidate_limit'
                elif scene_counts[task['scene_name']] >= int(thresholds.get('candidates_per_scene', 1)):
                    stop = 'scene_candidate_limit'
                elif task_audit['attempted_pairs'] >= int(thresholds.get('pair_attempt_limit', len(pool))):
                    stop = 'pair_attempt_limit'
            task_audit['search_stop'] = stop or 'pool_exhausted'
            task_audit['status'] = 'accepted_source_transition' if selected_ids else 'no_feasible_lingo_source'
        print(json.dumps(dict(direction=direction, accepted=count, audited_tasks=len(tasks))), flush=True)
    write_json(output/'transition_audit.json', dict(tasks=audit, attempts=attempts))
    write_json(output/'source_catalog.json', dict(original_tasks=tasks, sources=hoi+lingo, exclusions=exclusions))
    ids = {r['source_id'] for r in pool}
    write_json(output/'pool_selection.json', dict(selected_source_ids=sorted(ids),
        pool_not_attempted_source_ids=sorted(r['source_id'] for r in lingo if r['source_id'] not in ids),
        balance_scene_family_and_text=bool(expansion), maximum_source_duration_s=thresholds.get('maximum_source_duration_s'),
        duration_excluded_source_ids=[r['source_id'] for r in lingo
            if thresholds.get('maximum_source_duration_s') is not None and r['source_duration_s'] > thresholds.maximum_source_duration_s]))
    needed = {s['source_id'] for e in accepted for s in e['segments']}
    manifest = dict(schema_version=2, artifact_root=str(output), seed=int(cfg.seed),
        original_hosi=dict(task_count=len(tasks), tasks=tasks), sources=[records[s] for s in sorted(needed)],
        episodes=accepted, selection_uses_model_outputs=False, membership_frozen_before_bridge_generation=True, model_samples=0)
    write_json(output/'task_manifest.json', manifest)
    if expansion:
        selected = select_execution_pilot(accepted, records, int(cfg.multitask.execution.episode_limit))
        write_json(output/'execution_selection.json', dict(source_manifest=str(output/'task_manifest.json'),
            episode_ids=selected, unevaluated_episode_ids=[e['episode_id'] for e in accepted if e['episode_id'] not in selected],
            selection_uses_model_outputs=False, candidate_count=len(accepted)))
    torch.cuda.synchronize(device)
    summary = dict(schema_version=2, subphase=str(cfg.multitask.get('subphase', '5.5.2a.1')), status='completed', seed=int(cfg.seed), git_commit=commit,
        original_hosi_tasks=len(tasks), lingo_sources=len(lingo), excluded_lingo_sources=len(exclusions), lingo_pool_size=len(pool),
        accepted_by_direction={d:sum(e['direction'] == d for e in accepted) for d in directions}, task_audit_rows=len(audit),
        status_counts=dict(Counter(r['status'] for r in audit)), pair_attempts=len(attempts),
        candidates=len(accepted), inherited_candidates=inherited_count, coverage=source_coverage(accepted, records),
        coverage_by_direction={d:source_coverage([e for e in accepted if e['direction'] == d], records) for d in directions},
        model_output_used=False, generated_motion_samples=0,
        elapsed_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device), device=str(device),
        git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    write_json(output/'summary.json', summary)
    write_json(output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    print(json.dumps(summary), flush=True)


def labelled_spans(records):
    """Join contiguous safe annotations; every retained frame has a label."""
    spans = []
    for row in records:
        if row['action_type'] is None:
            continue
        if (spans and spans[-1]['source_stop_frame'] == row['start']
                and spans[-1]['source_scene'] == row['scene']):
            spans[-1]['source_stop_frame'] = row['stop']
            spans[-1]['actions'].append(row)
        else:
            spans.append(dict(source_dataset='LINGO', source_partition='test',
                source_scene=row['scene'], source_scene_family=row['family'],
                source_start_frame=row['start'], source_stop_frame=row['stop'],
                source_sequence_idx=row['sequence'], data_idx=row['data_idx'],
                actions=[row]))
    for span in spans:
        span['source_id'] = f'lingo-span-{span["source_start_frame"]}-{span["source_stop_frame"]}'
        span['text'] = '; then '.join(a['text'] for a in span['actions'])
        span['source_fps'] = 30
    return spans


def dataset_lingo_catalog(corpus, split):
    sequences, indices = np.unique(corpus.language['ori_sequence_idx'], return_index=True)
    indices = dict(zip(sequences.tolist(), indices.tolist()))
    test_scenes = set(split['test']['scenes'])
    annotations, exclusions = [], []
    for seq in range(len(corpus.starts)//2):
        start, stop = int(corpus.starts[seq]), int(corpus.ends[seq])
        scene = str(corpus.scenes[start])
        if scene not in test_scenes:
            continue
        index = indices.get(seq)
        text = str(corpus.language['text'][index][0]) if index is not None else 'unlabelled'
        kind = DATASET_ACTIONS.get(text)
        row = dict(sequence=seq, start=start, stop=stop, scene=scene,
            family=split['scene_to_family'][scene], data_idx=index, text=text, action_type=kind)
        annotations.append(row)
        if kind is None:
            exclusions.append(dict(row, reason='action_requires_prop_or_is_outside_static_allowlist'))
    return labelled_spans(annotations), exclusions


def standing_frames(joints, settings):
    """Upright supported stance; hand/object contact has no role in posture."""
    root = joints[:, 0]
    spine = joints[:, 12]-root
    tilt = torch.acos((spine[:, 1]/spine.norm(dim=-1)).clamp(-1, 1))*180/torch.pi
    thigh = joints[:, [1, 2]]-joints[:, [4, 5]]
    shin = joints[:, [7, 8]]-joints[:, [4, 5]]
    angle = torch.acos(((thigh*shin).sum(-1)/(thigh.norm(dim=-1)*shin.norm(dim=-1))).clamp(-1, 1))
    flexion = 180-angle*180/torch.pi
    speed = torch.cat((root.new_zeros(1), (root[1:]-root[:-1]).norm(dim=-1)*30))
    foot_gap = joints[:, [7, 8, 10, 11], 1].abs().amin(-1)
    mask = ((tilt <= settings.standing_tilt_deg) & (root[:, 1] >= settings.standing_pelvis_height_m)
        & (flexion.amax(-1) <= settings.standing_knee_flexion_deg)
        & (foot_gap <= settings.foot_support_m))
    return mask, dict(tilt=tilt, knee_flexion=flexion.amax(-1), root_speed=speed, foot_gap=foot_gap)


def standing_context(joints, settings):
    mask, values = standing_frames(joints, settings)
    return dict(passes=bool(mask.all() and (values['root_speed'] <= settings.standing_speed_m_s).all()), frames=len(joints),
        torso_tilt_max_deg=float(values['tilt'].max()),
        knee_flexion_max_deg=float(values['knee_flexion'].max()),
        pelvis_height_min_m=float(joints[:, 0, 1].min()),
        root_speed_max_m_s=float(values['root_speed'].max()),
        foot_distance_max_m=float(values['foot_gap'].max()))


def source_cut_options(span, joints, direction, settings):
    """Search all frames, then spread the fixed cut budget over distinct times."""
    width = int(settings.standing_context_frames)
    if len(joints) < settings.minimum_frames:
        return []
    standing, values = standing_frames(joints, settings)
    # A cut has no velocity edge to a discarded predecessor frame.
    windows = (standing.unfold(0, width, 1).all(-1)
        & (values['root_speed'].unfold(0, width, 1)[:, 1:] <= settings.standing_speed_m_s).all(-1))
    walking = torch.zeros(len(joints), dtype=torch.bool, device=joints.device)
    for action in span['actions']:
        if action['action_type'] == 'locomotion':
            walking[action['start']-span['source_start_frame']:action['stop']-span['source_start_frame']] = True
    walk_step = (joints[1:, 0]-joints[:-1, 0]).norm(dim=-1)*(walking[1:] & walking[:-1])
    distance = torch.cat((joints.new_zeros(1), walk_step.cumsum(0)))
    choices = []
    for context in torch.where(windows)[0].tolist():
        if direction == 'omomo_to_lingo':
            start, stop = context, min(len(joints), context+int(settings.maximum_frames))
        else:
            stop = context+width
            start = max(0, stop-int(settings.maximum_frames))
        if stop-start < settings.minimum_frames:
            continue
        travel = float(distance[stop-1]-distance[start])
        if travel < settings.minimum_walk_distance_m:
            continue
        raw_start, raw_stop = span['source_start_frame']+start, span['source_start_frame']+stop
        actions = [dict(a, start=max(a['start'], raw_start), stop=min(a['stop'], raw_stop))
            for a in span['actions'] if a['start'] < raw_stop and a['stop'] > raw_start]
        has_seat = any(a['action_type'] == 'seated' and a['stop']-a['start'] >= 10 for a in actions)
        choices.append(dict(start=start, stop=stop, source_frame_interval=[raw_start, raw_stop], actions=actions,
            has_static=has_seat, walk_distance_m=travel, standing_score=float(values['root_speed'][context+1:context+width].mean()),
            cut_frame=raw_start if direction == 'omomo_to_lingo' else raw_stop-1,
            internal_cut=start > 0 if direction == 'omomo_to_lingo' else stop < len(joints)))
    choices.sort(key=lambda c: (not c['has_static'], c['standing_score'], -(c['stop']-c['start']), c['cut_frame']))
    selected = []
    for choice in choices:
        if all(abs(choice['cut_frame']-other['cut_frame']) >= width for other in selected):
            selected.append(choice)
        if len(selected) == settings.cut_options_per_source:
            break
    return selected


def slice_source(motion, start, stop):
    return {key: value[start:stop] if key in MOTION_KEYS else value for key, value in motion.items()}


def chunked_source_motion(corpus, record, start, stop, model, device):
    pieces = [source_motion(corpus, record, list(range(first, min(first+128, stop))), model, device)
        for first in range(start, stop, 128)]
    return {key: torch.cat([piece[key] for piece in pieces]) if key in MOTION_KEYS else value
        for key, value in pieces[0].items()}


def dataset_support(motion, actions, model, scene, settings):
    gap = motion['joints'][:, [7, 8, 10, 11], 1].abs().amin(-1)
    result = dict(feet_supported=bool((gap <= settings.foot_support_m).all()),
        foot_distance_max_m=float(gap.max()), seated_intervals=[], static_interaction=False)
    for action in actions:
        if action['action_type'] != 'seated':
            continue
        start, stop = action['local_start'], action['local_stop']
        height = motion['joints'][start:stop, 0, 1]
        seated = torch.where((height < .65) & (height <= height.min()+.05))[0]+start
        if len(seated) < 10:
            continue
        patches = torch.stack([support_patches(motion['verts'][i:i+1], motion['joints'][i:i+1], model)
            for i in seated.tolist()])
        supported = seating_support_mask(patches, *scene)
        result['seated_intervals'].append(dict(text=action['text'], frame_indices=seated.tolist(),
            supported=bool(supported.all()), support_fraction=float(supported.float().mean()),
            support_points=patches[len(patches)//2].tolist()))
    result['static_interaction'] = bool(result['seated_intervals'])
    result['passes'] = result['feet_supported'] and all(s['supported'] for s in result['seated_intervals'])
    return result


def place_complete_omomo(corpus, source, motion, task, vertices, offset):
    """Place a complete source once; derive new goals from the actual result."""
    initial = transforms.axis_angle_to_matrix(motion['pose'][0, 0])
    initial_heading = transforms.matrix_to_euler_angles(initial, 'YXZ')[0]
    delta = motion['translation'].new_tensor(task['pelvis_goal'])-motion['translation'].new_tensor(task['start_location'])
    yaw = torch.atan2(-delta[2], delta[0])+math.pi/2-initial_heading
    rotation = yaw_matrix(yaw.reshape(1))[0]
    object_world = vertices @ motion['object_rotation'].transpose(-1, -2)+motion['object_translation'][:, None]
    shift = torch.zeros(3, device=rotation.device)
    shift[1] = -torch.minimum(motion['verts'][..., 1].min(), object_world[..., 1].min())
    shift[[0, 2]] = (shift.new_tensor(task['start_location'])+shift.new_tensor([offset[0], 0, offset[1]])
        -motion['joints'][0, 0] @ rotation.T)[[0, 2]]
    placed = transformed_motion(motion, rotation, shift)
    placed['object_translation'] = motion['object_translation'] @ rotation.T+shift
    placed['object_rotation'] = rotation @ motion['object_rotation']
    return placed, dict(yaw_rad=float(yaw), translation_m=shift.tolist(), anchor_offset_m=list(offset),
        source_frame_interval=[source['source_start_frame'], source['source_stop_frame']])


def balanced_tasks(tasks):
    groups = defaultdict(list)
    for row in tasks:
        groups[row['original_task']['scene_name']].append(row)
    return [group[i] for i in range(max(map(len, groups.values())))
        for _, group in sorted(groups.items()) if i < len(group)]


def pair_options(spans, options, usage, family_usage, settings):
    groups = {}
    for static in (True, False):
        group = [span for span in spans if any(c['has_static'] == static for c in options[span['source_id']])]
        groups[static] = sorted(group, key=lambda s: (usage[s['source_id']], family_usage[s['source_scene_family']], s['data_idx']))
    for attempt in range(int(settings.pair_attempt_limit)):
        static = attempt % 2 == 0
        group = groups[static] or groups[not static]
        if not group:
            return
        ordinal = attempt//2
        span = group[ordinal % len(group)]
        choices = [c for c in options[span['source_id']] if c['has_static'] == static] or options[span['source_id']]
        cycle = ordinal//len(group)
        cut = choices[cycle % len(choices)]
        yaw = settings.yaw_offsets_deg[(ordinal//8+cycle) % len(settings.yaw_offsets_deg)]
        offset = settings.root_offsets_m[(ordinal//4+cycle) % len(settings.root_offsets_m)]
        yield span, cut, float(yaw), list(offset)


def dataset_candidate(output, row, source, span, cut, direction, body, lingo, metrics):
    first, second = (body, lingo) if direction == 'omomo_to_lingo' else (lingo, body)
    identifier = f'{row["task_id"]}-{direction}-{span["source_id"]}-{cut["cut_frame"]}'
    witness = 'witnesses/'+identifier+'.pt'
    torch.save({name: {k:v.cpu() if torch.is_tensor(v) else v for k,v in motion.items() if k != 'verts'}
        for name,motion in dict(first=first, second=second).items()}, output/witness)
    hoi_interval = [source['source_start_frame'], source['source_stop_frame']]
    common = dict(scene_name=row['original_task']['scene_name'])
    hoiseg = dict(common, segment_id='omomo', source_dataset='OMOMO', source_id=source['source_id'],
        task_type='hoi', text=source['text'], source_frame_interval=hoi_interval,
        pelvis_goal=body['joints'][-1, 0].tolist(), object_goal=body['object_translation'][-1].tolist(),
        object_rotation_goal=body['object_rotation'][-1].tolist(), frame_count=len(body['pose']))
    static = metrics['lingo_support']['static_interaction']
    lingoseg = dict(common, segment_id='lingo', source_dataset='LINGO', source_id=span['source_id'],
        task_type='locomotion_static_interaction' if static else 'locomotion',
        text='; then '.join(a['text'] for a in cut['actions']), actions=cut['actions'],
        source_frame_interval=cut['source_frame_interval'], frame_count=len(lingo['pose']),
        pelvis_goal=lingo['joints'][-1, 0].tolist(), object_goal=None,
        contact_targets=metrics['lingo_support']['seated_intervals'])
    segments = [hoiseg, lingoseg] if direction == 'omomo_to_lingo' else [lingoseg, hoiseg]
    join = -1 if direction == 'omomo_to_lingo' else 0
    return dict(episode_id=identifier, direction=direction, original_hosi_task_id=row['task_id'],
        **common, segments=segments, body_identity=dict(gender=body['gender'], betas=body['betas'].tolist()),
        persistent_objects=[dict(object_id=row['original_task']['object_name'],
            geometry='data/test/rest_object_geo/'+row['original_task']['object_name']+'.ply',
            planned_translation=body['object_translation'][join].tolist(),
            planned_rotation=body['object_rotation'][join].tolist(), support=metrics['object_support'])],
        construction=dict(witness=witness, source_only=True, expert_samples=0, measurements=metrics,
            lingo_source_scene=span['source_scene'], lingo_source_scene_family=span['source_scene_family'],
            internal_lingo_cut=cut['internal_cut'], lingo_cut_frame=cut['cut_frame']),
        transition=dict(frames=41, context_frames=10, posture='standing', hands_may_contact_object=True),
        frame_count=len(first['pose'])+41+len(second['pose']))


@torch.no_grad()
def run_dataset_sources(cfg):
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered benchmark construction requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = Path(cfg.multitask.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    (output/'witnesses').mkdir()
    settings, thresholds, device = cfg.multitask.dataset, cfg.multitask.source_eligibility, cfg.device
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    corpora = {name:SourceCorpus(root, name) for name in ('OMOMO', 'LINGO')}
    tasks, originals = original_tasks(root, corpora['OMOMO'])
    records = {r['source_id']:r for r in originals}
    spans, exclusions = dataset_lingo_catalog(corpora['LINGO'], json.loads(Path(cfg.multitask.split_manifest).read_text()))
    directions = ['omomo_to_lingo', 'lingo_to_omomo']
    options = {direction:{} for direction in directions}
    for span in spans:
        joints = torch.as_tensor(np.array(corpora['LINGO'].joints[span['source_start_frame']:span['source_stop_frame']]), device=device, dtype=torch.float32)
        for direction in directions:
            options[direction][span['source_id']] = source_cut_options(span, joints, direction, settings)
    write_json(output/'source_catalog.json', dict(spans=spans, exclusions=exclusions, cut_options=options))
    models, objects = {}, {}
    accepted, task_audit, pair_audit = [], [], []
    usage, families, counts = Counter(), Counter(), Counter()
    for row in balanced_tasks(tasks):
        task, source = row['original_task'], dict(records[row['source_id']])
        seq = source['source_sequence_idx']
        source.update(source_start_frame=int(corpora['OMOMO'].starts[seq]), source_stop_frame=int(corpora['OMOMO'].ends[seq]))
        if source['gender'] not in models:
            models[source['gender']] = create_smplx_model(source['gender'], torch.device(device)).eval().requires_grad_(False)
        model = models[source['gender']]
        body_record = dict(source, _target_betas=corpora['OMOMO'].betas[seq], _target_gender=source['gender'])
        body_raw = chunked_source_motion(corpora['OMOMO'], body_record,
            source['source_start_frame'], source['source_stop_frame'], model, device)
        frame_slice = slice(source['source_start_frame'], source['source_stop_frame'])
        body_raw['object_translation'] = torch.as_tensor(np.array(corpora['OMOMO'].object_translation[frame_slice]), device=device, dtype=torch.float32)
        body_raw['object_rotation'] = torch.as_tensor(np.array(corpora['OMOMO'].object_rotation[frame_slice]), device=device, dtype=torch.float32)
        name = task['object_name']
        if name not in objects:
            rest = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(root/'data/test/rest_object_geo'/(name+'.ply')).vertices)), device=device, dtype=torch.float32)
            array, info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', name)
            objects[name] = (rest, torch.as_tensor(array, device=device, dtype=torch.float32)[None, None], info)
        rest, obj_sdf, obj_info = objects[name]
        scene = _load_scene(root, task['scene_name'], device)
        placement_audit, placed = [], None
        for offset in settings.omomo_anchor_offsets_m:
            body, placement = place_complete_omomo(corpora['OMOMO'], source, body_raw, task, rest, offset)
            geo = full_source_geometry(body, scene, obj_sdf, obj_info, rest,
                body['object_translation'], body['object_rotation'], thresholds)
            support = dataset_support(body, [], model, scene, settings)
            placement_audit.append(dict(placement=placement, geometry=geo, support=support))
            if geo['passes'] and support['passes']:
                placed = (body, placement, geo, support)
                break
        for direction in directions:
            audit = dict(task_id=row['task_id'], direction=direction, source_id=source['source_id'],
                scene_name=task['scene_name'], omomo_frame_interval=[source['source_start_frame'], source['source_stop_frame']],
                omomo_placements=placement_audit, selected=[], pair_attempts=0)
            task_audit.append(audit)
            if placed is None:
                audit['status'] = 'complete_omomo_geometry_or_support_failed'
                continue
            body, placement, body_geo, body_support = placed
            forward = direction == 'omomo_to_lingo'
            context = body['joints'][-10:] if forward else body['joints'][:10]
            boundary = standing_context(context, settings)
            join = -1 if forward else 0
            position, rotation = body['object_translation'][join], body['object_rotation'][join]
            support = object_support(rest @ rotation.T+position, scene)
            audit.update(standing=boundary, object_support=support)
            if not boundary['passes'] or not support['supported']:
                audit['status'] = 'omomo_join_stance_or_object_support_failed'
                continue
            if len(accepted) >= settings.candidate_limit or counts[direction] >= settings.candidate_limit//2:
                audit['status'] = 'not_attempted_candidate_budget'
                continue
            target_root, target_heading = body['joints'][join, 0], heading(body['joints'][join])
            templates = {}
            chosen_spans = set()
            for span, cut, yaw, offset in pair_options(spans, options[direction], usage, families, settings):
                if span['source_id'] in chosen_spans:
                    continue
                cache_key = (span['source_id'], cut['start'], cut['stop'])
                if cache_key not in templates:
                    # Limit the live body-surface cache; source intervals remain immutable.
                    if len(templates) == 4:
                        templates.pop(next(iter(templates)))
                    transfer = dict(span, _target_betas=body_record['_target_betas'], _target_gender=source['gender'])
                    templates[cache_key] = ground_source_motion(chunked_source_motion(corpora['LINGO'], transfer,
                        cut['source_frame_interval'][0], cut['source_frame_interval'][1], model, device))
                template = templates[cache_key]
                target = target_root+target_root.new_tensor([offset[0], 0, offset[1]])
                lingo, alignment = align_motion(template, 0 if forward else -1, target, target_heading+math.radians(yaw))
                lingo['object_translation'] = position.expand(len(lingo['pose']), -1).clone()
                lingo['object_rotation'] = rotation.expand(len(lingo['pose']), -1, -1).clone()
                action_intervals = [dict(a, local_start=a['start']-cut['source_frame_interval'][0],
                    local_stop=a['stop']-cut['source_frame_interval'][0]) for a in cut['actions']]
                lingo_support = dataset_support(lingo, action_intervals, model, scene, settings)
                stance = standing_context(lingo['joints'][:10] if forward else lingo['joints'][-10:], settings)
                lingo_geo = full_source_geometry(lingo, scene, obj_sdf, obj_info, rest, position, rotation, thresholds)
                passed = lingo_support['passes'] and stance['passes'] and lingo_geo['passes']
                if cut['has_static'] and not lingo_support['static_interaction']:
                    passed = False
                measurements = dict(omomo_geometry=body_geo, omomo_support=body_support, omomo_placement=placement,
                    omomo_standing=boundary, object_support=support, lingo_geometry=lingo_geo,
                    lingo_support=lingo_support, lingo_standing=stance, alignment=alignment,
                    ground_translation_m=template['ground_translation_m'])
                pair_audit.append(dict(task_id=row['task_id'], direction=direction, source_id=span['source_id'],
                    source_frame_interval=cut['source_frame_interval'], internal_cut=cut['internal_cut'],
                    has_static=cut['has_static'], passed=passed, measurements=measurements))
                audit['pair_attempts'] += 1
                if passed:
                    episode = dataset_candidate(output, row, source, span, cut, direction, body, lingo, measurements)
                    accepted.append(episode)
                    audit['selected'].append(episode['episode_id'])
                    chosen_spans.add(span['source_id'])
                    usage[span['source_id']] += 1
                    families[span['source_scene_family']] += 1
                    counts[direction] += 1
                    print(json.dumps(dict(candidate=episode['episode_id'], static=lingo_support['static_interaction'], count=len(accepted))), flush=True)
                if (len(audit['selected']) >= settings.candidates_per_task_direction or len(accepted) >= settings.candidate_limit
                        or counts[direction] >= settings.candidate_limit//2):
                    break
            audit['status'] = 'source_pair_selected' if audit['selected'] else 'no_source_pair_in_fixed_search'
        if len(task_audit) % 40 == 0:
            print(json.dumps(dict(audited_tasks=len(task_audit)//2, candidates=len(accepted))), flush=True)
    write_json(output/'construction_audit.json', dict(tasks=task_audit, pairs=pair_audit))
    manifest = dict(schema_version=3, artifact_root=str(output), seed=int(cfg.seed),
        original_hosi=dict(task_count=len(tasks), tasks=tasks), episodes=accepted,
        candidate_selection='dataset_source_geometry_and_standing', expert_samples=0,
        bridge_selection='one_fixed_kimodo_attempt_per_pair_then_next_pair', target_episodes=int(settings.target_episodes))
    write_json(output/'candidates.json', manifest)
    torch.cuda.synchronize(device)
    summary = dict(status='completed', subphase='5.6.1', git_commit=commit, seed=int(cfg.seed),
        original_tasks=len(tasks), source_spans=len(spans), excluded_action_intervals=len(exclusions),
        source_candidates=len(accepted), candidates_by_direction=dict(counts),
        static_candidates=sum(any(s['task_type'] == 'locomotion_static_interaction' for s in e['segments']) for e in accepted),
        internal_cut_candidates=sum(e['construction']['internal_lingo_cut'] for e in accepted),
        original_task_coverage=len({e['original_hosi_task_id'] for e in accepted}),
        scene_coverage=len({e['scene_name'] for e in accepted}), lingo_span_coverage=len(usage),
        task_statuses=dict(Counter(a['status'] for a in task_audit)), pair_attempts=len(pair_audit),
        elapsed_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(device),
        device=str(device), expert_samples=0,
        git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
