"""Place source interaction witnesses in HOSI scenes and build explicit chains."""

import json
import math
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .inbetween import heading
from .multitask import transition_edge, validate_episode, write_json
from .surface_edit import decode_body, load_object_sdf, yaw_matrix


def signed_query(points, sdf, info):
    extent = float(max(info['extents']))
    normalized = (points-points.new_tensor(info['centroid']))/(extent/2)
    signed = torch.nn.functional.grid_sample(sdf,
        normalized[..., [2, 1, 0]].reshape(1, -1, 1, 1, 3),
        padding_mode='border', align_corners=True).reshape(points.shape[:-1])*(extent/2)
    return signed, (normalized.abs() > 1).any(-1)


def geometry_measures(vertices, sdf, info, object_sdf, object_info, object_position, object_rotation):
    scene, outside = signed_query(vertices, sdf, info)
    relative = (vertices-object_position) @ object_rotation
    obj, _ = signed_query(relative, object_sdf, object_info)
    return dict(scene_penetration_mean_m=float((-scene).clamp_min(0).mean()),
        scene_penetration_max_m=float((-scene).clamp_min(0).max()),
        scene_outside_fraction=float(outside.float().mean()),
        object_penetration_max_m=float((-obj).clamp_min(0).max()),
        floor_penetration_max_m=float((-vertices[..., 1]).clamp_min(0).max()))


def geometry_passes(values):
    return (values['scene_penetration_mean_m'] <= .005
        and values['scene_penetration_max_m'] <= .05
        and values['scene_outside_fraction'] == 0
        and values['object_penetration_max_m'] <= .05
        and values['floor_penetration_max_m'] <= .01)


def transformed_motion(motion, rotation, translation):
    result = dict(motion)
    for key in ('joints', 'verts'):
        result[key] = motion[key] @ rotation.T+translation
    local = transforms.axis_angle_to_matrix(motion['pose'])
    local[:, 0] = rotation @ local[:, 0]
    result['pose'] = transforms.matrix_to_axis_angle(local)
    # SMPL-X rotates around the rest pelvis, not the world origin.
    rest_pelvis = motion['joints'][0, 0]-motion['translation'][0]
    result['translation'] = (motion['translation']+rest_pelvis) @ rotation.T+translation-rest_pelvis
    return result


def initial_to_task_rotation(source_joints, task):
    start = source_joints.new_tensor(task['start_location'])
    goal = source_joints.new_tensor(task['pelvis_goal'])
    target_heading = torch.atan2(-(goal[2]-start[2]), goal[0]-start[0])+math.pi/2
    return yaw_matrix((target_heading-heading(source_joints[0])).reshape(1))[0]


def support_patches(vertices, joints, model):
    """Two lower buttock patches near the pelvis on the native body surface."""
    pelvis = joints[-1, 0]
    dominant = model.lbs_weights.argmax(-1)
    near = (vertices[-1, :, [0, 2]]-pelvis[[0, 2]]).norm(dim=-1) <= .22
    patches = []
    for side in (1, 2):
        eligible = torch.where(near & (dominant == side))[0]
        selected = eligible[vertices[-1, eligible, 1].argsort()[:32]]
        patch = vertices[-1, selected].mean(0)
        patch[1] = vertices[-1, selected, 1].min()
        patches.append(patch)
    return torch.stack(patches)


def clear_straight_paths(start, ends, sdf, info, object_sdf, object_info, obj_pos, obj_rot):
    length = (ends[:, [0, 2]]-start[[0, 2]]).norm(dim=-1)
    # 31 samples cover the maximum registered 3m path at 10cm spacing.
    alpha = torch.linspace(0, 1, 31, device=ends.device)
    roots = start[None, None]+alpha[None, :, None]*(ends[:, None]-start)
    offsets = ends.new_tensor([[0, 0, 0], [.25, 0, 0], [-.25, 0, 0], [0, 0, .25], [0, 0, -.25]])
    heights = ends.new_tensor([.15, .45, .75, 1.05, 1.35, 1.65])
    probes = roots[:, :, None, None]+offsets[None, None, :, None]
    probes = probes.expand(-1, -1, -1, len(heights), -1).clone()
    probes[..., 1] = heights
    scene, outside = signed_query(probes, sdf, info)
    obj, _ = signed_query((probes-obj_pos) @ obj_rot, object_sdf, object_info)
    clear = ((scene.amin(dim=(1, 2, 3)) >= 0) & ~outside.flatten(1).any(-1)
             & (obj.amin(dim=(1, 2, 3)) >= 0) & (length >= .75) & (length <= 3))
    return clear, length


def segment_from_source(record, segment_id, scene, start, goal, initialization):
    return dict(segment_id=segment_id, task_type=record['task_type'],
        source_id=record['source_id'], source_dataset=record['source_dataset'], data_idx=record['data_idx'],
        source_scene=record['source_scene'], scene_name=scene, text=record['text'],
        start_location=list(start), pelvis_goal=list(goal), object_goal=None, scene_goal=None,
        initialization=initialization, data_idx_role='source_context_and_semantics',
        duration_s=record['source_duration_s'],
        source_frame_interval=[record['source_start_frame'], record['source_stop_frame']],
        contact_target=None, source_is_complete_recomposed_gt=False)


def make_episode(original, hoi, walk, sit, placed, hoi_context, object_position, object_rotation,
                 path_length, metrics, support, witness_path):
    task = original['original_task']
    scene = task['scene_name']
    start = placed['joints'][0, 0].tolist(); start[1] = 0
    goal = placed['joints'][-1, 0].tolist(); goal[1] = 0
    first = segment_from_source(hoi, 'hoi', scene, task['start_location'], task['pelvis_goal'], 'source_context')
    first.update(object_goal=task['object_goal'], original_hosi_task_id=original['task_id'],
        original_conditions=task, duration_s=task['episode_num']*1.4,
        exit_requirements=dict(object_supported=True, object_slow=True,
            object_rotation=object_rotation.tolist(), pelvis_goal=task['pelvis_goal']))
    second = segment_from_source(walk, 'walk', scene, task['pelvis_goal'], start, 'previous_generated_history')
    second.update(duration_s=math.ceil(path_length/.8/1.4)*1.4,
        path=[task['pelvis_goal'], start], entry_requirements=dict(hands_released=True),
        exit_requirements=dict(pelvis_goal=start, heading_rad=float(heading(placed['joints'][0]))))
    third = segment_from_source(sit, 'sit', scene, start, goal, 'previous_generated_history')
    third.update(scene_goal=placed['joints'][-1, 0].tolist(),
        initial_heading_rad=float(heading(placed['joints'][0])),
        terminal_heading_rad=float(heading(placed['joints'][-1])),
        source_to_target_placement=placed['placement'],
        contact_target=dict(target_id=f'{original["task_id"]}-seat', kind='seating_support_surface',
            support_points=support.tolist(), semantic_review='pending', scene_name=scene),
        exit_requirements=dict(seated=True, pelvis_goal=goal, seat_contact=True, feet_supported=True))
    return dict(episode_id=f'multitask-{original["task_id"]}', schema_version=1,
        scene_name=scene, scene_geometry=dict(sdf=f'data/hosi_test/Scene_sdf/{scene}_sdf.npy',
            sdf_info=f'data/hosi_test/Scene_sdf/{scene}_sdf_info.json',
            occupancy=f'data/test/Scene_vis/{scene}.npy'),
        original_hosi_task_id=original['task_id'], chain_type='hoi_walk_sit',
        body_identity=dict(source_id=hoi['source_id'], gender=hoi['gender'], body_parameters=hoi['body_parameters']),
        persistent_objects=[dict(object_id=task['object_name'], geometry=f'data/test/rest_object_geo/{task["object_name"]}.ply',
            planned_terminal_translation=object_position.tolist(), planned_terminal_rotation=object_rotation.tolist(),
            inference_transform='actual_achieved_transform_after_hoi', persist_through=['walk', 'sit'])],
        segments=[first, second, third], transitions=[
            transition_edge('hoi', 'walk', 'release_and_walk', 'hold_achieved_supported_object'),
            transition_edge('walk', 'sit', 'align_for_sitting', 'hold_achieved_supported_object')],
        construction_status='geometry_accepted_pending_semantic_review',
        construction=dict(boundary_geometry=metrics, path_length_m=path_length,
            witness=witness_path, witness_role='input_feasibility_only', complete_motion_gt=False,
            selection_uses_model_outputs=False),
        evaluation=dict(include_transition_frames=True, initial_state='source_context',
            successor_state='actual_generated_history', count_runtime_failures=True,
            metrics=['ordered_completion', 'episode_success', 'longest_completed_prefix',
                'pelvis_goal_error', 'object_goal_error', 'scene_penetration', 'object_penetration',
                'seat_contact', 'hand_release', 'foot_sliding', 'boundary_position_velocity_jump']))


@torch.no_grad()
def construct_hosi_chains(cfg, root, output, corpora, original, sources):
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    parameters = cfg.multitask
    device = cfg.device
    records = {r['source_id']: r for r in sources}
    def upright(boundary):
        return (boundary['torso_tilt_max_deg'] <= 25 and boundary['pelvis_height_min_m'] >= .65
                and boundary['foot_floor_distance_max_m'] <= .08)
    sit_pool = sorted([r for r in sources if r['source_dataset'] == 'LINGO' and r['text'].startswith('sit down')
        and upright(r['source_boundary_audit']['entry'])
        and r['source_boundary_audit']['exit']['pelvis_height_max_m'] <= .85], key=lambda r: r['data_idx'])
    sit_pool = sit_pool[:int(parameters.static_source_limit)]
    walk_pool = sorted([r for r in sources if r['task_type'] == 'locomotion'
        and upright(r['source_boundary_audit']['entry'])], key=lambda r: r['data_idx'])
    write_json(output/'fixed_source_selection.json', dict(sit_sources=[r['source_id'] for r in sit_pool],
        walk_sources=[r['source_id'] for r in walk_pool[:1]], rule='ascending data_idx after registered source boundary conditions'))
    # Empty source pools are a scientific construction result.
    if not sit_pool or not walk_pool:
        return [], [dict(task_id=r['task_id'], status='no_eligible_lingo_source') for r in original]
    walk = walk_pool[0]
    models, native_cache, object_cache = {}, {}, {}
    episodes, audit, accepted_scenes = [], [], set()
    current_scene, sdf, info = None, None, None
    step = float(parameters.anchor_spacing_m)
    x = torch.arange(-2.8, 2.8+step/2, step, device=device)
    z = torch.arange(-3.8, 3.8+step/2, step, device=device)
    gx, gz = torch.meshgrid(x, z, indexing='ij')
    centers = torch.stack((gx.flatten(), torch.zeros_like(gx).flatten(), gz.flatten()), -1)
    angles = torch.arange(8, device=device)*math.pi/4
    rotations = yaw_matrix(angles)
    candidate_rotation = rotations[:, None].expand(-1, len(centers), -1, -1).reshape(-1, 3, 3)
    candidate_centers = centers.repeat(8, 1)
    for row in original:
        task = row['original_task']
        hoi = records[row['source_id']]
        status = dict(task_id=row['task_id'], scene_name=task['scene_name'], source_id=hoi['source_id'])
        boundary = hoi['source_boundary_audit']['exit']
        reasons = []
        if not boundary['supported_slow']:
            reasons.append('source_object_requires_placement_or_settling')
        if not upright(boundary):
            reasons.append('source_exit_not_upright_supported')
        if reasons:
            audit.append(dict(status, status='ineligible_source', reasons=reasons)); continue
        if len(episodes) >= int(parameters.episode_limit):
            audit.append(dict(status, status='not_attempted_episode_cap')); continue
        if task['scene_name'] in accepted_scenes:
            audit.append(dict(status, status='not_attempted_scene_already_selected')); continue
        if current_scene != task['scene_name']:
            current_scene = task['scene_name']
            sdf = torch.as_tensor(np.load(root/f'data/hosi_test/Scene_sdf/{current_scene}_sdf.npy'),
                device=device, dtype=torch.float32)[None, None]
            info = json.loads((root/f'data/hosi_test/Scene_sdf/{current_scene}_sdf_info.json').read_text())
        if hoi['gender'] not in models:
            models[hoi['gender']] = create_smplx_model(hoi['gender'], torch.device(device)).eval().requires_grad_(False)
        model = models[hoi['gender']]
        corpus = corpora['OMOMO']
        source_context = corpus.native_motion(hoi, hoi['terminal_context_frames'], device, model)
        status['native_source_joint_max_error_m'] = source_context['source_joint_max_error_m']
        if source_context['source_joint_max_error_m'] > .001:
            audit.append(dict(status, status='native_source_roundtrip_failed')); continue
        entry_joints = torch.as_tensor(np.array(corpus.joints[hoi['initial_context_frames']]), device=device, dtype=torch.float32)
        task_rotation = initial_to_task_rotation(entry_joints, task)
        target_root = source_context['joints'][-1, 0].clone()
        target_root[[0, 2]] = target_root.new_tensor(task['pelvis_goal'])[[0, 2]]
        shift = target_root-source_context['joints'][-1, 0] @ task_rotation.T
        hoi_context = transformed_motion(source_context, task_rotation, shift)
        object_position = target_root.new_tensor(task['object_goal'])
        object_rotation = task_rotation @ torch.as_tensor(np.array(corpus.object_rotation[hoi['source_terminal_frame']]),
                                                           device=device, dtype=torch.float32)
        name = task['object_name']
        if name not in object_cache:
            mesh = trimesh.load_mesh(root/f'data/test/rest_object_geo/{name}.ply')
            obj_vertices = torch.as_tensor(zup_to_yup(np.asarray(mesh.vertices)), device=device, dtype=torch.float32)
            obj_sdf, obj_info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', name)
            object_cache[name] = (obj_vertices, torch.as_tensor(obj_sdf, device=device, dtype=torch.float32)[None, None], obj_info)
        object_vertices, object_sdf, object_info = object_cache[name]
        object_world = object_vertices @ object_rotation.T+object_position
        floor_distance = float(object_world[:, 1].min().abs())
        object_scene, object_outside = signed_query(object_world, sdf, info)
        endpoint = geometry_measures(hoi_context['verts'], sdf, info, object_sdf, object_info, object_position, object_rotation)
        endpoint.update(object_support_floor_distance_m=floor_distance,
            object_scene_penetration_max_m=float((-object_scene).clamp_min(0).max()),
            object_scene_outside_fraction=float(object_outside.float().mean()))
        status['hoi_terminal_witness'] = endpoint
        if not (geometry_passes(endpoint) and floor_distance <= .05
                and endpoint['object_scene_penetration_max_m'] <= .05 and not object_outside.any()):
            audit.append(dict(status, status='infeasible_hoi_terminal_witness')); continue
        found = False
        source_attempts = []
        for sit in sit_pool:
            cache_key = (sit['source_id'], hoi['gender'], tuple(source_context['betas'].tolist()))
            if cache_key not in native_cache:
                lingo = corpora['LINGO']
                frames = sit['initial_context_frames']+sit['terminal_context_frames']
                # Transfer source local rotations onto the persistent HOI body.
                pose = torch.as_tensor(np.concatenate((lingo.orient[frames, None],
                    lingo.pose[frames].reshape(-1, 21, 3)), axis=1), device=device, dtype=torch.float32)
                betas = source_context['betas']
                neutral = model.J_regressor @ (model.v_template+torch.einsum('vci,i->vc', model.shapedirs[..., :len(betas)], betas))
                roots = torch.as_tensor(np.array(lingo.joints[frames, 0]), device=device, dtype=torch.float32)
                roots[:, [0, 2]] -= roots[-1:, [0, 2]].clone()
                motion = dict(pose=pose, translation=roots-neutral[0], betas=betas, gender=hoi['gender'])
                motion['verts'], motion['joints'] = decode_body(motion, model)
                floor_shift = -motion['verts'][..., 1].min()
                for key in ('translation', 'joints', 'verts'):
                    motion[key][..., 1] += floor_shift
                motion['vertical_placement_shift_m'] = float(floor_shift)
                native_cache[cache_key] = (motion, support_patches(motion['verts'], motion['joints'], model))
            template, patch = native_cache[cache_key]
            attempt = dict(source_id=sit['source_id'])
            source_attempts.append(attempt)
            if float(template['joints'][:, [7, 8, 10, 11], 1].abs().amin(-1).max()) > .08:
                attempt['status'] = 'body_transfer_foot_support_failed'; continue
            support_world = torch.einsum('bij,kj->bki', candidate_rotation, patch)+candidate_centers[:, None]
            support_signed, support_outside = signed_query(support_world, sdf, info)
            below = support_world.clone(); below[..., 1] -= .04
            below_signed, _ = signed_query(below, sdf, info)
            mask = (support_signed.abs().amax(-1) <= .03) & (below_signed.amax(-1) < 0) & ~support_outside.any(-1)
            indices = torch.where(mask)[0]
            attempt['support_candidates'] = len(indices)
            if not len(indices):
                attempt['status'] = 'no_bilateral_support_anchor'; continue
            candidate_joint = torch.einsum('bij,tkj->btki', candidate_rotation[indices], template['joints'][[0, -1], :22])
            candidate_joint += candidate_centers[indices, None, None]
            signed, outside = signed_query(candidate_joint, sdf, info)
            mask = (signed.amin(dim=(1, 2)) >= -.02) & ~outside.flatten(1).any(-1)
            indices = indices[mask]
            attempt['joint_clear_candidates'] = len(indices)
            if not len(indices):
                attempt['status'] = 'coarse_body_collision'; continue
            ends = (torch.einsum('bij,j->bi', candidate_rotation[indices], template['joints'][0, 0])
                    +candidate_centers[indices]); ends[:, 1] = 0
            path_start = ends.new_tensor(task['pelvis_goal'])
            clear, length = clear_straight_paths(path_start, ends, sdf, info, object_sdf, object_info, object_position, object_rotation)
            indices, lengths = indices[clear], length[clear]
            order = lengths.argsort(stable=True)
            indices, lengths = indices[order], lengths[order]
            attempt['clear_path_candidates'] = len(indices)
            attempt['full_geometry_checks'] = []
            for index, path_length in zip(indices[:int(parameters.full_candidate_limit)].tolist(),
                                           lengths[:int(parameters.full_candidate_limit)].tolist()):
                placed = transformed_motion(template, candidate_rotation[index], candidate_centers[index])
                metrics = geometry_measures(placed['verts'], sdf, info, object_sdf, object_info, object_position, object_rotation)
                attempt['full_geometry_checks'].append(dict(candidate=index, **metrics))
                if not geometry_passes(metrics):
                    continue
                placed['placement'] = dict(yaw_rad=float(angles[index//len(centers)]),
                    terminal_root_xz_m=candidate_centers[index, [0, 2]].tolist(),
                    source_terminal_root_xz_m=np.asarray(corpora['LINGO'].joints[sit['source_terminal_frame'], 0, [0, 2]]).tolist(),
                    vertical_shift_m=template['vertical_placement_shift_m'], body_identity_source=hoi['source_id'])
                witness_path = f'witnesses/{row["task_id"]}.pt'
                dest = output/witness_path; dest.parent.mkdir(parents=True, exist_ok=True)
                torch.save(dict(hoi_context=hoi_context, sit_context=placed, support_points=support_world[index],
                    object_vertices=object_world, scene=task['scene_name']), dest)
                episode = make_episode(row, hoi, walk, sit, placed, hoi_context, object_position, object_rotation,
                    path_length, dict(hoi=endpoint, sit=metrics), support_world[index], witness_path)
                validate_episode(episode, records)
                episodes.append(episode); accepted_scenes.add(task['scene_name'])
                attempt.update(status='accepted_geometry', selected_candidate=index)
                audit.append(dict(status, status='accepted_geometry_pending_semantic_review', source_attempts=source_attempts,
                                  episode_id=episode['episode_id']))
                print(f'Constructed {episode["episode_id"]} in {task["scene_name"]}; {len(episodes)}/{parameters.episode_limit}', flush=True)
                found = True
                break
            if found:
                break
            attempt.setdefault('status', 'full_geometry_or_path_failed')
        if not found:
            audit.append(dict(status, status='no_feasible_seating_placement', source_attempts=source_attempts))
    return episodes, audit


def render_construction_previews(root, output, episodes, mesh_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    parents = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
    preview = output/'preview'; preview.mkdir()
    for episode in episodes:
        witness = torch.load(output/episode['construction']['witness'], map_location='cpu', weights_only=False)
        target = witness['sit_context']['joints'][-1, 0].numpy()
        mesh = trimesh.load_mesh(Path(mesh_root)/(episode['scene_name']+'.obj'))
        vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
        centers = vertices[faces].mean(1)
        local = np.linalg.norm(centers[:, [0, 2]]-target[[0, 2]], axis=-1) < 1.25
        local &= centers[:, 1] < 1.8
        local_faces = faces[local]
        local_faces = local_faces[::max(1, len(local_faces)//18000)]
        fig = plt.figure(figsize=(12, 5))
        top = fig.add_subplot(121)
        visible = centers[:, 1] < 1.8
        cloud = centers[visible][::max(1, int(visible.sum())//30000)]
        top.scatter(cloud[:, 0], cloud[:, 2], s=.3, c='#999999')
        route = np.array(episode['segments'][1]['path'])
        top.plot(route[:, 0], route[:, 2], 'o-', color='#a13c59', label='walk path')
        obj = witness['object_vertices'].numpy()
        top.scatter(obj[::5, 0], obj[::5, 2], s=1, c='#c28a25', label='persisted object')
        top.scatter(target[0], target[2], marker='*', s=100, c='#2877b5', label='seating goal')
        top.set(xlabel='World X (m)', ylabel='World Z (m)', title='Fixed task conditions; top view', xlim=(-3, 3), ylim=(-4, 4))
        top.set_aspect('equal'); top.legend(fontsize=8)
        ax = fig.add_subplot(122, projection='3d')
        tri = Poly3DCollection(vertices[local_faces][..., [0, 2, 1]], alpha=.22, facecolor='#aaaaaa', edgecolor='none')
        ax.add_collection3d(tri)
        for frame, color, label in [(0, '#269b8e', 'static entry'), (-1, '#2877b5', 'static exit')]:
            joints = witness['sit_context']['joints'][frame, :22].numpy()[:, [0, 2, 1]]
            for joint, parent in enumerate(parents):
                if parent >= 0:
                    ax.plot(*joints[[parent, joint]].T, color=color, linewidth=2)
            ax.scatter(*joints.T, c=color, s=8, label=label)
        support = witness['support_points'].numpy()[:, [0, 2, 1]]
        ax.scatter(*support.T, c='#d94435', s=40, label='support patches')
        ax.set(xlim=(target[0]-1.1, target[0]+1.1), ylim=(target[2]-1.1, target[2]+1.1), zlim=(0, 2),
               xlabel='X', ylabel='Z', zlabel='Y', title='Native body conditions and scene support')
        ax.view_init(elev=22, azim=-55); ax.set_box_aspect((1, 1, .9)); ax.legend(fontsize=7)
        fig.suptitle(episode['episode_id']+' | geometry witness, no generated motion', fontsize=11)
        fig.tight_layout(); fig.savefig(preview/(episode['episode_id']+'.png'), dpi=140); plt.close(fig)
