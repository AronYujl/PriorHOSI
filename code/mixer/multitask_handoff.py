"""Actual-state and timing eligibility for a cached multi-task predecessor."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .multitask import human_boundary_measures, validate_episode, write_json
from .multitask_geometry import geometry_measures, geometry_passes, signed_query
from .standing_transition import observed_motion, tail_measures
from .surface_edit import decode_body, load_object_sdf, native_hand_distances


MOTION_FIELDS = ('pose', 'translation', 'joints', 'object_translation', 'object_rotation')


def native_budget_frames(duration_s, fps=30, history_frames=2, stride=3):
    """Duration counts new motion after the initial coarse history."""
    return (history_frames-1)*stride + 1 + round(duration_s*fps)


def motion_slice(motion, start, stop):
    return {k: v[start:stop] if k in MOTION_FIELDS or k == 'verts' else v
            for k, v in motion.items()}


def source_handoff_checks(measures, thresholds, goal_tolerance_m):
    """A release bridge can change hand contact after the object settles."""
    return dict(
        pelvis_at_goal=measures['pelvis_goal_error_m'] <= goal_tolerance_m,
        object_at_goal=measures['object_goal_error_m'] <= goal_tolerance_m,
        object_supported=measures['object_floor_max_m'] <= thresholds.object_floor_m,
        object_translation_slow=measures['object_speed_max_m_s'] <= thresholds.object_speed_m_s,
        object_rotation_slow=measures['object_angular_speed_max_rad_s'] <= thresholds.object_angular_speed_rad_s,
        body_upright=measures['tilt_max_deg'] <= thresholds.tilt_deg,
        body_height=measures['root_height_min_m'] >= thresholds.root_height_m,
        body_foot_support=measures['foot_height_max_m'] <= thresholds.foot_height_m,
        body_geometry=geometry_passes(measures['body_geometry']),
        object_scene_geometry=(measures['object_scene_geometry']['scene_penetration_mean_m'] <= .005
            and measures['object_scene_geometry']['scene_penetration_max_m'] <= .05
            and measures['object_scene_geometry']['scene_outside_fraction'] == 0),
    )


def surface_scene_measures(points, sdf, info):
    signed, outside = signed_query(points, sdf, info)
    return dict(scene_penetration_mean_m=float((-signed).clamp_min(0).mean()),
        scene_penetration_max_m=float((-signed).clamp_min(0).max()),
        scene_outside_fraction=float(outside.float().mean()))


def cutoff_measures(motion, task, object_vertices, scene, obj, cfg):
    tail = motion_slice(motion, -10, None)
    values = tail_measures(tail, object_vertices, cfg.stand_wait.thresholds)
    goal = motion['joints'].new_tensor(task['pelvis_goal'])
    values['pelvis_goal_error_m'] = float((motion['joints'][-1, 0, [0, 2]]-goal[[0, 2]]).norm())
    values['object_goal_error_m'] = float((motion['object_translation'][-1]-goal.new_tensor(task['object_goal'])).norm())
    rotation_goal = goal.new_tensor(task['exit_requirements']['object_rotation'])
    delta = motion['object_rotation'][-1] @ rotation_goal.T
    values['object_goal_rotation_error_deg'] = float(transforms.matrix_to_axis_angle(delta).norm()*180/torch.pi)
    values['body_geometry'] = geometry_measures(tail['verts'], *scene, *obj,
        tail['object_translation'][:, None], tail['object_rotation'])
    object_world = object_vertices @ tail['object_rotation'].transpose(-1, -2) + tail['object_translation'][:, None]
    values['object_scene_geometry'] = surface_scene_measures(object_world, *scene)
    values['object_last_step_speed_m_s'] = float(((tail['object_translation'][-1]-tail['object_translation'][-2])*30).norm())
    values['observed_frames'] = len(motion['pose'])
    values['tail_frame_interval'] = [len(motion['pose'])-10, len(motion['pose'])]
    values['tail_timestamp_interval_s'] = [(len(motion['pose'])-10)/30, (len(motion['pose'])-1)/30]
    values['checks'] = source_handoff_checks(values, cfg.stand_wait.thresholds, cfg.multitask.handoff.goal_tolerance_m)
    values['failed_checks'] = [k for k, passed in values['checks'].items() if not passed]
    return values


def target_context_measures(target, motion, model, object_vertices, scene, obj):
    vertices, joints = decode_body(target, model)
    position, rotation = motion['object_translation'][-1], motion['object_rotation'][-1]
    object_world = object_vertices @ rotation.T + position
    geometry = geometry_measures(vertices, *scene, *obj, position, rotation)
    hands = native_hand_distances(joints, object_world[None].expand(len(joints), -1, -1))
    pose_delta = (transforms.axis_angle_to_matrix(target['pose'][0]) @
                  transforms.axis_angle_to_matrix(motion['pose'][-1]).transpose(-1, -2))
    return dict(body_geometry=geometry, human_boundary=human_boundary_measures(joints),
        hand_object_min_m=float(hands.min()),
        body_identity_error=float((target['betas']-motion['betas']).abs().max()),
        joint_reconstruction_max_error_m=float((joints-target['joints']).norm(dim=-1).max()),
        source_to_target_joint_mean_m=float((joints[0]-motion['joints'][-1]).norm(dim=-1).mean()),
        source_to_target_root_m=float((joints[0, 0]-motion['joints'][-1, 0]).norm()),
        source_to_target_rotation_max_deg=float(transforms.matrix_to_axis_angle(pose_delta).norm(dim=-1).max()*180/torch.pi),
        source_to_target_velocity_jump_m_s=float(((joints[1]-joints[0])-(motion['joints'][-1]-motion['joints'][-2])).norm(dim=-1).mean()*30),
        achieved_object_translation=position.tolist(), achieved_object_rotation=rotation.tolist())


def motion_trace(motion, object_vertices):
    joints = motion['joints']
    obj_world = object_vertices @ motion['object_rotation'].transpose(-1, -2) + motion['object_translation'][:, None]
    feet = joints[:, [7, 8, 10, 11]]
    contact = feet[:-1, :, 1].abs() <= .08
    speed = (feet[1:]-feet[:-1])[..., [0, 2]].norm(dim=-1)*30
    omega = transforms.matrix_to_axis_angle(motion['object_rotation'][1:] @ motion['object_rotation'][:-1].transpose(-1, -2))*30
    values = dict(time_s=torch.arange(len(joints), device=joints.device)/30,
        object_lowest_surface_m=obj_world[..., 1].amin(-1),
        foot_marker_distance_m=feet[..., 1].abs().amin(-1),
        object_speed_m_s=motion['object_translation'].diff(dim=0).norm(dim=-1)*30,
        object_angular_speed_rad_s=omega.norm(dim=-1),
        root_speed_m_s=joints[:, 0].diff(dim=0).norm(dim=-1)*30,
        contact_foot_speed_m_s=speed, near_floor_contact=contact,
        body_lowest_surface_m=motion['verts'][..., 1].amin(-1))
    stats = dict(near_floor_foot_speed_mean_m_s=float(speed[contact].mean()) if contact.any() else None,
        near_floor_contact_pairs=int(contact.sum()),
        near_floor_contact_pair_fraction=float(contact.float().mean()),
        unsupported_frame_fraction=float((values['foot_marker_distance_m'] > .08).float().mean()))
    return {k: v.cpu().numpy() for k, v in values.items()}, stats


def render_audit(output, episode, motion, target, object_vertices, trace, budget_frames, mesh_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import trimesh
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from .kinematic_composition import _PARENTS_22
    preview = output/'preview'
    preview.mkdir()
    time_s = trace['time_s']
    cutoff = (budget_frames-1)/30
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    plots = [('object_speed_m_s', .10, 'Object translation (m/s)'),
             ('object_angular_speed_rad_s', .5, 'Object rotation (rad/s)'),
             ('object_lowest_surface_m', .05, 'Object lowest surface (m)')]
    for ax, (key, limit, label) in zip(axes, plots):
        ax.plot(time_s[1:] if len(trace[key]) == len(time_s)-1 else time_s, trace[key], color='#2477a6')
        ax.axhline(limit, color='#b54040', linestyle='--', label='Handoff limit')
        if key == 'object_lowest_surface_m':
            ax.axhline(-limit, color='#b54040', linestyle='--')
            ax.axhline(0, color='#666666', linewidth=.6)
        ax.axvline(cutoff, color='#333333', linestyle=':', label='Published budget cutoff')
        ax.axvspan(cutoff-.3, cutoff, color='#888888', alpha=.15)
        ax.axvspan(time_s[-1]-.3, time_s[-1], color='#2477a6', alpha=.12)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[0].legend(loc='upper left', fontsize=8)
    axes[-1].set_xlabel('Time from first history sample (s)')
    fig.suptitle(episode['episode_id']+' | unchanged cached HOI trajectory')
    fig.savefig(preview/'object_handoff_trace.png', dpi=150)
    fig.savefig(preview/'object_handoff_trace.pdf')
    plt.close(fig)

    mesh = trimesh.load_mesh(Path(mesh_root)/(episode['scene_name']+'.obj'))
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    center = motion['joints'][-1, 0].cpu().numpy()
    centers = vertices[faces].mean(1)
    selected = (np.linalg.norm(centers[:, [0, 2]]-center[[0, 2]], axis=-1) < 1.8) & (centers[:, 1] < 1.8)
    faces = faces[selected][::max(1, int(selected.sum())//12000)]
    fig = plt.figure(figsize=(12, 5), constrained_layout=True)
    for panel, (frame, label) in enumerate([(budget_frames-1, 'Published budget'), (len(time_s)-1, 'Cached terminal')]):
        ax = fig.add_subplot(1, 2, panel+1, projection='3d')
        ax.add_collection3d(Poly3DCollection(vertices[faces][..., [0, 2, 1]], facecolor='#aaaaaa', alpha=.2, edgecolor='none'))
        for joints, color, name in [(motion['joints'][frame], '#2477a6', 'Actual body'), (target['joints'][0], '#348567', 'Prescribed bridge entry')]:
            points = joints[:22].cpu().numpy()[:, [0, 2, 1]]
            for child, parent in enumerate(_PARENTS_22):
                if parent >= 0:
                    ax.plot(*points[[parent, child]].T, color=color, linewidth=1.8)
            ax.scatter(*points.T, c=color, s=5, label=name)
        object_points = object_vertices.cpu().numpy() @ motion['object_rotation'][frame].cpu().numpy().T + motion['object_translation'][frame].cpu().numpy()
        ax.scatter(*object_points[::4, [0, 2, 1]].T, s=1, color='#bd8032', label='Actual object')
        ax.set(xlim=(center[0]-1.3, center[0]+1.3), ylim=(center[2]-1.3, center[2]+1.3), zlim=(-.1, 2),
            xlabel='X (m)', ylabel='Z (m)', zlabel='Y (m)', title=f'{label}: {frame/30:.2f} s')
        ax.view_init(elev=20, azim=-60)
        ax.set_box_aspect((1, 1, .9))
        ax.legend(fontsize=7)
    fig.savefig(preview/'actual_handoff_contexts.png', dpi=150)
    plt.close(fig)


@torch.no_grad()
def run_handoff_audit(cfg):
    from omegaconf import OmegaConf
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered handoff audit requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    settings = cfg.multitask.handoff
    output = Path(cfg.multitask.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(settings.task_manifest).read_text())
    episode, = manifest['episodes']
    validate_episode(episode, {r['source_id']: r for r in manifest['sources']})
    segment = episode['segments'][0]
    raw = torch.load(settings.source_motion, map_location=cfg.device, weights_only=False)[settings.source_arm]
    raw['gender'] = episode['body_identity']['gender']
    motion = observed_motion(raw)
    original = json.loads(Path(settings.source_metrics).read_text())
    model = create_smplx_model(motion['gender'], torch.device(cfg.device)).eval().requires_grad_(False)
    torch.cuda.synchronize(cfg.device)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(cfg.device)
    motion['verts'], rebuilt = decode_body(motion, model)
    reconstruction_error = float((rebuilt-motion['joints']).norm(dim=-1).max())
    body = episode['body_identity']['body_parameters']
    expected_betas = torch.as_tensor(np.array(np.load(body['path'], mmap_mode='r')[body['index']]), device=cfg.device)
    original_rows = json.loads((root/'data/hosi_test/data'/(episode['scene_name']+'.json')).read_text())
    identity = dict(body_parameters_match=bool(torch.equal(expected_betas, motion['betas'])),
        original_task_preserved=segment['original_conditions'] in original_rows,
        source_scene_match=original['scene'] == episode['scene_name'],
        source_object_match=original['object'] == episode['persistent_objects'][0]['object_id'],
        source_task_match=original['task'] == int(episode['original_hosi_task_id'].split('-')[-1]),
        native_reconstruction=reconstruction_error <= 1e-4,
        native_window_frame_count=len(motion['pose']) == native_budget_frames(original['windows']*1.4),
        legacy_padding_is_held=all(torch.equal(raw[k][-2:], raw[k][-3:-2].expand_as(raw[k][-2:])) for k in MOTION_FIELDS))
    geometry = episode['scene_geometry']
    scene = (torch.as_tensor(np.load(root/geometry['sdf']), dtype=torch.float32, device=cfg.device)[None, None],
             json.loads((root/geometry['sdf_info']).read_text()))
    array, info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', original['object'])
    obj = (torch.as_tensor(array, dtype=torch.float32, device=cfg.device)[None, None], info)
    mesh = trimesh.load_mesh(root/episode['persistent_objects'][0]['geometry'])
    object_vertices = torch.as_tensor(zup_to_yup(np.asarray(mesh.vertices)), dtype=torch.float32, device=cfg.device)
    frames = native_budget_frames(segment['duration_s'])
    timing = dict(source_export_frames=len(raw['pose']), observed_frames=len(motion['pose']),
        removed_held_padding_frames=len(raw['pose'])-len(motion['pose']), cached_windows=original['windows'],
        source_fps=30, initial_history_interval_s=.1,
        published_new_motion_budget_s=segment['duration_s'], published_observed_frame_limit=frames,
        excess_generated_frames=max(0, len(motion['pose'])-frames),
        cached_new_motion_duration_s=(len(motion['pose'])-4)/30,
        cached_last_timestamp_s=(len(motion['pose'])-1)/30,
        source_cache_within_budget=len(motion['pose']) <= frames)
    records, first_target = {}, None
    for label, stop in [('published_budget_prefix', min(frames, len(motion['pose']))), ('cached_terminal', len(motion['pose']))]:
        prefix = motion_slice(motion, 0, stop)
        record = cutoff_measures(prefix, segment, object_vertices, scene, obj, cfg)
        contexts = []
        for edge in episode['transitions']:
            ref = edge['target_context_reference']
            witness = torch.load(root/manifest['artifact_root']/ref['artifact'], map_location=cfg.device, weights_only=False)
            target = motion_slice(witness[ref['motion_key']], ref['frame_start'], ref['frame_stop'])
            if first_target is None:
                first_target = target
            values = target_context_measures(target, prefix, model, object_vertices, scene, obj)
            target_checks = dict(geometry=geometry_passes(values['body_geometry']),
                foot_support=values['human_boundary']['foot_floor_distance_max_m'] <= .08,
                hands_released=values['hand_object_min_m'] >= .08,
                body_identity=values['body_identity_error'] <= 1e-6,
                reconstruction=values['joint_reconstruction_max_error_m'] <= 1e-4)
            contexts.append(dict(source_segment=edge['source_segment'], target_segment=edge['target_segment'],
                scope='prescribed_context_against_achieved_hoi_object', reference=ref,
                predecessor_history_available=edge['source_segment'] == 'hoi',
                measures=values, checks=target_checks, target_admissible=all(target_checks.values())))
        record['target_contexts'] = contexts
        record['handoff_geometry_and_state_ready'] = not record['failed_checks'] and contexts[0]['target_admissible']
        records[label] = record
        torch.save({k: v.cpu() if torch.is_tensor(v) else v for k, v in motion_slice(prefix, -10, None).items()
                    if k != 'verts'}, output/(label+'_context.pt'))
    trace, foot_stats = motion_trace(motion, object_vertices)
    np.savez(output/'trace.npz', **trace)
    np.savez(output/'source_motion.npz', **{k: raw[k].cpu().numpy() for k in MOTION_FIELDS+('betas',)},
        gender=np.array(raw['gender']), fps=np.array(30), observed_frames=np.array(len(motion['pose'])))
    whole_geometry = geometry_measures(motion['verts'], *scene, *obj,
        motion['object_translation'][:, None], motion['object_rotation'])
    entry_checks = dict(source_identity=all(identity.values()), source_cache_within_budget=timing['source_cache_within_budget'],
        actual_terminal_handoff=records['cached_terminal']['handoff_geometry_and_state_ready'])
    render_audit(output, episode, motion, first_target, object_vertices, trace, min(frames, len(motion['pose'])), cfg.multitask.scene_mesh_root)
    torch.cuda.synchronize(cfg.device)
    result = dict(schema_version=1, subphase='5.5.1', status='completed', seed=int(cfg.seed),
        git_commit=commit, git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        episode_id=episode['episode_id'], source_motion=str(settings.source_motion), source_metrics=str(settings.source_metrics),
        source_arm=str(settings.source_arm), input_manifest=str(settings.task_manifest),
        original_goal_only_completed=original['terminal'][0]['metrics']['completed'],
        original_terminal_repair_attempted=original['terminal'][0]['attempted'],
        original_terminal_repair_steps=original['terminal'][0]['solver']['steps'],
        identity_checks=identity, source_joint_reconstruction_max_error_m=reconstruction_error,
        timing=timing, cutoffs=records, full_cache_body_geometry=whole_geometry, full_cache_foot_measures=foot_stats,
        execution_entry_checks=entry_checks, execution_entry_gate=all(entry_checks.values()),
        full_chain_status='not_executed', full_chain_metrics=None,
        successors=[dict(segment_id=s['segment_id'], status='awaiting_execution' if all(entry_checks.values()) else 'blocked_by_predecessor', metrics=None)
                    for s in episode['segments'][1:]],
        model_samples=0, device=str(cfg.device), elapsed_seconds=time.perf_counter()-started,
        peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated(cfg.device),
        interpretation='Cached original-protocol diagnostic; the budget prefix is not a shorter-budget model run.')
    write_json(output/'resolved_config.json', OmegaConf.to_container(cfg, resolve=True))
    write_json(output/'metrics.json', result)
    print(json.dumps(result), flush=True)
