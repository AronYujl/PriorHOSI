"""Kimodo acquisition bridges on a frozen source-only candidate manifest."""

import json
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .body_projection import native_rest_offsets
from .inbetween import dispatch, fit_native, heading, interpolate_rotations, motion_metrics
from .inbetween_contact import contact_intervals, contact_measures, correct_motion, surface_patches
from .multitask import hand_object_distances, write_json
from .source_eligibility import GRASPED_ENTRY, _load_scene, full_source_geometry
from .surface_edit import decode_body, load_object_sdf, yaw_matrix

PREDICTION_FIELDS = ('target_joints', 'root_positions', 'local_rot_mats', 'foot_contacts')


def prediction_row(arrays, index, device):
    """Select one motion sample; scalar metadata such as fps has no batch axis."""
    return {key:torch.as_tensor(arrays[key][index], device=device) for key in PREDICTION_FIELDS}


@torch.no_grad()
def bridge_condition(prefix, suffix, position, rotation, model, *, object_contexts=None):
    weights = ((torch.arange(61, device=position.device)-9)/42).clamp(0, 1)
    first = transforms.axis_angle_to_matrix(prefix['pose'][-10:])
    last = transforms.axis_angle_to_matrix(suffix['pose'][:10])
    rotations = interpolate_rotations(first[-1], last[0], weights)
    translation = prefix['translation'][-1]+weights[:, None]*(suffix['translation'][0]-prefix['translation'][-1])
    rotations[:10], rotations[51:] = first, last
    translation[:10], translation[51:] = prefix['translation'][-10:], suffix['translation'][:10]
    motion = dict(pose=transforms.matrix_to_axis_angle(rotations), translation=translation,
        betas=suffix['betas'], gender=suffix['gender'], object_translation=position.repeat(61, 1),
        object_rotation=rotation.repeat(61, 1, 1))
    if object_contexts is not None:
        for key in ('object_translation', 'object_rotation'):
            motion[key][:10] = object_contexts[0][key][-10:]
            motion[key][51:] = object_contexts[1][key][:10]
    # Keep native context axis angles themselves, including their branch choice.
    motion['pose'][:10], motion['pose'][51:] = prefix['pose'][-10:], suffix['pose'][:10]
    motion['verts'], motion['joints'] = decode_body(motion, model)
    canonical = yaw_matrix((-heading(motion['joints'][0])).reshape(1))[0]
    origin = motion['joints'][0, 0].clone()
    origin[1] = 0
    rotations[:, 0] = canonical @ rotations[:, 0]
    points = (motion['joints'][:, :22]-origin) @ canonical.T
    neutral = model.J_regressor @ (model.v_template+torch.einsum(
        'vci,i->vc', model.shapedirs[..., :len(motion['betas'])], motion['betas']))
    arrays = dict(local_rot_mats=rotations, root_positions=points[:, 0], joints=points,
        neutral_joints=neutral[:22], known=(torch.arange(61, device=position.device) < 10)
            | (torch.arange(61, device=position.device) >= 51))
    return motion, canonical, origin, arrays


def native_arrays(motion):
    keys = ('pose', 'translation', 'joints', 'betas', 'object_translation', 'object_rotation')
    return dict({k: motion[k].detach().cpu().numpy() for k in keys}, gender=np.array(motion['gender']), fps=np.array(30))


def load_bridge_object_sdf(root, object_name, device):
    """Native metrics and contact correction consume a [1,D,H,W] object SDF."""
    values, info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', object_name)
    return torch.as_tensor(values, device=device, dtype=torch.float32)[None], info


def acquisition_measures(motion, condition, object_vertices, hand_distance_m=.08):
    world = object_vertices @ motion['object_rotation'].transpose(-1, -2)+motion['object_translation'][:, None]
    distances = hand_object_distances(motion['joints'], world)
    target = object_vertices @ condition['object_rotation'][51].T+condition['object_translation'][51]
    desired = hand_object_distances(condition['joints'][51:52], target[None])[0] <= hand_distance_m
    contact = distances <= hand_distance_m
    matching = (contact | ~desired).all(-1)
    frames = torch.where(matching[10:])[0]+10
    return dict(source_contacting_hands=desired.tolist(),
        suffix_hand_distance_max_m=distances[51:].amax(0).tolist(),
        suffix_contact_recovered=bool(desired.any() and matching[51:].all()),
        contact_at_bridge_start=bool(matching[9]),
        first_matching_contact_frame=int(frames[0]) if len(frames) else None,
        hand_distances_m=distances.tolist(),
        object_translation_change_m=float((motion['object_translation']-condition['object_translation']).abs().max()),
        object_rotation_change=float((motion['object_rotation']-condition['object_rotation']).abs().max()))


def rotation_seam_measures(motion):
    rotations = transforms.axis_angle_to_matrix(motion['pose'])
    result = {}
    for label, first, last in [('entry', 9, 10), ('exit', 50, 51)]:
        delta = rotations[last] @ rotations[first].transpose(-1, -2)
        angles = transforms.matrix_to_axis_angle(delta).norm(dim=-1)*180/torch.pi
        result[label+'_rotation_jump_max_deg'] = float(angles.max())
        result[label+'_rotation_jump_mean_deg'] = float(angles.mean())
        result[label+'_root_rotation_jump_deg'] = float(angles[0])
    return result


def render_bridge(root, output, episode, motion, source_frames, scene_mesh_root, stages=None, video_name=None):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
    import trimesh
    from utils import zup_to_yup
    from .kinematic_composition import _PARENTS_22
    joints = motion['joints'][:, :22]
    points = joints[:, :, [0, 2, 1]]
    scene = trimesh.load_mesh(Path(scene_mesh_root)/(episode['scene_name']+'.obj'))
    vertices, faces = np.asarray(scene.vertices), np.asarray(scene.faces)
    center = joints[:, 0].mean(0)
    centers = vertices[faces].mean(1)
    visible = (np.linalg.norm(centers[:, [0, 2]]-center[[0, 2]], axis=-1) < 2.5) & (centers[:, 1] < 2.)
    faces = faces[visible][::max(1, int(visible.sum())//6000)]
    obj = trimesh.load_mesh(root/episode['persistent_objects'][0]['geometry'])
    object_rest = zup_to_yup(np.asarray(obj.vertices))
    object_faces = np.asarray(obj.faces)
    obj_world = object_rest @ motion['object_rotation'][0].T+motion['object_translation'][0]
    fig = plt.figure(figsize=(8, 6), dpi=100)
    ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(Poly3DCollection(vertices[faces][..., [0, 2, 1]], facecolor='#999999', alpha=.17, edgecolor='none'))
    object_artist = Poly3DCollection(obj_world[object_faces][..., [0, 2, 1]], facecolor='#bd8032', alpha=.65, edgecolor='none')
    ax.add_collection3d(object_artist)
    artist = Line3DCollection([], colors='#187b82', linewidths=2.5)
    ax.add_collection3d(artist)
    radius = max(1., float(np.ptp(joints[:, 0, [0, 2]], axis=0).max())/2+.65)
    ax.set(xlim=(center[0]-radius, center[0]+radius), ylim=(center[2]-radius, center[2]+radius),
        zlim=(-.05, 2.05), xlabel='X (m)', ylabel='Z (m)', zlabel='Y (m)')
    ax.set_box_aspect((radius, radius, 1.05))
    ax.view_init(elev=18, azim=-60)
    parents = np.array(_PARENTS_22[1:])
    title = fig.suptitle(episode['episode_id'])
    writer = FFMpegWriter(fps=15, codec='libx264', extra_args=['-pix_fmt', 'yuv420p', '-crf', '22'])
    filename = video_name or ('source_and_bridge.mp4' if stages is None else 'actual_chain.mp4')
    with writer.saving(fig, str(output/filename), dpi=100):
        for frame in range(0, len(points), 2):
            artist.set_segments(np.stack((points[frame, 1:], points[frame, parents]), axis=1))
            phase = 'LINGO source reference' if frame < source_frames else 'Kimodo acquisition bridge'
            if stages is not None:
                phase = next(s['label'] for s in stages if s['start_frame'] <= frame < s['stop_frame'])
                world = object_rest @ motion['object_rotation'][frame].T+motion['object_translation'][frame]
                object_artist.set_verts(world[object_faces][..., [0, 2, 1]])
            title.set_text(f'{episode["original_hosi_task_id"]} | {phase} | {frame/30:.2f} s')
            writer.grab_frame()
        fig.savefig(output/('final_grasp.png' if stages is None else 'final_frame.png'), dpi=150)
    plt.close(fig)


def adapt_bridge(cfg, root, dest, episode, condition, canonical, origin, model, prediction, source_frames,
                 *, require_suffix_contact=True):
    import trimesh
    from utils import zup_to_yup
    scene = _load_scene(root, episode['scene_name'], cfg.device)
    obj = episode['persistent_objects'][0]
    rest = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(root/obj['geometry']).vertices)), device=cfg.device, dtype=torch.float32)
    object_sdf, info = load_bridge_object_sdf(root, obj['object_id'], cfg.device)
    targets = prediction['target_joints'] @ canonical+origin
    roots = prediction['root_positions'] @ canonical+origin
    rotation = prediction['local_rot_mats'].clone()
    rotation[:, 0] = canonical.T @ rotation[:, 0]
    fitted, translation, fit_audit = fit_native(rotation, roots, targets,
        transforms.axis_angle_to_matrix(condition['pose']), condition['translation'], native_rest_offsets(model, condition['betas']), cfg)
    motion = dict(condition, pose=transforms.matrix_to_axis_angle(fitted), translation=translation)
    motion['pose'][:10], motion['pose'][51:] = condition['pose'][:10], condition['pose'][51:]
    vertices, motion['joints'] = decode_body(motion, model)
    np.savez(dest/'adapted_motion.npz', **native_arrays(motion))
    patches = surface_patches(model, vertices[51], cfg.inbetween.contact.patch_vertices)
    mask, intervals = contact_intervals(prediction['foot_contacts'], cfg.inbetween.contact.minimum_contact_frames)
    before = motion_metrics(motion, vertices, condition, cfg, rest, *scene, object_sdf, info)
    before.update(contact_measures(vertices, patches, mask))
    before.update(rotation_seam_measures(motion))
    corrected, vertices, correction = correct_motion(motion, model, patches, mask, *scene, object_sdf, info, cfg, dest)
    after = motion_metrics(corrected, vertices, condition, cfg, rest, *scene, object_sdf, info)
    after.update(contact_measures(vertices, patches, mask))
    after.update(rotation_seam_measures(corrected))
    geometry = full_source_geometry(dict(corrected, verts=vertices), scene, object_sdf[:, None], info, rest,
        corrected['object_translation'], corrected['object_rotation'], cfg.multitask.source_eligibility)
    contacts = acquisition_measures(corrected, condition, rest,
        hand_distance_m=float(cfg.multitask.source_eligibility.hand_distance_m))
    contact_pass = contacts['suffix_contact_recovered'] if require_suffix_contact else True
    gates = dict(contexts=after['endpoint_joint_max_error_m'] < 1e-5,
        finite=after['finite'], full_frame_geometry=geometry['passes'], suffix_contact=contact_pass,
        object_fixed=contacts['object_translation_change_m'] == 0 and contacts['object_rotation_change'] == 0)
    np.savez(dest/'motion.npz', **native_arrays(corrected))
    write_json(dest/'contacts.json', dict(intervals=intervals, mask=mask.tolist(),
        patch_vertices=patches.tolist(),
        free_contact_frame_fraction=float(mask[10:51].any(-1).float().mean()), **contacts))
    row = dict(episode_id=episode['episode_id'], before=before, after=after, geometry=geometry, acquisition=contacts,
        fit=fit_audit, correction=correction, gates=gates, bridge_gate=all(gates.values()), source_frames=source_frames,
        object_support=obj['support'], new_bridge_frames=51 if require_suffix_contact else 41,
        suffix_contact_required=require_suffix_contact, artifact=str(dest))
    write_json(dest/'metrics.json', row)
    return corrected, row


def run_source_bridges(cfg):
    from omegaconf import OmegaConf
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered bridge run requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = Path(cfg.multitask.bridge.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'
    inputs.mkdir()
    manifest = json.loads(Path(cfg.multitask.bridge.source_manifest).read_text())
    episodes = [r for r in manifest['episodes'] if r['direction'] == GRASPED_ENTRY]
    write_json(output/'frozen_candidate_ids.json', dict(episode_ids=[r['episode_id'] for r in episodes],
        source_manifest=str(cfg.multitask.bridge.source_manifest), source_membership_changed=False))
    torch.cuda.synchronize(cfg.device)
    torch.cuda.reset_peak_memory_stats(cfg.device)
    started = time.perf_counter()
    models, prepared, arrays = {}, [], {}
    for episode in episodes:
        witness = torch.load(Path(manifest['artifact_root'])/episode['construction']['witness'], map_location=cfg.device, weights_only=False)
        source, target = witness['lingo_motion'], witness['target_context']
        if target['gender'] not in models:
            models[target['gender']] = create_smplx_model(target['gender'], torch.device(cfg.device)).eval().requires_grad_(False)
        model = models[target['gender']]
        obj = episode['persistent_objects'][0]
        position = target['translation'].new_tensor(obj['planned_translation'])
        rotation = target['translation'].new_tensor(obj['planned_rotation'])
        condition, canonical, origin, sample = bridge_condition(source, target, position, rotation, model)
        for key, value in sample.items():
            arrays.setdefault(key, []).append(value.cpu().numpy())
        torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in condition.items() if k != 'verts'}, inputs/(episode['episode_id']+'.pt'))
        prepared.append((episode, condition, canonical, origin, source))
    if episodes:
        np.savez(inputs/'conditions.npz', **{k: np.stack(v) for k,v in arrays.items()})
        generation_cfg = OmegaConf.merge(cfg, dict(hosi_output_dir=str(output), inbetween=dict(input_dir=str(inputs), models=['kimodo'])))
        dispatch(generation_cfg, root)
        raw = {k:torch.from_numpy(v).to(cfg.device) for k,v in np.load(output/'kimodo/prediction.npz').items()}
    records = []
    native_peak = torch.cuda.max_memory_allocated(cfg.device)
    for ordinal, (episode, condition, canonical, origin, source) in enumerate(prepared):
        dest = output/episode['episode_id']
        dest.mkdir()
        model = models[condition['gender']]
        corrected, row = adapt_bridge(cfg, root, dest, episode, condition, canonical, origin, model,
            prediction_row(raw, ordinal, cfg.device), len(source['pose']))
        native_peak = max(native_peak, row['correction']['peak_cuda_allocated_bytes'])
        final = native_arrays(corrected)
        full = dict(final)
        for key in ('pose', 'translation', 'joints'):
            full[key] = np.concatenate((source[key].cpu().numpy(), final[key][10:]))
        for key in ('object_translation', 'object_rotation'):
            full[key] = np.repeat(final[key][:1], len(full['pose']), axis=0)
        np.savez(dest/'source_and_bridge.npz', **full)
        records.append(row)
        render_bridge(root, dest, episode, full, len(source['pose']), cfg.multitask.scene_mesh_root)
        print(json.dumps(dict(episode_id=episode['episode_id'], gates=row['gates'])), flush=True)
    torch.cuda.synchronize(cfg.device)
    write_json(output/'metrics.json', dict(status='completed', subphase='5.5.2a.1', git_commit=commit,
        git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        seed=int(cfg.seed), candidate_count=len(episodes), bridge_samples=len(episodes),
        bridge_passes=sum(r['bridge_gate'] for r in records), records=records, elapsed_seconds=time.perf_counter()-started,
        source_manifest=str(cfg.multitask.bridge.source_manifest), membership_changed=False, full_expert_chain_executed=False,
        device=str(cfg.device), kimodo_device=str(cfg.inbetween.kimodo_device),
        peak_cuda_allocated_bytes=max(native_peak, torch.cuda.max_memory_allocated(cfg.device))))


def dataset_order(episodes):
    """Fix coverage order before any interpolation output is read."""
    from collections import Counter
    pool, ordered = list(episodes), []
    directions, scenes, tasks, sources = Counter(), Counter(), Counter(), Counter()
    while pool:
        def key(episode):
            source = next(s for s in episode['segments'] if s['source_dataset'] == 'LINGO')
            return (directions[episode['direction']], scenes[episode['scene_name']],
                tasks[episode['original_hosi_task_id']], source['task_type'] == 'locomotion',
                sources[source['source_id']], episode['episode_id'])
        episode = min(pool, key=key)
        ordered.append(episode)
        pool.remove(episode)
        directions[episode['direction']] += 1
        scenes[episode['scene_name']] += 1
        tasks[episode['original_hosi_task_id']] += 1
        sources[next(s['source_id'] for s in episode['segments'] if s['source_dataset'] == 'LINGO')] += 1
    return ordered


def stitch_dataset_motion(first, bridge, second):
    """Append only the 41 new bridge frames between two intact source clips."""
    tracks = ('pose', 'translation', 'joints', 'object_translation', 'object_rotation')
    result = {key:value for key,value in first.items() if key != 'verts'}
    for key in tracks:
        result[key] = torch.cat((first[key], bridge[key][10:51], second[key]))
    return result


def inference_episode(episode, first):
    """Export model inputs with no future source or interpolation trajectory."""
    segments, cursor = [], 0
    for index, source in enumerate(episode['segments']):
        if index:
            cursor += 41
        fields = ('segment_id', 'task_type', 'text', 'pelvis_goal', 'object_goal')
        segment = {key:source[key] for key in fields}
        segment.update(start_frame=cursor, stop_frame=cursor+source['frame_count'])
        if source['source_dataset'] == 'OMOMO':
            segment['object_rotation_goal'] = source['object_rotation_goal']
        else:
            segment['object_policy'] = 'stationary_at_achieved_boundary_transform'
            segment['contact_targets'] = [dict(text=c['text'], support_points=c['support_points'],
                frame_interval=[min(c['frame_indices']), max(c['frame_indices'])+1])
                for c in source['contact_targets']]
        segments.append(segment)
        cursor = segment['stop_frame']
    obj = episode['persistent_objects'][0]
    initial = {key:first[key][:10].detach().cpu().tolist()
        for key in ('pose', 'translation', 'object_translation', 'object_rotation')}
    return dict(episode_id=episode['episode_id'], scene_name=episode['scene_name'], fps=30,
        scene_sdf='data/hosi_test/Scene_sdf/'+episode['scene_name']+'_sdf.npy',
        scene_sdf_info='data/hosi_test/Scene_sdf/'+episode['scene_name']+'_sdf_info.json',
        body_identity=episode['body_identity'], object=dict(object_id=obj['object_id'], geometry=obj['geometry']),
        instruction='; then '.join(s['text'] for s in segments), initial_context=initial,
        segments=segments, transition=dict(start_frame=segments[0]['stop_frame'],
            stop_frame=segments[1]['start_frame'], posture='standing', hands_may_contact_object=True),
        frame_count=cursor, duration_s=(cursor-1)/30)


@torch.no_grad()
def dataset_motion_metrics(motion, task, model, scene, rest, object_sdf, object_info, thresholds):
    """Measure a native prediction using task conditions, without source GT."""
    vertices = torch.cat([decode_body(dict(motion, pose=motion['pose'][i:i+128],
        translation=motion['translation'][i:i+128]), model)[0] for i in range(0, len(motion['pose']), 128)])
    geometry = full_source_geometry(dict(motion, verts=vertices), scene, object_sdf, object_info, rest,
        motion['object_translation'], motion['object_rotation'], thresholds)
    joints = motion['joints']
    feet = joints[:, [7, 8, 10, 11]]
    contact = feet[..., 1].abs() <= .08
    foot_pairs = contact[1:] & contact[:-1]
    speed = (feet[1:, :, [0, 2]]-feet[:-1, :, [0, 2]]).norm(dim=-1)*30
    world = rest @ motion['object_rotation'].transpose(-1, -2)+motion['object_translation'][:, None]
    hands = hand_object_distances(joints, world)
    stages = []
    for segment in task['segments']:
        begin, stop = segment['start_frame'], segment['stop_frame']
        goal = joints.new_tensor(segment['pelvis_goal'])
        row = dict(segment_id=segment['segment_id'],
            pelvis_goal_error_m=float((joints[stop-1, 0, [0, 2]]-goal[[0, 2]]).norm()))
        if segment['object_goal'] is not None:
            row['object_goal_error_m'] = float((motion['object_translation'][stop-1]-joints.new_tensor(segment['object_goal'])).norm())
            row['hand_contact_frame_fraction'] = float((hands[begin:stop].min(-1).values <= .08).float().mean())
        else:
            row['stationary_object_displacement_max_m'] = float((motion['object_translation'][begin:stop]-motion['object_translation'][begin]).norm(dim=-1).max())
        stages.append(row)
    return dict(geometry=geometry, segments=stages,
        foot_floor_distance_max_m=float(feet[..., 1].abs().amin(-1).max()),
        near_floor_foot_speed_m_s=float(speed[foot_pairs].mean()) if foot_pairs.any() else None,
        near_floor_contact_frame_fraction=float(contact.any(-1).float().mean()),
        finite=bool(torch.isfinite(joints).all()), observed_frames=len(joints), expected_frames=task['frame_count'])


def run_dataset_bridges(cfg):
    from omegaconf import OmegaConf
    from utils import create_smplx_model, zup_to_yup
    from .source_eligibility import dataset_support
    import trimesh
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered dataset bridges require a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    settings = cfg.multitask.dataset
    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(settings.source_manifest).read_text())
    ordered = dataset_order(manifest['episodes'])
    episodes = [e for i,e in enumerate(ordered) if i % settings.lane_count == settings.lane_index]
    write_json(output/'candidate_order.json', dict(all_candidate_ids=[e['episode_id'] for e in ordered],
        lane_candidate_ids=[e['episode_id'] for e in episodes], lane_index=int(settings.lane_index)))
    torch.cuda.synchronize(cfg.device)
    started = time.perf_counter()
    models, rows = {}, []
    for batch_start in range(0, len(episodes), int(settings.bridge_batch_size)):
        group = output/f'batch-{batch_start//settings.bridge_batch_size:03d}'
        inputs = group/'inputs'
        inputs.mkdir(parents=True)
        arrays, prepared = {}, []
        for episode in episodes[batch_start:batch_start+settings.bridge_batch_size]:
            witness = torch.load(Path(manifest['artifact_root'])/episode['construction']['witness'],
                map_location=cfg.device, weights_only=False)
            first, second = witness['first'], witness['second']
            gender = first['gender']
            if gender not in models:
                models[gender] = create_smplx_model(gender, torch.device(cfg.device)).eval().requires_grad_(False)
            obj = episode['persistent_objects'][0]
            condition, canonical, origin, sample = bridge_condition(first, second,
                first['translation'].new_tensor(obj['planned_translation']),
                first['translation'].new_tensor(obj['planned_rotation']), models[gender], object_contexts=(first, second))
            for key,value in sample.items():
                arrays.setdefault(key, []).append(value.cpu().numpy())
            prepared.append((episode, first, second, condition, canonical, origin))
        np.savez(inputs/'conditions.npz', **{k:np.stack(v) for k,v in arrays.items()})
        generation_cfg = OmegaConf.merge(cfg, dict(hosi_output_dir=str(group),
            inbetween=dict(input_dir=str(inputs), models=['kimodo'])))
        dispatch(generation_cfg, root)
        raw = dict(np.load(group/'kimodo/prediction.npz'))
        for index,(episode, first, second, condition, canonical, origin) in enumerate(prepared):
            dest = output/episode['episode_id']
            dest.mkdir()
            model = models[first['gender']]
            corrected, bridge = adapt_bridge(cfg, root, dest, episode, condition, canonical, origin, model,
                prediction_row(raw, index, cfg.device), len(first['pose']), require_suffix_contact=False)
            scene = _load_scene(root, episode['scene_name'], cfg.device)
            obj = episode['persistent_objects'][0]
            rest = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(root/obj['geometry']).vertices)), device=cfg.device, dtype=torch.float32)
            obj_sdf, obj_info = load_bridge_object_sdf(root, obj['object_id'], cfg.device)
            full = stitch_dataset_motion(first, corrected, second)
            task = inference_episode(episode, first)
            metrics = dataset_motion_metrics(full, task, model, scene, rest, obj_sdf[:, None], obj_info, cfg.multitask.source_eligibility)
            vertices, _ = decode_body(corrected, model)
            bridge_support = dataset_support(dict(corrected, verts=vertices), [], model, scene, settings)
            tracks = ('pose', 'translation', 'joints', 'object_translation', 'object_rotation')
            source_preserved = all(torch.equal(full[key][:len(first[key])], first[key])
                and torch.equal(full[key][-len(second[key]):], second[key]) for key in tracks)
            contexts = all(torch.equal(corrected[key][:10], first[key][-10:])
                and torch.equal(corrected[key][51:], second[key][:10])
                for key in ('pose', 'translation', 'object_translation', 'object_rotation'))
            gates = dict(bridge=bridge['bridge_gate'], full_motion_geometry=metrics['geometry']['passes'],
                bridge_foot_support=bridge_support['passes'], source_preserved=source_preserved,
                contexts_preserved=contexts, frame_count=len(full['pose']) == episode['frame_count'])
            row = dict(episode_id=episode['episode_id'], accepted=all(gates.values()), gates=gates,
                bridge_metrics=str(dest/'metrics.json'), bridge_support=bridge_support,
                complete_motion_metrics=metrics, motion=str(dest/'source_composition.npz'),
                inference_task=task, direction=episode['direction'], artifact=str(dest))
            np.savez(dest/'source_composition.npz', **native_arrays(full))
            write_json(dest/'construction_result.json', row)
            rows.append(row)
            print(json.dumps(dict(episode_id=episode['episode_id'], accepted=row['accepted'], gates=gates)), flush=True)
    torch.cuda.synchronize(cfg.device)
    summary = dict(status='completed', subphase='5.6.1', git_commit=commit,
        seed=int(cfg.seed), lane_index=int(settings.lane_index), source_candidates=len(episodes),
        accepted=sum(r['accepted'] for r in rows), records=rows, expert_samples=0, kimodo_samples=len(rows),
        elapsed_seconds=time.perf_counter()-started, device=str(cfg.device),
        git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip())
    write_json(output/'summary.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)


def render_dataset_case(arguments):
    root, destination, episode, motion_path, mesh_root = arguments
    motion = dict(np.load(motion_path))
    first, second = episode['segments']
    count = first['frame_count']
    stages = [dict(label=first['source_dataset']+' dataset motion', start_frame=0, stop_frame=count),
        dict(label='Kimodo inbetween', start_frame=count, stop_frame=count+41),
        dict(label=second['source_dataset']+' dataset motion', start_frame=count+41, stop_frame=len(motion['pose']))]
    render_bridge(Path(root), Path(destination), episode, motion, count, mesh_root, stages=stages, video_name='preview.mp4')
    return str(Path(destination)/'preview.mp4')


def publish_dataset(cfg):
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    root = Path(__file__).resolve().parents[2]
    settings = cfg.multitask.dataset
    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(settings.source_manifest).read_text())
    ordered = dataset_order(manifest['episodes'])
    results = [r for directory in settings.bridge_roots
        for r in json.loads((Path(directory)/'summary.json').read_text())['records']]
    by_id = {r['episode_id']:r for r in results}
    selected = [e for e in ordered if by_id[e['episode_id']]['accepted']][:settings.target_episodes]
    tasks = [by_id[e['episode_id']]['inference_task'] for e in selected]
    write_json(output/'inference_tasks.json', dict(schema_version=1, seed=int(cfg.seed), episodes=tasks,
        input_contract='initial_context_and_task_goals_only', construction_motion_is_imitation_gt=False))
    write_json(output/'original_hosi_tasks.json', manifest['original_hosi'])
    write_json(output/'construction_manifest.json', dict(schema_version=3, source_manifest=str(settings.source_manifest),
        bridge_roots=list(settings.bridge_roots), episodes=selected,
        results=[{k:v for k,v in r.items() if k != 'inference_task'} for r in results],
        selected_ids=[e['episode_id'] for e in selected],
        reserve_ids=[e['episode_id'] for e in ordered if by_id[e['episode_id']]['accepted'] and e not in selected],
        rejected_ids=[r['episode_id'] for r in results if not r['accepted']]))
    static = [e for e in selected if any(s['task_type'] == 'locomotion_static_interaction' for s in e['segments'])]
    preview_episodes = (static+ [e for e in selected if e not in static])[:settings.render_limit]
    arguments = [(str(root), by_id[e['episode_id']]['artifact'], e, by_id[e['episode_id']]['motion'], cfg.multitask.scene_mesh_root)
        for e in preview_episodes]
    with ProcessPoolExecutor(max_workers=4) as executor:
        videos = list(executor.map(render_dataset_case, arguments))
    lines = ['# Dataset-motion benchmark', '', f'{len(selected)} constructed episodes; {len(static)} include static interaction.', '',
        '[Inference conditions](inference_tasks.json) · [Construction ledger](construction_manifest.json)', '',
        '| Episode | Order | LINGO content | Preview |', '|---|---|---|---|']
    video_map = {e['episode_id']:v for e,v in zip(preview_episodes, videos)}
    for e in selected:
        source = next(s for s in e['segments'] if s['source_dataset'] == 'LINGO')
        preview = f'[video]({video_map[e["episode_id"]]})' if e['episode_id'] in video_map else ''
        lines.append(f'| {e["episode_id"]} | {e["direction"]} | {source["text"]} | {preview} |')
    (output/'review.md').write_text('\n'.join(lines)+'\n')
    summary = dict(status='completed', subphase='5.6.1', seed=int(cfg.seed),
        git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        original_hosi_tasks=manifest['original_hosi']['task_count'], source_candidates=len(ordered),
        bridge_attempts=len(results), successful_constructions=sum(r['accepted'] for r in results),
        published_episodes=len(selected), published_by_direction=dict(Counter(e['direction'] for e in selected)),
        static_interaction_episodes=len(static), internal_cut_episodes=sum(e['construction']['internal_lingo_cut'] for e in selected),
        original_task_coverage=len({e['original_hosi_task_id'] for e in selected}), scene_coverage=len({e['scene_name'] for e in selected}),
        lingo_span_coverage=len({s['source_id'] for e in selected for s in e['segments'] if s['source_dataset'] == 'LINGO'}),
        native_frames=sum(e['frame_count'] for e in selected), previews=len(videos), expert_samples=0,
        failure_gates=dict(Counter(k for r in results for k,v in r['gates'].items() if not v)),
        inference_tasks=str(output/'inference_tasks.json'), review=str(output/'review.md'))
    write_json(output/'summary.json', summary)
    print(json.dumps(summary), flush=True)
