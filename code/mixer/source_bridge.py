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
def bridge_condition(prefix, suffix, position, rotation, model):
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
    world = object_vertices @ motion['object_rotation'][0].T+motion['object_translation'][0]
    distances = hand_object_distances(motion['joints'], world[None].expand(len(motion['joints']), -1, -1))
    desired = hand_object_distances(condition['joints'][51:52], world[None])[0] <= hand_distance_m
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


def render_bridge(root, output, episode, motion, source_frames, scene_mesh_root, stages=None):
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
    with writer.saving(fig, str(output/('source_and_bridge.mp4' if stages is None else 'actual_chain.mp4')), dpi=100):
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


def adapt_bridge(cfg, root, dest, episode, condition, canonical, origin, model, prediction, source_frames):
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
    gates = dict(contexts=after['endpoint_joint_max_error_m'] < 1e-5,
        finite=after['finite'], full_frame_geometry=geometry['passes'], suffix_contact=contacts['suffix_contact_recovered'],
        object_fixed=contacts['object_translation_change_m'] == 0 and contacts['object_rotation_change'] == 0)
    np.savez(dest/'motion.npz', **native_arrays(corrected))
    write_json(dest/'contacts.json', dict(intervals=intervals, mask=mask.tolist(),
        patch_vertices=patches.tolist(),
        free_contact_frame_fraction=float(mask[10:51].any(-1).float().mean()), **contacts))
    row = dict(episode_id=episode['episode_id'], before=before, after=after, geometry=geometry, acquisition=contacts,
        fit=fit_audit, correction=correction, gates=gates, bridge_gate=all(gates.values()), source_frames=source_frames,
        object_support=obj['support'], new_bridge_frames=51, artifact=str(dest))
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
