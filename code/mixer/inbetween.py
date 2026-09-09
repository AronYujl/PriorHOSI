"""Native endpoint preparation, pretrained bridge dispatch and SMPL-X evaluation."""

import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .body_projection import native_rest_offsets
from .continuation_outcomes import write_json
from .kinematic_composition import _forward_kinematics, _PARENTS_22
from .standing_transition import observed_motion, scene_measures, object_measures, tail_measures
from .surface_edit import decode_body, load_object_sdf, yaw_matrix


def heading(joints):
    across = joints[..., 2, :]-joints[..., 1, :] + joints[..., 17, :]-joints[..., 16, :]
    return torch.atan2(across[..., 2], -across[..., 0])


def interpolate_rotations(first, last, weights):
    angle = transforms.matrix_to_axis_angle(first.transpose(-1, -2) @ last)
    return first[None] @ transforms.axis_angle_to_matrix(angle[None]*weights[:, None, None])


def fk(rotation, translation, offsets):
    rest = offsets[None].expand(len(rotation), -1, -1).clone()
    rest[:, 0] += translation
    global_rotation, points = _forward_kinematics(rotation, rest)
    return global_rotation[:, :22], points[:, :22]


def select_reference(sources):
    """Select an actual upright, low-wrist pose before any bridge is sampled."""
    candidates = []
    for row, motion in sources:
        joints = motion['joints']
        spine = joints[:, 12]-joints[:, 0]
        tilt = torch.acos((spine[:, 1]/spine.norm(dim=-1)).clamp(-1, 1))
        foot = joints[:, [7, 8, 10, 11], 1].abs().amin(-1)
        wrist_height = joints[:, [20, 21], 1]-joints[:, :1, 1]
        wrist = wrist_height.mean(-1)
        thigh = torch.nn.functional.normalize(joints[:, [4, 5]]-joints[:, [1, 2]], dim=-1)
        shin = torch.nn.functional.normalize(joints[:, [7, 8]]-joints[:, [4, 5]], dim=-1)
        knee = torch.acos((thigh*shin).sum(-1).clamp(-1, 1)).amax(-1)
        valid = ((tilt < np.deg2rad(10)) & (foot < .08) & (joints[:, 0, 1] > .65)
            & (knee < np.deg2rad(25)) & (wrist_height.amax(-1) < .1))
        for frame in torch.where(valid)[0].tolist():
            candidates.append((float(wrist[frame]+.25*tilt[frame]), row['task'], frame,
                float(tilt[frame]*180/np.pi), float(wrist[frame]), float(knee[frame]*180/np.pi)))
    chosen = min(candidates)
    index = next(i for i, (row, _) in enumerate(sources) if row['task'] == chosen[1])
    return sources[index][1]['pose'][chosen[2]].clone(), sources[index][1]['joints'][chosen[2]], dict(
        task=chosen[1], frame=chosen[2], upright_tilt_deg=chosen[3], wrist_relative_height_m=chosen[4], knee_flexion_deg=chosen[5],
        score=chosen[0], eligible_pose_count=len(candidates), criterion='min mean wrist height relative to pelvis + 0.25*tilt radians; torso<10deg, knees<25deg, wrists<pelvis+0.1m and standing support')


@torch.no_grad()
def prepare_inputs(cfg, root):
    from utils import create_smplx_model
    previous = json.loads(Path(cfg.inbetween.source_results).read_text())
    records = {r['task']: r for r in previous['records'] if r['prompt_index'] == 0}
    sources = []
    models = {}
    for task_id in previous['source_audit']['selected']:
        previous_motion = torch.load(Path(records[task_id]['metrics_path']).with_name('motion.pt'), map_location='cpu', weights_only=False)
        row = previous_motion['source']
        motion = torch.load(row['path'], map_location=cfg.device, weights_only=False)[cfg.stand_wait.source_arm]
        motion = observed_motion(motion)
        motion['gender'] = previous_motion['motion']['gender']
        sources.append((row, motion))
        if motion['gender'] not in models:
            models[motion['gender']] = create_smplx_model(motion['gender'], torch.device(cfg.device)).eval().requires_grad_(False)
    pose_ref, joints_ref, reference_record = select_reference(sources)
    output = Path(cfg.inbetween.input_dir)
    output.mkdir(parents=True)
    write_json(output/'standing_reference.json', reference_record)
    arrays = {k: [] for k in ('local_rot_mats', 'root_positions', 'joints', 'neutral_joints', 'known')}
    metadata = []
    for row, source in sources:
        model = models[source['gender']]
        offsets = native_rest_offsets(model, source['betas'])
        reference = transforms.axis_angle_to_matrix(pose_ref)
        target_rotation = reference.clone()
        target_rotation[0] = yaw_matrix((heading(source['joints'][-1])-heading(joints_ref)).reshape(1))[0] @ reference[0]
        target = dict(source, pose=transforms.matrix_to_axis_angle(target_rotation)[None],
            translation=source['translation'][-1:].clone())
        target_vertices, target_joints = decode_body(target, model)
        target['translation'][:, 1] -= target_vertices[..., 1].min()
        target_vertices, target_joints = decode_body(target, model)
        time_weight = ((torch.arange(cfg.inbetween.frames, device=cfg.device)-9)/42).clamp(0, 1)
        source_rotation = transforms.axis_angle_to_matrix(source['pose'][-10:])
        rotation = interpolate_rotations(source_rotation[-1], target_rotation, time_weight)
        translation = source['translation'][-1:]+time_weight[:, None]*(target['translation']-source['translation'][-1:])
        rotation[:10] = source_rotation
        translation[:10] = source['translation'][-10:]
        condition = dict(source, pose=transforms.matrix_to_axis_angle(rotation), translation=translation,
            object_translation=source['object_translation'][-1:].expand(cfg.inbetween.frames, -1).clone(),
            object_rotation=source['object_rotation'][-1:].expand(cfg.inbetween.frames, -1, -1).clone())
        condition['verts'], condition['joints'] = decode_body(condition, model)
        canonical_yaw = -heading(condition['joints'][0])
        canonical_rotation = yaw_matrix(canonical_yaw.reshape(1))[0]
        origin = condition['joints'][0, 0].clone(); origin[1] = 0
        local_rotations = rotation.clone(); local_rotations[:, 0] = canonical_rotation @ rotation[:, 0]
        canonical_joints = (canonical_rotation @ (condition['joints'][:, :22]-origin)[..., None]).squeeze(-1)
        neutral = model.J_regressor @ (model.v_template+torch.einsum('vci,i->vc', model.shapedirs[..., :len(source['betas'])], source['betas']))
        known = (torch.arange(len(rotation), device=cfg.device) < 10) | (torch.arange(len(rotation), device=cfg.device) >= 51)
        for key, value in dict(local_rot_mats=local_rotations, root_positions=canonical_joints[:, 0],
                joints=canonical_joints, neutral_joints=neutral[:22], known=known).items():
            arrays[key].append(value.cpu().numpy())
        dest = output/f'task-{row["task"]:03d}'
        dest.mkdir()
        torch.save(dict(source=row, motion={k: v.cpu() if torch.is_tensor(v) else v for k, v in condition.items() if k != 'verts'},
            source_tail={k: v[-10:].cpu() if torch.is_tensor(v) and v.ndim and v.shape[0] == len(source['pose']) and k != 'betas'
                else v.cpu() if torch.is_tensor(v) else v for k, v in source.items()},
            canonical_rotation=canonical_rotation.cpu(), origin=origin.cpu()), dest/'condition.pt')
        meta = dict(row, gender=source['gender'], target_reference=reference_record,
            target_surface_min_y_m=float(target_vertices[..., 1].min()),
            source_reconstruction_max_m=float((condition['joints'][:10]-source['joints'][-10:]).norm(dim=-1).max()))
        metadata.append(meta)
        print(json.dumps(dict(prepared_task=row['task'], source_error_m=meta['source_reconstruction_max_m'])), flush=True)
    np.savez(output/'conditions.npz', **{k: np.stack(v) for k, v in arrays.items()})
    write_json(output/'index.json', dict(tasks=metadata, fps=30, frames=61, prefix_frames=10, suffix_start=51,
        standing_reference=reference_record, source_results=str(cfg.inbetween.source_results)))


def dispatch(cfg, root):
    external_root = Path(cfg.inbetween.external_root)
    output = Path(cfg.hosi_output_dir)
    processes = []
    for name in cfg.inbetween.models:
        external_repo = external_root/('kimodo' if name == 'kimodo' else 'diffusion-motion-inbetweening')
        dest = output/name
        command = [cfg.inbetween[name+'_python'], str(root/'code/mixer/inbetween_external.py'), '--model', name,
            '--input', str(Path(cfg.inbetween.input_dir)/'conditions.npz'), '--output', str(dest),
            '--root', str(external_root), '--device', cfg.inbetween[name+'_device'], '--steps', str(cfg.inbetween.kimodo_steps)]
        environment = dict(os.environ, PYTHONPATH=str(external_repo))
        environment['OMP_NUM_THREADS'] = '4'
        log = (output/(name+'_generation.log')).open('w')
        processes.append((name, subprocess.Popen(command, cwd=external_repo, env=environment, stdout=log, stderr=subprocess.STDOUT), log))
    statuses = {}
    for name, process, log in processes:
        statuses[name] = process.wait()
        log.close()
    write_json(output/'generation_exit_codes.json', statuses)
    if any(statuses.values()):
        raise RuntimeError(f'external generation failed: {statuses}')


def fit_native(rotation, root_positions, targets, condition_rotation, condition_translation, offsets, cfg):
    """Fit only free frames; conditioned poses are the exact boundary values."""
    unknown = torch.arange(10, 51, device=rotation.device)
    six = transforms.matrix_to_rotation_6d(rotation[unknown]).detach().clone().requires_grad_(True)
    translation = (root_positions[unknown]-offsets[0]).detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([six, translation], lr=cfg.inbetween.native_fit_lr)
    base = rotation.detach()
    torch.cuda.synchronize(rotation.device)
    started = time.perf_counter()
    for _ in range(cfg.inbetween.native_fit_iterations):
        fitted_rotation = condition_rotation.clone().index_copy(0, unknown, transforms.rotation_6d_to_matrix(six))
        fitted_translation = condition_translation.clone().index_copy(0, unknown, translation)
        _, points = fk(fitted_rotation, fitted_translation, offsets)
        position_loss = (points[unknown]-targets[unknown]).square().mean()
        rotation_loss = (fitted_rotation[unknown]-base[unknown]).square().mean()
        acceleration_loss = (points[2:]-2*points[1:-1]+points[:-2]).square().mean()
        loss = position_loss+.002*rotation_loss+10*acceleration_loss
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    with torch.no_grad():
        fitted_rotation = condition_rotation.clone().index_copy(0, unknown, transforms.rotation_6d_to_matrix(six))
        fitted_translation = condition_translation.clone().index_copy(0, unknown, translation)
        _, fitted_joints = fk(fitted_rotation, fitted_translation, offsets)
        torch.cuda.synchronize(rotation.device)
    return fitted_rotation.detach(), fitted_translation.detach(), dict(
        native_fit_seconds=time.perf_counter()-started, iterations=cfg.inbetween.native_fit_iterations,
        mean_joint_target_error_m=float((fitted_joints[unknown]-targets[unknown]).norm(dim=-1).mean()),
        final_fit_loss=float(loss), fixed_context_frames=20)


def motion_metrics(motion, vertices, condition, cfg, object_vertices, sdf, info, object_sdf, object_info):
    metrics = tail_measures(motion, object_vertices, cfg.stand_wait.thresholds, condition['joints'][9, 0])
    metrics.update(scene_measures(vertices[9:52], sdf, info))
    metrics.update(object_measures(dict(motion, verts=vertices), object_sdf, object_info))
    joints = motion['joints']
    velocity = (joints[1:]-joints[:-1])*30
    for label, boundary in [('entry', 9), ('exit', 51)]:
        metrics[label+'_velocity_jump_mean_m_s'] = float((velocity[boundary]-velocity[boundary-1]).norm(dim=-1).mean())
        metrics[label+'_root_velocity_jump_m_s'] = float((velocity[boundary, 0]-velocity[boundary-1, 0]).norm())
    indices = torch.cat((torch.arange(10, device=joints.device), torch.arange(51, 61, device=joints.device)))
    metrics['endpoint_joint_max_error_m'] = float((joints[indices]-condition['joints'][indices]).norm(dim=-1).max())
    rotation = transforms.axis_angle_to_matrix(motion['pose'][indices])
    desired = transforms.axis_angle_to_matrix(condition['pose'][indices])
    metrics['endpoint_rotation_max_deg'] = float(transforms.matrix_to_axis_angle(rotation@desired.transpose(-1, -2)).norm(dim=-1).max()*180/np.pi)
    metrics['bridge_body_min_y_m'] = float(vertices[9:52, :, 1].min())
    metrics['bridge_body_max_lowest_y_m'] = float(vertices[9:52, :, 1].amin(-1).max())
    feet = joints[9:52, [7, 8, 10, 11]]
    contact = feet[:-1, :, 1].abs() < .08
    foot_speed = (feet[1:]-feet[:-1])[..., [0, 2]].norm(dim=-1)*30
    metrics['bridge_contact_foot_speed_mean_m_s'] = float(foot_speed[contact].mean()) if contact.any() else None
    metrics['bridge_unsupported_fraction'] = float((feet[..., 1].abs().amin(-1) > .08).float().mean())
    metrics['geometry_success'] = bool(metrics['scene_penetration_mean_m'] <= .005 and metrics['scene_penetration_max_m'] <= .05 and metrics['scene_outside_fraction'] == 0)
    metrics['finite'] = bool(torch.isfinite(joints).all() and torch.isfinite(vertices).all())
    return metrics


def evaluate(cfg, root):
    import trimesh
    from utils import create_smplx_model, zup_to_yup
    output = Path(cfg.hosi_output_dir)
    inputs = Path(cfg.inbetween.input_dir)
    index = json.loads((inputs/'index.json').read_text())
    models, records = {}, []
    for name in cfg.inbetween.models:
        raw = {k: torch.from_numpy(v).to(cfg.device) for k, v in np.load(output/name/'prediction.npz').items()}
        for ordinal, row in enumerate(index['tasks']):
            data = torch.load(inputs/f'task-{row["task"]:03d}/condition.pt', map_location=cfg.device, weights_only=False)
            condition = data['motion']
            if row['gender'] not in models:
                models[row['gender']] = create_smplx_model(row['gender'], torch.device(cfg.device)).eval().requires_grad_(False)
            model = models[row['gender']]
            offsets = native_rest_offsets(model, condition['betas'])
            inverse = data['canonical_rotation'].T
            targets = raw['target_joints'][ordinal]
            roots = raw['root_positions'][ordinal]
            if name == 'condmdi':
                targets = torch.nn.functional.interpolate(targets.flatten(1).T[None], size=61, mode='linear', align_corners=True)[0].T.reshape(61, 22, 3)
                roots = torch.nn.functional.interpolate(roots.T[None], size=61, mode='linear', align_corners=True)[0].T
            targets = (inverse @ targets[..., None]).squeeze(-1)+data['origin']
            roots = (inverse @ roots[..., None]).squeeze(-1)+data['origin']
            condition_rotation = transforms.axis_angle_to_matrix(condition['pose'])
            if name == 'kimodo':
                rotation = raw['local_rot_mats'][ordinal].clone()
                rotation[:, 0] = inverse @ rotation[:, 0]
            else:
                rotation = condition_rotation.clone()
            raw_motion = dict(condition, pose=transforms.matrix_to_axis_angle(rotation), translation=roots-offsets[0])
            raw_vertices, raw_motion['joints'] = decode_body(raw_motion, model)
            fitted_rotation, fitted_translation, audit = fit_native(rotation, roots, targets, condition_rotation,
                condition['translation'], offsets, cfg)
            motion = dict(condition, pose=transforms.matrix_to_axis_angle(fitted_rotation), translation=fitted_translation)
            vertices, motion['joints'] = decode_body(motion, model)
            sdf_root = root/'data/hosi_test/Scene_sdf'
            sdf = torch.from_numpy(np.load(sdf_root/(row['scene']+'_sdf.npy'))).float().to(cfg.device)[None, None]
            info = json.loads((sdf_root/(row['scene']+'_sdf_info.json')).read_text())
            object_sdf, object_info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', row['object'])
            object_sdf = torch.from_numpy(object_sdf).float().to(cfg.device)[None]
            object_vertices = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(root/'data/test/rest_object_geo'/(row['object']+'.ply')).vertices)), device=cfg.device, dtype=torch.float32)
            metrics = motion_metrics(motion, vertices, condition, cfg, object_vertices, sdf, info, object_sdf, object_info)
            endpoints = torch.cat((torch.arange(10, device=cfg.device), torch.arange(51, 61, device=cfg.device)))
            audit['raw_target_endpoint_max_error_m'] = float((targets[endpoints]-condition['joints'][endpoints, :22]).norm(dim=-1).max())
            if name == 'kimodo':
                audit['raw_native_endpoint_metrics'] = motion_metrics(raw_motion, raw_vertices, condition, cfg, object_vertices, sdf, info, object_sdf, object_info)
            dest = output/name/f'task-{row["task"]:03d}'
            dest.mkdir()
            torch.save(dict(motion={k: v.detach().cpu() if torch.is_tensor(v) else v for k, v in motion.items()},
                target_joints=targets.cpu(), model=name, source=row), dest/'motion.pt')
            np.savez(dest/'motion.npz', pose=motion['pose'].cpu().numpy(), translation=motion['translation'].cpu().numpy(),
                joints=motion['joints'].cpu().numpy(), betas=motion['betas'].cpu().numpy(), gender=np.array(row['gender']),
                object_translation=motion['object_translation'].cpu().numpy(), object_rotation=motion['object_rotation'].cpu().numpy(), fps=np.array(30))
            record = dict(task=row['task'], model=name, scene=row['scene'], object=row['object'], metrics=metrics, audit=audit)
            write_json(dest/'metrics.json', record)
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(output/'metrics.json', dict(records=records, seed=42, scope='fixed endpoint-conditioned deployment diagnostic'))


def render(cfg, root):
    """Save paired skeletal animations and the complete twelve-case overview."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    from mpl_toolkits.mplot3d.art3d import Line3DCollection
    import trimesh
    from utils import zup_to_yup

    output = Path(cfg.hosi_output_dir)
    index = json.loads((Path(cfg.inbetween.input_dir)/'index.json').read_text())
    preview = output/'preview'
    preview.mkdir()
    parents = np.array(_PARENTS_22[1:])
    colors = {'kimodo':'#7257c7', 'condmdi':'#007c91'}
    clips = []
    for row in index['tasks']:
        tracks = {name: np.load(output/name/f'task-{row["task"]:03d}/motion.npz')['joints']
                  for name in cfg.inbetween.models}
        condition = torch.load(Path(cfg.inbetween.input_dir)/f'task-{row["task"]:03d}/condition.pt', map_location='cpu', weights_only=False)['motion']
        origin = condition['joints'][9, 0].numpy().copy(); origin[1] = 0
        vertices = zup_to_yup(np.asarray(trimesh.load_mesh(root/'data/test/rest_object_geo'/(row['object']+'.ply')).vertices))
        object_world = vertices @ condition['object_rotation'][-1].numpy().T+condition['object_translation'][-1].numpy()
        object_world = object_world-origin
        stride = max(1, len(object_world)//450)
        clips.append((row, {k:(v-origin)[..., [0,2,1]] for k,v in tracks.items()},
            (condition['joints'].numpy()-origin)[..., [0,2,1]], object_world[::stride][:, [0,2,1]]))

    def create_axis(fig, location, clip):
        row, tracks, condition, obj = clip
        ax = fig.add_subplot(*location, projection='3d')
        ax.set_title(f'Task {row["task"]:03d} · {row["object"]}', fontsize=9)
        ax.scatter(obj[:,0],obj[:,1],obj[:,2],s=.8,c='#aaaaaa',alpha=.18)
        for frame, color in [(9, '#777777'), (51, '#348a4d')]:
            pose = condition[frame, :22]
            lines = np.stack((pose[1:],pose[parents]),1)
            ax.add_collection3d(Line3DCollection(lines,colors=color,linewidths=.8,alpha=.22))
        ax.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],[0]*5,color='#aaaaaa',linewidth=.6)
        artists = {}
        for name in cfg.inbetween.models:
            artist = Line3DCollection([],colors=colors[name],linewidths=2)
            ax.add_collection3d(artist); artists[name] = artist
        ax.set(xlim=(-.85,.85),ylim=(-.85,.85),zlim=(-.05,1.85))
        ax.set_box_aspect((1,1,1.15))
        ax.view_init(elev=12,azim=-55)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([0,1])
        return artists

    def update(artists, clip, frame):
        for name, artist in artists.items():
            pose = clip[1][name][frame, :22]
            artist.set_segments(np.stack((pose[1:],pose[parents]),1))

    for clip in clips:
        fig = plt.figure(figsize=(7,6),dpi=100)
        artists = create_axis(fig,(1,1,1),clip)
        title = fig.suptitle('Kimodo: purple · CondMDI: teal | prescribed endpoints: grey / green', fontsize=10)
        writer = FFMpegWriter(fps=30, codec='libx264', extra_args=['-pix_fmt','yuv420p','-crf','20'])
        with writer.saving(fig,str(preview/f'task-{clip[0]["task"]:03d}.mp4'),dpi=100):
            for frame in range(61):
                update(artists,clip,frame)
                title.set_text(f'Kimodo: purple · CondMDI: teal | {frame/30:.2f} s')
                writer.grab_frame()
        plt.close(fig)
    fig = plt.figure(figsize=(12,12),dpi=100)
    artists = [create_axis(fig,(4,3,i+1),clip) for i,clip in enumerate(clips)]
    title = fig.suptitle('HOI → prescribed standing | Kimodo: purple · CondMDI: teal',fontsize=13)
    fig.subplots_adjust(left=.01,right=.99,bottom=.01,top=.94,wspace=.02,hspace=.12)
    writer = FFMpegWriter(fps=30, codec='libx264', extra_args=['-pix_fmt','yuv420p','-crf','20'])
    with writer.saving(fig,str(preview/'all_twelve.mp4'),dpi=100):
        for frame in range(61):
            for artist,clip in zip(artists,clips):
                update(artist,clip,frame)
            title.set_text(f'HOI → prescribed standing | Kimodo: purple · CondMDI: teal | {frame/30:.2f} s')
            writer.grab_frame()
            if frame == 30:
                fig.savefig(preview/'all_twelve_midpoint.png',dpi=150)
    plt.close(fig)


def run_inbetween(cfg):
    from omegaconf import OmegaConf
    root = Path(__file__).resolve().parents[2]
    torch.set_num_threads(4)
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('reportable inbetween run requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = Path(cfg.hosi_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json(output/(cfg.inbetween.stage+'_provenance.json'), dict(commit=commit, stage=cfg.inbetween.stage,
        config=OmegaConf.to_container(cfg, resolve=True)))
    {'prepare': prepare_inputs, 'generate': dispatch, 'evaluate': evaluate, 'render': render}[cfg.inbetween.stage](cfg, root)
