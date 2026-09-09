"""Model-independent multi-task conditions and measured source boundaries."""

import copy
import json
import pickle
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .inbetween import heading


ACTION_TYPES = {
    'walk': 'locomotion',
    'sit down on chair': 'static_object_interaction',
    'sit down on office chair': 'static_object_interaction',
    'sit down on sofa': 'static_object_interaction',
    'sit down on couch': 'static_object_interaction',
    'stand up from seat': 'static_object_interaction',
}


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')


def source_type(text):
    return ACTION_TYPES.get(text)


def source_record(corpus, data_idx, language, starts, ends, scene, partition):
    sequence = int(language['ori_sequence_idx'][data_idx])
    start, stop = int(language['start_idx'][data_idx]), int(ends[sequence])
    text = str(language['text'][data_idx][0])
    return dict(
        source_id=f'{corpus.lower()}-{data_idx}', source_dataset=corpus,
        data_idx=int(data_idx), source_sequence_idx=sequence, source_scene=scene,
        source_partition=partition, text=text,
        task_type='hoi' if corpus == 'OMOMO' else source_type(text),
        source_sequence_start=int(starts[sequence]), source_start_frame=start,
        source_stop_frame=stop, source_fps=30, frame_interval='[start, stop)',
        initial_context_frames=list(range(start, min(start+10, stop))),
        terminal_context_frames=list(range(max(start, stop-10), stop)),
        source_terminal_frame=stop-1,
        semantic_end_frame=int(language['end_range'][data_idx]),
        source_duration_s=(stop-start)/30,
        source_is_complete_recomposed_gt=False,
    )


class SourceCorpus:
    """Read corpus-qualified indices without constructing a training dataset."""
    def __init__(self, root, name):
        self.name = name
        self.root = Path(root)/('data/test' if name == 'OMOMO' else 'data/dataset')
        with (self.root/'language_motion_dict/language_motion_dict__inter_and_loco__16.pkl').open('rb') as handle:
            self.language = pickle.load(handle)
        self.starts = np.load(self.root/'start_idx.npy')
        self.ends = np.load(self.root/'end_idx.npy')
        self.joints = np.load(self.root/'human_joints_aligned.npy', mmap_mode='r')
        self.orient = np.load(self.root/'human_orient.npy', mmap_mode='r')
        self.pose = np.load(self.root/'human_pose.npy', mmap_mode='r')
        self.betas = np.load(self.root/'betas.npy', mmap_mode='r')
        with (self.root/'gender.pkl').open('rb') as handle:
            self.genders = pickle.load(handle)
        with (self.root/'scene_name.pkl').open('rb') as handle:
            self.scenes = pickle.load(handle)
        if name == 'OMOMO':
            self.object_rotation = np.load(self.root/'object_rot_mat.npy', mmap_mode='r')
            self.object_translation = np.load(self.root/'object_trans.npy', mmap_mode='r')
            with (self.root/'object_name.pkl').open('rb') as handle:
                self.objects = pickle.load(handle)

    def record(self, data_idx, partition):
        sequence = int(self.language['ori_sequence_idx'][data_idx])
        scene = (str(self.scenes[sequence]) if self.name == 'OMOMO'
                 else str(self.scenes[int(self.starts[sequence])]))
        result = source_record(self.name, data_idx, self.language, self.starts,
                               self.ends, scene, partition)
        result['gender'] = str(self.genders[sequence])
        result['body_parameters'] = dict(path=str(self.root/'betas.npy'), index=sequence)
        if self.name == 'OMOMO':
            result['object_name'] = str(self.objects[sequence])
        return result

    def native_motion(self, record, frames, device, model):
        """Raw world root and local body rotations in native SMPL-X Y-up."""
        from utils import run_smplx_model, SMPLX_JOINTS_28
        pose = torch.as_tensor(np.concatenate((self.orient[frames, None],
            self.pose[frames].reshape(-1, 21, 3)), axis=1), device=device, dtype=torch.float32)
        if self.name == 'OMOMO':
            # Same world-frame correction as datasets.infbagel; local rotations
            # retain the native template frame.
            correction = pose.new_tensor([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
            pose[:, 0] = transforms.matrix_to_axis_angle(correction @ transforms.axis_angle_to_matrix(pose[:, 0]))
        betas = torch.as_tensor(np.array(self.betas[record['source_sequence_idx']]),
                                device=device, dtype=torch.float32)
        neutral = model.J_regressor @ (model.v_template + torch.einsum(
            'vci,i->vc', model.shapedirs[..., :len(betas)], betas))
        pelvis = torch.as_tensor(np.array(self.joints[frames, 0]), device=device, dtype=torch.float32)
        translation = pelvis-neutral[0]
        vertices, joints = run_smplx_model(pose, translation, betas, record['gender'],
                                          joints_ind=SMPLX_JOINTS_28, smpl_model=model)
        reference = torch.as_tensor(np.array(self.joints[frames]), device=device, dtype=torch.float32)
        return dict(pose=pose, translation=translation, joints=joints, verts=vertices,
                    betas=betas, gender=record['gender'],
                    source_joint_max_error_m=float((joints-reference).norm(dim=-1).max()))


def original_tasks(root, corpus):
    tasks, sources = [], {}
    for path in sorted((Path(root)/'data/hosi_test/data').glob('*.json')):
        for index, task in enumerate(json.loads(path.read_text())):
            data_idx = int(task['data_idx'])
            source = corpus.record(data_idx, 'OMOMO-test')
            sources[source['source_id']] = source
            tasks.append(dict(task_id=f'hosi-{len(tasks):03d}', original_file=str(path.relative_to(root)),
                original_row=index, original_task=task, source_id=source['source_id']))
    return tasks, list(sources.values())


def lingo_sources(corpus, split):
    language = corpus.language
    sequence = np.asarray(language['ori_sequence_idx'])
    first = np.flatnonzero(np.asarray(language['start_idx']) == corpus.starts[sequence])
    # Mirror aliases are already resolved in the fixed v3 split. Use original
    # sequences once; duplicated language windows use the smallest data_idx.
    by_sequence = {}
    for index in first.tolist():
        seq = int(sequence[index])
        if seq < len(corpus.starts)//2 and seq not in by_sequence:
            by_sequence[seq] = index
    test_scenes = set(split['test']['scenes'])
    records, exclusions = [], []
    for seq, index in sorted(by_sequence.items()):
        scene = str(corpus.scenes[int(corpus.starts[seq])])
        if scene not in test_scenes:
            continue
        record = corpus.record(index, 'test')
        record['source_scene_family'] = split['scene_to_family'][scene]
        reasons = []
        if source_type(record['text']) is None:
            reasons.append('action_outside_v1_allowlist')
        if language['left_hand_inter_frame'][index] != -1 or language['right_hand_inter_frame'][index] != -1:
            reasons.append('annotated_hand_interaction')
        if record['source_stop_frame']-record['source_start_frame'] <= 48:
            reasons.append('source_too_short_for_native_window')
        record['source_eligible'] = not reasons
        record['exclusion_reasons'] = reasons
        (exclusions if reasons else records).append(record)
    return records, exclusions


def human_boundary_measures(joints, fps=30):
    root = joints[:, 0]
    angle = heading(joints)
    delta = torch.atan2((angle[1:]-angle[:-1]).sin(), (angle[1:]-angle[:-1]).cos())
    spine = joints[:, 12]-root
    tilt = torch.acos((spine[:, 1]/spine.norm(dim=-1)).clamp(-1, 1))*180/torch.pi
    feet = joints[:, [7, 8, 10, 11]]
    return dict(root_position=root[-1].tolist(), heading_rad=float(angle[-1]),
        root_speed_max_m_s=float(((root[1:]-root[:-1])*fps).norm(dim=-1).max()),
        heading_speed_max_rad_s=float((delta*fps).abs().max()),
        pelvis_height_min_m=float(root[:, 1].min()),
        pelvis_height_max_m=float(root[:, 1].max()),
        foot_floor_distance_max_m=float(feet[..., 1].abs().amin(-1).max()),
        foot_speed_mean_m_s=float(((feet[1:]-feet[:-1])*fps).norm(dim=-1).mean()),
        torso_tilt_max_deg=float(tilt.max()))


def object_boundary_measures(joints, translation, rotation, vertices, fps=30):
    world = (rotation @ vertices.T).transpose(-1, -2)+translation[:, None]
    hands = joints[:, [22, 23, 24, 25, 26, 27]]
    distances = (hands[:, :, None]-world[:, None]).norm(dim=-1).amin(-1)
    omega = transforms.matrix_to_axis_angle(rotation[1:] @ rotation[:-1].transpose(-1, -2))*fps
    result = dict(hand_object_min_m=float(distances.min()),
        object_floor_distance_max_m=float(world[..., 1].amin(-1).abs().max()),
        object_speed_max_m_s=float(((translation[1:]-translation[:-1])*fps).norm(dim=-1).max()),
        object_angular_speed_max_rad_s=float(omega.norm(dim=-1).max()),
        object_translation=translation[-1].tolist(), object_rotation=rotation[-1].tolist())
    result['supported_slow'] = (result['object_floor_distance_max_m'] <= .05
        and result['object_speed_max_m_s'] <= .10 and result['object_angular_speed_max_rad_s'] <= .5)
    result['hands_released'] = result['hand_object_min_m'] >= .08
    result['required_exit_action'] = ('walk' if result['supported_slow'] and result['hands_released']
        else 'release' if result['supported_slow'] else 'place_then_release')
    return result


@torch.no_grad()
def audit_source_boundaries(corpora, records, device):
    import trimesh
    from utils import zup_to_yup
    objects = {}
    for record in records:
        corpus = corpora[record['source_dataset']]
        boundaries = {}
        for name, frame_key in [('entry', 'initial_context_frames'), ('exit', 'terminal_context_frames')]:
            frames = record[frame_key]
            joints = torch.as_tensor(np.array(corpus.joints[frames]), device=device, dtype=torch.float32)
            values = human_boundary_measures(joints)
            if corpus.name == 'OMOMO':
                obj = record['object_name']
                if obj not in objects:
                    mesh = trimesh.load_mesh(corpus.root/'rest_object_geo'/(obj+'.ply'))
                    objects[obj] = torch.as_tensor(zup_to_yup(np.asarray(mesh.vertices)), device=device, dtype=torch.float32)
                translation = torch.as_tensor(np.array(corpus.object_translation[frames]), device=device, dtype=torch.float32)
                rotation = torch.as_tensor(np.array(corpus.object_rotation[frames]), device=device, dtype=torch.float32)
                values.update(object_boundary_measures(joints, translation, rotation, objects[obj]))
            boundaries[name] = values
        record['source_boundary_audit'] = boundaries
    return records


def transition_edge(source, target, action, object_policy, duration_s=1.4):
    return dict(source_segment=source, target_segment=target, action=action,
        duration_s=duration_s, context_before_s=.3, context_after_s=.3,
        boundary_source='actual_generated_context', target_context='planned_successor_context',
        body_identity='preserve_episode_body', object_policy=object_policy,
        evaluate_transition_frames=True,
        requirements=['scene_collision', 'body_support', 'contact_change', 'position_velocity_continuity'])


def successor_condition(segment, generated_history, persistent_objects):
    """Carry achieved state into a successor; data_idx remains provenance."""
    condition = copy.deepcopy(segment)
    condition['initial_history'] = generated_history
    condition['persistent_objects'] = persistent_objects
    condition['start_location'] = generated_history['joints'][-1, 0].clone()
    condition['start_location'][1] = 0
    condition['initialization'] = 'previous_generated_history'
    return condition


def validate_episode(episode, sources):
    """Scientific input invariants for a continuous single-scene episode."""
    segments = episode['segments']
    ids = [item['segment_id'] for item in segments]
    if len(set(ids)) != len(ids):
        raise ValueError('segment IDs must be unique')
    if len(episode['transitions']) != len(segments)-1:
        raise ValueError('each neighbouring pair requires one transition')
    for index, segment in enumerate(segments):
        record = sources[segment['source_id']]
        if segment['data_idx'] != record['data_idx'] or segment['source_dataset'] != record['source_dataset']:
            raise ValueError('source dataset and data_idx must resolve together')
        if segment['scene_name'] != episode['scene_name']:
            raise ValueError('an episode uses one persistent target scene')
        if index and segment['initialization'] != 'previous_generated_history':
            raise ValueError('a successor must inherit generated history')
        if segment['task_type'] == 'static_object_interaction' and not segment['contact_target']:
            raise ValueError('static interaction requires an explicit contact target')
    for index, edge in enumerate(episode['transitions']):
        if (edge['source_segment'], edge['target_segment']) != (ids[index], ids[index+1]):
            raise ValueError('transition order must follow the episode')
        if not edge['evaluate_transition_frames']:
            raise ValueError('continuous episode evaluation includes transitions')
    return True


def run_multitask(cfg):
    from .multitask_geometry import construct_hosi_chains, render_construction_previews
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered construction requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output = Path(cfg.multitask.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    corpora = {name: SourceCorpus(root, name) for name in ('OMOMO', 'LINGO')}
    tasks, hoi = original_tasks(root, corpora['OMOMO'])
    split = json.loads(Path(cfg.multitask.split_manifest).read_text())
    lingo, exclusions = lingo_sources(corpora['LINGO'], split)
    torch.cuda.synchronize(cfg.device)
    audit_source_boundaries(corpora, hoi+lingo, cfg.device)
    torch.cuda.synchronize(cfg.device)
    write_json(output/'original_hosi_tasks.json', dict(tasks=tasks))
    write_json(output/'sources.json', dict(sources=hoi+lingo))
    write_json(output/'source_exclusions.json', dict(records=exclusions))
    episodes, construction_audit = construct_hosi_chains(cfg, root, output, corpora, tasks, hoi+lingo)
    write_json(output/'candidate_episodes.json', dict(schema_version=1, episodes=episodes))
    write_json(output/'construction_audit.json', dict(records=construction_audit))
    render_construction_previews(root, output, episodes, cfg.multitask.scene_mesh_root)
    torch.cuda.synchronize(cfg.device)
    summary = dict(schema_version=1, seed=int(cfg.seed), git_commit=commit,
        original_hosi_tasks=len(tasks), original_hosi_scenes=len({r['original_task']['scene_name'] for r in tasks}),
        unique_omomo_sources=len(hoi), lingo_sources=len(lingo), excluded_lingo_sources=len(exclusions),
        lingo_action_counts=dict(Counter(r['text'] for r in lingo)),
        exclusion_reason_counts=dict(Counter(reason for r in exclusions for reason in r['exclusion_reasons'])),
        omomo_exit_requirements=dict(Counter(r['source_boundary_audit']['exit']['required_exit_action'] for r in hoi)),
        geometry_accepted_episodes=len(episodes),
        construction_status_counts=dict(Counter(r['status'] for r in construction_audit)),
        construction_audit_rows=len(construction_audit),
        selected_episode_ids=[r['episode_id'] for r in episodes],
        semantic_review='pending',
        elapsed_seconds=time.perf_counter()-started, device=str(cfg.device),
        peak_allocated_bytes=torch.cuda.max_memory_allocated(cfg.device), model_samples=0)
    write_json(output/'source_audit_summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)
