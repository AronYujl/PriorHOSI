"""Execute frozen source-selected chains using the history each stage produced."""

import json
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from .continuation_outcomes import native_tracks
from .inbetween import dispatch
from .multitask import hand_object_distances, human_boundary_measures, write_json
from .multitask_handoff import motion_slice
from .source_bridge import adapt_bridge, bridge_condition, native_arrays, prediction_row, render_bridge
from .source_eligibility import (_load_scene, full_source_geometry, object_support,
    source_support)
from .standing_transition import load_texts, make_history, native_local_goal, sample_transition
from .surface_edit import decode_body, load_object_sdf

TRACK_KEYS = ('pose', 'translation', 'joints', 'object_translation', 'object_rotation', 'contact')


def window_plan(initial_frames, total_frames):
    """Forty-two new 30 Hz frames per window; the final budget may be partial."""
    return [min(42, total_frames-start) for start in range(initial_frames, total_frames, 42)]


def append_window(history, window, new_frames=42):
    # Native interpolation exposes 46 frames: the first four end at the current
    # state, followed by 42 genuinely new frames. Keep all existing history.
    return dict(history, **{key: torch.cat((history[key], window[key][4:4+new_frames])) for key in TRACK_KEYS})


def append_bridge(history, bridge):
    return dict(history, **{key: torch.cat((history[key], bridge[key][10:])) for key in TRACK_KEYS})


def geometric_contacts(motion, vertices):
    world = vertices @ motion['object_rotation'].transpose(-1, -2)+motion['object_translation'][:, None]
    hands = hand_object_distances(motion['joints'], world)
    feet = (motion['joints'][:, [10, 11], None]-world[:, None]).norm(dim=-1).amin(-1)
    return torch.cat((hands, feet), -1).lt(.05).float()


def contact_track(coarse):
    return torch.nn.functional.interpolate(coarse.T[None], size=46,
        mode='linear', align_corners=True)[0].T


def save_motion(path, motion):
    torch.save({k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in motion.items() if k != 'verts'}, path)
    np.savez(path.with_suffix('.npz'), **native_arrays(motion), contact=motion['contact'].detach().cpu().numpy())


def boundary_seam(history, window):
    before = (history['joints'][-1]-history['joints'][-4])*10
    after = (window['joints'][6]-window['joints'][3])*10
    rotations = transforms.axis_angle_to_matrix(window['pose'][3:5])
    jump = transforms.matrix_to_axis_angle(rotations[1] @ rotations[0].transpose(-1, -2)).norm(dim=-1)
    return dict(joint_context_max_error_m=float((window['joints'][[0, 3]]-history['joints'][[-4, -1]]).norm(dim=-1).max()),
        object_context_max_error_m=float((window['object_translation'][[0, 3]]-history['object_translation'][[-4, -1]]).abs().max()),
        object_rotation_context_max_error=float((window['object_rotation'][[0, 3]]-history['object_rotation'][[-4, -1]]).abs().max()),
        joint_velocity_jump_mean_m_s=float((after-before).norm(dim=-1).mean()),
        local_rotation_jump_mean_deg=float(jump.mean()*180/torch.pi), local_rotation_jump_max_deg=float(jump.max()*180/torch.pi))


def stage_checks(*, finite, goal_error, geometry, support, history_error, budget_ok, object_fixed=True):
    return dict(finite=finite, goal=goal_error <= .10, geometry=geometry,
        body_support=support, native_history=history_error <= 1e-4,
        frame_budget=budget_ok, persistent_object=object_fixed)


@torch.no_grad()
def evaluate_motion(motion, model, scene, obj, thresholds, goal, source_record=None):
    vertices, sdf, info = obj
    motion = dict(motion)
    motion['verts'], rebuilt = decode_body(motion, model)
    reconstruction = float((rebuilt-motion['joints']).norm(dim=-1).max())
    geometry = full_source_geometry(motion, scene, sdf, info, vertices,
        motion['object_translation'], motion['object_rotation'], thresholds)
    tail = human_boundary_measures(motion['joints'][-10:])
    world = vertices @ motion['object_rotation'].transpose(-1, -2)+motion['object_translation'][:, None]
    hand_distances = hand_object_distances(motion['joints'], world)
    feet = motion['joints'][:, [7, 8, 10, 11]]
    foot_mask = feet[..., 1].abs() <= .08
    velocity = (feet[1:]-feet[:-1]).norm(dim=-1)*30
    paired = foot_mask[:-1] & foot_mask[1:]
    support = (source_support(motion, source_record, model, scene) if source_record is not None
        else dict(passes=bool(feet[..., 1].abs().amin(-1).max() <= .08),
            foot_distance_max_m=float(feet[..., 1].abs().amin(-1).max())))
    pelvis_error = float((motion['joints'][-1, 0, [0, 2]]-motion['joints'].new_tensor(goal)[[0, 2]]).norm())
    result = dict(geometry=geometry, support=support, tail=tail, pelvis_goal_error_m=pelvis_error,
        native_reconstruction_max_error_m=reconstruction,
        contact_frame_fraction_5cm=float((hand_distances <= .05).any(-1).float().mean()),
        contact_frame_fraction_8cm=float((hand_distances <= .08).any(-1).float().mean()),
        hand_distance_min_m=float(hand_distances.min()), hand_distance_mean_m=float(hand_distances.mean()),
        foot_contact_frame_fraction=float(foot_mask.any(-1).float().mean()),
        foot_sliding_speed_m_s=float(velocity[paired].mean()) if bool(paired.any()) else None,
        terminal_object_support=object_support(world[-1], scene), frame_count=len(motion['pose']))
    return result


@torch.no_grad()
def sample_hoi_window(cfg, sampler, dataset, history, task, model, embedding, codec, step, total_windows):
    from test_infbagel_hosi import decode_sample_window
    reference = history['object_rotation'][-4]
    clean, mat, reference, error = make_history(cfg, dataset, history, task, model,
        object_reference=reference, preserve_object_history=True, contact_history=history['contact'])
    fixed = clean[:, :2].clone()
    goal = native_local_goal(dataset, history, task, model, mat, task['pelvis_goal'], planar=True)
    object_goal = (mat.new_tensor(task['object_goal'])[None]-mat[:, :3, 3]) @ mat[0, :3, :3]
    sequence = dataset.ori_sequence_idx[task['data_idx']]
    seq_name = dataset.scene_name[sequence]
    bps = codec.recompute_bps(dataset.obj_rest_verts[task['object_name']], reference)
    one = torch.ones(1, device=cfg.device, dtype=torch.bool)
    zero = ~one
    pi, end_pi, length = [torch.tensor([x], device=cfg.device, dtype=torch.long)
        for x in (step*42, step*42+48, total_windows*42+6)]
    torch.cuda.synchronize(cfg.device)
    started = time.perf_counter()
    samples, _ = sampler.p_sample_loop(fixed, mat, torch.zeros(1, dtype=torch.long, device=cfg.device),
        embedding, goal, torch.zeros_like(goal), object_goal, zero, one, pi, end_pi, length,
        one, one, one, bps[:, None], None, reference, dataset.obj_rest_verts,
        seq_name_dict={0:seq_name}, obj_rot_mat_prefix=torch.eye(3, device=cfg.device)[None])
    torch.cuda.synchronize(cfg.device)
    seconds = time.perf_counter()-started
    result = samples[-1]
    decoded = decode_sample_window(cfg, result, dataset, mat)
    object_rotation = decoded['object_rot_mat'].reshape(16, 3, 3) @ reference[0]
    world = dict(points_world=decoded['points_orig'].reshape(16, 28, 3).cpu(),
        global_rot_6d=decoded['global_rot_6d'].reshape(16, 132).cpu(),
        object_translation_world=decoded['obj_trans_orig'].reshape(16, 3).cpu(),
        object_rotation_world=object_rotation.cpu())
    position_error = float((decoded['obj_trans_orig'][:, :2]-history['object_translation'][[-4, -1]][None]).abs().max())
    rotation_error = float((object_rotation[:2]-history['object_rotation'][[-4, -1]]).abs().max())
    audit = dict(history_max_error_m=error, fixed_history_max_error=float((result[:, :2, :219]-fixed[..., :219]).abs().max()),
        object_history_max_error_m=position_error, object_rotation_history_max_error=rotation_error,
        generation_seconds=seconds, hoi_forward_calls=500, hsi_forward_calls=0, seed=int(cfg.seed),
        progress=[int(pi), int(end_pi), int(length)], local_pelvis_goal=goal.tolist(),
        object_condition='BPS and rotation reference recomputed from actual first history frame')
    snapshot = dict(conditioned_history=fixed.cpu(), mat=mat.cpu(), object_reference=reference.cpu(), object_bps=bps.cpu())
    return world, audit, result.cpu(), snapshot


def run_acquisition(cfg, root, destination, episode, history, target, model, object_vertices, cached=None):
    from omegaconf import OmegaConf
    destination.mkdir()
    inputs = destination/'inputs'
    inputs.mkdir()
    condition, canonical, origin, arrays = bridge_condition(history, target,
        history['object_translation'][-1], history['object_rotation'][-1], model)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in condition.items() if k != 'verts'}, inputs/'condition.pt')
    np.savez(inputs/'conditions.npz', **{k:v.cpu().numpy()[None] for k,v in arrays.items()})
    generation = OmegaConf.merge(cfg, dict(hosi_output_dir=str(destination),
        inbetween=dict(input_dir=str(inputs), models=['kimodo'], kimodo_device=str(cfg.device))))
    if cached is None:
        dispatch(generation, root)
        prediction_path = destination/'kimodo/prediction.npz'
    else:
        previous = torch.load(cached['condition'], map_location=cfg.device, weights_only=False)
        keys = ('pose', 'translation', 'joints', 'object_translation', 'object_rotation', 'betas')
        if previous['gender'] != condition['gender'] or any(not torch.equal(previous[k],condition[k]) for k in keys):
            raise ValueError('cached Kimodo sample has a different native condition')
        prediction_path = Path(cached['prediction'])
        write_json(destination/'generation_reference.json', dict(reused_prediction=str(prediction_path),
            verified_native_condition=str(cached['condition']), new_generation_seconds=0., new_model_samples=0))
    with np.load(prediction_path) as arrays:
        prediction = prediction_row(arrays, 0, cfg.device)
    motion, record = adapt_bridge(cfg, root, destination, episode, condition, canonical, origin, model, prediction, len(history['pose']))
    motion['contact'] = geometric_contacts(motion, object_vertices)
    save_motion(destination/'actual_motion.pt', motion)
    record['actual_prefix_max_error_m'] = float((motion['joints'][:10]-history['joints'][-10:]).abs().max())
    record['actual_prefix_pose_exact'] = torch.equal(motion['pose'][:10], history['pose'][-10:])
    record['actual_prefix_translation_exact'] = torch.equal(motion['translation'][:10], history['translation'][-10:])
    record['gates']['actual_history'] = (record['actual_prefix_max_error_m'] < 1e-5
        and record['actual_prefix_pose_exact'] and record['actual_prefix_translation_exact'])
    record['bridge_gate'] = all(record['gates'].values())
    record['generation_reused_from'] = str(prediction_path) if cached is not None else None
    return motion, record


def blocked(stage, predecessor, reasons):
    return dict(stage=stage, status='blocked_by_predecessor', predecessor=predecessor,
        reasons=reasons, metrics=None, generated_frames=0, model_windows=0)


def cached_hsi_window(path, history, progress):
    """Reuse denoising only when the actual history and progress match."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if saved['audit']['progress'] != list(progress):
        raise ValueError('cached first HSI window has different progress')
    if any(not torch.equal(saved['actual_input'][key], history[key][-10:].cpu()) for key in TRACK_KEYS):
        raise ValueError('cached first HSI window has different actual history')
    audit = dict(saved['audit'], reused_denoising_from=str(path),
        cached_generation_seconds=saved['audit']['generation_seconds']+saved['audit'].get('cached_generation_seconds',0.),
        generation_seconds=0., hsi_forward_calls=0)
    return saved['world'], audit, saved['clean']


def run_actual_history(cfg):
    import hydra
    import trimesh
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from priors.core.window_codec import WindowStateCodec
    from priors.hoi.models import load_trained_hoi_prior
    from test_infbagel_lingo_hsi import _remap_checkpoint_keys
    from utils import create_smplx_model, zup_to_yup
    root = Path(__file__).resolve().parents[2]
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], cwd=root, text=True).strip():
        raise RuntimeError('registered actual-history execution requires a clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    settings = cfg.multitask.execution
    output = Path(settings.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(Path(settings.source_manifest).read_text())
    selection = json.loads(Path(settings.selection).read_text())
    cache = {}
    if settings.initial_window_cache is not None:
        references = json.loads(Path(settings.initial_window_cache).read_text())
        if references['source_manifest'] != str(settings.source_manifest):
            raise ValueError('initial-window cache uses a different source manifest')
        cache = {(key,0):path for key,path in references['windows'].items()}
    resume = dict(completed_episode_ids=[], windows={}, kimodo={})
    if settings.resume is not None:
        resume = json.loads(Path(settings.resume).read_text())
        if resume['source_manifest'] != str(settings.source_manifest) or resume['fixed_selection'] != str(settings.selection):
            raise ValueError('execution resume uses different frozen inputs')
        cache.update({(key,int(step)):path for key,windows in resume['windows'].items() for step,path in windows.items()})
    episode_map = {e['episode_id']:e for e in manifest['episodes']}
    lane_ids = selection['episode_ids'][int(settings.lane_index)::int(settings.lane_count)]
    lane_ids = [key for key in lane_ids if key not in resume['completed_episode_ids']]
    episodes = [episode_map[key] for key in lane_ids]
    sources = {s['source_id']:s for s in manifest['sources']}
    write_json(output/'frozen_selection.json', dict(episode_ids=lane_ids, all_pilot_episode_ids=selection['episode_ids'],
        source_manifest=str(settings.source_manifest), lane_index=int(settings.lane_index), lane_count=int(settings.lane_count),
        completed_episode_ids_retained=resume['completed_episode_ids']))
    torch.set_num_threads(4)
    torch.manual_seed(int(cfg.seed))
    torch.cuda.synchronize(cfg.device)
    torch.cuda.reset_peak_memory_stats(cfg.device)
    started = time.perf_counter()
    texts = sorted({s['text'] for e in episodes for s in e['segments']})
    embeddings = load_texts(OmegaConf.merge(cfg, dict(stand_wait=dict(prompts=texts)))) if texts else []
    embeddings = {text:embeddings[i:i+1] for i,text in enumerate(texts)}
    dataset, current_scene, hsi_model, hoi_model, codec, sampler = None, None, None, None, None, None
    models, objects, results = {}, {}, []
    native_peak = torch.cuda.max_memory_allocated(cfg.device)
    for episode in episodes:
        destination = output/episode['episode_id']
        destination.mkdir()
        lingo, hoi = episode['segments']
        task = hoi['original_conditions']
        if dataset is None:
            dataset = InfBaGelDataset(**OmegaConf.merge(cfg.dataset, dict(test_scene_name=episode['scene_name'])))
            dataset.obj_rest_verts = {k:v.to(cfg.device) for k,v in dataset.obj_rest_verts.items()}
            hsi_model = hydra.utils.instantiate(OmegaConf.merge(cfg.model.infbagel, dict(ckpt=str(cfg.hsi_ckpt_path))))
            state, _ = _remap_checkpoint_keys(torch.load(cfg.hsi_ckpt_path, map_location='cpu', weights_only=False))
            hsi_model.load_state_dict(state, strict=True)
            hsi_model = hsi_model.to(cfg.device).eval().requires_grad_(False)
            del state
            sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
            sampler.dataset = dataset
            sampler.hsi_sampler.set_dataset_and_model(dataset, hsi_model)
        elif current_scene != episode['scene_name']:
            dataset.set_test_scene(episode['scene_name'])
        current_scene = episode['scene_name']
        scene = _load_scene(root, current_scene, cfg.device)
        witness = torch.load(Path(manifest['artifact_root'])/episode['construction']['witness'], map_location=cfg.device, weights_only=False)
        source, target = witness['lingo_motion'], witness['target_context']
        gender = source['gender']
        if gender not in models:
            models[gender] = create_smplx_model(gender, torch.device(cfg.device)).eval().requires_grad_(False)
        model = models[gender]
        object_name = task['object_name']
        if object_name not in objects:
            rest = torch.as_tensor(zup_to_yup(np.asarray(trimesh.load_mesh(root/episode['persistent_objects'][0]['geometry']).vertices)), device=cfg.device, dtype=torch.float32)
            array, info = load_object_sdf(root/'data/object/rest_object_sdf_256_npy_files', object_name)
            objects[object_name] = (rest, torch.as_tensor(array, device=cfg.device, dtype=torch.float32)[None, None], info)
        obj = objects[object_name]
        history = motion_slice(source, 0, 10)
        fixed_object = episode['persistent_objects'][0]
        history['object_translation'] = history['translation'].new_tensor(fixed_object['planned_translation']).repeat(10, 1)
        history['object_rotation'] = history['translation'].new_tensor(fixed_object['planned_rotation']).repeat(10, 1, 1)
        history['contact'] = geometric_contacts(history, obj[0])
        history.pop('verts', None)
        save_motion(destination/'initial_history.pt', history)
        windows = []
        plan = window_plan(10, len(source['pose']))
        for step, frames in enumerate(plan):
            progress = (6+42*step, 54+42*step, len(source['pose']))
            if (episode['episode_id'],step) in cache:
                world, audit, clean = cached_hsi_window(cache[(episode['episode_id'],step)], history, progress)
            else:
                world, audit, clean = sample_transition(cfg, sampler, dataset, history, task, model, embeddings[lingo['text']],
                    goal=lingo['pelvis_goal'], scene_goal=lingo['scene_goal'], progress=progress,
                    is_locomotion=lingo['text'] in ('walk', 'stand up from seat'), seed=int(cfg.seed))
            window = native_tracks(cfg, dataset, world, task, False, models, body_parameters=True)
            window['object_translation'] = history['object_translation'][-1:].repeat(46, 1)
            window['object_rotation'] = history['object_rotation'][-1:].repeat(46, 1, 1)
            window['contact'] = torch.zeros(46, 4, device=cfg.device)
            audit.update(boundary_seam(history, window), appended_frames=frames)
            torch.save(dict(world=world, clean=clean, actual_input={k:history[k][-10:].cpu() for k in TRACK_KEYS}, audit=audit), destination/f'hsi-window-{step:02d}.pt')
            windows.append(audit)
            history = append_window(history, window, frames)
            native_peak = max(native_peak, torch.cuda.max_memory_allocated(cfg.device))
            print(json.dumps(dict(episode_id=episode['episode_id'], stage='hsi', window=step,
                new_frames=frames, seconds=audit['generation_seconds'])), flush=True)
        save_motion(destination/'hsi_motion.pt', history)
        hsi_metrics = evaluate_motion(history, model, scene, obj, cfg.multitask.source_eligibility,
            lingo['pelvis_goal'], sources[lingo['source_id']])
        hsi_history_error = max([hsi_metrics['native_reconstruction_max_error_m']]+
            [max(w['history_max_error_m'], w['joint_context_max_error_m']) for w in windows])
        hsi_checks = stage_checks(finite=all(bool(torch.isfinite(history[k]).all()) for k in TRACK_KEYS),
            goal_error=hsi_metrics['pelvis_goal_error_m'], geometry=hsi_metrics['geometry']['passes'],
            support=hsi_metrics['support']['passes'], history_error=hsi_history_error, budget_ok=len(history['pose']) == len(source['pose']),
            object_fixed=bool((history['object_translation'] == history['object_translation'][:1]).all()
                and (history['object_rotation'] == history['object_rotation'][:1]).all()))
        hsi_result = dict(stage='hsi', status='passed' if all(hsi_checks.values()) else 'failed_guard',
            gates=hsi_checks, metrics=hsi_metrics, windows=windows, model_windows=len(windows), generated_frames=sum(plan),
            new_model_windows=sum('reused_denoising_from' not in w for w in windows),
            reused_model_windows=sum('reused_denoising_from' in w for w in windows),
            reused_initial_windows=int(bool(windows) and 'reused_denoising_from' in windows[0]))
        write_json(destination/'hsi_metrics.json', hsi_result)
        stages = [dict(label='Initial source context', start_frame=0, stop_frame=10),
            dict(label='HSIPrior generated predecessor', start_frame=10, stop_frame=len(history['pose']))]
        failures = [key for key,value in hsi_checks.items() if not value]
        bridge_result = blocked('kimodo', 'hsi', failures)
        hoi_result = blocked('hoi', 'hsi', failures)
        longest = int(not failures)
        if not failures:
            bridge, bridge_record = run_acquisition(cfg, root, destination/'bridge', episode, history, target, model, obj[0],
                cached=resume['kimodo'].get(episode['episode_id']))
            start = len(history['pose'])
            history = append_bridge(history, bridge)
            stages.append(dict(label='Kimodo acquisition bridge', start_frame=start, stop_frame=len(history['pose'])))
            bridge_result = dict(stage='kimodo', status='passed' if bridge_record['bridge_gate'] else 'failed_guard',
                gates=bridge_record['gates'], metrics=bridge_record, generated_frames=51, model_windows=1)
            longest += int(bridge_record['bridge_gate'])
            failures = [key for key,value in bridge_record['gates'].items() if not value]
            hoi_result = blocked('hoi', 'kimodo', failures)
            native_peak = max(native_peak, bridge_record['correction']['peak_cuda_allocated_bytes'])
        if longest == 2:
            if hoi_model is None:
                hoi_model, _ = load_trained_hoi_prior(str(cfg.ckpt_path), torch.device(cfg.device), weight_variant=str(cfg.checkpoint_weight_variant))
                sampler.hoi_adapter.set_dataset_and_model(dataset, hoi_model)
                codec = WindowStateCodec(dataset.min_torch, dataset.max_torch, dataset.obj_min_torch,
                    dataset.obj_max_torch, bps_path=root/'code/bps.pt')
            sampler.hoi_adapter.reset_sampling_audit()
            torch.manual_seed(int(cfg.seed))
            start = len(history['pose'])
            hoi_history = motion_slice(history, -10, None)
            hoi_windows = []
            total_windows = int(task['episode_num'])
            for step in range(total_windows):
                world, audit, clean, snapshot = sample_hoi_window(cfg, sampler.hoi_adapter, dataset, hoi_history,
                    task, model, embeddings[hoi['text']], codec, step, total_windows)
                window = native_tracks(cfg, dataset, world, task, False, models, body_parameters=True)
                window['contact'] = contact_track(clean[0, :, 228:232].to(cfg.device))
                audit.update(boundary_seam(hoi_history, window), appended_frames=42)
                torch.save(dict(world=world, clean=clean, snapshot=snapshot,
                    actual_input={k:hoi_history[k][-10:].cpu() for k in TRACK_KEYS}, audit=audit), destination/f'hoi-window-{step:02d}.pt')
                hoi_history = append_window(hoi_history, window)
                history = append_window(history, window)
                hoi_windows.append(audit)
                native_peak = max(native_peak, torch.cuda.max_memory_allocated(cfg.device))
                print(json.dumps(dict(episode_id=episode['episode_id'], stage='hoi', window=step, seconds=audit['generation_seconds'])), flush=True)
            save_motion(destination/'hoi_motion.pt', hoi_history)
            hoi_metrics = evaluate_motion(hoi_history, model, scene, obj, cfg.multitask.source_eligibility, task['pelvis_goal'])
            hoi_metrics['object_goal_error_m'] = float((hoi_history['object_translation'][-1]-hoi_history['translation'].new_tensor(task['object_goal'])).norm())
            history_error = max([hoi_metrics['native_reconstruction_max_error_m']]+
                [max(w['history_max_error_m'], w['joint_context_max_error_m'], w['object_history_max_error_m'], w['object_rotation_history_max_error']) for w in hoi_windows])
            hoi_checks = stage_checks(finite=all(bool(torch.isfinite(hoi_history[k]).all()) for k in TRACK_KEYS),
                goal_error=max(hoi_metrics['pelvis_goal_error_m'], hoi_metrics['object_goal_error_m']),
                geometry=hoi_metrics['geometry']['passes'], support=hoi_metrics['support']['passes'],
                history_error=history_error, budget_ok=len(hoi_history['pose']) == 10+42*total_windows)
            hoi_result = dict(stage='hoi', status='passed' if all(hoi_checks.values()) else 'failed_guard', gates=hoi_checks,
                metrics=hoi_metrics, windows=hoi_windows, generated_frames=42*total_windows, model_windows=total_windows,
                goal_completed=max(hoi_metrics['pelvis_goal_error_m'], hoi_metrics['object_goal_error_m']) <= float(settings.goal_tolerance_m))
            longest += int(all(hoi_checks.values()))
            stages.append(dict(label='HOIPrior generated successor', start_frame=start, stop_frame=len(history['pose'])))
        save_motion(destination/'actual_chain.pt', history)
        results.append(dict(episode_id=episode['episode_id'], scene=episode['scene_name'],
            source_ids=[s['source_id'] for s in episode['segments']], stages=dict(hsi=hsi_result, kimodo=bridge_result, hoi=hoi_result),
            frame_stages=stages, longest_completed_prefix=longest, chain_success=longest == 3,
            observed_frames=len(history['pose']), duration_s=(len(history['pose'])-1)/30,
            source_context_frames=10, source_membership_changed=False, source_terminal_substituted=False,
            artifact=str(destination), hoi_protocol='frozen P15+ArmB native rollout; Phase2.34 offline editor not applied'))
        write_json(destination/'metrics.json', results[-1])
        if settings.render:
            render_bridge(root, destination, episode, native_arrays(history), 10, cfg.multitask.scene_mesh_root, stages=stages)
        print(json.dumps(dict(episode_id=episode['episode_id'], longest_completed_prefix=longest, observed_frames=len(history['pose']))), flush=True)
    torch.cuda.synchronize(cfg.device)
    summary = dict(status='completed', subphase=str(cfg.multitask.subphase), git_commit=commit,
        git_commit_at_completion=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip(),
        seed=int(cfg.seed), device=str(cfg.device), lane_index=int(settings.lane_index), episodes=len(results), records=results,
        chain_passes=sum(r['chain_success'] for r in results),
        stage_status_counts={stage:dict(Counter(r['stages'][stage]['status'] for r in results)) for stage in ('hsi', 'kimodo', 'hoi')},
        source_manifest=str(settings.source_manifest), selection=str(settings.selection),
        elapsed_seconds=time.perf_counter()-started, peak_cuda_allocated_bytes=max(native_peak, torch.cuda.max_memory_allocated(cfg.device)))
    write_json(output/'metrics.json', summary)
    print(json.dumps({k:v for k,v in summary.items() if k != 'records'}), flush=True)
