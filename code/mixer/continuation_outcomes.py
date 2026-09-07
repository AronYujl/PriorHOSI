"""Bounded, paired consequences of cached decisions under the native B1 rollout.

Test-source records remain diagnostic data. The sampler, state advance and native
metric formulas are shared with the existing evaluator; no policy is learned here.
"""
import copy
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch

from .candidate_selection import move_tree, retained_endpoint_features
from .single_side_waypoint import random_snapshot, restore_random

ARMS = ('W0', 'Wplus', 'Wminus')


def write_json(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def exact_tree(first, second):
    """Compare recoverable state values, including representation and conditions."""
    if torch.is_tensor(first):
        return (torch.is_tensor(second) and first.dtype == second.dtype
                and first.shape == second.shape and torch.equal(first, second))
    if isinstance(first, dict):
        return (isinstance(second, dict) and first.keys() == second.keys()
                and all(exact_tree(first[k], second[k]) for k in first))
    if isinstance(first, (list, tuple)):
        return type(first) is type(second) and len(first) == len(second) and all(
            exact_tree(a, b) for a, b in zip(first, second))
    return first == second


def require_training_source(record):
    """Boundary for any future consumer of this diagnostic format."""
    if not record['training_allowed']:
        raise ValueError('test-source continuation outcomes are excluded from training')
    return record


def branch_horizon(window, total_windows, limit=2):
    if limit > 2:
        raise ValueError('paired continuation budget permits at most two new windows')
    last = min(window + limit + 1, total_windows)
    return list(range(window + 1, last)), last == total_windows


def retained_ranges(window, total_windows, observed_windows):
    """Each native coarse frame belongs to exactly one committed window."""
    offset = 0
    result = {}
    for i in range(observed_windows):
        count = 16 if window + i == total_windows - 1 else 14
        result[('current', 'next_1', 'next_2')[i]] = (offset, offset + count)
        offset += count
    return result


def _world_record(cfg, dataset, clean, context):
    from test_infbagel_hosi import decode_sample_window
    output = decode_sample_window(cfg, clean, dataset, context['mat'])
    rotation = (context['obj_rot_mat_prefix'] @ output['object_rot_mat'].reshape(-1, 3, 3)
                @ context['obj_rot_mat_ref']).reshape(1, 16, 3, 3)
    return dict(points_world=output['points_orig'].reshape(1, 16, 28, 3),
                global_rot_6d=output['global_rot_6d'].reshape(1, 16, 22, 6),
                object_translation_world=output['obj_trans_orig'],
                object_rotation_world=rotation, contact=output['contact_label']), output


def _verify_world(decoded, saved):
    keys = ('points_world', 'object_translation_world', 'global_rot_6d', 'object_rotation_world')
    errors = {k: float((decoded[k].cpu().reshape_as(saved[k])-saved[k]).abs().max()) for k in keys}
    if any(errors.values()):
        raise AssertionError('cached selected world decode differs: ' + str(errors))
    return errors


@torch.no_grad()
def generate_branch(cfg, dataset, hoi, hsi, record, source_episode, cached,
                    task, arm, output_dir, protocol):
    """Commit a cached current action, then call the shared native state advance."""
    import hydra
    from astar import get_path
    from test_infbagel_hosi import (get_guidance_from_json, prepare_next_window,
                                   sample_step, seed_everything)
    device = torch.device(cfg.device)
    ambient = random_snapshot(device)
    began = time.perf_counter()
    calls = dict(HOI_calls=0, HSI_calls=0, generated_windows=0, attempted_windows=0)
    payload = dict(source=record, action=arm, training_allowed=False, windows=[],
                   consistency=[], costs=calls, failure=None)
    hooks = []
    def count_hoi(module, args):
        calls['HOI_calls'] += 1
    def count_hsi(module, args):
        calls['HSI_calls'] += 1
    try:
        # Construction and all per-branch RNG state are isolated from traversal.
        seed_everything(record['rng']['episode_seed'])
        sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
        sampler.set_dataset_and_model(dataset, hoi, hsi_model=hsi)
        seed_everything(record['rng']['episode_seed'])
        sampler.inner_hoi.sample_calls = record['window'] + 1
        sampler.compose_calls = (record['window'] + 1) * 500
        context = move_tree(cached['snapshot']['replay_context'], device)
        clean = cached['snapshot']['edited'].to(device).clone()
        offsets = cached['snapshot']['rest_offsets'].to(device).clone()
        world, previous = _world_record(cfg, dataset, clean, context)
        if not torch.isfinite(clean).all():
            raise FloatingPointError('nonfinite cached action')
        if arm == record['selected']:
            payload['current_world_exact'] = _verify_world(world, source_episode['windows'][record['window']])
        payload['windows'].append(dict(absolute_window=record['window'], clean=clean.cpu(),
            context=move_tree(context, 'cpu'), world=move_tree(world, 'cpu'), cached=True,
            editor=cached['editor']))
        cond = get_guidance_from_json(cfg, task)
        cond['raw_text'] = dataset.text[task['data_idx']][0]
        cond['text_emb'] = context['text_emb'].clone()
        trajectory = get_path(np.asarray(task['start_location'])[[0, 2]],
                              np.asarray(task['pelvis_goal'])[[0, 2]], dataset)
        steps, terminal = branch_horizon(record['window'], record['total_windows'],
                                         protocol['generation']['max_extra_windows'])
        hooks = [hoi.register_forward_pre_hook(count_hoi), hsi.register_forward_pre_hook(count_hsi)]
        compatible = arm == record['selected']
        for step in steps:
            mat, fixed, object_points = prepare_next_window(
                cfg, dataset, step, record['scene'], source_episode['test_idx'],
                context['seq_name_dict'], dataset.obj_rest_verts,
                context['obj_rot_mat_ref'], context['obj_rot_mat_prefix'],
                previous['points_orig'], previous['obj_trans_orig'],
                previous['object_rot_mat'], previous['global_rot_6d'], previous['contact_label'])
            pi = torch.tensor([step * 42], device=device, dtype=torch.long)
            calls['attempted_windows'] += 1
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            previous = sample_step(cfg, step, mat, fixed, sampler, copy.deepcopy(cond), trajectory,
                pi, pi+48, context['seq_length'].clone(), context['obj_bps_data'].clone(),
                object_points, dataset.obj_rest_verts, {}, context['seq_name_dict'],
                context['obj_rot_mat_ref'].clone(), {'rest_human_offsets': offsets[0, 0].clone()},
                context['obj_rot_mat_prefix'].clone())
            torch.cuda.synchronize(device)
            seconds = time.perf_counter() - start
            calls['generated_windows'] += 1
            snap = sampler.scene_editor.motion_records[-1]
            new_context = sampler._window_context
            clean = snap['edited'].to(device)
            world, decoded = _world_record(cfg, dataset, clean, new_context)
            item = dict(absolute_window=step, clean=clean.cpu(),
                        context=move_tree({k: v for k, v in new_context.items()
                            if k not in ('obj_rest_verts', 'obj_vert_normals', 'static_occ_cache')}, 'cpu'),
                        world=move_tree(world, 'cpu'), cached=False, generation_seconds=seconds,
                        editor=sampler.scene_editor.records[-1], sample_calls=sampler.inner_hoi.sample_calls)
            payload['windows'].append(item)
            if not torch.isfinite(clean).all():
                raise FloatingPointError('nonfinite continuation')
            if not torch.equal(clean[:, :2], fixed):
                raise AssertionError('continuation changed its actual history')
            if sampler.inner_hoi.sample_calls != step + 1:
                raise AssertionError('continuation sample counter differs from absolute window')
            if not exact_tree(previous, decoded):
                raise AssertionError('shared native world decode differs')
            if compatible:
                # Only the actual same history is compared. The old next choice
                # may be an offset; its W0 trial is still the exact B1 comparator.
                path = Path(record['decision']['path']).with_name(
                    f'task-{record["task"]:03d}-window-{step:03d}-W0.pt')
                old = torch.load(path, map_location='cpu', weights_only=False)['snapshot']
                check = dict(window=step, action='W0', source=str(path),
                             motion_exact=torch.equal(snap['edited'], old['edited']),
                             context_exact=exact_tree(item['context'], old['replay_context']))
                payload['consistency'].append(check)
                if not check['motion_exact'] or not check['context_exact']:
                    raise AssertionError('same-state frozen-B1 replay mismatch: ' + str(check))
                compatible = source_episode['corrections'][step]['waypoint_decision']['selected'] == 'W0'
        payload['termination'] = 'original_terminal' if terminal else 'budget_censored'
        payload['terminal_observed'] = terminal
    except Exception as error:
        payload['failure'] = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
        payload['termination'] = ('nonfinite' if isinstance(error, FloatingPointError) else 'restore_or_execution_failure')
        payload['terminal_observed'] = False
    finally:
        for hook in hooks:
            hook.remove()
        torch.cuda.synchronize(device)
        calls.update(wall_seconds=time.perf_counter()-began,
                     generation_seconds=sum(w.get('generation_seconds', 0.) for w in payload['windows']),
                     peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        restore_random(ambient, device)
        with (Path(output_dir) / (arm+'.pt')).open('xb') as handle:
            torch.save(payload, handle)
    return payload


def _concat_world(payload):
    ranges = retained_ranges(payload['source']['window'], payload['source']['total_windows'], len(payload['windows']))
    world = {k: torch.cat([w['world'][k][:, :b-a] for w, (a, b) in
                           zip(payload['windows'], ranges.values())], dim=1)[0]
             for k in payload['windows'][0]['world']}
    return world, ranges


@torch.no_grad()
def native_tracks(cfg, dataset, world, task, terminal, smpl_cache):
    """Native interpolation/SMPL-X on actual observed frames, without censored padding."""
    from utils import (interpolate_joints, interp_object, interp_jrot,
                       create_smplx_model, run_smplx_model)
    from utils import SMPLX_JOINTS_28
    import pytorch3d.transforms as transforms
    device = torch.device(cfg.device)
    index = dataset.ori_sequence_idx[task['data_idx']]
    betas = torch.as_tensor(dataset.betas[index], device=device)
    transl = torch.as_tensor(dataset.transl[index], device=device)
    gender = dataset.gender[index]
    points = interpolate_joints(world['points_world'].reshape(-1, 84).to(device), cfg.interp_s)
    obj_trans, obj_rot = interp_object(world['object_translation_world'].numpy(),
                                      world['object_rotation_world'].reshape(-1, 9).numpy(), cfg.interp_s)
    obj_trans = torch.from_numpy(obj_trans).to(device).float()
    obj_rot = torch.from_numpy(obj_rot).to(device).float().reshape(-1, 3, 3)
    global_rot = transforms.rotation_6d_to_matrix(world['global_rot_6d'].to(device))
    local_rot = dataset.quat_ik_torch(global_rot)
    local_q = transforms.matrix_to_quaternion(local_rot)
    local_rot = transforms.quaternion_to_matrix(interp_jrot(local_q, cfg.interp_s))
    root_trans = points.reshape(-1, 28, 3)[:, 0] + transl
    pose = transforms.matrix_to_axis_angle(local_rot).reshape(-1, 22, 3)
    if gender not in smpl_cache:
        smpl_cache[gender] = create_smplx_model(gender, device, batch_size=1)
    verts, joints = run_smplx_model(pose, root_trans, betas[None].repeat(len(pose), 1), gender,
                                  joints_ind=SMPLX_JOINTS_28, smpl_model=smpl_cache[gender])
    # A censored final coarse frame is observed once; the two held samples
    # manufactured by the native end-of-task interpolation are excluded here.
    n = len(verts) if terminal else (len(world['points_world'])-1)*cfg.interp_s+1
    return dict(verts=verts[:n], joints=joints[:n], object_translation=obj_trans[:n],
                object_rotation=obj_rot[:n])


def _mean(values, mask=None):
    if mask is not None:
        values = values[mask]
    return float(values.double().mean()) if values.numel() else None


def _relative(hands, translation, rotation):
    return (rotation.transpose(-1, -2)[:, None] @ (hands-translation[:, None])[..., None]).squeeze(-1)


@torch.no_grad()
def evaluate_state(cfg, dataset, record, task, branches, smpl_cache, scene_sdf, scene_info, protocol):
    """All raw components and W0 differences share an actual world-time prefix."""
    from test_infbagel_hosi import compute_scene_sdf_penetration, _subsample_seed
    from eval_metrics import determine_floor_height_and_contacts, compute_foot_sliding_for_smpl
    from .waypoint_control import decode_motion
    from .relational import source_floor_height
    device = torch.device(cfg.device)
    original = torch.load(record['episode']['path'], map_location='cpu', weights_only=False)
    offsets = original['corrections'][record['window']]['rest_offsets'].to(device)
    rest = dataset.obj_rest_verts[record['object']].to(device)
    points128 = rest[torch.linspace(0, len(rest)-1, 128, device=device).long()][None]
    # Match the original episode's native 10475-vertex object subset exactly.
    gen = torch.Generator().manual_seed(_subsample_seed(42, record['scene'], original['test_idx']))
    indices = torch.randperm(len(rest), generator=gen)[:10475].to(device)
    trajectories, ranges, tracks, fk = {}, {}, {}, {}
    for arm, payload in branches.items():
        if not payload['windows']:
            continue
        world, arm_ranges = _concat_world(payload)
        trajectories[arm] = world; ranges[arm] = arm_ranges
        tracks[arm] = native_tracks(cfg, dataset, world, task, payload['terminal_observed'], smpl_cache)
        decoded = []
        for w, (a, b) in zip(payload['windows'], arm_ranges.values()):
            context = move_tree(w['context'], device)
            motion = w['clean'].to(device)
            state = decode_motion(motion, dataset, offsets, context, points128)
            decoded.append({k: state[k][:, :b-a] for k in
                            ('human', 'object_surface', 'object_translation_world', 'object_rotation_world')})
        fk[arm] = {k: torch.cat([d[k] for d in decoded], dim=1)[0] for k in decoded[0]}
    if 'W0' not in tracks:
        return dict(state_id=record['state_id'], task=record['task'], error='missing_W0', branches={}), {}
    ref = tracks['W0']; ref_fk = fk['W0']; ref_world = trajectories['W0']
    current_end = ranges['W0']['current'][1]
    active = ref_world['contact'][:current_end, :2].to(device) > .95
    active_hands = active.any(0)
    fixed_rel = _relative(ref_fk['human'][:current_end, 22:24],
                          ref_fk['object_translation_world'][:current_end], ref_fk['object_rotation_world'][:current_end])
    anchors = torch.stack([fixed_rel[active[:, i], i].mean(0) if active_hands[i]
                           else torch.full((3,), float('nan'), device=device) for i in range(2)])
    floor, _ = source_floor_height(ref_fk['human'][:current_end][None])
    floor = floor[0]
    result = dict(state_id=record['state_id'], task=record['task'], scene=record['scene'],
                  source_record=record['record_id'], source_selected=record['selected'], branches={},
                  training_allowed=False, test_set_development=True,
                  contact_reference=dict(source='actual current W0 activity/anchors; frozen for the paired state',
                      active_hands=active_hands.tolist(), current_active_frames=active.sum(0).tolist(),
                      release_timing='unknown; predicted-contact transitions descriptive, not new control rules'))
    previous_fk = None
    if record['window']:
        prev = original['corrections'][record['window']-1]
        prev_state = decode_motion(prev['edited'].to(device),dataset,offsets,
                                   move_tree(prev['replay_context'],device),points128)
        previous_fk = {k:v[:,13] for k,v in prev_state.items() if k in
                       ('human','object_translation_world','object_rotation_world')}
    arrays = {}; components = {}
    for arm, track in tracks.items():
        payload = branches[arm]
        world = trajectories[arm]; coarse = fk[arm]
        relative = _relative(track['joints'][:, [24, 26]], track['object_translation'], track['object_rotation'])
        ref_relative = _relative(ref['joints'][:, [24, 26]], ref['object_translation'], ref['object_rotation'])
        object_surface = (track['object_rotation'][:, None] @ rest[None, :, :, None]).squeeze(-1) + track['object_translation'][:, None]
        distances = torch.cdist(track['joints'][:, [24, 26]], object_surface).amin(-1)
        predicted = world['contact'][:, :2].to(device) > .95
        coarse_rel = _relative(coarse['human'][:, 22:24], coarse['object_translation_world'], coarse['object_rotation_world'])
        fixed_drift = (coarse_rel-anchors[None]).norm(dim=-1)
        feet = coarse['human'][:, (7, 8, 10, 11)]
        ref_feet = ref_fk['human'][:, (7, 8, 10, 11)]
        stance = ref_feet[..., 1]-floor < floor.new_tensor([.08, .08, .04, .04])
        own_stance = feet[..., 1]-floor < floor.new_tensor([.08, .08, .04, .04])
        query = torch.cat((coarse['human'], coarse['object_surface']), dim=1)[None]
        context0 = payload['windows'][0]['context']
        occupied, nearest = dataset.get_nearest_free_voxel(query, context0['scene_flag'].to(device))
        distance_sq = (query-nearest).double().square().sum(-1)[0]
        occupied = occupied[0]
        components[arm] = dict(track=track,world=world,coarse=coarse,relative=relative,
            object_surface=object_surface,distances=distances,predicted=predicted,
            fixed_drift=fixed_drift,feet=feet,own_stance=own_stance,
            distance_sq=distance_sq,occupied=occupied)

    def measure(arm, a, end, native_b):
        c = components[arm]
        track,world,coarse = c['track'],c['world'],c['coarse']
        relative,object_surface,distances = c['relative'],c['object_surface'],c['distances']
        predicted,fixed_drift,feet = c['predicted'],c['fixed_drift'],c['feet']
        own_stance,distance_sq,occupied = c['own_stance'],c['distance_sq'],c['occupied']
        native_a = a*cfg.interp_s
        if end <= a or native_b <= native_a:
            return dict(status='no_common_prefix')
        sl = slice(native_a, native_b); csl = slice(a, end)
        hscene = compute_scene_sdf_penetration(track['verts'][sl], record['scene']+'_sdf', scene_sdf, scene_info)
        oscene = compute_scene_sdf_penetration(object_surface[sl][:, indices], record['scene']+'_sdf', scene_sdf, scene_info)
        j = track['joints'][sl].cpu().numpy().copy()
        fragment_floor = float(determine_floor_height_and_contacts(j))
        fs = float(compute_foot_sliding_for_smpl(j.copy(), fragment_floor))
        hand_rows = []
        for i in range(2):
            activity = bool(active_hands[i])
            hand_rows.append(dict(hand=i, fixed_active=activity,
                surface_mean_m=_mean(distances[sl, i]),
                surface_per_frame_m=distances[sl, i].tolist(),
                coverage_5cm=_mean((distances[sl, i]<.05).float()),
                active_surface_mean_m=_mean(distances[sl, i]) if activity else None,
                anchor_vs_W0_m=_mean((relative[sl, i]-ref_relative[sl, i]).norm(dim=-1)) if activity else None,
                fixed_anchor_drift_FK_m=_mean(fixed_drift[csl, i]) if activity else None,
                predicted_coverage=_mean(predicted[csl, i].float()),
                relative_speed_m_per_s=_mean((relative[max(0,native_a-1):native_b, i][1:]
                     -relative[max(0,native_a-1):native_b, i][:-1]).norm(dim=-1)*30),
                reference_status='active' if activity else 'N/A_no_active_evidence'))
        vstart = max(1, a)
        paired_end = min(end, len(ref_feet))
        support_mask = stance[vstart:paired_end] & stance[vstart-1:paired_end-1]
        foot_speed = (feet[vstart:paired_end]-feet[vstart-1:paired_end-1])[..., (0, 2)].norm(dim=-1)/.1
        joint_speed = (coarse['human'][vstart:paired_end, :22]-coarse['human'][vstart-1:paired_end-1, :22]).norm(dim=-1)/.1
        root = track['joints'][native_b-1, 0].clone(); root[1] = 0
        target = root.new_tensor(task['pelvis_goal'])
        end_obj = track['object_translation'][native_b-1]
        m = dict(status='ok', coarse_range=[a, end], native_range=[native_a, native_b],
            native_frames=native_b-native_a, native_surface_HS_s_mean=hscene[1],
            native_surface_HS_s_max=hscene[2], native_surface_HS_frame_ratio=hscene[3],
            native_surface_OS_s_mean=oscene[1], native_surface_OS_s_max=oscene[2],
            native_surface_OS_frame_ratio=oscene[3], native_fragment_FS_cm=fs,
            native_fragment_floor_m=fragment_floor, hands=hand_rows,
            contact_any_5cm=_mean((distances[sl]<.05).any(-1).float()),
            support_speed_m_per_s=_mean(foot_speed, support_mask), support_count=int(support_mask.sum()),
            foot_speed_per_foot_m_per_s=foot_speed.mean(0).tolist(),
            foot_predicted_support_coverage=own_stance[csl].float().mean(0).tolist(),
            world_joint_speed_m_per_s=_mean(joint_speed),
            human_scene_RMS_cm=float(distance_sq[csl, :24].mean().sqrt()*100),
            object_scene_RMS_cm=float(distance_sq[csl, 24:].mean().sqrt()*100),
            human_occupied_fraction=_mean(occupied[csl, :24].float()),
            object_occupied_fraction=_mean(occupied[csl, 24:].float()),
            human_goal_error_cm=float((root-target).norm()*100),
            object_goal_error_3D_cm=float((end_obj-root.new_tensor(task['object_goal'])).norm()*100),
            progress=(record['window']*14+end)/(record['total_windows']*14+2),
            metric_source='native-compatible fragment surface/SMPL-X30fps; FK/voxel10fps proxies separately named')
        return m

    for arm, track in tracks.items():
        payload = branches[arm]
        c = components[arm]
        world,coarse,relative = c['world'],c['coarse'],c['relative']
        fixed_drift,distances,predicted = c['fixed_drift'],c['distances'],c['predicted']
        row = dict(action=arm, cached_local_guard=record['cached_guard'][arm],
                   diagnostic_only=arm != 'W0' and not record['cached_guard'][arm]['accepted'],
                   termination=payload['termination'], terminal_observed=payload['terminal_observed'],
                   failure=payload['failure'], costs=payload['costs'], consistency=payload['consistency'],
                   slices={}, paired_reference={}, delta_vs_W0={}, continuity=[])
        slices = dict(ranges[arm], cumulative=(0, len(world['points_world'])))
        for name, (a,b) in slices.items():
            end = min(b,len(ref_world['points_world']))
            native_b = min(end*cfg.interp_s,len(track['joints']),len(ref['joints']))
            row['slices'][name] = measure(arm,a,end,native_b)
            row['paired_reference'][name] = measure('W0',a,end,native_b)
        row['actual_observed'] = dict(coarse_frames=len(world['points_world']),native_frames=len(track['joints']),
            generated_windows=payload['costs']['generated_windows'])
        if previous_fk is not None:
            row['continuity'].append(dict(at='current',
                root_step_m=float((coarse['human'][0,0]-previous_fk['human'][0,0]).norm()),
                joint_step_mean_m=float((coarse['human'][0]-previous_fk['human'][0]).norm(dim=-1).mean()),
                object_step_m=float((coarse['object_translation_world'][0]-previous_fk['object_translation_world'][0]).norm())))
        for name, (a,b) in ranges[arm].items():
            if a:
                row['continuity'].append(dict(at=name,
                    root_step_m=float((coarse['human'][a,0]-coarse['human'][a-1,0]).norm()),
                    joint_step_mean_m=float((coarse['human'][a]-coarse['human'][a-1]).norm(dim=-1).mean()),
                    object_step_m=float((coarse['object_translation_world'][a]-coarse['object_translation_world'][a-1]).norm())))
        endpoint = retained_endpoint_features(track['joints'],track['object_translation'],task,len(track['joints']))
        row['actual_observed'].update(human_goal_error_cm=100*endpoint['root_endpoint_m'],
                                     object_goal_error_3D_cm=100*endpoint['object_endpoint_m'])
        if payload['terminal_observed'] and not payload['failure']:
            row['terminal'] = dict(completed=endpoint['completed'],
                human_goal_error_cm=100*endpoint['root_endpoint_m'],
                object_goal_error_3D_cm=100*endpoint['object_endpoint_m'],
                absolute_window=record['total_windows']-1, semantics='native_retained_v2 original terminal')
        else:
            row['terminal'] = None
        arrays[arm] = dict(world=world, native_joints=track['joints'].cpu(),
            native_object_translation=track['object_translation'].cpu(), native_object_rotation=track['object_rotation'].cpu(),
            coarse_fk_human=coarse['human'].cpu(), object_points128=coarse['object_surface'].cpu(),
            fixed_anchor_drift_FK_m=fixed_drift.cpu(), native_hand_relative=relative.cpu(),
            native_hand_surface_distance_m=distances.cpu(), predicted_contact=predicted.cpu())
        result['branches'][arm] = row
    pair_and_classify(result, protocol['labels'])
    return result, arrays


def metric_deltas(value, base):
    return {k: value[k]-base[k] for k in value.keys() & base.keys()
            if isinstance(value[k], (int,float)) and not isinstance(value[k], bool)
            and isinstance(base[k], (int,float)) and not isinstance(base[k], bool)}


def pair_and_classify(result, limits):
    base = result['branches']['W0']
    for arm, row in result['branches'].items():
        failures = {}; ambiguities = []
        for name, m in row['slices'].items():
            b = row.get('paired_reference',base['slices']).get(name)
            if b is None or m['status'] != 'ok' or b['status'] != 'ok':
                ambiguities.append(name+':missing_common_prefix'); continue
            delta = metric_deltas(m,b)
            delta['hands'] = [metric_deltas(h,r) for h,r in zip(m['hands'],b['hands'])]
            row['delta_vs_W0'][name] = delta
            f = []
            if arm != 'W0':
                for i,(h,r,d) in enumerate(zip(m['hands'],b['hands'],delta['hands'])):
                    if h['fixed_active']:
                        for key,margin in [('active_surface_mean_m',limits['contact_distance_increase_m']),
                                           ('anchor_vs_W0_m',limits['contact_anchor_increase_m'])]:
                            if d[key]>margin: f.append(f'hand{i}:{key}')
                        if d['coverage_5cm'] < -limits['contact_fraction_drop']: f.append(f'hand{i}:coverage_5cm')
                        if h['predicted_coverage'] < r['predicted_coverage']-limits['contact_fraction_drop']:
                            ambiguities.append(f'{name}:hand{i}:predicted_release_or_lost_contact')
                support = delta.get('support_speed_m_per_s')
                if support is not None and support>limits['support_speed_increase_m_per_s']:f.append('support_speed')
                if m['world_joint_speed_m_per_s']<b['world_joint_speed_m_per_s']*limits['world_joint_speed_retention']:
                    f.append('world_joint_speed_retention')
                for key in ('human_goal_error_cm','object_goal_error_3D_cm'):
                    if delta[key]>limits['endpoint_increase_cm']:f.append(key)
            failures[name] = f
        current = bool(failures.get('current'))
        future = [bool(failures[n]) for n in ('next_1','next_2') if n in failures]
        immediate = current
        delayed = not current and any(future)
        recovery = bool(future) and (current or any(future[:-1])) and not future[-1]
        c = row['delta_vs_W0'].get('cumulative',{})
        scene_ok = all(c.get(k,float('inf'))<=margin for k,margin in (
            ('human_scene_RMS_cm',limits['human_scene_RMS_increase_cm']),
            ('object_scene_RMS_cm',limits['object_scene_RMS_increase_cm']),
            ('human_occupied_fraction',limits['occupied_fraction_increase']),
            ('object_occupied_fraction',limits['occupied_fraction_increase'])))
        contact_valid = any(result['contact_reference']['active_hands'])
        benefit = c.get('native_surface_HS_s_mean',0)<0 and c.get('native_surface_OS_s_mean',float('inf'))<=0
        physical_compatible = (arm!='W0' and not any(failures.values())
                    and scene_ok and contact_valid and not row['failure'] and not ambiguities)
        feasible = physical_compatible and row['cached_local_guard']['accepted']
        compatible = benefit and feasible
        category = ('immediate_damage' if immediate else 'delayed_damage' if delayed else
                    'beneficial_compatible' if compatible else 'ambiguous_or_censored' if
                    ambiguities or row['failure'] or not contact_valid else 'no_useful_alternative')
        if recovery:category='recovery'
        row['diagnostics'] = dict(category=category,immediate_damage=immediate,delayed_damage=delayed,
            recovery=recovery,beneficial_compatible=compatible,quality_protected=feasible,
            scene_benefit=benefit,scene_proxy_protected=scene_ok,failures=failures,
            diagnostic_beneficial_compatible=benefit and physical_compatible,
            ambiguities=ambiguities,budget_censored=row['termination']=='budget_censored',
            terminal_inference='observed' if row['terminal_observed'] else 'ambiguous_or_censored')
    result['beneficial_compatible_actions'] = [a for a,r in result['branches'].items() if r['diagnostics']['beneficial_compatible']]
    result['no_useful_alternative'] = (not result['beneficial_compatible_actions']
        if all(not r['failure'] for r in result['branches'].values()) else None)


def run_continuation_scene(cfg):
    import hydra
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from priors.hoi.models import load_trained_hoi_prior
    from utils import init_model
    from test_infbagel_hosi import seed_everything
    if cfg.get('run_id') and subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('reportable continuation requires clean worktree')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    started = time.perf_counter()
    protocol_path = Path(cfg.continuation.protocol)
    root = protocol_path.resolve().parents[2]
    protocol = json.loads(protocol_path.read_text())
    inventory = json.loads((root/protocol['state_manifest']).read_text())
    records = inventory['records']
    if len(records) > protocol['generation']['max_source_records']:
        raise ValueError('source decision budget exceeded')
    if len({r['state_id'] for r in records})*len(ARMS)*protocol['generation']['max_extra_windows'] > protocol['generation']['max_new_windows']:
        raise ValueError('registered generation budget exceeded')
    state_ids = cfg.continuation.state_ids
    selected = {}
    for r in records:
        if r['scene']==cfg.continuation.scene and (state_ids is None or r['state_id'] in state_ids):
            selected.setdefault(r['state_id'],r)
    out = Path(cfg.hosi_output_dir);out.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'resolved.yaml')
    device = torch.device(cfg.device)
    seed_everything(42)
    dataset = InfBaGelDataset(**cfg.dataset)
    dataset.obj_rest_verts={k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
    hoi,_ = load_trained_hoi_prior(cfg.ckpt_path,device,weight_variant=cfg.checkpoint_weight_variant)
    hoi.eval().requires_grad_(False)
    hsi = init_model(OmegaConf.merge(cfg.model.infbagel,{'ckpt':cfg.hsi_ckpt_path}),device=device,eval=True)
    hsi.eval().requires_grad_(False)
    scene = str(cfg.continuation.scene)
    task_rows = json.loads((root/'experiments/tasks/conditional_repair_native_development_s42_20260907.json').read_text())['tasks']
    task_lookup={r['canonical_ordinal']:r for r in task_rows}
    native = json.loads((root/'data/hosi_test/data'/(scene+'.json')).read_text())
    scene_key=scene+'_sdf'
    sdf_root=root/'data/hosi_test/Scene_sdf'
    sdf={scene_key:np.load(sdf_root/(scene_key+'.npy'))}
    sdf_info={scene_key:json.loads((sdf_root/(scene_key+'_info.json')).read_text())}
    torch.cuda.reset_peak_memory_stats(device)
    smpl_cache={};rows=[]
    for state_id,source_record in selected.items():
        record=copy.deepcopy(source_record)
        for asset in [record['episode'],record['decision'],*record['candidates'].values()]:
            asset['path']=str(root/asset['path'])
        task=native[task_lookup[record['task']]['test_idx']]
        dest=out/state_id;dest.mkdir()
        recovered_at=time.perf_counter()
        episode=torch.load(root/record['episode']['path'],map_location='cpu',weights_only=False)
        cached={a:torch.load(root/record['candidates'][a]['path'],map_location='cpu',weights_only=False) for a in ARMS}
        recovery_seconds=time.perf_counter()-recovered_at
        order=ARMS if int(state_id.split('-')[-1])%2==0 else tuple(reversed(ARMS))
        if state_id in cfg.continuation.reuse_states:
            previous=Path(cfg.continuation.resume_source)/state_id
            branches={}
            for arm in order:
                old=previous/(arm+'.pt')
                payload=torch.load(old,map_location='cpu',weights_only=False)
                if payload['failure'] or payload['source']['state_id']!=state_id or payload['action']!=arm:
                    raise ValueError('resume requires a successful matching generated branch')
                branches[arm]=payload
                (dest/(arm+'.pt')).symlink_to(old.resolve())
            write_json(dest/'generation_source.json',dict(source=str(previous),new_HOI_calls=0,new_HSI_calls=0))
        else:
            branches={a:generate_branch(cfg,dataset,hoi,hsi,record,episode,cached[a],task,a,dest,protocol) for a in order}
        torch.cuda.synchronize(device);evaluate_at=time.perf_counter()
        try:
            result,arrays=evaluate_state(cfg,dataset,record,task,branches,smpl_cache,sdf,sdf_info,protocol)
            with (dest/'tracks.pt').open('xb') as handle:torch.save(arrays,handle)
        except Exception as error:
            result=dict(state_id=state_id,task=record['task'],scene=scene,branches={a:dict(costs=b['costs'],failure=b['failure']) for a,b in branches.items()},
                        evaluation_failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        torch.cuda.synchronize(device)
        result.update(recovery_seconds=recovery_seconds,evaluation_seconds=time.perf_counter()-evaluate_at,
                      evaluation_peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
                      source_records=[r['record_id'] for r in records if r['state_id']==state_id],branch_order=list(order),
                      reused_generation=state_id in cfg.continuation.reuse_states)
        write_json(dest/'outcomes.json',result)
        rows.append(result)
        print(json.dumps(dict(state=state_id,task=record['task'],
            costs={a:b['costs'] for a,b in branches.items()},evaluation_failure=result.get('evaluation_failure'),
            labels={a:r.get('diagnostics',{}).get('category') for a,r in result['branches'].items()})),flush=True)
    errors=sum(bool(r.get('evaluation_failure')) or bool(r.get('error')) or
               any(b.get('failure') for b in r['branches'].values()) for r in rows)
    write_json(out/'lane.json',dict(technical_errors=errors,commit_at_start=commit,commit_at_completion=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        scene=scene,states=list(selected),wall_seconds=time.perf_counter()-started,
        generated_windows=sum(r['branches'][a]['costs']['generated_windows'] for r in rows for a in r['branches']),
        new_generated_windows=sum(r['branches'][a]['costs']['generated_windows'] for r in rows if not r['reused_generation'] for a in r['branches']),
        training_allowed=False))
    if errors:
        raise RuntimeError(f'{errors} states retain technical failures; inspect saved outcomes')


def accumulate_branch_cost(total, cost):
    """Calls and elapsed work add; a device allocation peak is a maximum."""
    for key,value in cost.items():
        if key == 'peak_allocated_bytes':
            total[key]=max(total.get(key,0),value)
        else:
            total[key]=total.get(key,0)+value


def summarize_continuation_outcomes(run_root):
    """Descriptive state pairs and task/scene aggregates for this targeted cohort."""
    import csv
    from collections import Counter, defaultdict
    root=Path(run_root)
    protocol=json.loads((root/'protocol.json').read_text())
    inventory=json.loads(Path(protocol['state_manifest']).read_text())
    paths=sorted(root.glob('lanes/*/state-*/outcomes.json'))
    states=[json.loads(p.read_text()) for p in paths]
    output=root/'analysis';output.mkdir(exist_ok=False)
    scalars=['native_surface_HS_s_mean','native_surface_OS_s_mean','native_fragment_FS_cm',
             'contact_any_5cm','human_goal_error_cm','object_goal_error_3D_cm',
             'support_speed_m_per_s','world_joint_speed_m_per_s',
             'human_scene_RMS_cm','object_scene_RMS_cm']
    table=[];adopted=[];all_sides=[];errors=[];checks=[];cost=Counter()
    for state,path in zip(states,paths):
        if state.get('evaluation_failure') or state.get('error'):
            errors.append(dict(state=state['state_id'],error=state.get('evaluation_failure',state.get('error'))))
        for arm,row in state['branches'].items():
            accumulate_branch_cost(cost,row['costs'])
            if row.get('failure'):errors.append(dict(state=state['state_id'],arm=arm,error=row['failure']))
            checks.extend(row.get('consistency',[]))
            if 'diagnostics' not in row:continue
            entry=dict(state_id=state['state_id'],task=state['task'],scene=state['scene'],arm=arm,
                       diagnostic=row['diagnostics'],deltas=row['delta_vs_W0'],source=str(path))
            if arm!='W0':all_sides.append(entry)
            if arm==state['source_selected']:adopted.append(entry)
            for horizon,delta in row['delta_vs_W0'].items():
                flat=dict(state_id=state['state_id'],task=state['task'],scene=state['scene'],arm=arm,
                          horizon=horizon,category=row['diagnostics']['category'],
                          local_accepted=row['cached_local_guard']['accepted'],termination=row['termination'])
                flat.update({k:delta.get(k) for k in scalars})
                for i,h in enumerate(delta['hands']):
                    for key in ('coverage_5cm','surface_mean_m','anchor_vs_W0_m','fixed_anchor_drift_FK_m'):
                        flat[f'hand{i}_{key}']=h.get(key)
                table.append(flat)
    if table:
        with (output/'paired_components.csv').open('x') as f:
            writer=csv.DictWriter(f,fieldnames=list(table[0]));writer.writeheader();writer.writerows(table)
    task_rows=[]
    for task in sorted({r['task'] for r in adopted}):
        rows=[r for r in adopted if r['task']==task]
        means={}
        for horizon in ('current','next_1','next_2','cumulative'):
            for key in scalars:
                values=[r['deltas'][horizon][key] for r in rows if key in r['deltas'].get(horizon,{})]
                means[horizon+':'+key]=sum(values)/len(values) if values else None
        task_rows.append(dict(task=task,scene=rows[0]['scene'],states=len(rows),adopted_deltas=means,
                              labels=dict(Counter(r['diagnostic']['category'] for r in rows))))
    scene_rows=[]
    for scene in sorted({r['scene'] for r in task_rows}):
        rows=[r for r in task_rows if r['scene']==scene]
        scene_rows.append(dict(scene=scene,tasks=len(rows),states=sum(r['states'] for r in rows),
            adopted_task_mean_deltas={k:sum(v)/len(v) if v else None for k in rows[0]['adopted_deltas']
                for v in [[r['adopted_deltas'][k] for r in rows if r['adopted_deltas'][k] is not None]]}))
    def diagnostic_counts(rows):
        return dict(categories=dict(Counter(r['diagnostic']['category'] for r in rows)),
            flags={k:sum(bool(r['diagnostic'][k]) for r in rows) for k in
                ('immediate_damage','delayed_damage','recovery','beneficial_compatible','scene_benefit','diagnostic_beneficial_compatible')},
            current_failure_terms=dict(Counter(f for r in rows for f in r['diagnostic']['failures'].get('current',[]))),
            future_failure_terms=dict(Counter(f for r in rows for n in ('next_1','next_2') for f in r['diagnostic']['failures'].get(n,[]))))
    usable=[dict(state_id=r['state_id'],task=r['task'],actions=r.get('beneficial_compatible_actions',[]))
            for r in states if r.get('beneficial_compatible_actions')]
    seen={r['state_id'] for r in states};expected={r['state_id'] for r in inventory['records']}
    summary=dict(schema_version=1,phase='2.24',baseline_commit='f202825',
        source_records=inventory['record_count'],expected_unique_states=inventory['unique_states'],
        evaluated_states=len(states),tasks=len({r['task'] for r in states}),scenes=len({r['scene'] for r in states}),
        missing_states=sorted(expected-seen),duplicate_source_records=inventory['record_count']-inventory['unique_states'],
        technical_integrity=not errors and seen==expected and all(c['motion_exact'] and c['context_exact'] for c in checks),
        errors=errors,paired_B1_cache_checks=len(checks),costs=dict(cost),
        adopted_unique_states=diagnostic_counts(adopted),all_signed_branches=diagnostic_counts(all_sides),
        beneficial_compatible_states=usable,no_useful_alternative_states=sum(r.get('no_useful_alternative') is True for r in states),
        diagnostic_ambiguity_states=sum(r.get('no_useful_alternative') is None for r in states),
        terminal_branches=sum(bool(b.get('terminal_observed')) for r in states for b in r['branches'].values()),
        censored_branches=sum(b.get('termination')=='budget_censored' for r in states for b in r['branches'].values()),
        task_rows=task_rows,scene_rows=scene_rows,state_files=[str(p) for p in paths],
        training_allowed=False,test_set_development=True,
        interpretation='Targeted adopted-state diagnostic; shared B1 two-window horizon, task nesting retained. No population-unbiased, whole-task quality or HSI added-value claim.')
    write_json(output/'summary.json',summary)
    write_json(output/'adopted_states.json',adopted)
    write_json(output/'signed_branches.json',all_sides)
    return summary


def render_continuation_outcomes(run_root,device='cuda:7',comparison='waypoint',tasks=None):
    """Fixed representative states, side-by-side actual W0/W+/W− continuations."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation,FFMpegWriter
    root=Path(run_root)
    repo=Path(__file__).resolve().parents[2]
    protocol=json.loads((root/'protocol.json').read_text())
    summary=json.loads((root/'analysis/summary.json').read_text())
    states=[(Path(p),json.loads(Path(p).read_text())) for p in summary['state_files']]
    out=root/'visualizations';out.mkdir(exist_ok=False)
    parents=[-1,0,0,0,1,2,3,4,5,6,7,8,9,9,9,12,13,14,16,17,18,19,20,20,20,21,21,21]
    records=[]
    for task in (tasks if tasks is not None else protocol['reporting']['representative_tasks']):
        path,state=min((p,s) for p,s in states if s['task']==task)
        tracks=torch.load(path.parent/'tracks.pt',map_location='cpu',weights_only=False)
        arms = ARMS if comparison == 'waypoint' else (state['source_selected'],
            'C1_'+state['source_selected'], 'C2_'+state['source_selected'])
        captions = arms if comparison == 'waypoint' else ('C0 B1', 'C1 local', 'C2 persistent')
        tracks = {a: tracks[a] for a in arms}
        full=np.concatenate([v['coarse_fk_human'].numpy().reshape(-1,3) for v in tracks.values()])
        lo=full.min(0)-[.5,.2,.5];hi=full.max(0)+[.5,.2,.5]
        scene=state['scene'];sdf_root=repo/'data/hosi_test/Scene_sdf'
        sdf=torch.as_tensor(np.load(sdf_root/(scene+'_sdf.npy')),device=device)
        meta=json.loads((sdf_root/(scene+'_sdf_info.json')).read_text());scale=max(meta['extents'])/2
        occupied=torch.nonzero(sdf.abs()*scale<.015)
        surface=(occupied.float()/(sdf.shape[0]-1)*2-1)*scale+torch.tensor(meta['centroid'],device=device)
        surface=surface[((surface>=torch.tensor(lo,device=device))&(surface<=torch.tensor(hi,device=device))).all(-1)]
        if len(surface)>2500:surface=surface[torch.linspace(0,len(surface)-1,2500,device=device).long()]
        surface=surface.cpu().numpy()
        fig=plt.figure(figsize=(13,4.6));lines=[];objects=[];titles=[]
        colors=['#666666','#2271ad','#1d9963'];center=(lo+hi)/2;radius=max(hi-lo)/2
        for i,arm in enumerate(arms):
            ax=fig.add_subplot(1,3,i+1,projection='3d')
            ax.scatter(surface[:,0],surface[:,2],surface[:,1],s=.5,c='#aaaaaa',alpha=.15)
            lines.append([ax.plot([],[],[],color=colors[i],lw=2)[0] for _ in parents[1:]])
            objects.append(ax.scatter([],[],[],s=3,c='#444444',alpha=.6))
            titles.append(ax.set_title(arm))
            ax.set(xlim=(center[0]-radius,center[0]+radius),ylim=(center[2]-radius,center[2]+radius),
                   zlim=(center[1]-radius,center[1]+radius),xlabel='X (m)',ylabel='Z (m)',zlabel='Y (m)')
            ax.view_init(elev=20,azim=-55);ax.set_box_aspect((1,1,1))
        title=fig.suptitle('')
        def update(frame):
            for i,arm in enumerate(arms):
                t=tracks[arm];available=frame<len(t['native_joints'])
                j=t['native_joints'][frame].numpy() if available else None
                o=t['object_points128'][min(frame//3,len(t['object_points128'])-1)].numpy() if available else np.empty((0,3))
                for line,k in zip(lines[i],range(1,len(parents))):
                    seg=j[[k,parents[k]]] if available else np.empty((0,3))
                    line.set_data(seg[:,0],seg[:,2]);line.set_3d_properties(seg[:,1])
                objects[i]._offsets3d=(o[:,0],o[:,2],o[:,1])
                label=(state['branches'][arm]['diagnostics']['category'] if comparison == 'waypoint'
                       else state['branches'][arm]['termination'])
                titles[i].set_text(f'{captions[i]}: {label}' if available else f'{captions[i]}: observed interval ended')
            stage='current' if frame<42 else 'next_1' if frame<84 else 'next_2'
            title.set_text(f'Targeted task {task:03d} / {state["state_id"]} / {frame/30:.2f}s / {stage}')
        frames=list(range(0,max(len(t['native_joints']) for t in tracks.values()),3))
        animation=FuncAnimation(fig,update,frames=frames,interval=100)
        target=out/f'task-{task:03d}-{state["state_id"]}.mp4'
        animation.save(target,writer=FFMpegWriter(fps=10),dpi=85)
        pictures=[]
        for i,frame in enumerate(np.linspace(0,frames[-1],4).astype(int)):
            update(int(frame));picture=out/f'task-{task:03d}-frame{i}.png';fig.savefig(picture,dpi=90);pictures.append(str(picture))
        plt.close(fig)
        records.append(dict(task=task,state=state['state_id'],video=str(target),frames=pictures))
    write_json(out/'review.json',dict(selection='Fixed333/376/421/20/375; lexicographically first registered state for each task, independent of outcomes',
        records=records,human_blind_review_pending=True,
        limits='SMPL-X skeleton plus128 object points and sparse scene SDF. Quantitative contact uses full object surface; renders support temporal comparison only.'))
    return records
