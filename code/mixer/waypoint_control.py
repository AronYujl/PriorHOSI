"""Paired first-window controls and attribution of a fixed complete-candidate pool."""
import json
import time
from pathlib import Path

import numpy as np
import torch

from .candidate_selection import move_tree, saved_endpoint_features
from .relational import RelationalGeometry, source_floor_height
from .temporal_preservation import root_local_features


def waypoint_variants(waypoint_world, path_xz, initial_root, bounds, occupancy,
                      navigation=True, offset_m=.1):
    """Fixed local path normal and input-only feasibility, without random draws."""
    path = torch.as_tensor(path_xz, device=waypoint_world.device, dtype=waypoint_world.dtype)
    result = dict(applicable=False, reasons=[], variants={}, normal=None)
    if not navigation:
        result['reasons'].append('no_intermediate_navigation')
        return result
    if len(path) < 2:
        result['reasons'].append('degenerate_path')
        return result
    index = int((path-waypoint_world[[0, 2]]).square().sum(-1).argmin())
    if index == len(path)-1:
        result['reasons'].append('original_waypoint_is_path_endpoint')
        return result
    tangent = path[index+1]-path[index]
    if float(tangent.norm()) == 0:
        result['reasons'].append('zero_path_tangent')
        return result
    normal = torch.stack((-tangent[1], tangent[0]))/tangent.norm()
    result.update(path_index=index, normal=normal.tolist())
    for name, sign in (('Wplus', 1), ('Wminus', -1)):
        waypoint = waypoint_world.clone()
        waypoint[[0, 2]] += sign*offset_m*normal
        query = waypoint.clone(); query[1] = initial_root[1]
        reason = None
        if not bool(((query >= bounds[:3]) & (query < bounds[3:6])).all()):
            reason = 'outside_scene'
        elif bool(occupancy(query.reshape(1, 1, 3)).any()):
            reason = 'occupied_offset_at_initial_pelvis_height'
        result['variants'][name] = dict(waypoint_world=waypoint.tolist(), query=query.tolist(), reason=reason)
        if reason:
            result['reasons'].append(name+':'+reason)
    result['applicable'] = not result['reasons']
    return result


def local_waypoint(world, mat):
    from utils import transform_points
    return transform_points(world.reshape(1, 1, 3), torch.linalg.inv(mat)).reshape(1, 3)


def decode_motion(motion, dataset, offsets, context, points):
    geometry = RelationalGeometry(motion, dataset, offsets, context, points)
    state = geometry.decode(motion.new_zeros(*motion.shape[:2], geometry.dimension))
    state['local_human'] = root_local_features(geometry, state)
    return state


def coordinate_difference(first, second):
    tensors = dict(human=second['human']-first['human'],
                   root=second['human'][..., 0, :]-first['human'][..., 0, :],
                   object=second['object_translation_world']-first['object_translation_world'],
                   local_human=second['local_human']-first['local_human'])
    return {k+'_rms_mm': float(v[:, 2:].double().square().mean().sqrt()*1000) for k, v in tensors.items()}


def _masked_mean(values, mask):
    count = int(mask.sum())
    return float(values[mask].double().mean()) if count else 0.


@torch.no_grad()
def control_metrics(state, reference, clean_reference, dataset, context, normal, sign):
    """Fixed reference masks and object-relative anchors; coarse window proxies."""
    human, base = state['human'], reference['human']
    object_rotation = state['object_rotation_world']
    relative = (object_rotation.transpose(-1, -2)[..., None, :, :] @
                (human[..., 22:24, :]-state['object_translation_world'][..., None, :])[..., None]).squeeze(-1)
    anchors = (reference['object_rotation_world'].transpose(-1, -2)[..., None, :, :] @
               (base[..., 22:24, :]-reference['object_translation_world'][..., None, :])[..., None]).squeeze(-1)
    active = clean_reference[:, 2:, 228:230] > .95
    contact = (relative[:, 2:]-anchors[:, 2:]).norm(dim=-1)
    hands = torch.cdist(human[:, 2:, 22:24].flatten(0, 1), state['object_surface'][:, 2:].flatten(0, 1)).amin((-1, -2))
    floor, _ = source_floor_height(base)
    stance = base[..., (7, 8, 10, 11), 1]-floor[:, None, None] < base.new_tensor([.08, .08, .04, .04])
    support = stance[:, 2:] & stance[:, 1:-1]
    feet = human[..., (7, 8, 10, 11), :]
    speed = (feet[:, 2:]-feet[:, 1:-1])[..., (0, 2)].norm(dim=-1)/.1
    query = torch.cat((human[:, 2:], state['object_surface'][:, 2:]), -2)
    occupied, nearest = dataset.get_nearest_free_voxel(query, context['scene_flag'])
    distance = (query-nearest).double().square().sum(-1)
    directed = lambda delta: float((delta[:, 2:, (0, 2)]*normal).sum(-1).mean()*sign)
    return dict(root_directed_m=directed(human[:, :, 0]-base[:, :, 0]),
                object_directed_m=directed(state['object_translation_world']-reference['object_translation_world']),
                contact_anchor_m=_masked_mean(contact, active), contact_count=int(active.sum()),
                contact_surface_distance_m=float(hands.mean()), contact_fraction=float((hands < .1).float().mean()),
                support_speed_m_per_s=_masked_mean(speed, support), support_count=int(support.sum()),
                human_scene_RMS_cm=float(distance[..., :24].mean().sqrt()*100),
                object_scene_RMS_cm=float(distance[..., 24:].mean().sqrt()*100),
                human_occupied_fraction=float(occupied[..., :24].float().mean()),
                object_occupied_fraction=float(occupied[..., 24:].float().mean()),
                world_joint_speed_m_per_s=float((human[:, 2:, :22]-human[:, 1:-1, :22]).norm(dim=-1).mean()/.1),
                **coordinate_difference(reference, state))


def quality_margins(gate):
    return dict(contact_anchor_m=gate['contact_anchor_increase_m'],
                contact_surface_distance_m=gate['contact_surface_distance_increase_m'],
                support_speed_m_per_s=gate['support_speed_increase_m_per_s'],
                human_scene_RMS_cm=gate['human_scene_RMS_increase_cm'],
                object_scene_RMS_cm=gate['object_scene_RMS_increase_cm'],
                human_occupied_fraction=gate['occupied_fraction_increase'],
                object_occupied_fraction=gate['occupied_fraction_increase'])


def _write(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def _append(path, value):
    with Path(path).open('a') as handle:
        handle.write(json.dumps(value, allow_nan=False)+'\n')


def _model_input_trace(model):
    trace = dict(calls=0)
    def capture(module, args):
        trace['calls'] += 1
        if trace['calls'] == 1:
            trace.update(latent=args[0].detach().clone(), timestep=args[1].detach().clone(),
                         text=args[2].detach().clone(), bps=args[3].detach().clone(),
                         goals=args[4].detach().clone(), progress=args[5].detach().clone())
        elif not torch.equal(trace['goals'], args[4]):
            raise AssertionError('model goal changed within fixed control window')
    return trace, model.register_forward_pre_hook(capture)


def execute_waypoint_scene(run_root, scene_name, resolved_config, device='cuda:0'):
    """A/B/C in one scene, frozen cached first-window conditions and full B1."""
    import inspect
    import subprocess
    import hydra
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from priors.hoi.models import load_trained_hoi_prior
    from test_infbagel_hosi import seed_everything
    from utils import init_model, transform_points
    from astar import get_path
    from .composed_sampler import HOSIComposedSampler
    from .scene_calibration import recover_temporal_window
    if subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('reportable waypoint probe requires clean worktree')
    root = Path(run_root); protocol = json.loads((root/'protocol.json').read_text())
    cfg = OmegaConf.load(resolved_config)
    if not cfg.waypoint_probe.enabled:
        return dict(enabled=False, generated_windows=0)
    cfg.device = device; cfg.dataset.device = device; cfg.dataset.vis = True
    cfg.dataset.load_object_payload = False; cfg.dataset.test_scene_name = scene_name
    cfg.sampler.pelvis.hoi_adapter.device = device; cfg.sampler.pelvis.hsi_sampler.device = device
    dataset = InfBaGelDataset(**cfg.dataset)
    dataset.obj_rest_verts = {k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
    hoi, metadata = load_trained_hoi_prior(cfg.ckpt_path, torch.device(device), weight_variant=cfg.checkpoint_weight_variant)
    hoi.eval().requires_grad_(False)
    hsi = init_model(OmegaConf.merge(cfg.model.infbagel, {'ckpt':cfg.hsi_ckpt_path}), device=device, eval=True)
    hsi.eval().requires_grad_(False)
    slots = json.loads(Path(protocol['candidate_manifest']).read_text())['candidates']
    tasks = json.loads(Path(protocol['task_manifest']).read_text())['tasks']
    tasks = [t for t in tasks if t['scene_name'] == scene_name]
    native = json.loads((Path(cfg.dataset.folder).parent/'hosi_test/data'/(scene_name+'.json')).read_text())
    out = root/('scene-'+scene_name); out.mkdir(exist_ok=False)
    start = time.perf_counter(); generated = 0
    torch.cuda.reset_peak_memory_stats(device)
    for task_row in tasks:
        ordinal = task_row['canonical_ordinal']; task = native[task_row['test_idx']]
        pool = {s['candidate']:s for s in slots if s['task'] == ordinal}
        saved = {i:torch.load(s['motion_path'], map_location='cpu', weights_only=False) for i,s in pool.items()}
        point_cloud = dataset.obj_rest_verts[task['object_name']]
        points = point_cloud[torch.linspace(0, len(point_cloud)-1, 128, device=device).long()][None]
        # A uses world coordinates from each window; later histories are descriptive.
        stages = {}; audits = {}
        for index, motion in saved.items():
            stages[index] = []
            for w, (snapshot, world) in enumerate(zip(motion['corrections'], motion['windows'])):
                _, converted, recovery, context, offsets = recover_temporal_window(
                    dataset, motion, snapshot, world, task, points, device)
                current = {stage:decode_motion(converted[stage], dataset, offsets, context, points)
                           for stage in ('raw_source', 'proposal', 'edited')}
                stages[index].append(current)
                _append(out/'cache_recovery.jsonl', dict(task=ordinal, candidate=index, window=w, audit=recovery))
            # B computes physical endpoints before reading the sealed native metrics.
            endpoints = saved_endpoint_features(motion, task, device)
            expected = json.loads(Path(pool[index]['audit_path']).read_text())['metrics']
            errors = dict(root_cm=abs(endpoints['root_endpoint_m']*100-expected['xy_points_err']),
                          object_cm=abs(endpoints['object_endpoint_m']*100-expected['end_obj_trans_err']))
            if max(errors.values()) > 1e-5 or endpoints['completed'] != expected['completed']:
                raise AssertionError('retained endpoint differs from native evaluation')
            _append(out/'endpoints.jsonl', dict(task=ordinal, candidate=index, **endpoints, errors=errors))
        for index in (1, 2, 3):
            for w, (base, altered) in enumerate(zip(stages[0], stages[index])):
                differences = {stage:coordinate_difference(base[stage], altered[stage]) for stage in base}
                ratios = {key:(differences['edited'][key]/value if value >= .001 else None)
                          for key, value in differences['raw_source'].items()}
                _append(out/'cache_differences.jsonl', dict(task=ordinal, candidate=index, window=w,
                        differences=differences, edited_over_raw=ratios,
                        history_common_exact=torch.equal(saved[0]['corrections'][w]['raw_source'][:, :2],
                                                         saved[index]['corrections'][w]['raw_source'][:, :2])))
        snapshot = saved[1]['corrections'][0]
        context = move_tree(snapshot['replay_context'], device)
        context.update(obj_rest_verts=dataset.obj_rest_verts, obj_vert_normals={})
        offsets = snapshot['rest_offsets'].to(device)
        fixed = saved[0]['corrections'][0]['raw_source'][:, :2].to(device)
        if not torch.equal(fixed.cpu(), snapshot['raw_source'][:, :2]):
            raise AssertionError('first-window initial histories differ across seeds')
        path = get_path(np.asarray(task['start_location'])[[0, 2]], np.asarray(task['pelvis_goal'])[[0, 2]], dataset)
        world_waypoint = transform_points(context['pelvis_goal'].reshape(1, 1, 3), context['mat']).reshape(3)
        initial_root = stages[0][0]['raw_source']['human'][0, 0, 0]
        proposal = waypoint_variants(world_waypoint, path, initial_root, dataset.scene_grid_torch.to(device),
                     lambda p:dataset.get_occ_for_points(p, None, context['scene_flag']),
                     bool(context['is_loco'].all() and context['need_pelvis_dir'].all()), cfg.waypoint_probe.offset_m)
        proposal.update(task=ordinal, scene=scene_name, object=task['object_name'], original_waypoint_world=world_waypoint.tolist())
        _append(out/'applicability.jsonl', proposal)
        sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
        sampler.set_dataset_and_model(dataset, hoi, hsi_model=hsi)
        normal = fixed.new_tensor(proposal['normal'] if proposal['normal'] is not None else [0., 0.])
        arms = ['W0'] + (['Wplus', 'Wminus'] if proposal['applicable'] else [])
        if proposal['applicable'] and ordinal in protocol['generation']['repeat_tasks']:
            arms.append('Wplus_repeat')
        arm_results = {}; arm_traces = {}; decoded = {}
        for arm in arms:
            conditions = move_tree(context, device)
            if arm != 'W0':
                key = 'Wplus' if arm == 'Wplus_repeat' else arm
                world = fixed.new_tensor(proposal['variants'][key]['waypoint_world'])
                conditions['pelvis_goal'] = local_waypoint(world, context['mat'])
            if sampler.inference_engineering:
                conditions['static_occ_cache'] = {}
            keys = inspect.signature(HOSIComposedSampler.p_sample_loop).parameters
            arguments = {k:v for k,v in conditions.items() if k in keys}
            arguments.update(fixed_points=fixed.clone(), human_dict={'rest_human_offsets':offsets.clone()})
            seed_everything(42+ordinal); sampler.inner_hoi.sample_calls = 0
            sampler.scene_editor.records.clear(); sampler.scene_editor.motion_records.clear()
            trace, hook = _model_input_trace(hoi)
            torch.cuda.synchronize(device); began = time.perf_counter()
            try:
                sampler.p_sample_loop(**arguments)
            finally:
                hook.remove()
            torch.cuda.synchronize(device); seconds = time.perf_counter()-began; generated += 1
            current = move_tree(sampler.scene_editor.motion_records[-1], device)
            record = sampler.scene_editor.records[-1]
            expected_goal = conditions['pelvis_goal'].clone(); expected_goal[:, 1] = 0
            if not torch.equal(trace['goals'][:, :3], expected_goal) or trace['calls'] != 500:
                raise AssertionError('waypoint did not reach the fixed HOI model interface')
            if not torch.equal(current['edited'][:, :2], fixed) or not torch.isfinite(current['edited']).all():
                raise AssertionError('fixed history/finite control failed')
            if arm == 'W0':
                for stage in ('raw_source', 'proposal', 'edited'):
                    if not torch.equal(current[stage].cpu(), saved[0]['corrections'][0][stage]):
                        raise AssertionError('W0 bitwise cached regression failed: '+stage)
            else:
                for key in ('latent', 'text', 'bps', 'progress'):
                    if not torch.equal(trace[key], arm_traces['W0'][key]):
                        raise AssertionError('paired first model input changed: '+key)
                if not torch.equal(trace['goals'][:, 3:], arm_traces['W0']['goals'][:, 3:]):
                    raise AssertionError('waypoint changed object/scene goal encoding')
            states = {stage:decode_motion(current[stage], dataset, offsets, context, points)
                      for stage in ('raw_source', 'edited')}
            sign = -1 if arm == 'Wminus' else 1
            if arm == 'W0':
                decoded = states
            metrics = {stage:control_metrics(state, decoded[stage],
                       current[stage] if arm == 'W0' else arm_results['W0'][stage],
                       dataset, context, normal, sign) for stage,state in states.items()}
            if arm == 'Wplus_repeat':
                if any(not torch.equal(current[k], arm_results['Wplus'][k]) for k in ('raw_source', 'proposal', 'edited')):
                    raise AssertionError('repeated waypoint output differs')
                if any(not torch.equal(trace[k], arm_traces['Wplus'][k]) for k in ('latent', 'goals', 'progress')):
                    raise AssertionError('repeated model input differs')
            arm_results[arm] = current; arm_traces[arm] = trace
            row = dict(task=ordinal, arm=arm, scene=scene_name, object=task['object_name'],
                       seed=42+ordinal, seconds=seconds, model_calls=trace['calls'], metrics=metrics,
                       editor=record, applicable=proposal['applicable'],
                       integrity=dict(history=True, finite=True, goals=True, paired_inputs=True, W0_bitwise=arm=='W0', repeat_exact=arm=='Wplus_repeat'))
            _append(out/'control_results.jsonl', row)
            payload = dict(task=ordinal, arm=arm, mode='independent_first_window_control',
                           snapshot=move_tree(current, 'cpu'), states=move_tree(states, 'cpu'),
                           model_trace=move_tree(trace, 'cpu'), context=move_tree({k:v for k,v in conditions.items()
                           if k not in ('obj_rest_verts','obj_vert_normals','static_occ_cache')}, 'cpu'),
                           proposal=proposal)
            with (out/f'task-{ordinal:03d}-{arm}.pt').open('xb') as handle:
                torch.save(payload, handle)
            print('complete', ordinal, arm, 'seconds', round(seconds, 3), flush=True)
    summary = dict(tasks=len(tasks), generated_windows=generated, seconds=time.perf_counter()-start,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(device), new_HSI_calls=0,
                   full_native_rollouts=0, checkpoint_metadata=metadata)
    _write(out/'summary.json', summary)
    return summary


def summarize_waypoint_probe(run_root, device='cuda:7'):
    from .scene_calibration import paired_local_metrics
    root = Path(run_root); protocol = json.loads((root/'protocol.json').read_text())
    out = root/'analysis'; out.mkdir(exist_ok=False)
    def read(name):
        return [json.loads(line) for p in sorted(root.glob('scene-*/'+name)) for line in p.read_text().splitlines()]
    controls = read('control_results.jsonl'); applicability = read('applicability.jsonl')
    differences = read('cache_differences.jsonl'); endpoints = read('endpoints.jsonl')
    if len(applicability) != 28 or len(endpoints) != 112 or sum(r['arm']=='W0' for r in controls) != 28:
        raise ValueError('incomplete registered A/B/C coverage')
    def average(rows):
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    attribution = {}
    for scope, selected in (('first_window', [r for r in differences if r['window']==0]),
                            ('later_windows_descriptive', [r for r in differences if r['window']>0])):
        by_task = {}
        for task in sorted(set(r['task'] for r in selected)):
            data = [r for r in selected if r['task']==task]
            by_task[str(task)] = {stage:average([r['differences'][stage] for r in data])
                                  for stage in ('raw_source','proposal','edited')}
        means = {stage:average([r[stage] for r in by_task.values()]) for stage in ('raw_source','proposal','edited')}
        attribution[scope] = dict(comparisons=len(selected), per_task=by_task, task_means=means,
            common_history_comparisons=sum(r['history_common_exact'] for r in selected),
            edited_over_raw={k:means['edited'][k]/v if v>=.001 else None for k,v in means['raw_source'].items()})
    source = Path(protocol['source_run'])/'analysis'
    native = json.loads((source/'candidate_native_metrics.json').read_text())
    decisions = json.loads((source/'decisions.json').read_text())
    old = json.loads((source/'per_task.json').read_text())
    groups = {g:{} for g in ('registered','completed_only','unprotected','CG')}
    picks = []; hs = 'scene_human_penetration_s_mean'
    for d in decisions:
        task = d['task']; name = str(task)
        pool = {r['candidate']:r['metrics'] for r in native if r['task']==task}
        cg = pool[d['selected']['CG']]
        strict = [i for i in d['acceptable'] if pool[i]['completed']>=cg['completed']
                  and pool[i]['contact_percent']>=cg['contact_percent']-.02
                  and pool[i]['foot_sliding']<=cg['foot_sliding']+.01
                  and pool[i]['scene_obj_penetration_s_mean']<=cg['scene_obj_penetration_s_mean']+.02
                  and all(pool[i][k]<=cg[k]+1 for k in ('xy_points_err','end_obj_trans_err'))]
        options = dict(registered=strict or [d['selected']['CG']],
                       completed_only=[i for i in pool if pool[i]['completed']>=cg['completed']], unprotected=list(pool))
        for group, ids in options.items():
            index = min(ids, key=lambda i:(pool[i][hs], i))
            groups[group][name] = pool[index]
            picks.append(dict(task=task, oracle=group, selected=index, feasible=ids))
        groups['CG'][name] = cg
        if groups['registered'][name] != old['oracle'][name]:
            raise AssertionError('sealed protected oracle reconstruction differs')
    oracle = dict(means={k:average(list(v.values())) for k,v in groups.items()}, picks=picks,
                  contrasts={g+'-CG':paired_local_metrics(groups['CG'],groups[g],device)
                             for g in ('registered','completed_only','unprotected')}, registered_exact=True,
                  scope='descriptive per-task oracle; not a deployable or aggregate-tradeoff bound')
    applicable = [r for r in applicability if r['applicable']]
    eligible = set(r['task'] for r in applicable)
    data = {(r['task'],r['arm']):r for r in controls}
    gate = protocol['gate']; gates = {}; contrasts = {}; means = {}; scene_means = {}; useful = []
    gates['applicable_coverage'] = len(eligible)>=gate['minimum_applicable_pairs']
    gates['integrity'] = all(all(r['editor']['final_guards'].values()) and not r['editor']['invalid_proposal']
                             and r['editor']['history_exact'] and r['editor']['common_exact'] and r['editor']['contact_exact']
                             for r in controls)
    margins = quality_margins(gate)
    if eligible:
        for stage in ('raw_source','edited'):
            base = {str(t):data[(t,'W0')]['metrics'][stage] for t in sorted(eligible)}
            for arm in ('Wplus','Wminus'):
                group = {str(t):data[(t,arm)]['metrics'][stage] for t in sorted(eligible)}
                name = stage+'_'+arm
                contrasts[name] = paired_local_metrics(base, group, device)
                means[name] = dict(W0=average(list(base.values())), altered=average(list(group.values())))
                values = [r['root_directed_m'] for r in group.values()]
                scenes = {a['scene'] for a in applicable}
                scene_means[name] = {s:float(np.mean([group[str(a['task'])]['root_directed_m'] for a in applicable if a['scene']==s])) for s in scenes}
                gates[name+'_root_mean'] = np.mean(values)>=gate['each_sign_mean_directed_root_m']
                gates[name+'_root_positive_fraction'] = np.mean(np.asarray(values)>0)>=gate['each_sign_positive_task_fraction']
                gates[name+'_root_CI'] = contrasts[name]['root_directed_m']['ci'][0]>0
                gates[name+'_root_scenes'] = all(v>=0 for v in scene_means[name].values())
                gates[name+'_object_mean'] = np.mean([r['object_directed_m'] for r in group.values()])>=0
                for metric, margin in margins.items():
                    gates[name+'_'+metric] = contrasts[name][metric]['ci'][1]<=margin
                gates[name+'_contact_fraction'] = contrasts[name]['contact_fraction']['ci'][0]>=-gate['contact_fraction_drop']
                ratios = [group[t]['world_joint_speed_m_per_s']/base[t]['world_joint_speed_m_per_s']
                          for t in group if base[t]['world_joint_speed_m_per_s']>0]
                gates[name+'_speed_retention'] = bool(ratios) and np.mean(ratios)>=gate['world_joint_speed_retention']
        for task in sorted(eligible):
            base = data[(task,'W0')]['metrics']['edited']; passes = {}
            for arm in ('Wplus','Wminus'):
                value = data[(task,arm)]['metrics']['edited']
                passes[arm] = (value['root_directed_m']>=.01 and value['object_directed_m']>=0
                    and all(value[k]-base[k]<=v for k,v in margins.items())
                    and value['contact_fraction']-base['contact_fraction']>=-.05
                    and value['world_joint_speed_m_per_s']>=.95*base['world_joint_speed_m_per_s'])
            useful.append(dict(task=task, passed=all(passes.values()), arms=passes))
    gates['useful_pairs'] = bool(useful) and sum(r['passed'] for r in useful)/len(useful)>=gate['useful_task_fraction']
    gates = {k:bool(v) for k,v in gates.items()}
    numerical = all(gates.values())
    result = dict(phase='2.21',tasks=28,cache_candidates=112,cache_windows=len(read('cache_recovery.jsonl')),
                  applicable_pairs=len(eligible),applicability=applicability,
                  generated_windows=len(controls),repeat_windows=sum(r['arm']=='Wplus_repeat' for r in controls),
                  gates=gates,control_means=means,scene_directed_root_means=scene_means,
                  useful_pairs=useful,numerical_gate_passed=numerical,
                  decision=('INCONCLUSIVE_COVERAGE' if not gates['applicable_coverage'] else ('REQUIRES_VISUAL_REVIEW' if numerical else 'NO-GO')),
                  endpoint_max_errors={k:max(r['errors'][k] for r in endpoints) for k in ('root_cm','object_cm')},
                  endpoint_completed_exact=True,visual_review_pending=True,test_set_development=True,
                  HSI_forwards=0,complete_native_rollouts=0,route_pool_started=False,HSI_selection_started=False,
                  source_attribution=attribution,oracles=oracle,
                  cost=dict(generation_seconds=sum(r['seconds'] for r in controls),model_calls=sum(r['model_calls'] for r in controls),
                            lanes=[json.loads(p.read_text()) for p in sorted(root.glob('scene-*/summary.json'))]))
    _write(out/'summary.json',result); _write(out/'control_contrasts.json',contrasts)
    _write(out/'oracle_per_task.json',groups); _write(out/'endpoints.json',endpoints)
    return result
