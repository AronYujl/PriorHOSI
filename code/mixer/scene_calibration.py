"""Independent-development diagnostics for one globally fixed evidence scale."""
import json
from pathlib import Path

import numpy as np
import torch


def save_development_episode(editor, record_start, output_dir, episode, windows, seconds):
    records = editor.records[record_start:]
    stem = 'episode-%03d' % episode['canonical_ordinal']
    payload = dict(episode=episode, window_count=len(windows), records=records,
                   generation_seconds=seconds, quality_evaluated=False,
                   metric_scope='relation/voxel window proxies; no native mesh-SDF evaluation')
    directory = Path(output_dir)
    with (directory / (stem + '.json')).open('x') as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
    with (directory / (stem + '.pt')).open('xb') as handle:
        torch.save(dict(episode=episode, windows=windows, corrections=editor.motion_records), handle)
    editor.motion_records.clear()
    return payload


def replay_development_episode(sampler, source_dir, ordinal, object_vertices, output_dir, device):
    """Evaluate saved sources/conditions without sampling a new HOI trajectory."""
    import time
    paths = sorted(Path(source_dir).glob(f'*-shard*/episode-{ordinal:03d}.pt'))
    if len(paths) != 1:
        raise ValueError(f'expected one saved episode for ordinal {ordinal}, found {len(paths)}')
    saved = torch.load(paths[0], map_location='cpu', weights_only=False)
    metadata = json.loads(paths[0].with_suffix('.json').read_text())
    if len(saved['corrections']) != len(metadata['records']):
        raise ValueError('saved source/context coverage mismatch')
    editor = sampler.scene_editor
    start_record, started = len(editor.records), time.perf_counter()
    def move(items):
        if torch.is_tensor(items):
            return items.to(device)
        if isinstance(items, dict):
            return {k: move(v) for k, v in items.items()}
        return items
    for snapshot, old_record in zip(saved['corrections'], metadata['records']):
        context = move(snapshot['replay_context'])
        context['obj_rest_verts'] = object_vertices
        sampler.inner_hoi.sample_calls = snapshot['window']
        bps = snapshot['local_bps']
        output = editor.edit(sampler, snapshot['raw_source'].to(device), move(snapshot['hoi_arguments']),
            None if bps is None else bps.to(device), context, snapshot['rest_offsets'].to(device),
            old_record['seed'])
        current = editor.motion_records[-1]
        raw_equal = torch.equal(output.cpu(), snapshot['raw_source'])
        reference_equal = torch.equal(current['reference'], snapshot['reference'])
        mask_equal = torch.equal(current['contact_mask'], snapshot['contact_mask'])
        if not (raw_equal and reference_equal and mask_equal):
            raise AssertionError('saved replay source/reference/contact changed')
        regression = None
        if editor.diagnostics.get('probe') == 'relation_compatible':
            old = next(p['parameters'] for p in snapshot['probe_motions'] if p['view'] == 'lambda0_short_edit')
            new = next(p['parameters'] for p in current['probe_motions'] if p['view'] == 'G0_short_edit')
            regression = torch.equal(old, new)
            if not regression:
                raise AssertionError('lambda0 saved real-window regression failed')
        editor.records[-1]['replay_audit'] = dict(source_path=str(paths[0]), raw_source_exact=raw_equal,
            reference_exact=reference_equal, contact_mask_exact=mask_equal, lambda0_parameters_exact=regression)
    return save_development_episode(editor, start_record, output_dir, saved['episode'],
                                    saved['windows'], time.perf_counter()-started)


def estimate_global_scale(episodes, calibration_scenes, target_ratio=.1,
                          minimum_active_windows=6):
    """Scene-balanced median of positive-scene source gradient norm ratios.

    Inputs must be passive lambda1 records. No verification row participates in
    selection, and no tiny-signal denominator is replaced by an arbitrary floor.
    """
    by_scene = {scene: [] for scene in calibration_scenes}
    inactive = zero_signal = 0
    for episode in episodes:
        scene = episode['episode']['scene_name']
        if scene not in by_scene:
            continue
        for record in episode['records']:
            if record['mode'] != 'calibrate':
                raise ValueError('scale estimation requires passive source records')
            scene_energy = record['source_terms']['human_scene'][0] + record['source_terms']['object_scene'][0]
            if scene_energy == 0:
                inactive += 1
                continue
            ratios = []
            for iteration in record['iterations']:
                gradients = iteration['parameter_gradient_norms']
                numerator, denominator = gradients['explicit'][0], gradients['hsi_evidence'][0]
                if not np.isfinite(numerator) or not np.isfinite(denominator):
                    raise ValueError('nonfinite source gradient')
                if denominator == 0:
                    zero_signal += 1
                else:
                    ratios.append(numerator / denominator)
            if ratios:
                by_scene[scene].append(float(np.median(ratios)))
    if any(not values for values in by_scene.values()) or sum(map(len, by_scene.values())) < minimum_active_windows:
        raise ValueError('insufficient positive-scene calibration coverage')
    scene_medians = {scene: float(np.median(values)) for scene, values in by_scene.items()}
    unrounded = target_ratio * float(np.median(list(scene_medians.values())))
    coefficient = float(format(unrounded, '.2g'))
    return dict(lambda_dp=coefficient, unrounded=unrounded, target_ratio=target_ratio,
                calibration_scenes=list(calibration_scenes), scene_medians=scene_medians,
                active_windows={scene: len(values) for scene, values in by_scene.items()},
                inactive_windows=inactive, zero_hsi_level_count=zero_signal,
                rule='target times median(scene median(window median(level norm ratios))), two significant figures')


def summarize_relation_compatible(source_root, output_dir, device='cuda:7'):
    """Task-paired local screening; all outcomes and failure strata stay visible."""
    episodes = [json.loads(p.read_text()) for p in sorted(Path(source_root).glob('*-shard*/episode-*.json'))]
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    groups = ('G0','G1','G2','G3')
    short = {g: {} for g in groups}
    rays = {str(s): {g: {} for g in groups[1:]} for s in (.001,.005)}
    scene_by_task, window_rows, projection, ray_rows, short_rows = {}, [], [], [], []
    def scalar_metrics(values):
        return {k: float(v[0]) for k,v in values.items()}
    def average(rows):
        return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}
    def write(name, value):
        with (directory/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    seen = set()
    for episode in episodes:
        identity = episode['episode']; ordinal = identity['canonical_ordinal']; task = f'{ordinal:03d}'
        if task in seen:raise ValueError('duplicate replay task')
        seen.add(task);scene_by_task[task]=identity['scene_name']
        task_short={g:[] for g in groups}
        task_rays={s:{g:[] for g in groups[1:]} for s in rays}
        for record in episode['records']:
            audit=record['replay_audit']; diagnostic=record['view_diagnostic']
            if not all(audit[k] for k in ('raw_source_exact','reference_exact','contact_mask_exact','lambda0_parameters_exact')):
                raise ValueError('failed source or lambda0 pairing')
            if not all(diagnostic[k] for k in ('ambient_rng_preserved','scene_storage_unchanged','context_unchanged','source_anchors_unchanged')):
                raise ValueError('failed diagnostic purity')
            if not record['history_exact'] or not record['contact_exact']:raise ValueError('changed history/contact')
            source=scalar_metrics(record['source_metrics'])
            window_rows.append(dict(task=task,scene=identity['scene_name'],object=identity['object_name'],
                window=record['window'],source_metrics=source,source_domain=record['source_domain'],
                active_anchors=diagnostic['source_contact_count']))
            for g in groups:
                item=diagnostic['short_edits'][g];r=item['record']
                m=scalar_metrics(r['final_metrics'])
                m.update(physical_rms_m=item['physical_rms_m'],accepted_steps=sum(i['solver']['accepted'][0] for i in r['iterations']),
                    seconds=r['seconds'],teacher_seconds=r['teacher_seconds'],solver_seconds=r['solver_seconds'])
                task_short[g].append(m)
                short_rows.append(dict(task=task,scene=identity['scene_name'],object=identity['object_name'],window=record['window'],group=g,
                    metrics=m,source_metrics=source,reasons=[i['solver']['reason'][0] for i in r['iterations']]))
                if not r['history_exact'] or not r['contact_exact']:raise ValueError('short editor changed history/contact')
                for i in r['iterations']:
                    if 'projection' in i:projection.append(dict(kind='short',task=task,window=record['window'],group=g,level=i['level'],**i['projection']))
            window_rays={s:{g:[] for g in groups[1:]} for s in rays}
            for iteration in record['iterations']:
                g=iteration['group']
                if not iteration['static_conditions_equal'] or not iteration['base_prediction_equal']:raise ValueError('unpaired teacher')
                if iteration['projection'] is not None:
                    projection.append(dict(kind='ray',task=task,window=record['window'],group=g,level=iteration['level'],draw=iteration['draw'],**iteration['projection']))
                for probe in iteration['physical_step_probes']:
                    scale=str(probe['target_rms_m']);m=scalar_metrics(probe['metrics'])
                    m.update(physical_rms_m=probe['actual_rms_m'],matched=float(probe['reason']=='matched'),admissible=float(probe['domain_admissible']))
                    window_rays[scale][g].append(m)
                    ray_rows.append(dict(task=task,scene=identity['scene_name'],object=identity['object_name'],window=record['window'],group=g,
                        scale=scale,draw=iteration['draw'],level=iteration['level'],reason=probe['reason'],metrics=m,source_metrics=source,
                        source_outside=record['source_domain']['outside_points'][0]>0))
            for s in rays:
                for g in groups[1:]:task_rays[s][g].append(average(window_rays[s][g]))
        for g in groups:short[g][task]=average(task_short[g])
        for s in rays:
            for g in groups[1:]:rays[s][g][task]=average(task_rays[s][g])
    if len(episodes)!=24 or len(window_rows)!=68:raise ValueError('incomplete registered24/68 coverage')
    write('window_sources.json',window_rows);write('ray_rows.json',ray_rows);write('short_rows.json',short_rows);write('projection.json',projection)
    def scene_means(source):
        return {scene:average([m for task,m in source.items() if scene_by_task[task]==scene]) for scene in sorted(set(scene_by_task.values()))}
    paired_results,means={},{}
    for kind,cells in [('short',short)]+[('ray'+s,rows) for s,rows in rays.items()]:
        means[kind]={g:average(list(rows.values())) for g,rows in cells.items()}
        for unit in ('task','scene'):
            data=cells if unit=='task' else {g:scene_means(rows) for g,rows in cells.items()}
            write(f'{kind}_{unit}_inputs.json',data)
            paired_results[kind+'_'+unit]={f'G2-{b}':paired_local_metrics(data[b],data['G2'],device) for b in cells if b!='G2'}
    feasibility={}
    for scale in rays:
        rows=[r for r in ray_rows if r['scale']==scale]
        keys=lambda r:(r['task'],r['window'],r['draw'],r['level'])
        common=set(keys(r) for r in rows)
        for g in groups[1:]:common &= {keys(r) for r in rows if r['group']==g and r['metrics']['matched'] and r['metrics']['admissible']}
        feasibility[scale]={'common_queries':len(common),'total_queries':len(rows)//3,'groups':{}}
        for g in groups[1:]:
            own=[r for r in rows if r['group']==g]
            reasons={k:sum(r['reason']==k for r in own) for k in sorted(set(r['reason'] for r in own))}
            subset=[r['metrics'] for r in own if keys(r) in common]
            feasibility[scale]['groups'][g]=dict(n=len(own),matched=sum(r['metrics']['matched'] for r in own),
                admissible=sum(r['metrics']['admissible'] for r in own),reasons=reasons,
                common_mean=average(subset) if subset else None)
    ray_projection=[p for p in projection if p['kind']=='ray']
    keep_fraction=sum(p['r_keep'] is not None and p['r_keep']>=.001 for p in ray_projection)/len(ray_projection)
    hs,os,contact,stance='human_scene_residual_cm','object_scene_residual_cm','contact_anchor_drift_cm','stance_increment_cm'
    raycontact=paired_results['ray0.005_task']['G2-G1'][contact]
    gates={}
    gates['contact_reduction']=means['ray0.005']['G2'][contact]<=.2*means['ray0.005']['G1'][contact] and raycontact['ci'][1]<0
    gates['motion_retained']=keep_fraction>=.95 and all(feasibility[s]['groups'][g]['matched']/feasibility[s]['groups'][g]['n']>=.95 for s in rays for g in ('G2','G3'))
    for b in ('G0','G3'):
        contrast=paired_results['short_task']['G2-'+b]
        gates['hs_gain_vs_'+b]=(contrast[hs]['delta']<=-max(.01,.05*means['short'][b][hs]) and contrast[hs]['ci'][1]<0
                              and paired_results['short_scene']['G2-'+b][hs]['delta']<=0)
        gates['os_protection_vs_'+b]=contrast[os]['delta']<=0 and contrast[os]['ci'][1]<=.02
        for metric,point,upper in ((contact,.01,.05),(stance,.01,.05),('root_endpoint_shift_cm',.05,.10),('object_endpoint_shift_cm',.05,.10)):
            gates[metric+'_protection_vs_'+b]=contrast[metric]['delta']<=point and contrast[metric]['ci'][1]<=upper
    f=feasibility['0.005']['groups']
    gates['domain_ray_protection']=f['G2']['admissible']/f['G2']['n']>=f['G1']['admissible']/f['G1']['n']-.02
    gates['history_contact_and_pairing']=True
    # Predefined complete strata; no group defines the main comparison.
    strata={}
    for label,selector in [('scene',lambda r:r['scene']),('object',lambda r:r['object']),
                           ('source_domain',lambda r:'outside' if r['source_outside'] else 'inside'),
                           ('role',lambda r:'calibration' if r['scene'] in ('004','006','055') else 'verification')]:
        strata[label]={}
        for value in sorted(set(selector(r) for r in ray_rows)):
            strata[label][value]={}
            for scale in rays:
                strata[label][value][scale]={g:average([r['metrics'] for r in ray_rows if selector(r)==value and r['scale']==scale and r['group']==g]) for g in groups[1:]}
    write('strata.json',strata);write('paired.json',paired_results);write('means.json',means);write('feasibility.json',feasibility)
    worst=max(short['G2'],key=lambda t:(short['G2'][t][hs]-short['G0'][t][hs],-int(t)))
    result=dict(schema_version=1,subphase='2.14a',decision='GO' if all(gates.values()) else 'NO-GO',gates=gates,
        tasks=len(episodes),windows=len(window_rows),native_quality_evaluated=False,
        means=means,paired=paired_results,feasibility=feasibility,
        projection=dict(queries=len(projection),source_queries=len(ray_projection),keep_fraction=keep_fraction,
            r_keep_mean=float(np.mean([p['r_keep'] for p in ray_projection if p['r_keep'] is not None])),
            r_keep_median=float(np.median([p['r_keep'] for p in ray_projection if p['r_keep'] is not None])),
            normalized_residual_max=max(p['normalized_residual'] for p in projection),
            jv_after_max=max(p['jv_after'] for p in projection)),
        teacher_forwards=dict(hsi=sum(r['hsi_teacher_calls'] for e in episodes for r in e['records']),
                              hoi=sum(r['hoi_teacher_calls'] for e in episodes for r in e['records'])),
        peak_allocated_bytes=max(r['peak_allocated_bytes'] for e in episodes for r in e['records']),
        bootstrap=dict(replicates=10000,seed=42,percentiles=[2.5,97.5],primary_unit='task',device=device,dtype='float64'),
        selected_failure_task=worst,all_failures_retained=True)
    write('summary.json',result)
    return result


def paired_local_metrics(first, second, device):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from tools.paired_bootstrap import pair_sequence_names, discover_metrics
    names=pair_sequence_names(first,second)
    metrics=discover_metrics(first,second,names)['analyzed']
    index=torch.as_tensor(np.random.default_rng(42).integers(0,len(names),size=(10000,len(names))),device=device)
    delta=torch.tensor([[second[n][k]-first[n][k] for k in metrics] for n in names],dtype=torch.float64,device=device)
    if not torch.isfinite(delta).all():raise ValueError('nonfinite paired metric')
    samples=delta[index].mean(1)
    bounds=torch.quantile(samples,torch.tensor([.025,.975],dtype=torch.float64,device=device),dim=0)
    return {k:dict(delta=float(delta[:,i].mean()),ci=[float(bounds[0,i]),float(bounds[1,i])],n=len(names)) for i,k in enumerate(metrics)}


def foot_quality_gates(contrasts, audit, motions, means):
    """Phase2.16's preregistered HS primary and native noninferiority gates."""
    candidate = 'B2_quality'
    primary = contrasts['task'][candidate+'-B1_no_hsi']['scene_human_penetration_s_mean']
    gates = dict(complete_tasks=True,
        native_hs_gain=primary['delta'] <= -.2052208120905562 and primary['ci'][1] < 0
            and contrasts['scene'][candidate+'-B1_no_hsi']['scene_human_penetration_s_mean']['delta'] <= 0,
        no_invalid_proposals=all(a['invalid_proposals']==0 for a in audit.values()),
        no_nonfinite_steps=all(a['nonfinite_steps']==0 for a in audit.values()),
        no_task_failures=all(m['task_failed']==0 for m in means.values()),
        history_common_contact=all(a['history_common_contact_exact'] and a['final_guards'] for a in audit.values()),
        world_history_continuity=all(m['history_world_max_abs_m']<=1e-5 for m in motions))
    for unit in ('task', 'scene'):
        for baseline in ('B0_hoi', 'B1_no_hsi'):
            c = contrasts[unit][candidate+'-'+baseline]
            gates[f'{unit}_fs_vs_{baseline}'] = c['foot_sliding']['ci'][1] <= .01
            for k in ('contact_percent','completed'):
                gates[f'{unit}_{k}_vs_{baseline}'] = c[k]['ci'][0] >= -.02
            for k in ('xy_points_err','end_obj_trans_err'):
                gates[f'{unit}_{k}_vs_{baseline}'] = c[k]['ci'][1] <= 1.
        c1 = contrasts[unit][candidate+'-B1_no_hsi']
        os = c1['scene_obj_penetration_s_mean']
        gates[unit+'_os_protection_B1'] = os['delta'] <= 0 and os['ci'][1] <= .02
        c0 = contrasts[unit][candidate+'-B0_hoi']
        keys = ('scene_human_penetration_s_mean','scene_obj_penetration_s_mean')
        gates[unit+'_total_scene_gain_B0'] = all(c0[k]['delta']<=0 for k in keys) and any(c0[k]['ci'][1]<0 for k in keys)
    return gates


def summarize_conditional_repair(source_root, output_dir, task_manifest, device='cuda:7',
                                 foot_quality=False):
    """Complete development rollouts, paired by task and by scene, with fixed gates."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    groups = ('B0_hoi', 'B1_no_hsi', 'B2_hsi_repair')
    if foot_quality:
        groups += ('B2_quality',)
    tasks, records, motions, scene_by_task = {g:{} for g in groups}, {g:[] for g in groups}, [], {}
    def write(name, value):
        with (directory/name).open('x') as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
    def average(rows):
        return {k:float(np.mean([r[k] for r in rows])) for k in rows[0]}
    selection = json.loads(Path(task_manifest).read_text())['tasks']
    expected = {'%03d'%r['canonical_ordinal'] for r in selection}
    native_unavailable = set()
    for g in groups:
        paths = sorted(Path(source_root).glob(f'{g}-shard*/episode-audit-*.json'))
        for path in paths:
            ep = json.loads(path.read_text())
            identity = ep['metrics']; task = '%03d' % identity['canonical_ordinal']
            saved_path = path.with_name(path.name.replace('episode-audit-', 'episode-motion-')).with_suffix('.pt')
            saved = torch.load(saved_path, map_location='cpu', weights_only=False)
            ep['records'] = ep['sampler_audit']['composition']['scene_edit']['records'][-len(saved['windows']):]
            if task in tasks[g]:
                raise ValueError('duplicate conditional-repair task')
            scene_by_task[task] = identity['scene_name']
            native_unavailable.update(k for k,v in ep['metrics'].items() if v is None)
            metrics = {k:float(v) for k,v in ep['metrics'].items()
                       if isinstance(v,(float,int,bool)) and k not in ('canonical_ordinal','test_idx')}
            proxy = average([{k:float(v[0]) for k,v in r['final_metrics'].items()} for r in ep['records']])
            metrics.update({'proxy_'+k:v for k,v in proxy.items()})
            metrics['generation_seconds'] = ep['generation_seconds']
            metrics['mean_repair_rms_mm'] = float(np.mean([r['repair_rms_mm'] for r in ep['records']]))
            metrics['invalid_proposal_fraction'] = float(np.mean([r['invalid_proposal'] for r in ep['records']]))
            for r in ep['records']:
                records[g].append(dict(r,task=task,scene=identity['scene_name'],object=identity['object_name']))
            joints = saved['evaluated_joints_world'].to(device)
            velocity = (joints[1:]-joints[:-1]).norm(dim=-1)
            root_velocity = velocity[:,0]
            metrics.update(root_path_cm=float(root_velocity.sum()*100),
                mean_joint_frame_displacement_cm=float(velocity.mean()*100),
                near_stationary_root_fraction=float((root_velocity<.001).float().mean()))
            seams = []
            for previous,current in zip(saved['windows'],saved['windows'][1:]):
                a=previous['points_world'].to(device).reshape(1,16,28,3)
                b=current['points_world'].to(device).reshape(1,16,28,3)
                seams.append(float((a[:,-2:]-b[:,:2]).abs().max()))
            metrics['history_world_max_abs_m'] = max(seams,default=0.)
            tasks[g][task] = metrics
            motions.append(dict(group=g,task=task,scene=identity['scene_name'],object=identity['object_name'],
                frame_count=len(joints),history_world_max_abs_m=metrics['history_world_max_abs_m'],
                generation_seconds=ep['generation_seconds']))
        if set(tasks[g]) != expected:
            raise ValueError(f'incomplete registered task coverage: {g}, {len(tasks[g])}')
    scene = {g:{s:average([r for n,r in tasks[g].items() if scene_by_task[n]==s])
                for s in sorted(set(scene_by_task.values()))} for g in groups}
    means = {g:average(list(tasks[g].values())) for g in groups}
    contrasts = {}
    for unit,data in (('task',tasks),('scene',scene)):
        contrasts[unit]={f'{b}-{a}':paired_local_metrics(data[a],data[b],device)
                        for i,a in enumerate(groups) for b in groups[i+1:]}
    repair=records[groups[2]]
    audit={g:dict(windows=len(records[g]),
        modified_windows=sum(r['modified'] for r in records[g]),
        fallback_windows=sum(r['fallback'] for r in records[g]),
        invalid_proposals=sum(r['invalid_proposal'] for r in records[g]),
        hsi_calls=sum(r['hsi_calls'] for r in records[g]),
        accepted_fit_steps=sum(sum(sum(t['accepted']) for t in s.get('fit',[])) for r in records[g] for s in r['steps']),
        nonfinite_steps=sum(s['reason'].startswith('nonfinite') for r in records[g] for s in r['steps']),
        history_common_contact_exact=all(r['history_exact'] and r['common_exact'] and r['contact_exact'] for r in records[g]),
        final_guards=all(all(r['final_guards'].values()) for r in records[g]),
        timing_mean_seconds={k:float(np.mean([r[k] for r in records[g]]))
                             for k in ('geometry_seconds','hsi_seconds','fit_seconds','seconds')},
        peak_allocated_bytes=max(r['peak_allocated_bytes'] or 0 for r in records[g])) for g in groups}
    gates=dict(complete_tasks=True, no_invalid_proposals=all(a['invalid_proposals']==0 for a in audit.values()),
        history_common_contact=all(a['history_common_contact_exact'] and a['final_guards'] for a in audit.values()),
        world_history_continuity=all(m['history_world_max_abs_m']<=1e-5 for m in motions),
        repair_occurs=sum(r['modified'] for r in repair)/len(repair)>=.5,
        fallback_fraction=sum(r['fallback'] for r in repair)/len(repair)<=.5)
    primary=contrasts['task']['B2_hsi_repair-B1_no_hsi']['foot_sliding']
    gates['native_fs_gain']=primary['delta']<=-.01 and primary['ci'][1]<0 and contrasts['scene']['B2_hsi_repair-B1_no_hsi']['foot_sliding']['delta']<=0
    for unit in ('task','scene'):
        for b in groups[:2]:
            contrast=contrasts[unit]['B2_hsi_repair-'+b]
            for k in ('contact_percent','completed'):
                gates[f'{unit}_{k}_vs_{b}']=contrast[k]['ci'][0]>=-.02
            for k in ('xy_points_err','end_obj_trans_err'):
                gates[f'{unit}_{k}_vs_{b}']=contrast[k]['ci'][1]<=1.
        c1=contrasts[unit]['B2_hsi_repair-B1_no_hsi'];c0=contrasts[unit]['B2_hsi_repair-B0_hoi']
        scene_keys=('scene_human_penetration_s_mean','scene_obj_penetration_s_mean')
        gates[unit+'_scene_protection_B1']=all(c1[k]['delta']<=0 and c1[k]['ci'][1]<=.02 for k in scene_keys)
        gates[unit+'_total_scene_gain_B0']=all(c0[k]['delta']<=0 for k in scene_keys) and any(c0[k]['ci'][1]<0 for k in scene_keys)
    result=dict(phase='2.15',tasks=len(expected),scenes=len(scene[groups[0]]),groups=groups,means=means,audit=audit,gates=gates,
        numerical_gate_passed=all(gates.values()),
        decision='REQUIRES_FULL_MOTION_REVIEW' if all(gates.values()) else 'NO-GO',
        full469_started=False,native_unavailable=sorted(native_unavailable),
        metric_scope='complete native HOSI metrics; window FK/voxel proxies separately named',
        bootstrap=dict(replicates=10000,seed=42,confidence=.95,units=['task','scene']))
    if foot_quality:
        gates = foot_quality_gates(contrasts, audit, motions, means)
        gates['complete_native_metrics'] = not native_unavailable
        result.update(phase='2.16', gates=gates, numerical_gate_passed=all(gates.values()),
            decision='REQUIRES_FULL_MOTION_REVIEW' if all(gates.values()) else 'NO-GO',
            primary_metric='scene_human_penetration_s_mean', minimum_absolute_hs_gain=.2052208120905562,
            foot_energy_epsilon_m2=1e-12, test_set_development=True)
        diagnosis = {}
        for g in groups:
            trials = [t for r in records[g] for s in r['steps'] for f in s.get('fit',[]) for t in f['trials']]
            rejected = [t for t in trials if not all(t['admissible'])]
            reasons = {k:sum(not all(t['guards'][k]) for t in rejected)
                       for k in ('domain','contact','human_scene','stance','feet','common','finite')}
            rms = np.array([r['repair_rms_mm'] for r in records[g]])
            diagnosis[g] = dict(trials=len(trials), rejected_trials=len(rejected),
                rejection_reasons_cooccurring=reasons,
                rejection_single_reason={k:sum(not all(t['guards'][k]) and
                    sum(not all(v) for v in t['guards'].values())==1 for t in rejected) for k in reasons},
                search_exhausted=sum(f['reason'].count('search_exhausted') for r in records[g]
                    for s in r['steps'] for f in s.get('fit',[])),
                rms_mm=dict(mean=float(rms.mean()),median=float(np.median(rms)),
                    p95=float(np.quantile(rms,.95)),max=float(rms.max())),
                legacy_disallowed_but_new_admissible=sum(all(t['admissible']) and
                    not all(t['foot']['legacy_pass']) for t in trials if 'foot' in t),
                legacy_disallowed_but_new_accepted=sum(any(t['accepted']) and
                    not all(t['foot']['legacy_pass']) for t in trials if 'foot' in t),
                empty_support_windows=sum(r['foot']['active_count']==[0] for r in records[g] if 'foot' in r))
        object_by_task = {'%03d'%r['canonical_ordinal']:r['object_name'] for r in selection}
        objects = {g:{obj:average([row for task,row in tasks[g].items() if object_by_task[task]==obj])
                    for obj in sorted(set(object_by_task.values()))} for g in groups}
        # Task rows are the statistical units, also within each object stratum.
        write('object_means.json', objects)
        write('repair_diagnosis.json', diagnosis)
    for name,data in (('task_metrics.json',tasks),('scene_metrics.json',scene),('paired.json',contrasts),
                      ('window_records.json',records),('motion_audit.json',motions),('summary.json',result)):
        write(name,data)
    for g in groups:
        for unit,data in (('task',tasks),('scene',scene)):
            sub=directory/(g+'_'+unit);sub.mkdir()
            write(str(sub.relative_to(directory)/'per_sequence_metrics.json'),data[g])
    return result


def recover_temporal_window(dataset, saved, snapshot, world_window, task, object_points, device):
    """Recover omitted context from redundant saved world/local rigid transforms.

    Saved global rotations obey R_world = mat.R @ R_local. Object rotations obey
    R_object_world = prefix @ R_relative @ R_reference. Invert those equations;
    never infer a heading from coordinate names or another arm's trajectory.
    """
    from pytorch3d import transforms
    from .relational import RelationalGeometry
    from .conditional_repair import ConstrainedPoseFit, locked_encode
    c = {k:v.to(device) if torch.is_tensor(v) else v for k,v in snapshot.items()}
    w = {k:v.to(device) if torch.is_tensor(v) else v for k,v in world_window.items()}
    local = transforms.rotation_6d_to_matrix(c['edited'][...,84:216].reshape(1,16,22,6))
    world = transforms.rotation_6d_to_matrix(w['global_rot_6d'].reshape(1,16,22,6))
    mat = torch.eye(4,device=device)[None]
    mat[:,:3,:3] = world[:,0,0] @ local[:,0,0].transpose(-1,-2)
    position = dataset.denormalize_torch(c['edited'][...,:84]).reshape(1,16,28,3)
    mat[:,:3,3] = (w['points_world'].reshape(1,16,28,3)[:,0,0]
                   - (mat[:,:3,:3] @ position[:,0,0,:,None]).squeeze(-1))
    index = task['data_idx']
    sequence = dataset.ori_sequence_idx[index]
    offsets = torch.tensor(dataset.rest_human_offsets[sequence].astype(np.float32),device=device)
    start = int(dataset.start_ind[index])
    reference = torch.tensor(dataset.object_rot_mat[start],device=device,dtype=torch.float32)[None]
    relative = c['edited'][0,0,219:228].reshape(3,3)
    prefix = (w['object_rotation_world'][0,0].reshape(3,3)
              @ torch.linalg.inv(reference[0]) @ torch.linalg.inv(relative))
    context = dict(mat=mat,obj_rot_mat_ref=reference,obj_rot_mat_prefix=prefix[None],
                   scene_flag=torch.tensor([0],device=device))
    geometry = RelationalGeometry(c['proposal'],dataset,offsets,context,object_points)
    source = RelationalGeometry(c['raw_source'],dataset,offsets,context,object_points)
    fitter = ConstrainedPoseFit(geometry,source,context['scene_flag'],foot_guard_mode='quality')
    state = geometry.decode(c['repair_parameters'])
    audit = dict(fk_max_abs_m=float((state['human']-c['final_fk']).abs().max()),
        proposal_fk_max_abs_m=float((fitter.proposal['human']-c['proposal_fk']).abs().max()),
        source_anchor_max_abs_m=float((fitter.objective.hand_anchor-c['source_anchor']).abs().max()),
        object_world_max_abs_m=float((state['object_translation_world']-w['object_translation_world']).abs().max()),
        object_rotation_max_abs=float((state['object_rotation_world']-w['object_rotation_world'].reshape(1,16,3,3)).abs().max()),
        output_reencode_max_abs=float((locked_encode(geometry,c['repair_parameters'])-c['edited']).abs().max()),
        contact_exact=torch.equal(fitter.objective.contact,c['contact_mask']),
        stance_exact=('repair_stance_mask' not in c or torch.equal(fitter.protection.stance,c['repair_stance_mask'])))
    if any(audit[k]>1e-5 for k in audit if k.endswith(('_m','_abs'))) or not audit['contact_exact'] or not audit['stance_exact']:
        raise AssertionError('Temporal cache recovery mismatch: '+str(audit))
    # Use the actual saved source anchor, retaining its original finite precision.
    fitter.objective.hand_anchor = c['source_anchor'].clone()
    fitter.contact_distance = fitter.objective.contact_residual(fitter.proposal).norm(dim=-1)
    fitter.contact_limit = fitter.contact_distance.clamp_min(.005)+.001
    return fitter,c,audit,context,offsets


def temporal_window_metrics(fitter, parameters, dt=.1):
    """Independent world/contact proxies, explicitly separate from native15."""
    from .temporal_preservation import physical_derivatives, masked_square_mean
    state=fitter.geometry.decode(parameters)
    human=state['human']; mask=fitter.protection.stance
    speed=(human[:,2:,(7,8,10,11),:]-human[:,1:-1,(7,8,10,11),:])[...,(0,2)].norm(dim=-1)/dt
    contact=fitter.objective.contact
    distance=fitter.objective.contact_residual(state).norm(dim=-1)
    _,metrics=fitter.protection.evaluate(state)
    relative=(state['object_rotation_world'].transpose(-1,-2)[...,None,:,:] @
              (human[...,22:24,:]-state['object_translation_world'][...,None,:])[...,None]).squeeze(-1)
    source=fitter.objective.hand_anchor
    # Derivatives cross only continuous active grasp segments, including history.
    active=fitter.geometry.base[...,228:230]>.95
    vmask=active[:,2:] & active[:,1:-1]
    amask=vmask & active[:,:-2]
    v=((relative[:,2:]-relative[:,1:-1])-(source[:,2:]-source[:,1:-1]))/dt
    a=((relative[:,2:]-2*relative[:,1:-1]+relative[:,:-2])-(source[:,2:]-2*source[:,1:-1]+source[:,:-2]))/dt**2
    def masked(value,m):return float(torch.where(m,value,torch.zeros_like(value)).sum()/m.sum().clamp_min(1))
    return dict(support_speed_m_per_s=masked(speed,mask),support_count=int(mask.sum()),
        contact_anchor_cm=masked(distance,contact)*100,contact_count=int(contact.sum()),
        human_scene_residual_cm=float(metrics['human_scene_residual_cm']),
        human_occupied_fraction=float(metrics['human_occupied_fraction']),
        toe_height_above_fixed_floor_cm=float((human[:,2:,(10,11),1]-fitter.protection.floor_height[:,None,None]).mean()*100),
        joint_speed_m_per_s=float((human[:,2:,:22]-human[:,1:-1,:22]).norm(dim=-1).mean()/dt),
        root_path_m=float((human[:,2:,0]-human[:,1:-1,0]).norm(dim=-1).sum()),
        near_stationary_fraction=float(((human[:,2:,:22]-human[:,1:-1,:22]).norm(dim=-1)/dt<.01).float().mean()),
        hand_object_velocity_error_m2_per_s2=masked(v.square().mean(-1),vmask),
        hand_object_acceleration_error_m2_per_s4=masked(a.square().mean(-1),amask),
        hand_velocity_count=int(vmask.sum()),hand_acceleration_count=int(amask.sum()))


def replay_temporal_scene(source_root, output_dir, resolved_config, task_manifest, scene_name, device):
    """Fixed cached windows, no expert forward and no stitched native score."""
    import subprocess
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from .temporal_preservation import TemporalPreservation
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('Reportable Temporal replay requires clean worktree')
    config=OmegaConf.load(resolved_config)
    options=OmegaConf.to_container(config.sampler.pelvis.scene_editor.temporal,resolve=True)
    module=TemporalPreservation(**options)
    config.dataset.device=device;config.dataset.vis=True;config.dataset.load_object_payload=False
    config.dataset.test_scene_name=scene_name
    dataset=InfBaGelDataset(**config.dataset)
    root=Path(source_root);out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    selection=json.loads(Path(task_manifest).read_text())
    tasks=[t for t in selection['tasks'] if t['scene_name']==scene_name]
    native=json.loads((Path(config.dataset.folder).parent/'hosi_test/data'/ (scene_name+'.json')).read_text())
    records=[]
    with torch.no_grad():
        for arm in ('B1_no_hsi','B2_quality'):
            for task in tasks:
                ordinal=task['canonical_ordinal']
                path=next(root.glob(f'{arm}-*/episode-motion-{ordinal:03d}.pt'))
                saved=torch.load(path,map_location='cpu',weights_only=False)
                points=dataset.obj_rest_verts[task['object_name']].to(device)
                points=points[torch.linspace(0,len(points)-1,128,device=device).long()][None]
                motions=[]
                for snapshot,window in zip(saved['corrections'],saved['windows']):
                    fitter,c,audit,context,offsets=recover_temporal_window(dataset,saved,snapshot,window,native[task['test_idx']],points,device)
                    timestamps=torch.arange(16,device=device,dtype=torch.float64)[None]*.1
                    valid=torch.ones(1,16,device=device,dtype=torch.bool);links=valid[:,1:].clone()
                    incoming=fitter.geometry.decode(c['repair_parameters'])
                    before=temporal_window_metrics(fitter,c['repair_parameters'])
                    output,parameters,trace=module.apply(fitter,c['edited'],c['repair_parameters'],timestamps,valid,links)
                    after=temporal_window_metrics(fitter,parameters)
                    state=fitter.geometry.decode(parameters)
                    final_guards={k:bool(v.all()) for k,v in fitter.guards(parameters).items()}
                    record=dict(arm=arm,task=ordinal,scene=scene_name,object=task['object_name'],window=c['window'],
                        recovery=audit,before=before,after=after,temporal=trace,final_guards=final_guards,
                        history_exact=torch.equal(output[:,:2],c['edited'][:,:2]),
                        common_exact=all(torch.equal(output[...,a:b],c['edited'][...,a:b]) for a,b in ((0,3),(84,90),(216,232))),
                        temporal_rms_mm=float((state['human'][:,2:]-incoming['human'][:,2:]).square().mean().sqrt()*1000))
                    records.append(record)
                    motions.append(dict(window=c['window'],raw_source=c['raw_source'].cpu(),proposal=c['proposal'].cpu(),
                        incoming=c['edited'].cpu(),output=output.cpu(),incoming_parameters=c['repair_parameters'].cpu(),parameters=parameters.cpu(),
                        source_human=fitter.objective.anchor['human'].cpu(),incoming_human=incoming['human'].cpu(),output_human=state['human'].cpu(),
                        object_translation_world=state['object_translation_world'].cpu(),object_rotation_world=state['object_rotation_world'].cpu(),
                        timestamps_seconds=timestamps.cpu(),valid_frames=valid.cpu(),valid_links=links.cpu(),
                        source_anchor=fitter.objective.hand_anchor.cpu(),stance_mask=fitter.protection.stance.cpu(),
                        context={k:v.cpu() for k,v in context.items()},rest_offsets=offsets.cpu()))
                    with (out/'per_window.jsonl').open('a') as handle:handle.write(json.dumps(record,allow_nan=False)+'\n')
                torch.save(dict(arm=arm,task=task,mode='independent_window_replay',windows=motions),out/f'{arm}-{ordinal:03d}.pt')
                print(arm,ordinal,len(motions),'complete',flush=True)
    (out/'summary.json').write_text(json.dumps(dict(windows=len(records),tasks=len(tasks),expert_calls=0),indent=2))
    return records


def summarize_temporal_replay(run_root, task_manifest, device='cuda:7'):
    """Task-paired A2 decision; cached independent windows never native episodes."""
    root=Path(run_root)
    records=[json.loads(line) for p in sorted(root.glob('scene-*/per_window.jsonl')) for line in p.read_text().splitlines()]
    tasks=json.loads(Path(task_manifest).read_text())['tasks']
    arms=('B1_no_hsi','B2_quality');tables={};scene_tables={};paired={};scene_paired={};means={}
    for arm in arms:
        tables[arm]={};scene_tables[arm]={};means[arm]={}
        for stage in ('before','after'):
            table={}
            for task in tasks:
                subset=[r for r in records if r['arm']==arm and r['task']==task['canonical_ordinal']]
                if not subset:raise ValueError('Missing preregistered Temporal task')
                table[str(task['canonical_ordinal'])]={k:float(np.mean([r[stage][k] for r in subset])) for k in subset[0][stage]}
                for k in ('velocity_error_m2_per_s2','acceleration_error_m2_per_s4','velocity_normalized','acceleration_normalized'):
                    table[str(task['canonical_ordinal'])][k]=float(np.mean([r['temporal'][stage][k][0] for r in subset]))
                table[str(task['canonical_ordinal'])]['source_normalized_sum']=sum(table[str(task['canonical_ordinal'])][k] for k in ('velocity_normalized','acceleration_normalized'))
            tables[arm][stage]=table
            scene_tables[arm][stage]={scene:{k:float(np.mean([table[str(t['canonical_ordinal'])][k] for t in tasks if t['scene_name']==scene])) for k in next(iter(table.values()))} for scene in sorted({t['scene_name'] for t in tasks})}
            means[arm][stage]={k:float(np.mean([t[k] for t in table.values()])) for k in next(iter(table.values()))}
        paired[arm]=paired_local_metrics(tables[arm]['before'],tables[arm]['after'],device)
        scene_paired[arm]=paired_local_metrics(scene_tables[arm]['before'],scene_tables[arm]['after'],device)
    independent={}
    for arm in arms:
        independent[arm]={}
        other=arms[1] if arm==arms[0] else arms[0]
        for key in ('support_speed_m_per_s','contact_anchor_cm'):
            independent[arm][key]=(means[arm]['before'][key]>0 and
                means[arm]['after'][key]<=.95*means[arm]['before'][key] and
                paired[arm][key]['ci'][1]<0 and scene_paired[arm][key]['delta']<=0 and
                paired[other][key]['ci'][0]<=0)
    gates=dict(coverage=len(records)==248 and all(sum(r['arm']==arm for r in records)==124 for arm in arms),
        hard_guards=all(all(r['final_guards'].values()) for r in records),
        fixed_history_common=all(r['history_exact'] and r['common_exact'] for r in records),
        finite_execution=all(r['temporal']['reason']=='optimized' for r in records),
        source_dynamics=all(means[a]['after']['source_normalized_sum']<means[a]['before']['source_normalized_sum'] for a in arms),
        nonzero_changes=all(any(r['temporal']['changed'] for r in records if r['arm']==a) for a in arms),
        independent_quality=any(v for arm in independent.values() for v in arm.values()))
    for arm in arms:
        b,a=means[arm]['before'],means[arm]['after']
        gates[arm+'_scene_retained']=a['human_scene_residual_cm']<=b['human_scene_residual_cm']+.01 and a['human_occupied_fraction']<=b['human_occupied_fraction']
        gates[arm+'_no_toe_lift']=a['toe_height_above_fixed_floor_cm']<=b['toe_height_above_fixed_floor_cm']+.1
        gates[arm+'_motion_retained']=a['joint_speed_m_per_s']>=.95*b['joint_speed_m_per_s']
    counts={arm:dict(windows=sum(r['arm']==arm for r in records),
        changed=sum(r['temporal']['changed'] for r in records if r['arm']==arm),
        accepted_steps=sum(sum(s['accepted']) for r in records if r['arm']==arm for s in r['temporal']['steps']),
        mean_rms_mm=float(np.mean([r['temporal_rms_mm'] for r in records if r['arm']==arm])),
        mean_seconds=float(np.mean([r['temporal']['seconds'] for r in records if r['arm']==arm]))) for arm in arms}
    result=dict(phase='2.17',mode='independent_window_replay',test_set_development=True,
        tasks=28,scenes=4,windows=248,expert_calls=0,native_rollouts_started=0,
        decision='NUMERICAL-PASS-VISUAL-PENDING' if all(gates.values()) else 'NO-GO',
        visual_review_pending=True,gates=gates,independent_quality=independent,means=means,
        counts=counts,paired_task=paired,paired_scene=scene_paired,
        limitation='Window replay is not a causally consistent task rollout; no native15 or completion claim.')
    output=root/'analysis';output.mkdir(exist_ok=False)
    for name,value in [('summary.json',result),('task_metrics.json',tables),('scene_metrics.json',scene_tables)]:
        (output/name).write_text(json.dumps(value,indent=2,allow_nan=False))
    return result
