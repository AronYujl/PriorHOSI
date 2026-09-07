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


def summarize_conditional_repair(source_root, output_dir, task_manifest, device='cuda:7'):
    """Complete development rollouts, paired by task and by scene, with fixed gates."""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    groups = ('B0_hoi', 'B1_no_hsi', 'B2_hsi_repair')
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
                        for a,b in ((groups[0],groups[1]),(groups[0],groups[2]),(groups[1],groups[2]))}
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
    for name,data in (('task_metrics.json',tasks),('scene_metrics.json',scene),('paired.json',contrasts),
                      ('window_records.json',records),('motion_audit.json',motions),('summary.json',result)):
        write(name,data)
    for g in groups:
        for unit,data in (('task',tasks),('scene',scene)):
            sub=directory/(g+'_'+unit);sub.mkdir()
            write(str(sub.relative_to(directory)/'per_sequence_metrics.json'),data[g])
    return result
