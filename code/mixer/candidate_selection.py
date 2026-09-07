"""Read-only scene compatibility and whole-episode selection on a fixed pool."""
import inspect
import json
import time
from pathlib import Path

import torch

from .composed_sampler import HOSIComposedSampler
from .input_views import KnownEmptyObjectView, masked_object_arguments


def move_tree(value, device):
    if torch.is_tensor(value):
        return value.detach().to(device).clone()
    if isinstance(value, dict):
        return {k: move_tree(v, device) for k, v in value.items()}
    return value


class CandidatePoolSampler(HOSIComposedSampler):
    """B1 sampling with a scoped generation seed and optional context recording.

    The evaluator retains seed42 for task setup and metric subsampling. Offset0
    follows the original sampler directly. No scoring occurs during generation.
    """
    def __init__(self, candidate_seed_offset=0, record_replay_context=False, **kwargs):
        super().__init__(**kwargs)
        self.candidate_seed_offset = int(candidate_seed_offset)
        self.record_replay_context = record_replay_context

    def _hsi_context(self, *args, **kwargs):
        context = super()._hsi_context(*args, **kwargs)
        self._window_context = context
        return context

    def p_sample_loop(self, *args, **kwargs):
        if self.candidate_seed_offset:
            device = args[0].device if args else kwargs['fixed_points'].device
            with torch.random.fork_rng(devices=[device.index] if device.type == 'cuda' else []):
                # Only the CPU initial seed is read to seed the private HOI stream.
                torch.set_rng_state(torch.Generator().manual_seed(
                    torch.initial_seed() + self.candidate_seed_offset).get_state())
                result = super().p_sample_loop(*args, **kwargs)
        else:
            result = super().p_sample_loop(*args, **kwargs)
        if self.record_replay_context:
            bound = inspect.signature(HOSIComposedSampler.p_sample_loop).bind(self, *args, **kwargs)
            context = {k: v for k, v in self._window_context.items()
                       if k not in ('obj_rest_verts', 'obj_vert_normals', 'static_occ_cache')}
            self.scene_editor.motion_records[-1].update(
                replay_context=move_tree(context, 'cpu'),
                rest_offsets=move_tree(bound.arguments['human_dict']['rest_human_offsets'], 'cpu'))
        return result

    def audit_dict(self):
        result = super().audit_dict()
        result['candidate_pool'] = dict(seed_offset=self.candidate_seed_offset,
                                       record_replay_context=self.record_replay_context)
        return result


def prediction_frames(window_index, length, valid_frames=None):
    """Only newly predicted frames; history overlap contributes no second score."""
    valid = torch.ones(length, dtype=torch.bool) if valid_frames is None else valid_frames.clone().bool()
    valid[:2] = False
    local = torch.arange(length)[valid.cpu()]
    if not len(local):
        raise ValueError('candidate has zero valid prediction frames')
    return valid, local + 14 * window_index


def mismatch_dynamic(common):
    """One fixed spatial correspondence control, retaining native occupied labels."""
    changed = list(common)
    changed[15] = torch.cat((common[15][:1], torch.roll(common[15][1:], 16, dims=2)), 0)
    return tuple(changed)


def reconstruction_error(prediction, clean, valid):
    active = prediction[:, valid, :84]
    if not torch.isfinite(active).all():
        raise FloatingPointError('nonfinite HSI position prediction')
    return float((active.double() - clean[:, valid, :84].double()).square().mean())


@torch.no_grad()
def score_window(sampler, clean, context, task, window_index, protocol, valid_frames=None):
    """Raw full/static heads, common private noise, clean scene queries, no edit."""
    valid, global_frames = prediction_frames(window_index, clean.shape[1], valid_frames)
    valid = valid.to(clean.device)
    if not torch.isfinite(clean).all():
        raise FloatingPointError('nonfinite candidate motion')
    diffusion = sampler.hsi_sampler
    devices = [clean.device.index] if clean.is_cuda else []
    scene_seed = 42 + 800000000 + task * 1000 + window_index
    # Query once per window: native geometry sampling is independent of level.
    with torch.random.fork_rng(devices=devices):
        torch.set_rng_state(torch.Generator().manual_seed(scene_seed).get_state())
        t = torch.full((len(clean),), protocol['levels'][0], device=clean.device, dtype=torch.long)
        common = sampler._hsi_model_arguments(clean, clean, t, context)
    common = masked_object_arguments(common)
    wrong = mismatch_dynamic(common)
    observations = []
    for repeat in range(protocol['repetitions']):
        for level in protocol['levels']:
            seed = 42 + 700000000 + task * 100000 + window_index * 1000 + repeat * 100 + level
            generator = torch.Generator(device=clean.device).manual_seed(seed)
            noise = torch.randn(clean.shape, device=clean.device, dtype=clean.dtype, generator=generator)
            t = torch.full((len(clean),), level, device=clean.device, dtype=torch.long)
            noisy = diffusion.q_sample(clean, t, noise)
            noisy[:, :2] = clean[:, :2]
            empty = KnownEmptyObjectView()
            empty.begin_window(clean, seed)
            view = empty.for_step(noisy, level)
            arguments = list(common); arguments[1] = t
            mismatched = list(wrong); mismatched[1] = t
            full, static = sampler._hsi_predict_pair(view, tuple(arguments))
            mismatch = diffusion.student_model(view, *mismatched, is_sample=True)
            values = {name: reconstruction_error(pred, clean, valid)
                      for name, pred in (('full', full), ('static', static), ('mismatch', mismatch))}
            values.update(level=level, repetition=repeat, noise_seed=seed,
                          rotation_l1_full=float((full[:, valid, 84:216]-clean[:, valid, 84:216]).abs().mean()))
            observations.append(values)
    means = {name: sum(r[name] for r in observations)/len(observations)
             for name in ('full', 'static', 'mismatch')}
    return dict(**means, dp_difference=means['static']-means['full'],
                observations=observations, global_frames=global_frames.tolist(),
                coordinates=len(global_frames)*84, scene_seed=scene_seed, hsi_calls=3*len(observations))


def aggregate_scores(windows):
    frames = [f for w in windows for f in w['global_frames']]
    if len(frames) != len(set(frames)):
        raise ValueError('scoring windows contain repeated global prediction frames')
    count = sum(w['coordinates'] for w in windows)
    if count == 0:
        raise ValueError('candidate has zero scored coordinates')
    return {name: sum(w[name]*w['coordinates'] for w in windows)/count
            for name in ('full', 'static', 'mismatch')}


def select_candidates(features, scores, protocol):
    """Consumes inference features only; native reports are outside this API."""
    reference = features[0]
    accepted, reasons = [], {}
    for index, f in sorted(features.items()):
        rejected = []
        if not f['valid']:
            rejected.append('invalid_motion_or_editor_guard')
        for key in ('root_endpoint_m', 'object_endpoint_m'):
            limit = max(reference[key], .10)
            if f[key] > limit or (reference[key] < .10 and f[key] >= .10):
                rejected.append(key)
        if f['contact_distance_m'] > reference['contact_distance_m']+.01:
            rejected.append('contact_distance')
        if f['contact_fraction'] < reference['contact_fraction']-.05:
            rejected.append('contact_fraction')
        if f['support_speed_m_per_s'] > reference['support_speed_m_per_s']+.01:
            rejected.append('support_speed')
        reasons[index] = rejected
        if not rejected:
            accepted.append(index)
    if not accepted:
        return dict(acceptable=[],budget_set=[],rejections=reasons,selected=dict(C0=0,CG=0,CS=0,CF=0,CM=0),
                    fallback='no_acceptable_candidate',budget=None)
    geometry = min(accepted, key=lambda i: (features[i]['energy'], i))
    best = features[geometry]['energy']
    budget = max(protocol['budget_relative']*best, protocol['budget_absolute'])
    eligible = [i for i in accepted if features[i]['energy'] <= best+budget]
    selected = dict(C0=0, CG=geometry)
    for name, score in (('CS','static'),('CF','full'),('CM','mismatch')):
        # A technical score failure is a failed experiment, not a preference.
        if not all(torch.isfinite(torch.tensor(scores[i][score])) for i in eligible):
            raise FloatingPointError('nonfinite score in selection')
        selected[name] = min(eligible, key=lambda i: (scores[i][score], i))
    return dict(acceptable=accepted,budget_set=eligible,rejections=reasons,
                selected=selected,fallback=None,budget=budget)


class _CaptureConditions:
    """Reuse native sample_step's condition construction without a reverse chain."""
    def __init__(self, sampler, clean):
        self.sampler, self.dataset, self.clean = sampler, sampler.dataset, clean

    def p_sample_loop(self, *args, **kwargs):
        bound = inspect.signature(HOSIComposedSampler.p_sample_loop).bind(self.sampler,*args,**kwargs)
        bound.apply_defaults()
        values = bound.arguments
        names = inspect.signature(HOSIComposedSampler._hsi_context).parameters
        self.context = self.sampler._hsi_context(**{k: (len(self.clean) if k=='batch' else values[k])
                                                  for k in names if k!='self'})
        return [self.clean], None


@torch.no_grad()
def recover_score_context(sampler, cfg, saved, index, task, data, trajectory, geometry_context, offsets):
    """Recover absent legacy conditions through the unchanged native entrypoint."""
    from test_infbagel_hosi import get_guidance_from_json, sample_step
    clean = saved['corrections'][index]['edited'].to(cfg.device)
    capture = _CaptureConditions(sampler,clean)
    cond = get_guidance_from_json(cfg,task)
    cond['text_emb'] = data['text_clip_embedding'].to(cfg.device)[None]
    cond['raw_text'] = sampler.dataset.text[task['data_idx']][0]
    pi = torch.tensor([index*42], device=cfg.device, dtype=torch.long)
    length = torch.tensor([len(saved['windows'])*42+6],device=cfg.device,dtype=torch.long)
    output = sample_step(cfg,index,geometry_context['mat'],clean[:,:2],capture,cond,trajectory,
        pi,pi+48,length,data['obj_bps_data'].to(cfg.device)[None],None,
        sampler.dataset.obj_rest_verts,{}, {0:data['seq_name']},geometry_context['obj_rot_mat_ref'],
        {'rest_human_offsets': offsets.clone()},geometry_context['obj_rot_mat_prefix'])
    world = saved['windows'][index]['points_world'].to(cfg.device)
    error = float((output['points_orig']-world).abs().max())
    if error > 1e-5:
        raise AssertionError('native context reconstruction world mismatch: '+str(error))
    return capture.context, error


def retained_endpoint_features(human_world, object_world, task, retained_frames):
    """Native endpoint definitions on the actual emitted, interpolated tracks."""
    if not 0 < retained_frames <= min(len(human_world), len(object_world)):
        raise ValueError('retained frame count is outside the output tracks')
    root = human_world[retained_frames - 1, 0].clone()
    root[1] = 0
    obj = object_world[retained_frames - 1]
    root_error = (root - root.new_tensor(task['pelvis_goal'])).norm()
    object_error = (obj - obj.new_tensor(task['object_goal'])).norm()
    return dict(root_endpoint_m=float(root_error), object_endpoint_m=float(object_error),
                completed=bool(float(root_error) * 100 < 10. and float(object_error) * 100 < 10.),
                retained_frames=retained_frames, endpoint_frame=retained_frames - 1)


def saved_endpoint_features(saved, task, device):
    """Retained SMPL-X joints and native object interpolation, no metric input."""
    from utils import interp_object
    translated, _ = interp_object(saved['stitched']['object_translation_world'],
                                 saved['stitched']['object_rotation_world'], saved['interp_s'])
    human = saved['evaluated_joints_world'].to(device)
    obj = torch.as_tensor(translated, device=device, dtype=torch.float32)
    return retained_endpoint_features(human, obj, task, len(human))


@torch.no_grad()
def candidate_features(dataset, saved, contexts, offsets, task, records,
                       endpoint_mode='legacy_XZ'):
    from .relational import RelationalGeometry
    from .relational import source_floor_height
    points = dataset.obj_rest_verts[saved['object_name']].to(contexts[0]['mat'].device)
    points = points[torch.linspace(0,len(points)-1,128,device=points.device).long()][None]
    energies, contacts, speeds, all_humans = [], [], [], []
    valid = True
    for c,ctx,rest in zip(saved['corrections'],contexts,offsets):
        clean = c['edited'].to(points.device)
        geometry = RelationalGeometry(clean,dataset,rest,ctx,points)
        state = geometry.decode(clean.new_zeros(*clean.shape[:2],geometry.dimension))
        human,surface=state['human'],state['object_surface']
        query=torch.cat((human[:,2:],surface[:,2:]),-2)
        occupied, nearest=dataset.get_nearest_free_voxel(query,ctx['scene_flag'])
        distance=(query-nearest).double().square().sum(-1)
        energies.append((distance[...,:24].mean(-1)+distance[...,24:].mean(-1))/(3*.05**2))
        hand=torch.cdist(human[:,2:,22:24].flatten(0,1),surface[:,2:].flatten(0,1)).amin((-1,-2))
        contacts.append(hand)
        floor,_=source_floor_height(human)
        stance=human[...,(7,8,10,11),1]-floor[:,None,None]<human.new_tensor([.08,.08,.04,.04])
        mask=stance[:,2:] & stance[:,1:-1]
        speed=(human[:,2:,(7,8,10,11)][:,:,:,(0,2)]-human[:,1:-1,(7,8,10,11)][:,:,:,(0,2)]).norm(dim=-1)/.1
        speeds.append(speed[mask])
        all_humans.append(human)
        valid &= bool(torch.isfinite(clean).all()) and bool(torch.isfinite(query).all())
    seam=max((float((a[:,-2:]-b[:,:2]).abs().max()) for a,b in zip(all_humans,all_humans[1:])),default=0.)
    valid &= seam<=1e-5 and all(r['history_exact'] and r['common_exact'] and r['contact_exact']
                and not r['invalid_proposal'] and all(r['final_guards'].values()) for r in records)
    contact=torch.cat(contacts);support=torch.cat(speeds)
    # External goal geometry, calculated directly before native report loading.
    root=all_humans[-1][0,-1,0]
    obj=state['object_translation_world'][0,-1]
    features = dict(valid=valid,energy=float(torch.cat(energies,1).mean()),
        root_endpoint_m=float((root[[0,2]]-root.new_tensor(task['pelvis_goal'])[[0,2]]).norm()),
        object_endpoint_m=float((obj[[0,2]]-obj.new_tensor(task['object_goal'])[[0,2]]).norm()),
        contact_distance_m=float(contact.mean()),contact_fraction=float((contact<.10).float().mean()),
        contact_frame_count=len(contact),support_speed_m_per_s=float(support.mean()) if len(support) else 0.,
        support_pair_count=len(support),history_world_max_abs_m=seam)
    if endpoint_mode == 'native_retained_v2':
        features.update(saved_endpoint_features(saved, task, points.device))
    elif endpoint_mode != 'legacy_XZ':
        raise ValueError('unknown endpoint mode: ' + endpoint_mode)
    return features


def _task_data(dataset, task):
    """Load the original text/BPS condition without materializing every BPS file."""
    import numpy as np
    from utils import zup_to_yup
    idx = task['data_idx']; seq = dataset.ori_sequence_idx[idx]
    name = dataset.scene_name[seq]
    offset = int(dataset.start_ind[idx]-dataset.ori_sequence_start_idx[seq])
    bps = np.load(Path(dataset.dest_obj_bps_npy_folder)/(name+'.npy'),mmap_mode='r')
    text = dataset.text[idx][0]
    embedding = torch.from_numpy(dataset.clip_features[dataset.text2features_idx[text]].copy()).float().reshape(1,-1)
    return dict(seq_name=name,text_clip_embedding=embedding/embedding.norm(dim=1,keepdim=True),
                obj_bps_data=zup_to_yup(torch.from_numpy(bps[offset:offset+1].copy()).float()))


def score_candidate_scene(run_root, scene_name, resolved_config, device='cuda:0'):
    """Read cached complete candidates, score once, then emit immutable choices."""
    import subprocess
    import numpy as np
    import hydra
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from utils import init_model
    from astar import get_path
    from .scene_calibration import recover_temporal_window
    from tools.experiment import sha256_file
    if subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('reportable candidate scoring requires clean worktree')
    root=Path(run_root);protocol=json.loads((root/'protocol.json').read_text())
    slots=json.loads((root/'candidate_manifest.json').read_text())['candidates']
    tasks=json.loads(Path(protocol['task_manifest']).read_text())['tasks']
    tasks=[t for t in tasks if t['scene_name']==scene_name]
    cfg=OmegaConf.load(resolved_config);cfg.device=device;cfg.dataset.device=device
    cfg.sampler.pelvis.hoi_adapter.device=device;cfg.sampler.pelvis.hsi_sampler.device=device
    cfg.dataset.vis=True;cfg.dataset.load_object_payload=False;cfg.dataset.test_scene_name=scene_name
    dataset=InfBaGelDataset(**cfg.dataset)
    dataset.obj_rest_verts={k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
    model_cfg=OmegaConf.merge(cfg.model.infbagel,{'ckpt':cfg.hsi_ckpt_path})
    model=init_model(model_cfg,device=device,eval=True).eval().requires_grad_(False)
    sampler=hydra.utils.instantiate(cfg.sampler.pelvis)
    sampler.dataset=dataset;sampler.hsi_sampler.set_dataset_and_model(dataset,model)
    out=root/'scores'/scene_name;out.mkdir(parents=True,exist_ok=False)
    native=json.loads((Path(cfg.dataset.folder).parent/'hosi_test/data'/(scene_name+'.json')).read_text())
    started=time.perf_counter();all_rows=[]
    torch.cuda.reset_peak_memory_stats(device)
    for task_row in tasks:
        ordinal=task_row['canonical_ordinal'];task=native[task_row['test_idx']]
        data=_task_data(dataset,task)
        trajectory=get_path(np.asarray(task['start_location'])[[0,2]],np.asarray(task['pelvis_goal'])[[0,2]],dataset)
        features,scores={},{}
        for slot in sorted((s for s in slots if s['task']==ordinal),key=lambda s:s['candidate']):
            path=Path(slot['motion_path']);saved=torch.load(path,map_location='cpu',weights_only=False)
            audit=json.loads(Path(slot['audit_path']).read_text())['sampler_audit']
            records=audit['composition']['scene_edit']['records'][-len(saved['windows']):]
            if len(records)!=len(saved['corrections']):
                raise ValueError('candidate correction/record coverage mismatch')
            expected_seed=42+ordinal+protocol['generation']['candidate_seed_offsets'][slot['candidate']]
            if [r['seed'] for r in records] != [expected_seed+i*1000003 for i in range(len(records))]:
                raise AssertionError('candidate generation seed mismatch')
            points=dataset.obj_rest_verts[saved['object_name']].to(device)
            points=points[torch.linspace(0,len(points)-1,128,device=device).long()][None]
            contexts,offsets,windows,recovery=[],[],[],[]
            hsi_seconds=0.
            torch.cuda.synchronize(device);start=time.perf_counter()
            for i,(snapshot,world) in enumerate(zip(saved['corrections'],saved['windows'])):
                fitter,c,validation,geom,rest=recover_temporal_window(dataset,saved,snapshot,world,task,points,device)
                recovered,error=recover_score_context(sampler,cfg,saved,i,task,data,trajectory,geom,rest)
                if 'replay_context' in snapshot:
                    context=move_tree(snapshot['replay_context'],device)
                    context.update(obj_rest_verts=dataset.obj_rest_verts,obj_vert_normals={})
                    if sampler.inference_engineering:context['static_occ_cache']={}
                    discrepancies={k:float((context[k].double()-recovered[k].double()).abs().max())
                                   for k in recovered if torch.is_tensor(recovered[k])}
                    if any(v>1e-5 for v in discrepancies.values()):
                        raise AssertionError('captured/native recovered conditions disagree: '+str(discrepancies))
                    validation['captured_context_max_abs']=max(discrepancies.values())
                else:
                    context=recovered
                contexts.append(context);offsets.append(rest)
                validation.update(context_world_max_abs_m=error,context_recovered='replay_context' not in snapshot)
                recovery.append(validation)
                torch.cuda.synchronize(device);query_start=time.perf_counter()
                window=score_window(sampler,c['edited'],context,ordinal,i,protocol['score'])
                torch.cuda.synchronize(device);hsi_seconds+=time.perf_counter()-query_start
                window.update(window_index=i,generation_seed=records[i]['seed'])
                windows.append(window)
                with (out/'per_window.jsonl').open('a') as handle:
                    handle.write(json.dumps(dict(task=ordinal,candidate=slot['candidate'],**window),allow_nan=False)+'\n')
            torch.cuda.synchronize(device);seconds=time.perf_counter()-start
            features[slot['candidate']]=candidate_features(dataset,saved,contexts,offsets,task,records)
            scores[slot['candidate']]=aggregate_scores(windows)
            after=sha256_file(path)
            if after!=slot['motion_sha256']:
                raise AssertionError('candidate file changed during scoring')
            row=dict(task=ordinal,candidate=slot['candidate'],scores=scores[slot['candidate']],
                features=features[slot['candidate']],hsi_seconds=hsi_seconds,motion_path=str(path),motion_sha256=after,
                recovery=recovery,windows=len(windows),scoring_seconds=seconds,
                hsi_calls=sum(w['hsi_calls'] for w in windows),
                generation_seconds=sum(r['seconds'] for r in records),
                geometry_seconds=sum(r['geometry_seconds'] for r in records))
            all_rows.append(row)
            with (out/'per_candidate.jsonl').open('a') as handle:handle.write(json.dumps(row,allow_nan=False)+'\n')
            print('scored',ordinal,slot['candidate'],len(windows),'windows',flush=True)
        selection_start=time.perf_counter()
        decision=select_candidates(features,scores,protocol['selection'])
        decision['selection_seconds']=time.perf_counter()-selection_start
        decision.update(task=ordinal,scene=scene_name,object=task_row['object_name'])
        decision['selected_files']={name:dict(motion_path=next(s['motion_path'] for s in slots if s['task']==ordinal and s['candidate']==idx),
            motion_sha256=next(s['motion_sha256'] for s in slots if s['task']==ordinal and s['candidate']==idx))
            for name,idx in decision['selected'].items()}
        with (out/'selections.jsonl').open('a') as handle:handle.write(json.dumps(decision,allow_nan=False)+'\n')
        print('selected',ordinal,decision['selected'],flush=True)
    result=dict(scene=scene_name,candidates=len(all_rows),tasks=len(tasks),seconds=time.perf_counter()-started,
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


def summarize_candidate_selection(run_root, device='cuda:7'):
    """Load native reports only after frozen choices have been saved."""
    import numpy as np
    from .scene_calibration import paired_local_metrics
    root=Path(run_root);protocol=json.loads((root/'protocol.json').read_text())
    slots=json.loads((root/'candidate_manifest.json').read_text())['candidates']
    rows=[json.loads(line) for p in sorted((root/'scores').glob('*/per_candidate.jsonl')) for line in p.read_text().splitlines()]
    decisions=[json.loads(line) for p in sorted((root/'scores').glob('*/selections.jsonl')) for line in p.read_text().splitlines()]
    if len(slots)!=112 or len(rows)!=112 or len(decisions)!=28:
        raise ValueError('fixed112 candidates /28 selection coverage incomplete')
    out=root/'analysis';out.mkdir(exist_ok=False)
    def write(name,value):
        with (out/name).open('x') as h:json.dump(value,h,indent=2,allow_nan=False)
    def mean(values):
        return {k:float(np.mean([v[k] for v in values])) for k in values[0]}
    native={};cost={};groups={k:{} for k in ('B0','C0','CG','CS','CF','CM','oracle')}
    for slot in slots:
        audit=json.loads(Path(slot['audit_path']).read_text())
        metric=audit['metrics']
        native[(slot['task'],slot['candidate'])]={k:float(v) for k,v in metric.items()
            if isinstance(v,(int,float,bool)) and k not in ('test_idx','canonical_ordinal')}
        if any(v is None for v in metric.values()):raise ValueError('missing native metric')
        cost[(slot['task'],slot['candidate'])]=audit['generation_seconds']
    scene_by_task={str(d['task']):d['scene'] for d in decisions}
    hs='scene_human_penetration_s_mean';oskey='scene_obj_penetration_s_mean'
    oracle_rows=[]
    for d in decisions:
        task=d['task'];name=str(task)
        for g,idx in d['selected'].items():groups[g][name]=native[(task,idx)]
        baseline=next(Path(protocol['baseline_root']).glob(f'B0_hoi-*/episode-audit-{task:03d}.json'))
        b=json.loads(baseline.read_text())['metrics']
        groups['B0'][name]={k:float(b[k]) for k in groups['CG'][name]}
        geometry=groups['CG'][name]
        eligible=[]
        for idx in d['acceptable']:
            m=native[(task,idx)]
            if (m['completed']>=geometry['completed'] and m['contact_percent']>=geometry['contact_percent']-.02
                and m['foot_sliding']<=geometry['foot_sliding']+.01 and m[oskey]<=geometry[oskey]+.02
                and all(m[k]<=geometry[k]+1. for k in ('xy_points_err','end_obj_trans_err'))):eligible.append(idx)
        oracle=min(eligible,key=lambda idx:(native[(task,idx)][hs],idx)) if eligible else d['selected']['CG']
        groups['oracle'][name]=native[(task,oracle)]
        oracle_rows.append(dict(task=task,eligible=eligible,selected=oracle,budget_set=d['budget_set'],
                                improved_over_CG=native[(task,oracle)][hs]<geometry[hs]))
    scenes={g:{s:mean([v for n,v in data.items() if scene_by_task[n]==s])
               for s in sorted(set(scene_by_task.values()))} for g,data in groups.items()}
    means={g:mean(list(v.values())) for g,v in groups.items()}
    comparisons=[('CF','CG'),('CF','CS'),('CF','C0'),('CS','CG'),('CG','C0'),('CF','CM'),('oracle','CG'),('CF','B0')]
    contrasts={unit:{a+'-'+b:paired_local_metrics(data[b],data[a],device) for a,b in comparisons}
               for unit,data in (('task',groups),('scene',scenes))}
    primary=contrasts['task']['CF-CG'][hs]
    gates=dict(primary_HS=primary['delta']<=-.05*means['CG'][hs] and primary['ci'][1]<0
               and contrasts['scene']['CF-CG'][hs]['delta']<=0)
    for unit in ('task','scene'):
        for b in ('CG','C0'):
            c=contrasts[unit]['CF-'+b]
            gates[f'{unit}_OS_{b}']=c[oskey]['delta']<=0 and c[oskey]['ci'][1]<=.02
            gates[f'{unit}_FS_{b}']=c['foot_sliding']['ci'][1]<=.01
            for k in ('contact_percent','completed'):gates[f'{unit}_{k}_{b}']=c[k]['ci'][0]>=-.02
            for k in ('xy_points_err','end_obj_trans_err'):gates[f'{unit}_{k}_{b}']=c[k]['ci'][1]<=1.
    testable=any(len(d['budget_set'])>1 for d in decisions)
    valid=all(r['features']['valid'] for r in rows)
    # Invalid physical proposals remain in the pool and denominators, but cannot win A.
    gates['selected_valid']=all(next(r['features']['valid'] for r in rows if r['task']==d['task'] and r['candidate']==d['selected']['CF']) for d in decisions)
    agreement={g:sum(d['selected']['CF']==d['selected'][g] for d in decisions)/28 for g in ('CG','CS','CM','C0')}
    main_calls=sum(r['hsi_calls']//3 for r in rows)
    result=dict(phase='2.20',tasks=28,candidates=112,scenes=4,test_set_development=True,
        means=means,gates=gates,numerical_gate_passed=all(gates.values()),testable=testable,
        decision=('BLOCKED_SINGLETON_POOL' if not testable else ('REQUIRES_VISUAL_REVIEW' if all(gates.values()) else 'NO-GO')),
        agreement_CF=agreement,multiple_budget_tasks=sum(len(d['budget_set'])>1 for d in decisions),
        empty_acceptable_tasks=sum(not d['acceptable'] for d in decisions),all_pool_candidates_valid=valid,
        oracle_improvement_tasks=sum(o['improved_over_CG'] for o in oracle_rows),
        costs=dict(total_K4_generation_seconds=sum(cost.values()),mean_K4_generation_seconds_per_task=sum(cost.values())/28,
            new_generation_seconds=sum(v for (t,i),v in cost.items() if i>0),
            geometry_seconds=sum(r['geometry_seconds'] for r in rows),
            scoring_with_recovery_seconds=sum(r['scoring_seconds'] for r in rows),
            HSI_three_branches_seconds=sum(r['hsi_seconds'] for r in rows),
            selection_seconds=sum(d['selection_seconds'] for d in decisions),
            CF_required_forward_calls=main_calls,CS_diagnostic_forward_calls=main_calls,CM_diagnostic_forward_calls=main_calls),
        visual_review_pending=True,additional441_started=False,full469_started=False)
    strata={obj:{g:mean([data[str(d['task'])] for d in decisions if d['object']==obj]) for g,data in groups.items()}
            for obj in sorted(set(d['object'] for d in decisions))}
    write('summary.json',result);write('contrasts.json',contrasts);write('per_task.json',groups)
    write('per_scene.json',scenes);write('object_strata.json',strata);write('oracle.json',oracle_rows)
    write('candidate_native_metrics.json',[dict(task=t,candidate=i,metrics=m) for (t,i),m in native.items()])
    write('decisions.json',decisions)
    return result
