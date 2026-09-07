"""Paired generation with locally refreshed or persistent executed hand relations."""
import copy
import json
import subprocess
import time
import traceback
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

from priors.core.window_codec import project_to_so3
from .candidate_selection import move_tree
from .continuation_outcomes import (write_json, exact_tree, _world_record,
                                   branch_horizon, evaluate_state)
from .single_side_waypoint import random_snapshot, restore_random
from .kinematic_composition import (_expand_rest_offsets, _local_from_global,
                                    _forward_kinematics, _apply_rotation)


def relation_geometry(clean, dataset, offsets, context, rest):
    """Differentiable native FK and object-local palms, using the native prefix."""
    batch, frames = clean.shape[:2]
    positions = dataset.denormalize_torch(clean[..., :84]).reshape(batch, frames, 28, 3)
    rotation = transforms.rotation_6d_to_matrix(clean[..., 84:216].reshape(batch, frames, 22, 6))
    bones = _expand_rest_offsets(offsets, batch, frames, positions).clone()
    bones[..., 0, :] = positions[..., 0, :]
    _, fk = _forward_kinematics(_local_from_global(rotation), bones)
    mat = context['mat'].to(clean)
    human = _apply_rotation(mat[:, None, None, :3, :3], fk) + mat[:, None, None, :3, 3]
    object_local = dataset.denormalize_torch(clean[..., 216:219], is_object=True)
    trans = _apply_rotation(mat[:, None, :3, :3], object_local) + mat[:, None, :3, 3]
    rot = (context['obj_rot_mat_prefix'].to(clean).reshape(batch, 1, 3, 3)
           @ project_to_so3(clean[..., 219:228].reshape(batch, frames, 3, 3))
           @ context['obj_rot_mat_ref'].to(clean).reshape(batch, 1, 3, 3))
    relative = _apply_rotation(rot[:, :, None].transpose(-1, -2), human[..., 22:24, :] - trans[:, :, None])
    # Object coordinates avoid materializing a separate full surface per frame.
    distance = torch.cdist(relative.flatten(0, 1), rest[None].expand(batch*frames, -1, -1)).amin(-1).reshape(batch, frames, 2)
    return dict(relative=relative, distance=distance, contact=clean[..., 228:230], human=human)


class RelationMemory:
    def __init__(self, version, settings):
        self.version, self.settings = version, settings
        self.anchor = None
        self.active = None
        self.ever_trusted = None
        self.events = []

    @torch.no_grad()
    def observe(self, geometry, explicit_release=None):
        """Only executed geometry enters this transition; candidate contact is absent."""
        relative, distance, contact = (geometry[k] for k in ('relative', 'distance', 'contact'))
        p = self.settings
        trusted = ((distance <= p['trust_distance_m']).all(1)
                   & (contact > p['trust_contact_threshold']).all(1)
                   & ((relative[:, -1]-relative[:, 0]).norm(dim=-1) <= p['trust_anchor_step_m']))
        ambiguous = ((distance > p['ambiguous_distance_m']).all(1)
                     & (contact <= p['trust_contact_threshold']).all(1))
        release = torch.zeros_like(trusted) if explicit_release is None else explicit_release.bool()
        if self.active is None:
            self.active = torch.zeros_like(trusted)
            self.anchor = torch.zeros_like(relative[:, 0])
            self.ever_trusted = torch.zeros_like(trusted)
        previous = self.active.clone()
        suspended = previous & ambiguous & ~release
        acquired = ~previous & trusted & ~release
        regrasp = acquired & self.ever_trusted
        active = (previous | acquired) & ~release & ~suspended
        refresh = active if self.version == 'C1' else acquired
        self.anchor = torch.where(refresh[..., None], relative.mean(1), self.anchor)
        self.active = active
        self.ever_trusted |= acquired
        event = dict(active=active.tolist(), trusted_observation=trusted.tolist(),
            acquired=acquired.tolist(), explicit_release=release.tolist(), suspended_ambiguous=suspended.tolist(),
            regrasp_ambiguous=regrasp.tolist(), unknown=(~active & ~release).tolist(),
            anchor_object_m=self.anchor.tolist(), refreshed=refresh.tolist(),
            history_distance_m=distance.tolist(), history_contact=contact.tolist())
        self.events.append(event)
        return event


def relation_energy(geometry, memory, settings):
    distance = torch.relu(geometry['distance'][:, 2:] - settings['distance_tolerance_m']).square()
    drift = (geometry['relative'][:, 2:]-memory.anchor[:, None]).norm(dim=-1)
    anchor = torch.relu(drift-settings['anchor_tolerance_m']).square()
    mask = memory.active[:, None].to(distance)
    d = (distance*mask).mean()
    a = (anchor*mask).mean()
    return settings['distance_weight']*d + settings['anchor_weight']*a, d, a


class RelationalGuidance:
    def __init__(self, version, settings):
        self.settings = settings
        self.memory = RelationMemory(version, settings)
        self.window_records = []

    def begin_window(self, sampler, context, human_dict, original_guidance):
        self.dataset, self.context = sampler.dataset, context
        self.offsets = human_dict['rest_human_offsets']
        name = context['seq_name_dict'][0].split('_')[1]
        self.rest = sampler.dataset.obj_rest_verts[name]
        self.variance = original_guidance.posterior_variance
        self.telemetry = []
        self.has_active = bool(self.memory.active.any())

    def apply(self, posterior, clean, fixed, step):
        p = self.settings
        if step >= p['last_steps'] or not self.has_active:
            return posterior
        with torch.enable_grad():
            x = clean.detach().requires_grad_(True)
            g = relation_geometry(x, self.dataset, self.offsets, self.context, self.rest)
            energy, distance, anchor = relation_energy(g, self.memory, p)
            gradient = torch.autograd.grad(-energy, x)[0]
        update = (gradient * p['scale'] * self.variance[step]).clamp(-p['update_clamp'], p['update_clamp'])
        result = posterior + update
        result[:, :2] = fixed
        self.telemetry.append(torch.stack((energy.detach(), distance.detach(), anchor.detach(),
            gradient.abs().max(), gradient.square().mean().sqrt(), update.abs().max(),
            update.square().mean().sqrt(), (~torch.isfinite(gradient)).sum().to(energy))))
        return result

    @torch.no_grad()
    def finish_sampling(self, clean):
        self.raw_source = clean.detach().cpu().clone()
        audit = torch.stack(self.telemetry).cpu().tolist() if self.telemetry else []
        self.window_records.append(dict(state=copy.deepcopy(self.memory.events[-1]),
            energy_columns=['total','distance_m2','anchor_m2','gradient_max','gradient_rms','update_max','update_rms','nonfinite_elements'],
            steps=list(range(self.settings['last_steps']-1, 0, -1)) if audit else [], telemetry=audit))


@torch.no_grad()
def generate_relational_branch(cfg, dataset, hoi, hsi, record, episode, cached, task,
                               arm, version, destination, protocol, resume_current=None):
    import hydra
    from astar import get_path
    from test_infbagel_hosi import (get_guidance_from_json, prepare_next_window,
                                   sample_step, seed_everything)
    device = torch.device(cfg.device)
    ambient = random_snapshot(device)
    start_wall = time.perf_counter()
    cost = dict(HOI_calls=0, HSI_calls=0, generated_windows=0, attempted_windows=0)
    payload = dict(source=record, action=arm, version=version, training_allowed=False,
                   windows=[], costs=cost, consistency=[], failure=None)
    reused_cost = None
    hooks = []
    try:
        seed_everything(record['rng']['episode_seed'])
        sampler = hydra.utils.instantiate(cfg.sampler.pelvis)
        sampler.set_dataset_and_model(dataset, hoi, hsi_model=hsi)
        seed_everything(record['rng']['episode_seed'])
        sampler.inner_hoi.sample_calls = record['window']
        sampler.compose_calls = record['window']*500
        guide = RelationalGuidance(version, protocol['relation'])
        sampler.relation_guidance = guide
        context = move_tree(cached['snapshot']['replay_context'], device)
        source = cached['snapshot']['edited'].to(device)
        offsets = cached['snapshot']['rest_offsets'][0, 0].to(device)
        rest = dataset.obj_rest_verts[record['object']]
        if record['window']:
            previous_snapshot = episode['corrections'][record['window']-1]
            evidence_clean = previous_snapshot['edited'].to(device)[:, 12:14]
            evidence_context = move_tree(previous_snapshot['replay_context'], device)
        else:
            evidence_clean, evidence_context = source[:, :2], context
        guide.memory.observe(relation_geometry(evidence_clean, dataset, offsets, evidence_context, rest))
        if resume_current is not None:
            recovered = torch.load(resume_current,map_location='cpu',weights_only=False)
            saved = recovered['windows'][0]
            if (len(recovered['windows']) != 1 or recovered['source']['state_id'] != record['state_id']
                    or recovered['action'] != arm or recovered['version'] != version
                    or recovered['failure']['message'] != 'pre-editor sample identity differs'):
                raise ValueError('resume contract requires the retained current window from the recorder failure')
            reused_cost = recovered['costs']
            for key in ('HOI_calls','HSI_calls','generated_windows','attempted_windows'):
                cost[key] = reused_cost[key]
            payload['resume_source'] = str(resume_current)
        cond = get_guidance_from_json(cfg, task)
        cond['raw_text'] = dataset.text[task['data_idx']][0]
        cond['text_emb'] = context['text_emb'].clone()
        trajectory = get_path(np.asarray(task['start_location'])[[0,2]], np.asarray(task['pelvis_goal'])[[0,2]], dataset)
        steps, terminal = branch_horizon(record['window'], record['total_windows'], 2)
        def count_hoi(module, args): cost['HOI_calls'] += 1
        def count_hsi(module, args): cost['HSI_calls'] += 1
        hooks = [hoi.register_forward_pre_hook(count_hoi), hsi.register_forward_pre_hook(count_hsi)]
        for step in [record['window'], *steps]:
            if step == record['window']:
                fixed = source[:, :2].clone()
                # The exact saved arm conditions include the original signed waypoint.
                arguments = dict(context, fixed_points=fixed,
                    need_scene=context['need_scene'], need_pi=context['need_pi'],
                    need_pelvis_dir=context['need_pelvis_dir'], obj_rest_verts=dataset.obj_rest_verts,
                    obj_vert_normals={}, human_dict={'rest_human_offsets': offsets})
            else:
                guide.memory.observe(relation_geometry(clean[:, 12:14], dataset, offsets, context, rest))
                mat, fixed, points = prepare_next_window(cfg,dataset,step,record['scene'],episode['test_idx'],
                    context['seq_name_dict'],dataset.obj_rest_verts,context['obj_rot_mat_ref'],context['obj_rot_mat_prefix'],
                    previous['points_orig'],previous['obj_trans_orig'],previous['object_rot_mat'],
                    previous['global_rot_6d'],previous['contact_label'])
            reuse = resume_current is not None and step == record['window']
            cost['attempted_windows'] += int(not reuse)
            torch.cuda.synchronize(device); started = time.perf_counter()
            if reuse:
                guide.raw_source = saved['raw_source'].clone()
                guide.raw_source[:, :2] = saved['clean'][:, :2]
                guide.window_records.append(copy.deepcopy(saved['relation']))
                snapshot = dict(edited=saved['clean'],raw_source=guide.raw_source)
                new_context = move_tree(saved['context'],device)
                editor_record = saved['editor']
            elif step == record['window']:
                output = sampler.p_sample_loop(**arguments)
            else:
                pi = torch.tensor([step*42], device=device, dtype=torch.long)
                sample_step(cfg,step,mat,fixed,sampler,copy.deepcopy(cond),trajectory,pi,pi+48,
                    context['seq_length'].clone(),context['obj_bps_data'].clone(),points,dataset.obj_rest_verts,{},
                    context['seq_name_dict'],context['obj_rot_mat_ref'].clone(),
                    {'rest_human_offsets':offsets},context['obj_rot_mat_prefix'].clone())
            torch.cuda.synchronize(device); seconds = time.perf_counter()-started
            cost['generated_windows'] += int(not reuse)
            if reuse:
                seconds = saved['generation_seconds']
                sampler.inner_hoi.sample_calls = step + 1
                sampler.compose_calls = (step + 1)*500
            else:
                snapshot = sampler.scene_editor.motion_records[-1]
                new_context = sampler._window_context
                editor_record = sampler.scene_editor.records[-1]
            clean = snapshot['edited'].to(device)
            world, previous = _world_record(cfg,dataset,clean,new_context)
            small_context = move_tree({k:v for k,v in new_context.items()
                if k not in ('obj_rest_verts','obj_vert_normals','static_occ_cache')},'cpu')
            window_audit = guide.window_records[-1]
            payload['windows'].append(dict(absolute_window=step,clean=clean.cpu(),context=small_context,
                world=move_tree(world,'cpu'),cached=False,generation_seconds=seconds,
                editor=editor_record,raw_source=guide.raw_source,reused_generation=reuse,
                relation=window_audit,sample_calls=sampler.inner_hoi.sample_calls))
            if step == record['window']:
                payload['consistency'].append(dict(current_context_exact=exact_tree(small_context,cached['snapshot']['replay_context']),
                    current_history_exact=torch.equal(clean[:,:2].cpu(),source[:,:2].cpu())))
                if not all(payload['consistency'][-1].values()):
                    raise AssertionError('current conditions/history differ from registered cached arm')
            if not torch.isfinite(clean).all() or not torch.equal(clean[:,:2],fixed):
                raise AssertionError('generated finite/history contract failed')
            if not torch.equal(snapshot['raw_source'],guide.raw_source):
                raise AssertionError('pre-editor sample identity differs')
            for stage, motion in [('sampled',guide.raw_source.to(device)),('edited',clean)]:
                geom = relation_geometry(motion,dataset,offsets,new_context,rest)
                energy, d, a = relation_energy(geom,guide.memory,protocol['relation'])
                window_audit[stage] = dict(energy=float(energy),distance_energy_m2=float(d),anchor_energy_m2=float(a),
                    hand_surface_m=geom['distance'].cpu().tolist(),object_local_palms_m=geom['relative'].cpu().tolist())
            context = new_context
        payload.update(termination='original_terminal' if terminal else 'budget_censored',terminal_observed=terminal)
    except Exception as error:
        payload.update(failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()),
                       termination='execution_failure',terminal_observed=False)
    finally:
        for hook in hooks: hook.remove()
        torch.cuda.synchronize(device)
        cost.update(wall_seconds=time.perf_counter()-start_wall,
                    generation_seconds=sum(w['generation_seconds'] for w in payload['windows']),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        cost['new_generated_windows'] = cost['generated_windows'] - (reused_cost['generated_windows'] if reused_cost else 0)
        cost['new_HOI_calls'] = cost['HOI_calls'] - (reused_cost['HOI_calls'] if reused_cost else 0)
        if reused_cost:
            cost['wall_seconds'] += reused_cost['wall_seconds']
        restore_random(ambient,device)
        with (Path(destination)/(version+'_'+arm+'.pt')).open('xb') as f:torch.save(payload,f)
    return payload


def run_relational_scene(cfg):
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from priors.hoi.models import load_trained_hoi_prior
    from utils import init_model
    from test_infbagel_hosi import seed_everything
    if cfg.get('run_id') and subprocess.check_output(['git','status','--porcelain'],text=True).strip():
        raise RuntimeError('reportable relational sampling requires clean worktree')
    commit = subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    started = time.perf_counter()
    path = Path(cfg.relational_sampling.protocol)
    root = path.resolve().parents[2]
    protocol = json.loads(path.read_text())
    metric_protocol = json.loads((root/protocol['source_protocol']).read_text())
    records = json.loads((root/protocol['state_manifest']).read_text())['records']
    selected = {}
    for record in records:
        if record['scene']==cfg.relational_sampling.scene and (cfg.relational_sampling.state_ids is None or record['state_id'] in cfg.relational_sampling.state_ids):
            selected.setdefault(record['state_id'], record)
    out = Path(cfg.hosi_output_dir); out.mkdir(parents=True,exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg,resolve=True)),out/'resolved.yaml')
    seed_everything(42); device = torch.device(cfg.device)
    dataset = InfBaGelDataset(**cfg.dataset)
    dataset.obj_rest_verts = {k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
    hoi,_ = load_trained_hoi_prior(cfg.ckpt_path,device,weight_variant=cfg.checkpoint_weight_variant)
    hoi.eval().requires_grad_(False)
    hsi = init_model(OmegaConf.merge(cfg.model.infbagel,{'ckpt':cfg.hsi_ckpt_path}),device=device,eval=True)
    hsi.eval().requires_grad_(False)
    scene = str(cfg.relational_sampling.scene)
    task_rows = json.loads((root/'experiments/tasks/conditional_repair_native_development_s42_20260907.json').read_text())['tasks']
    lookup = {r['canonical_ordinal']:r for r in task_rows}
    native = json.loads((root/'data/hosi_test/data'/(scene+'.json')).read_text())
    key = scene+'_sdf'; sdf_root = root/'data/hosi_test/Scene_sdf'
    sdf = {key:np.load(sdf_root/(key+'.npy'))}
    info = {key:json.loads((sdf_root/(key+'_info.json')).read_text())}
    rows=[]; smpl_cache={}
    torch.cuda.reset_peak_memory_stats(device)
    for state_id, source_record in selected.items():
        record = copy.deepcopy(source_record)
        for asset in [record['episode'],record['decision'],*record['candidates'].values()]:asset['path']=str(root/asset['path'])
        task = native[lookup[record['task']]['test_idx']]
        dest=out/state_id;dest.mkdir()
        episode=torch.load(record['episode']['path'],map_location='cpu',weights_only=False)
        arms = ['W0',record['selected']]
        cached={a:torch.load(record['candidates'][a]['path'],map_location='cpu',weights_only=False) for a in arms}
        c0_dir = next((root/protocol['source_run']/'lanes').glob('*/'+state_id))
        branches={a:torch.load(c0_dir/(a+'.pt'),map_location='cpu',weights_only=False) for a in arms}
        for a in arms:(dest/('C0_'+a+'.pt')).symlink_to((c0_dir/(a+'.pt')).resolve())
        versions = ['C1','C2'] if int(state_id.split('-')[-1])%2==0 else ['C2','C1']
        order = arms if int(state_id.split('-')[-1])%2==0 else arms[::-1]
        for version in versions:
            for arm in order:
                name=version+'_'+arm
                resume = (Path(cfg.relational_sampling.resume_source)/state_id/(name+'.pt')
                          if state_id in cfg.relational_sampling.reuse_states else None)
                branches[name]=generate_relational_branch(cfg,dataset,hoi,hsi,record,episode,cached[arm],task,arm,version,dest,protocol,resume)
                record['cached_guard'][name]=record['cached_guard'][arm]
        errors = [name for name,b in branches.items() if b['failure']]
        torch.cuda.synchronize(device); began=time.perf_counter()
        try:
            if errors:raise RuntimeError('generation failed: '+str(errors))
            result,arrays=evaluate_state(cfg,dataset,record,task,branches,smpl_cache,sdf,info,metric_protocol)
            with (dest/'tracks.pt').open('xb') as f:torch.save(arrays,f)
            for b in result['branches'].values():
                b['legacy_proxy_labels']=b.pop('diagnostics')
            result.pop('beneficial_compatible_actions'); result.pop('no_useful_alternative')
        except Exception as error:
            result=dict(state_id=state_id,task=record['task'],scene=scene,
                branches={k:dict(costs=b['costs'],failure=b['failure']) for k,b in branches.items()},
                evaluation_failure=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        torch.cuda.synchronize(device)
        result.update(evaluation_seconds=time.perf_counter()-began,source_records=[r['record_id'] for r in records if r['state_id']==state_id],
                      new_generated_windows=sum(b['costs']['new_generated_windows'] for k,b in branches.items() if k.startswith('C')))
        write_json(dest/'outcomes.json',result);rows.append(result)
        print(json.dumps(dict(state=state_id,task=record['task'],generated=result['new_generated_windows'],errors=errors,
                              evaluation_failure=result.get('evaluation_failure'))),flush=True)
        if errors or result.get('evaluation_failure'):break
    errors=sum(bool(r.get('evaluation_failure')) for r in rows)
    write_json(out/'lane.json',dict(technical_errors=errors,commit_at_start=commit,
        commit_at_completion=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),scene=scene,
        states=list(selected),completed_states=len(rows),wall_seconds=time.perf_counter()-started,
        new_generated_windows=sum(r['new_generated_windows'] for r in rows),peak_allocated_bytes=torch.cuda.max_memory_allocated(device)))
    if errors:raise RuntimeError(f'{errors} relational states failed; original artifacts retained')


def direct_quality(value, reference, limits):
    """Direct physical margins; anchors and intermediate goals remain diagnostics."""
    failures=[]
    for i,(hand,base) in enumerate(zip(value['hands'],reference['hands'])):
        if base['fixed_active']:
            if hand['surface_mean_m']-base['surface_mean_m']>limits['contact_distance_increase_m']:
                failures.append(f'hand{i}:distance')
            if hand['coverage_5cm']-base['coverage_5cm'] < -limits['contact_fraction_drop']:
                failures.append(f'hand{i}:coverage')
    if value['support_speed_m_per_s'] is not None and reference['support_speed_m_per_s'] is not None:
        if value['support_speed_m_per_s']-reference['support_speed_m_per_s']>limits['support_speed_increase_m_per_s']:
            failures.append('support_speed')
    if value['world_joint_speed_m_per_s']<reference['world_joint_speed_m_per_s']*limits['world_joint_speed_retention']:
        failures.append('world_joint_speed')
    return failures


def future_improvement(value, reference, limits):
    gains=[]
    for horizon in ('next_1','next_2'):
        if horizon not in value['slices'] or horizon not in reference['slices']:continue
        v,r=value['slices'][horizon],reference['slices'][horizon]
        for i,(h,b) in enumerate(zip(v['hands'],r['hands'])):
            if b['fixed_active']:
                if h['coverage_5cm']-b['coverage_5cm']>=limits['contact_fraction_drop']:gains.append(f'{horizon}:hand{i}:coverage')
                if b['surface_mean_m']-h['surface_mean_m']>=limits['contact_distance_increase_m']:gains.append(f'{horizon}:hand{i}:distance')
        if v['support_speed_m_per_s'] is not None and r['support_speed_m_per_s'] is not None:
            if r['support_speed_m_per_s']-v['support_speed_m_per_s']>=limits['support_speed_increase_m_per_s']:gains.append(horizon+':support_speed')
    return gains


def summarize_relational_sampling(run_root):
    import csv
    from collections import Counter, defaultdict
    from .continuation_outcomes import accumulate_branch_cost
    root=Path(run_root)
    protocol=json.loads((root/'protocol.json').read_text())
    limits=json.loads(Path(protocol['source_protocol']).read_text())['labels']
    out=root/'analysis';out.mkdir(exist_ok=False)
    table=[];details=[];errors=[];cost=Counter();checks=[];memory_counts=Counter();memory_changes=[]
    for path in sorted(root.glob('lanes/*/state-*/outcomes.json')):
        state=json.loads(path.read_text())
        if state.get('evaluation_failure'):
            errors.append(dict(state_id=state['state_id'],error=state['evaluation_failure']));continue
        selected=state['source_selected'];branches=state['branches'];base=branches['W0']
        payloads={}
        for version in ('C1','C2'):
            for arm in ('W0',selected):
                name=version+'_'+arm;row=branches[name]
                payload=torch.load(path.parent/(name+'.pt'),map_location='cpu',weights_only=False)
                payloads[name]=payload
                accumulate_branch_cost(cost,payload['costs'])
                checks.extend(payload['consistency'])
                events=[w['relation']['state'] for w in payload['windows']]
                known=any(events[0]['active'][0])
                engaged=any(any(e['active'][0]) for e in events)
                ambiguous=any(any(e[k][0]) for e in events for k in ('suspended_ambiguous','regrasp_ambiguous'))
                memory_counts[version+'_initial_trusted_branches']+=known
                memory_counts[version+'_ambiguous_branches']+=ambiguous
                memory_counts[version+'_applied_steps']+=sum(len(w['relation']['steps']) for w in payload['windows'])
                memory_counts[version+'_explicit_release_hands']+=sum(sum(e['explicit_release'][0]) for e in events)
                memory_counts[version+'_nonfinite_gradient_elements']+=sum(t[-1] for w in payload['windows'] for t in w['relation']['telemetry'])
                for w in payload['windows']:
                    audit=w['relation']
                    memory_changes.append(dict(state_id=state['state_id'],version=version,arm=arm,window=w['absolute_window'],
                        active=audit['state']['active'],sampled_energy=audit['sampled']['energy'],edited_energy=audit['edited']['energy'],
                        editor_energy_delta=audit['edited']['energy']-audit['sampled']['energy'],
                        gradient_max=max((t[3] for t in audit['telemetry']),default=0.),update_max=max((t[5] for t in audit['telemetry']),default=0.)))
                failure_w0={h:direct_quality(m,base['slices'][h],limits) for h,m in row['slices'].items()}
                same=branches[arm]
                failure_same={h:direct_quality(m,same['slices'][h],limits) for h,m in row['slices'].items()}
                gains=future_improvement(row,same,limits)
                cumulative=row['slices']['cumulative'];baseline=base['slices']['cumulative']
                scene_benefit=(cumulative['native_surface_HS_s_mean']<baseline['native_surface_HS_s_mean'] and
                               cumulative['native_surface_OS_s_mean']<=baseline['native_surface_OS_s_mean'])
                proxy_ok=row['legacy_proxy_labels']['scene_proxy_protected']
                terminal_regression=bool(same['terminal'] and same['terminal']['completed'] and not row['terminal']['completed'])
                joint_quality=bool(arm!='W0' and scene_benefit and proxy_ok and not any(failure_w0.values())
                                   and not terminal_regression)
                joint=joint_quality and engaged and not ambiguous
                item=dict(state_id=state['state_id'],task=state['task'],scene=state['scene'],version=version,arm=arm,
                    failures_vs_C0_W0=failure_w0,failures_vs_C0_same=failure_same,future_gains_vs_C0=gains,
                    scene_benefit=scene_benefit,scene_proxy_protected=proxy_ok,joint_candidate=joint,
                    joint_quality_candidate=joint_quality,ever_engaged=engaged,
                    initial_trust=known,ambiguity=ambiguous,terminal_regression=terminal_regression,
                    terminal=row['terminal'],termination=row['termination'])
                if version=='C2':
                    c1=branches['C1_'+arm]
                    item['future_gains_vs_C1']=future_improvement(row,c1,limits)
                    item['failures_vs_C1']={h:direct_quality(m,c1['slices'][h],limits) for h,m in row['slices'].items()}
                details.append(item)
                for comparator,reference in [('C0_same',same),('C0_W0',base)]+([('C1_same',branches['C1_'+arm])] if version=='C2' else []):
                    for horizon,metrics in row['slices'].items():
                        ref=reference['slices'][horizon]
                        flat=dict(state_id=state['state_id'],task=state['task'],scene=state['scene'],version=version,
                                  condition='W0' if arm=='W0' else 'adopted',comparator=comparator,horizon=horizon)
                        for key in ('native_surface_HS_s_mean','native_surface_OS_s_mean','native_fragment_FS_cm','contact_any_5cm',
                                    'support_speed_m_per_s','world_joint_speed_m_per_s','human_goal_error_cm','object_goal_error_3D_cm'):
                            flat[key]=metrics[key]-ref[key] if metrics[key] is not None and ref[key] is not None else None
                        for i,(h,r) in enumerate(zip(metrics['hands'],ref['hands'])):
                            for key in ('surface_mean_m','coverage_5cm','anchor_vs_W0_m'):
                                flat[f'hand{i}_{key}']=h[key]-r[key] if h[key] is not None and r[key] is not None else None
                        table.append(flat)
        for arm in ('W0',selected):
            a,b=payloads['C1_'+arm],payloads['C2_'+arm]
            checks.append(dict(state_id=state['state_id'],arm=arm,
                first_window_C1_C2_exact=torch.equal(a['windows'][0]['clean'],b['windows'][0]['clean'])))
    fields=list(table[0]) if table else []
    if table:
        with (out/'paired_components.csv').open('x') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(table)
    numeric=fields[7:]
    def aggregate(rows, keys):
        groups=defaultdict(list)
        for r in rows:groups[tuple(r[k] for k in keys)].append(r)
        output=[]
        for group,values in sorted(groups.items()):
            row=dict(zip(keys,group));row['count']=len(values)
            for k in numeric:
                valid=[r[k] for r in values if r.get(k) is not None]
                row[k]=sum(valid)/len(valid) if valid else None
            output.append(row)
        return output
    tasks=aggregate(table,['task','scene','version','condition','comparator','horizon'])
    scenes=aggregate(tasks,['scene','version','condition','comparator','horizon'])
    counts={}
    for version in ('C1','C2'):
        d=[r for r in details if r['version']==version and r['arm']!='W0']
        counts[version]=dict(adopted=len(d),joint_states=sum(r['joint_candidate'] for r in d),
            joint_quality_states=sum(r['joint_quality_candidate'] for r in d),
            joint_tasks=sorted({r['task'] for r in d if r['joint_candidate']}),
            improved_joint_tasks=sorted({r['task'] for r in d if r['joint_candidate'] and r['future_gains_vs_C0']}),
            scene_benefit_states=sum(r['scene_benefit'] for r in d),
            direct_failure_states=sum(any(r['failures_vs_C0_W0'].values()) for r in d),
            future_improved_states=sum(bool(r['future_gains_vs_C0']) for r in d))
    c2=[r for r in details if r['version']=='C2' and r['arm']!='W0']
    extra={r['task'] for r in c2 if r['joint_candidate'] and r['future_gains_vs_C1'] and not any(r['failures_vs_C1'].values())}
    regress={r['task'] for r in c2 if any(r['failures_vs_C1'].values())}
    w0_reg={v:{r['task'] for r in details if r['version']==v and r['arm']=='W0' and any(r['failures_vs_C0_same'].values())} for v in ('C1','C2')}
    all_checks=all(all(v for k,v in c.items() if k.endswith('_exact')) for c in checks)
    integrity=not errors and len(details)==88 and cost['generated_windows']<=264 and all_checks
    upgrade=bool(integrity and len(counts['C2']['improved_joint_tasks'])>=2 and len(extra)>=2 and len(regress)<=len(extra)
                 and len(w0_reg['C2'])<=len(w0_reg['C1']))
    summary=dict(status='completed' if integrity else 'incomplete',engineering_integrity=integrity,
        state_files=[str(p) for p in sorted(root.glob('lanes/*/state-*/outcomes.json'))],
        counts=counts,cost=dict(cost),relation_counts=dict(memory_counts),C2_extra_joint_tasks=sorted(extra),
        C2_direct_regression_tasks_vs_C1=sorted(regress),W0_regression_tasks={k:sorted(v) for k,v in w0_reg.items()},
        upgrade_native28=upgrade,decision='PREPARE_FIXED_NATIVE28_POLICY' if upgrade else 'DO_NOT_UPGRADE_FIXED_RELATION_SAMPLING',
        errors=errors,paired_checks=len(checks),all_pairing_checks=all_checks,
        denominator=dict(source_records=26,states=22,tasks=12,scenes=4,new_branches=88),
        inference='Selected-state descriptive paired evidence; no full-task success or HSI transfer claim for censored branches.')
    for name,value in [('summary',summary),('state_details',details),('task_means',tasks),('scene_means',scenes),
                       ('pairing_checks',checks),('relation_editor_effects',memory_changes)]:write_json(out/(name+'.json'),value)
    return summary
