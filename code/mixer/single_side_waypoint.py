"""Protected single-side waypoint decisions with one native window commit."""
import copy
import inspect
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from .candidate_selection import CandidatePoolSampler, move_tree, score_window
from .composed_sampler import HOSIComposedSampler
from .waypoint_control import (waypoint_variants, local_waypoint, decode_motion,
                               control_metrics, quality_margins)
from .waypoint_protection import assess_waypoint_quality


ORDER = ('W0', 'Wplus', 'Wminus')


def independent_variants(*args, **kwargs):
    result = waypoint_variants(*args, **kwargs)
    result['valid_sides'] = [a for a in ORDER[1:]
                             if a in result['variants'] and result['variants'][a]['reason'] is None]
    return result


def geometry_energy(metrics):
    return (metrics['human_scene_RMS_cm']**2 + metrics['object_scene_RMS_cm']**2) / 75.


def window_choices(metrics, checks, selection, scores=None):
    accepted = ['W0'] + [a for a in ORDER[1:] if a in checks and checks[a]['accepted']]
    energies = {a: geometry_energy(metrics[a]) for a in accepted}
    g = min(accepted, key=lambda a: (energies[a], ORDER.index(a)))
    budget = energies[g] + max(selection['budget_relative'] * energies[g], selection['budget_absolute'])
    allowed = [a for a in accepted if energies[a] <= budget]
    h = g if scores is None else min(allowed, key=lambda a: (scores[a]['full'], ORDER.index(a)))
    return dict(accepted=accepted, budget_set=allowed, energies=energies, budget=budget, G=g, H=h)


def random_snapshot(device):
    return (random.getstate(), np.random.get_state(), torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == 'cuda' else None)


def restore_random(snapshot, device):
    random.setstate(snapshot[0]); np.random.set_state(snapshot[1]); torch.set_rng_state(snapshot[2])
    if device.type == 'cuda':
        torch.cuda.set_rng_state(snapshot[3], device)


def fork_sampler(sampler):
    """Share frozen networks/assets, isolate mutable sampling and editor state."""
    branch = copy.copy(sampler)
    branch.hoi_adapter = copy.copy(sampler.hoi_adapter)
    branch.hoi_adapter.inner = copy.copy(sampler.inner_hoi)
    inner = branch.inner_hoi
    inner.audit = copy.deepcopy(sampler.inner_hoi.audit)
    inner.guidance_audit = copy.deepcopy(sampler.inner_hoi.guidance_audit)
    inner.guidance_rest_vertex_cache = dict(sampler.inner_hoi.guidance_rest_vertex_cache)
    inner.local_bps_digest = sampler.inner_hoi.local_bps_digest.copy()
    branch.hsi_sampler = copy.copy(sampler.hsi_sampler)
    branch.scene_editor = copy.copy(sampler.scene_editor)
    branch.scene_editor.records = []
    branch.scene_editor.motion_records = []
    return branch


def editor_valid(record):
    return (not record['invalid_proposal'] and record['history_exact'] and
            record['common_exact'] and record['contact_exact'] and all(record['final_guards'].values()))


@torch.no_grad()
def per_hand_diagnostics(state, reference, baseline):
    human = state['human']
    distances = torch.cdist(human[:, 2:, 22:24].flatten(0, 1),
                            state['object_surface'][:, 2:].flatten(0, 1)).amin(-1)
    def relative(s):
        return (s['object_rotation_world'].transpose(-1, -2)[..., None, :, :] @
                (s['human'][..., 22:24, :] - s['object_translation_world'][..., None, :])[..., None]).squeeze(-1)
    change = relative(state) - relative(reference)
    active = baseline[:, 2:, 228:230].flatten(0, 1) > .95
    rows = []
    for i in range(2):
        mask = active[:, i]
        rows.append(dict(hand=i, active_frames=int(mask.sum()),
            surface_m=distances[:, i].tolist(),
            active_surface_mean_m=float(distances[mask, i].mean()) if bool(mask.any()) else None,
            anchor_drift_m=change[:, 2:, i].norm(dim=-1).flatten().tolist(),
            relative_velocity_change_m_per_s=((change[:, 2:, i]-change[:, 1:-1, i])/.1).norm(dim=-1).flatten().tolist()))
    return rows


class SingleSideWaypointSampler(CandidatePoolSampler):
    def __init__(self, waypoint_policy='off', waypoint_protocol=None,
                 waypoint_output_dir=None, candidate_order=ORDER, **kwargs):
        super().__init__(**kwargs)
        self.waypoint_policy = waypoint_policy
        self.candidate_order = tuple(candidate_order)
        self.waypoint_output_dir = waypoint_output_dir
        self.decision_records = []
        if waypoint_policy != 'off':
            self.protocol = json.loads(Path(waypoint_protocol).read_text())
            self.repo = Path(waypoint_protocol).resolve().parents[2]
            source = json.loads((self.repo / self.protocol['quality_reference']).read_text())
            self.quality = source['gate']
            self.score_protocol = json.loads((self.repo / self.protocol['checkpoint_and_asset_reference']).read_text())['score']
            tasks = json.loads((self.repo / self.protocol['task_manifest']).read_text())['tasks']
            self.tasks = {r['canonical_ordinal']: r for r in tasks}

    def p_sample_loop(self, *args, **kwargs):
        if self.waypoint_policy == 'off':
            return super().p_sample_loop(*args, **kwargs)
        from astar import get_path
        from utils import transform_points
        bound = inspect.signature(HOSIComposedSampler.p_sample_loop).bind(self, *args, **kwargs)
        call = {k:v for k,v in bound.arguments.items() if k != 'self'}
        device = call['fixed_points'].device
        task_id = int(torch.initial_seed()) - 42
        window = self.inner_hoi.sample_calls
        task_row = self.tasks[task_id]
        native_path = self.repo / 'data/hosi_test/data' / (task_row['scene_name'] + '.json')
        task = json.loads(native_path.read_text())[task_row['test_idx']]
        points = self.dataset.obj_rest_verts[task['object_name']].to(device)
        points = points[torch.linspace(0, len(points)-1, 128, device=device).long()][None]
        offsets = call['human_dict']['rest_human_offsets']
        initial_rng = random_snapshot(device)
        torch.cuda.synchronize(device); started = time.perf_counter()
        branches, motions, metrics, checks, hands, seconds, traces = {}, {}, {}, {}, {}, {}, {}
        context = None
        proposal = None
        directory = Path(self.waypoint_output_dir) if self.waypoint_output_dir else None
        if directory:
            directory.mkdir(parents=True, exist_ok=True)
        # W0 determines fixed masks; the signed generation order remains interchangeable.
        order = ['W0'] + [a for a in self.candidate_order if a != 'W0']
        cache_matches = {}
        for arm in order:
            if arm != 'W0' and arm not in proposal['valid_sides']:
                continue
            restore_random(initial_rng, device)
            branch = fork_sampler(self)
            arguments = move_tree(call, device)
            if arm != 'W0':
                arguments['pelvis_goal'] = local_waypoint(
                    points.new_tensor(proposal['variants'][arm]['waypoint_world']), call['mat'])
            from .waypoint_control import _model_input_trace, validate_model_trace
            trace, hook = _model_input_trace(branch.student_model)
            torch.cuda.synchronize(device); began = time.perf_counter()
            try:
                output = CandidatePoolSampler.p_sample_loop(branch, **arguments)
            finally:
                hook.remove()
            torch.cuda.synchronize(device); seconds[arm] = time.perf_counter()-began
            traces[arm] = move_tree(trace, 'cpu')
            clean = output[0][-1]
            snapshot = branch.scene_editor.motion_records[-1]
            record = branch.scene_editor.records[-1]
            if directory:
                with (directory / f'task-{task_id:03d}-window-{window:03d}-{arm}.pt').open('xb') as handle:
                    torch.save(dict(snapshot=snapshot, editor=record, trace=traces[arm]), handle)
            validate_model_trace(trace, arguments['pelvis_goal'], record)
            if not torch.isfinite(clean).all() or not torch.equal(clean[:, :2], call['fixed_points']):
                raise AssertionError('candidate finite/history contract failed')
            if arm == 'W0':
                post_rng = random_snapshot(device)
                context = branch._window_context
                base = decode_motion(clean, self.dataset, offsets, context, points)
                path = get_path(np.asarray(task['start_location'])[[0, 2]],
                                np.asarray(task['pelvis_goal'])[[0, 2]], self.dataset)
                world = transform_points(context['pelvis_goal'].reshape(1, 1, 3), context['mat']).reshape(3)
                proposal = independent_variants(world, path, base['human'][0, 0, 0],
                    self.dataset.scene_grid_torch.to(device),
                    lambda p: self.dataset.get_occ_for_points(p, None, context['scene_flag']),
                    bool(context['is_loco'].all() and context['need_pelvis_dir'].all()),
                    self.protocol['generation']['offset_m'])
                normal = points.new_tensor(proposal['normal'] or [0., 0.])
                baseline = clean
            else:
                for key in ('latent', 'text', 'bps', 'progress'):
                    if not torch.equal(traces['W0'][key], traces[arm][key]):
                        raise AssertionError('unpaired candidate model input: ' + key)
                if not torch.equal(traces['W0']['goals'][:, 3:], traces[arm]['goals'][:, 3:]):
                    raise AssertionError('candidate changed fixed object/scene goals')
            state = decode_motion(clean, self.dataset, offsets, context, points)
            metrics[arm] = control_metrics(state, base, baseline, self.dataset, context,
                                           normal, -1 if arm == 'Wminus' else 1)
            hands[arm] = per_hand_diagnostics(state, base, baseline)
            limits = dict(quality_margins(self.quality), contact_fraction=self.quality['contact_fraction_drop'])
            check = assess_waypoint_quality(metrics[arm], metrics['W0'], limits,
                                            self.quality['world_joint_speed_retention'])
            if not editor_valid(record):
                check['failures'].append('editor_invalid')
                check['accepted'] = False
            checks[arm] = check
            if window == 0:
                cache = self.repo / self.protocol['source_run'] / ('scene-' + task_row['scene_name']) / f'task-{task_id:03d}-{arm}.pt'
                old = torch.load(cache, map_location='cpu', weights_only=False)
                cache_matches[arm] = all(torch.equal(old['snapshot'][k], snapshot[k]) for k in ('raw_source','proposal','edited'))
                if not cache_matches[arm]:
                    raise AssertionError('first-window source cache mismatch')
            branches[arm] = branch
            motions[arm] = clean
        choices = window_choices(metrics, checks, self.protocol['selection'])
        scores = {}; scoring_error = None; scoring_seconds = 0.
        calls = [0]
        if self.waypoint_policy == 'H' and len(choices['budget_set']) > 1:
            def count(module, inputs):
                calls[0] += 1
            hook = self.hsi_sampler.student_model.register_forward_pre_hook(count)
            torch.cuda.synchronize(device); began = time.perf_counter()
            try:
                for arm in choices['budget_set']:
                    scores[arm] = score_window(self, motions[arm], move_tree(context, device),
                                               task_id, window, self.score_protocol)
                choices = window_choices(metrics, checks, self.protocol['selection'], scores)
            except FloatingPointError as error:
                scoring_error = str(error)
            finally:
                hook.remove()
            torch.cuda.synchronize(device); scoring_seconds = time.perf_counter()-began
        chosen = choices[self.waypoint_policy]
        selected = branches[chosen]
        self.hoi_adapter = selected.hoi_adapter
        self.compose_calls = selected.compose_calls
        self.scene_editor.records.append(selected.scene_editor.records[-1])
        self.scene_editor.motion_records.append(selected.scene_editor.motion_records[-1])
        restore_random(post_rng, device)
        torch.cuda.synchronize(device)
        decision = dict(task=task_id, scene=task_row['scene_name'], object=task['object_name'],
            window=window, policy=self.waypoint_policy, proposal=proposal, quality=checks,
            metrics=metrics, hands=hands, scores=scores, choices=choices, selected=chosen,
            fallback=('input_inapplicable' if len(motions)==1 else 'no_qualified_offset' if len(choices['accepted'])==1
                      else 'W0_selected') if chosen=='W0' else None,
            score_error=scoring_error, cache_matches=cache_matches,
            generation_seconds=seconds, scoring_seconds=scoring_seconds,
            total_seconds=time.perf_counter()-started, HSI_calls=calls[0],
            HOI_calls=sum(t['calls'] for t in traces.values()), generated_candidates=len(motions),
            logical_sample_calls=self.inner_hoi.sample_calls,
            episode_seed=42+task_id, window_seed=42+task_id+window*1000003,
            common_score_context='W0', selected_editor_valid=editor_valid(self.scene_editor.records[-1]))
        self.decision_records.append(decision)
        self.scene_editor.records[-1]['waypoint_decision'] = decision
        self.scene_editor.motion_records[-1]['waypoint_decision'] = decision
        self.scene_editor.motion_records[-1]['waypoint_candidates'] = {
            arm: dict(motion=motions[arm].detach().cpu(), context=move_tree({k:v for k,v in
                branches[arm]._window_context.items() if k not in ('obj_rest_verts','obj_vert_normals','static_occ_cache')}, 'cpu'))
            for arm in motions}
        if directory:
            with (directory / f'task-{task_id:03d}-window-{window:03d}-decision.json').open('x') as handle:
                json.dump(decision, handle, indent=2, allow_nan=False)
        return [motions[chosen]], []

    def audit_dict(self):
        audit = super().audit_dict()
        audit['single_side_waypoint'] = dict(policy=self.waypoint_policy,
            windows=len(self.decision_records),
            generated_candidates=sum(r['generated_candidates'] for r in self.decision_records),
            HOI_calls=sum(r['HOI_calls'] for r in self.decision_records),
            HSI_calls=sum(r['HSI_calls'] for r in self.decision_records),
            note='base sampler audits describe committed branches; waypoint totals include all trials')
        return audit


def analyze_single_side_cache(protocol_path, output_dir, device='cuda:7'):
    """Recompute acceptance and geometry from per-window records, retaining all28."""
    protocol = json.loads(Path(protocol_path).read_text())
    repo = Path(protocol_path).resolve().parents[2]
    source = repo / protocol['source_run']
    rows = [json.loads(line) for p in sorted(source.glob('scene-*/control_results.jsonl'))
            for line in p.read_text().splitlines() if json.loads(line)['arm'] != 'Wplus_repeat']
    data = {(r['task'], r['arm']):r for r in rows}
    tasks = json.loads((repo / protocol['task_manifest']).read_text())['tasks']
    quality = json.loads((repo / protocol['quality_reference']).read_text())['gate']
    limits = dict(quality_margins(quality), contact_fraction=quality['contact_fraction_drop'])
    table = []
    for task in tasks:
        ordinal = task['canonical_ordinal']
        current = {a:data[ordinal,a] for a in ORDER if (ordinal,a) in data}
        reference = current['W0']['metrics']['edited']
        metrics = {a:r['metrics']['edited'] for a,r in current.items()}
        checks = {a:assess_waypoint_quality(m,reference,limits,quality['world_joint_speed_retention'])
                  for a,m in metrics.items()}
        payloads = {}
        for arm, row in current.items():
            path = source / ('scene-' + task['scene_name']) / f'task-{ordinal:03d}-{arm}.pt'
            payload = torch.load(path, map_location=device, weights_only=False)
            assert payload['task'] == ordinal and payload['arm'] == arm
            assert torch.isfinite(payload['snapshot']['edited']).all()
            assert editor_valid(row['editor'])
            payloads[arm] = payload
        base = payloads['W0']
        for arm, payload in payloads.items():
            assert torch.equal(payload['snapshot']['edited'][:,:2],base['snapshot']['edited'][:,:2])
            for k in ('latent','text','bps','progress'):
                assert torch.equal(payload['model_trace'][k],base['model_trace'][k])
        choices = window_choices(metrics,checks,protocol['selection'])
        table.append(dict(task=ordinal,scene=task['scene_name'],object=task['object_name'],
            applicability=base['proposal'],metrics=metrics,checks=checks,choices=choices,
            delta={a:{k:m[k]-reference[k] for k in reference} for a,m in metrics.items()},
            useful_sides=[a for a in choices['accepted'] if geometry_energy(metrics[a])<geometry_energy(reference)],
            hands={a:per_hand_diagnostics(p['states']['edited'],base['states']['edited'],
                   base['snapshot']['edited']) for a,p in payloads.items()},
            sources={a:str(source / ('scene-'+task['scene_name']) / f'task-{ordinal:03d}-{a}.pt') for a in current},
            H='pending_formal_scoring',integrity='IDs, finite, paired history/input and original editor guards verified'))
    result = dict(tasks=len(table),cached_windows=len(rows),
        accepted_offset_tasks=sum(len(t['choices']['accepted'])>1 for t in table),
        geometric_improvement_tasks=[t['task'] for t in table if t['useful_sides']],
        decision='PROCEED_TO_G28_H28' if any(t['useful_sides'] for t in table) else 'STOP_NO_GEOMETRIC_HEADROOM',
        table=table,test_set_development=True,new_HOI_calls=0,new_HSI_calls=0)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'cache_analysis.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    import csv
    with (out/'cache_tasks.csv').open('x') as handle:
        writer=csv.writer(handle)
        writer.writerow(['task','scene','applicable','accepted','useful_sides','G','E_W0','E_G','HS_delta_cm','OS_delta_cm'])
        for t in table:
            g=t['choices']['G'];b=t['metrics']['W0'];m=t['metrics'][g]
            writer.writerow([t['task'],t['scene'],t['applicability']['applicable'],
                '|'.join(t['choices']['accepted']),'|'.join(t['useful_sides']),g,
                geometry_energy(b),geometry_energy(m),m['human_scene_RMS_cm']-b['human_scene_RMS_cm'],
                m['object_scene_RMS_cm']-b['object_scene_RMS_cm']])
    return {k:v for k,v in result.items() if k!='table'}


def verify_single_side_interface(resolved_config, output_dir, device='cuda:0'):
    """Registered task372: original B1, both candidate orders, and G/H pairing."""
    import hydra
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from priors.hoi.models import load_trained_hoi_prior
    from utils import init_model
    from test_infbagel_hosi import seed_everything
    cfg=OmegaConf.load(resolved_config)
    cfg.device=device;cfg.dataset.device=device;cfg.dataset.vis=True
    cfg.dataset.load_object_payload=False
    cfg.dataset.test_scene_name='a3df624b-0917-46e9-ac15-fab766276c72'
    cfg.sampler.pelvis.hoi_adapter.device=device;cfg.sampler.pelvis.hsi_sampler.device=device
    dataset=InfBaGelDataset(**cfg.dataset)
    dataset.obj_rest_verts={k:v.to(device) for k,v in dataset.obj_rest_verts.items()}
    hoi,_=load_trained_hoi_prior(cfg.ckpt_path,torch.device(device),weight_variant=cfg.checkpoint_weight_variant)
    hoi.eval().requires_grad_(False)
    hsi=init_model(OmegaConf.merge(cfg.model.infbagel,{'ckpt':cfg.hsi_ckpt_path}),device=device,eval=True)
    hsi.eval().requires_grad_(False)
    repo=Path(cfg.sampler.pelvis.waypoint_protocol).resolve().parents[2]
    protocol=json.loads(Path(cfg.sampler.pelvis.waypoint_protocol).read_text())
    source=repo/protocol['source_run']/('scene-'+cfg.dataset.test_scene_name)/'task-372-W0.pt'
    cached=torch.load(source,map_location='cpu',weights_only=False)
    keys=inspect.signature(HOSIComposedSampler.p_sample_loop).parameters
    call={k:v for k,v in cached['context'].items() if k in keys}
    call.update(fixed_points=cached['snapshot']['raw_source'][:,:2],
                human_dict={'rest_human_offsets':cached['snapshot']['rest_offsets']},
                obj_rest_verts=dataset.obj_rest_verts,obj_vert_normals={})
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    results={};captures={}
    for mode,policy,order in [('off','off',ORDER),('G','G',ORDER),('G_reverse','G',('W0','Wminus','Wplus')),
                              ('H','H',ORDER),('H_reverse','H',('W0','Wminus','Wplus'))]:
        config=OmegaConf.merge(cfg.sampler.pelvis,dict(waypoint_policy=policy,candidate_order=list(order),
                                                    waypoint_output_dir=str(out/mode)))
        sampler=hydra.utils.instantiate(config);sampler.set_dataset_and_model(dataset,hoi,hsi_model=hsi)
        seed_everything(414);before=random_snapshot(torch.device(device))
        generated=sampler.p_sample_loop(**move_tree(call,device))[0][-1]
        if mode=='off':
            assert torch.equal(generated.cpu(),cached['snapshot']['edited'])
            off_rng=random_snapshot(torch.device(device))
        else:
            after=random_snapshot(torch.device(device))
            assert before[0]==after[0] and np.array_equal(before[1][1],after[1][1])
            assert torch.equal(after[2],off_rng[2]) and torch.equal(after[3],off_rng[3])
            assert sampler.inner_hoi.sample_calls==1 and len(sampler.scene_editor.records)==1
            captures[mode]=sampler.scene_editor.motion_records[-1]['waypoint_candidates']
            results[mode]=sampler.decision_records[-1]
        with (out/(mode+'.pt')).open('xb') as handle:
            torch.save(dict(output=generated.cpu(),snapshots=sampler.scene_editor.motion_records),handle)
    for mode in ('G_reverse','H','H_reverse'):
        for arm in ORDER:
            assert torch.equal(captures['G'][arm]['motion'],captures[mode][arm]['motion'])
    assert results['G']['selected']==results['G_reverse']['selected']
    assert results['H']['selected']==results['H_reverse']['selected']
    assert results['H']['scores']==results['H_reverse']['scores']
    result=dict(task=372,passed=True,off_B1_exact=True,candidate_order_exact=True,
                G_H_candidates_exact=True,score_order_exact=True,single_commit=True,
                random_state_preserved=True,generated_windows=13,decisions=results)
    with (out/'interface.json').open('x') as handle:
        json.dump(result,handle,indent=2,allow_nan=False)
    return dict(passed=True,generated_windows=13)
