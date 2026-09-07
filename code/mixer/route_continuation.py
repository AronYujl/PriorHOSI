"""Same-current native continuations with one persistent signed route."""
import copy
import csv
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from .candidate_selection import move_tree
from .continuation_outcomes import (accumulate_branch_cost, evaluate_state,
                                   exact_tree, generate_branch, write_json)
from .relational_sampling import direct_quality, future_improvement


def smoothstep(u):
    u = np.clip(u, 0., 1.)
    return u**3 * (10. + u * (-15. + 6. * u))


def sustained_route(path, anchor_index, selected_waypoint, lookahead):
    """Deform the original sampled polyline with a fixed quintic offset envelope."""
    path = np.asarray(path, dtype=np.float64)
    j = int(anchor_index)
    if not 0 < j < len(path)-1:
        raise ValueError('registered detour anchor must be interior')
    arc = np.r_[0., np.linalg.norm(np.diff(path, axis=0), axis=1).cumsum()]
    entry = max(0., arc[j]-lookahead)
    leave = min(arc[-1], arc[j]+2*lookahead)
    before = smoothstep((arc-entry)/(arc[j]-entry))
    after = 1.-smoothstep((arc-arc[j])/(leave-arc[j]))
    weight = np.where(arc <= arc[j], before, after)
    offset = np.asarray(selected_waypoint, dtype=np.float64)-path[j]
    if abs(np.linalg.norm(offset)-.1) > 1e-5:
        raise ValueError('cached signed waypoint is not the registered10cm offset')
    result = path + weight[:, None]*offset
    result[0], result[-1], result[j] = path[0], path[-1], selected_waypoint
    return result, dict(original_path_xz=path.tolist(), detour_path_xz=result.tolist(),
        original_arc_m=arc.tolist(), weight=weight.tolist(), anchor_index=j,
        offset_xz_m=offset.tolist(), entry_arc_m=float(entry), anchor_arc_m=float(arc[j]),
        exit_arc_m=float(leave), lookahead_m=float(lookahead),
        original_length_m=float(arc[-1]), detour_length_m=float(np.linalg.norm(np.diff(result, axis=0), axis=1).sum()))


def route_goal(path, mat, lookahead):
    """The existing native nearest-point/lookahead rule, for command auditing."""
    position = mat[0, :3, 3].detach().cpu().numpy()[[0, 2]][None]
    index = int(np.argmin(np.linalg.norm(position-path, axis=1)))
    ahead = math.ceil(len(path)/np.linalg.norm(np.diff(path, axis=0), axis=1).sum()*lookahead)
    chosen = min(index+ahead, len(path)-1)
    point = torch.tensor([path[chosen, 0], 0., path[chosen, 1]], device=mat.device,
                         dtype=torch.float32).reshape(1, 1, 3)
    from utils import transform_points
    local = transform_points(point, torch.inverse(mat)).reshape(1, 3)
    return local, dict(nearest_index=index, target_index=chosen, world=point.flatten().tolist(),
                      local=local.flatten().tolist())


def sweep_samples(path, arc, entry, leave):
    """Sample every affected polyline segment at spacing at most2cm."""
    selected = np.flatnonzero((arc[1:] >= entry) & (arc[:-1] <= leave))
    samples = []
    for i in selected:
        count = max(1, math.ceil(np.linalg.norm(path[i+1]-path[i])/.02))
        samples.append(path[i]+np.arange(count)[:, None]/count*(path[i+1]-path[i]))
    return np.concatenate([*samples, path[selected[-1]+1: selected[-1]+2]])


@torch.no_grad()
def envelope_audit(dataset, cached, record, route):
    """Translate the committed frame13 body/full object; measure geometric exposure."""
    from .waypoint_control import decode_motion
    device = dataset.obj_rest_verts[record['object']].device
    snapshot = cached['snapshot']
    context = move_tree(snapshot['replay_context'], device)
    rest = dataset.obj_rest_verts[record['object']]
    state = decode_motion(snapshot['edited'].to(device), dataset,
                          snapshot['rest_offsets'].to(device), context, rest[None])
    human, obj = state['human'][0, 13], state['object_surface'][0, 13]
    grid = dataset.scene_grid_torch.to(human)
    occupancy = dataset.scene_occ[int(context['scene_flag'].flatten()[0])]
    arc = np.asarray(route['original_arc_m'])
    result = dict(reference='actual committed current frame13; fixed height/orientation',
                  scope='FK24 plus full object mesh; translated fixed-pose envelope only')
    for name in ('original', 'detour'):
        path = np.asarray(route[name+'_path_xz'])
        samples = sweep_samples(path, arc, route['entry_arc_m'], route['exit_arc_m'])
        positions = human.new_tensor(samples)
        displacement = human.new_zeros(len(samples), 3)
        displacement[:, [0, 2]] = positions-human[0, [0, 2]]
        row = dict(route_samples=len(samples), points={})
        for kind, points in [('human', human), ('object', obj)]:
            occupied = outside = total = 0
            # This chunks query memory only; every full-mesh point is measured.
            for delta in displacement.split(16):
                query = points[None]+delta[:, None]
                inside = ((query >= grid[:3]) & (query < grid[3:6])).all(-1)
                voxel = ((query-grid[:3])/((grid[3:6]-grid[:3])/grid[6:])).long()
                valid = voxel[inside]
                occupied += int((occupancy[valid[:, 0], valid[:, 1], valid[:, 2]] != 0).sum())
                outside += int((~inside).sum())
                total += inside.numel()
            row['points'][kind] = dict(vertices_per_pose=len(points), count=total,
                occupied_inside_count=occupied, outside_count=outside,
                occupied_inside_fraction=occupied/total, outside_fraction=outside/total)
        result[name] = row
    return result


class RouteControl:
    """Audit the changed path and the exact model conditions without extra sampling."""
    def __init__(self, cfg, record, cached, pulse, dataset):
        self.cfg, self.record, self.cached = cfg, record, cached
        self.pulse, self.dataset = pulse, dataset
        self.lookahead = float(cfg.get('hsi_lookahead_m', .8))
        self.audit = {'commands': [], 'model_calls_checked': 0}
        self.expected = None

    def prepare(self, path, context):
        decision = json.loads(Path(self.record['decision']['path']).read_text())
        proposal = decision['proposal']
        waypoint = np.asarray(proposal['variants'][self.record['selected']]['waypoint_world'])[[0, 2]]
        j = proposal['path_index']
        # Recover the original W0 waypoint in world coordinates from its cache.
        w0 = torch.load(self.record['candidates']['W0']['path'], map_location='cpu', weights_only=False)
        c = w0['snapshot']['replay_context']
        from utils import transform_points
        world = transform_points(c['pelvis_goal'].reshape(1, 1, 3), c['mat']).flatten().numpy()[[0, 2]]
        error = float(np.max(np.abs(world-path[j])))
        if error > 1e-5:
            raise AssertionError('cached original waypoint and reconstructed route disagree')
        self.original = path
        self.detour, route = sustained_route(path, j, waypoint, self.lookahead)
        self.audit.update(route)
        self.audit['cached_W0_anchor_error_m'] = error
        if not bool(context['is_loco'].all()):
            raise ValueError('registered adopted waypoint requires navigation')
        torch.cuda.synchronize(context['mat'].device)
        started = time.perf_counter()
        self.audit['envelope'] = envelope_audit(self.dataset, self.cached, self.record, self.audit)
        torch.cuda.synchronize(context['mat'].device)
        self.audit['envelope_seconds'] = time.perf_counter()-started
        return self.detour

    def before_step(self, step, mat, fixed):
        self.expected, changed = route_goal(self.detour, mat, self.lookahead)
        _, original = route_goal(self.original, mat, self.lookahead)
        # HOI's pelvis y is explicitly zeroed in its argument preparation.
        self.expected = self.expected.clone()
        self.expected[:, 1] = 0
        self.audit['commands'].append(dict(window=step, original_same_history=original,
            sustained=changed, world_goal_change_m=float(np.linalg.norm(
                np.asarray(changed['world'])-np.asarray(original['world'])))))

    def model_hook(self, module, args):
        goal = args[4][:, :3]
        if not torch.equal(goal, self.expected):
            raise AssertionError('actual HOI model pelvis goal differs from registered route')
        self.audit['model_calls_checked'] += 1

    def after_step(self, item):
        context = item['context']
        first = len(self.audit['commands']) == 1
        row = self.audit['commands'][-1]
        row['actual_context_goal'] = context['pelvis_goal'].flatten().tolist()
        if first:
            old = self.pulse['windows'][1]
            current = {k: v for k, v in context.items() if k != 'pelvis_goal'}
            reference = {k: v for k, v in old['context'].items() if k != 'pelvis_goal'}
            row['first_future_nonroute_context_exact'] = exact_tree(current, reference)
            row['first_future_history_exact'] = torch.equal(item['clean'][:, :2], old['clean'][:, :2])
            if not row['first_future_nonroute_context_exact'] or not row['first_future_history_exact']:
                raise AssertionError('first future non-route conditions/history changed')


def run_route_scene(cfg):
    """One scene lane; native loader, generator, state advance and metrics are shared."""
    from omegaconf import OmegaConf
    from datasets.infbagel import InfBaGelDataset
    from priors.hoi.models import load_trained_hoi_prior
    from utils import init_model
    from test_infbagel_hosi import seed_everything
    if cfg.get('run_id') and subprocess.check_output(['git', 'status', '--porcelain'], text=True).strip():
        raise RuntimeError('reportable route continuation requires clean worktree')
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    started = time.perf_counter()
    protocol_path = Path(cfg.route_continuation.protocol)
    root = protocol_path.resolve().parents[2]
    protocol = json.loads(protocol_path.read_text())
    metrics_protocol = json.loads((root/protocol['source_protocol']).read_text())
    records = json.loads((root/protocol['state_manifest']).read_text())['records']
    selected = {}
    for record in records:
        if record['scene'] == cfg.route_continuation.scene and (
                cfg.route_continuation.state_ids is None or record['state_id'] in cfg.route_continuation.state_ids):
            selected.setdefault(record['state_id'], record)
    out = Path(cfg.hosi_output_dir)
    out.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)), out/'resolved.yaml')
    device = torch.device(cfg.device)
    seed_everything(42)
    dataset = InfBaGelDataset(**cfg.dataset)
    dataset.obj_rest_verts = {k: v.to(device) for k, v in dataset.obj_rest_verts.items()}
    hoi, _ = load_trained_hoi_prior(cfg.ckpt_path, device, weight_variant=cfg.checkpoint_weight_variant)
    hoi.eval().requires_grad_(False)
    hsi = init_model(OmegaConf.merge(cfg.model.infbagel, {'ckpt': cfg.hsi_ckpt_path}), device=device, eval=True)
    hsi.eval().requires_grad_(False)
    scene = str(cfg.route_continuation.scene)
    lookup = {r['canonical_ordinal']: r for r in json.loads((root/
        'experiments/tasks/conditional_repair_native_development_s42_20260907.json').read_text())['tasks']}
    native = json.loads((root/'data/hosi_test/data'/(scene+'.json')).read_text())
    key = scene+'_sdf'
    sdf_root = root/'data/hosi_test/Scene_sdf'
    sdf = {key: np.load(sdf_root/(key+'.npy'))}
    info = {key: json.loads((sdf_root/(key+'_info.json')).read_text())}
    torch.cuda.reset_peak_memory_stats(device)
    rows, smpl_cache = [], {}
    for state_id, source_record in selected.items():
        record = copy.deepcopy(source_record)
        for asset in [record['episode'], record['decision'], *record['candidates'].values()]:
            asset['path'] = str(root/asset['path'])
        task = native[lookup[record['task']]['test_idx']]
        dest = out/state_id
        dest.mkdir()
        (dest/'new').mkdir()
        old_dir = next((root/protocol['source_run']/'lanes').glob('*/'+state_id))
        arm = record['selected']
        branches = {a: torch.load(old_dir/(a+'.pt'), map_location='cpu', weights_only=False) for a in ('W0', arm)}
        for a in ('W0', arm):
            (dest/(a+'.pt')).symlink_to((old_dir/(a+'.pt')).resolve())
        cached = torch.load(record['candidates'][arm]['path'], map_location='cpu', weights_only=False)
        episode = torch.load(record['episode']['path'], map_location='cpu', weights_only=False)
        control = RouteControl(cfg, record, cached, branches[arm], dataset)
        hook = hoi.register_forward_pre_hook(control.model_hook)
        try:
            candidate = generate_branch(cfg, dataset, hoi, hsi, record, episode, cached, task,
                                        arm, dest/'new', protocol, route_control=control)
        finally:
            hook.remove()
        (dest/'sustained.pt').symlink_to((dest/'new'/(arm+'.pt')).resolve())
        write_json(dest/'route.json', control.audit)
        if candidate['failure']:
            write_json(dest/'generation_failure.json', candidate['failure'])
            raise RuntimeError('route generation failed; original branch retained')
        if not torch.equal(candidate['windows'][0]['clean'], branches[arm]['windows'][0]['clean']):
            raise AssertionError('cached current action changed')
        branches['sustained'] = candidate
        record['cached_guard']['sustained'] = record['cached_guard'][arm]
        torch.cuda.synchronize(device)
        began = time.perf_counter()
        result, arrays = evaluate_state(cfg, dataset, record, task, branches, smpl_cache, sdf, info, metrics_protocol)
        torch.cuda.synchronize(device)
        result['evaluation_seconds'] = time.perf_counter()-began
        with (dest/'tracks.pt').open('xb') as f:
            torch.save(arrays, f)
        result['current_exact'] = True
        old_metrics = json.loads((old_dir/'outcomes.json').read_text())['branches']
        baseline_checks = []
        for a in ('W0', arm):
            for horizon, value in result['branches'][a]['slices'].items():
                previous_value = old_metrics[a]['slices'][horizon]
                for field in METRICS:
                    baseline_checks.append(value[field] == previous_value[field])
        result['baseline_scalar_checks'] = len(baseline_checks)
        result['baseline_scalars_exact'] = all(baseline_checks)
        if not result['baseline_scalars_exact']:
            write_json(dest/'baseline_comparison_failure.json', result)
            raise AssertionError('cached reference native metrics differ from sealed evaluation')
        result['route'] = control.audit
        result['source_records'] = [r['record_id'] for r in records if r['state_id'] == state_id]
        write_json(dest/'outcomes.json', result)
        rows.append(result)
        print(json.dumps(dict(state=state_id, task=record['task'], new_windows=candidate['costs']['generated_windows'],
                              commands=[r['world_goal_change_m'] for r in control.audit['commands']])), flush=True)
    write_json(out/'lane.json', dict(commit_at_start=commit, commit_at_completion=subprocess.check_output(
        ['git', 'rev-parse', 'HEAD'], text=True).strip(), scene=scene, states=list(selected),
        completed_states=len(rows), technical_errors=0, wall_seconds=time.perf_counter()-started,
        new_generated_windows=sum(r['branches']['sustained']['costs']['generated_windows'] for r in rows)))


METRICS = ('native_surface_HS_s_mean', 'native_surface_OS_s_mean', 'native_fragment_FS_cm',
           'contact_any_5cm', 'support_speed_m_per_s', 'world_joint_speed_m_per_s',
           'human_goal_error_cm', 'object_goal_error_3D_cm')


def route_verdict(state, limits, motion_mm):
    """The prospectively fixed joint candidate and future improvement definition."""
    branches = state['branches']
    new, pulse, base = branches['sustained'], branches[state['source_selected']], branches['W0']
    failures = {name: {h: direct_quality(m, ref['slices'][h], limits) for h, m in new['slices'].items()}
                for name, ref in [('pulse', pulse), ('W0', base)]}
    n, b = new['slices']['cumulative'], base['slices']['cumulative']
    benefit = n['native_surface_HS_s_mean'] < b['native_surface_HS_s_mean'] and n['native_surface_OS_s_mean'] <= b['native_surface_OS_s_mean']
    terminal_loss = any(ref['terminal'] and ref['terminal']['completed'] and not new['terminal']['completed']
                        for ref in (pulse, base))
    command = max((c['world_goal_change_m'] for c in state['route']['commands']), default=0.)
    realized = command >= .01 and motion_mm >= 1.
    gains = future_improvement(new, pulse, limits)
    joint = bool(benefit and new['diagnostics']['scene_proxy_protected']
                 and not any(f for rows in failures.values() for f in rows.values())
                 and not terminal_loss and any(state['contact_reference']['active_hands']))
    return dict(state_id=state['state_id'], task=state['task'], scene=state['scene'],
        failures=failures, scene_benefit=benefit, joint_candidate=joint, future_improvements=gains,
        qualifying=bool(joint and gains and realized), realized=realized,
        max_command_change_m=command, future_root_displacement_mm=motion_mm,
        direct_regression_vs_pulse=any(failures['pulse'].values()), terminal_regression=bool(terminal_loss))


def grouped_means(rows, fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[k] for k in fields)].append(row)
    result = []
    for identity, group in groups.items():
        row = dict(zip(fields, identity))
        row['count'] = len(group)
        for k in (*METRICS, *(k for k in group[0] if k.startswith('hand'))):
            values = [r[k] for r in group if r[k] is not None]
            row[k] = sum(values)/len(values) if values else None
        result.append(row)
    return result


def summarize_routes(run_root):
    root = Path(run_root)
    protocol = json.loads((root/'protocol.json').read_text())
    limits = json.loads(Path(protocol['source_protocol']).read_text())['labels']
    attribution = json.loads((Path(protocol['source_run'])/'analysis/frame_ownership_attribution.json').read_text())
    delayed = {r['state'] for r in attribution['rows'] if r['delayed_component_group']
               and not r['segments']['current_cached_available']['exceeds_registered_hand_margin']}
    out = root/'analysis'
    out.mkdir(exist_ok=False)
    details, table, checks, envelopes = [], [], [], []
    costs = Counter()
    for path in sorted(root.glob('lanes/*/state-*/outcomes.json')):
        state = json.loads(path.read_text())
        new = state['branches']['sustained']
        accumulate_branch_cost(costs, new['costs'])
        costs['evaluation_seconds'] += state['evaluation_seconds']
        tracks = torch.load(path.parent/'tracks.pt', map_location='cpu', weights_only=False)
        current_end = new['slices']['current']['coarse_range'][1]
        delta = (tracks['sustained']['coarse_fk_human'][current_end:, 0]
                 - tracks[state['source_selected']]['coarse_fk_human'][current_end:, 0])
        motion_mm = float(delta.norm(dim=-1).mean()*1000)
        verdict = route_verdict(state, limits, motion_mm)
        verdict['delayed_new_subgroup'] = state['state_id'] in delayed
        details.append(verdict)
        checks.append(dict(state_id=state['state_id'], current_exact=state['current_exact'],
            first_nonroute_exact=state['route']['commands'][0]['first_future_nonroute_context_exact'],
            first_history_exact=state['route']['commands'][0]['first_future_history_exact'],
            model_calls_checked=state['route']['model_calls_checked']))
        envelopes.append(dict(state_id=state['state_id'], task=state['task'], scene=state['scene'],
                              **state['route']['envelope']))
        for comparator, reference in [('pulse', state['branches'][state['source_selected']]), ('W0', state['branches']['W0'])]:
            for horizon, metrics in new['slices'].items():
                ref = reference['slices'][horizon]
                row = dict(state_id=state['state_id'], task=state['task'], scene=state['scene'],
                           comparator=comparator, horizon=horizon)
                for key in METRICS:
                    row[key] = metrics[key]-ref[key] if metrics[key] is not None and ref[key] is not None else None
                for i, (hand, base) in enumerate(zip(metrics['hands'], ref['hands'])):
                    for key in ('surface_mean_m', 'coverage_5cm', 'anchor_vs_W0_m'):
                        row[f'hand{i}_{key}'] = hand[key]-base[key] if hand[key] is not None and base[key] is not None else None
                table.append(row)
    qualifying = sorted({r['task'] for r in details if r['qualifying']})
    regressed = sorted({r['task'] for r in details if r['direct_regression_vs_pulse']})
    upgrade = len(qualifying) >= 2 and len(regressed) <= len(qualifying)
    task_means = grouped_means(table, ('task', 'scene', 'comparator', 'horizon'))
    scene_means = grouped_means(task_means, ('scene', 'comparator', 'horizon'))
    summary = dict(states=len(details), tasks=len({r['task'] for r in details}), scenes=len({r['scene'] for r in details}),
        delayed_new_states=len(delayed), realized_states=sum(r['realized'] for r in details),
        scene_benefit_states=sum(r['scene_benefit'] for r in details), joint_states=sum(r['joint_candidate'] for r in details),
        qualifying_tasks=qualifying, regression_tasks=regressed, upgrade_discussion=upgrade,
        decision='DISCUSS_FIXED_FULL_POLICY' if upgrade else 'DO_NOT_UPGRADE_SUSTAINED_ROUTE',
        costs=dict(costs), checks=checks, state_details=details, task_means=task_means, scene_means=scene_means,
        state_files=[str(p) for p in sorted(root.glob('lanes/*/state-*/outcomes.json'))],
        delayed_subgroup_details=[r for r in details if r['delayed_new_subgroup']])
    with (out/'paired_components.csv').open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)
    for name, data in [('summary', summary), ('state_details', details), ('task_means', task_means),
                       ('scene_means', scene_means), ('envelope_audits', envelopes), ('pairing_checks', checks)]:
        write_json(out/(name+'.json'), data)
    return summary


def render_route_summary(run_root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(run_root)
    summary = json.loads((root/'analysis/summary.json').read_text())
    states = [json.loads(Path(p).read_text()) for p in summary['state_files']]
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    index = np.arange(len(states))
    labels = [s['state_id'].split('-')[-1] for s in states]
    axes[0, 0].bar(index, [max(c['world_goal_change_m'] for c in s['route']['commands'])*100 for s in states], color='#3572a5')
    axes[0, 0].set(title='Executed future goal change at same history', ylabel='Maximum change (cm)')
    for key, label, color in [('native_surface_HS_s_mean', 'HS', '#28786b'), ('native_surface_OS_s_mean', 'OS', '#d48629')]:
        values = [s['branches']['sustained']['slices']['cumulative'][key]-s['branches'][s['source_selected']]['slices']['cumulative'][key] for s in states]
        axes[0, 1].plot(index, values, 'o-', label=label, color=color)
    axes[0, 1].axhline(0, color='gray', lw=1)
    axes[0, 1].set(title='Sustained minus pulse: cumulative native surfaces', ylabel='Native depth-sum units')
    axes[0, 1].legend()
    for s, x in zip(states, index):
        new, old = s['branches']['sustained'], s['branches'][s['source_selected']]
        for horizon, marker in [('next_1', 'o'), ('next_2', '^')]:
            differences = [100*(h['coverage_5cm']-b['coverage_5cm']) for h, b in zip(new['slices'][horizon]['hands'], old['slices'][horizon]['hands']) if b['fixed_active']]
            if differences:
                axes[1, 0].scatter(x, min(differences), marker=marker, color='#a64242' if min(differences)<0 else '#28786b')
    axes[1, 0].axhline(0, color='gray', lw=1)
    axes[1, 0].set(title='Worst active-hand coverage change per future slice', ylabel='Percentage points; circle next1 / triangle next2')
    for kind, label in [('human', 'FK body24'), ('object', 'Full object mesh')]:
        values = [100*(s['route']['envelope']['detour']['points'][kind]['occupied_inside_fraction']-
                       s['route']['envelope']['original']['points'][kind]['occupied_inside_fraction']) for s in states]
        axes[1, 1].plot(index, values, 'o-', label=label)
    axes[1, 1].axhline(0, color='gray', lw=1)
    axes[1, 1].set(title='Fixed-pose swept occupancy: detour minus original', ylabel='Percentage points; diagnostic proxy')
    axes[1, 1].legend()
    for ax in axes.flatten():
        ax.set_xticks(index, labels, rotation=90)
        ax.set_xlabel('Registered state (nested within12 tasks /4 scenes)')
    fig.suptitle('Same-current sustained-route continuation — seed42,22 states')
    fig.tight_layout()
    target = root/'visualizations/route-summary.png'
    fig.savefig(target, dpi=150)
    plt.close(fig)
    return str(target)
