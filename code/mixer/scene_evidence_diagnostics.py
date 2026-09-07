"""Fixed-source, paired-noise observations; probes never enter rollout history."""
import time

import torch

from .body_groups import ROTATION_GROUPS
from .input_views import masked_object_arguments
from .scene_evidence import (EXPLICIT_TERMS, SceneEvidenceTeacher, domain_accepts,
                             domain_state, editable_mask, epsilon_from_x0, local_armijo)
from .scene_views import TEACHER_VIEWS, temporal_environment


def values(items):
    return {key: value.detach().cpu().tolist() for key, value in items.items()}


def alignment(first, second):
    a, b = first.double().flatten(), second.double().flatten()
    na, nb = a.norm(), b.norm()
    dot = a.dot(b)
    finite = torch.isfinite(torch.stack((na, nb, dot))).all()
    if not finite:
        raise FloatingPointError('nonfinite diagnostic alignment')
    zero = bool((na == 0) | (nb == 0))
    return dict(dot=float(dot), first_norm=float(na), second_norm=float(nb),
                cosine=None if zero else float(dot / (na * nb)),
                cosine_reason='zero_norm' if zero else None)


def group_norms(gradient):
    groups = {'translation': gradient[..., :3], 'yaw': gradient[..., 3:4]}
    local = gradient[..., 4:].reshape(*gradient.shape[:2], 21, 3)
    for name, joints in ROTATION_GROUPS.items():
        indices = [j - 1 for j in joints if j > 0]
        groups[name] = local[..., indices, :]
    return {name: float(value.double().norm()) for name, value in groups.items()}


def physical_rms(state, origin):
    human = (state['human'][:, 2:] - origin['human'][:, 2:]).square().sum(-1).mean()
    obj = (state['object_surface'][:, 2:] - origin['object_surface'][:, 2:]).square().sum(-1).mean()
    return ((human + obj) / 2).sqrt()


@torch.no_grad()
def physical_probe(geometry, objective, parameters, gradient, target, grid):
    """Match physical amplitude on the unchanged descent ray through tanh decode."""
    origin = geometry.decode(parameters)
    origin_domain = domain_state(origin, grid)
    norm = gradient.double().norm()
    if not torch.isfinite(norm):
        raise FloatingPointError('nonfinite probe direction')
    lo, hi = 0., 1.
    if norm == 0:
        hi = 0.
        reason = 'zero_direction'
    else:
        for _ in range(21):
            state = geometry.decode(parameters - hi * gradient)
            if physical_rms(state, origin) >= target or hi == 2**20:
                break
            hi *= 2
        attained = physical_rms(state, origin) >= target
        reason = 'matched' if attained else 'bounded_or_weak_direction'
        if attained:
            for _ in range(20):
                mid = (lo + hi) / 2
                state = geometry.decode(parameters - mid * gradient)
                if physical_rms(state, origin) < target:
                    lo = mid
                else:
                    hi = mid
    proposal = parameters - hi * gradient
    state = geometry.decode(proposal)
    terms, metrics = objective.evaluate(state)
    valid = bool(domain_accepts(origin_domain, domain_state(state, grid)).all())
    delta_h = state['human'][:, 2:] - origin['human'][:, 2:]
    movements = {name: float(delta_h[..., list(joints), :].square().sum(-1).mean().sqrt())
                 if joints else 0. for name, joints in ROTATION_GROUPS.items()}
    return dict(target_rms_m=target, actual_rms_m=float(physical_rms(state, origin)),
                ray_scalar=hi, reason=reason, domain_admissible=valid,
                terms=values({key: terms[key] for key in EXPLICIT_TERMS}),
                metrics=values(metrics), human_group_rms_m=movements,
                object_translation_rms_m=float((state['object_translation_world'][:, 2:] -
                    origin['object_translation_world'][:, 2:]).square().sum(-1).mean().sqrt()),
                yaw_rms_rad=float(state['yaw'][:, 2:].square().mean().sqrt())), proposal


def _synchronize(source):
    if source.is_cuda:
        torch.cuda.synchronize(source.device)


@torch.no_grad()
def run_fixed_source_views(editor, teacher, geometry, objective, parameters, reference, seed):
    """One shared model/query bundle per draw/level, all views at the same source."""
    sampler, context = teacher.sampler, teacher.context
    source = geometry.base
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(source.device).clone() if source.is_cuda else None
    scene_storage = sampler.hsi_sampler.dataset.scene_occ.clone()
    context_before = {k: v.clone() for k, v in context.items() if torch.is_tensor(v)}
    mask = editable_mask(source)
    denominator = mask.flatten(1).sum(1)
    grid = sampler.dataset.scene_grid_torch.to(source)
    origin = geometry.decode(parameters)
    with torch.enable_grad():
        p = parameters.detach().requires_grad_(True)
        motion = geometry.encode(geometry.decode(p), source[:, :2])
        terms, _ = objective.evaluate(geometry.decode(p))
        term_gradients = {key: torch.autograd.grad(terms[key].sum(), p, retain_graph=True)[0].detach()
                          for key in EXPLICIT_TERMS}
    term_gradients['explicit'] = sum(term_gradients.values())
    records, motions, directions = [], [], {}
    teacher_seconds = probe_seconds = 0.
    hoi_calls = hsi_calls = 0
    n_draws = editor.diagnostics['noise_draws']
    for draw in range(n_draws):
        paired = SceneEvidenceTeacher(sampler, teacher.arguments, teacher.local_bps, context,
            reference, int(seed) + 1000003 * draw, teacher.beta, teacher.lambda_dp)
        for level in editor.noise_levels:
            _synchronize(source)
            started = time.perf_counter()
            _, legacy_record = paired.query(reference, level, capture=True)
            if any(value != 0. for value in legacy_record['hoi_reference_rms']):
                raise AssertionError('HOI reference failed to cancel at reconstructed source')
            bundle = paired.query_bundle
            legacy = bundle['common']
            environment, query = temporal_environment(legacy, reference, context, sampler.hsi_sampler)
            mismatch, wrong_query = temporal_environment(legacy, reference, context,
                sampler.hsi_sampler, editor.diagnostics['mismatch_local_x_m'])
            predictions = [(bundle['cond'], bundle['base'])]
            for common in (environment, mismatch):
                predictions.append(sampler._hsi_predict_pair(bundle['view'], masked_object_arguments(common)))
                paired.hsi_calls += 2
            _synchronize(source)
            teacher_seconds += time.perf_counter() - started
            static_equal = all(torch.equal(legacy[i], common[i])
                               for common in (environment, mismatch) for i in (0, 16))
            static_equal &= all(torch.equal(legacy[15][:1], c[15][:1]) for c in (environment, mismatch))
            base_equal = all(torch.equal(bundle['base'], pair[1]) for pair in predictions)
            if not static_equal or not base_equal:
                raise AssertionError('temporal view changed static input or base prediction')
            alpha, sigma = bundle['alpha'], bundle['sigma']
            started = time.perf_counter()
            for view_name, common, (cond, base) in zip(TEACHER_VIEWS, (legacy, environment, mismatch), predictions):
                raw = cond - base
                delta = torch.zeros_like(reference)
                delta[..., :216] = (epsilon_from_x0(bundle['view'], cond, alpha, sigma) -
                                   epsilon_from_x0(bundle['view'], base, alpha, sigma))[..., :216]
                unscaled = alpha / sigma * delta * mask
                scaled = teacher.lambda_dp * unscaled
                with torch.enable_grad():
                    p = parameters.detach().requires_grad_(True)
                    motion = geometry.encode(geometry.decode(p), source[:, :2])
                    g_unit, = torch.autograd.grad(((unscaled * motion).flatten(1).sum(1) / denominator).sum(), p)
                g_scaled = teacher.lambda_dp * g_unit.detach()
                directions[draw, level, view_name] = g_scaled
                gradients = dict(term_gradients, hoi_reference=torch.zeros_like(parameters),
                                 hsi_unscaled=g_unit, hsi_scaled=g_scaled)
                block_rms = lambda x: dict(position=float(x[:, 2:, :84].square().mean().sqrt()),
                                          rotation=float(x[:, 2:, 84:216].square().mean().sqrt()))
                probes = []
                for target in editor.diagnostics['physical_probe_rms_m']:
                    probe, proposal = physical_probe(geometry, objective, parameters, g_scaled, target, grid)
                    probes.append(probe)
                    if level == editor.noise_levels[0] and draw == 0:
                        state = geometry.decode(proposal)
                        motions.append(dict(view=view_name, target_rms_m=target,
                            human_world=state['human'].cpu(), object_surface_world=state['object_surface'].cpu(),
                            parameters=proposal.cpu()))
                record = dict(level=level, draw=draw, view=view_name,
                    candidate_kind='reconstructed_source', alpha=float(alpha), sigma=float(sigma),
                    raw_x0_rms=block_rms(raw), epsilon_delta_rms=block_rms(delta),
                    weighted_unscaled_rms=block_rms(unscaled), weighted_scaled_rms=block_rms(scaled),
                    hoi_reference_rms=legacy_record['hoi_reference_rms'],
                    static_conditions_equal=static_equal, base_prediction_equal=base_equal,
                    parameter_gradient_norms={k: [float(v.double().norm())] for k, v in gradients.items()},
                    parameter_group_norms={k: group_norms(v) for k, v in gradients.items()},
                    per_term_alignment={k: alignment(g_scaled, v) for k, v in term_gradients.items()},
                    temporal_occupied_fraction=common[15][1:].float().flatten(1).mean(1).cpu().tolist(),
                    overlay_added_fraction=(legacy[15][1:] != environment[15][1:]).float().flatten(1).mean(1).cpu().tolist(),
                    physical_step_probes=probes,
                    solver=dict(accepted=[False], trials=[], reason=['fixed_source_probe']))
                records.append(record)
            _synchronize(source)
            probe_seconds += time.perf_counter() - started
        hoi_calls += paired.hoi_calls
        hsi_calls += paired.hsi_calls
    # Same-source lambda0 short editor, kept wholly outside actual history.
    zero_teacher = SceneEvidenceTeacher(sampler, teacher.arguments, teacher.local_bps,
                                       context, reference, seed, teacher.beta, 0.)
    _synchronize(source)
    started = time.perf_counter()
    edited_parameters = parameters.clone()
    short_trace = []
    for level in editor.noise_levels:
        current = geometry.decode(edited_parameters)
        candidate = geometry.encode(current, source[:, :2])
        direction, _ = zero_teacher.query(candidate, level)
        anchor = edited_parameters
        def evaluate(proposal):
            state = geometry.decode(proposal)
            x = geometry.encode(state, source[:, :2])
            terms, _ = objective.evaluate(state)
            return ((direction * mask * (x-candidate)).flatten(1).sum(1) / denominator
                    + sum(terms[k] for k in EXPLICIT_TERMS)
                    + .5 * editor.prox_weight * (proposal-anchor).square().flatten(1).sum(1))
        current_domain = domain_state(current, grid)
        edited_parameters, trace = local_armijo(edited_parameters, evaluate,
            lambda a: domain_accepts(current_domain, domain_state(geometry.decode(a), grid)),
            editor.initial_step, editor.shrink, editor.c1, editor.max_backtracks)
        short_trace.append(trace)
    short_state = geometry.decode(edited_parameters)
    short_terms, short_metrics = objective.evaluate(short_state)
    _synchronize(source)
    short_seconds = time.perf_counter() - started
    motions.append(dict(view='lambda0_short_edit', parameters=edited_parameters.cpu(),
        human_world=short_state['human'].cpu(), object_surface_world=short_state['object_surface'].cpu()))
    hoi_calls += zero_teacher.hoi_calls
    teacher.hoi_calls, teacher.hsi_calls = hoi_calls, hsi_calls
    editor.diagnostic_motions = motions
    pairs = [(TEACHER_VIEWS[1], TEACHER_VIEWS[0]), (TEACHER_VIEWS[1], TEACHER_VIEWS[2])]
    cross_view = [dict(level=k, draw=d, first=a, second=b,
                      **alignment(directions[d, k, a], directions[d, k, b]))
                  for k in editor.noise_levels for d in range(n_draws) for a, b in pairs]
    stability = [dict(level=k, view=v, first_draw=a, second_draw=b,
                      **alignment(directions[a, k, v], directions[b, k, v]))
                 for k in editor.noise_levels for v in TEACHER_VIEWS
                 for a in range(n_draws) for b in range(a+1, n_draws)]
    ambient = torch.equal(cpu_rng, torch.get_rng_state())
    if source.is_cuda:
        ambient &= torch.equal(cuda_rng, torch.cuda.get_rng_state(source.device))
    storage_equal = torch.equal(scene_storage, sampler.hsi_sampler.dataset.scene_occ)
    context_equal = all(torch.equal(v, context[k]) for k, v in context_before.items())
    if not ambient or not storage_equal or not context_equal:
        raise AssertionError('fixed-source diagnostic mutated RNG, scene storage or context')
    return dict(iterations=records, teacher_seconds=teacher_seconds, probe_seconds=probe_seconds,
        lambda0_short_seconds=short_seconds,
        extra_environment_grid_queries=n_draws * len(editor.noise_levels) * 2 * sampler.hsi_sampler.temp_voxel_num,
        ambient_rng_preserved=ambient, scene_storage_unchanged=storage_equal, context_unchanged=context_equal,
        native_quality_evaluated=False, source_contact_count=int(objective.contact.sum()),
        temporal_query_frames=query['frames'], query_center_world=query['centers'].cpu().tolist(),
        environment_outside_fraction=query['outside_fraction'].cpu().tolist(),
        mismatch_center_world=wrong_query['centers'].cpu().tolist(),
        mismatch_outside_fraction=wrong_query['outside_fraction'].cpu().tolist(),
        added_overlay_fraction=(legacy[15][1:] != environment[15][1:]).float().flatten(1).mean(1).cpu().tolist(),
        goal_context={k: context[k].cpu().tolist() for k in
            ('pelvis_goal', 'object_goal', 'scene_goal', 'is_object', 'is_loco', 'need_pelvis_dir')},
        cross_view_alignment=cross_view, draw_stability=stability,
        lambda0_short_edit=dict(terms=values({k: short_terms[k] for k in EXPLICIT_TERMS}),
            metrics=values(short_metrics), physical_rms_m=float(physical_rms(short_state, origin)), trace=short_trace))


@torch.no_grad()
def run_relation_compatible(editor, teacher, geometry, objective, parameters, reference, seed):
    """Paired DP rays and four same-budget editors on an immutable source."""
    from .relation_projection import contact_jacobian, project_dp_gradient
    from .scene_evidence import SceneEvidenceEditor
    sampler, context, source = teacher.sampler, teacher.context, geometry.base
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state(source.device).clone() if source.is_cuda else None
    storage = sampler.hsi_sampler.dataset.scene_occ.clone()
    frozen_context = {k: v.clone() for k, v in context.items() if torch.is_tensor(v)}
    frozen_anchor, frozen_contact = objective.hand_anchor.clone(), objective.contact.clone()
    mask = editable_mask(source)
    denominator = mask.flatten(1).sum(1)
    grid = sampler.dataset.scene_grid_torch.to(source)
    origin = geometry.decode(parameters)
    _synchronize(source)
    started = time.perf_counter()
    jacobian, residual = contact_jacobian(objective, parameters)
    _synchronize(source)
    jacobian_seconds = time.perf_counter() - started
    with torch.enable_grad():
        p = parameters.detach().requires_grad_(True)
        terms, _ = objective.evaluate(geometry.decode(p))
        term_gradients = {key: torch.autograd.grad(terms[key].sum(), p, retain_graph=True)[0].detach()
                          for key in EXPLICIT_TERMS}
    records, motions, directions = [], [], {}
    teacher_seconds = probe_seconds = 0.
    hoi_calls = hsi_calls = 0
    for draw in range(editor.diagnostics['noise_draws']):
        paired = SceneEvidenceTeacher(sampler, teacher.arguments, teacher.local_bps, context,
            reference, int(seed) + 1000003 * draw, teacher.beta, teacher.lambda_dp,
            'environment_only_temporal')
        for level in editor.noise_levels:
            _synchronize(source)
            started = time.perf_counter()
            _, teacher_record = paired.query(reference, level, capture=True)
            bundle = paired.query_bundle
            correct = bundle['common']
            mismatch, wrong_query = temporal_environment(correct, reference, context,
                sampler.hsi_sampler, editor.diagnostics['mismatch_local_x_m'])
            wrong_cond, wrong_base = sampler._hsi_predict_pair(bundle['view'], masked_object_arguments(mismatch))
            paired.hsi_calls += 2
            static_equal = (all(torch.equal(correct[i], mismatch[i]) for i in (0, 16))
                            and torch.equal(correct[15][:1], mismatch[15][:1]))
            base_equal = torch.equal(bundle['base'], wrong_base)
            if not static_equal or not base_equal or any(teacher_record['hoi_reference_rms']):
                raise AssertionError('relation probe lost static/base/reference pairing')
            _synchronize(source)
            teacher_seconds += time.perf_counter() - started
            started = time.perf_counter()
            alpha, sigma = bundle['alpha'], bundle['sigma']
            gradients = {}
            for group, cond, base in (('G1', bundle['cond'], bundle['base']), ('G3', wrong_cond, wrong_base)):
                delta = torch.zeros_like(reference)
                delta[..., :216] = (epsilon_from_x0(bundle['view'], cond, alpha, sigma) -
                                   epsilon_from_x0(bundle['view'], base, alpha, sigma))[..., :216]
                component = alpha / sigma * delta * mask
                with torch.enable_grad():
                    p = parameters.detach().requires_grad_(True)
                    x = geometry.encode(geometry.decode(p), source[:, :2])
                    gradients[group], = torch.autograd.grad((component*x).sum() / denominator.sum(), p)
            gradients['G2'], projection_b = project_dp_gradient(objective, parameters, gradients['G1'], jacobian)
            raw_c = gradients['G3']
            gradients['G3'], projection_c = project_dp_gradient(objective, parameters, raw_c, jacobian)
            for group in ('G1', 'G2', 'G3'):
                gradient = gradients[group].detach()
                projection = {'G1': None, 'G2': projection_b, 'G3': projection_c}[group]
                weak = projection is not None and (projection['r_keep'] is not None
                                                   and projection['r_keep'] < editor.diagnostics['minimum_keep_ratio'])
                probes = []
                for target in editor.diagnostics['physical_probe_rms_m']:
                    probe, proposal = physical_probe(geometry, objective, parameters,
                        torch.zeros_like(gradient) if weak else editor.lambda_dp * gradient, target, grid)
                    if weak:
                        probe['reason'] = 'weak_projected_direction'
                    probes.append(probe)
                    if level == editor.noise_levels[0] and draw == 0:
                        state = geometry.decode(proposal)
                        motions.append(dict(view=group, target_rms_m=target,
                            human_world=state['human'].cpu(), object_surface_world=state['object_surface'].cpu(),
                            parameters=proposal.cpu()))
                directions[draw, level, group] = gradient
                records.append(dict(group=group, view=group, level=level, draw=draw,
                    pairing_id=f'{seed}:{draw}:{level}', candidate_kind='reconstructed_source',
                    alpha=float(alpha), sigma=float(sigma), projection=projection,
                    static_conditions_equal=static_equal, base_prediction_equal=base_equal,
                    hoi_reference_rms=teacher_record['hoi_reference_rms'],
                    gradient_semantics='unscaled DP gradient; actual ray is -lambda*g',
                    gradient_norm=float(gradient.double().norm()), parameter_group_norms=group_norms(gradient),
                    raw_parameter_group_norms=group_norms(raw_c if group == 'G3' else gradients['G1']),
                    per_term_gradient_norms={k: float(v.double().norm()) for k, v in term_gradients.items()},
                    per_term_descent_alignment={k: alignment(-gradient, v) for k, v in term_gradients.items()},
                    physical_step_probes=probes))
            _synchronize(source)
            probe_seconds += time.perf_counter() - started
        hoi_calls += paired.hoi_calls
        hsi_calls += paired.hsi_calls
    short_edits = {}
    for group, coefficient, projected, shift in (
            ('G0', 0., False, 0.), ('G1', editor.lambda_dp, False, 0.),
            ('G2', editor.lambda_dp, True, 0.),
            ('G3', editor.lambda_dp, True, editor.diagnostics['mismatch_local_x_m'])):
        short = SceneEvidenceEditor(enabled=True, mode='edit', lambda_dp=coefficient,
            hoi_reference_weight=editor.hoi_reference_weight, noise_levels=editor.noise_levels,
            initial_step=editor.initial_step, shrink=editor.shrink, c1=editor.c1,
            max_backtracks=editor.max_backtracks, prox_weight=editor.prox_weight,
            teacher_scene_view='environment_only_temporal', dp_proxy='parameter',
            relation_projection=projected, mismatch_local_x_m=shift)
        result = short.edit(sampler, source, teacher.arguments, teacher.local_bps,
                            context, geometry.offsets, seed)
        snapshot, record = short.motion_records[0], short.records[0]
        state = geometry.decode(snapshot['parameters'].to(source))
        short_edits[group] = dict(record=record, physical_rms_m=float(physical_rms(state, origin)))
        motions.append(dict(view=group+'_short_edit', parameters=snapshot['parameters'],
            encoded=result.cpu(), human_world=state['human'].cpu(), object_surface_world=state['object_surface'].cpu()))
        hoi_calls += record['hoi_teacher_calls']
        hsi_calls += record['hsi_teacher_calls']
    ambient = torch.equal(cpu_rng, torch.get_rng_state())
    if source.is_cuda:
        ambient &= torch.equal(cuda_rng, torch.cuda.get_rng_state(source.device))
    storage_equal = torch.equal(storage, sampler.hsi_sampler.dataset.scene_occ)
    context_equal = all(torch.equal(v, context[k]) for k, v in frozen_context.items())
    anchor_equal = torch.equal(frozen_anchor, objective.hand_anchor) and torch.equal(frozen_contact, objective.contact)
    if not (ambient and storage_equal and context_equal and anchor_equal):
        raise AssertionError('relation diagnostic changed RNG/context/storage/source anchors')
    teacher.hoi_calls, teacher.hsi_calls = hoi_calls, hsi_calls
    editor.diagnostic_motions = motions
    return dict(iterations=records, teacher_seconds=teacher_seconds, probe_seconds=probe_seconds,
        jacobian_seconds=jacobian_seconds, source_residual_norm=float(residual.double().norm()),
        ambient_rng_preserved=ambient, scene_storage_unchanged=storage_equal,
        context_unchanged=context_equal, source_anchors_unchanged=anchor_equal,
        native_quality_evaluated=False, source_contact_count=int(objective.contact.sum()),
        mismatch_outside_fraction=wrong_query['outside_fraction'].cpu().tolist(),
        short_edits=short_edits,
        cross_view_alignment=[dict(level=k, draw=d, first=a, second=b,
            **alignment(directions[d,k,a], directions[d,k,b]))
            for k in editor.noise_levels for d in range(editor.diagnostics['noise_draws'])
            for a,b in (('G2','G1'),('G2','G3'))],
        draw_stability=[dict(level=k, view=v, first_draw=a, second_draw=b,
            **alignment(directions[a,k,v], directions[b,k,v]))
            for k in editor.noise_levels for v in ('G1','G2','G3')
            for a in range(editor.diagnostics['noise_draws']) for b in range(a+1,editor.diagnostics['noise_draws'])])
