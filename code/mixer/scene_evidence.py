"""Frozen denoising evidence and relation-space post-window editing.

Armijo compares one frozen local linear teacher surrogate. Its values across
teacher refreshes are not a global density or a monotone likelihood sequence.
"""
import time

import torch

from .input_views import KnownEmptyObjectView, masked_object_arguments
from .relational import RelationalGeometry, RelationalObjective


EXPLICIT_TERMS = ('residual', 'contact', 'stance', 'endpoint',
                  'human_scene', 'object_scene')


def epsilon_from_x0(noisy, prediction, alpha, sigma):
    return (noisy - alpha * prediction) / sigma


def evidence_direction(hoi_delta, hsi_delta, alpha, sigma, beta, lambda_dp, mask):
    direction = beta * hoi_delta.clone()
    direction[..., :216] += lambda_dp * hsi_delta[..., :216]
    return ((alpha / sigma) * direction * mask).detach()


def editable_mask(motion):
    mask = torch.ones_like(motion)
    mask[:, :2] = 0
    mask[..., 228:] = 0
    return mask


def domain_state(state, grid):
    points = torch.cat((state['human'][:, 2:], state['object_surface'][:, 2:]), -2)
    lower, upper = grid[:3], grid[3:6]
    distance = ((lower - points).clamp_min(0) + (points - upper).clamp_min(0)).norm(dim=-1)
    inside = ((points >= lower) & (points < upper)).all(-1)
    voxel = ((points - lower) / ((upper - lower) / grid[6:])).long()
    query_valid = ((voxel >= 0) & (voxel < grid[6:])).all(-1)
    return distance, inside, query_valid


def domain_accepts(current, trial):
    distance, inside, _ = current
    new_distance, new_inside, _ = trial
    return ((~inside | new_inside) & (new_distance <= distance)).flatten(1).all(1)


def local_armijo(parameters, evaluate, admissible, initial_step=1., shrink=.5,
                 c1=1e-4, max_backtracks=10):
    """One independently accepted step per cell, returning the last valid state."""
    with torch.enable_grad():
        origin = parameters.detach().requires_grad_(True)
        value = evaluate(origin)
        gradient, = torch.autograd.grad(value.sum(), origin)
    if not torch.isfinite(gradient).all() or not torch.isfinite(value).all():
        raise FloatingPointError('nonfinite scene-edit local objective/gradient')
    gradient = gradient.detach()
    norm2 = gradient.square().flatten(1).sum(1)
    active = norm2 > 0
    accepted = torch.zeros_like(active)
    result = parameters.detach().clone()
    trials = []
    step = initial_step
    for _ in range(max_backtracks):
        if not active.any():
            break
        with torch.no_grad():
            proposal = parameters - step * gradient
            trial_value = evaluate(proposal)
            valid = admissible(proposal)
            take = (active & valid & torch.isfinite(trial_value)
                    & (trial_value < value.detach())
                    & (trial_value <= value.detach() - c1 * step * norm2))
            result[take] = proposal[take]
        trials.append(dict(step=step, value=trial_value.detach().cpu().tolist(),
                           admissible=valid.cpu().tolist(), accepted=take.cpu().tolist()))
        accepted |= take
        active &= ~take
        step *= shrink
    return result, dict(value=value.detach().cpu().tolist(),
                        gradient_norm=norm2.sqrt().cpu().tolist(),
                        accepted=accepted.cpu().tolist(),
                        reason=['accepted' if a else ('zero_gradient' if n == 0 else 'search_exhausted')
                                for a, n in zip(accepted.tolist(), norm2.tolist())], trials=trials)


class SceneEvidenceTeacher:
    """Pure raw-head queries using conditions prepared once by native generation."""

    def __init__(self, sampler, arguments, local_bps, context, source, seed,
                 beta=1., lambda_dp=.1):
        self.sampler, self.arguments = sampler, arguments
        self.local_bps, self.context = local_bps, context
        self.source = source.detach()
        self.beta, self.lambda_dp = beta, lambda_dp
        self.generator = torch.Generator(device=source.device).manual_seed(
            (int(seed) + 32452843) % (2**63 - 1))
        self.scene_generator = torch.Generator(device='cpu').manual_seed(
            (int(seed) + 49979687) % (2**63 - 1))
        self.empty = KnownEmptyObjectView()
        self.empty.begin_window(source, seed)
        self.hoi_calls = self.hsi_calls = 0

    @torch.no_grad()
    def query(self, candidate, level):
        diffusion = self.sampler.inner_hoi.diffusion
        t = torch.full((candidate.shape[0],), level, device=candidate.device, dtype=torch.long)
        alpha = diffusion.sqrt_alpha_bar[level].to(candidate)
        sigma = diffusion.sqrt_one_minus_alpha_bar[level].to(candidate)
        noise = torch.randn(candidate.shape, device=candidate.device,
                            dtype=candidate.dtype, generator=self.generator)
        z_edit = diffusion.q_sample(candidate, t, noise)
        z_ref = diffusion.q_sample(self.source, t, noise)
        hoi_delta, hsi_delta = torch.zeros_like(candidate), torch.zeros_like(candidate)
        if self.beta:
            edit = self.sampler._hoi_raw_x0(z_edit, t, self.arguments, self.local_bps)
            reference = self.sampler._hoi_raw_x0(z_ref, t, self.arguments, self.local_bps)
            hoi_delta = epsilon_from_x0(z_edit, edit, alpha, sigma) - epsilon_from_x0(z_ref, reference, alpha, sigma)
            self.hoi_calls += 2
        if self.lambda_dp:
            # Both geometric query inputs are clean. Object geometry is complete;
            # only the denoiser's view has known-empty object/contact modalities.
            # Native occupancy queries sample object vertices with CPU randperm.
            # Carry that stream separately, restoring both ambient RNG states.
            devices = [candidate.device.index] if candidate.is_cuda else []
            with torch.random.fork_rng(devices=devices):
                torch.set_rng_state(self.scene_generator.get_state())
                common = self.sampler._hsi_model_arguments(candidate, candidate, t, self.context)
                self.scene_generator.set_state(torch.get_rng_state())
            view = self.empty.for_step(z_edit, level)
            cond, base = self.sampler._hsi_predict_pair(view, masked_object_arguments(common))
            hsi_delta[..., :216] = (epsilon_from_x0(view, cond, alpha, sigma)
                                    - epsilon_from_x0(view, base, alpha, sigma))[..., :216]
            self.hsi_calls += 2
        mask = editable_mask(candidate)
        direction = evidence_direction(hoi_delta, hsi_delta, alpha, sigma,
                                       self.beta, self.lambda_dp, mask)
        self.components = {
            'hoi_reference': ((alpha / sigma) * self.beta * hoi_delta * mask).detach(),
            'hsi_evidence': ((alpha / sigma) * self.lambda_dp * hsi_delta * mask).detach(),
        }
        if not torch.isfinite(direction).all():
            raise FloatingPointError('nonfinite scene-edit teacher; stop and retain run failure')
        def rms(x):
            return ((x * mask).square().flatten(1).sum(1) / mask.flatten(1).sum(1)).sqrt().cpu().tolist()
        return direction, dict(level=level, alpha=float(alpha), sigma=float(sigma),
                               hoi_reference_rms=rms(hoi_delta), hsi_evidence_rms=rms(hsi_delta),
                               direction_rms=rms(direction))


class SceneEvidenceEditor:
    """Bounded relation editing usable on a native or cached source window."""

    def __init__(self, enabled=False, mode='edit', lambda_dp=.1,
                 hoi_reference_weight=1., noise_levels=(300, 264, 229, 193, 157, 121, 86, 50),
                 initial_step=1., shrink=.5, c1=1e-4, max_backtracks=10,
                 prox_weight=1., record_motion=True):
        if mode not in ('edit', 'reconstruct_only', 'calibrate'):
            raise ValueError('scene_edit mode must be edit, reconstruct_only or calibrate')
        if not noise_levels or any(k <= 0 or k >= 499 for k in noise_levels):
            raise ValueError('scene_edit requires interior canonical noise levels')
        self.enabled, self.mode = enabled, mode
        self.lambda_dp, self.hoi_reference_weight = lambda_dp, hoi_reference_weight
        self.noise_levels = tuple(noise_levels)
        self.initial_step, self.shrink, self.c1 = initial_step, shrink, c1
        self.max_backtracks, self.prox_weight = max_backtracks, prox_weight
        self.record_motion = record_motion
        self.records, self.motion_records = [], []
        self.cell = ('disabled' if not enabled else
                     ('reconstruct' if mode == 'reconstruct_only' else ('dp_edit' if lambda_dp else 'lambda0')))

    @torch.no_grad()
    def edit(self, sampler, source, arguments, local_bps, context, offsets, seed):
        if not self.enabled:
            return source
        if source.is_cuda:
            torch.cuda.synchronize(source.device)
        start = time.perf_counter()
        name = context['seq_name_dict'][0].split('_')[1]
        vertices = context['obj_rest_verts'][name]
        indices = torch.linspace(0, len(vertices) - 1, 128, device=vertices.device).long()
        geometry = RelationalGeometry(source, sampler.dataset, offsets, context, vertices[indices][None])
        parameters = source.new_zeros(*source.shape[:2], geometry.dimension)
        initial = geometry.decode(parameters)
        reference = geometry.encode(initial, source[:, :2]).detach()
        objective = RelationalObjective(geometry, context['scene_flag'], initial['human'][:, 2:],
                                        source_floor=True, source_stance_velocity=True)
        teacher = SceneEvidenceTeacher(sampler, arguments, local_bps, context, reference, seed,
                                        self.hoi_reference_weight, self.lambda_dp)
        grid = sampler.dataset.scene_grid_torch.to(source)
        mask = editable_mask(source)
        denominator = mask.flatten(1).sum(1)
        initial_domain = domain_state(initial, grid)
        iterations, teacher_seconds, solver_seconds = [], 0., 0.
        initial_terms, initial_metrics = objective.evaluate(initial)
        for level in self.noise_levels if self.mode in ('edit', 'calibrate') else ():
            current_state = geometry.decode(parameters)
            candidate = geometry.encode(current_state, source[:, :2]).detach()
            current_domain = domain_state(current_state, grid)
            if source.is_cuda:
                torch.cuda.synchronize(source.device)
            started = time.perf_counter()
            direction, record = teacher.query(candidate, level)
            if source.is_cuda:
                torch.cuda.synchronize(source.device)
            teacher_seconds += time.perf_counter() - started
            origin = parameters.detach()
            def evaluate(proposal):
                state = geometry.decode(proposal)
                motion = geometry.encode(state, source[:, :2])
                terms, _ = objective.evaluate(state)
                linear = (direction * mask * (motion - candidate)).flatten(1).sum(1) / denominator
                proximal = .5 * self.prox_weight * (proposal - origin).square().flatten(1).sum(1)
                return linear + sum(terms[key] for key in EXPLICIT_TERMS) + proximal
            def admissible(proposal):
                return domain_accepts(current_domain, domain_state(geometry.decode(proposal), grid))
            started = time.perf_counter()
            with torch.enable_grad():
                probe = parameters.detach().requires_grad_(True)
                probe_state = geometry.decode(probe)
                probe_motion = geometry.encode(probe_state, source[:, :2])
                probe_terms, _ = objective.evaluate(probe_state)
                gradients = {}
                for key, component in teacher.components.items():
                    scalar = (component * probe_motion).flatten(1).sum(1) / denominator
                    gradients[key], = torch.autograd.grad(scalar.sum(), probe, retain_graph=True)
                gradients['explicit'], = torch.autograd.grad(
                    sum(probe_terms[key] for key in EXPLICIT_TERMS).sum(), probe)
            record['parameter_gradient_norms'] = {
                key: value.flatten(1).norm(dim=1).cpu().tolist()
                for key, value in gradients.items()}
            record['parameter_gradient_dots'] = {
                first + '_dot_' + second: (gradients[first] * gradients[second]).flatten(1).sum(1).cpu().tolist()
                for first, second in (('hoi_reference', 'hsi_evidence'),
                                      ('hoi_reference', 'explicit'), ('hsi_evidence', 'explicit'))}
            if self.mode == 'calibrate':
                solver = dict(accepted=[False] * source.shape[0], trials=[],
                              reason=['passive_measurement'] * source.shape[0])
            else:
                parameters, solver = local_armijo(parameters, evaluate, admissible,
                                                   self.initial_step, self.shrink, self.c1, self.max_backtracks)
            if source.is_cuda:
                torch.cuda.synchronize(source.device)
            solver_seconds += time.perf_counter() - started
            terms, _ = objective.evaluate(geometry.decode(parameters))
            record.update(solver=solver, explicit_terms={k: terms[k].cpu().tolist() for k in EXPLICIT_TERMS})
            iterations.append(record)
        final_state = geometry.decode(parameters)
        result = geometry.encode(final_state, source[:, :2]).detach()
        final_terms, final_metrics = objective.evaluate(final_state)
        final_domain = domain_state(final_state, grid)
        if not torch.isfinite(result).all():
            raise FloatingPointError('nonfinite scene-edit output')
        if source.is_cuda:
            torch.cuda.synchronize(source.device)
        def values(items):
            return {k: v.detach().cpu().tolist() for k, v in items.items()}
        def domain_record(items):
            return dict(outside_points=(~items[1]).flatten(1).sum(1).cpu().tolist(),
                        invalid_queries=(~items[2]).flatten(1).sum(1).cpu().tolist(),
                        exterior_distance_max=items[0].flatten(1).max(1).values.cpu().tolist())
        returned = source if self.mode == 'calibrate' else result
        self.records.append(dict(window=sampler.inner_hoi.sample_calls, mode=self.mode,
            returned_source_exact=torch.equal(returned, source),
            seed=int(seed), seconds=time.perf_counter() - start,
            teacher_seconds=teacher_seconds, solver_seconds=solver_seconds,
            hoi_teacher_calls=teacher.hoi_calls, hsi_teacher_calls=teacher.hsi_calls,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(source.device) if source.is_cuda else None,
            history_exact=torch.equal(result[:, :2], source[:, :2]),
            contact_exact=torch.equal(result[..., 228:], source[..., 228:]),
            reconstruction_rms=float((reference-source).square().mean().sqrt()),
            edit_rms=float((result-reference).square().mean().sqrt()),
            source_terms=values({k: initial_terms[k] for k in EXPLICIT_TERMS}),
            final_terms=values({k: final_terms[k] for k in EXPLICIT_TERMS}),
            source_metrics=values(initial_metrics), final_metrics=values(final_metrics),
            source_domain=domain_record(initial_domain), final_domain=domain_record(final_domain),
            iterations=iterations))
        if self.record_motion:
            self.motion_records.append(dict(window=sampler.inner_hoi.sample_calls,
                raw_source=source.cpu().clone(), reference=reference.cpu(), edited=returned.cpu(),
                parameters=parameters.cpu(), stance_mask=objective.stance.cpu(),
                contact_mask=objective.contact.cpu()))
        return returned

    def audit_dict(self):
        return dict(enabled=self.enabled, mode=self.mode, placement='post_window',
                    lambda_dp=self.lambda_dp, hoi_reference_weight=self.hoi_reference_weight,
                    noise_levels=self.noise_levels, prediction_source='raw_x0',
                    weighting='alpha_over_sigma', explicit_terms=EXPLICIT_TERMS,
                    scene_query_rng='independent_cpu_generator_forked_per_query',
                    hsi_teacher_input='known_empty_forward_trajectory',
                    explicit_weights={key: 1. for key in EXPLICIT_TERMS},
                    physical_scales_m=dict(contact=.05, stance=.02, endpoint=.10, scene=.05),
                    translation_bound_m=.10, angular_component_bound_deg=10.,
                    solver=dict(initial_step=self.initial_step, shrink=self.shrink,
                                c1=self.c1, max_backtracks=self.max_backtracks,
                                prox_weight=self.prox_weight),
                    records=self.records)
