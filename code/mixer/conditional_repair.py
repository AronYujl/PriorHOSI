"""Partial-noise scene-conditioned pose repair with fixed proposal common motion."""
import time

import torch
from pytorch3d import transforms

from .input_views import KnownEmptyObjectView, masked_object_arguments
from .kinematic_composition import _local_from_global
from .relational import RelationalGeometry, RelationalObjective
from .relation_projection import contact_jacobian, nullspace_projection, synchronize
from .scene_evidence import SceneEvidenceEditor, domain_state, domain_accepts, local_armijo


def ddim_repair_step(diffusion, solver, current, clean, level, index):
    alpha = diffusion.sqrt_alpha_bar[level].to(current)
    sigma = diffusion.sqrt_one_minus_alpha_bar[level].to(current)
    epsilon = (current-alpha*clean)/sigma
    indices = torch.full((len(current),), index, device=current.device, dtype=torch.long)
    # DDIMSolver's numpy previous-alpha array is float64; the expert input is float32.
    result = solver.ddim_step(clean, epsilon, indices).to(current)
    result[:, :2] = clean[:, :2]
    return result


def locked_encode(geometry, parameters):
    """Reconstruct coherent future FK while copying immutable redundant channels."""
    if not parameters.any():
        return geometry.base.clone()
    result = geometry.encode(geometry.decode(parameters), geometry.base[:, :2])
    result[..., :3] = geometry.base[..., :3]
    result[..., 84:90] = geometry.base[..., 84:90]
    result[..., 216:] = geometry.base[..., 216:]
    return result


def project_local_pose(objective, parameters, gradient):
    """The tangent solve owns63 local coordinates; fixed columns never enter SVD."""
    jacobian, _ = contact_jacobian(objective, parameters)
    local, _ = nullspace_projection(jacobian[..., 4:], gradient[:, 2:, 4:])
    future = torch.cat((torch.zeros_like(gradient[:, 2:, :4]), local), -1)
    return torch.cat((torch.zeros_like(gradient[:, :2]), future), 1)


def support_foot_energy(human, stance):
    """World X/Z displacement energy in m²; fixed pairs1->2 through14->15.

    Average over active frame-foot pairs, summing the two horizontal axes.
    Empty support is vacuous (energy0/count0), never evidence of improvement.
    """
    feet = human[..., (7, 8, 10, 11), :].double()
    squared = (feet[:, 2:, :, (0, 2)] - feet[:, 1:-1, :, (0, 2)]).square().sum(-1)
    count = stance.flatten(1).sum(1)
    energy = torch.where(stance, squared, torch.zeros_like(squared)).flatten(1).sum(1)
    return energy / count.clamp_min(1), count


class ConstrainedPoseFit:
    """Finite FK guards, source anchors and local-only projected pose fitting."""

    def __init__(self, geometry, source_geometry, scene_flag, iterations=4,
                 proposal_weight=.25, initial_step=.25, max_backtracks=10,
                 foot_guard_mode='increment', foot_energy_epsilon_m2=1e-12):
        self.geometry = geometry
        self.foot_guard_mode = foot_guard_mode
        self.foot_energy_epsilon_m2 = foot_energy_epsilon_m2
        self.iterations, self.proposal_weight = iterations, proposal_weight
        self.initial_step, self.max_backtracks = initial_step, max_backtracks
        self.zero = geometry.base.new_zeros(*geometry.base.shape[:2], geometry.dimension)
        self.proposal = geometry.decode(self.zero)
        self.objective = RelationalObjective(source_geometry, scene_flag,
            self.proposal['human'][:, 2:], source_floor=True, source_stance_velocity=True)
        self.objective.geometry = geometry  # anchors remain from raw source
        self.protection = RelationalObjective(geometry, scene_flag,
            self.proposal['human'][:, 2:], source_floor=True, source_stance_velocity=True)
        self.grid = geometry.dataset.scene_grid_torch.to(geometry.base)
        self.domain = domain_state(self.proposal, self.grid)
        self.contact_distance = self.objective.contact_residual(self.proposal).norm(dim=-1)
        self.contact_limit = self.contact_distance.clamp_min(.005) + .001
        _, self.metrics = self.protection.evaluate(self.proposal)
        self.proposal_foot_energy, self.support_count = support_foot_energy(
            self.proposal['human'], self.protection.stance)
        self.invalid_proposal = bool(((self.contact_distance > .05) & self.objective.contact).any())

    def guards(self, parameters):
        state = self.geometry.decode(parameters)
        _, metrics = self.protection.evaluate(state)
        contact = self.objective.contact_residual(state).norm(dim=-1)
        feet = (state['human'][:, 2:, (7, 8, 10, 11)]
                - self.proposal['human'][:, 2:, (7, 8, 10, 11)]).norm(dim=-1)
        energy, count = support_foot_energy(state['human'], self.protection.stance)
        legacy = metrics['stance_increment_cm'] <= .05
        quality = energy <= self.proposal_foot_energy + self.foot_energy_epsilon_m2
        self.last_foot_metrics = dict(active_count=count,
            proposal_energy_m2=self.proposal_foot_energy, candidate_energy_m2=energy,
            energy_delta_m2=energy-self.proposal_foot_energy,
            stance_increment_cm=metrics['stance_increment_cm'],
            legacy_pass=legacy, quality_pass=quality,
            max_foot_displacement_cm=feet.flatten(1).max(1).values*100)
        checks = dict(
            domain=domain_accepts(self.domain, domain_state(state, self.grid)),
            contact=((contact <= self.contact_limit) | ~self.objective.contact).flatten(1).all(1),
            human_scene=metrics['human_scene_residual_cm'] <= self.metrics['human_scene_residual_cm'] + .01,
            stance={'increment': legacy, 'quality': quality}[self.foot_guard_mode],
            feet=(feet <= .02).flatten(1).all(1),
            common=(parameters[..., :4] == 0).flatten(1).all(1),
            finite=torch.isfinite(state['human']).flatten(1).all(1))
        return checks

    def fit(self, parameters, target_local):
        geometry = self.geometry
        target_local = target_local.detach()
        def evaluate(p):
            bounded = p.tanh() * geometry.future_mask
            delta = geometry.angle_scale * bounded[..., 4:].reshape(*p.shape[:2], 21, 3)
            local = geometry.local_rotation[..., 1:, :, :] @ transforms.axis_angle_to_matrix(delta)
            pose = ((local[:, 2:] - target_local[:, 2:, 1:]) / geometry.angle_scale).square().flatten(1).sum(1) / 2
            return pose + self.proposal_weight / 2 * bounded[..., 4:].square().flatten(1).sum(1)
        guard_records = []
        def admissible(p):
            checks = self.guards(p)
            guard_records.append((checks, self.last_foot_metrics))
            return torch.stack(list(checks.values())).all(0)
        traces = []
        for _ in range(self.iterations):
            guard_records.clear()
            parameters, trace = local_armijo(parameters, evaluate, admissible,
                initial_step=self.initial_step, max_backtracks=self.max_backtracks,
                gradient_transform=lambda p, g: project_local_pose(self.objective, p, g))
            for trial, (checks, foot) in zip(trace['trials'], guard_records):
                trial['guards'] = {k:v.cpu().tolist() for k,v in checks.items()}
                trial['foot'] = {k:v.cpu().tolist() for k,v in foot.items()}
            traces.append(trace)
        return parameters, traces


class ConditionalRepairEditor(SceneEvidenceEditor):
    """Shared frozen lambda0 proposal followed by one optional HSI DDIM chain."""

    def __init__(self, repair_enabled=True, baseline_hoi=False, repair_start_step=99,
                 repair_timesteps=(99, 79, 59, 39, 19), repair_eta=0.,
                 repair_prediction_type='x0', fit_iterations=4, proposal_weight=.25,
                 fit_initial_step=.25, fit_max_backtracks=10,
                 foot_guard_mode='increment', foot_energy_epsilon_m2=1e-12, **kwargs):
        super().__init__(**kwargs)
        if self.lambda_dp != 0 or self.mode != 'edit':
            raise ValueError('conditional repair requires the frozen lambda0 edit proposal')
        if list(repair_timesteps) != [99, 79, 59, 39, 19] or repair_start_step != 99 or repair_eta != 0 or repair_prediction_type != 'x0':
            raise ValueError('conditional repair uses the registered five-step DDIM x0 chain')
        self.repair_enabled, self.baseline_hoi = repair_enabled, baseline_hoi
        self.repair_timesteps = list(repair_timesteps)
        self.fit_options = dict(iterations=fit_iterations, proposal_weight=proposal_weight,
            initial_step=fit_initial_step, max_backtracks=fit_max_backtracks,
            foot_guard_mode=foot_guard_mode, foot_energy_epsilon_m2=foot_energy_epsilon_m2)
        self.cell = 'B0_hoi' if baseline_hoi else ('B2_hsi_repair' if repair_enabled else 'B1_no_hsi')
        if repair_enabled and not baseline_hoi and foot_guard_mode == 'quality':
            self.cell = 'B2_quality'

    @torch.no_grad()
    def edit(self, sampler, source, arguments, local_bps, context, offsets, seed):
        synchronize(source)
        started = time.perf_counter()
        if self.baseline_hoi:
            proposal = source
        else:
            proposal = super().edit(sampler, source, arguments, local_bps, context, offsets, seed)
        synchronize(source)
        geometry_seconds = time.perf_counter() - started
        name = context['seq_name_dict'][0].split('_')[1]
        vertices = context['obj_rest_verts'][name]
        indices = torch.linspace(0, len(vertices)-1, 128, device=vertices.device).long()
        points = vertices[indices][None]
        source_geometry = RelationalGeometry(source, sampler.dataset, offsets, context, points)
        geometry = RelationalGeometry(proposal, sampler.dataset, offsets, context, points)
        fitter = ConstrainedPoseFit(geometry, source_geometry, context['scene_flag'], **self.fit_options)
        parameters = fitter.zero.clone()
        source_state = source_geometry.decode(parameters)
        source_objective = RelationalObjective(source_geometry, context['scene_flag'],
            source_state['human'][:, 2:], source_floor=True, source_stance_velocity=True)
        _, source_metrics = source_objective.evaluate(source_state)
        result = proposal.clone()
        hsi_seconds, fit_seconds, calls, steps = 0., 0., 0, []
        enabled = self.repair_enabled and not self.baseline_hoi
        if enabled:
            generator = torch.Generator(device=source.device).manual_seed((int(seed)+86028121) % (2**63-1))
            scene_generator = torch.Generator().manual_seed((int(seed)+49979687) % (2**63-1))
            empty = KnownEmptyObjectView()
            empty.begin_window(proposal, seed)
            diffusion = sampler.inner_hoi.diffusion
            solver = sampler.hsi_sampler.solver
            if solver.ddim_timesteps[:5].tolist() != list(reversed(self.repair_timesteps)):
                raise ValueError('HSI DDIM solver differs from the registered repair schedule')
            t = torch.full((len(source),), 99, device=source.device, dtype=torch.long)
            noise = torch.randn(source.shape, device=source.device, dtype=source.dtype, generator=generator)
            current = diffusion.q_sample(proposal, t, noise)
        for index, level in enumerate(self.repair_timesteps if not self.baseline_hoi else []):
            trace = dict(level=level, accepted=False, reason='invalid_proposal' if fitter.invalid_proposal else 'pending')
            if fitter.invalid_proposal:
                steps.append(trace)
                continue
            synchronize(source)
            start = time.perf_counter()
            if enabled:
                t = torch.full((len(source),), level, device=source.device, dtype=torch.long)
                devices = [source.device.index] if source.is_cuda else []
                with torch.random.fork_rng(devices=devices):
                    torch.set_rng_state(scene_generator.get_state())
                    common = sampler._hsi_model_arguments(result, result, t, context)
                    scene_generator.set_state(torch.get_rng_state())
                view = empty.for_step(current, level)
                prediction = sampler.hsi_sampler.student_model(view, *masked_object_arguments(common), is_sample=True)
                calls += 1
                finite = bool(torch.isfinite(prediction[..., :216]).all())
                target = _local_from_global(transforms.rotation_6d_to_matrix(
                    prediction[..., 84:216].reshape(*source.shape[:2], 22, 6))) if finite else None
            else:
                finite, target = True, geometry.local_rotation
            synchronize(source)
            hsi_seconds += time.perf_counter()-start if enabled else 0.
            before = parameters.clone()
            start = time.perf_counter()
            if finite:
                try:
                    parameters, trace['fit'] = fitter.fit(parameters, target)
                    result = locked_encode(geometry, parameters)
                    trace.update(accepted=bool((parameters != before).any()),
                        reason='fitted' if (parameters != before).any() else 'unchanged')
                except FloatingPointError as error:
                    trace.update(reason='nonfinite_fit', error=str(error))
            else:
                trace['reason'] = 'nonfinite_prediction'
            synchronize(source)
            fit_seconds += time.perf_counter()-start
            steps.append(trace)
            if enabled:
                # Constrained clean prediction determines both epsilon and scheduler update.
                current = ddim_repair_step(diffusion, solver, current, result, level, 4-index)
        final_state = geometry.decode(parameters)
        _, final_metrics = source_objective.evaluate(final_state)
        _, proposal_metrics = source_objective.evaluate(fitter.proposal)
        rms = float((final_state['human'][:, 2:]-fitter.proposal['human'][:, 2:]).square().mean().sqrt()*1000)
        history_exact = torch.equal(result[:, :2], source[:, :2])
        common_exact = all(torch.equal(result[..., a:b], proposal[..., a:b]) for a,b in ((0,3),(84,90),(216,232)))
        if not history_exact or not common_exact:
            raise AssertionError('conditional repair changed immutable history/common motion')
        values = lambda d: {k: v.detach().cpu().tolist() for k,v in d.items()}
        final_guards = fitter.guards(parameters)
        record = dict(window=sampler.inner_hoi.sample_calls, seed=int(seed), cell=self.cell,
            history_exact=history_exact, common_exact=common_exact, contact_exact=torch.equal(result[..., 228:], source[..., 228:]),
            source_metrics=values(source_metrics), proposal_metrics=values(proposal_metrics), final_metrics=values(final_metrics),
            repair_rms_mm=rms, modified=rms>=1., fallback=bool(torch.equal(result, proposal)),
            invalid_proposal=fitter.invalid_proposal, hsi_calls=calls, hsi_seconds=hsi_seconds,
            fit_seconds=fit_seconds, geometry_seconds=geometry_seconds, steps=steps,
            final_guards={k: bool(v.all()) for k,v in final_guards.items()},
            foot=values(fitter.last_foot_metrics), foot_guard_mode=fitter.foot_guard_mode,
            foot_energy_epsilon_m2=fitter.foot_energy_epsilon_m2,
            max_source_anchor_cm=float((source_objective.contact_residual(final_state).norm(dim=-1)*source_objective.contact).max()*100),
            source_outside_points=int((~fitter.domain[1]).sum()),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(source.device) if source.is_cuda else None)
        synchronize(source)
        record['seconds'] = time.perf_counter()-started
        if self.baseline_hoi:
            self.records.append(record)
            if self.record_motion:
                self.motion_records.append({})
        else:
            geometry_record = self.records[-1]
            record['geometry_edit'] = geometry_record
            self.records[-1] = record
        if self.record_motion:
            self.motion_records[-1].update(raw_source=source.cpu().clone(), proposal=proposal.cpu(), edited=result.cpu(),
                repair_parameters=parameters.cpu(), source_anchor=source_objective.hand_anchor.cpu(),
                contact_mask=source_objective.contact.cpu(), proposal_fk=fitter.proposal['human'].cpu(),
                final_fk=final_state['human'].cpu(), repair_stance_mask=fitter.protection.stance.cpu())
        return result

    def audit_dict(self):
        return dict(super().audit_dict(), cell=self.cell, repair_enabled=self.repair_enabled,
            baseline_hoi=self.baseline_hoi, repair_timesteps=self.repair_timesteps,
            repair_sampler='DDIMSolver(500,25)', repair_eta=0., repair_prediction_type='x0',
            hsi_condition='full', repair_cfg=0., fit_options=self.fit_options)
