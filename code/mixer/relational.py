"""Differentiable common human-object motion with articulated body corrections."""

import math
import time

import torch
from pytorch3d import transforms

from priors.core.window_codec import project_to_so3
from .diagnostics import decode_human
from .kinematic_composition import (
    _ATTACHED_MARKER_PARENTS, _FK_EXTRA_TO_POSITION, _apply_rotation,
    _expand_rest_offsets, _forward_kinematics, _local_from_global,
)


CELL_KEYS = ('a00', 'a10', 'a01', 'a11')


@torch.no_grad()
def source_floor_height(human):
    """Native low-speed/DBSCAN height rule on nonpadded scale-3 source FK.

    One-dimensional core components are ordered by their first input index,
    matching DBSCAN's assignment of a border point reachable from two clusters.
    Native evaluation includes noise in its minimum-median comparison.
    """
    toes = human.detach()[:, :, (10, 11), :].to(torch.float64)
    index = torch.arange((human.shape[1] - 1) * 3 + 1, device=human.device)
    base = index // 3
    fraction = (index % 3).to(toes.dtype)[None, :, None, None] / 3
    dense = (toes[:, base] * (1 - fraction)
             + toes[:, (base + 1).clamp_max(human.shape[1] - 1)] * fraction).to(human.dtype)
    speed = (dense[:, 1:] - dense[:, :-1]).norm(dim=-1)
    speed = torch.cat((speed, speed[:, -1:]), dim=1)
    selected = speed < .005
    floors = []
    counts = selected.flatten(1).sum(1)
    for points, mask in zip(dense, selected):
        heights = points[..., 1].transpose(0, 1).flatten()
        values = heights[mask.transpose(0, 1).flatten()].to(torch.float64)
        count = len(values)
        if count == 0:
            floors.append(human.new_zeros(()))
            continue
        neighbors = (values[:, None] - values[None, :]).abs() <= .005
        core = torch.where(neighbors.sum(1) >= 3)[0]
        if len(core) == 0:
            floors.append(values.quantile(.5).to(human.dtype))
            continue
        ordered = core[values[core].argsort()]
        gaps = values[ordered][1:] - values[ordered][:-1] > .005
        groups = torch.cat((gaps.new_zeros(1), gaps)).long().cumsum(0)
        priorities = torch.empty_like(ordered)
        for group in range(int(groups[-1]) + 1):
            member = groups == group
            priorities[member] = ordered[member].min()
        labels = torch.where(neighbors[:, ordered], priorities[None, :], count).amin(1)
        medians = torch.stack([values[labels == label].quantile(.5) for label in labels.unique()])
        floors.append(medians.min().to(human.dtype))
    return torch.stack(floors), counts


class RelationalGeometry:
    """67 residual coordinates/frame: translation, yaw, and 21 local rotations.

    Cached expert inputs are frozen. Residual coordinates and reconstruction
    remain differentiable, including at zero. A batch of residuals can broadcast
    one cached window to the four independent optimization cells.
    """

    translation_scale = 0.10
    angle_scale = math.radians(10)
    dimension = 4 + 21 * 3

    def __init__(self, clean, dataset, rest_offsets, context, object_points):
        self.base = clean.detach()
        self.dataset = dataset
        self.positions = dataset.denormalize_torch(self.base[..., :84]).reshape(
            *clean.shape[:2], 28, 3,
        )
        self.global_rotation = transforms.rotation_6d_to_matrix(
            self.base[..., 84:216].reshape(*clean.shape[:2], 22, 6),
        )
        self.local_rotation = _local_from_global(self.global_rotation)
        self.offsets = _expand_rest_offsets(rest_offsets, *clean.shape[:2], self.positions)
        self.offsets[..., 0, :] = self.positions[..., 0, :]
        self.mat = context['mat'].to(clean)
        self.world_rotation = self.mat[:, :3, :3]
        self.world_translation = self.mat[:, :3, 3]
        self.prefix = context['obj_rot_mat_prefix'].reshape(-1, 3, 3).to(clean)
        self.reference = context['obj_rot_mat_ref'].reshape(clean.shape[0], -1, 3, 3)[:, 0].to(clean)
        self.object_position = dataset.denormalize_torch(self.base[..., 216:219], is_object=True)
        object_relative = project_to_so3(self.base[..., 219:228].reshape(*clean.shape[:2], 3, 3))
        self.object_rotation_world = self.prefix[:, None] @ object_relative @ self.reference[:, None]
        self.object_points = object_points.to(clean)
        mask = clean.new_ones(1, clean.shape[1], 1)
        mask[:, :2] = 0
        self.future_mask = mask

    def world_points(self, points):
        return _apply_rotation(self.world_rotation[:, None, None], points) + self.world_translation[:, None, None]

    def decode(self, parameters):
        bounded = parameters.tanh() * self.future_mask
        translation = self.translation_scale * bounded[..., :3]
        yaw = self.angle_scale * bounded[..., 3]
        local_delta = self.angle_scale * bounded[..., 4:].reshape(*parameters.shape[:2], 21, 3)
        cosine, sine, zero = yaw.cos(), yaw.sin(), torch.zeros_like(yaw)
        one = torch.ones_like(yaw)
        common_rotation = torch.stack(
            (cosine, zero, sine, zero, one, zero, -sine, zero, cosine), dim=-1,
        ).reshape(*yaw.shape, 3, 3)
        root = self.positions[..., 0, :] + translation
        root_rotation = common_rotation @ self.local_rotation[..., 0, :, :]
        local = torch.cat((
            root_rotation[..., None, :, :],
            self.local_rotation[..., 1:, :, :] @ transforms.axis_angle_to_matrix(local_delta),
        ), dim=-3)
        offsets = self.offsets.expand(parameters.shape[0], -1, -1, -1).clone()
        offsets[..., 0, :] = root
        global_rotation, local_fk = _forward_kinematics(local, offsets)
        human = self.world_points(local_fk)
        object_position = root + _apply_rotation(
            common_rotation, self.object_position - self.positions[..., 0, :],
        )
        object_translation_world = (
            _apply_rotation(self.world_rotation[:, None], object_position)
            + self.world_translation[:, None]
        )
        common_world_rotation = (
            self.world_rotation[:, None] @ common_rotation
            @ self.world_rotation[:, None].transpose(-1, -2)
        )
        object_rotation_world = common_world_rotation @ self.object_rotation_world
        object_surface = (
            _apply_rotation(object_rotation_world[..., None, :, :], self.object_points[:, None])
            + object_translation_world[..., None, :]
        )
        return {
            'human': human, 'local_fk': local_fk, 'global_rotation': global_rotation,
            'object_position': object_position, 'object_translation_world': object_translation_world,
            'object_rotation_world': object_rotation_world, 'object_surface': object_surface,
            'bounded_parameters': bounded, 'translation': translation,
            'yaw': yaw, 'local_delta': local_delta,
        }

    def encode(self, state, fixed_points):
        global_rotation, fk = state['global_rotation'], state['local_fk']
        positions = self.positions.expand(fk.shape[0], -1, -1, -1).clone()
        positions[..., :22, :] = fk[..., :22, :]
        for fk_index, position_index in _FK_EXTRA_TO_POSITION.items():
            positions[..., position_index, :] = fk[..., fk_index, :]
        for marker, parent in _ATTACHED_MARKER_PARENTS.items():
            vector = self.positions[..., marker, :] - self.positions[..., parent, :]
            local_vector = _apply_rotation(self.global_rotation[..., parent, :, :].transpose(-1, -2), vector)
            positions[..., marker, :] = positions[..., parent, :] + _apply_rotation(
                global_rotation[..., parent, :, :], local_vector,
            )
        object_relative = (
            self.prefix[:, None].transpose(-1, -2) @ state['object_rotation_world']
            @ self.reference[:, None].transpose(-1, -2)
        )
        result = torch.cat((
            self.dataset.normalize_torch(positions).flatten(-2),
            transforms.matrix_to_rotation_6d(global_rotation).flatten(-2),
            self.dataset.normalize_torch(state['object_position'], is_object=True),
            object_relative.flatten(-2), self.base[..., 228:232].expand(fk.shape[0], -1, -1),
        ), dim=-1)
        # Keep the original redundant history representation exactly.
        return torch.cat((fixed_points.expand(fk.shape[0], -1, -1), result[:, 2:]), dim=1)


def _mean(value):
    return value.flatten(1).mean(1)


def _masked_mean(value, mask):
    # An empty contact/stance set contributes zero by definition.
    return (value * mask).flatten(1).sum(1) / mask.expand_as(value).flatten(1).sum(1).clamp_min(1)


class RelationalObjective:
    """Physical residuals and fixed source relations shared by all four cells."""

    def __init__(self, geometry, scene_flag, hsi_target, source_floor=False,
                 source_stance_velocity=False):
        self.geometry = geometry
        self.scene_flag = scene_flag
        self.hsi_target = hsi_target.detach()
        self.source_stance_velocity = source_stance_velocity
        zero = geometry.base.new_zeros(*geometry.base.shape[:2], geometry.dimension)
        with torch.no_grad():
            self.anchor = geometry.decode(zero)
        human = self.anchor['human']
        self.source_feet = human[..., (7, 8, 10, 11), :].detach()
        relative = human[..., 22:24, :] - self.anchor['object_translation_world'][..., None, :]
        self.hand_anchor = _apply_rotation(
            self.anchor['object_rotation_world'][..., None, :, :].transpose(-1, -2), relative,
        )
        self.contact = (geometry.base[..., 228:230] > 0.95)[:, 2:]
        heights = human[..., (7, 8, 10, 11), 1]
        self.floor_height = human.new_zeros(human.shape[0])
        self.floor_sample_count = human.new_zeros(human.shape[0])
        if source_floor:
            self.floor_height, self.floor_sample_count = source_floor_height(human)
            heights = heights - self.floor_height[:, None, None]
        stance = heights < heights.new_tensor((0.08, 0.08, 0.04, 0.04))
        self.stance = stance[:, 2:] & stance[:, 1:-1]

    def evaluate(self, state):
        human = state['human']
        surface = state['object_surface']
        points = torch.cat((human[:, 2:], surface[:, 2:]), dim=-2)
        with torch.no_grad():
            occupied, nearest = self.geometry.dataset.get_nearest_free_voxel(
                points.detach(), self.scene_flag.expand(points.shape[0]),
            )
        self.last_scene_query = {'occupied': occupied, 'nearest': nearest}
        distance_squared = (points - nearest).square().sum(-1)
        hand_target = (
            _apply_rotation(state['object_rotation_world'][..., None, :, :], self.hand_anchor)
            + state['object_translation_world'][..., None, :]
        )
        hand_squared = (human[:, 2:, 22:24] - hand_target[:, 2:]).square().sum(-1)
        feet = human[..., (7, 8, 10, 11), :]
        velocity_squared = (feet[:, 2:, :, (0, 2)] - feet[:, 1:-1, :, (0, 2)]).square().sum(-1)
        foot_correction = feet - self.source_feet
        increment_squared = (foot_correction[:, 2:, :, (0, 2)]
                             - foot_correction[:, 1:-1, :, (0, 2)]).square().sum(-1)
        support_height = human[:, 2:, (10, 11), 1].min(-1).values
        root_endpoint = human[:, -1, 0] - self.anchor['human'][:, -1, 0]
        object_endpoint = state['object_translation_world'][:, -1] - self.anchor['object_translation_world'][:, -1]
        residual = state['bounded_parameters'][:, 2:]
        contact_mse = _masked_mean(hand_squared, self.contact)
        stance_mse = _masked_mean(velocity_squared, self.stance)
        stance_increment_mse = _masked_mean(increment_squared, self.stance)
        terms = {
            'residual': _mean(residual[..., :3].square()) + _mean(residual[..., 3:4].square()) + _mean(residual[..., 4:].square()),
            'contact': contact_mse / (3 * 0.05**2),
            'stance': (stance_increment_mse if self.source_stance_velocity else stance_mse) / (2 * 0.02**2),
            'floor': _mean((support_height - 0.02).square()) / 0.02**2,
            'endpoint': (root_endpoint.square().mean(-1) + object_endpoint.square().mean(-1)) / 0.10**2,
            'hsi': _mean((human[:, 2:] - self.hsi_target).square()) / 0.05**2,
            'human_scene': _mean(distance_squared[..., :24]) / (3 * 0.05**2),
            'object_scene': _mean(distance_squared[..., 24:]) / (3 * 0.05**2),
        }
        metrics = {
            'source_floor_height_cm': self.floor_height.expand(human.shape[0]) * 100,
            'source_floor_sample_count': self.floor_sample_count.to(human).expand(human.shape[0]),
            'source_stance_count': self.stance.flatten(1).sum(1).to(human).expand(human.shape[0]),
            'absolute_toe_height_cm': _mean(human[:, 2:, (10, 11), 1]) * 100,
            'human_scene_residual_cm': _mean(distance_squared[..., :24]).sqrt() * 100,
            'object_scene_residual_cm': _mean(distance_squared[..., 24:]).sqrt() * 100,
            'human_occupied_fraction': _mean(occupied[..., :24].float()),
            'object_occupied_fraction': _mean(occupied[..., 24:].float()),
            'contact_anchor_drift_cm': contact_mse.sqrt() * 100,
            'stance_displacement_cm': stance_mse.sqrt() * 100,
            'stance_increment_cm': stance_increment_mse.sqrt() * 100,
            'root_endpoint_shift_cm': root_endpoint.norm(dim=-1) * 100,
            'object_endpoint_shift_cm': object_endpoint.norm(dim=-1) * 100,
            'translation_max_cm': state['translation'].abs().flatten(1).max(1).values * 100,
            'angle_component_max_deg': torch.cat((state['yaw'][..., None], state['local_delta'].flatten(-2)), -1).abs().flatten(1).max(1).values * (180 / math.pi),
        }
        return terms, metrics


def optimize_relational_cells(geometry, objective, steps=20, learning_rate=0.05,
                              cells=CELL_KEYS, include_floor=True, trace=None,
                              solver='adam', max_backtracks=20):
    """Optimize registered cells from identical zero residuals with the selected solver."""
    if solver != 'adam':
        return {'armijo': _optimize_relational_armijo}[solver](
            geometry, objective, steps, learning_rate, cells, include_floor,
            trace, max_backtracks,
        )
    with torch.enable_grad():
        parameters = geometry.base.new_zeros(len(cells), geometry.base.shape[1], geometry.dimension, requires_grad=True)
        optimizer = torch.optim.Adam([parameters], lr=learning_rate)
        use_hsi = parameters.new_tensor([int(cell[1]) for cell in cells])
        use_scene = parameters.new_tensor([int(cell[2]) for cell in cells])
        gradient_finite = torch.ones(len(cells), dtype=torch.bool, device=parameters.device)
        gradient_max = parameters.new_zeros(len(cells))
        initial_terms, initial_metrics = objective.evaluate(geometry.decode(parameters))
        common_terms = ('residual', 'contact', 'stance', 'floor', 'endpoint') if include_floor else (
            'residual', 'contact', 'stance', 'endpoint',
        )
        for _ in range(steps):
            terms, _ = objective.evaluate(geometry.decode(parameters))
            loss = sum(terms[name] for name in common_terms)
            loss = loss + use_hsi * terms['hsi'] + use_scene * (terms['human_scene'] + terms['object_scene'])
            optimizer.zero_grad(set_to_none=True)
            if trace is not None:
                trace.before_backward(parameters, terms, loss, common_terms, optimizer)
            loss.sum().backward()
            gradient_finite &= torch.isfinite(parameters.grad).flatten(1).all(1)
            gradient_max = torch.maximum(gradient_max, parameters.grad.detach().abs().flatten(1).max(1).values)
            optimizer.step()
            if trace is not None:
                trace.after_step(parameters, optimizer)
        with torch.no_grad():
            state = geometry.decode(parameters)
            final_terms, final_metrics = objective.evaluate(state)
            encoded = geometry.encode(state, geometry.base[:, :2])
            if trace is not None:
                final_loss = sum(final_terms[name] for name in common_terms)
                final_loss = final_loss + use_hsi * final_terms['hsi'] + use_scene * (
                    final_terms['human_scene'] + final_terms['object_scene'])
                trace.finish(parameters, final_terms, final_loss)
    metrics = {**final_metrics, **{'energy_' + k: v for k, v in final_terms.items()}}
    metrics.update({'before_' + k: v.detach() for k, v in initial_metrics.items()})
    metrics.update({'before_energy_' + k: v.detach() for k, v in initial_terms.items()})
    metrics['gradient_finite'] = gradient_finite.float()
    metrics['gradient_max'] = gradient_max
    return encoded, parameters.detach(), metrics


def _complete_energy(terms, common_terms, use_hsi, use_scene):
    loss = sum(terms[name] for name in common_terms)
    return loss + use_hsi * terms['hsi'] + use_scene * (
        terms['human_scene'] + terms['object_scene'])


def _optimize_relational_armijo(geometry, objective, steps, learning_rate, cells,
                                include_floor, trace, max_backtracks):
    """Steepest descent with independent sufficient-decrease searches per cell.

    Each trial re-evaluates the complete objective and its current voxel targets.
    Stop codes: 0 gradient budget, 1 zero gradient, 2 exhausted line search.
    """
    with torch.enable_grad():
        parameters = geometry.base.new_zeros(
            len(cells), geometry.base.shape[1], geometry.dimension, requires_grad=True)
        use_hsi = parameters.new_tensor([int(cell[1]) for cell in cells])
        use_scene = parameters.new_tensor([int(cell[2]) for cell in cells])
        common_terms = ('residual', 'contact', 'stance', 'floor', 'endpoint') if include_floor else (
            'residual', 'contact', 'stance', 'endpoint')
        gradient_terms = common_terms + (('hsi',) if any(cell[1] == '1' for cell in cells) else ())
        gradient_terms += ('human_scene', 'object_scene') if any(cell[2] == '1' for cell in cells) else ()
        initial_terms, initial_metrics = objective.evaluate(geometry.decode(parameters))
        initial_loss = _complete_energy(initial_terms, common_terms, use_hsi, use_scene)
        active = torch.ones(len(cells), device=parameters.device, dtype=torch.bool)
        gradient_finite = active.clone()
        energy_finite = torch.isfinite(initial_loss)
        gradient_max = parameters.new_zeros(len(cells))
        gradients = parameters.new_zeros(len(cells))
        evaluations = parameters.new_ones(len(cells))
        trials = parameters.new_zeros(len(cells))
        updates = parameters.new_zeros(len(cells))
        stop_code = parameters.new_zeros(len(cells))
        for iteration in range(steps):
            if not active.any():
                break
            terms, _ = objective.evaluate(geometry.decode(parameters))
            loss = _complete_energy(terms, common_terms, use_hsi, use_scene)
            evaluations += 1
            energy_finite &= torch.isfinite(loss)
            parameters.grad = None
            if trace is not None:
                trace.before_backward(parameters, terms, loss, gradient_terms,
                                      active, objective.last_scene_query)
            loss.sum().backward()
            gradient = parameters.grad.detach().clone()
            gradients += 1
            gradient_finite &= torch.isfinite(gradient).flatten(1).all(1)
            current_max = gradient.abs().flatten(1).amax(1)
            gradient_max = torch.maximum(gradient_max, current_max)
            zero_gradient = active & (current_max == 0)
            stop_code = torch.where(zero_gradient, stop_code.new_ones(()), stop_code)
            pending = active & ~zero_gradient
            accepted = torch.zeros_like(active)
            next_parameters = parameters.detach().clone()
            accepted_alpha = parameters.new_zeros(len(cells))
            alpha = parameters.new_full((len(cells),), learning_rate)
            slope = -gradient.square().flatten(1).sum(1)
            with torch.no_grad():
                for trial in range(max_backtracks):
                    if not pending.any():
                        break
                    candidate = torch.where(
                        pending[:, None, None],
                        parameters - alpha[:, None, None] * gradient,
                        next_parameters,
                    )
                    candidate_terms, _ = objective.evaluate(geometry.decode(candidate))
                    candidate_loss = _complete_energy(candidate_terms, common_terms, use_hsi, use_scene)
                    evaluations += 1
                    trials += pending.to(trials)
                    energy_finite &= torch.isfinite(candidate_loss)
                    bound = loss.detach() + 1e-4 * alpha * slope
                    take = pending & (candidate_loss <= bound) & (candidate_loss < loss.detach())
                    if trace is not None:
                        trace.trial(iteration, trial, pending, alpha, candidate_terms,
                                    candidate_loss, bound, take)
                    next_parameters = torch.where(take[:, None, None], candidate, next_parameters)
                    accepted_alpha = torch.where(take, alpha, accepted_alpha)
                    accepted |= take
                    pending &= ~take
                    alpha *= .5
                parameters.copy_(next_parameters)
                updates += accepted.to(updates)
                stop_code = torch.where(pending, stop_code.new_full((), 2), stop_code)
                active &= accepted
                if trace is not None:
                    trace.after_step(gradient, accepted, accepted_alpha)
        with torch.no_grad():
            state = geometry.decode(parameters)
            final_terms, final_metrics = objective.evaluate(state)
            final_loss = _complete_energy(final_terms, common_terms, use_hsi, use_scene)
            evaluations += 1
            energy_finite &= torch.isfinite(final_loss)
            encoded = geometry.encode(state, geometry.base[:, :2])
            if trace is not None:
                trace.finish(parameters, final_terms, final_loss, objective.last_scene_query)
    metrics = {**final_metrics, **{'energy_' + k: v for k, v in final_terms.items()}}
    metrics.update({'before_' + k: v.detach() for k, v in initial_metrics.items()})
    metrics.update({'before_energy_' + k: v.detach() for k, v in initial_terms.items()})
    metrics.update(
        before_energy_total=initial_loss.detach(), energy_total=final_loss,
        gradient_finite=gradient_finite.float(), gradient_max=gradient_max,
        energy_finite=energy_finite.float(), gradient_evaluations=gradients,
        objective_evaluations=evaluations, line_search_trials=trials,
        optimizer_updates=updates, optimizer_stop_code=stop_code,
    )
    return encoded, parameters.detach(), metrics


def relational_problem(sampler, current, previous_x0, timesteps, context,
                       rest_offsets, clean, source_floor=False,
                       source_stance_velocity=False):
    """Use the same frozen source and HSI target in window and rollout experiments."""
    cond, uncond = sampler._hsi_prediction_pair(current, previous_x0, timesteps, context)
    object_name = context['seq_name_dict'][0].split('_')[1]
    vertices = context['obj_rest_verts'][object_name]
    indices = torch.linspace(0, len(vertices) - 1, 128, device=vertices.device).long()
    geometry = RelationalGeometry(clean, sampler.dataset, rest_offsets, context, vertices[indices][None])
    conditional_fk = decode_human(cond, sampler.dataset, rest_offsets)[2]
    unconditioned_fk = decode_human(uncond, sampler.dataset, rest_offsets)[2]
    increment = _apply_rotation(context['mat'][:, None, None, :3, :3], conditional_fk - unconditioned_fk)
    zero = clean.new_zeros(*clean.shape[:2], geometry.dimension)
    target = geometry.decode(zero)['human'][:, 2:] + increment
    return geometry, RelationalObjective(
        geometry, context['scene_flag'], target, source_floor=source_floor,
        source_stance_velocity=source_stance_velocity,
    ), cond, uncond


class RelationalCorrector:
    """Feed one registered relation cell into the shared reverse chain."""

    def __init__(self, cell='a01', steps=(10, 1, 0), optimizer_steps=20,
                 learning_rate=0.05, include_floor=True, record_motion=False, source_floor=False,
                 source_stance_velocity=False, optimizer_diagnostic=False,
                 solver='adam', max_backtracks=20, solver_diagnostic=False):
        self.cell = cell
        self.steps = tuple(steps)
        self.optimizer_steps = int(optimizer_steps)
        self.learning_rate = float(learning_rate)
        self.include_floor = include_floor
        self.records = []
        self.record_motion = record_motion or optimizer_diagnostic or solver_diagnostic
        self.source_floor = source_floor
        self.source_stance_velocity = source_stance_velocity
        self.optimizer_diagnostic = optimizer_diagnostic
        self.solver = solver
        self.max_backtracks = int(max_backtracks)
        self.solver_diagnostic = solver_diagnostic
        self.motion_records = []

    @torch.no_grad()
    def record_geometry(self, geometry, objective, parameters, window, step):
        """Observe a fixed-order transform decomposition without changing inference."""
        translation_parameters = torch.zeros_like(parameters)
        translation_parameters[..., :3] = parameters[..., :3]
        common_parameters = translation_parameters.clone()
        common_parameters[..., 3] = parameters[..., 3]
        states = {
            'source': objective.anchor,
            'translation_only': geometry.decode(translation_parameters),
            'common': geometry.decode(common_parameters),
            'corrected': geometry.decode(parameters),
        }
        self.motion_records.append({
            'window': window, 'step': step,
            'stance_mask': objective.stance.detach().cpu().clone(),
            'source_floor_height_m': objective.floor_height.detach().cpu().clone(),
            'source_floor_sample_count': objective.floor_sample_count.detach().cpu().clone(),
            'parameters': parameters.detach().cpu().clone(),
            'states': {
                name: {key: state[key].detach().cpu().clone() for key in (
                    'human', 'object_translation_world', 'object_rotation_world',
                    'translation', 'yaw', 'local_delta',
                )} for name, state in states.items()
            },
        })

    def correct(self, sampler, current, previous_x0, timesteps, context,
                rest_offsets, clean, step):
        if step not in self.steps:
            return clean
        if current.is_cuda:
            torch.cuda.synchronize(current.device)
            torch.cuda.reset_peak_memory_stats(current.device)
        started = time.perf_counter()
        geometry, objective, _, _ = relational_problem(
            sampler, current, previous_x0, timesteps, context, rest_offsets, clean,
            source_floor=self.source_floor,
            source_stance_velocity=self.source_stance_velocity,
        )
        trace = None
        if self.solver_diagnostic:
            from .relational_diagnostics import RelationalArmijoTrace, diagnose_relational_solver
            trace = RelationalArmijoTrace(self.learning_rate, self.max_backtracks)
        elif self.optimizer_diagnostic:
            from .relational_diagnostics import RelationalOptimizerTrace, diagnose_relational_optimizer
            trace = RelationalOptimizerTrace()
        encoded, parameters, metrics = optimize_relational_cells(
            geometry, objective, self.optimizer_steps, self.learning_rate,
            cells=(self.cell,), include_floor=self.include_floor, trace=trace,
            solver=self.solver, max_backtracks=self.max_backtracks,
        )
        if current.is_cuda:
            torch.cuda.synchronize(current.device)
        elapsed = time.perf_counter() - started
        names = list(metrics)
        values = torch.stack([metrics[name][0] for name in names]).detach().cpu().tolist()
        self.records.append({
            'window': sampler.inner_hoi.sample_calls, 'step': step,
            'seconds': elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(current.device) if current.is_cuda else None,
            'history_exact': torch.equal(encoded[:, :2], clean[:, :2]),
            'contact_exact': torch.equal(encoded[..., 228:], clean[..., 228:]),
            'metrics': dict(zip(names, values)),
        })
        if self.record_motion:
            self.record_geometry(geometry, objective, parameters,
                                 sampler.inner_hoi.sample_calls, step)
        if self.solver_diagnostic:
            if current.is_cuda:
                torch.cuda.synchronize(current.device)
            diagnostic_started = time.perf_counter()
            self.motion_records[-1]['solver_diagnostic'] = diagnose_relational_solver(
                geometry, objective, trace, self.cell, self.include_floor,
            )
            if current.is_cuda:
                torch.cuda.synchronize(current.device)
            self.records[-1]['solver_diagnostic_seconds'] = time.perf_counter() - diagnostic_started
            self.records[-1]['solver_diagnostic_peak_allocated_bytes'] = (
                torch.cuda.max_memory_allocated(current.device) if current.is_cuda else None)
        elif self.optimizer_diagnostic:
            if current.is_cuda:
                torch.cuda.synchronize(current.device)
            diagnostic_started = time.perf_counter()
            self.motion_records[-1]['optimizer_diagnostic'] = diagnose_relational_optimizer(
                geometry, objective, trace, self.optimizer_steps, self.learning_rate,
                self.cell, self.include_floor,
            )
            if current.is_cuda:
                torch.cuda.synchronize(current.device)
            self.records[-1]['optimizer_diagnostic_seconds'] = time.perf_counter() - diagnostic_started
            self.records[-1]['optimizer_diagnostic_peak_allocated_bytes'] = (
                torch.cuda.max_memory_allocated(current.device) if current.is_cuda else None)
        return encoded

    def audit_dict(self):
        return {
            'cell': self.cell, 'steps': list(self.steps),
            'optimizer_steps': self.optimizer_steps, 'learning_rate': self.learning_rate,
            'include_floor': self.include_floor,
            'source_floor': self.source_floor,
            'source_stance_velocity': self.source_stance_velocity,
            'solver': self.solver, 'max_backtracks': self.max_backtracks,
            'solver_diagnostic': self.solver_diagnostic,
            'solver_shadow_gradient_steps': 20 * len(self.records) if self.solver_diagnostic else 0,
            'optimizer_diagnostic': self.optimizer_diagnostic,
            'shadow_optimizer_gradient_steps': self.optimizer_steps * len(self.records) if self.optimizer_diagnostic else 0,
            'optimizer_gradient_steps': (
                sum(int(record['metrics']['gradient_evaluations']) for record in self.records)
                if self.solver == 'armijo' else self.optimizer_steps * len(self.records)),
            'insertion': 'clean_before_posterior_and_previous_x0',
            'object_pose': 'hoi_with_common_human_object_transform',
            'calls': len(self.records), 'records': self.records,
        }
