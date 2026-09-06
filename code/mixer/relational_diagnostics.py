"""Four-cell constrained-window experiment on generated HOI hypotheses."""

import time

import torch

from .diagnostics import HSIInputDiagnostic
from .relational import CELL_KEYS, relational_problem, optimize_relational_cells
from .kinematic_composition import _apply_rotation


class RelationalOptimizerTrace:
    """Observe the actual optimizer trajectory without changing its gradients."""

    def __init__(self):
        self.parameters = []
        self.terms = []
        self.losses = []
        self.gradients = []
        self.exp_avg = []
        self.exp_avg_sq = []
        self.initial_term_gradients = {}
        self.settings = {}

    def before_backward(self, parameters, terms, loss, common_terms, optimizer):
        if not self.parameters:
            self.initial_term_gradients = {
                name: torch.autograd.grad(terms[name].sum(), parameters, retain_graph=True)[0].detach().clone()
                for name in common_terms
            }
            group = optimizer.param_groups[0]
            self.settings = {name: group[name] for name in ('lr', 'betas', 'eps', 'weight_decay', 'amsgrad')}
        self._state(parameters, terms, loss)

    def _state(self, parameters, terms, loss):
        self.parameters.append(parameters.detach().clone())
        self.terms.append({name: value.detach().clone() for name, value in terms.items()})
        self.losses.append(loss.detach().clone())

    def after_step(self, parameters, optimizer):
        state = optimizer.state[parameters]
        self.gradients.append(parameters.grad.detach().clone())
        self.exp_avg.append(state['exp_avg'].detach().clone())
        self.exp_avg_sq.append(state['exp_avg_sq'].detach().clone())

    def finish(self, parameters, terms, loss):
        self._state(parameters, terms, loss)

    def payload(self):
        return {
            'settings': self.settings,
            'parameters': torch.stack(self.parameters).cpu(),
            'terms': {name: torch.stack([terms[name] for terms in self.terms]).cpu()
                      for name in self.terms[0]},
            'loss': torch.stack(self.losses).cpu(),
            'gradients': torch.stack(self.gradients).cpu(),
            'exp_avg': torch.stack(self.exp_avg).cpu(),
            'exp_avg_sq': torch.stack(self.exp_avg_sq).cpu(),
            'initial_term_gradients': {name: value.cpu() for name, value in self.initial_term_gradients.items()},
        }


def diagnose_relational_optimizer(geometry, objective, reference, steps, learning_rate,
                                  cell, include_floor):
    """Same-source contact ablation and first-step probe; neither feeds sampling."""

    class ContactOffObjective:
        def evaluate(self, state):
            terms, metrics = objective.evaluate(state)
            terms['contact'] = terms['contact'] * 0
            return terms, metrics

    shadow = RelationalOptimizerTrace()
    _, shadow_parameters, _ = optimize_relational_cells(
        geometry, ContactOffObjective(), steps, learning_rate,
        cells=(cell,), include_floor=include_floor, trace=shadow,
    )
    with torch.no_grad():
        shadow_original_terms, _ = objective.evaluate(geometry.decode(shadow_parameters))
        unnormalized_parameters = -learning_rate * reference.gradients[0]
        unnormalized_terms, _ = objective.evaluate(geometry.decode(unnormalized_parameters))
        source = objective.anchor
        rotation = source['object_rotation_world'][..., None, :, :]
        translation = source['object_translation_world'][..., None, :]
        hands = source['human'][..., 22:24, :]
        roundtrip = hands - (_apply_rotation(rotation, objective.hand_anchor) + translation)
        # Reuse the cached float32 geometry in float64. This separates errors in
        # the cached rotation from arithmetic in the subsequent anchor round trip.
        relative64 = hands.double() - translation.double()
        anchor64 = _apply_rotation(rotation.double().transpose(-1, -2), relative64)
        roundtrip64 = hands.double() - (
            _apply_rotation(rotation.double(), anchor64) + translation.double())
        orthogonality64 = rotation.double() @ rotation.double().transpose(-1, -2) - torch.eye(
            3, device=rotation.device, dtype=torch.float64)
    return {
        'reference': reference.payload(), 'contact_off': shadow.payload(),
        'contact_off_final_original_terms': {name: value.cpu() for name, value in shadow_original_terms.items()},
        'unnormalized_first_proposal': {
            'parameters': unnormalized_parameters.cpu(),
            'terms': {name: value.cpu() for name, value in unnormalized_terms.items()},
        },
        'contact_mask': objective.contact.cpu().clone(),
        'hand_anchor': objective.hand_anchor.cpu().clone(),
        'source_contact_residual': roundtrip.cpu(),
        'source_contact_residual_float64': roundtrip64.cpu(),
        'object_rotation_orthogonality_float64': orthogonality64.cpu(),
        'geometry': {name: value.detach().cpu().clone() for name, value in vars(geometry).items()
                     if torch.is_tensor(value)},
        'hsi_target': objective.hsi_target.cpu().clone(),
        'scene_flag': objective.scene_flag.cpu().clone(),
    }


class RelationalPrototypeDiagnostic(HSIInputDiagnostic):
    probe_name = 'relational_constrained_window'
    contrast_names = CELL_KEYS

    def __init__(self, steps=(10, 1, 0), seed=42, optimizer_steps=20, learning_rate=0.05):
        super().__init__(steps, seed)
        self.optimizer_steps = int(optimizer_steps)
        self.learning_rate = float(learning_rate)

    def begin_window(self, current):
        self.window_index += 1

    def _observe(self, sampler, current, previous_x0, timesteps, context,
                 rest_offsets, hoi_clean, step):
        geometry, objective, cond, uncond = relational_problem(
            sampler, current, previous_x0, timesteps, context, rest_offsets, hoi_clean,
        )
        if current.is_cuda:
            torch.cuda.synchronize(current.device)
            torch.cuda.reset_peak_memory_stats(current.device)
        started = time.perf_counter()
        encoded, parameters, metrics = optimize_relational_cells(
            geometry, objective, self.optimizer_steps, self.learning_rate,
        )
        if current.is_cuda:
            torch.cuda.synchronize(current.device)
        elapsed = time.perf_counter() - started
        names = list(metrics)
        values = torch.stack([metrics[name] for name in names], dim=1).detach().cpu().tolist()
        self.records.append({
            'window_index': self.window_index, 'step': step,
            'contrasts': {cell: dict(zip(names, row)) for cell, row in zip(CELL_KEYS, values)},
            'optimization_batch_seconds': elapsed,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(current.device) if current.is_cuda else None,
        })
        torch.save({
            'current': current.cpu(), 'previous_x0': previous_x0.cpu(), 'hoi_clean': hoi_clean.cpu(),
            'hsi_cond': cond.cpu(), 'hsi_uncond': uncond.cpu(),
            'hsi_input': sampler.hsi_input_view.for_step(current, step).cpu(),
            'cell_names': CELL_KEYS, 'cell_predictions': encoded.cpu(),
            'cell_parameters': parameters.cpu(), 'mat': context['mat'].cpu(),
            'object_rotation_prefix': geometry.prefix.cpu(),
            'object_rotation_reference': geometry.reference.cpu(), 'rest_offsets': rest_offsets.cpu(),
        }, self.output_dir / (
            f'state-{self.episode["canonical_ordinal"]:03d}'
            f'-w{self.window_index:03d}-t{step:03d}.pt'
        ))
