"""Four-cell constrained-window experiment on generated HOI hypotheses."""

import time

import torch

from .diagnostics import HSIInputDiagnostic, decode_human
from .kinematic_composition import _apply_rotation
from .relational import CELL_KEYS, RelationalGeometry, RelationalObjective, optimize_relational_cells


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
        cond, uncond = sampler._hsi_prediction_pair(current, previous_x0, timesteps, context)
        object_name = context['seq_name_dict'][0].split('_')[1]
        vertices = context['obj_rest_verts'][object_name]
        indices = torch.linspace(0, len(vertices) - 1, 128, device=vertices.device).long()
        geometry = RelationalGeometry(hoi_clean, sampler.dataset, rest_offsets, context, vertices[indices][None])
        conditional_fk = decode_human(cond, sampler.dataset, rest_offsets)[2]
        unconditioned_fk = decode_human(uncond, sampler.dataset, rest_offsets)[2]
        increment = _apply_rotation(context['mat'][:, None, None, :3, :3], conditional_fk - unconditioned_fk)
        zero = hoi_clean.new_zeros(*hoi_clean.shape[:2], geometry.dimension)
        target = geometry.decode(zero)['human'][:, 2:] + increment
        objective = RelationalObjective(geometry, context['scene_flag'], target)
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
