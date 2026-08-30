"""Gate producers for the composed sampler.

A gate is anything the composed sampler can call as
``gate(step=..., current=..., hoi=..., hsi=...) -> scalar or tensor in [0,1]``,
or a bare scalar/tensor.  The keyword signature is the MixerMDM modularity
property made mechanical: a gate sees the step index and both experts' x_hat_0,
and nothing else.  It never sees a model, a weight or an internal feature, so
either expert can be swapped without retraining the gate.

Everything here is a REFERENCE gate: fixed rules, no learned parameters. They
exist to bracket what a learned gate has to beat, and two of them are anchors of
the operator rather than candidates.

Why object identity is the first conditioning signal worth trying, and why that
is not the analogous mistake to the HSI dose story: on this benchmark the object
penetration mass concentrates by OBJECT, and the concentration is a property of
the episode rather than of the model.  Measured on all 469 HOSI-test episodes,
Spearman rho between the HOI-alone row and the July released row is +0.827 for
object penetration and +0.825 for human penetration, and the per-object means
span 37x (clothesstand 128.5, tripod 97.8, smalltable 55.4, monitor 29.6,
smallbox 19.4, floorlamp 10.0, suitcase 3.5).  clothesstand and tripod are 29%
of episodes but 65.7% of the mass.  Object identity is a task input known before
the first denoising step, so a gate can condition on it -- unlike the HSI
guidance-dose case, where the corresponding rank correlation was +0.056 and a
uniform intervention taxed 225 episodes that never needed it.
"""

from typing import Dict, Optional, Sequence

import torch

from .composition import human_gate_mask


class ConstantGate:
    """One value everywhere.  ``ConstantGate(0)`` and ``ConstantGate(1)`` are the anchors."""

    def __init__(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f'gate value must lie in [0, 1], got {value}')
        self.value = value

    def __call__(self, *, step, current, hoi, hsi):
        del step, current, hoi, hsi
        return self.value

    def is_identically_zero_at(self, step):
        """Lets the sampler skip the HSI expert's two forward passes entirely."""
        del step
        return self.value == 0.0

    def describe(self):
        return {'kind': 'constant', 'value': self.value}


class ScheduleGate:
    """A gate that varies with the reverse step index only.

    ``late`` (the default) hands the early, structure-setting steps to HOI and
    lets HSI act only near the end; ``early`` does the reverse.  Which way round
    is an open empirical question, not a settled one: the argument for ``late``
    is that object manipulation is the harder constraint and should set the
    trajectory's coarse structure, and the argument for ``early`` is that scene
    collision is decided by coarse structure and is expensive to fix late.  Both
    are cheap to measure once an HSI checkpoint exists.
    """

    def __init__(self, timesteps, peak=0.5, mode='late'):
        if mode not in ('late', 'early'):
            raise ValueError(f"mode must be 'late' or 'early', got {mode!r}")
        peak = float(peak)
        if not 0.0 <= peak <= 1.0:
            raise ValueError(f'peak must lie in [0, 1], got {peak}')
        self.timesteps = int(timesteps)
        self.peak = peak
        self.mode = mode

    def __call__(self, *, step, current, hoi, hsi):
        del current, hoi, hsi
        return self._value_at(step)

    def _value_at(self, step):
        progress = 1.0 - (float(step) / max(self.timesteps - 1, 1))
        fraction = progress if self.mode == 'late' else 1.0 - progress
        return self.peak * fraction

    def is_identically_zero_at(self, step):
        """True on the steps this schedule hands entirely to HOI.

        Worth having rather than merely correct: ``late`` is exactly 0 at
        ``step == timesteps - 1`` and ``early`` is exactly 0 at ``step == 0``, and
        a schedule reaching 0 over a RANGE (peak 0, or a future shape with a flat
        head) then costs one forward per step instead of three.
        """
        return self._value_at(step) == 0.0

    def describe(self):
        return {
            'kind': 'schedule', 'mode': self.mode, 'peak': self.peak,
            'timesteps': self.timesteps,
        }


class ObjectConditionedGate:
    """Per-batch-element gate keyed on the episode's object name.

    The gate is a [B, 1, 1] tensor, so it varies across the batch and is constant
    over frames and channels; the human channel mask is applied by ``compose_x0``
    on top.  ``default`` covers an object with no entry.

    This is the reference gate the penetration measurement points at: object
    identity is known at episode start, the penetration mass concentrates 37x by
    object, and the concentration transfers between models (rho +0.827), so a
    per-object dose can spend scene-awareness where it is needed without taxing
    the episodes that are already clean.
    """

    def __init__(self, values: Dict[str, float], default: float = 0.0):
        for name, value in values.items():
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f'gate value for {name!r} must lie in [0, 1], got {value}')
        if not 0.0 <= float(default) <= 1.0:
            raise ValueError(f'default gate value must lie in [0, 1], got {default}')
        self.values = {str(name): float(value) for name, value in values.items()}
        self.default = float(default)
        self._object_names: Optional[Sequence[str]] = None

    def set_object_names(self, names: Sequence[str]):
        """Supply this window's per-batch-element object names.

        The composed sampler's gate signature carries only expert outputs, by
        design, so episode metadata is injected here rather than smuggled through
        the call.  The evaluator knows the object name before the first step.
        """
        self._object_names = [str(name) for name in names]
        return self

    def __call__(self, *, step, current, hoi, hsi):
        del step, hsi
        if self._object_names is None:
            raise ValueError(
                'ObjectConditionedGate needs set_object_names() before sampling'
            )
        reference = hoi if hoi is not None else current
        batch = reference.shape[0]
        if len(self._object_names) != batch:
            raise ValueError(
                f'have {len(self._object_names)} object names for a batch of {batch}'
            )
        values = [self.values.get(name, self.default) for name in self._object_names]
        return torch.tensor(
            values, device=reference.device, dtype=reference.dtype,
        ).reshape(batch, 1, 1)

    def is_identically_zero_at(self, step):
        """Only when every object in THIS window is gated to zero.

        Batch-dependent, so it needs the names; without them it answers False and
        the expert runs, which is the conservative direction.
        """
        del step
        if self._object_names is None:
            return False
        return all(
            self.values.get(name, self.default) == 0.0
            for name in self._object_names
        )

    def describe(self):
        return {
            'kind': 'object_conditioned', 'values': dict(self.values),
            'default': self.default,
            'object_names': list(self._object_names) if self._object_names else None,
        }


class ChannelBlockGate:
    """A fixed per-channel gate, for ablating which channels HSI may touch.

    Useful for the measurement the channel mask makes obvious but does not
    answer: the mask keeps object and contact at HOI, and within the remaining
    human channels one may still want to ask whether HSI should drive joint
    POSITIONS (0:84) but not joint ROTATIONS (84:216), since scene collision is a
    positional constraint.
    """

    def __init__(self, positions=0.0, rotations=0.0):
        for name, value in (('positions', positions), ('rotations', rotations)):
            if not 0.0 <= float(value) <= 1.0:
                raise ValueError(f'{name} must lie in [0, 1], got {value}')
        self.positions = float(positions)
        self.rotations = float(rotations)

    def __call__(self, *, step, current, hoi, hsi):
        del step, hsi
        reference = hoi if hoi is not None else current
        gate = torch.zeros(
            reference.shape[-1], device=reference.device, dtype=reference.dtype,
        )
        gate[:84] = self.positions
        gate[84:216] = self.rotations
        return gate * human_gate_mask(device=gate.device, dtype=gate.dtype)

    def is_identically_zero_at(self, step):
        del step
        return self.positions == 0.0 and self.rotations == 0.0

    def describe(self):
        return {
            'kind': 'channel_block', 'positions': self.positions,
            'rotations': self.rotations,
        }
