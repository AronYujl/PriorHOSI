"""The preregistered gating operator.

    x_hat_0,h = G * x_hat_0^HSI + (1 - G) * x_hat_0^HOI

``G`` is a per-(batch, frame, channel) gate in [0, 1].  Two anchors define the
operator and are asserted as properties rather than argued for in prose:

    G == 0  ->  the HOI expert's output, bit for bit
    G == 1  ->  the HSI expert's output, bit for bit

Both anchors short-circuit: at ``G == 0`` the HSI tensor is never read, so the
anchor row is reproducible without an HSI checkpoint and cannot be perturbed by
a non-finite value on the unused side.  That is a correctness property, not an
optimization -- ``0 * nan`` is ``nan``, so the naive arithmetic would let a
dead expert contaminate an anchor that is meant to be identical to running the
other expert alone.

There is no HOSI ground truth by design (``docs/plan/OVERVIEW.md``), so nothing
here is fit against a target; ``G`` is produced by a mixer that reads only the
expert outputs (MixerMDM's modularity property, 2504.01019), which is what
keeps the experts swappable without retraining it.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ExpertOutputs:
    """One denoising step's x_hat_0 from each expert, in the shared contract.

    Both tensors are ``[B, window_frames, 232]`` in the frozen cross-branch
    representation (``code/priors/core``).  ``hsi`` is ``None`` on the HOI-alone
    anchor and ``hoi`` is ``None`` on the HSI-alone anchor: an absent expert is
    absent, not a zero tensor, so an anchor row cannot silently average against
    a tensor of zeros.
    """

    hoi: Optional[torch.Tensor] = None
    hsi: Optional[torch.Tensor] = None

    def present(self):
        """The experts that actually produced a tensor, in contract order."""
        names = []
        if self.hoi is not None:
            names.append('hoi')
        if self.hsi is not None:
            names.append('hsi')
        return tuple(names)


def gate_is_identity(gate, value):
    """True when ``gate`` is uniformly ``value`` (0 or 1), so an anchor applies.

    A Python/numpy scalar, a 0-d tensor and a full tensor all answer the same
    question.  Anything not exactly ``value`` everywhere returns False, so a
    gate that is 0 on all but one channel takes the full arithmetic path.
    """
    if value not in (0, 1):
        raise ValueError(f'anchor value must be 0 or 1, got {value!r}')
    if gate is None:
        return False
    if isinstance(gate, torch.Tensor):
        if gate.numel() == 0:
            return False
        return bool(torch.all(gate == value).item())
    return bool(gate == value)


def compose_x0(outputs, gate, state=None):
    """Blend two experts' x_hat_0 under ``gate``.

    ``state`` is RESERVED and unused.  It is the slot the LLM state machine
    (CLoSD-style) will fill with the discrete task state that selects or
    conditions the gate; it is accepted now so that adding the state machine
    does not change this signature or any call site.  Passing anything other
    than None today raises, so no caller can come to depend on a meaning this
    function does not yet implement.
    """
    if state is not None:
        raise NotImplementedError(
            'compose_x0 accepts `state` only as a reserved parameter; the LLM '
            'state machine is not implemented, so no state may be supplied yet'
        )
    if not isinstance(outputs, ExpertOutputs):
        raise TypeError(f'outputs must be ExpertOutputs, got {type(outputs)!r}')

    # Anchors first, and before any shape or dtype check on the unused side:
    # G == 0 must reproduce "run HOIPrior alone" exactly, including on a config
    # that has no HSI expert loaded at all.
    if gate_is_identity(gate, 0):
        if outputs.hoi is None:
            raise ValueError('gate is identically 0 but the HOI expert produced nothing')
        return outputs.hoi
    if gate_is_identity(gate, 1):
        if outputs.hsi is None:
            raise ValueError('gate is identically 1 but the HSI expert produced nothing')
        return outputs.hsi

    if outputs.hoi is None or outputs.hsi is None:
        raise ValueError(
            'a non-anchor gate needs both experts; present: '
            f'{outputs.present()}'
        )
    if outputs.hoi.shape != outputs.hsi.shape:
        raise ValueError(
            f'expert output shapes disagree: hoi {tuple(outputs.hoi.shape)} '
            f'vs hsi {tuple(outputs.hsi.shape)}'
        )
    if isinstance(gate, torch.Tensor):
        try:
            torch.broadcast_shapes(gate.shape, outputs.hoi.shape)
        except RuntimeError as error:
            raise ValueError(
                f'gate shape {tuple(gate.shape)} does not broadcast against '
                f'expert output shape {tuple(outputs.hoi.shape)}'
            ) from error
        if bool(((gate < 0) | (gate > 1)).any().item()):
            raise ValueError('gate must lie in [0, 1]')
    elif not (0.0 <= float(gate) <= 1.0):
        raise ValueError('gate must lie in [0, 1]')

    return gate * outputs.hsi + (1.0 - gate) * outputs.hoi
