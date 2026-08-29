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

The gate is masked to the human channels, and that is forced by measurement
rather than chosen.  HSI is never supervised on channels 216:232:
``priors/hsi/data.py:253`` calls ``codec.encode`` with no object arguments and
``core/window_codec.py:215`` starts from ``torch.zeros``, so every HSI training
target is exactly zero there.  Blending against that zero is not a compromise
between two opinions, it is a pull toward the origin of the normalized box:

* object translation 216:219 -- the error is ``G * |x| * half_range`` per axis,
  and the box half range is [3.0880, 1.0918, 3.0581] m, i.e. up to 4.481 m of
  L2 object displacement per unit of gate.
* contact 228:232 -- the composed value is ``contact_HOI * (1 - G)``, so the
  gate scales every contact label down monotonically, straight into the metric
  the contact budget is written against.
* object rotation 219:228 -- measured invariant: a uniform positive scale
  leaves the polar factor unchanged, so ``project_to_so3((1-G) * R) == R`` to
  9.99e-16 over 64x4 random cases.  The mask covers these nine channels for
  uniformity, not because they need it, and only ``G == 1`` degenerates (the
  zero matrix has no polar factor).

So ``HUMAN_GATE_MASK`` is 1 on 0:216 and 0 on 216:232, and it is the default.
The object and contact channels always come from HOI.  Measurements behind the
numbers: ``.claude/scratch/phase2-blend/blend_space.json``.
"""

from dataclasses import dataclass
from typing import Optional

import torch

from priors.core.representation import REPRESENTATION

#: The first channel HSI is not supervised on.  Named rather than spelled 216 so
#: that a change to the frozen representation moves the mask with it.
OBJECT_CHANNEL_START = REPRESENTATION.field('object_translation').start


def human_gate_mask(*, device=None, dtype=torch.float32):
    """A [232] mask that is 1 on the human channels and 0 on object/contact.

    Multiplying any gate by this is what keeps the object and contact channels
    equal to the HOI expert's prediction regardless of what the gate producer
    emits, so scene-awareness cannot leak into channels no HSI checkpoint has
    ever been supervised on.
    """
    mask = torch.zeros(REPRESENTATION.dimension, device=device, dtype=dtype)
    mask[:OBJECT_CHANNEL_START] = 1
    return mask


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


def compose_x0(outputs, gate, state=None, channel_mask='human'):
    """Blend two experts' x_hat_0 under ``gate``.

    ``channel_mask`` defaults to ``'human'``, which multiplies the gate by
    ``human_gate_mask()`` so the object and contact channels come from HOI
    whatever the gate says.  Pass ``None`` to apply the gate to all 232 channels,
    which is the literal preregistered operator and is what the HSI-alone anchor
    needs; pass a tensor to supply your own mask.

    The two anchors are exact and UNMASKED: ``G == 0`` returns the HOI tensor and
    ``G == 1`` returns the HSI tensor, bit for bit, whatever ``channel_mask``
    says.  They are diagnostics -- "run this one expert alone and report it" --
    not compositions, and their whole value is being bitwise identical to the
    single-expert row.  That does make the operator discontinuous at ``G == 1``
    under the default mask: the limit from below is HSI-on-human with
    HOI-on-object, while exactly 1 is HSI everywhere, including HSI's
    never-supervised zeros on 216:232.  A learned gate reaches that only by
    emitting exactly 1.0 on all 232 channels of every batch element at once,
    which is the HSI-alone case by any reading.

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

    if channel_mask is not None:
        if isinstance(channel_mask, str):
            if channel_mask != 'human':
                raise ValueError(
                    f"channel_mask must be 'human', None or a tensor, got {channel_mask!r}"
                )
            mask = human_gate_mask(
                device=outputs.hoi.device, dtype=outputs.hoi.dtype,
            )
        elif isinstance(channel_mask, torch.Tensor):
            mask = channel_mask.to(device=outputs.hoi.device, dtype=outputs.hoi.dtype)
            try:
                torch.broadcast_shapes(mask.shape, outputs.hoi.shape)
            except RuntimeError as error:
                raise ValueError(
                    f'channel_mask shape {tuple(mask.shape)} does not broadcast '
                    f'against expert output shape {tuple(outputs.hoi.shape)}'
                ) from error
        else:
            raise TypeError(
                f"channel_mask must be 'human', None or a tensor, got {type(channel_mask)!r}"
            )
        gate = gate * mask

    return gate * outputs.hsi + (1.0 - gate) * outputs.hoi
