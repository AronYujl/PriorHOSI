"""The preregistered gating operator.

    x_hat_0,h = G * x_hat_0^HSI + (1 - G) * x_hat_0^HOI

``G`` is a per-(batch, frame, channel) gate in [0, 1], and it is ALWAYS masked to
the human channels.  Channels 216:232 -- object translation, object rotation and
contact -- come from the HOI expert at every value of ``G``, ``G == 1`` included.
That is a hard requirement of the operator on this benchmark, not a knob; see
"Why the mask is not optional" below.

One anchor is exact and bitwise:

    G == 0  ->  the HOI expert's output, bit for bit

It short-circuits: at ``G == 0`` the HSI tensor is never read, so the anchor row
is reproducible without an HSI checkpoint and cannot be perturbed by a non-finite
value on the unused side.  That is a correctness property, not an optimization --
``0 * nan`` is ``nan``, so the naive arithmetic would let a dead expert
contaminate an anchor that is meant to be identical to running the other expert
alone.

REVISION TO THE PREREGISTRATION (2026-08-30, on the user's instruction).  The
preregistration also declared ``G == 1 -> the HSI expert's output, bit for bit``.
That anchor does not hold on HOSI-test and is WITHDRAWN.  HSIPrior receives no
gradient on 216:232, so "HSIPrior alone" here is not "a scene expert generating
without object help"; it is an object channel filled with the training target's
exact zeros -- an object at the centre of the normalized box (up to 4.481 m of L2
displacement), a zero 3x3 matrix that has no polar factor at all, and every
contact label at 0.  That row is not a baseline for anything.  The anchor now
reads:

    G == 1  ->  the HSI expert on 0:216, the HOI expert on 216:232

which is exactly what the masked arithmetic produces, so the operator is now
CONTINUOUS at 1: the value at 1 equals the limit from below.  The withdrawn
short-circuit skipped the mask, which meant a gate reaching 1.0 anywhere hit a
different operator than a gate at 0.999 -- a discontinuity in a quantity a
learned gate is free to move through.

There is no HOSI ground truth by design (``docs/plan/OVERVIEW.md``), so nothing
here is fit against a target; ``G`` is produced by a mixer that reads only the
expert outputs (MixerMDM's modularity property, 2504.01019), which is what
keeps the experts swappable without retraining it.

Why the mask is not optional.  HSI is never supervised on channels 216:232:
``priors/hsi/data.py:253`` calls ``codec.encode`` with no object arguments and
``core/window_codec.py:215`` starts from ``torch.zeros``, so every HSI training
target is exactly zero there.  Blending against that zero is not a compromise
between two opinions, it is a pull toward the centre of the normalized box:

* object translation 216:219 -- the error is ``G * |x| * half_range`` per axis,
  and the box half range is [3.0880, 1.0918, 3.0581] m, i.e. up to 4.481 m of
  L2 object displacement per unit of gate.
* contact 228:232 -- the composed value is ``contact_HOI * (1 - G)``, so the
  gate scales every contact label down monotonically, straight into the metric
  the contact budget is written against.
* object rotation 219:228 -- measured invariant: a uniform positive scale
  leaves the polar factor unchanged, so ``project_to_so3((1-G) * R) == R`` to
  9.99e-16 over 64x4 random cases.  The mask covers these nine channels for
  uniformity, not because they need it.  The one value that WOULD degenerate is
  ``G == 1``, where the scale is 0 and the zero matrix has no polar factor at
  all -- and that value is now inside the mask's protection rather than outside
  it, which is the second reason the ``G == 1`` short-circuit had to go.

So ``human_gate_mask()`` is 1 on 0:216 and 0 on 216:232, and it is MANDATORY.
The object and contact channels always come from HOI.  ``channel_mask=None``
is refused: it produced rows whose object channel was pulled toward the box
centre by the gate, and no such row is valid.  A caller may still pass its own
mask tensor -- for a per-frame or per-batch-element mask -- but it must be
exactly zero on 216:232, which ``_validate_channel_mask`` enforces.
Measurements behind the numbers: ``.claude/scratch/phase2-blend/blend_space.json``.
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
    anchor (``G == 0``): an absent expert is absent, not a zero tensor, so an
    anchor row cannot silently average against a tensor of zeros.

    ``hoi`` being None is no longer a usable state at any gate value.  The mask
    on 216:232 makes the HOI prediction a REQUIRED input even at ``G == 1``,
    which is the 2026-08-30 revision recorded in the module docstring: there is
    no HSI-alone row on this benchmark, because HSI has no object channel to
    produce.  The field stays Optional so the failure is a checked ValueError
    naming that revision rather than an AttributeError in the arithmetic.
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


def _validate_channel_mask(channel_mask, reference):
    """Resolve ``channel_mask`` to a tensor that is provably zero on 216:232.

    The invariant this enforces is the operator's, not this function's: the
    object and contact channels come from the HOI expert at EVERY gate value,
    because HSIPrior's training target is exactly zero on all sixteen of them.
    Enforcing it here rather than trusting the caller is the point -- ``None``
    (the old "literal unmasked operator") and a hand-built all-ones tensor are
    the same invalid row, so both are refused by one check on the tensor.
    """
    if channel_mask is None:
        raise ValueError(
            'channel_mask=None is refused: the gate must be masked to the human '
            'channels at every gate value.  HSIPrior is never supervised on '
            '216:232, so an unmasked blend pulls the object toward the centre of '
            'the normalized box (up to 4.481 m of L2 displacement per unit gate) '
            'and scales every contact label by (1-G).  No such row is valid; see '
            "the 2026-08-30 revision in this module's docstring."
        )
    if isinstance(channel_mask, str):
        if channel_mask != 'human':
            raise ValueError(
                f"channel_mask must be 'human' or a tensor, got {channel_mask!r}"
            )
        return human_gate_mask(device=reference.device, dtype=reference.dtype)
    if not isinstance(channel_mask, torch.Tensor):
        raise TypeError(
            f"channel_mask must be 'human' or a tensor, got {type(channel_mask)!r}"
        )
    mask = channel_mask.to(device=reference.device, dtype=reference.dtype)
    if mask.shape[-1] != REPRESENTATION.dimension:
        raise ValueError(
            f'channel_mask must end in {REPRESENTATION.dimension} channels so the '
            f'object/contact block is addressable, got {tuple(mask.shape)}'
        )
    try:
        torch.broadcast_shapes(mask.shape, reference.shape)
    except RuntimeError as error:
        raise ValueError(
            f'channel_mask shape {tuple(mask.shape)} does not broadcast against '
            f'expert output shape {tuple(reference.shape)}'
        ) from error
    if bool((mask[..., OBJECT_CHANNEL_START:] != 0).any().item()):
        raise ValueError(
            'channel_mask must be exactly 0 on channels '
            f'{OBJECT_CHANNEL_START}:{REPRESENTATION.dimension}: the object and '
            'contact channels come from the HOI expert at every gate value'
        )
    return mask


def compose_x0(outputs, gate, state=None, channel_mask='human'):
    """Blend two experts' x_hat_0 under ``gate``.

    ``channel_mask`` defaults to ``'human'``, which multiplies the gate by
    ``human_gate_mask()`` so the object and contact channels come from HOI
    whatever the gate says.  It is MANDATORY: ``None`` is refused, and a caller
    supplying its own tensor must make it zero on 216:232.  A gate value of 1 is
    inside that mask, not outside it.

    One anchor is exact: ``G == 0`` returns the HOI tensor bit for bit, without
    reading the HSI side at all.  ``G == 1`` takes the ordinary masked path, so
    it is HSI on 0:216 and HOI on 216:232 and equals its own limit from below.
    The old bitwise ``G == 1 -> HSI everywhere`` short-circuit was withdrawn on
    2026-08-30; the module docstring records why.

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

    # The mask is validated BEFORE the G == 0 short-circuit, deliberately.
    # `mixer_channel_mask: null` alongside the default `mixer_gate: 0` would
    # otherwise be accepted by exactly the config that is the SAFE default, and
    # fail only once someone raised the gate.  It reads nothing from the unused
    # expert, so the anchor's nan protection below is untouched.
    reference = outputs.hoi if outputs.hoi is not None else outputs.hsi
    if reference is None:
        raise ValueError('compose_x0 needs at least one expert output')
    mask = _validate_channel_mask(channel_mask, reference)

    # The anchor comes before any shape or dtype check on the unused side:
    # G == 0 must reproduce "run HOIPrior alone" exactly, including on a config
    # that has no HSI expert loaded at all.
    if gate_is_identity(gate, 0):
        if outputs.hoi is None:
            raise ValueError('gate is identically 0 but the HOI expert produced nothing')
        return outputs.hoi

    # No G == 1 short-circuit.  Every gate above 0 takes the masked arithmetic,
    # so 216:232 comes from HOI at 1 exactly as it does at 0.999.
    if outputs.hoi is None:
        raise ValueError(
            'every gate above 0 needs the HOI expert, including gate == 1: the '
            'object and contact channels 216:232 always come from HOI because '
            'HSIPrior is never supervised on them.  The preregistered '
            '"G == 1 -> HSIPrior alone" anchor was withdrawn on 2026-08-30 for '
            f'that reason.  Present: {outputs.present()}'
        )
    if outputs.hsi is None:
        raise ValueError(
            f'a non-zero gate needs the HSI expert; present: {outputs.present()}'
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

    gate = gate * mask.to(device=outputs.hoi.device, dtype=outputs.hoi.dtype)
    return gate * outputs.hsi + (1.0 - gate) * outputs.hoi
