"""Phase 2 composition: the gating operator and the per-expert evaluator adapters.

Nothing in this package trains.  It composes two frozen, path-disjoint expert
priors -- HOIPrior (``code/priors/hoi``) and HSIPrior (``code/priors/hsi``) --
and adapts each expert's sampler to the shared HOSI-test evaluator so that the
single-expert anchor rows and the composed row are produced by one measurement
protocol.
"""

from .body_groups import (
    DEFAULT_BODY_GROUP_GATE,
    GROUP_NAMES,
    POSITION_GROUPS,
    ROTATION_GROUPS,
    body_group_channel_gate,
    describe_body_groups,
)
from .composed_sampler import HOSIComposedSampler
from .composition import (
    OBJECT_CHANNEL_START,
    ExpertOutputs,
    compose_x0,
    gate_is_identity,
    human_gate_mask,
)
from .gates import (
    BodyGroupGate,
    ChannelBlockGate,
    ConstantGate,
    ObjectConditionedGate,
    ScheduleGate,
)
from .hoi_adapter import HOIExpertSamplerAdapter, SceneBlindDatasetView

__all__ = [
    'DEFAULT_BODY_GROUP_GATE',
    'GROUP_NAMES',
    'OBJECT_CHANNEL_START',
    'POSITION_GROUPS',
    'ROTATION_GROUPS',
    'BodyGroupGate',
    'ChannelBlockGate',
    'ConstantGate',
    'ExpertOutputs',
    'HOIExpertSamplerAdapter',
    'HOSIComposedSampler',
    'ObjectConditionedGate',
    'SceneBlindDatasetView',
    'ScheduleGate',
    'body_group_channel_gate',
    'compose_x0',
    'describe_body_groups',
    'gate_is_identity',
    'human_gate_mask',
]
