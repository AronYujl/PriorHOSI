"""Phase 2 composition: the gating operator and the per-expert evaluator adapters.

Nothing in this package trains.  It composes two frozen, path-disjoint expert
priors -- HOIPrior (``code/priors/hoi``) and HSIPrior (``code/priors/hsi``) --
and adapts each expert's sampler to the shared HOSI-test evaluator so that the
single-expert anchor rows and the composed row are produced by one measurement
protocol.
"""

from .composition import ExpertOutputs, compose_x0, gate_is_identity
from .hoi_adapter import HOIExpertSamplerAdapter, SceneBlindDatasetView

__all__ = [
    'ExpertOutputs',
    'compose_x0',
    'gate_is_identity',
    'HOIExpertSamplerAdapter',
    'SceneBlindDatasetView',
]
