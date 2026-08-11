"""Independent State-Compositional Prior scaffolding.

Three layers:

* ``priors.core`` - the frozen contract both experts and the future mixer share;
* ``priors.hoi``  - the HOIPrior expert, owned by ``phase/01b-hoi``;
* ``priors.hsi``  - the HSIPrior expert, owned by ``phase/01c-hsi``.

The seven historical package-level names are still available, but the five that
live in an expert package or in ``core.expert_api`` resolve lazily through
``__getattr__``.  Eagerly importing an expert here would make
``import priors.core.<module>`` fail on a branch that has deleted the other
expert's package, which is exactly what the three-layer split exists to prevent.
"""

from .core.representation import REPRESENTATION, masked_reconstruction_loss

__all__ = [
    "HSIPrior", "HOIPrior", "REPRESENTATION", "assert_parameter_independence",
    "build_expert", "load_trained_hoi_prior", "masked_reconstruction_loss",
]

_LAZY = {
    "assert_parameter_independence": ".core.expert_api",
    "build_expert": ".core.expert_api",
    "HOIPrior": ".hoi.models",
    "load_trained_hoi_prior": ".hoi.models",
    "HSIPrior": ".hsi.models",
}


def __getattr__(name):
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 'priors' has no attribute {name!r}")
    from importlib import import_module
    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
