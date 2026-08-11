"""Frozen cross-branch contract shared by HOIPrior, HSIPrior and the mixer.

Nothing in this package may import from ``priors.hoi`` or ``priors.hsi`` at
module import time.  Every module here is content-hashed by
``tests/core/test_contract_freeze.py``: changing one is by definition
cross-branch communication and requires the user's explicit approval plus a
matching update on the other expert branch.

The package is deliberately side-effect free and re-exports nothing, so
``import priors.core.<module>`` never pulls in an expert package.
"""
