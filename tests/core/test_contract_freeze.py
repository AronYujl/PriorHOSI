"""Freeze ``code/priors/core/`` as the cross-branch contract.

``core/`` is the only code HOIPrior (``phase/01b-hoi``), HSIPrior
(``phase/01c-hsi``) and the future mixer all execute.  Because the two expert
branches are meant to be recombined by a directory graft rather than a git
merge, a change to ``core/`` cannot be reviewed on one branch alone: it is
cross-branch communication by definition.

This module pins the exact bytes of every file under ``core/`` and proves that
``core/`` never depends on an expert package at import time.
"""

import ast
import hashlib
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
CORE = REPO / "code/priors/core"

FREEZE_MESSAGE = textwrap.dedent(
    """
    code/priors/core/ is the FROZEN cross-branch contract shared by HOIPrior,
    HSIPrior and the mixer.  Changing any file under it is by definition
    cross-branch communication, not a local edit: it requires the user's
    explicit approval AND a matching update on the other expert branch, and it
    invalidates the sealed D2-X / D2-AI / W3 comparison points on both branches
    at once.  If the change is genuinely intended, update the expected hash in
    tests/core/test_contract_freeze.py in the same commit and record the
    cross-branch decision in the plan.
    """
).strip()

# sha256 of every tracked file under code/priors/core/ as of the three-layer
# split of code/priors (Phase 1B, 2026-08-11).
EXPECTED_SHA256 = {
    "__init__.py": "6d14fb16affe4d360cf6d281fa1f8432a68e2e4f307988b38d32608e4c653958",
    "contracts.py": "17fb5dec1c99896cb3d8cae1aa5b8091a9823d9bdaf7c3fbddad9bffbc61f77f",
    "ddpm.py": "41268a193e92d39ca58b48ce70c6a0e140d81337004839af217fa9b40dd19e5b",
    "diffusion_schedule.py": "b4d9cf74174d63de30f75acb3f687e87f824e75b147f3a2efcfd3d76befd5b09",
    "expert_api.py": "7f0f336502d22f50367f2e70b2272a3278b7b828040287921e34dff0806f640c",
    "representation.py": "a510b4ddfb4f6b60e3917219a87898a717d91cbfb858d0993f3e054b5a1abf74",
    "window_codec.py": "74ed335330425bbc0941d99f9f816c8b81d0eebbc59d2d680770f964a3312b53",
}

FORBIDDEN_PREFIXES = ("priors.hoi", "priors.hsi")


def _core_files():
    return sorted(
        path for path in CORE.rglob("*.py")
        if "__pycache__" not in path.parts
    )


class ContractFreezeTests(unittest.TestCase):
    def test_core_file_set_is_exactly_the_frozen_set(self):
        actual = {str(path.relative_to(CORE)) for path in _core_files()}
        self.assertEqual(actual, set(EXPECTED_SHA256), FREEZE_MESSAGE)

    def test_every_core_file_matches_its_frozen_hash(self):
        actual = {
            str(path.relative_to(CORE)):
                hashlib.sha256(path.read_bytes()).hexdigest()
            for path in _core_files()
        }
        self.assertEqual(actual, EXPECTED_SHA256, FREEZE_MESSAGE)


class CoreLayeringTests(unittest.TestCase):
    def test_no_core_module_imports_an_expert_package_at_import_time(self):
        offenders = []
        for path in _core_files():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only; lazy in-function is fine
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    prefix = "." * node.level
                    names = [f"{prefix}{node.module or ''}"]
                for name in names:
                    if name.startswith(("..hoi", "..hsi", ".hoi", ".hsi")) or \
                            name.startswith(FORBIDDEN_PREFIXES):
                        offenders.append(f"{path.name}:{node.lineno} {name}")
        self.assertEqual(offenders, [], FREEZE_MESSAGE)

    def test_core_imports_with_both_expert_packages_blocked(self):
        """Prove, not assert, that ``core`` stands alone.

        ``phase/01c-hsi`` deletes ``priors/hoi``; this blocks both expert
        packages from the import system and imports every core module, which is
        the same failure mode a deleted directory would produce.
        """
        program = textwrap.dedent(
            f"""
            import sys
            sys.path.insert(0, {str(REPO / 'code')!r})

            class Block:
                def find_spec(self, name, path=None, target=None):
                    if name.startswith(("priors.hoi", "priors.hsi")):
                        raise ImportError("expert package blocked: " + name)
                    return None

            sys.meta_path.insert(0, Block())
            import importlib
            for module in (
                "priors",
                "priors.core",
                "priors.core.contracts",
                "priors.core.ddpm",
                "priors.core.diffusion_schedule",
                "priors.core.expert_api",
                "priors.core.representation",
                "priors.core.window_codec",
            ):
                importlib.import_module(module)
            from priors.core.expert_api import build_expert
            try:
                build_expert("hoi")
            except ImportError:
                pass
            else:
                raise SystemExit("build_expert('hoi') must fail when priors.hoi is absent")
            try:
                build_expert("hsi", init_checkpoint="checkpoint/checkpoint.pth")
            except ValueError as error:
                assert "randomly initialized" in str(error), error
            else:
                raise SystemExit("the random-initialization guard did not fire")
            print("OK")
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", program],
            capture_output=True, text=True, cwd=str(REPO),
        )
        self.assertEqual(
            result.returncode, 0,
            f"core must import with priors.hoi/priors.hsi absent:\n{result.stderr}",
        )
        self.assertIn("OK", result.stdout)


if __name__ == "__main__":
    unittest.main()
