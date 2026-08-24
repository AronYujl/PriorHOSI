"""The frozen D2-AI training recipe and the arms composed on top of it.

Before 2026-08-11 every full-budget arm restated the whole recipe inline: ~52
keys per file, of which 22 were byte-identical restatements of the base config.
"Same recipe as the sealed baseline" was therefore a hand-maintained guarantee,
and one missed line would silently produce an uncontrolled comparison.

These tests make that guarantee mechanical. They pin the sealed value of every
recipe key and assert that each lineage arm still resolves to it, so an edit to
`recipe/d2ai.yaml` cannot quietly redefine what P8/P9/P10 were measured against.
"""

import os
import unittest
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "code" / "config"
# config_train_hoi_prior.yaml declares repo_root: ${oc.env:ROOT_DIR}.
os.environ.setdefault("ROOT_DIR", str(REPO))

# The sealed D2-AI recipe, as extracted from the eleven arms' resolved configs.
SEALED_RECIPE = {
    "mode": "d2ai-full-budget",
    "d2ai_full_budget": True,
    "max_processed_windows": 299520000,
    "num_gpus": 4,
    "batch_size": 512,
    "effective_batch_size": 2048,
    "num_workers": 4,
    "dataset_limit": 0,
    "optimizer_name": "Adam",
    "scheduler_name": "none",
    "gradient_clipping": False,
    "gradient_clip_norm": None,
    "minimum_lr_ratio": 1.0,
    "weight_decay": 0.0,
    "amp": False,
    "max_consecutive_amp_overflows": 0,
    "primary_weight_variant": "online",
    "ema_decays": [],
    "validation_windows": 32768,
    "validation_batch_size": 512,
    "validation_interval_windows": 3072000,
    "checkpoint_interval_windows": 3072000,
    "fk_foot_temporal_routing": True,
    "fk_weight": 0.3569973401779424,
    "object_surface_weight": 0.4772322188400037,
}

# Arm -> the factor it manipulates, with the sealed value it was run at.
LINEAGE_ARMS = {
    "p8h1": {"hand_object_contact_weight": 50.0},
    "p8l1": {"hand_object_contact_weight": 50.0, "max_processed_windows": 61440000},
    "p9w1": {"hand_object_contact_weight": 1.0},
    "p9w3": {"hand_object_contact_weight": 3.0},
    "p9w5": {"hand_object_contact_weight": 5.0},
    "p9w8": {"hand_object_contact_weight": 8.0},
    "p9w10": {"hand_object_contact_weight": 10.0},
    "p9w15": {"hand_object_contact_weight": 15.0},
    "p10_hinge": {"hand_object_contact_weight": 3.0, "hand_object_contact_hinge": 0.02},
    "p10_detach": {
        "hand_object_contact_weight": 3.0,
        "hand_object_contact_detach_object": True,
    },
    "p10_both": {
        "hand_object_contact_weight": 3.0,
        "hand_object_contact_hinge": 0.02,
        "hand_object_contact_detach_object": True,
    },
    "p11_rootdetach": {
        "hand_object_contact_weight": 3.0,
        "hand_object_contact_detach_root": True,
    },
}


def _resolve(config_name, overrides=()):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    return OmegaConf.to_container(cfg, resolve=True)


class RecipeFileTests(unittest.TestCase):
    def test_base_config_declares_the_sealed_contact_mask_mode(self):
        resolved = _resolve("config_train_hoi_prior")
        self.assertEqual(resolved["hand_object_contact_mask_mode"], "sealed")

    def test_the_recipe_is_a_global_package(self):
        """Without `@package _global_` the keys land under a `recipe:` node."""
        text = (CONFIG_DIR / "recipe" / "d2ai.yaml").read_text(encoding="utf-8")
        self.assertTrue(text.startswith("# @package _global_"))

    def test_the_recipe_states_exactly_the_sealed_keys(self):
        text = (CONFIG_DIR / "recipe" / "d2ai.yaml").read_text(encoding="utf-8")
        declared = {
            line.split(":", 1)[0].strip()
            for line in text.splitlines()
            if line and not line.startswith((" ", "#")) and ":" in line
        }
        self.assertEqual(declared, set(SEALED_RECIPE))


class LineageArmTests(unittest.TestCase):
    def test_every_arm_inherits_the_sealed_recipe(self):
        for arm in LINEAGE_ARMS:
            with self.subTest(arm=arm):
                resolved = _resolve(f"config_train_hoi_prior_{arm}")
                overridden = LINEAGE_ARMS[arm]
                for key, expected in SEALED_RECIPE.items():
                    want = overridden.get(key, expected)
                    self.assertEqual(
                        resolved[key], want,
                        f"{arm} resolved {key}={resolved[key]!r}, expected {want!r}",
                    )

    def test_every_arm_applies_its_own_manipulated_factor(self):
        for arm, factors in LINEAGE_ARMS.items():
            with self.subTest(arm=arm):
                resolved = _resolve(f"config_train_hoi_prior_{arm}")
                for key, expected in factors.items():
                    self.assertEqual(resolved[key], expected)

    def test_the_geometry_term_is_off_unless_an_arm_turns_it_on(self):
        """The 2x2 must be readable from the config, not inferred from a name."""
        both = _resolve("config_train_hoi_prior_p10_both")
        hinge = _resolve("config_train_hoi_prior_p10_hinge")
        detach = _resolve("config_train_hoi_prior_p10_detach")
        w3 = _resolve("config_train_hoi_prior_p9w3")
        self.assertEqual(
            (w3["hand_object_contact_hinge"], w3["hand_object_contact_detach_object"]),
            (0.0, False),
        )
        self.assertEqual(
            (hinge["hand_object_contact_hinge"], hinge["hand_object_contact_detach_object"]),
            (0.02, False),
        )
        self.assertEqual(
            (detach["hand_object_contact_hinge"], detach["hand_object_contact_detach_object"]),
            (0.0, True),
        )
        self.assertEqual(
            (both["hand_object_contact_hinge"], both["hand_object_contact_detach_object"]),
            (0.02, True),
        )

    def test_arms_state_only_their_difference(self):
        """The point of the refactor: an arm file is short enough to read."""
        for arm in LINEAGE_ARMS:
            with self.subTest(arm=arm):
                path = CONFIG_DIR / f"config_train_hoi_prior_{arm}.yaml"
                lines = [
                    line for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and not line.strip().startswith("#")
                ]
                self.assertLessEqual(len(lines), 16, f"{arm} restates the recipe again")
                self.assertIn("  - recipe: d2ai", lines)

    def test_p11_is_a_single_factor_contrast_against_sealed_w3(self):
        # Normalize the three identity fields before resolving so their derived
        # output/checkpoint/metrics/state paths also share one comparison value.
        identity = ("run_id=contrast", "exp_name=contrast", "subphase=contrast")
        p11 = _resolve("config_train_hoi_prior_p11_rootdetach", identity)
        w3 = _resolve("config_train_hoi_prior_p9w3", identity)
        differing = {key for key in set(p11) | set(w3) if p11.get(key) != w3.get(key)}
        self.assertEqual(differing, {"hand_object_contact_detach_root"})


if __name__ == "__main__":
    unittest.main()
