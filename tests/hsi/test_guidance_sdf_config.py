"""Pin mesh-SDF guidance config composition and sampler defaults."""

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import Sampler  # noqa: E402


class GuidanceSdfConfigTests(unittest.TestCase):
    def test_wrapper_resolves_the_frozen_guidance_overrides(self):
        with patch.dict(os.environ, {"ROOT_DIR": str(REPO)}, clear=False):
            with initialize_config_dir(config_dir=str(REPO / "code" / "config"), version_base=None):
                cfg = compose(config_name="config_sample_infbagel_lingo_hsi_p16gq")
                resolved = OmegaConf.to_container(cfg, resolve=True)
        self.assertEqual(resolved["sample_type"], "diffusion")
        self.assertTrue(resolved["use_guidance"])
        self.assertTrue(resolved["export_motion"])
        self.assertEqual(resolved["shard_count"], 8)
        self.assertEqual(resolved["hsi_guidance_sdf_proxy"], "area512")
        self.assertEqual(resolved["hsi_guidance_sdf_weight"], 4879)
        self.assertTrue(resolved["formal_preflight"])
        self.assertTrue(resolved["formal_attestation"])
        self.assertEqual(
            resolved["formal_attestation_protocol"],
            "p16-gq-preflight-attestation-v1",
        )
        self.assertEqual(
            resolved["expected_checkpoint_sha256"],
            "5daaf813ca82878868602840760f35df43b642d73f73cb37e24bb5a4dbf62b4c",
        )
        self.assertEqual(resolved["dataset"]["hsi_mesh_root"], resolved["lingo_mesh_root"])
        self.assertTrue(str(resolved["lingo_output_dir"]).startswith(str(REPO) + "/results/"))

    def test_plain_shard_index_composes_all_eight_values(self):
        with patch.dict(os.environ, {"ROOT_DIR": str(REPO)}, clear=False):
            with initialize_config_dir(config_dir=str(REPO / "code" / "config"), version_base=None):
                for shard_index in range(8):
                    cfg = compose(
                        config_name="config_sample_infbagel_lingo_hsi_p16gq",
                        overrides=["shard_index=%d" % shard_index],
                    )
                    self.assertEqual(int(cfg.shard_index), shard_index)
                    self.assertEqual(int(cfg.shard_count), 8)

    def test_sampler_mesh_knobs_are_off_by_default_and_opt_in(self):
        def make(**kwargs):
            return Sampler(
                device="cpu", mask_ind=0, emb_f=None, batch_size=1, channel=232,
                auto_regre_num=1, timesteps=500, ddim_timesteps=25, cm_timesteps=16,
                **kwargs
            )

        default = make()
        self.assertIsNone(default.hsi_guidance_sdf_proxy)
        self.assertEqual(default.hsi_guidance_sdf_weight, 0.0)
        default.dataset = MagicMock()
        self.assertIsNone(default._scene_sdf_geometry("unused-off-path"))
        default.dataset.scene_geometry.assert_not_called()
        arm = make(hsi_guidance_sdf_proxy="area512", hsi_guidance_sdf_weight=4879)
        self.assertEqual(arm.hsi_guidance_sdf_proxy, "area512")
        self.assertEqual(arm.hsi_guidance_sdf_weight, 4879.0)


if __name__ == "__main__":
    unittest.main()
