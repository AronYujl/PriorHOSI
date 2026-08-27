"""CPU tests for the formal P16-GQ dependency and provenance gate."""

import hashlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi.preflight import (  # noqa: E402
    SEALED_CHECKPOINT_SHA256,
    SDF_CACHE_PROTOCOL_ID,
    run_formal_preflight,
    sdf_cache_path,
    sealed_checkpoint_sha256,
)
from priors.hsi.scene_field import sdf_cache_protocol_identity  # noqa: E402
from priors.hsi import gq_shards  # noqa: E402
import priors.hsi.preflight as preflight_module  # noqa: E402


class FormalPreflightTests(unittest.TestCase):
    @staticmethod
    def _fixture(root):
        repo = root / "checkout"
        dataset = repo / "data" / "dataset"
        mesh_root = repo / "data" / "Scene_mesh"
        cache = repo / ".cache" / "hsi_sdf"
        scene_name = "fixture"
        (dataset / "Scene").mkdir(parents=True)
        mesh_path = mesh_root / scene_name / "mesh_low.obj"
        mesh_path.parent.mkdir(parents=True)
        mesh_path.write_text(
            "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n", encoding="utf-8"
        )
        occupancy = np.zeros((300, 100, 400), dtype=np.bool_)
        occupancy[:, 0, :] = True
        np.save(dataset / "Scene" / (scene_name + ".npy"), occupancy)
        cache.mkdir(parents=True)
        mesh_digest = hashlib.sha256(mesh_path.read_bytes()).hexdigest()
        cache_path = sdf_cache_path(cache, scene_name, mesh_digest)
        np.savez(
            cache_path,
            field=np.zeros((2, 2, 2), dtype="<f4"),
            origin=np.zeros(3, dtype=np.float64),
            meta=np.array(json.dumps({
                "cache_protocol_id": SDF_CACHE_PROTOCOL_ID,
                "mesh_filename": "mesh_low.obj",
                "mesh_sha256": mesh_digest,
                "mesh_sha256_prefix": mesh_digest[:16],
                "scene_name": scene_name,
                "voxel_size_m": 0.02,
                "padding_m": 0.20,
                "exact_band_voxels": 1,
                "build_version": 1,
                "field_dtype": "<f4",
                "origin_dtype": "<f8",
                "field_shape": [2, 2, 2],
                "origin_shape": [3],
                "filename_binding": sdf_cache_protocol_identity()["filename_binding"],
                "metadata_binding": sdf_cache_protocol_identity()["metadata_binding"],
                "watertight": 0.0,
            })),
        )
        checkpoint = repo / "results" / "sealed.pth"
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b"temporary checkpoint fixture")
        expected = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        return repo, dataset, mesh_root, cache, checkpoint, expected, scene_name

    def test_preflight_exercises_real_proxy_and_scene_geometry_from_complete_cache(self):
        with tempfile.TemporaryDirectory() as raw_root:
            repo, dataset, mesh_root, cache, checkpoint, expected, scene_name = self._fixture(
                Path(raw_root)
            )
            with patch.dict(os.environ, {"INFBAGEL_SDF_CACHE": str(cache)}, clear=False), patch.object(
                preflight_module, "SEALED_CHECKPOINT_SHA256", expected
            ):
                result = run_formal_preflight(
                    repo_root=repo,
                    checkpoint_path=checkpoint,
                    dataset_root=dataset,
                    mesh_root=mesh_root,
                    scene_names=[scene_name],
                    expected_checkpoint_sha256=expected,
                )
            self.assertEqual(result["checkpoint"]["sha256"], expected)
            self.assertEqual(result["proxy"]["weights_shape"], [512, 22])
            self.assertEqual(result["proxy"]["offsets_shape"], [512, 22, 3])
            self.assertEqual(result["proxy"]["posedirs_shape"], [512, 3, 189])
            self.assertEqual(result["sdf_cache"]["scenes"][0]["scene_name"], scene_name)
            self.assertEqual(result["sdf_cache"]["protocol"], sdf_cache_protocol_identity())
            self.assertEqual(
                result["sdf_cache"]["scenes"][0]["protocol_id"],
                SDF_CACHE_PROTOCOL_ID,
            )

    def test_missing_cache_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as raw_root:
            repo, dataset, mesh_root, cache, checkpoint, expected, scene_name = self._fixture(
                Path(raw_root)
            )
            (sdf_cache_path(cache, scene_name, hashlib.sha256(
                (mesh_root / scene_name / "mesh_low.obj").read_bytes()
            ).hexdigest())).unlink()
            with patch.dict(os.environ, {"INFBAGEL_SDF_CACHE": str(cache)}, clear=False), patch.object(
                preflight_module, "SEALED_CHECKPOINT_SHA256", expected
            ):
                with self.assertRaises(FileNotFoundError):
                    run_formal_preflight(
                        repo_root=repo,
                        checkpoint_path=checkpoint,
                        dataset_root=dataset,
                        mesh_root=mesh_root,
                        scene_names=[scene_name],
                        expected_checkpoint_sha256=expected,
                    )

    def test_cache_mesh_hash_mismatch_is_a_hard_failure(self):
        with tempfile.TemporaryDirectory() as raw_root:
            repo, dataset, mesh_root, cache, checkpoint, expected, scene_name = self._fixture(
                Path(raw_root)
            )
            mesh_digest = hashlib.sha256(
                (mesh_root / scene_name / "mesh_low.obj").read_bytes()
            ).hexdigest()
            cache_path = sdf_cache_path(cache, scene_name, mesh_digest)
            with np.load(cache_path, allow_pickle=False) as data:
                field = np.asarray(data["field"])
                origin = np.asarray(data["origin"])
            np.savez(
                cache_path,
                field=field,
                origin=origin,
                meta=np.array(
                    json.dumps(
                        {
                            "cache_protocol_id": SDF_CACHE_PROTOCOL_ID,
                            "mesh_sha256": mesh_digest,
                            "mesh_sha256_prefix": "0" * 16,
                            "scene_name": scene_name,
                            "voxel_size_m": 0.02,
                            "padding_m": 0.20,
                            "exact_band_voxels": 1,
                            "build_version": 1,
                            "field_dtype": "<f4",
                            "origin_dtype": "<f8",
                            "field_shape": [2, 2, 2],
                            "origin_shape": [3],
                            "filename_binding": sdf_cache_protocol_identity()["filename_binding"],
                            "metadata_binding": sdf_cache_protocol_identity()["metadata_binding"],
                        }
                    )
                ),
            )
            with patch.dict(os.environ, {"INFBAGEL_SDF_CACHE": str(cache)}, clear=False), patch.object(
                preflight_module, "SEALED_CHECKPOINT_SHA256", expected
            ):
                with self.assertRaises(RuntimeError):
                    run_formal_preflight(
                        repo_root=repo,
                        checkpoint_path=checkpoint,
                        dataset_root=dataset,
                        mesh_root=mesh_root,
                        scene_names=[scene_name],
                        expected_checkpoint_sha256=expected,
                    )

    def test_sealed_checkpoint_mismatch_is_rejected_before_sampling(self):
        with tempfile.TemporaryDirectory() as raw_root:
            checkpoint = Path(raw_root) / "checkpoint.pth"
            checkpoint.write_bytes(b"not the sealed checkpoint")
            with self.assertRaises(RuntimeError):
                sealed_checkpoint_sha256(checkpoint, SEALED_CHECKPOINT_SHA256)
            with self.assertRaises(RuntimeError):
                sealed_checkpoint_sha256(checkpoint, "0" * 64)

    def test_formal_paths_reject_parent_checkout_roots(self):
        with self.assertRaises(RuntimeError):
            run_formal_preflight(
                repo_root=REPO,
                checkpoint_path=REPO.parent / "InfBaGel-hsi" / "checkpoint.pth",
                dataset_root=REPO / "data" / "dataset",
                mesh_root=REPO / "data" / "Scene_mesh",
                scene_names=["fixture"],
            )

    def test_sealed_evaluator_has_no_p16_gate_delta(self):
        source = (REPO / "code" / "test_infbagel_lingo_hsi.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_formal_preflight", source)
        self.assertNotIn('"formal_preflight"', source)
        self.assertEqual(
            gq_shards.verify_sealed_evaluator_unchanged(REPO),
            "4f25a6e67ab5104f2b10b41acbafa7ef257814751e0c402f0e28581b7b9eac0f",
        )

    def test_proxy_runtime_has_no_full_source_loader_or_parent_checkout_path(self):
        source = (REPO / "code" / "priors" / "hsi" / "body_proxy.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("_repository_root", source)
        self.assertNotIn("model_path", source)
        self.assertNotIn("smpl_models", source)


if __name__ == "__main__":
    unittest.main()
