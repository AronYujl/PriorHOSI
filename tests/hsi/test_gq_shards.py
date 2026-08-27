"""Hermetic and adversarial tests for the independent P16-GQ wrapper."""

import copy
from contextlib import ExitStack
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from priors.hsi import gq_shards  # noqa: E402
from priors.hsi.scene_field import SceneGeometry, sdf_cache_protocol_identity  # noqa: E402


def _raw_payload(
    ordinal,
    *,
    output_dir,
    checkpoint_path,
    checkpoint_sha256,
    shard_block,
):
    selected = shard_block["selected_episode_ordinals"]
    selected_window_total = shard_block["selected_window_total"]
    scene_name = shard_block["selected_episodes"][0]["scene_name"]
    record_key = "%s:%06d" % (scene_name, ordinal)
    return {
        "schema_version": 4,
        "model_name": "hsi_b_lingo_full_v2_epoch222",
        "checkpoint": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        },
        "output_dir": str(output_dir),
        "seed": 42,
        "sample_type": "diffusion",
        "guided": True,
        "rds": {"available": False},
        "sampling_body": "smplx_vertices_10475",
        "fps": 30.0,
        "sequence_count": len(selected),
        "scene_count": 1,
        "scene_summary": {},
        "timing": {
            "per_window_wall_seconds": None,
            "total_sampling_seconds": None,
            "window_count": selected_window_total,
            "denoiser_calls_per_window": 1000,
            "sampler_steps_per_window": 500,
            "cuda_synchronized": True,
            "batch_size": 1,
            "aits": None,
            "avg_fps": None,
            "aggregate_fps": None,
            "rtf": None,
            "total_generation_seconds": None,
            "timed_sequence_count": 0,
            "avg_frames_per_seq": None,
            "avg_end_to_end_episode_seconds": None,
            "warmup_sequences_required": 0,
            "warmup_sequences_excluded": 0,
            "protocol_complete": False,
            "timing_valid": False,
            "timing_invalid_reason": "unit",
        },
        "sharding": {
            "shard_index": ordinal,
            "shard_count": shard_block["shard_count"],
            "canonical_episode_total": shard_block["canonical_episode_total"],
            "canonical_window_total": shard_block["canonical_window_total"],
            "shard_episode_ordinals": selected,
            "shard_window_total": selected_window_total,
            "partition_rule": shard_block["partition_rule"],
            "per_episode_seeding": shard_block["per_episode_seeding"],
            "timing_valid": False,
        },
        "latency_subset": {"enabled": False},
        "metrics": {
            record_key: {
                "canonical_ordinal": selected[0],
                "scene_name": scene_name,
                "source_sequence_index": ordinal,
                "window_count": float(selected_window_total),
                "pen_ratio": 0.0,
                "sampling_seconds": None,
                "per_window_wall_seconds": None,
                "excluded_as_warmup": False,
            }
        },
    }


class ReceiptFixture(unittest.TestCase):
    """Create eight tiny receipt-backed shards without touching real data."""

    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name) / "checkout"
        self.root.mkdir()
        self.cache = self.root / "cache"
        self.cache.mkdir()
        self.checkpoint = self.root / "checkpoint.pth"
        self.checkpoint.write_bytes(b"fixture checkpoint")
        self.checkpoint_sha256 = hashlib.sha256(self.checkpoint.read_bytes()).hexdigest()
        self.manifest_dir = self.root / "episodes"
        self.manifest_dir.mkdir()
        self.mesh_root = self.root / "mesh"
        self.output_root = self.root / "results"
        self.mesh_records = {}
        for ordinal in range(8):
            scene_name = "scene-%d" % ordinal
            manifest = self.manifest_dir / (scene_name + ".json")
            manifest.write_text(
                json.dumps(
                    [
                        {
                            "scene_name": scene_name,
                            "object_name": None,
                            "episode_num": 1,
                        }
                    ],
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            mesh = self.mesh_root / scene_name / "mesh_low.obj"
            mesh.parent.mkdir(parents=True)
            mesh.write_text("fixture mesh %d\n" % ordinal, encoding="utf-8")
            cache_file = self.cache / (scene_name + ".npz")
            cache_file.write_bytes(b"fixture cache %d" % ordinal)
            self.mesh_records[scene_name] = (mesh, cache_file)

    def tearDown(self):
        self.tempdir.cleanup()

    def contract(self):
        stack = ExitStack()
        stack.enter_context(
            patch.multiple(
                gq_shards,
                SEALED_CHECKPOINT_SHA256=self.checkpoint_sha256,
                GQ_EPISODES=8,
                GQ_WINDOWS=8,
            )
        )
        stack.enter_context(
            patch.dict(
                gq_shards.os.environ,
                {"INFBAGEL_SDF_CACHE": str(self.cache)},
                clear=False,
            )
        )
        return stack

    def resolved(self, shard_index, output_root=None):
        output_root = self.output_root if output_root is None else Path(output_root)
        return {
            "formal_wrapper": gq_shards.FORMAL_WRAPPER_RELATIVE_PATH,
            "formal_preflight": True,
            "formal_attestation": True,
            "formal_attestation_protocol": gq_shards.ATTESTATION_PROTOCOL,
            "shard_count": 8,
            "shard_index": shard_index,
            "expected_checkpoint_sha256": self.checkpoint_sha256,
            "lingo_hsi_mode": "sample",
            "sample_type": "diffusion",
            "use_guidance": True,
            "export_motion": True,
            "hsi_progress_fix": True,
            "hsi_guidance_sdf_proxy": "area512",
            "hsi_guidance_sdf_weight": 4879,
            "load_scene": True,
            "load_scene_goal": True,
            "load_pelvis_goal": True,
            "seed": 42,
            "dataset": {"hsi_mesh_root": str(self.mesh_root)},
            "lingo_mesh_root": str(self.mesh_root),
            "lingo_episode_dir": str(self.manifest_dir),
            "lingo_output_dir": str(output_root),
            "ckpt_path": str(self.checkpoint),
        }

    def preflight(self, scene_names):
        scenes = []
        for scene_name in sorted(scene_names):
            mesh, cache_file = self.mesh_records[scene_name]
            scenes.append(
                {
                    "path": str(cache_file),
                    "sha256": hashlib.sha256(cache_file.read_bytes()).hexdigest(),
                    "size_bytes": cache_file.stat().st_size,
                    "field_shape": [2, 2, 2],
                    "origin": [0.0, 0.0, 0.0],
                    "mesh_sha256": hashlib.sha256(mesh.read_bytes()).hexdigest(),
                    "metadata": {},
                    "protocol_id": sdf_cache_protocol_identity()["id"],
                    "scene_name": scene_name,
                    "mesh_path": str(mesh),
                }
            )
        return {
            "checkpoint": {
                "path": str(self.checkpoint),
                "sha256": self.checkpoint_sha256,
                "size_bytes": self.checkpoint.stat().st_size,
            },
            "proxy": {
                "asset_sha256": gq_shards.BODY_PROXY_ASSET_SHA256,
                "asset_size_bytes": gq_shards.BODY_PROXY_ASSET_SIZE_BYTES,
                "source_sha256": gq_shards.SMPLX_SOURCE_SHA256,
                "weights_shape": [512, 22],
                "offsets_shape": [512, 22, 3],
                "posedirs_shape": [512, 3, 189],
            },
            "sdf_cache": {
                "root": str(self.cache),
                "protocol": sdf_cache_protocol_identity(),
                "build_version": 1,
                "voxel_size": 0.02,
                "pad": 0.2,
                "band": 1,
                "scenes": scenes,
            },
        }

    def write_raw(self, path, ordinal, resolved, shard_block):
        output_dir = path.parent.parent
        path.parent.mkdir(parents=True)
        payload = _raw_payload(
            ordinal,
            output_dir=output_dir,
            checkpoint_path=self.checkpoint,
            checkpoint_sha256=self.checkpoint_sha256,
            shard_block=shard_block,
        )
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return payload

    def make_attested_shards(self, output_root=None):
        output_root = self.output_root if output_root is None else Path(output_root)
        paths = []
        manifest = gq_shards._load_episode_manifest(self.manifest_dir)
        for ordinal in range(8):
            resolved = self.resolved(ordinal, output_root=output_root)
            shard_block = gq_shards._episode_shard_block(manifest, ordinal)
            scene_names = sorted({item["scene_name"] for item in shard_block["selected_episodes"]})
            preflight = self.preflight(scene_names)
            command = gq_shards.build_evaluator_command(
                self.root, "/verified/python", ordinal
            )
            attestation, output = gq_shards._build_preflight_attestation(
                repo_root=self.root,
                config_name=gq_shards.FORMAL_CONFIG_NAME,
                resolved=resolved,
                manifest=manifest,
                shard_index=ordinal,
                scene_names=scene_names,
                preflight=preflight,
                command=command,
                sealed_evaluator_sha256=gq_shards.SEALED_EVALUATOR_SHA256,
            )
            gq_shards._write_json_atomic(gq_shards.preflight_attestation_path(output), attestation)
            self.write_raw(output, ordinal, resolved, shard_block)
            gq_shards._attach_execution_receipt(output)
            paths.append(output)
        return paths


class AttestationIdentityTests(ReceiptFixture):
    def test_preflight_scene_comparison_uses_unique_scene_set(self):
        with self.contract():
            resolved = self.resolved(0)
            preflight = self.preflight(["scene-0"])
            gq_shards._validate_preflight_observation(
                preflight,
                resolved,
                ["scene-0", "scene-0"],
            )

    def test_attestation_digest_contains_observed_contract_fields(self):
        with self.contract():
            paths = self.make_attested_shards()
            attestation_path = gq_shards.preflight_attestation_path(paths[0])
            attestation = gq_shards._load_json(attestation_path)
            self.assertTrue(attestation["wrapper_used"])
            self.assertTrue(attestation["preflight_passed"])
            self.assertEqual(attestation["treatment"]["sdf_weight"], 4879)
            self.assertEqual(
                attestation["treatment"]["config_flags"]["dataset_hsi_mesh_root"],
                str(self.mesh_root),
            )
            self.assertEqual(
                attestation["assets"]["area512_index"]["expected_sha256"],
                gq_shards.AREA512_INDEX_SHA256,
            )
            self.assertEqual(
                attestation["assets"]["area512_index"]["observed_sha256"],
                gq_shards.AREA512_INDEX_SHA256,
            )
            self.assertEqual(
                attestation["assets"]["proxy_tables"]["observed_sha256"],
                gq_shards.BODY_PROXY_ASSET_SHA256,
            )
            self.assertEqual(
                attestation["assets"]["source_smplx"]["observed_sha256"],
                gq_shards.SMPLX_SOURCE_SHA256,
            )
            self.assertEqual(
                attestation["treatment"]["checkpoint_expected_sha256"],
                self.checkpoint_sha256,
            )
            self.assertEqual(
                attestation["treatment"]["checkpoint_observed_sha256"],
                self.checkpoint_sha256,
            )
            self.assertEqual(attestation["episode_shard"]["shard_index"], 0)
            self.assertEqual(
                attestation["output_binding"]["canonical_output_path"], str(paths[0])
            )
            self.assertEqual(
                attestation["attestation_digest"],
                gq_shards._canonical_digest(gq_shards._attestation_without_digest(attestation)),
            )
            receipt = gq_shards._load_json(gq_shards.execution_receipt_path(paths[0]))
            self.assertEqual(receipt["attestation_digest"], attestation["attestation_digest"])
            self.assertEqual(receipt["canonical_output_path"], str(paths[0]))
            self.assertEqual(receipt["raw_payload_sha256"], gq_shards.sha256_file(paths[0]))

    def test_legacy_or_arbitrary_raw_payload_cannot_acquire_identity(self):
        raw = {"guided": True, "sample_type": "diffusion", "checkpoint": {"checkpoint_sha256": "0" * 64}}
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "arbitrary.json"
            original = json.dumps(raw, sort_keys=True)
            path.write_text(original, encoding="utf-8")
            for route in (
                gq_shards.canonical_treatment_identity,
                gq_shards.attach_treatment_identity,
                gq_shards.decorate_payload_file,
            ):
                with self.subTest(route=route.__name__), self.assertRaises(RuntimeError):
                    route(raw)
            with self.assertRaises(RuntimeError):
                gq_shards.decorate_payload_file(path)
            self.assertEqual(path.read_text(encoding="utf-8"), original)

    def test_direct_decorate_route_is_disabled_and_cli_has_no_decorate_command(self):
        with tempfile.TemporaryDirectory() as raw_root:
            path = Path(raw_root) / "payload.json"
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "disabled"):
                gq_shards.decorate_payload_file(path)
            with self.assertRaises(SystemExit):
                gq_shards._parser().parse_args(["decorate", str(path)])

    def test_mapping_merge_route_requires_a_receipt_backed_file(self):
        with self.assertRaisesRegex(ValueError, "execution receipts"):
            gq_shards.merge_gq_shard_payloads([{}])

    def test_missing_receipt_and_direct_sealed_output_are_rejected(self):
        with self.contract():
            paths = self.make_attested_shards()
            receipt_path = gq_shards.execution_receipt_path(paths[0])
            receipt_path.unlink()
            with self.assertRaisesRegex(ValueError, "missing P16-GQ execution receipt"):
                gq_shards.validate_attested_shard_file(paths[0])
            direct = paths[1]
            gq_shards.preflight_attestation_path(direct).unlink()
            gq_shards.execution_receipt_path(direct).unlink()
            with self.assertRaisesRegex(ValueError, "missing P16-GQ preflight attestation"):
                gq_shards.validate_attested_shard_file(direct)

    def test_receipt_tamper_is_rejected(self):
        with self.contract():
            paths = self.make_attested_shards()
            receipt_path = gq_shards.execution_receipt_path(paths[0])
            receipt = gq_shards._load_json(receipt_path)
            receipt["raw_payload_sha256"] = "0" * 64
            receipt_path.write_text(json.dumps(receipt, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "execution receipt binding mismatch"):
                gq_shards.validate_attested_shard_file(paths[0])

    def test_raw_payload_tamper_and_raw_claim_mutation_are_rejected(self):
        with self.contract():
            paths = self.make_attested_shards()
            payload = gq_shards.load_payload(paths[0])
            payload["guided"] = False
            paths[0].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw payload is not a guided"):
                gq_shards.validate_attested_shard_file(paths[0])

            paths = self.make_attested_shards(output_root=self.root / "results-second")
            payload = gq_shards.load_payload(paths[0])
            payload["metrics"][next(iter(payload["metrics"]))]["pen_ratio"] = 1.0
            paths[0].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "execution receipt binding mismatch"):
                gq_shards.validate_attested_shard_file(paths[0])

    def test_resolved_config_treatment_flag_and_hash_tamper_are_rejected(self):
        with self.contract():
            paths = self.make_attested_shards()
            attestation_path = gq_shards.preflight_attestation_path(paths[0])
            attestation = gq_shards._load_json(attestation_path)
            attestation["resolved_config"]["values"]["hsi_guidance_sdf_weight"] = 30572
            attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "attestation digest mismatch"):
                gq_shards.validate_attested_shard_file(paths[0])

            paths = self.make_attested_shards(output_root=self.root / "results-second")
            attestation_path = gq_shards.preflight_attestation_path(paths[0])
            attestation = gq_shards._load_json(attestation_path)
            attestation["treatment"]["proxy_tables_sha256"] = "0" * 64
            attestation_path.write_text(json.dumps(attestation, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "attestation digest mismatch"):
                gq_shards.validate_attested_shard_file(paths[0])

    def test_shard_replay_and_output_replay_are_rejected(self):
        with self.contract():
            paths = self.make_attested_shards()
            with self.assertRaisesRegex(ValueError, "replayed or missing shard"):
                gq_shards.merge_gq_shard_files([paths[0]] * 8, expected_episodes=8, expected_windows=8)

            replay = self.root / "replay" / "different-output" / "evaluation" / "per_sequence_metrics.json"
            replay.parent.mkdir(parents=True)
            original_attestation = gq_shards.preflight_attestation_path(paths[0])
            original_receipt = gq_shards.execution_receipt_path(paths[0])
            shutil.copyfile(paths[0], replay)
            shutil.copyfile(original_attestation, gq_shards.preflight_attestation_path(replay))
            shutil.copyfile(original_receipt, gq_shards.execution_receipt_path(replay))
            with self.assertRaisesRegex(ValueError, "output_dir|output binding"):
                gq_shards.validate_attested_shard_file(replay)

    def test_checkpoint_tamper_is_rejected(self):
        with self.contract():
            paths = self.make_attested_shards()
            self.checkpoint.write_bytes(b"tampered checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint changed"):
                gq_shards.validate_attested_shard_file(paths[0])

    def test_episode_30573_is_rejected_anywhere_in_a_raw_claim(self):
        with self.contract():
            paths = self.make_attested_shards()
            payload = gq_shards.load_payload(paths[0])
            payload["metrics"][next(iter(payload["metrics"]))]["episode_id"] = 30573
            paths[0].write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "forbidden 30573"):
                gq_shards.validate_attested_shard_file(paths[0])

    def test_canonical_end_to_end_mocked_wrapper_creates_attestation_before_child(self):
        observed = {}

        def fake_resolved(_repo_root, _config_name, shard_index):
            return self.resolved(shard_index)

        def fake_preflight(**kwargs):
            return self.preflight(kwargs["scene_names"])

        def fake_child(command, cwd, env, text, stdout, stderr, check):
            del cwd, env, text, stdout, stderr, check
            index = int(command[-1].split("=", 1)[1])
            resolved = self.resolved(index)
            output = gq_shards._canonical_shard_output_path(
                self.root, resolved, index, self.checkpoint_sha256
            )
            observed["attestation_before_child"] = gq_shards.preflight_attestation_path(output).is_file()
            observed["receipt_before_child"] = gq_shards.execution_receipt_path(output).exists()
            manifest = gq_shards._load_episode_manifest(self.manifest_dir)
            block = gq_shards._episode_shard_block(manifest, index)
            self.write_raw(output, index, resolved, block)
            observed["raw_before_receipt"] = output.read_bytes()
            return subprocess.CompletedProcess(command, 0, "Wrote %s\n" % output)

        with self.contract(), patch.object(
            gq_shards,
            "verify_sealed_evaluator_unchanged",
            return_value=gq_shards.SEALED_EVALUATOR_SHA256,
        ), patch.object(gq_shards, "_resolve_formal_config", side_effect=fake_resolved), patch.object(
            gq_shards, "run_formal_preflight", side_effect=fake_preflight
        ), patch.object(gq_shards.subprocess, "run", side_effect=fake_child):
            with patch.dict(
                gq_shards.os.environ,
                {"INFBAGEL_SDF_CACHE": str(self.cache)},
                clear=False,
            ):
                output = gq_shards.run_gq_shard(
                    self.root,
                    python="/verified/python",
                    shard_index=0,
                )
                verified = gq_shards.validate_attested_shard_file(output)
        self.assertTrue(observed["attestation_before_child"])
        self.assertFalse(observed["receipt_before_child"])
        self.assertEqual(output, Path(verified["payload_path"]))
        self.assertEqual(output.read_bytes(), observed["raw_before_receipt"])
        self.assertFalse("treatment_identity" in verified["payload"])
        self.assertTrue(gq_shards.execution_receipt_path(output).is_file())

    def test_successful_receipt_backed_merge_and_identity_is_not_inferred(self):
        with self.contract():
            paths = self.make_attested_shards()
            merged = gq_shards.merge_gq_shard_files(
                paths, expected_episodes=8, expected_windows=8
            )
            self.assertEqual(merged["sequence_count"], 8)
            self.assertEqual(merged["sharding"]["merged_shard_count"], 8)
            self.assertTrue(merged["treatment_attestation"]["wrapper_used"])
            self.assertEqual(
                len(merged["treatment_attestation"]["attestation_digests"]), 8
            )


class WrapperGovernanceTests(unittest.TestCase):
    def test_plain_shard_index_override_is_configured_for_all_eight_shards(self):
        for index in range(8):
            command = gq_shards.build_evaluator_command(REPO, "/verified/python", index)
            self.assertEqual(command[-1], "shard_index=%d" % index)
            self.assertEqual(
                [item for item in command if item.startswith("shard_index=")],
                ["shard_index=%d" % index],
            )
            self.assertEqual(len(command), 5)

    def test_caller_config_overrides_and_altered_config_name_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "does not accept caller config overrides"):
            gq_shards.build_evaluator_command(
                REPO, "/verified/python", 0, extra_overrides=("sdf_weight=30573",)
            )
        with self.assertRaisesRegex(ValueError, "committed formal config"):
            gq_shards.build_evaluator_command(
                REPO, "/verified/python", 0, config_name="other"
            )

    def test_formal_config_requires_the_independent_attesting_wrapper(self):
        config = (
            REPO / "code" / "config" / "config_sample_infbagel_lingo_hsi_p16gq.yaml"
        ).read_text(encoding="utf-8")
        self.assertIn("formal_wrapper: code/priors/hsi/gq_shards.py", config)
        self.assertIn("formal_attestation: true", config)
        self.assertEqual(
            gq_shards.P16_GQ_CONFIG_FLAGS["formal_wrapper"],
            "code/priors/hsi/gq_shards.py",
        )

    def test_sealed_evaluator_is_byte_identical_to_parent(self):
        digest = gq_shards.verify_sealed_evaluator_unchanged(REPO)
        self.assertEqual(digest, gq_shards.SEALED_EVALUATOR_SHA256)
        source = (REPO / gq_shards.SEALED_EVALUATOR_RELATIVE_PATH).read_text(
            encoding="utf-8"
        )
        self.assertNotIn("run_formal_preflight", source)

    def test_final_delta_mechanically_excludes_sealed_and_core(self):
        result = gq_shards.verify_complete_wrapper_delta(REPO)
        self.assertNotIn(str(gq_shards.SEALED_EVALUATOR_RELATIVE_PATH), result["changed_paths"])
        self.assertTrue("code/priors/hsi/gq_shards.py" in result["changed_paths"])


class CacheResidencyContractTests(unittest.TestCase):
    def test_one_geometry_reuses_the_same_device_dtype_cuda_view(self):
        geometry = SceneGeometry(
            "cache-view",
            np.zeros((2, 2, 2), dtype="<f4"),
            np.zeros(3, dtype="<f8"),
            0.02,
            is_watertight=True,
        )
        first = geometry._view(torch.device("cpu"), torch.float32)
        second = geometry._view(torch.device("cpu"), torch.float32)
        self.assertIs(first, second)
        for key in ("field", "min", "max", "last"):
            self.assertIs(first[key], second[key])

    def test_sealed_metrics_and_sampler_use_the_same_scene_key_without_eviction(self):
        evaluator = (REPO / "code" / "test_infbagel_lingo_hsi.py").read_text(
            encoding="utf-8"
        )
        dataset = (REPO / "code" / "datasets" / "infbagel_mix.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("geometries: Dict[str, SceneGeometry] = {}", evaluator)
        self.assertIn("geometries[scene_name] = SceneGeometry.from_scene(", evaluator)
        self.assertIn(
            "compute_metric_record(\n            vertices, joints, geometries[scene_name]",
            evaluator,
        )
        self.assertNotIn("geometries.pop", evaluator)
        self.assertNotIn("geometries.clear", evaluator)
        self.assertNotIn("SceneGeometry.cache_clear", evaluator)
        self.assertIn("SceneGeometry.from_scene(", dataset)
        self.assertIn("cache_dir=default_cache_dir(),", dataset)
        self.assertIn("dataset_root=Path(self.lingo_dataset.folder),", dataset)


class GovernanceContractTests(unittest.TestCase):
    def test_registry_and_plan_keep_the_binding_p16_gq_directions(self):
        records = [
            json.loads(line)
            for line in (REPO / "experiments" / "registry.jsonl").read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        ]
        record = next(
            item for item in records
            if item.get("experiment_id") == "p1-hsi-b-p16gq-s42-20260827"
        )
        fragment_keys = record["config"]["fragment_keys"]
        self.assertIn("ckpt_path", fragment_keys)
        self.assertIn("formal_wrapper", fragment_keys)
        self.assertIn("formal_attestation", fragment_keys)
        self.assertNotIn("shard_index", fragment_keys)
        self.assertEqual(
            record["config"]["shard_wrapper"]["identity_fields"],
            [
                "attestation_schema_version",
                "attestation_digest",
                "wrapper_used",
                "preflight_passed",
                "resolved_config.sha256",
                "treatment.guidance_mode",
                "treatment.guidance_version",
                "treatment.sdf_weight",
                "treatment.sdf_margin_m",
                "treatment.floor_threshold_m",
                "treatment.area512_index_sha256",
                "treatment.proxy_tables_sha256",
                "treatment.source_smplx_sha256",
                "treatment.checkpoint_expected_sha256",
                "treatment.checkpoint_observed_sha256",
                "treatment.scene_mesh_sdf_cache_protocol",
                "treatment.config_flags",
                "episode_shard.manifest.sha256",
                "episode_shard.shard_index",
                "episode_shard.selected_episode_ordinals",
                "output_binding.canonical_output_path",
                "receipt.raw_payload_sha256",
                "receipt.receipt_digest",
            ],
        )
        self.assertIn("steady-state and late-shard", record["config"]["shard_wrapper"]["memory_headroom"])
        guards = record["config"]["guards"]
        self.assertIn("nonincrease", guards["rav_jitter_1s"])
        self.assertIn("nondecrease", guards["still_frac_1s"])
        self.assertIn("retained", guards["rav_mean_1s"])
        self.assertIn("uncorrected", guards["theta_head_exp"])
        self.assertIn("not report-only, blind, independent, or confirmatory", guards["affordance107"])
        self.assertIn("must not significantly increase", guards["contact_count"])
        self.assertIn("must not significantly decrease", guards["contact_count_exterior"])

        plan = (REPO / "docs" / "plan" / "PHASE_1C_HSI.md").read_text(
            encoding="utf-8"
        )
        plan = " ".join(plan.split())
        for phrase in (
            "preflight attestation",
            "execution receipt",
            "raw shard payload remains byte-for-byte unchanged",
            "Merge is never allowed to infer identity from a raw shard",
            "steady-state and late-shard memory headroom",
            "12 captions / 290 of 375 episodes were viewed",
            "full375 facing are required parallel reports",
            "not report-only, blind, independent, or confirmatory analysis",
        ):
            self.assertIn(phrase, plan)


if __name__ == "__main__":
    unittest.main()
