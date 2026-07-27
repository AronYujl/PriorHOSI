"""Authority CPU contracts for the fixed D2-AD0 coordinate repair."""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import torch
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from priors.d2ad import (  # noqa: E402
    BPS_YUP_TENSOR_SHA256,
    DEFAULT_QUERY_WORKERS,
    OBJECT_MAPPING_SHA256,
    OBJECT_NAMES,
    REST_MESH_MANIFEST_SHA256,
    REST_MESH_SHA256,
    D2ADBatchCollator,
    D2ADPriorWindowDataset,
    LocalObjectBPSBuilder,
    local_bps_basis,
)
from priors.diffusion import HOIPriorSampler  # noqa: E402
from priors.interaction_adapter import (  # noqa: E402
    ADAPTER_PARAMETER_COUNT,
    ASSIGNMENT_SHA256,
    BPS_SHA256,
    CENTER_INDICES,
    CLUSTER_SIZES,
    LOCAL_BASIS_COORDINATE_SYSTEM,
    LocalObjectInteractionAdapter,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AC,
    HOI_ARCHITECTURE_D2AD,
    HOIPrior,
    assert_parameter_independence,
    build_expert,
    load_trained_hoi_prior,
)
from train_hoi_prior import (  # noqa: E402
    _d2ac_gradient_audit,
    _locked_loss_weights,
    _optimization_contract,
    _validate_d2ad_contract,
)
from tools import run_hoi_d2ad_internal as d2ad_internal  # noqa: E402
from tools import run_hoi_d2ad_native_evaluation as d2ad_native  # noqa: E402
from tools.diagnose_hoi_d2ad import RUN_ID as CPU_RUN_ID  # noqa: E402
from tools.run_hoi_d2ad_internal import (  # noqa: E402
    RUN_ID_RE as INTERNAL_RUN_ID_RE,
    TRAINING_RUN_ID as INTERNAL_TRAINING_RUN_ID,
)
from tools.run_hoi_d2ad_native_evaluation import (  # noqa: E402
    RUN_ID_RE as NATIVE_RUN_ID_RE,
    TRAINING_RUN_ID as NATIVE_TRAINING_RUN_ID,
)
from tools.smoke_hoi_d2ad import (  # noqa: E402
    FORMAL_RUN_ID,
    RUN_ID as SMOKE_RUN_ID,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _yaw(angle_degrees: float) -> torch.Tensor:
    angle = math.radians(angle_degrees)
    cosine, sine = math.cos(angle), math.sin(angle)
    return torch.tensor(
        [
            [cosine, 0.0, sine],
            [0.0, 1.0, 0.0],
            [-sine, 0.0, cosine],
        ],
        dtype=torch.float32,
    )


def _inputs(batch: int = 2, *, dtype: torch.dtype = torch.float32):
    generator = torch.Generator().manual_seed(123)
    return {
        "noisy": torch.randn(batch, 16, 232, generator=generator, dtype=dtype),
        "timesteps": torch.arange(batch, dtype=torch.long) * 249 % 500,
        "text": torch.randn(batch, 768, generator=generator, dtype=dtype),
        "global_bps": torch.randn(
            batch, 1024, 3, generator=generator, dtype=dtype,
        ),
        "goals": torch.randn(batch, 9, generator=generator, dtype=dtype),
        "progress": torch.randn(batch, 3, generator=generator, dtype=dtype),
        "local_bps": torch.randn(
            batch, 1024, 3, generator=generator, dtype=dtype,
        ),
    }


def _forward(model: HOIPrior, values):
    return model(
        values["noisy"],
        values["timesteps"],
        values["text"],
        values["global_bps"],
        values["goals"],
        values["progress"],
        local_object_bps=values["local_bps"],
    )


class D2ADGeometryTests(unittest.TestCase):
    def test_immutable_assets_and_coordinate_hashes(self):
        builder = LocalObjectBPSBuilder(
            ROOT, query_workers=DEFAULT_QUERY_WORKERS,
        )
        metadata = builder.contract_metadata()
        self.assertEqual(_sha256(ROOT / "code/bps.pt"), BPS_SHA256)
        self.assertEqual(metadata["basis_yup_tensor_sha256"], BPS_YUP_TENSOR_SHA256)
        self.assertEqual(metadata["rest_mesh_manifest_sha256"], REST_MESH_MANIFEST_SHA256)
        self.assertEqual(metadata["object_mapping_sha256"], OBJECT_MAPPING_SHA256)
        self.assertEqual(tuple(metadata["object_mapping"]), OBJECT_NAMES)
        self.assertEqual(metadata["query_parameters"], {"k": 1, "eps": 0.0, "p": 2})
        self.assertTrue(metadata["full_rest_mesh"])
        self.assertFalse(metadata["mesh_subsample"])
        self.assertFalse(metadata["window_condition_cache"])
        self.assertEqual(
            {
                path.name: _sha256(path)
                for path in sorted(
                    (ROOT / "data/object/rest_object_geo").glob("*.ply")
                )
            },
            dict(REST_MESH_SHA256),
        )
        basis = local_bps_basis(ROOT / "code/bps.pt")
        self.assertEqual(tuple(basis.shape), (1024, 3))
        self.assertEqual(basis.dtype, torch.float32)
        self.assertTrue(torch.isfinite(basis).all())
        self.assertEqual(
            hashlib.sha256(basis.contiguous().numpy().tobytes()).hexdigest(),
            BPS_YUP_TENSOR_SHA256,
        )
        adapter = LocalObjectInteractionAdapter(
            basis_coordinate_system=LOCAL_BASIS_COORDINATE_SYSTEM,
        )
        self.assertEqual(
            adapter.partition_metadata["center_indices"],
            list(CENTER_INDICES),
        )
        self.assertEqual(
            adapter.partition_metadata["cluster_sizes"],
            list(CLUSTER_SIZES),
        )
        self.assertEqual(
            adapter.partition_metadata["assignment_sha256"],
            ASSIGNMENT_SHA256,
        )
        torch.testing.assert_close(adapter.bps_basis, basis, rtol=0.0, atol=0.0)

    def test_common_global_yaw_equivariance_and_worker_exactness(self):
        world_to_local = torch.stack((_yaw(-31), _yaw(12), _yaw(79)))
        object_rotation = torch.stack((_yaw(17), _yaw(-48), _yaw(135)))
        object_indices = torch.tensor((0, 3, 12), dtype=torch.long)
        builder_one = LocalObjectBPSBuilder(ROOT, query_workers=1)
        original, indices_one = builder_one.build(
            world_to_local,
            object_rotation,
            object_indices,
            return_indices=True,
        )
        common = _yaw(53).expand(3, -1, -1)
        rotated, indices_rotated = builder_one.build(
            world_to_local @ common.transpose(-1, -2),
            common @ object_rotation,
            object_indices,
            return_indices=True,
        )
        self.assertTrue(torch.equal(indices_one, indices_rotated))
        self.assertLessEqual(float((original - rotated).abs().max()), 1.0e-6)

        builder_three = LocalObjectBPSBuilder(ROOT, query_workers=3)
        threaded, indices_three = builder_three.build(
            world_to_local,
            object_rotation,
            object_indices,
            return_indices=True,
        )
        self.assertTrue(torch.equal(indices_one, indices_three))
        self.assertTrue(torch.equal(original, threaded))
        builder_all = LocalObjectBPSBuilder(ROOT, query_workers=-1)
        all_workers, indices_all = builder_all.build(
            world_to_local,
            object_rotation,
            object_indices,
            return_indices=True,
        )
        repeated, repeated_indices = builder_three.build(
            world_to_local,
            object_rotation,
            object_indices,
            return_indices=True,
        )
        order = torch.tensor((2, 0, 1), dtype=torch.long)
        reordered, reordered_indices = builder_three.build(
            world_to_local[order],
            object_rotation[order],
            object_indices[order],
            return_indices=True,
        )
        self.assertTrue(torch.equal(indices_one, indices_all))
        self.assertTrue(torch.equal(original, all_workers))
        self.assertTrue(torch.equal(indices_three, repeated_indices))
        self.assertTrue(torch.equal(threaded, repeated))
        self.assertTrue(torch.equal(reordered_indices, indices_one[order]))
        self.assertTrue(torch.equal(reordered, original[order]))

        adapter = LocalObjectInteractionAdapter(
            basis_coordinate_system=LOCAL_BASIS_COORDINATE_SYSTEM,
        )
        original_features = adapter.local_features(original)
        rotated_features = adapter.local_features(rotated)
        self.assertEqual(tuple(original_features.shape), (3, 16, 10))
        self.assertTrue(torch.isfinite(original_features).all())
        self.assertLessEqual(
            float((original_features - rotated_features).abs().max()),
            1.0e-6,
        )

    def test_dataset_collator_and_evaluator_builder_are_identical(self):
        dataset = D2ADPriorWindowDataset(
            str(ROOT),
            "hoi",
            partition="internal_validation",
            limit=1,
            split_manifest=str(
                ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
        )
        item = dataset[0]
        collated = D2ADBatchCollator(ROOT, query_workers=1)([item])
        sequence = str(dataset.scene_names[int(item["sequence_index"])])
        transform = torch.eye(4, dtype=torch.float32)[None]
        transform[:, :3, :3] = item["world_to_local_rotation"].transpose(-1, -2)
        evaluator = LocalObjectBPSBuilder(
            ROOT, query_workers=1,
        ).build_from_evaluator_inputs(
            transform,
            item["object_rotation_reference"][None],
            [sequence],
        )
        self.assertTrue(torch.equal(collated["local_object_bps"], evaluator))
        self.assertEqual(collated["local_object_bps"].dtype, torch.float32)
        self.assertTrue(torch.isfinite(collated["local_object_bps"]).all())
        self.assertGreater(float(collated["local_bps_build_seconds"]), 0.0)

        changed = LocalObjectBPSBuilder(ROOT, query_workers=1).build(
            item["world_to_local_rotation"][None],
            (_yaw(37) @ item["object_rotation_reference"])[None],
            item["object_geometry_index"][None],
        )
        self.assertGreater(
            float((changed - collated["local_object_bps"]).abs().mean()),
            1.0e-4,
        )

    def test_invalid_geometry_inputs_fail_closed(self):
        builder = LocalObjectBPSBuilder(ROOT, query_workers=1)
        with self.assertRaisesRegex(ValueError, "query_workers"):
            LocalObjectBPSBuilder(ROOT, query_workers=0)
        with self.assertRaisesRegex(ValueError, "finite"):
            builder.build(
                torch.full((1, 3, 3), float("nan")),
                torch.eye(3)[None],
                [0],
            )
        with self.assertRaisesRegex(ValueError, "object name"):
            builder.object_indices_from_names(["sub0_not-an-object_000"])
        with self.assertRaisesRegex(ValueError, "transforms"):
            builder.build_from_evaluator_inputs(
                torch.eye(3)[None], torch.eye(3)[None], ["clothesstand"],
            )


class D2ADModelTests(unittest.TestCase):
    def test_exact_parameter_count_api_and_required_local_condition(self):
        model = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AD,
        )
        self.assertIsInstance(model, HOIPrior)
        self.assertEqual(
            sum(parameter.numel() for parameter in model.parameters()),
            30_023_145,
        )
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in model.network.interaction_adapter.parameters()
            ),
            ADAPTER_PARAMETER_COUNT,
        )
        self.assertEqual(float(model.network.interaction_adapter.alpha), 0.0)
        signature = inspect.signature(HOIPrior.forward)
        self.assertEqual(
            list(signature.parameters)[:7],
            [
                "self",
                "noisy",
                "timesteps",
                "text_embedding",
                "object_bps",
                "goals",
                "progress",
            ],
        )
        local = signature.parameters["local_object_bps"]
        self.assertEqual(local.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIsNone(local.default)
        values = _inputs(batch=2)
        output = _forward(model, values)
        self.assertEqual(tuple(output.shape), (2, 16, 232))
        with self.assertRaisesRegex(ValueError, "requires current human-local"):
            model(
                values["noisy"],
                values["timesteps"],
                values["text"],
                values["global_bps"],
                values["goals"],
                values["progress"],
            )
        d2ac = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AC,
        )
        with self.assertRaisesRegex(ValueError, "restricted to D2-AD"):
            _forward(d2ac, values)

    def test_d2ac_and_d2ad_trainable_schema_and_initialization_are_identical(self):
        torch.manual_seed(42)
        d2ac = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AC,
        )
        torch.manual_seed(42)
        d2ad = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AD,
        )
        ac_parameters = dict(d2ac.named_parameters())
        ad_parameters = dict(d2ad.named_parameters())
        self.assertEqual(
            {name: tuple(value.shape) for name, value in ac_parameters.items()},
            {name: tuple(value.shape) for name, value in ad_parameters.items()},
        )
        for name in ac_parameters:
            self.assertTrue(torch.equal(ac_parameters[name], ad_parameters[name]), name)
        self.assertEqual(
            d2ac.network.interaction_adapter.basis_coordinate_system,
            "raw_z_up_global_query",
        )
        self.assertEqual(
            d2ad.network.interaction_adapter.basis_coordinate_system,
            LOCAL_BASIS_COORDINATE_SYSTEM,
        )

    def test_alpha_zero_shared_trunk_parity(self):
        torch.manual_seed(42)
        base = build_expert(
            "hoi", dim_model=512, num_heads=16, num_layers=8,
        )
        torch.manual_seed(99)
        d2ad = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AD,
        )
        missing, unexpected = d2ad.load_state_dict(base.state_dict(), strict=False)
        self.assertTrue(missing)
        self.assertFalse(unexpected)
        base.eval()
        d2ad.eval()
        values = _inputs(batch=1)
        with torch.no_grad():
            expected = base(
                values["noisy"],
                values["timesteps"],
                values["text"],
                values["global_bps"],
                values["goals"],
                values["progress"],
            )
            actual = _forward(d2ad, values)
        self.assertLessEqual(float((expected - actual).abs().max()), 1.0e-6)

    def test_initial_and_activated_adapter_gradients(self):
        model = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AD,
        )
        values = _inputs(batch=2)
        prediction = _forward(model, values)
        (prediction - values["noisy"]).square().mean().backward()
        initial = _d2ac_gradient_audit(model, require_adapter_paths=False)
        self.assertTrue(initial["alpha"]["finite"])
        self.assertTrue(initial["alpha"]["nonzero"])
        model.zero_grad(set_to_none=True)
        with torch.no_grad():
            model.network.interaction_adapter.alpha.copy_(
                torch.atanh(torch.tensor(0.1))
            )
        prediction = _forward(model, values)
        (prediction - values["noisy"]).square().mean().backward()
        activated = _d2ac_gradient_audit(model, require_adapter_paths=True)
        self.assertAlmostEqual(activated["gate_value"], 0.1, places=5)
        self.assertTrue(all(
            group["finite"] and group["nonzero"]
            for group in activated["adapter_groups"].values()
        ))

    def test_local_correspondence_permutation_is_causal(self):
        model = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AD,
        ).eval()
        values = _inputs(batch=1)
        with torch.no_grad():
            model.network.interaction_adapter.set_diagnostic_variant("full")
            model.network.interaction_adapter.set_gate_override(0.1)
            full = _forward(model, values)
            model.network.interaction_adapter.set_diagnostic_variant(
                "local_correspondence_permuted"
            )
            permuted = _forward(model, values)
        self.assertGreater(float((full - permuted).abs().max()), 1.0e-8)

    def test_feature_extremes_roles_dtype_and_batch_propagate(self):
        adapter = LocalObjectInteractionAdapter(
            basis_coordinate_system=LOCAL_BASIS_COORDINATE_SYSTEM,
        ).double()
        for value in (
            torch.zeros(3, 1024, 3, dtype=torch.float64),
            torch.full((3, 1024, 3), 3.0, dtype=torch.float64),
            torch.full((3, 1024, 3), 1.0e4, dtype=torch.float64),
        ):
            features = adapter.local_features(value)
            self.assertEqual(tuple(features.shape), (3, 16, 10))
            self.assertEqual(features.dtype, torch.float64)
            self.assertTrue(torch.isfinite(features).all())
        motion = torch.randn(
            3,
            16,
            512,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(7),
        )
        output = adapter(motion, torch.randn(
            3,
            1024,
            3,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(8),
        ))
        self.assertEqual(tuple(output.shape), (3, 16, 512))
        self.assertEqual(output.dtype, torch.float64)
        self.assertTrue(torch.isfinite(output).all())
        self.assertGreater(
            float((adapter.part_embedding[0] - adapter.part_embedding[1]).abs().max()),
            0.0,
        )


class D2ADGovernanceTests(unittest.TestCase):
    def merged_config(self):
        base = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior.yaml")
        d2ad = OmegaConf.load(ROOT / "code/config/config_train_hoi_prior_d2ad.yaml")
        config = OmegaConf.merge(base, d2ad)
        config.repo_root = str(ROOT)
        config.split_manifest = str(
            ROOT / "experiments/splits/omomo_hoi_train_validation_seed42.json"
        )
        return config

    def test_exact_training_contract_and_optimizer(self):
        config = self.merged_config()
        _validate_d2ad_contract(config, 4)
        self.assertEqual(config.local_bps_query_workers, DEFAULT_QUERY_WORKERS)
        self.assertEqual(config.num_workers, 4)
        self.assertEqual(
            _locked_loss_weights(config),
            {
                "fk": 0.3569973401779424,
                "object_surface": 0.4772322188400037,
                "velocity": 0.1,
                "terminal_goal": 1.0,
            },
        )
        self.assertEqual(_optimization_contract(config)["optimizer"], "Adam")
        self.assertEqual(_optimization_contract(config)["scheduler"], "none")

    def test_contract_rejects_mutations(self):
        for field, value in (
            ("init_checkpoint", "/tmp/released.pth"),
            ("weight_init_checkpoint", "/tmp/d2ac.pth"),
            ("pause_after_windows", 3_072_000),
            ("fk_foot_temporal_routing", False),
            ("local_bps_query_workers", 1),
            ("num_workers", 0),
            ("d2ac_interaction_adapter", True),
        ):
            config = self.merged_config()
            setattr(config, field, value)
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, "D2-AD"):
                    _validate_d2ad_contract(config, 4)

    @staticmethod
    def _checkpoint(variant: str, adapter_contract=None):
        value = {
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "model_config": {
                "dim_model": 512,
                "num_heads": 16,
                "num_layers": 8,
                "architecture_variant": variant,
            },
        }
        if variant != HOI_ARCHITECTURE_BASE:
            value["architecture_variant"] = variant
            value["interaction_adapter_contract"] = adapter_contract
        return value

    def test_checkpoint_variant_and_geometry_provenance_rejection(self):
        d2ac_contract = {
            "bps_sha256": BPS_SHA256,
            "assignment_sha256": ASSIGNMENT_SHA256,
            "adapter_parameters": ADAPTER_PARAMETER_COUNT,
        }
        d2ad_contract = {
            **d2ac_contract,
            "basis_coordinate_system": LOCAL_BASIS_COORDINATE_SYSTEM,
            "basis_yup_tensor_sha256": BPS_YUP_TENSOR_SHA256,
            "rest_mesh_manifest_sha256": REST_MESH_MANIFEST_SHA256,
            "object_mapping_sha256": OBJECT_MAPPING_SHA256,
            "query_backend": "scipy.spatial.cKDTree.query",
            "query_parameters": {"k": 1, "eps": 0.0, "p": 2},
            "query_workers": DEFAULT_QUERY_WORKERS,
            "full_rest_mesh": True,
            "mesh_subsample": False,
            "stored_per_window_local_bps": False,
        }
        with tempfile.TemporaryDirectory() as temporary:
            temporary = Path(temporary)
            ac_path = temporary / "d2ac.pth"
            ad_path = temporary / "d2ad.pth"
            malformed_path = temporary / "d2ad-malformed.pth"
            torch.save(
                self._checkpoint(HOI_ARCHITECTURE_D2AC, d2ac_contract),
                ac_path,
            )
            torch.save(
                self._checkpoint(HOI_ARCHITECTURE_D2AD, d2ad_contract),
                ad_path,
            )
            torch.save(
                self._checkpoint(
                    HOI_ARCHITECTURE_D2AD,
                    {**d2ad_contract, "query_backend": "approximate"},
                ),
                malformed_path,
            )
            with self.assertRaisesRegex(ValueError, "architecture variant mismatch"):
                load_trained_hoi_prior(
                    str(ac_path),
                    torch.device("cpu"),
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AD,
                )
            with self.assertRaisesRegex(ValueError, "architecture variant mismatch"):
                load_trained_hoi_prior(
                    str(ad_path),
                    torch.device("cpu"),
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AC,
                )
            with self.assertRaisesRegex(ValueError, "local-geometry provenance"):
                load_trained_hoi_prior(
                    str(malformed_path),
                    torch.device("cpu"),
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AD,
                )
            wrong_workers_path = temporary / "d2ad-wrong-workers.pth"
            torch.save(
                self._checkpoint(
                    HOI_ARCHITECTURE_D2AD,
                    {**d2ad_contract, "query_workers": 1},
                ),
                wrong_workers_path,
            )
            with self.assertRaisesRegex(ValueError, "local-geometry provenance"):
                load_trained_hoi_prior(
                    str(wrong_workers_path),
                    torch.device("cpu"),
                    expected_architecture_variant=HOI_ARCHITECTURE_D2AD,
                )

    def test_hsi_independence_mixer_contract_and_sampler_routing(self):
        hoi = build_expert(
            "hoi",
            dim_model=512,
            num_heads=16,
            num_layers=8,
            architecture_variant=HOI_ARCHITECTURE_D2AD,
        )
        hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
        assert_parameter_independence(hoi, hsi)
        output = _forward(hoi, _inputs(batch=1))
        self.assertEqual(tuple(output.shape), (1, 16, 232))

        class Dataset:
            load_scene = False

        sampler = HOIPriorSampler(device="cpu", auto_regre_num=2, timesteps=500)
        sampler.set_dataset_and_model(Dataset(), hoi)
        self.assertIsInstance(sampler.local_bps_builder, LocalObjectBPSBuilder)
        base = build_expert(
            "hoi", dim_model=32, num_heads=4, num_layers=1,
        )
        sampler.set_dataset_and_model(Dataset(), base)
        self.assertIsNone(sampler.local_bps_builder)

    def test_static_scope_and_official_evaluator_hashes(self):
        d2ad = (ROOT / "code/priors/d2ad.py").read_text(encoding="utf-8")
        model = (ROOT / "code/priors/models.py").read_text(encoding="utf-8")
        for forbidden in (
            "from eval_metrics",
            "contact_label",
            "near_ground",
            "np.save",
            "torch.save",
            "pickle.dump",
            "@lru_cache",
        ):
            self.assertNotIn(forbidden, d2ad)
        self.assertNotIn("local_object_bps", inspect.signature(
            build_expert("hsi", dim_model=32, num_heads=4, num_layers=1).forward
        ).parameters)
        self.assertNotIn("scene_condition", inspect.signature(HOIPrior.forward).parameters)
        self.assertIn("global_bps_token_preserved", (
            ROOT / "code/train_hoi_prior.py"
        ).read_text(encoding="utf-8"))
        self.assertIn("local_object_bps", model)
        locked = {
            "code/test_infbagel_hoi.py":
                "22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524",
            "code/eval_metrics.py":
                "445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547",
            "code/config/config_eval_hoi_prior.yaml":
                "89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73",
            "tools/run_hoi_d2x_evaluation.py":
                "b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31",
            "code/priors/interaction_diagnostic.py":
                "e9a0157f80695469a53a5333b20685cb3c66d042b0ccd621b86164238764bcc5",
        }
        for relative, expected in locked.items():
            self.assertEqual(_sha256(ROOT / relative), expected)

    def test_registered_lifecycle_ids_and_runner_protocol_are_fixed(self):
        self.assertEqual(
            CPU_RUN_ID,
            "p1-hoi-d2ad-cpu-contract-s42-20260728",
        )
        self.assertEqual(
            SMOKE_RUN_ID,
            "p1-hoi-d2ad-gpu-smoke-s42-20260728",
        )
        self.assertEqual(
            FORMAL_RUN_ID,
            "p1-hoi-d2ad-local-frame-interaction-adapter-s42-20260728",
        )
        self.assertEqual(INTERNAL_TRAINING_RUN_ID, FORMAL_RUN_ID)
        self.assertEqual(NATIVE_TRAINING_RUN_ID, FORMAL_RUN_ID)
        self.assertIsNotNone(INTERNAL_RUN_ID_RE.fullmatch(
            "p1-hoi-d2ad-local-frame-interaction-adapter-internal-s42-20260728"
        ))
        self.assertIsNotNone(NATIVE_RUN_ID_RE.fullmatch(
            "p1-hoi-d2ad-native-eval-s42-20260728"
        ))
        internal_source = (
            ROOT / "tools/run_hoi_d2ad_internal.py"
        ).read_text(encoding="utf-8")
        native_source = (
            ROOT / "tools/run_hoi_d2ad_native_evaluation.py"
        ).read_text(encoding="utf-8")
        smoke_source = (
            ROOT / "tools/smoke_hoi_d2ad.py"
        ).read_text(encoding="utf-8")
        self.assertIn("local_bps_builder.build(", internal_source)
        self.assertIn("base.current_bps(", internal_source)
        self.assertIn("paired_noise_identity", internal_source)
        self.assertNotIn(
            'item["object_bps"]',
            inspect.getsource(d2ad_internal.rollout_chunk),
        )
        self.assertIn("sealed_d2ac_descriptive_comparison", native_source)
        self.assertIn('"selection_use": False', native_source)
        self.assertIn("registered_formal_cross_attention_score_elements_estimate", smoke_source)

    def test_native_descriptive_comparison_is_reported_but_not_selected(self):
        control = {"sequence": {"contact_f1": 0.50}}
        target = {"sequence": {"contact_f1": 0.60}}
        sealed_d2ac = {"sequence": {"contact_f1": 0.55}}
        calls = []

        def fake_compare(first, second):
            calls.append((first, second))
            return {
                "target_minus_control_contact_f1": {
                    "first_mean": float(
                        next(iter(second.values()))["contact_f1"]
                    ),
                    "second_mean": float(
                        next(iter(first.values()))["contact_f1"]
                    ),
                },
            }

        with mock.patch.object(
            d2ad_native, "_d2ac_per_sequence", sealed_d2ac,
        ):
            with mock.patch.object(
                d2ad_native, "_d2ac_aggregate", {"contact_f1": 0.55},
            ):
                with mock.patch.object(
                    d2ad_native.d2ac, "compare_records", fake_compare,
                ):
                    value = d2ad_native.compare_records(control, target)
        self.assertEqual(calls, [(control, target), (sealed_d2ac, target)])
        descriptive = value["target_vs_sealed_d2ac_descriptive"]
        self.assertFalse(descriptive["selection_use"])
        self.assertEqual(
            descriptive["sealed_d2ac_target_metrics"],
            {"contact_f1": 0.55},
        )


if __name__ == "__main__":
    unittest.main()
