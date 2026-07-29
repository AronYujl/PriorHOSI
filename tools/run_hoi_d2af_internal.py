#!/usr/bin/env python3
"""Run the fixed D2-AF0 sqrt-alpha-bar reliability causal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import pickle
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "code"))

import utils as author_utils  # noqa: E402
from datasets.utils import get_smpl_parents  # noqa: E402
from priors.contact_alignment import (  # noqa: E402
    PHASE_OFFSETS as COHORT_PHASE_OFFSETS,
    PRIOR_ROLLOUT_OFFSETS,
)
from priors.d2af_diagnostic import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIRECT_HAND_INDICES,
    FK_PALM_INDICES,
    GT_CONTACT_FINITE_SEQUENCE_COUNT,
    GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256,
    HISTORY_MAX_ABS,
    PHYSICAL_THRESHOLDS_CM,
    ROLE_NAMES,
    SELECTION_SHA256,
    TEMPORAL_ANCHORS,
    VARIANTS,
    internal_mechanism_gate,
    paired_comparisons,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SENTINELS,
    SQRT_ALPHA_BAR_SHA256,
    canonical_diffusion_schedule,
    diffusion_schedule_contract_metadata,
)
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AF, load_trained_hoi_prior  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    ROUTING_SLOTS,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    diffusion_reliability_contract_metadata,
    validate_diffusion_reliability_contract,
)
from tools import run_hoi_d2ac_internal as base  # noqa: E402
from tools import run_hoi_d2ae_internal as rollout_core  # noqa: E402


SUBPHASE = "1B-D2-AF0-internal"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-sqrt-alpha-bar-reliability-internal"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
TRAINING_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2af-sqrt-alpha-bar-reliability"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
FAILURE_CLASSIFICATION = "diffusion-reliability-contract-failure-stop"
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
DIFFUSION_STEPS = 500


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    if chunk_index < 0 or window_index not in range(base.WINDOWS_PER_SEQUENCE):
        raise ValueError("invalid D2-AF sampler seed coordinates")
    return f"D2:d2af-shared:chunk:{chunk_index}:window:{window_index}"


def checkpoint_contract(
    path: Path,
    expected_sha256: str,
    training_run_id: str,
) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_name = f"{training_run_id}_windows061440000.pth"
    if actual != expected_sha256:
        raise ValueError(f"D2-AF final checkpoint hash mismatch: {actual}")
    if path.name != expected_name:
        raise ValueError("D2-AF internal requires the fixed final checkpoint basename")
    checkpoint = torch.load(path, map_location="cpu")
    initialization = checkpoint.get("weight_initialization", {})
    reliability = validate_diffusion_reliability_contract(
        checkpoint.get("diffusion_reliability_contract")
    )
    resume = checkpoint.get("resume_contract", {})
    checks = {
        "schema_version": checkpoint.get("schema_version") == 2,
        "checkpoint_type": checkpoint.get("checkpoint_type") == "hoi_prior_phase1b",
        "window_state_codec": checkpoint.get("window_state_codec")
        == "state-compositional-v1",
        "expert": checkpoint.get("expert") == "hoi",
        "run_id": checkpoint.get("run_id") == training_run_id,
        "seed": checkpoint.get("seed") == 42,
        "processed_windows": checkpoint.get("processed_windows") == 61_440_000,
        "processed_frames": checkpoint.get("processed_frames") == 983_040_000,
        "optimizer_updates": checkpoint.get("optimizer_updates") == 30_000,
        "world_size": checkpoint.get("world_size") == 4,
        "effective_batch_size": checkpoint.get("effective_batch_size") == 2048,
        "architecture_variant": (
            checkpoint.get("architecture_variant") == HOI_ARCHITECTURE_D2AF
            and checkpoint.get("model_config", {}).get("architecture_variant")
            == HOI_ARCHITECTURE_D2AF
        ),
        "independent_reliability_provenance": (
            reliability == diffusion_reliability_contract_metadata()
            and checkpoint.get("sparse_relation_contract") is None
        ),
        "data_contract": (
            checkpoint.get("data_contract_sha256")
            == base.EXPECTED_DATA_CONTRACT_SHA256
        ),
        "split": checkpoint.get("split_sha256") == base.EXPECTED_SPLIT_SHA256,
        "random_initialization": (
            checkpoint.get("initialization") == "random"
            and initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("source_checkpoint_sha256") is None
            and initialization.get("source_model_state_sha256") is None
            and initialization.get("restored_components") == []
            and all(
                initialization.get(name) == 0
                for name in (
                    "old_optimizer_states_loaded",
                    "old_ema_models_loaded",
                    "old_scheduler_states_loaded",
                    "old_scaler_states_loaded",
                    "old_rng_states_loaded",
                )
            )
            and initialization.get("initial_model_state_sha256")
            == EXPECTED_INITIAL_MODEL_STATE_SHA256
        ),
        "no_ema": checkpoint.get("ema_models") == {},
        "online_model": isinstance(checkpoint.get("model"), dict),
        "d2x_routing": resume.get("fk_foot_temporal_routing") is True,
        "d2ab_disabled": resume.get("d2ab_predicted_support_no_slip") is False,
        "d2ac_disabled": resume.get("d2ac_interaction_adapter") is not True,
        "d2ad_disabled": resume.get("d2ad_local_frame_interaction_adapter") is not True,
        "predecessor_sparse_relation_variant_disabled": (
            resume.get("d2ae_sparse_relation_field") is not True
        ),
        "d2af_enabled": (
            resume.get("d2af_sqrt_alpha_bar_reliability") is True
            and resume.get("architecture_variant") == HOI_ARCHITECTURE_D2AF
        ),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"D2-AF final checkpoint contract mismatch: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
        "run_id": training_run_id,
        "git_commit": checkpoint.get("git_commit"),
        "checks": checks,
        "initial_model_state_sha256": initialization.get(
            "initial_model_state_sha256"
        ),
        "diffusion_reliability_contract": reliability,
    }


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": SUBPHASE,
        "mode": "sqrt-alpha-bar-reliability-internal-causal-diagnostic",
        "seed": 42,
        "git_commit": base.git_output("rev-parse", "HEAD"),
        "repo_root": str(REPO),
        "python": str(Path(sys.executable).resolve()),
        "device": args.device,
        "batch_size": args.batch_size,
        "target_checkpoint": {
            "path": str(args.target_checkpoint.resolve()),
            "sha256": args.target_sha256,
            "run_id": args.training_run_id,
            "processed_windows": 61_440_000,
            "weight_variant": "online",
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
        },
        "selection": {
            "partition": "internal_validation",
            "source": "sealed D2-O cohort",
            "phase_offsets": [14, 56, 98],
            "sequences": 64,
            "windows_per_sequence": 3,
            "windows": 192,
            "sha256": SELECTION_SHA256,
        },
        "variants": list(VARIANTS),
        "sampling": {
            "diffusion_steps": DIFFUSION_STEPS,
            "paired_initial_latent_and_posterior_noise": True,
            "same_exogenous_conditions_and_window_order": True,
            "path_local_generated_history_restoration": True,
            "causal_window_overlap": (
                "previous sampled tail [start+42,start+45] equals next "
                "history [next_start,next_start+3]"
            ),
            "history_restoration": True,
            "global_bps": "recomputed_from_each generated object reference",
            "relation_source": "current diffusion state x_t only",
            "relation_builder_shared_with_training": True,
            "timestep_source": "same current reverse-loop timestep per sample",
            "future_gt": False,
            "previous_predicted_x0_relation": False,
            "stored_relation": False,
            "scene": False,
            "cfg": False,
            "guidance": False,
            "dynamic_perception": False,
            "generator_draws_per_window": {
                "initial_latent": 1,
                "posterior_noise": 499,
                "timestep_zero_noise": "zeros_without_generator_draw",
            },
        },
        "diffusion_reliability": {
            "formula": (
                "motion + sqrt_alpha_bar[current_timestep] "
                "* tanh(alpha) * routed_relation"
            ),
            "schedule": diffusion_schedule_contract_metadata(),
            "full_rho": "canonical sqrt_alpha_bar at every reverse step",
            "unit_rho": "force rho=1 only; all other state remains paired",
            "relation_gate_ablated": "force tanh(alpha)=0 at every model call",
            "temporal_correspondence_permuted": (
                "geometry slot k receives (k+2) mod 4 with canonical rho"
            ),
            "left_right_role_swapped": (
                "swap pooled left/right blocks before projection with canonical rho"
            ),
            "global_scaling_including_history_anchor": True,
            "per_anchor_scaling": False,
            "unit_rho_training": False,
            "selection_use": False,
        },
        "relation": {
            "rest_object_points": [100, 3],
            "temporal_anchors": list(TEMPORAL_ANCHORS),
            "roles": list(ROLE_NAMES),
            "capture": [
                "pooled_block_norm_by_timestep_anchor_role",
                "pooled_block_variance_by_timestep_anchor_role",
                "raw_relation_norm_by_timestep_anchor",
                "rho_by_timestep",
                "raw_writeback_norm_variance_by_timestep_anchor",
                "attenuated_writeback_norm_variance_by_timestep_anchor",
                "temporal_permutation_sensitivity",
                "role_swap_sensitivity",
                "gate",
            ],
            "selection_use": False,
        },
        "metrics": {
            "semantic_contact_units": ["left_hand", "right_hand", "union"],
            "semantic_thresholds": [0.5, 0.75, 0.95],
            "physical_thresholds_cm": list(PHYSICAL_THRESHOLDS_CM),
            "direct_hand_indices": list(DIRECT_HAND_INDICES),
            "fk_palm_indices": list(FK_PALM_INDICES),
            "gt_contact_frame_definition": (
                "target direct-hand union physical distance below 5cm"
            ),
            "gt_contact_distance_finite_mask": (
                "fixed target-derived sequence mask; no missing-value imputation"
            ),
            "gt_contact_distance_finite_sequence_count": (
                GT_CONTACT_FINITE_SEQUENCE_COUNT
            ),
            "gt_contact_distance_finite_sequence_names_sha256": (
                GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256
            ),
            "penetration_zero_denominator": (
                "undefined ratio plus unchanged paired absolute difference"
            ),
            "paired_unit": "sequence",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "assets": {
            "data_contract_sha256": base.EXPECTED_DATA_CONTRACT_SHA256,
            "split_sha256": base.EXPECTED_SPLIT_SHA256,
            "normalization_sha256": base.EXPECTED_NORMALIZATION_SHA256,
            "bps_sha256": base.BPS_SHA256,
            "sparse_mapping_sha256": SPARSE_POINT_MAPPING_SHA256,
            "sparse_manifest_sha256": SPARSE_POINT_MANIFEST_SHA256,
            "sparse_tensor_sha256": SPARSE_POINT_TENSOR_SHA256,
            "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
        },
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_writes": 0,
        "checkpoint_selection": False,
        "official_test_used": False,
        "native_runs_regardless_of_internal_mechanism": True,
        "output_dir": str(args.output_dir.resolve()),
        "metrics_path": str(args.metrics.resolve()),
    }


class RelationCapture:
    """Capture bounded GPU tensors and synchronize once after each rollout."""

    FIXED_SHAPES = {
        "pooled_block_norm": (4, 3),
        "pooled_block_variance": (4, 3),
        "relation_norm": (4,),
        "temporal_permutation_sensitivity": (4,),
        "role_swap_sensitivity": (4,),
        "gate": (1,),
        "raw_writeback_norm": (16,),
        "attenuated_writeback_norm": (16,),
    }
    TRACE_KEYS = (
        "pooled_block_norm",
        "pooled_block_variance",
        "relation_norm",
        "temporal_permutation_sensitivity",
        "role_swap_sensitivity",
        "gate",
        "rho",
        "raw_writeback_norm_by_anchor",
        "raw_writeback_variance_by_anchor",
        "attenuated_writeback_norm_by_anchor",
        "attenuated_writeback_variance_by_anchor",
    )

    def __init__(self) -> None:
        self.traces: Dict[str, List[torch.Tensor]] = {
            key: [] for key in self.TRACE_KEYS
        }
        self.rho_batch_spread: List[torch.Tensor] = []
        self.finite: torch.Tensor | None = None
        self.calls = 0

    @staticmethod
    def _anchor_reduce(value: torch.Tensor) -> torch.Tensor:
        if value.shape != (16,):
            raise ValueError("D2-AF writeback token norm must have shape [16]")
        slots = torch.as_tensor(ROUTING_SLOTS, device=value.device)
        return torch.stack([
            value[slots == slot].mean() for slot in range(4)
        ])

    @staticmethod
    def _anchor_variance(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (16, 512):
            raise ValueError("D2-AF writeback delta must have shape [B,16,512]")
        slots = torch.as_tensor(ROUTING_SLOTS, device=value.device)
        return torch.stack([
            value[:, slots == slot].float().var(unbiased=False)
            for slot in range(4)
        ])

    def hook(self, module, inputs, output) -> None:
        snapshot = module.snapshot()
        expected = set(self.FIXED_SHAPES) | {"rho"}
        if snapshot is None or set(snapshot) != expected:
            raise RuntimeError("D2-AF diffusion-reliability capture is incomplete")
        for key, shape in self.FIXED_SHAPES.items():
            value = snapshot[key]
            if tuple(value.shape) != shape:
                raise ValueError(f"D2-AF relation snapshot {key} is invalid")
        rho = snapshot["rho"]
        motion = inputs[0]
        if (
            rho.ndim != 1
            or rho.shape[0] != motion.shape[0]
            or output.shape != motion.shape
        ):
            raise ValueError("D2-AF rho/writeback capture batch shape is invalid")
        rho = rho.detach().to(dtype=torch.float32)
        delta = (output - motion).detach().to(dtype=torch.float32)
        raw_delta = delta / rho[:, None, None]
        values = {
            key: snapshot[key].detach().to(dtype=torch.float32)
            for key in (
                "pooled_block_norm",
                "pooled_block_variance",
                "relation_norm",
                "temporal_permutation_sensitivity",
                "role_swap_sensitivity",
                "gate",
            )
        }
        values.update({
            "rho": rho.mean().reshape(1),
            "raw_writeback_norm_by_anchor": self._anchor_reduce(
                snapshot["raw_writeback_norm"].detach().to(dtype=torch.float32)
            ),
            "raw_writeback_variance_by_anchor": self._anchor_variance(raw_delta),
            "attenuated_writeback_norm_by_anchor": self._anchor_reduce(
                snapshot["attenuated_writeback_norm"].detach().to(
                    dtype=torch.float32
                )
            ),
            "attenuated_writeback_variance_by_anchor": self._anchor_variance(
                delta
            ),
        })
        for key, value in values.items():
            self.traces[key].append(value)
        spread = rho.max() - rho.min()
        self.rho_batch_spread.append(spread)
        if self.finite is None:
            self.finite = torch.ones((), dtype=torch.bool, device=motion.device)
        for value in values.values():
            self.finite.logical_and_(torch.isfinite(value).all())
        self.calls += 1

    def result(self) -> Dict[str, object]:
        if self.calls != DIFFUSION_STEPS:
            raise ValueError(
                f"D2-AF expected {DIFFUSION_STEPS} relation calls, got {self.calls}"
            )
        if self.finite is None or not bool(self.finite.detach().cpu()):
            raise ValueError("D2-AF relation capture contains non-finite values")
        stacked = {
            key: torch.stack(values)
            for key, values in self.traces.items()
        }
        spread = torch.stack(self.rho_batch_spread)
        if float(spread.max().detach().cpu()) > 1.0e-7:
            raise ValueError("D2-AF rho differs within a shared-timestep batch")
        rho = stacked["rho"].reshape(DIFFUSION_STEPS)
        expected = canonical_diffusion_schedule()["sqrt_alpha_bar"].to(rho)
        expected = expected.flip(0)
        canonical_max_abs = float((rho - expected).abs().max().detach().cpu())
        unit_max_abs = float((rho - 1.0).abs().max().detach().cpu())
        if canonical_max_abs <= 1.0e-7:
            rho_mode = "canonical"
        elif unit_max_abs <= 1.0e-7:
            rho_mode = "unit"
        else:
            raise ValueError("D2-AF rho trace is neither canonical nor unit-rho")
        timestep_values = {
            "timesteps": list(reversed(range(DIFFUSION_STEPS))),
            "axis": {
                "temporal_anchors": list(TEMPORAL_ANCHORS),
                "roles": list(ROLE_NAMES),
            },
            "values": {
                key: value.detach().cpu().tolist()
                for key, value in sorted(stacked.items())
            },
        }
        sentinels = {
            str(timestep): float(
                rho[DIFFUSION_STEPS - 1 - timestep].detach().cpu()
            )
            for timestep in SQRT_ALPHA_BAR_SENTINELS
        }
        return {
            "forward_calls": self.calls,
            "rho_mode": rho_mode,
            "rho_canonical_max_abs": canonical_max_abs,
            "rho_unit_max_abs": unit_max_abs,
            "rho_batch_spread_max_abs": float(spread.max().detach().cpu()),
            "rho_sentinels": sentinels,
            "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
            "axis": {
                "temporal_anchors": list(TEMPORAL_ANCHORS),
                "roles": list(ROLE_NAMES),
            },
            "values": {
                key: value.mean(dim=0).detach().cpu().tolist()
                for key, value in sorted(stacked.items())
            },
            "by_timestep": timestep_values,
            "by_timestep_sha256": sha256_json(timestep_values),
        }


def aggregate_relation_windows(
    records: Sequence[Mapping[str, object]],
) -> Dict[str, object]:
    if not records:
        raise ValueError("D2-AF relation appendix is empty")
    keys = set(records[0]["values"])
    trace_keys = set(records[0]["by_timestep"]["values"])
    if any(set(record["values"]) != keys for record in records):
        raise ValueError("D2-AF relation appendix average keys differ")
    if any(
        set(record["by_timestep"]["values"]) != trace_keys
        or record["by_timestep"]["timesteps"]
        != list(reversed(range(DIFFUSION_STEPS)))
        for record in records
    ):
        raise ValueError("D2-AF relation appendix timestep axes differ")
    by_timestep = {
        "timesteps": list(reversed(range(DIFFUSION_STEPS))),
        "axis": records[0]["by_timestep"]["axis"],
        "values": {
            key: np.mean(
                np.asarray([
                    record["by_timestep"]["values"][key] for record in records
                ], dtype=np.float64),
                axis=0,
            ).tolist()
            for key in sorted(trace_keys)
        },
    }
    return {
        "window_records": len(records),
        "forward_calls_per_window": DIFFUSION_STEPS,
        "axis": records[0]["axis"],
        "rho_modes": sorted({str(record["rho_mode"]) for record in records}),
        "rho_trace_hashes": [
            str(record["by_timestep_sha256"]) for record in records
        ],
        "values": {
            key: np.mean(
                np.asarray(
                    [record["values"][key] for record in records],
                    dtype=np.float64,
                ),
                axis=0,
            ).tolist()
            for key in sorted(keys)
        },
        "by_timestep": by_timestep,
        "by_timestep_sha256": sha256_json(by_timestep),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def _model_variant(variant: str) -> str:
    return "full" if variant == "full_rho" else variant


def _relation_window_identity(
    variant: str,
    windows: Sequence[Mapping[str, object]],
) -> bool:
    expected_mode = "unit" if variant == "unit_rho" else "canonical"
    return all(
        window.get("rho_mode") == expected_mode
        and window.get("sqrt_alpha_bar_sha256") == SQRT_ALPHA_BAR_SHA256
        and int(window.get("forward_calls", -1)) == DIFFUSION_STEPS
        and float(window.get("rho_batch_spread_max_abs", float("inf"))) <= 1.0e-7
        and (
            float(window.get("rho_unit_max_abs", float("inf"))) <= 1.0e-7
            if expected_mode == "unit"
            else float(
                window.get("rho_canonical_max_abs", float("inf"))
            ) <= 1.0e-7
        )
        for window in windows
    )


def main() -> None:
    args = parse_args()
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AF internal run id must use the locked stem and actual date"
        )
    if not TRAINING_RUN_ID_RE.fullmatch(args.training_run_id):
        raise ValueError("invalid D2-AF formal training run id")
    if not re.fullmatch(r"[0-9a-f]{64}", args.target_sha256):
        raise ValueError("D2-AF target SHA-256 must be lowercase hexadecimal")
    if args.batch_size <= 0 or 64 % args.batch_size:
        raise ValueError("D2-AF internal batch size must evenly divide 64")
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(sys.executable).resolve() != Path(configured_python).resolve()
    ):
        raise ValueError("D2-AF internal requires the absolute INFBAGEL_PYTHON")
    config = resolved_config(args)
    if args.resolve_only:
        base.exclusive_json(args.resolved_config.resolve(), config)
        return
    if (
        os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi"
        or socket.gethostname() != "node01"
    ):
        raise RuntimeError("D2-AF internal is restricted to the HOI worker")
    if base.git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-AF internal refuses a dirty worker checkout")
    if json.loads(args.resolved_config.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-AF internal runtime differs from archived config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-AF internal requires worker CUDA")
    if args.output_dir.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir.resolve()}")
    if args.metrics.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.metrics.resolve()}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    author_utils.SMPL_DIR = str((REPO / "smpl_models").resolve())
    try:
        checkpoint = checkpoint_contract(
            args.target_checkpoint.resolve(),
            args.target_sha256,
            args.training_run_id,
        )
        asset_hashes = {
            "normalization": sha256_file((REPO / "data/train/norm.npy").resolve()),
            "bps": sha256_file((REPO / "code/bps.pt").resolve()),
            "split": sha256_file(
                REPO / "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
            "sparse_mapping": SPARSE_POINT_MAPPING_SHA256,
            "sparse_manifest": SPARSE_POINT_MANIFEST_SHA256,
            "sparse_tensor": SPARSE_POINT_TENSOR_SHA256,
            "sqrt_alpha_bar": SQRT_ALPHA_BAR_SHA256,
        }
        if asset_hashes != {
            "normalization": base.EXPECTED_NORMALIZATION_SHA256,
            "bps": base.BPS_SHA256,
            "split": base.EXPECTED_SPLIT_SHA256,
            "sparse_mapping": SPARSE_POINT_MAPPING_SHA256,
            "sparse_manifest": SPARSE_POINT_MANIFEST_SHA256,
            "sparse_tensor": SPARSE_POINT_TENSOR_SHA256,
            "sqrt_alpha_bar": SQRT_ALPHA_BAR_SHA256,
        }:
            raise ValueError(f"D2-AF internal asset hash mismatch: {asset_hashes}")

        base.seed_everything(42)
        dataset = PriorWindowDataset(
            str(REPO),
            "hoi",
            partition="internal_validation",
            split_manifest=(
                "experiments/splits/omomo_hoi_train_validation_seed42.json"
            ),
        )
        selection = base.select_contact_holdout(dataset)
        if (
            selection["sha256"] != SELECTION_SHA256
            or selection["sequences"] != 64
            or selection["windows"] != 192
            or selection["phase_offsets"] != [14, 56, 98]
        ):
            raise ValueError("D2-AF internal selection contract mismatch")
        triples = selection["triples"]
        causal_overlap = rollout_core.causal_overlap_contract(dataset, triples)
        causal_overlap_path = output_dir / "causal_window_overlap.json"
        base.exclusive_json(causal_overlap_path, {
            **causal_overlap,
            "run_id": args.run_id,
        })
        parents_24 = torch.from_numpy(
            get_smpl_parents(use_joints24=True).copy()
        ).long().to(device)
        parents_22 = torch.from_numpy(
            get_smpl_parents(use_joints24=False).copy()
        ).long().to(device)
        targets = base.prepare_targets(dataset, triples, parents_24.cpu())
        for target, triple in zip(targets, triples):
            target["sequence_index"] = int(
                dataset[triple[0]]["sequence_index"].item()
            )
        rest_vertices = base.load_rest_vertices(dataset, triples, device)
        penetration_assets = base.load_penetration_assets(REPO)
        betas = np.load(REPO / "data/train/betas.npy", mmap_mode="r")
        translations = np.load(
            REPO / "data/train/transl_aligned.npy", mmap_mode="r"
        )
        with (REPO / "data/train/gender.pkl").open("rb") as handle:
            genders = pickle.load(handle)
        smpl_cache: Dict[str, torch.nn.Module] = {}
        diffusion = GaussianDiffusion(DIFFUSION_STEPS).to(device)
        model, metadata = load_trained_hoi_prior(
            str(args.target_checkpoint.resolve()),
            device,
            weight_variant="online",
            expected_architecture_variant=HOI_ARCHITECTURE_D2AF,
        )
        if metadata["data_contract_sha256"] != base.EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError("D2-AF internal checkpoint data-contract mismatch")
        model.eval()
        model_before = state_dict_sha256(model)

        # The established rollout/metric body remains unchanged; only these
        # two small hooks bind the independent reliability capture and paired
        # RNG identity in this process.
        rollout_core.RelationCapture = RelationCapture
        rollout_core.sampler_seed_label = sampler_seed_label

        variants: Dict[str, object] = {}
        records_by_variant: Dict[str, List[Dict[str, object]]] = {}
        noise_by_variant: Dict[str, object] = {}
        conditioning_by_variant: Dict[str, object] = {}
        relation_by_variant: Dict[str, object] = {}
        for variant in VARIANTS:
            model.network.set_sparse_relation_diagnostic_variant(
                _model_variant(variant)
            )
            model.network.set_sparse_relation_gate_override(None)
            model.network.set_sparse_relation_rho_override(None)
            records: List[Dict[str, object]] = []
            decoded_chunks = []
            noise_streams = []
            conditioning_streams = []
            relation_windows = []
            history_max_abs = 0.0
            for chunk_index, offset in enumerate(
                range(0, len(triples), args.batch_size)
            ):
                selected = triples[offset:offset + args.batch_size]
                rollout = rollout_core.rollout_chunk(
                    model,
                    diffusion,
                    dataset,
                    selected,
                    device,
                    rest_vertices,
                    parents_24,
                    chunk_index=chunk_index,
                )
                decoded_chunks.append(rollout["decoded_steps"])
                noise_streams.extend(rollout["noise_streams"])
                for row in rollout["conditioning_streams"]:
                    row["path_local_provenance"][
                        "intervention_scope_per_model_call"
                    ] = (
                        "rho_or_gate_or_temporal_geometry_blocks_or_"
                        "left_right_pooled_blocks_only"
                    )
                conditioning_streams.extend(rollout["conditioning_streams"])
                relation_windows.extend(rollout["relation_windows"])
                history_max_abs = max(
                    history_max_abs, float(rollout["history_max_abs"])
                )
                for target, generated in zip(
                    targets[offset:offset + args.batch_size],
                    rollout["generated"],
                ):
                    penetration = base.sequence_penetration(
                        generated,
                        sequence_index=int(target["sequence_index"]),
                        object_name=str(target["object_category"]),
                        device=device,
                        parents_22=parents_22,
                        betas=betas,
                        genders=genders,
                        translations=translations,
                        penetration_assets=penetration_assets,
                        smpl_cache=smpl_cache,
                    )
                    records.append(rollout_core.analyze_sequence(
                        target,
                        generated,
                        rest_vertices,
                        device,
                        penetration=penetration,
                    ))
            decoded_steps = base.concatenate_decoded_steps(decoded_chunks, device)
            kinematics = base.physical_summary(
                dataset, triples, decoded_steps, device,
            )
            sequence_names = [str(record["sequence"]) for record in records]
            mapped_kinematics = base.kinematics_by_sequence(
                kinematics, sequence_names,
            )
            for record in records:
                record["kinematics"] = mapped_kinematics[str(record["sequence"])]
            semantic_geometry = base._summary_for_records(
                records, include_categories=True,
            )
            penetration_summary = base.aggregate_penetration(records)
            relation_summary = aggregate_relation_windows(relation_windows)
            finite = bool(
                base.all_finite(semantic_geometry)
                and base.all_finite(kinematics)
                and base.all_finite(relation_summary)
                and all(
                    all(
                        value is None or math.isfinite(float(value))
                        for key, value in record["penetration"].items()
                        if key not in {"finite", "excluded_by_official_contract"}
                    )
                    for record in records
                )
            )
            complete = rollout_core.variant_complete(records, kinematics)
            variant_value = {
                "variant": variant,
                "rho_mode": "unit" if variant == "unit_rho" else "canonical",
                "history_max_abs": history_max_abs,
                "aggregate": {
                    "semantic_and_geometry": semantic_geometry,
                    "kinematics": kinematics["aggregate"],
                    "penetration": penetration_summary,
                    "diffusion_reliability": relation_summary,
                },
                "kinematics_full": kinematics,
                "per_sequence": records,
                "noise_streams": noise_streams,
                "conditioning_streams": conditioning_streams,
                "relation_windows": relation_windows,
                "finite": finite,
                "all_fields_reported": complete,
            }
            variant_path = output_dir / f"{variant}.json"
            base.exclusive_json(variant_path, variant_value)
            variants[variant] = {
                "artifact": {
                    "path": str(variant_path),
                    "sha256": sha256_file(variant_path),
                    "bytes": variant_path.stat().st_size,
                },
                "rho_mode": variant_value["rho_mode"],
                "history_max_abs": history_max_abs,
                "aggregate": variant_value["aggregate"],
                "finite": finite,
                "all_fields_reported": complete,
            }
            records_by_variant[variant] = records
            noise_by_variant[variant] = noise_streams
            conditioning_by_variant[variant] = conditioning_streams
            relation_by_variant[variant] = {
                "aggregate": relation_summary,
                "per_window": [
                    {
                        key: value
                        for key, value in window.items()
                        if key != "by_timestep"
                    }
                    for window in relation_windows
                ],
            }
            torch.cuda.empty_cache()

        model.network.set_sparse_relation_diagnostic_variant("full")
        model.network.set_sparse_relation_gate_override(None)
        model.network.set_sparse_relation_rho_override(None)
        model.network.set_sparse_relation_capture(False)
        model_after = state_dict_sha256(model)
        comparisons = paired_comparisons(records_by_variant)
        finite_masks = [
            comparisons[f"full_rho_vs_{variant}"][
                "other_minus_full_rho_gt_contact_distance_cm"
            ]["finite_sequence_names"]
            for variant in VARIANTS[1:]
        ]
        gt_contact_mask_exact = bool(
            all(mask == finite_masks[0] for mask in finite_masks[1:])
            and len(finite_masks[0]) == GT_CONTACT_FINITE_SEQUENCE_COUNT
            and base.sequence_names_sha256(finite_masks[0])
            == GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256
        )
        paired_noise_identity = all(
            noise_by_variant[variant] == noise_by_variant["full_rho"]
            for variant in VARIANTS[1:]
        )
        paired_exogenous_identity = all(
            [
                {
                    "chunk_index": row["chunk_index"],
                    "window_index": row["window_index"],
                    "exogenous": row["exogenous"],
                }
                for row in conditioning_by_variant[variant]
            ]
            == [
                {
                    "chunk_index": row["chunk_index"],
                    "window_index": row["window_index"],
                    "exogenous": row["exogenous"],
                }
                for row in conditioning_by_variant["full_rho"]
            ]
            for variant in VARIANTS[1:]
        )
        initial_history_identity = all(
            [
                row["path_local_model_inputs"]["sha256"]["fixed_history"]
                for row in conditioning_by_variant[variant]
                if int(row["window_index"]) == 0
            ]
            == [
                row["path_local_model_inputs"]["sha256"]["fixed_history"]
                for row in conditioning_by_variant["full_rho"]
                if int(row["window_index"]) == 0
            ]
            for variant in VARIANTS[1:]
        )
        path_local_provenance_exact = all(
            row["path_local_provenance"]["fixed_history_source"]
            == (
                "immutable_selected_window_history"
                if int(row["window_index"]) == 0
                else "previous_generated_tail_from_same_variant"
            )
            and row["path_local_provenance"]["frame_source"]
            == (
                "immutable_selected_window_frame"
                if int(row["window_index"]) == 0
                else "previous_generated_tail_from_same_variant"
            )
            and row["path_local_provenance"]["global_bps_reference"]
            == "same_path_local_frame.object_reference"
            and row["path_local_provenance"]["local_goal_reference"]
            == "same_path_local_frame"
            and row["path_local_provenance"]["relation_rotation_reference"]
            == "same_path_local_frame"
            and row["path_local_provenance"][
                "intervention_scope_per_model_call"
            ] == (
                "rho_or_gate_or_temporal_geometry_blocks_or_"
                "left_right_pooled_blocks_only"
            )
            for variant in VARIANTS
            for row in conditioning_by_variant[variant]
        )
        generator_draw_contract_exact = all(
            row["draw_contract"] == {
                "initial_latent_draws": 1,
                "posterior_noise_draws": 499,
                "total_generator_draws": 500,
                "draw_shape": [
                    len(triples[offset:offset + args.batch_size]), 16, 232,
                ],
                "timestep_zero_noise": "zeros_without_generator_draw",
            }
            for variant in VARIANTS
            for offset, row in zip(
                (
                    offset
                    for offset in range(0, len(triples), args.batch_size)
                    for _ in range(base.WINDOWS_PER_SEQUENCE)
                ),
                noise_by_variant[variant],
            )
        )
        paired_noise_path = output_dir / "paired_noise.json"
        base.exclusive_json(paired_noise_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "shared": paired_noise_identity,
            "variants": noise_by_variant,
        })
        conditioning_path = output_dir / "paired_conditioning.json"
        base.exclusive_json(conditioning_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "shared_exogenous": paired_exogenous_identity,
            "shared_initial_history": initial_history_identity,
            "later_model_inputs": "path-local after causal rollout divergence",
            "path_local_provenance_exact": path_local_provenance_exact,
            "variants": conditioning_by_variant,
        })
        relation_path = output_dir / "diffusion_reliability_appendix.json"
        base.exclusive_json(relation_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "selection_use": False,
            "schedule": diffusion_schedule_contract_metadata(),
            "temporal_anchors": list(TEMPORAL_ANCHORS),
            "roles": list(ROLE_NAMES),
            "variants": relation_by_variant,
        })

        sampler_source = inspect.getsource(rollout_core.rollout_chunk)
        diffusion_source = inspect.getsource(GaussianDiffusion.sample)
        contract = {
            "checkpoint_contract": all(checkpoint["checks"].values()),
            "checkpoint_architecture_variant": (
                metadata["architecture_variant"] == HOI_ARCHITECTURE_D2AF
            ),
            "diffusion_reliability_contract_exact": (
                metadata.get("diffusion_reliability_contract")
                == diffusion_reliability_contract_metadata()
            ),
            "asset_hashes_exact": True,
            "selection_exact": True,
            "causal_window_overlap_exact": causal_overlap["all_exact"] is True,
            "gt_contact_finite_mask_exact": gt_contact_mask_exact,
            "paired_noise_identity": paired_noise_identity,
            "generator_draw_contract_exact": generator_draw_contract_exact,
            "paired_exogenous_condition_identity": paired_exogenous_identity,
            "paired_initial_history_identity": initial_history_identity,
            "path_local_condition_provenance": path_local_provenance_exact,
            "history_restoration": all(
                float(variants[variant]["history_max_abs"]) <= HISTORY_MAX_ABS
                for variant in VARIANTS
            ),
            "all_variants_finite": all(
                bool(variants[variant]["finite"]) for variant in VARIANTS
            ),
            "all_fields_reported": all(
                bool(variants[variant]["all_fields_reported"])
                for variant in VARIANTS
            ),
            "relation_capture_complete": all(
                int(window["forward_calls"]) == DIFFUSION_STEPS
                and bool(window["metadata"]["finite"])
                and str(window["metadata"]["device"]).startswith("cuda")
                for variant in VARIANTS
                for window in relation_by_variant[variant]["per_window"]
            ),
            "rho_variant_identity": all(
                _relation_window_identity(
                    variant,
                    relation_by_variant[variant]["per_window"],
                )
                for variant in VARIANTS
            ),
            "canonical_schedule_hash": (
                diffusion_schedule_contract_metadata()["sqrt_alpha_bar_sha256"]
                == SQRT_ALPHA_BAR_SHA256
            ),
            "model_state_unchanged": model_before == model_after,
            "parameter_grad_buffers_clear": all(
                parameter.grad is None for parameter in model.parameters()
            ),
            "relation_capture_descriptive_only": True,
            "penetration_zero_denominator_explicit": True,
            "current_state_relation_metadata_forwarded": all(
                name in diffusion_source
                for name in (
                    "rest_object_points",
                    "world_to_local_rotation",
                    "object_rotation_reference",
                    "position_minimum",
                    "position_maximum",
                    "object_minimum",
                    "object_maximum",
                )
            ),
            "current_timestep_forwarded": "timesteps" in diffusion_source,
            "sampler_future_gt_absent": "future_gt" not in sampler_source,
            "sampler_previous_x0_relation_absent": "previous_x0" not in sampler_source,
            "sampler_scene_absent": "Scene" not in sampler_source,
            "sampler_stored_relation_absent": "stored_relation" not in sampler_source,
            "sampler_cpu_dynamic_geometry_absent": all(
                token not in sampler_source
                for token in ("cKDTree", "scipy", "cdist", "full_mesh")
            ),
            "global_bps_recomputed_unchanged": (
                "base.current_bps(" in sampler_source
            ),
            "optimizer_absent": True,
            "checkpoint_write_absent": True,
            "official_test_absent": True,
        }
        decision = internal_mechanism_gate(contract, comparisons)
        relation_module = model.network.sparse_relation_field
        assert relation_module is not None
        result = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "completed",
            "seed": 42,
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "selection": {
                key: value for key, value in selection.items() if key != "triples"
            },
            "target_checkpoint": checkpoint,
            "checkpoint_metadata": {
                key: value
                for key, value in metadata.items()
                if key != "sparse_relation_contract"
            },
            "learned_sparse_relation": {
                "alpha": float(relation_module.alpha.detach().cpu()),
                "gate": float(torch.tanh(relation_module.alpha.detach()).cpu()),
                "contract": relation_module.contract_metadata(),
            },
            "diffusion_reliability": {
                "schedule": diffusion_schedule_contract_metadata(),
                "global_scaling_including_history_anchor": True,
                "unit_rho_training": False,
                "unit_rho_diagnostic_only": True,
            },
            "assets": {
                **asset_hashes,
                "data_contract_sha256": base.EXPECTED_DATA_CONTRACT_SHA256,
                "penetration_hand_vertex_ids_sha256": penetration_assets[
                    "hand_ids_sha256"
                ],
            },
            "variants": variants,
            "comparisons": comparisons,
            "contract": contract,
            "decision": decision,
            "internal_status": decision["internal_status"],
            "paired_noise": {
                "path": str(paired_noise_path),
                "sha256": sha256_file(paired_noise_path),
            },
            "paired_conditioning": {
                "path": str(conditioning_path),
                "sha256": sha256_file(conditioning_path),
            },
            "causal_window_overlap": {
                "path": str(causal_overlap_path),
                "sha256": sha256_file(causal_overlap_path),
                "all_exact": causal_overlap["all_exact"],
            },
            "diffusion_reliability_appendix": {
                "path": str(relation_path),
                "sha256": sha256_file(relation_path),
                "selection_use": False,
            },
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_writes": 0,
            "checkpoint_selection": False,
            "consistency_started": False,
            "official_test_used": False,
            "native_evaluation_authorized": bool(
                decision["native_evaluation_authorized"]
            ),
            "gpu": {
                "device": str(device),
                "name": torch.cuda.get_device_name(device),
                "maximum_allocated_bytes": int(
                    torch.cuda.max_memory_allocated(device)
                ),
                "maximum_reserved_bytes": int(
                    torch.cuda.max_memory_reserved(device)
                ),
            },
        }
        base.exclusive_json(args.metrics.resolve(), result)
    except Exception as error:
        failure = {
            "schema_version": 1,
            "run_id": args.run_id,
            "phase": "p1",
            "subphase": SUBPHASE,
            "status": "failed",
            "seed": 42,
            "git_commit": base.git_output("rev-parse", "HEAD"),
            "runtime_seconds": time.perf_counter() - started,
            "classification": FAILURE_CLASSIFICATION,
            "failure_type": type(error).__name__,
            "failure": str(error),
            "optimizer_created": False,
            "training_updates": 0,
            "checkpoint_writes": 0,
            "checkpoint_selection": False,
            "consistency_started": False,
            "official_test_used": False,
        }
        failure_path = output_dir / "failure.json"
        if not failure_path.exists():
            base.exclusive_json(failure_path, failure)
        if not args.metrics.resolve().exists():
            base.exclusive_json(args.metrics.resolve(), failure)
        raise


if __name__ == "__main__":
    main()
