#!/usr/bin/env python3
"""Run the fixed D2-AG0 self-conditioned relation source causal diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

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
from priors.d2ag_diagnostic import (  # noqa: E402
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    DIRECT_HAND_INDICES,
    FK_PALM_INDICES,
    GT_CONTACT_FINITE_SEQUENCE_COUNT,
    GT_CONTACT_FINITE_SEQUENCE_NAMES_SHA256,
    HISTORY_MAX_ABS,
    PHYSICAL_THRESHOLDS_CM,
    REFERENCE_VARIANT,
    ROLE_NAMES,
    SELECTION_SHA256,
    SOURCE_IS_CURRENT_STEP_COUNTS,
    TEMPORAL_ANCHORS,
    VARIANTS,
    internal_mechanism_gate,
    paired_comparisons,
    reported_only_quantities,
)
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion  # noqa: E402
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SHA256,
    diffusion_schedule_contract_metadata,
)
from priors.gradient_routing import state_dict_sha256  # noqa: E402
from priors.models import HOI_ARCHITECTURE_D2AG, load_trained_hoi_prior  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402
from priors.sparse_relation import (  # noqa: E402
    D2AG_DIAGNOSTIC_OBJECT_DISPLACEMENT_M,
    D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
    D2AG_SELF_CONDITION_PROBABILITY,
    D2AG_VARIABLE_ANCHORS,
    ROUTING_SLOTS,
    SPARSE_POINT_MANIFEST_SHA256,
    SPARSE_POINT_MAPPING_SHA256,
    SPARSE_POINT_TENSOR_SHA256,
    SPARSE_RELATION_PARAMETER_COUNT,
    selfcond_relation_source_contract_metadata,
    validate_selfcond_relation_source_contract,
)
from tools import run_hoi_d2ac_internal as base  # noqa: E402
from tools import run_hoi_d2ae_internal as rollout_core  # noqa: E402


SUBPHASE = "1B-D2-AG0-internal"
RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-selfcond-relation-source-internal"
    r"(?:-r[1-9][0-9]*)?-s42-(?P<date>[0-9]{8})$"
)
TRAINING_RUN_ID_RE = re.compile(
    r"^p1-hoi-d2ag-selfcond-relation-source"
    r"(?:-r[1-9][0-9]*)?-s42-[0-9]{8}$"
)
FAILURE_CLASSIFICATION = "selfcond-relation-source-contract-failure-stop"
EXPECTED_INITIAL_MODEL_STATE_SHA256 = (
    "b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c"
)
DIFFUSION_STEPS = 500
SOURCE_LABEL_CURRENT = "x_t"
SOURCE_LABEL_PREVIOUS = "prev_x0"
EXPECTED_INTERNAL_STREAM_COORDINATES = tuple(
    (chunk_index, window_index)
    for chunk_index in range(8)
    for window_index in range(3)
)
INTERVENTION_SCOPE = (
    "anchor_source_or_high_t_gating_or_object_delta_or_temporal_blocks_or_"
    "role_blocks_only"
)
SELFCOND_APPENDIX_FILENAME = "self_conditioning_appendix.json"
REPORTED_ONLY_FILENAME = "reported_only_quantities.json"
# The D2-AG formal training run has not executed yet, so its sealed lineage
# hashes cannot exist.  They are deliberately unfilled and fail closed: the
# runner refuses to start until the completion record supplies them.  Never
# substitute a D2-AE or D2-AF value here.
FORMAL_LINEAGE_SEALED: Dict[str, Optional[str]] = {
    "training_run_id": None,
    "checkpoint_source_commit": None,
    "execution_target_commit": None,
    "manifest_sha256": None,
    "metrics_sha256": None,
    "training_state_sha256": None,
    "resume_contract_sha256": None,
    "final_checkpoint_sha256": None,
    "final_model_state_sha256": None,
}
FORMAL_CADENCE_WINDOWS = tuple(range(3_072_000, 61_440_000 + 1, 3_072_000))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def load_json_object(path: Path) -> Dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"D2-AG expected JSON object: {path}")
    return value


def sampler_seed_label(chunk_index: int, window_index: int) -> str:
    """D2-AG-specific paired sampler namespace, identical across variants."""
    return (
        "p1-hoi-d2ag-selfcond-relation-source-internal"
        f"-chunk{chunk_index:02d}-window{window_index:02d}"
    )


def validate_internal_batch_size(batch_size: int) -> None:
    if batch_size != 8:
        raise ValueError("D2-AG internal batch size must be exactly 8")


def sealed_lineage_contract(args: argparse.Namespace) -> Dict[str, object]:
    """Fail closed until the formal run seals its own lineage hashes.

    D2-AG's formal training has not run, so no sealed value can be legitimate
    yet.  Filling these in with a predecessor's hashes would silently bind the
    diagnostic to the wrong checkpoint, so the runner refuses instead.
    """
    unfilled = sorted(
        key for key, value in FORMAL_LINEAGE_SEALED.items() if value is None
    )
    if unfilled:
        raise ValueError(
            "D2-AG sealed formal lineage is not registered yet: "
            + ", ".join(unfilled)
            + "; complete the formal run and record its hashes before the "
            "internal diagnostic"
        )
    supplied = {
        "manifest_sha256": str(args.formal_manifest_sha256),
        "metrics_sha256": str(args.training_metrics_sha256),
        "training_state_sha256": str(args.training_state_sha256),
        "resume_contract_sha256": str(args.resume_contract_sha256),
    }
    mismatched = sorted(
        key for key, value in supplied.items()
        if FORMAL_LINEAGE_SEALED[key] != value
    )
    if mismatched or FORMAL_LINEAGE_SEALED["training_run_id"] != str(
        args.training_run_id
    ):
        raise ValueError(
            f"D2-AG formal lineage mismatch: {mismatched or ['training_run_id']}"
        )
    actual = {
        "formal_manifest": sha256_file(args.formal_manifest.resolve()),
        "training_metrics": sha256_file(args.training_metrics.resolve()),
        "training_state": sha256_file(args.training_state.resolve()),
        "resume_contract": sha256_file(args.resume_contract.resolve()),
        "target_checkpoint": sha256_file(args.target_checkpoint.resolve()),
    }
    expected = {
        "formal_manifest": FORMAL_LINEAGE_SEALED["manifest_sha256"],
        "training_metrics": FORMAL_LINEAGE_SEALED["metrics_sha256"],
        "training_state": FORMAL_LINEAGE_SEALED["training_state_sha256"],
        "resume_contract": FORMAL_LINEAGE_SEALED["resume_contract_sha256"],
        "target_checkpoint": FORMAL_LINEAGE_SEALED["final_checkpoint_sha256"],
    }
    if actual != expected or str(args.target_sha256) != str(
        expected["target_checkpoint"]
    ):
        raise ValueError(f"D2-AG formal artifact hash mismatch: {actual}")
    return {
        "training_run_id": str(args.training_run_id),
        "artifacts": {
            name: {"sha256": value} for name, value in sorted(actual.items())
        },
        "checkpoint_source_commit": FORMAL_LINEAGE_SEALED[
            "checkpoint_source_commit"
        ],
        "execution_target_commit": FORMAL_LINEAGE_SEALED[
            "execution_target_commit"
        ],
        "final_checkpoint_sha256": FORMAL_LINEAGE_SEALED[
            "final_checkpoint_sha256"
        ],
        "final_model_state_sha256": FORMAL_LINEAGE_SEALED[
            "final_model_state_sha256"
        ],
        "cadence_main_checkpoints": len(FORMAL_CADENCE_WINDOWS),
        "cadence_rng_sidecars": 4 * len(FORMAL_CADENCE_WINDOWS),
        "checks": {"sealed_lineage_registered": True},
    }


class RelationCapture:
    """Capture bounded GPU tensors and synchronize once after each rollout.

    Every reverse step enters the field, including the highest-``t`` step, so the
    500-call contract is satisfied naturally and no synthetic snapshot is
    inserted.  Each step records its own source identity, derived from the exact
    per-frame source-minus-current L2 rather than from Python object identity:
    the high-``t`` gate builds a new tensor with ``torch.where``, so identity
    would silently under-count its fallback steps.
    """

    FIXED_SHAPES = {
        "pooled_block_norm": (4, 3),
        "pooled_block_variance": (4, 3),
        "relation_norm": (4,),
        "temporal_permutation_sensitivity": (4,),
        "role_swap_sensitivity": (4,),
        "gate": (1,),
        "writeback_norm": (16,),
        "relation_source_minus_current_l2": (16,),
        "relation_source_history_max_abs": (1,),
        "relation_source_estimate": (1,),
        "relation_source_is_current": (1,),
    }
    TRACE_KEYS = (
        "pooled_block_norm",
        "pooled_block_variance",
        "relation_norm",
        "temporal_permutation_sensitivity",
        "role_swap_sensitivity",
        "gate",
        "writeback_norm_by_anchor",
        "writeback_variance_by_anchor",
        "source_minus_current_l2_by_anchor",
        "source_history_max_abs",
    )

    def __init__(self) -> None:
        self.traces: Dict[str, List[torch.Tensor]] = {
            key: [] for key in self.TRACE_KEYS
        }
        self.source_is_current: List[bool] = []
        self.finite: Optional[torch.Tensor] = None
        self.calls = 0

    @staticmethod
    def _anchor_reduce(value: torch.Tensor) -> torch.Tensor:
        if value.shape != (16,):
            raise ValueError("D2-AG per-frame trace must have shape [16]")
        slots = torch.as_tensor(ROUTING_SLOTS, device=value.device)
        return torch.stack([
            value[slots == slot].mean() for slot in range(4)
        ])

    @staticmethod
    def _anchor_variance(value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3 or value.shape[1:] != (16, 512):
            raise ValueError("D2-AG writeback delta must have shape [B,16,512]")
        slots = torch.as_tensor(ROUTING_SLOTS, device=value.device)
        return torch.stack([
            value[:, slots == slot].float().var(unbiased=False)
            for slot in range(4)
        ])

    def hook(self, module, inputs, output) -> None:
        snapshot = module.snapshot()
        if snapshot is None or set(snapshot) != set(self.FIXED_SHAPES):
            raise RuntimeError("D2-AG self-conditioning capture is incomplete")
        for key, shape in self.FIXED_SHAPES.items():
            if tuple(snapshot[key].shape) != shape:
                raise ValueError(f"D2-AG relation snapshot {key} is invalid")
        motion = inputs[0]
        if output.shape != motion.shape:
            raise ValueError("D2-AG writeback capture batch shape is invalid")
        if float(snapshot["relation_source_estimate"].item()) != 0.0:
            raise RuntimeError(
                "D2-AG rollout captured an estimate pass; sampling must never "
                "run the training-side estimate forward"
            )
        history_max_abs = float(
            snapshot["relation_source_history_max_abs"].item()
        )
        if history_max_abs != 0.0:
            raise ValueError(
                "D2-AG relation source history pin is not exact during rollout"
            )
        delta = (output - motion).detach().to(dtype=torch.float32)
        source_delta = snapshot["relation_source_minus_current_l2"].detach().to(
            dtype=torch.float32
        )
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
            "writeback_norm_by_anchor": self._anchor_reduce(
                snapshot["writeback_norm"].detach().to(dtype=torch.float32)
            ),
            "writeback_variance_by_anchor": self._anchor_variance(delta),
            "source_minus_current_l2_by_anchor": self._anchor_reduce(
                source_delta
            ),
            "source_history_max_abs": snapshot[
                "relation_source_history_max_abs"
            ].detach().to(dtype=torch.float32),
        })
        for key, value in values.items():
            self.traces[key].append(value)
        # Exact zero over every frame means the field read the current state
        # bitwise, which is the registered definition of an ``x_t``-source step.
        self.source_is_current.append(
            bool(float(source_delta.abs().max()) == 0.0)
        )
        if self.finite is None:
            self.finite = torch.ones((), dtype=torch.bool, device=motion.device)
        for value in values.values():
            self.finite.logical_and_(torch.isfinite(value).all())
        self.calls += 1

    def result(self) -> Dict[str, object]:
        if self.calls != DIFFUSION_STEPS:
            raise ValueError(
                f"D2-AG expected {DIFFUSION_STEPS} relation calls, got {self.calls}"
            )
        if self.finite is None or not bool(self.finite.detach().cpu()):
            raise ValueError("D2-AG relation capture contains non-finite values")
        stacked = {
            key: torch.stack(values) for key, values in self.traces.items()
        }
        sources = [
            SOURCE_LABEL_CURRENT if flag else SOURCE_LABEL_PREVIOUS
            for flag in self.source_is_current
        ]
        history_max_abs = float(
            stacked["source_history_max_abs"].abs().max().detach().cpu()
        )
        if history_max_abs != 0.0:
            raise ValueError("D2-AG history pin drifted during the rollout")
        timestep_values = {
            "timesteps": list(reversed(range(DIFFUSION_STEPS))),
            "axis": {
                "temporal_anchors": list(TEMPORAL_ANCHORS),
                "roles": list(ROLE_NAMES),
            },
            "sources": sources,
            "values": {
                key: value.detach().cpu().tolist()
                for key, value in sorted(stacked.items())
            },
        }
        return {
            "forward_calls": self.calls,
            "sources": sources,
            "source_is_x_t_count": sum(self.source_is_current),
            "first_step_source": sources[0],
            "source_history_max_abs": history_max_abs,
            "estimate_pass_observed": False,
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
        raise ValueError("D2-AG relation appendix is empty")
    keys = set(records[0]["values"])
    trace_keys = set(records[0]["by_timestep"]["values"])
    if any(set(record["values"]) != keys for record in records):
        raise ValueError("D2-AG relation appendix average keys differ")
    if any(
        set(record["by_timestep"]["values"]) != trace_keys
        or record["by_timestep"]["timesteps"]
        != list(reversed(range(DIFFUSION_STEPS)))
        for record in records
    ):
        raise ValueError("D2-AG relation appendix timestep axes differ")
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
        "source_is_x_t_counts": sorted({
            int(record["source_is_x_t_count"]) for record in records
        }),
        "first_step_sources": sorted({
            str(record["first_step_source"]) for record in records
        }),
        "source_trace_hashes": [
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


def _model_variant(variant: str) -> str:
    return "full" if variant == REFERENCE_VARIANT else variant


def expected_first_step_source(variant: str) -> str:
    """Registered first-step source label per variant.

    Every variant enters the first reverse step with ``prev_x0`` absent, so the
    source *starts* as ``x_t``.  The object-displacement gate then adds its fixed
    delta to that source, which makes the first step no longer bitwise equal to
    the current state; its label is therefore ``prev_x0``.  All other variants,
    including the reference path, keep the exact ``x_t`` first step registered in
    EP:7060-7068.
    """
    return (
        SOURCE_LABEL_PREVIOUS
        if variant == "object_displaced_counterfactual"
        else SOURCE_LABEL_CURRENT
    )


def _relation_window_identity(
    variant: str,
    windows: Sequence[Mapping[str, object]],
) -> bool:
    expected_count = SOURCE_IS_CURRENT_STEP_COUNTS[variant]
    expected_first = expected_first_step_source(variant)
    return all(
        int(window.get("forward_calls", -1)) == DIFFUSION_STEPS
        and int(window.get("source_is_x_t_count", -1)) == expected_count
        and window.get("first_step_source") == expected_first
        and window.get("estimate_pass_observed") is False
        and float(window.get("source_history_max_abs", 1.0)) == 0.0
        for window in windows
    )


def resolved_config(args: argparse.Namespace) -> Dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": args.run_id,
        "phase": "p1",
        "subphase": SUBPHASE,
        "mode": "selfcond-relation-source-internal-causal-diagnostic",
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
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
        },
        "formal_lineage": {
            "manifest": {
                "path": str(args.formal_manifest.resolve()),
                "sha256": args.formal_manifest_sha256,
            },
            "metrics": {
                "path": str(args.training_metrics.resolve()),
                "sha256": args.training_metrics_sha256,
            },
            "training_state": {
                "path": str(args.training_state.resolve()),
                "sha256": args.training_state_sha256,
            },
            "resume_contract": {
                "path": str(args.resume_contract.resolve()),
                "sha256": args.resume_contract_sha256,
            },
            "cadence_main_checkpoints": len(FORMAL_CADENCE_WINDOWS),
            "cadence_rng_sidecars": 4 * len(FORMAL_CADENCE_WINDOWS),
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
            "relation_source": (
                "previous step raw x0_hat on variable anchors, current x_t on "
                "anchor 0 and the two history frames"
            ),
            "first_reverse_step_source": "current_noisy_state",
            "relation_builder_shared_with_training": True,
            "timestep_source": "same current reverse-loop timestep per sample",
            "prepare_clean_x0_applied_to_relation_source": False,
            "relation_source_so3_projected": False,
            "clean_target": False,
            "future_gt": False,
            "stored_relation": False,
            "stored_per_frame_bps": False,
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
        "self_conditioning": {
            "formula": "motion + tanh(alpha) * routed_relation(relation_source)",
            "contract": selfcond_relation_source_contract_metadata(),
            "variable_anchors": list(D2AG_VARIABLE_ANCHORS),
            "history_anchor_source": "current_noisy_state",
            "train_selection_probability": D2AG_SELF_CONDITION_PROBABILITY,
            "relation_exposure_fraction": 1.0,
            "relation_zero_branch": False,
            "sqrt_alpha_bar_attenuation": False,
            "gates": {
                REFERENCE_VARIANT: (
                    "previous raw x0_hat on the variable anchors at every step "
                    "after the first"
                ),
                "source_substituted_xt": (
                    "variable anchors revert to the current x_t at every step; "
                    "the path stays active and in distribution"
                ),
                "high_t_restricted": (
                    "self-conditioned source only for t < "
                    f"{D2AG_HIGH_T_SELF_CONDITION_CUTOFF}; higher timesteps fall "
                    "back to x_t and are never zeroed"
                ),
                "object_displaced_counterfactual": (
                    "fixed metric "
                    f"{list(D2AG_DIAGNOSTIC_OBJECT_DISPLACEMENT_M)} m translation "
                    "on the object channels of the variable anchors only"
                ),
                "temporal_correspondence_permuted": (
                    "geometry slot k receives (k+2) mod 4"
                ),
                "left_right_role_swapped": (
                    "swap pooled left/right blocks before projection"
                ),
            },
            "registered_source_is_x_t_counts": dict(
                SOURCE_IS_CURRENT_STEP_COUNTS
            ),
            "gate_ablation_variant_used": False,
        },
        "relation": {
            "rest_object_points": [100, 3],
            "temporal_anchors": list(TEMPORAL_ANCHORS),
            "roles": list(ROLE_NAMES),
            "capture": [
                "pooled_block_norm_by_timestep_anchor_role",
                "pooled_block_variance_by_timestep_anchor_role",
                "raw_relation_norm_by_timestep_anchor",
                "writeback_norm_variance_by_timestep_anchor",
                "source_minus_current_l2_by_timestep_anchor",
                "source_label_by_timestep",
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
            "reported_only_quantities": [
                "per_sequence_contact_structure",
                "anchor_vs_fk_palm_correspondence",
            ],
            "reported_only_enter_gate": False,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--target-checkpoint", type=Path, required=True)
    parser.add_argument("--target-sha256", required=True)
    parser.add_argument("--training-run-id", required=True)
    parser.add_argument("--formal-manifest", type=Path, required=True)
    parser.add_argument("--formal-manifest-sha256", required=True)
    parser.add_argument("--training-metrics", type=Path, required=True)
    parser.add_argument("--training-metrics-sha256", required=True)
    parser.add_argument("--training-state", type=Path, required=True)
    parser.add_argument("--training-state-sha256", required=True)
    parser.add_argument("--resume-contract", type=Path, required=True)
    parser.add_argument("--resume-contract-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--resolved-config", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=base.DEFAULT_BATCH_SIZE)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resolve-only", action="store_true")
    return parser.parse_args()


def artifact_closure_entry(
    path: Path,
    metrics_path: Path,
    run_id: str,
    *,
    artifact_id: str,
) -> Dict[str, object]:
    if path.is_symlink():
        raise ValueError("D2-AG internal closure artifact must not be a symlink")
    path = path.resolve()
    root = metrics_path.resolve().parent
    try:
        relative_path = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError(
            "D2-AG internal closure artifact must share the metrics run root"
        ) from error
    return {
        "artifact_id": artifact_id,
        "relative_path": relative_path,
        "run_id": run_id,
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }


def internal_decision_evidence(
    decision: Mapping[str, object],
) -> Dict[str, object]:
    evidence = {
        "checks": decision.get("checks"),
        "contract_passed": decision.get("contract_passed"),
        "source_provenance_passed": decision.get("source_provenance_passed"),
        "high_t_provenance_passed": decision.get("high_t_provenance_passed"),
        "object_following_passed": decision.get("object_following_passed"),
        "temporal_routing_passed": decision.get("temporal_routing_passed"),
        "role_binding_passed": decision.get("role_binding_passed"),
        "mechanism_passed": decision.get("mechanism_passed"),
        "internal_status": decision.get("internal_status"),
        "classification": decision.get("classification"),
        "native_evaluation_authorized": decision.get(
            "native_evaluation_authorized"
        ),
        "reported_only_quantities_used": decision.get(
            "reported_only_quantities_used"
        ),
    }
    return {**evidence, "canonical_sha256": sha256_json(evidence)}


def main() -> None:
    args = parse_args()
    run_id_match = RUN_ID_RE.fullmatch(args.run_id)
    actual_date = datetime.now().astimezone().strftime("%Y%m%d")
    if run_id_match is None or run_id_match.group("date") != actual_date:
        raise ValueError(
            "D2-AG internal run id must use the locked stem and actual date"
        )
    if not TRAINING_RUN_ID_RE.fullmatch(args.training_run_id):
        raise ValueError("invalid D2-AG formal training run id")
    for name in (
        "target_sha256",
        "formal_manifest_sha256",
        "training_metrics_sha256",
        "training_state_sha256",
        "resume_contract_sha256",
    ):
        if not re.fullmatch(r"[0-9a-f]{64}", str(getattr(args, name))):
            raise ValueError(f"{name} must be lowercase hexadecimal SHA-256")
    validate_internal_batch_size(args.batch_size)
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    if (
        not configured_python
        or not Path(configured_python).is_absolute()
        or Path(sys.executable).resolve() != Path(configured_python).resolve()
    ):
        raise ValueError("D2-AG internal requires the absolute INFBAGEL_PYTHON")
    formal_lineage = sealed_lineage_contract(args)
    config = resolved_config(args)
    if args.resolve_only:
        base.exclusive_json(args.resolved_config.resolve(), config)
        return
    if (
        os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi"
        or socket.gethostname() != "node01"
    ):
        raise RuntimeError("D2-AG internal is restricted to the HOI worker")
    if base.git_output("status", "--porcelain=v1", "--untracked-files=all"):
        raise RuntimeError("D2-AG internal refuses a dirty worker checkout")
    if json.loads(args.resolved_config.read_text(encoding="utf-8")) != config:
        raise ValueError("D2-AG internal runtime differs from archived config")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-AG internal requires worker CUDA")
    if args.output_dir.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.output_dir.resolve()}")
    if args.metrics.resolve().exists():
        raise FileExistsError(f"refusing to overwrite {args.metrics.resolve()}")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True)
    started = time.perf_counter()
    author_utils.SMPL_DIR = str((REPO / "smpl_models").resolve())
    try:
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
            raise ValueError(f"D2-AG internal asset hash mismatch: {asset_hashes}")

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
            raise ValueError("D2-AG internal selection contract mismatch")
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
        import pickle  # noqa: PLC0415

        with (REPO / "data/train/gender.pkl").open("rb") as handle:
            genders = pickle.load(handle)
        smpl_cache: Dict[str, torch.nn.Module] = {}
        diffusion = GaussianDiffusion(DIFFUSION_STEPS).to(device)
        model, metadata = load_trained_hoi_prior(
            str(args.target_checkpoint.resolve()),
            device,
            weight_variant="online",
            expected_architecture_variant=HOI_ARCHITECTURE_D2AG,
        )
        if metadata["data_contract_sha256"] != base.EXPECTED_DATA_CONTRACT_SHA256:
            raise ValueError("D2-AG internal checkpoint data-contract mismatch")
        validate_selfcond_relation_source_contract(
            metadata.get("selfcond_relation_source_contract")
        )
        model.eval()
        model_before = state_dict_sha256(model)

        # The established rollout/metric body stays unchanged; these two hooks
        # bind the D2-AG capture and the D2-AG paired RNG namespace.
        rollout_core.RelationCapture = RelationCapture
        rollout_core.sampler_seed_label = sampler_seed_label

        variants: Dict[str, object] = {}
        records_by_variant: Dict[str, List[Dict[str, object]]] = {}
        noise_by_variant: Dict[str, object] = {}
        conditioning_by_variant: Dict[str, object] = {}
        relation_by_variant: Dict[str, object] = {}
        reported_only_by_variant: Dict[str, object] = {}
        for variant in VARIANTS:
            model.network.set_sparse_relation_diagnostic_variant(
                _model_variant(variant)
            )
            model.network.set_sparse_relation_gate_override(None)
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
                    ] = INTERVENTION_SCOPE
                    # ``relation_source`` is path-local generated state, so it
                    # legitimately differs between variants and must never enter
                    # the cross-variant identity set.
                    row["path_local_provenance"]["self_conditioning_source"] = (
                        "same_path_prev_x0"
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
            reported_only = reported_only_quantities(records)
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
                "schema_version": 1,
                "run_id": args.run_id,
                "training_run_id": args.training_run_id,
                "target_checkpoint_sha256": args.target_sha256,
                "variant": variant,
                "registered_source_is_x_t_count": (
                    SOURCE_IS_CURRENT_STEP_COUNTS[variant]
                ),
                "registered_first_step_source": expected_first_step_source(
                    variant
                ),
                "history_max_abs": history_max_abs,
                "aggregate": {
                    "semantic_and_geometry": semantic_geometry,
                    "kinematics": kinematics["aggregate"],
                    "penetration": penetration_summary,
                    "self_conditioning": relation_summary,
                },
                "kinematics_full": kinematics,
                "per_sequence": records,
                "noise_streams": noise_streams,
                "conditioning_streams": conditioning_streams,
                "relation_windows": relation_windows,
                "reported_only": reported_only,
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
                "registered_source_is_x_t_count": (
                    SOURCE_IS_CURRENT_STEP_COUNTS[variant]
                ),
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
            reported_only_by_variant[variant] = reported_only
            torch.cuda.empty_cache()

        model.network.set_sparse_relation_diagnostic_variant("full")
        model.network.set_sparse_relation_gate_override(None)
        model.network.set_sparse_relation_capture(False)
        model_after = state_dict_sha256(model)
        comparisons = paired_comparisons(records_by_variant)
        finite_masks = [
            comparisons[f"full_vs_{variant}"][
                "other_minus_full_gt_contact_distance_cm"
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
            noise_by_variant[variant] == noise_by_variant[REFERENCE_VARIANT]
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
                for row in conditioning_by_variant[REFERENCE_VARIANT]
            ]
            for variant in VARIANTS[1:]
        )
        first_window_inputs_identity = all(
            [
                row["path_local_model_inputs"]
                for row in conditioning_by_variant[variant]
                if int(row["window_index"]) == 0
            ]
            == [
                row["path_local_model_inputs"]
                for row in conditioning_by_variant[REFERENCE_VARIANT]
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
            and row["path_local_provenance"]["self_conditioning_source"]
            == "same_path_prev_x0"
            and row["path_local_provenance"][
                "intervention_scope_per_model_call"
            ] == INTERVENTION_SCOPE
            for variant in VARIANTS
            for row in conditioning_by_variant[variant]
        )
        # ``relation_source`` is path-local generated state and is deliberately
        # excluded from every identity set above: gates 1-3 exist precisely to
        # change it, so hashing it into the cross-variant contract would fail by
        # construction.  The exogenous conditions and the paired noise streams
        # are what actually protect the pairing.
        relation_source_excluded_from_identity = all(
            "relation_source" not in row["path_local_model_inputs"]["sha256"]
            and "relation_source" not in row["exogenous"]["sha256"]
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
            "shared_first_window_model_inputs": first_window_inputs_identity,
            "first_window_model_input_keys": sorted(
                conditioning_by_variant[REFERENCE_VARIANT][0][
                    "path_local_model_inputs"
                ]["sha256"]
            ),
            "later_model_inputs": "path-local after causal rollout divergence",
            "relation_source_in_identity_set": False,
            "relation_source_note": (
                "path-local generated state; gates 1-3 change it by design, so "
                "it is recorded as provenance and never hashed into the "
                "cross-variant identity set"
            ),
            "path_local_provenance_exact": path_local_provenance_exact,
            "variants": conditioning_by_variant,
        })
        relation_path = output_dir / SELFCOND_APPENDIX_FILENAME
        base.exclusive_json(relation_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "selection_use": False,
            "schedule": diffusion_schedule_contract_metadata(),
            "sqrt_alpha_bar_attenuation": False,
            "temporal_anchors": list(TEMPORAL_ANCHORS),
            "variable_anchors": list(D2AG_VARIABLE_ANCHORS),
            "roles": list(ROLE_NAMES),
            "registered_source_is_x_t_counts": dict(
                SOURCE_IS_CURRENT_STEP_COUNTS
            ),
            "variants": relation_by_variant,
        })
        reported_only_path = output_dir / REPORTED_ONLY_FILENAME
        base.exclusive_json(reported_only_path, {
            "schema_version": 1,
            "run_id": args.run_id,
            "reported_only": True,
            "enters_internal_mechanism_gate": False,
            "enters_native_gate": False,
            "selection_use": False,
            "note": (
                "per-sequence contact run structure/coverage and the "
                "anchor 24/26 versus official FK palm 22/23 comparison; both are "
                "registered as reported-only and never enter a decision"
            ),
            "variants": reported_only_by_variant,
        })

        sampler_source = inspect.getsource(rollout_core.rollout_chunk)
        diffusion_source = inspect.getsource(GaussianDiffusion.sample)
        relation_module = model.network.sparse_relation_field
        assert relation_module is not None
        contract = {
            "formal_lineage": all(formal_lineage["checks"].values()),
            "checkpoint_architecture_variant": (
                metadata["architecture_variant"] == HOI_ARCHITECTURE_D2AG
            ),
            "selfcond_contract_exact": (
                metadata.get("selfcond_relation_source_contract")
                == selfcond_relation_source_contract_metadata()
            ),
            "field_schedule_buffer_absent": (
                relation_module.sqrt_alpha_bar is None
            ),
            "asset_hashes_exact": True,
            "selection_exact": True,
            "causal_window_overlap_exact": causal_overlap["all_exact"] is True,
            "gt_contact_finite_mask_exact": gt_contact_mask_exact,
            "paired_noise_identity": paired_noise_identity,
            "generator_draw_contract_exact": generator_draw_contract_exact,
            "paired_exogenous_condition_identity": paired_exogenous_identity,
            "paired_first_window_model_input_identity": (
                first_window_inputs_identity
            ),
            "path_local_condition_provenance": path_local_provenance_exact,
            "relation_source_excluded_from_identity_set": (
                relation_source_excluded_from_identity
            ),
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
            "source_variant_identity": all(
                _relation_window_identity(
                    variant, relation_by_variant[variant]["per_window"],
                )
                for variant in VARIANTS
            ),
            "estimate_forward_absent_during_sampling": all(
                window["estimate_pass_observed"] is False
                for variant in VARIANTS
                for window in relation_by_variant[variant]["per_window"]
            ),
            "history_pin_exact_every_step": all(
                float(window["source_history_max_abs"]) == 0.0
                for variant in VARIANTS
                for window in relation_by_variant[variant]["per_window"]
            ),
            "six_variants_no_gate_ablation": (
                set(VARIANTS) == set(records_by_variant)
                and len(VARIANTS) == 6
                and "relation_gate_ablated" not in VARIANTS
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
            "reported_only_excluded_from_gate": True,
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
            "sampler_prev_x0_is_raw_x0_hat": (
                "prev_x0 = clean.detach()" in diffusion_source
                and diffusion_source.index("prev_x0 = clean.detach()")
                < diffusion_source.index("clean = prepare_clean_x0(")
            ),
            "sampler_shared_source_builder": (
                "build_d2ag_relation_source" in diffusion_source
            ),
            "sampler_future_gt_absent": "future_gt" not in sampler_source,
            "sampler_clean_target_absent": "clean_target" not in sampler_source,
            "sampler_scene_absent": "Scene" not in sampler_source,
            "sampler_stored_relation_absent": (
                "stored_relation" not in sampler_source
            ),
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
        artifact_paths = {
            **{variant: output_dir / f"{variant}.json" for variant in VARIANTS},
            "paired_noise": paired_noise_path,
            "paired_conditioning": conditioning_path,
            "causal_window_overlap": causal_overlap_path,
            "self_conditioning_appendix": relation_path,
            "reported_only_quantities": reported_only_path,
        }
        artifact_closure = {
            "schema_version": 1,
            "run_id": args.run_id,
            "training_run_id": args.training_run_id,
            "target_checkpoint_sha256": args.target_sha256,
            "artifacts": {
                artifact_id: artifact_closure_entry(
                    path, args.metrics, args.run_id, artifact_id=artifact_id,
                )
                for artifact_id, path in artifact_paths.items()
            },
        }
        if set(artifact_closure["artifacts"]) != set(VARIANTS) | {
            "paired_noise",
            "paired_conditioning",
            "causal_window_overlap",
            "self_conditioning_appendix",
            "reported_only_quantities",
        }:
            raise ValueError("D2-AG internal artifact closure is incomplete")
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
            "formal_lineage": formal_lineage,
            "checkpoint_metadata": {
                key: value
                for key, value in metadata.items()
                if key != "sparse_relation_contract"
            },
            "learned_sparse_relation": {
                "alpha": float(relation_module.alpha.detach().cpu()),
                "gate": float(torch.tanh(relation_module.alpha.detach()).cpu()),
                "contract": relation_module.contract_metadata(),
                "parameters": SPARSE_RELATION_PARAMETER_COUNT,
            },
            "self_conditioning": {
                "contract": selfcond_relation_source_contract_metadata(),
                "variable_anchors": list(D2AG_VARIABLE_ANCHORS),
                "train_selection_probability": D2AG_SELF_CONDITION_PROBABILITY,
                "relation_exposure_fraction": 1.0,
                "relation_zero_branch": False,
                "sqrt_alpha_bar_attenuation": False,
                "high_t_cutoff": D2AG_HIGH_T_SELF_CONDITION_CUTOFF,
                "object_displacement_m": list(
                    D2AG_DIAGNOSTIC_OBJECT_DISPLACEMENT_M
                ),
                "registered_source_is_x_t_counts": dict(
                    SOURCE_IS_CURRENT_STEP_COUNTS
                ),
                "gate_ablation_variant_used": False,
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
            "decision_evidence": internal_decision_evidence(decision),
            "artifact_closure": artifact_closure,
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
            "self_conditioning_appendix": {
                "path": str(relation_path),
                "sha256": sha256_file(relation_path),
                "selection_use": False,
            },
            "reported_only_quantities": {
                "path": str(reported_only_path),
                "sha256": sha256_file(reported_only_path),
                "enters_internal_mechanism_gate": False,
                "enters_native_gate": False,
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
            "native_runs_regardless_of_internal_mechanism": True,
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
