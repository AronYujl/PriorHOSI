from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
import sys
from unittest import mock

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

from priors.d2af_diagnostic import (  # noqa: E402
    CONTACT_F1_POINT_MINIMUM,
    KINEMATIC_METRICS,
    PROTECTION_METRICS,
    RELEASED_HIGHER_IS_BETTER,
    RELEASED_LOWER_IS_BETTER,
    VARIANTS,
    internal_mechanism_gate,
    native_gate,
    paired_comparisons,
)
from priors.diffusion_schedule import (  # noqa: E402
    SQRT_ALPHA_BAR_SENTINELS,
    SQRT_ALPHA_BAR_SHA256,
    canonical_diffusion_schedule,
)
from priors.models import HOI_ARCHITECTURE_D2AF  # noqa: E402
from tools import run_hoi_d2af_internal as internal_runner  # noqa: E402
from tools import run_hoi_d2af_native_evaluation as native_runner  # noqa: E402


LOCKED_SEQUENCE_NAMES = [
    "sub14_floorlamp_023", "sub8_largebox_018", "sub15_woodchair_041",
    "sub6_trashcan_018", "sub6_woodchair_070", "sub14_smalltable_025",
    "sub13_whitechair_059", "sub1_floorlamp_064", "sub7_woodchair_001",
    "sub14_smallbox_011", "sub12_tripod_074", "sub10_whitechair_080",
    "sub1_plasticbox_019", "sub11_monitor_016", "sub5_largetable_013",
    "sub9_smalltable_038", "sub13_woodchair_062", "sub6_whitechair_054",
    "sub11_monitor_117", "sub8_largebox_049", "sub15_woodchair_056",
    "sub9_smalltable_016", "sub3_largebox_016", "sub14_woodchair_010",
    "sub7_largebox_011", "sub7_largebox_010", "sub5_whitechair_059",
    "sub11_clothesstand_058", "sub11_monitor_013", "sub6_whitechair_021",
    "sub8_trashcan_006", "sub6_woodchair_032", "sub14_floorlamp_027",
    "sub6_largetable_006", "sub14_woodchair_008", "sub1_tripod_031",
    "sub7_whitechair_008", "sub14_woodchair_066", "sub13_tripod_047",
    "sub6_smalltable_022", "sub14_smalltable_047", "sub5_whitechair_037",
    "sub10_tripod_006", "sub14_floorlamp_044", "sub6_trashcan_032",
    "sub15_suitcase_022", "sub13_whitechair_066", "sub11_monitor_040",
    "sub5_whitechair_014", "sub10_whitechair_034", "sub12_largebox_054",
    "sub9_smalltable_019", "sub1_largetable_041", "sub12_plasticbox_054",
    "sub9_clothesstand_031", "sub13_clothesstand_055",
    "sub1_floorlamp_045", "sub7_whitechair_026", "sub1_suitcase_039",
    "sub11_floorlamp_071", "sub1_plasticbox_044", "sub10_floorlamp_001",
    "sub5_whitechair_040", "sub9_monitor_052",
]
LOCKED_GLOBAL_INDICES = [
    187744, 187786, 187828, 532348, 532390, 532432, 245496, 245538,
    245580, 458713, 458755, 458797, 487859, 487901, 487943, 204540,
    204582, 204624, 174960, 175002, 175044, 275490, 275532, 275574,
    517570, 517612, 517654, 196323, 196365, 196407, 148414, 148456,
    148498, 37791, 37833, 37875, 293838, 293880, 293922, 71163,
    71205, 71247, 403530, 403572, 403614, 586795, 586837, 586879,
    183854, 183896, 183938, 473930, 473972, 474014, 86494, 86536,
    86578, 536265, 536307, 536349, 246426, 246468, 246510, 583163,
    583205, 583247, 348797, 348839, 348881, 219328, 219370, 219412,
    489433, 489475, 489517, 489293, 489335, 489377, 433491, 433533,
    433575, 54514, 54556, 54598, 70669, 70711, 70753, 468523,
    468565, 468607, 557518, 557560, 557602, 481776, 481818, 481860,
    188434, 188476, 188518, 435934, 435976, 436018, 219041, 219083,
    219125, 323270, 323312, 323354, 508486, 508528, 508570, 227174,
    227216, 227258, 166466, 166508, 166550, 448892, 448934, 448976,
    208054, 208096, 208138, 430439, 430481, 430523, 25853, 25895,
    25937, 190885, 190927, 190969, 461379, 461421, 461463, 236349,
    236391, 236433, 175622, 175664, 175706, 75319, 75361, 75403,
    426727, 426769, 426811, 33244, 33286, 33328, 107408, 107450,
    107492, 583712, 583754, 583796, 283891, 283933, 283975, 126911,
    126953, 126995, 566081, 566123, 566165, 161507, 161549, 161591,
    271023, 271065, 271107, 511245, 511287, 511329, 309918, 309960,
    310002, 67298, 67340, 67382, 297380, 297422, 297464, 5521, 5563,
    5605, 431102, 431144, 431186, 579064, 579106, 579148,
]


def _contact_unit(value: float):
    return {
        "precision": value,
        "recall": value,
        "f1": value,
        "prediction_percent": value,
        "prediction_run_lengths": {"mean_frames": 2.0 + value},
    }


def _record(
    sequence: str,
    *,
    f1: float,
    distance: float,
    left_f1: float | None = None,
    right_f1: float | None = None,
):
    direct = {"thresholds_cm": {}}
    fk = {"thresholds_cm": {}}
    for threshold in (2.0, 5.0, 7.5, 10.0):
        key = f"{threshold:g}"
        direct["thresholds_cm"][key] = {
            "left_hand": _contact_unit(
                f1 if left_f1 is None or key != "5" else left_f1
            ),
            "right_hand": _contact_unit(
                f1 if right_f1 is None or key != "5" else right_f1
            ),
            "union": _contact_unit(f1),
        }
        fk["thresholds_cm"][key] = {
            unit: _contact_unit(f1)
            for unit in ("left_hand", "right_hand", "union")
        }
    return {
        "sequence": sequence,
        "semantic_vs_gt": {
            "thresholds": {
                "0.5": {
                    unit: _contact_unit(f1)
                    for unit in ("left_hand", "right_hand", "union")
                }
            }
        },
        "direct_physical_geometry_vs_gt": direct,
        "fk_physical_geometry_vs_gt": fk,
        "gt_contact_frame_direct_distance": {
            "union": {"mean_cm": distance},
        },
        "kinematics": {
            metric: 1.0 + index * 0.1
            for index, metric in enumerate(KINEMATIC_METRICS)
        },
        "penetration": {
            "hand_pen_loss_omomo": 0.2,
            "human_pen_loss_infbagel": 0.3,
        },
    }


def _internal_records(count=8):
    settings = {
        "full_rho": (0.82, 2.0, 0.82, 0.82),
        "unit_rho": (0.74, 2.8, 0.74, 0.74),
        "relation_gate_ablated": (0.62, 3.2, 0.62, 0.62),
        "temporal_correspondence_permuted": (0.68, 3.0, 0.68, 0.68),
        "left_right_role_swapped": (0.78, 2.2, 0.60, 0.60),
    }
    names = (
        LOCKED_SEQUENCE_NAMES
        if count == len(LOCKED_SEQUENCE_NAMES)
        else [f"sequence-{index:02d}" for index in range(count)]
    )
    result = {}
    for variant, (f1, distance, left, right) in settings.items():
        rows = []
        for index, name in enumerate(names):
            record = _record(
                name,
                f1=f1,
                distance=distance,
                left_f1=left,
                right_f1=right,
            )
            record.update({
                "sequence_index": index,
                "object_category": (
                    name.split("_", 2)[1] if "_" in name else "fixture"
                ),
                "positions": [index * 100, index * 100 + 42, index * 100 + 84],
                "pi": [14, 56, 98],
            })
            rows.append(record)
        result[variant] = rows
    return result


def _difference(lower: float = 0.01, upper: float = 0.02):
    return {
        "bootstrap_95_ci": [lower, upper],
        "first_mean": 0.70,
        "second_mean": 0.68,
    }


def _ratio(upper: float = 1.0):
    return {
        "bootstrap_95_ci": [0.90, upper],
        "mean_ratio": 0.95,
    }


def _native_inputs():
    comparison = {
        "penetration_mask_contract": {"passed": True},
        "target_minus_control_contact_f1": _difference(),
        "target_minus_control_contact_recall": _difference(),
        "target_minus_control_contact_precision": _difference(-0.01, 0.01),
        "target_over_control_protection": {
            metric: _ratio(1.05) for metric in PROTECTION_METRICS
        },
        "contact_f1_released_gap_closure": 0.30,
        "target_vs_sealed_d2ae_repair": {
            "target_minus_control_contact_f1": _difference(),
            "target_minus_control_contact_recall": _difference(),
            "target_over_control_protection": {
                "end_obj_trans_err": _ratio(0.98),
                "foot_sliding": _ratio(0.97),
            },
        },
    }
    internal = {
        "contract_passed": True,
        "internal_status": "passed",
        "relation_path_used": True,
        "schedule_reliability_passed": True,
        "temporal_routing_passed": True,
        "role_binding_passed": True,
        "mechanism_passed": True,
    }
    target_metrics = {
        metric: 1.0
        for metric in set(RELEASED_LOWER_IS_BETTER)
        | set(RELEASED_HIGHER_IS_BETTER)
    }
    target_metrics["contact_f1"] = CONTACT_F1_POINT_MINIMUM + 0.01
    baseline_ratios = {
        metric: 1.0
        for metric in set(RELEASED_LOWER_IS_BETTER)
        | set(RELEASED_HIGHER_IS_BETTER)
    }
    return internal, comparison, target_metrics, baseline_ratios


def _formal_payloads(
    metrics_sha256=None,
    *,
    final_sha256=None,
    resume_sha256=None,
    resume_checkpoint_bytes=361_283_695,
    resume_rng_sidecars=None,
):
    run_id = internal_runner.FORMAL_TRAINING_RUN_ID
    final_sha256 = (
        final_sha256 or internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256
    )
    resume_sha256 = (
        resume_sha256 or internal_runner.FORMAL_RESUME_CHECKPOINT_SHA256
    )
    final_name = f"{run_id}_windows061440000.pth"
    resume_name = f"{run_id}_windows006144000.pth"
    transition = {
        "mode": "explicit_bound_transition",
        "checkpoint_git_commit": (
            internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT
        ),
        "current_git_commit": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
        "diff_sha256": internal_runner.FORMAL_EXECUTION_DIFF_SHA256,
    }
    metrics = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "stable",
        "git_commit": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
        "processed_windows": 61_440_000,
        "processed_frames": 983_040_000,
        "optimizer_updates": 30_000,
        "terminal_model_state_sha256": (
            internal_runner.FORMAL_FINAL_MODEL_STATE_SHA256
        ),
        "resume_checkpoint": f"/formal/checkpoints/{resume_name}",
        "resume_checkpoint_git_commit": (
            internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT
        ),
        "resume_commit_provenance": transition,
        "checkpoint_hashes": [
            {
                "processed_windows": processed_windows,
                "path": (
                    f"/formal/checkpoints/{run_id}_windows"
                    f"{processed_windows:09d}.pth"
                ),
                "sha256": (
                    final_sha256
                    if processed_windows == 61_440_000
                    else f"{processed_windows // 3_072_000:064x}"
                ),
            }
            for processed_windows in internal_runner.FORMAL_CADENCE_WINDOWS[2:]
        ],
    }
    state = {
        "schema_version": 1,
        "run_id": run_id,
        "seed": 42,
        "status": "completed",
        "amp_overflow_skips": 0,
        "processed_windows": 61_440_000,
        "processed_frames": 983_040_000,
        "optimizer_updates": 30_000,
        "resume_checkpoint": f"/formal/checkpoints/{resume_name}",
        "resume_checkpoint_git_commit": (
            internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT
        ),
        "resume_commit_provenance": transition,
        "terminal_checkpoint": f"/formal/checkpoints/{final_name}",
        "terminal_checkpoint_sha256": (
            final_sha256
        ),
    }
    resume = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "resume-contract-passed",
        "all_checks_passed": True,
        "failed_checks": [],
        "read_only": True,
        "checks": {"all_registered_checks": True},
        "resume_commit_provenance": transition,
        "checkpoint": {
            "path": f"/formal/checkpoints/{resume_name}",
            "sha256": resume_sha256,
            "bytes": resume_checkpoint_bytes,
            "processed_windows": 6_144_000,
            "optimizer_updates": 3_000,
        },
        "rng_sidecars": (
            copy.deepcopy(resume_rng_sidecars)
            if resume_rng_sidecars is not None
            else []
        ),
        "continuation_contract": {
            "sha256": internal_runner.FORMAL_CONTINUATION_CONTRACT_SHA256,
            "accepted_lineage_optimizer_updates": 30_000,
            "actual_total_gpu_optimizer_updates": 31_500,
        },
    }
    manifest = {
        "schema_version": 1,
        "experiment_id": run_id,
        "phase": "p1",
        "seed": 42,
        "status": "completed",
        "ended_at": "2026-07-30T04:42:53+00:00",
        "metrics_file": {
            "path": f"results/experiments/{run_id}/metrics.json",
            "sha256": (
                metrics_sha256
                if metrics_sha256 is not None
                else internal_runner.FORMAL_METRICS_SHA256
            ),
        },
        "metrics": copy.deepcopy(metrics),
        "git": {
            "commit": internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT,
            "dirty": False,
        },
        "final_git": {
            "commit": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
            "dirty": False,
        },
        "commit_transition": {
            "mode": "explicit_bound_transition",
            "source_commit": internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT,
            "target_commit": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
            "diff_sha256": internal_runner.FORMAL_EXECUTION_DIFF_SHA256,
        },
    }
    checkpoint = {
        "processed_windows": 61_440_000,
        "processed_frames": 983_040_000,
        "optimizer_updates": 30_000,
        "sha256": final_sha256,
        "git_commit": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
        "model_state_sha256": (
            internal_runner.FORMAL_FINAL_MODEL_STATE_SHA256
        ),
        "checks": {"fixed_checkpoint_contract": True},
    }
    return manifest, metrics, state, resume, checkpoint


def _formal_cadence_fixture(root: Path):
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir()
    rows = []
    resume_rng_sidecars = []
    final_sha256 = None
    resume_sha256 = None
    resume_checkpoint_bytes = None
    for processed_windows in internal_runner.FORMAL_CADENCE_WINDOWS:
        optimizer_updates = processed_windows // 2048
        stem = (
            f"{internal_runner.FORMAL_TRAINING_RUN_ID}_windows"
            f"{processed_windows:09d}"
        )
        records = []
        main_name = f"{stem}.pth"
        main_path = checkpoint_dir / main_name
        main_path.write_bytes(f"checkpoint:{processed_windows}".encode())
        main_sha256 = hashlib.sha256(main_path.read_bytes()).hexdigest()
        main_record = {
            "bytes": main_path.stat().st_size,
            "kind": "checkpoint",
            "path": f"/formal/checkpoints/{main_name}",
            "sha256": main_sha256,
        }
        records.append(main_record)
        rng_records = []
        for rank in range(4):
            name = f"{stem}.rank{rank}.rng.pth"
            path = checkpoint_dir / name
            path.write_bytes(f"rng:{processed_windows}:{rank}".encode())
            record = {
                "bytes": path.stat().st_size,
                "kind": "rng_sidecar",
                "path": f"/formal/checkpoints/{name}",
                "rank": rank,
                "schema_valid": True,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            records.append(record)
            rng_records.append(record)
        rows.append({
            "processed_windows": processed_windows,
            "optimizer_updates": optimizer_updates,
            "files": records,
        })
        if processed_windows == 6_144_000:
            resume_sha256 = main_sha256
            resume_checkpoint_bytes = main_path.stat().st_size
            resume_rng_sidecars = [
                {
                    "binding_exact": True,
                    "bytes": record["bytes"],
                    "path": record["path"],
                    "rank": record["rank"],
                    "schema_keys": ["cuda", "numpy", "python", "torch"],
                    "schema_valid": True,
                    "sha256": record["sha256"],
                }
                for record in rng_records
            ]
        if processed_windows == 61_440_000:
            final_sha256 = main_sha256
            final_record = main_record
    assert final_sha256 is not None
    assert resume_sha256 is not None
    assert resume_checkpoint_bytes is not None
    completion = {
        "schema_version": 1,
        "status": "formal-training-completion-verified",
        "classification": (
            "d2af-formal-continuation-completion-verification"
        ),
        "run_id": internal_runner.FORMAL_TRAINING_RUN_ID,
        "execution_head": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
        "all_checks_passed": True,
        "failed_checks": [],
        "checks": {"all_cadence_files": True},
        "cadence_checkpoints": rows,
        "final_checkpoint": {
            "path": final_record["path"],
            "bytes": final_record["bytes"],
            "sha256": final_record["sha256"],
            "processed_windows": 61_440_000,
            "processed_frames": 983_040_000,
            "optimizer_updates": 30_000,
            "model_state_sha256": (
                internal_runner.FORMAL_FINAL_MODEL_STATE_SHA256
            ),
        },
    }
    return {
        "checkpoint_dir": checkpoint_dir,
        "completion": completion,
        "final_sha256": final_sha256,
        "resume_sha256": resume_sha256,
        "resume_checkpoint_bytes": resume_checkpoint_bytes,
        "resume_rng_sidecars": resume_rng_sidecars,
    }


def _hash_bundle(prefix, shapes):
    return {
        "shapes": copy.deepcopy(shapes),
        "sha256": {
            key: hashlib.sha256(
                f"{prefix}:{key}".encode("utf-8")
            ).hexdigest()
            for key in shapes
        },
    }


def _conditioning_variants():
    result = {}
    for variant in VARIANTS:
        rows = []
        for chunk_index, window_index in native_runner.EXPECTED_STREAM_COORDINATES:
            exogenous = _hash_bundle(
                f"exogenous:{chunk_index}:{window_index}",
                native_runner.EXOGENOUS_SHAPES,
            )
            path_prefix = (
                f"first:{chunk_index}"
                if window_index == 0
                else f"later:{variant}:{chunk_index}:{window_index}"
            )
            path_local = _hash_bundle(
                path_prefix, native_runner.MODEL_INPUT_SHAPES,
            )
            path_local["sha256"]["rest_object_points"] = (
                exogenous["sha256"]["rest_object_points"]
            )
            rows.append({
                "chunk_index": chunk_index,
                "window_index": window_index,
                "path_local_provenance": native_runner._expected_provenance(
                    window_index
                ),
                "exogenous": exogenous,
                "path_local_model_inputs": path_local,
            })
        result[variant] = rows
    return result


def _noise_rows():
    rows = []
    for chunk_index, window_index in native_runner.EXPECTED_STREAM_COORDINATES:
        label = internal_runner.sampler_seed_label(chunk_index, window_index)
        rows.append({
            "chunk_index": chunk_index,
            "window_index": window_index,
            "label": label,
            "seed": internal_runner.base.stable_seed(label),
            "generator_initial_state_sha256": hashlib.sha256(
                f"initial:{label}".encode("utf-8")
            ).hexdigest(),
            "generator_final_state_sha256": hashlib.sha256(
                f"final:{label}".encode("utf-8")
            ).hexdigest(),
            "draw_contract": {
                "initial_latent_draws": 1,
                "posterior_noise_draws": 499,
                "total_generator_draws": 500,
                "draw_shape": [8, 16, 232],
                "timestep_zero_noise": "zeros_without_generator_draw",
            },
        })
    return rows


def _constant_nested(shape, value=0.0):
    if not shape:
        return float(value)
    return [_constant_nested(shape[1:], value) for _ in range(shape[0])]


def _relation_windows(variant):
    mode = "unit" if variant == "unit_rho" else "canonical"
    rho = (
        [1.0] * 500
        if mode == "unit"
        else list(reversed(
            canonical_diffusion_schedule()["sqrt_alpha_bar"].tolist()
        ))
    )
    axis = {
        "temporal_anchors": [0, 5, 10, 15],
        "roles": ["left_hand", "right_hand", "pelvis"],
    }
    trace_values = {
        key: _constant_nested(shape)
        for key, shape in native_runner.RELATION_TRACE_SHAPES.items()
    }
    trace_values["rho"] = [[value] for value in rho]
    by_timestep = {
        "timesteps": list(reversed(range(500))),
        "axis": axis,
        "values": trace_values,
    }
    mean_values = {
        key: _constant_nested(shape)
        for key, shape in native_runner.RELATION_MEAN_SHAPES.items()
    }
    mean_values["rho"] = [sum(rho) / len(rho)]
    template = {
        "forward_calls": 500,
        "rho_mode": mode,
        "rho_canonical_max_abs": (
            0.0 if mode == "canonical" else max(
                abs(current - expected)
                for current, expected in zip(
                    rho,
                    reversed(
                        canonical_diffusion_schedule()[
                            "sqrt_alpha_bar"
                        ].tolist()
                    ),
                )
            )
        ),
        "rho_unit_max_abs": max(abs(value - 1.0) for value in rho),
        "rho_batch_spread_max_abs": 0.0,
        "rho_sentinels": {
            str(timestep): rho[499 - timestep]
            for timestep in SQRT_ALPHA_BAR_SENTINELS
        },
        "sqrt_alpha_bar_sha256": SQRT_ALPHA_BAR_SHA256,
        "axis": axis,
        "values": mean_values,
        "by_timestep": by_timestep,
        "by_timestep_sha256": internal_runner.sha256_json(by_timestep),
        "metadata": {
            "rest_object_points_shape": [8, 100, 3],
            "world_to_local_rotation_shape": [8, 3, 3],
            "object_rotation_reference_shape": [8, 3, 3],
            "device": "cuda:0",
            "dtype": "torch.float32",
            "finite": True,
        },
    }
    return [
        {
            "chunk_index": chunk_index,
            "window_index": window_index,
            **copy.deepcopy(template),
        }
        for chunk_index, window_index in native_runner.EXPECTED_STREAM_COORDINATES
    ]


def _causal_overlap(run_id):
    rows = []
    for cohort_index, name in enumerate(LOCKED_SEQUENCE_NAMES):
        starts = [
            100_000 + cohort_index * 1_000 + offset
            for offset in (0, 42, 84)
        ]
        rows.append({
            "cohort_index": cohort_index,
            "sequence": name,
            "global_indices": LOCKED_GLOBAL_INDICES[
                cohort_index * 3:(cohort_index + 1) * 3
            ],
            "phase_offsets": [14, 56, 98],
            "source_starts": starts,
            "source_ends": [start + 48 for start in starts],
            "source_start_offsets": [0, 42, 84],
            "sampled_tail_to_next_history": [
                {
                    "previous_tail": [starts[step] + 42, starts[step] + 45],
                    "next_history": [
                        starts[step + 1], starts[step + 1] + 3,
                    ],
                    "exact": True,
                }
                for step in range(2)
            ],
            "checks": {
                "single_sequence": True,
                "phase_offsets": True,
                "source_window_lengths": True,
                "sampled_window_lengths": True,
                "rollout_offsets": True,
                "history_overlap": True,
            },
        })
    return {
        "schema_version": 1,
        "run_id": run_id,
        "phase_offsets": [14, 56, 98],
        "prior_rollout_offsets": [0, 42, 84],
        "source_window_frames": 48,
        "subsample_stride": 3,
        "model_window_frames": 16,
        "history_frames": 2,
        "sequences": 64,
        "all_exact": True,
        "rows": rows,
    }


def _formal_summary_fixture():
    return {
        "checks": {"formal": True},
        "training_run_id": internal_runner.FORMAL_TRAINING_RUN_ID,
        "checkpoint_source_commit": (
            internal_runner.FORMAL_CHECKPOINT_SOURCE_COMMIT
        ),
        "execution_target_commit": (
            internal_runner.FORMAL_EXECUTION_TARGET_COMMIT
        ),
        "execution_diff_sha256": internal_runner.FORMAL_EXECUTION_DIFF_SHA256,
        "final_checkpoint_sha256": (
            internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256
        ),
        "final_model_state_sha256": (
            internal_runner.FORMAL_FINAL_MODEL_STATE_SHA256
        ),
        "cadence_main_checkpoints": 20,
        "cadence_rng_sidecars": 80,
        "artifacts": {
            name: {"sha256": hashlib.sha256(name.encode("utf-8")).hexdigest()}
            for name in (
                "manifest",
                "metrics",
                "training_state",
                "resume_contract",
            )
        },
    }


def _write_internal_fixture(root: Path, *, mechanism_negative=False):
    run_id = (
        "p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-"
        f"s42-{datetime.now().astimezone().strftime('%Y%m%d')}"
    )
    records = _internal_records(64)
    if mechanism_negative:
        records["unit_rho"] = copy.deepcopy(records["full_rho"])
    comparisons = paired_comparisons(records)
    contract = {
        "formal_lineage": True,
        "paired_noise_identity": True,
        "paired_exogenous_condition_identity": True,
        "paired_first_window_model_input_identity": True,
        "causal_window_overlap_exact": True,
        "generator_draw_contract_exact": True,
        "path_local_condition_provenance": True,
        "current_state_relation_metadata_forwarded": True,
        "current_timestep_forwarded": True,
        "rho_variant_identity": True,
        "canonical_schedule_hash": True,
    }
    decision = internal_mechanism_gate(contract, comparisons)
    conditioning = _conditioning_variants()
    noise = _noise_rows()
    diagnostic = root / "diagnostic"
    diagnostic.mkdir(parents=True)
    raw = {}
    artifact_paths = {}
    for variant in VARIANTS:
        relation_windows = _relation_windows(variant)
        relation_aggregate = internal_runner.aggregate_relation_windows(
            relation_windows
        )
        aggregate = {
            "semantic_and_geometry": {"variant": variant},
            "kinematics": {"variant": variant},
            "penetration": {"variant": variant},
            "diffusion_reliability": relation_aggregate,
        }
        value = {
            "schema_version": 1,
            "run_id": run_id,
            "training_run_id": internal_runner.FORMAL_TRAINING_RUN_ID,
            "target_checkpoint_sha256": (
                internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256
            ),
            "variant": variant,
            "rho_mode": "unit" if variant == "unit_rho" else "canonical",
            "history_max_abs": 0.0,
            "aggregate": aggregate,
            "per_sequence": records[variant],
            "noise_streams": noise,
            "conditioning_streams": conditioning[variant],
            "relation_windows": relation_windows,
            "finite": True,
            "all_fields_reported": True,
        }
        path = diagnostic / f"{variant}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        raw[variant] = value
        artifact_paths[variant] = path
    support = {
        "paired_noise": {
            "schema_version": 1,
            "run_id": run_id,
            "shared": True,
            "variants": {variant: noise for variant in VARIANTS},
        },
        "paired_conditioning": {
            "schema_version": 1,
            "run_id": run_id,
            "shared_exogenous": True,
            "shared_first_window_model_inputs": True,
            "first_window_model_input_keys": sorted(
                internal_runner.FIRST_WINDOW_MODEL_INPUT_KEYS
            ),
            "later_model_inputs": "path-local after causal rollout divergence",
            "path_local_provenance_exact": True,
            "variants": conditioning,
        },
        "causal_window_overlap": _causal_overlap(run_id),
        "diffusion_reliability_appendix": {
            "schema_version": 1,
            "run_id": run_id,
            "selection_use": False,
            "schedule": (
                internal_runner.diffusion_schedule_contract_metadata()
            ),
            "temporal_anchors": [0, 5, 10, 15],
            "roles": ["left_hand", "right_hand", "pelvis"],
            "variants": {
                variant: {
                    "aggregate": raw[variant]["aggregate"][
                        "diffusion_reliability"
                    ],
                    "per_window": [
                        {
                            key: item
                            for key, item in window.items()
                            if key != "by_timestep"
                        }
                        for window in raw[variant]["relation_windows"]
                    ],
                }
                for variant in VARIANTS
            },
        },
    }
    for artifact_id, value in support.items():
        path = diagnostic / native_runner.INTERNAL_ARTIFACT_FILENAMES[
            artifact_id
        ]
        path.write_text(json.dumps(value), encoding="utf-8")
        artifact_paths[artifact_id] = path
    closure_records = {
        artifact_id: {
            "artifact_id": artifact_id,
            "relative_path": str(path.relative_to(root)),
            "run_id": run_id,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size,
        }
        for artifact_id, path in artifact_paths.items()
    }
    formal = _formal_summary_fixture()
    summary = {
        "schema_version": 1,
        "status": "completed",
        "run_id": run_id,
        "selection": {
            "sha256": native_runner.INTERNAL_SELECTION_SHA256,
            "sequences": 64,
            "windows": 192,
        },
        "target_checkpoint": {
            "sha256": internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256,
            "run_id": internal_runner.FORMAL_TRAINING_RUN_ID,
        },
        "formal_lineage": copy.deepcopy(formal),
        "variants": {
            variant: {
                "artifact": {
                    "path": str(artifact_paths[variant]),
                    "sha256": closure_records[variant]["sha256"],
                    "bytes": closure_records[variant]["bytes"],
                },
                "aggregate": raw[variant]["aggregate"],
                "rho_mode": raw[variant]["rho_mode"],
                "history_max_abs": raw[variant]["history_max_abs"],
                "finite": True,
                "all_fields_reported": True,
            }
            for variant in VARIANTS
        },
        "comparisons": comparisons,
        "contract": contract,
        "decision": decision,
        "decision_evidence": internal_runner.internal_decision_evidence(
            decision
        ),
        "internal_status": decision["internal_status"],
        "artifact_closure": {
            "schema_version": 1,
            "run_id": run_id,
            "training_run_id": internal_runner.FORMAL_TRAINING_RUN_ID,
            "target_checkpoint_sha256": (
                internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256
            ),
            "artifacts": closure_records,
        },
        "paired_noise": {
            "sha256": closure_records["paired_noise"]["sha256"],
        },
        "paired_conditioning": {
            "sha256": closure_records["paired_conditioning"]["sha256"],
        },
        "causal_window_overlap": {
            "sha256": closure_records["causal_window_overlap"]["sha256"],
            "all_exact": True,
        },
        "diffusion_reliability_appendix": {
            "sha256": closure_records[
                "diffusion_reliability_appendix"
            ]["sha256"],
        },
        "native_evaluation_authorized": True,
    }
    metrics = root / "metrics.json"
    metrics.write_text(json.dumps(summary), encoding="utf-8")
    args = argparse.Namespace(
        internal_diagnostic=metrics,
        internal_diagnostic_sha256=hashlib.sha256(
            metrics.read_bytes()
        ).hexdigest(),
        target_sha256=internal_runner.FORMAL_FINAL_CHECKPOINT_SHA256,
        training_run_id=internal_runner.FORMAL_TRAINING_RUN_ID,
    )
    return args, formal, summary, artifact_paths


def _refresh_internal_fixture(args, summary, artifact_paths):
    payloads = {
        artifact_id: json.loads(path.read_text(encoding="utf-8"))
        for artifact_id, path in artifact_paths.items()
    }
    for variant in VARIANTS:
        raw = payloads[variant]
        summary["variants"][variant].update({
            "aggregate": raw["aggregate"],
            "rho_mode": raw["rho_mode"],
            "history_max_abs": raw["history_max_abs"],
            "finite": raw["finite"],
            "all_fields_reported": raw["all_fields_reported"],
        })
    records = {
        variant: payloads[variant]["per_sequence"] for variant in VARIANTS
    }
    summary["comparisons"] = paired_comparisons(records)
    summary["decision"] = internal_mechanism_gate(
        summary["contract"], summary["comparisons"],
    )
    summary["decision_evidence"] = internal_runner.internal_decision_evidence(
        summary["decision"]
    )
    summary["internal_status"] = summary["decision"]["internal_status"]
    for artifact_id, path in artifact_paths.items():
        path.write_text(
            json.dumps(payloads[artifact_id]), encoding="utf-8",
        )
        record = summary["artifact_closure"]["artifacts"][artifact_id]
        record["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        record["bytes"] = path.stat().st_size
        if artifact_id in VARIANTS:
            summary["variants"][artifact_id]["artifact"].update({
                "sha256": record["sha256"],
                "bytes": record["bytes"],
            })
        else:
            summary[artifact_id]["sha256"] = record["sha256"]
    args.internal_diagnostic.write_text(
        json.dumps(summary), encoding="utf-8"
    )
    args.internal_diagnostic_sha256 = hashlib.sha256(
        args.internal_diagnostic.read_bytes()
    ).hexdigest()


class D2AFDiagnosticGateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.comparisons = paired_comparisons(_internal_records())

    def test_five_paths_and_all_seven_internal_gates_pass(self):
        self.assertEqual(VARIANTS, (
            "full_rho",
            "unit_rho",
            "relation_gate_ablated",
            "temporal_correspondence_permuted",
            "left_right_role_swapped",
        ))
        self.assertEqual(set(self.comparisons), {
            f"full_rho_vs_{variant}" for variant in VARIANTS[1:]
        })
        decision = internal_mechanism_gate(
            {"all_contracts": True},
            self.comparisons,
        )
        self.assertEqual(decision["internal_status"], "passed")
        self.assertTrue(decision["schedule_reliability_passed"])
        self.assertTrue(decision["relation_path_used"])
        self.assertTrue(decision["temporal_routing_passed"])
        self.assertTrue(decision["role_binding_passed"])
        self.assertTrue(decision["mechanism_passed"])
        self.assertEqual(len(decision["checks"]), 7)
        self.assertTrue(decision["native_evaluation_authorized"])

    def test_schedule_failure_is_distinct_and_still_continues_native(self):
        comparisons = dict(self.comparisons)
        unit = dict(comparisons["full_rho_vs_unit_rho"])
        unit["full_rho_minus_other_direct_union_5cm_f1"] = _difference(
            -0.01, 0.01,
        )
        comparisons["full_rho_vs_unit_rho"] = unit
        decision = internal_mechanism_gate({"contract": True}, comparisons)
        self.assertEqual(decision["internal_status"], "schedule-negative")
        self.assertFalse(decision["schedule_reliability_passed"])
        self.assertFalse(decision["mechanism_passed"])
        self.assertTrue(decision["native_evaluation_authorized"])

    def test_native_classification_precedence_and_mechanism_unverified(self):
        internal, comparison, target, baseline = _native_inputs()
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-positive-candidate-stop",
        )
        self.assertTrue(decision["d2ae_repair_passed"])
        self.assertTrue(decision["d2x_candidate_transfer_passed"])
        self.assertTrue(decision["checkpoint_selected"])

        internal = dict(internal)
        internal["mechanism_passed"] = False
        internal["internal_status"] = "schedule-negative"
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-native-positive-mechanism-unverified-stop",
        )
        self.assertFalse(decision["checkpoint_selected"])

        comparison = dict(comparison)
        repair = dict(comparison["target_vs_sealed_d2ae_repair"])
        repair["target_minus_control_contact_f1"] = _difference(-0.01, 0.01)
        comparison["target_vs_sealed_d2ae_repair"] = repair
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-ae-repair-negative-stop",
        )

    def test_repair_ratio_threshold_is_strict(self):
        internal, comparison, target, baseline = _native_inputs()
        repair = comparison["target_vs_sealed_d2ae_repair"]
        repair["target_over_control_protection"]["end_obj_trans_err"] = _ratio(1.0)
        decision = native_gate(
            contract_passed=True,
            internal=internal,
            comparison=comparison,
            target_metrics=target,
            baseline_ratios=baseline,
        )
        self.assertFalse(
            decision["d2ae_repair_checks"][
                "af_over_ae_end_object_ratio_ci_upper_lt_1.0"
            ]
        )
        self.assertEqual(
            decision["classification"],
            "diffusion-reliability-ae-repair-negative-stop",
        )


class _SnapshotModule:
    def __init__(self, rho: torch.Tensor):
        self.value = {
            "pooled_block_norm": torch.ones(4, 3),
            "pooled_block_variance": torch.full((4, 3), 0.5),
            "relation_norm": torch.arange(1, 5, dtype=torch.float32),
            "temporal_permutation_sensitivity": torch.ones(4),
            "role_swap_sensitivity": torch.ones(4),
            "gate": torch.tensor([0.1]),
            "rho": rho,
            "raw_writeback_norm": torch.ones(16),
            "attenuated_writeback_norm": torch.ones(16) * rho.mean(),
        }

    def snapshot(self):
        return self.value


class D2AFRelationCaptureTests(unittest.TestCase):
    def _capture(self, unit: bool):
        capture = internal_runner.RelationCapture()
        motion = torch.zeros(1, 16, 512)
        schedule = canonical_diffusion_schedule()["sqrt_alpha_bar"]
        for timestep in reversed(range(500)):
            rho = torch.ones(1) if unit else schedule[timestep].reshape(1)
            module = _SnapshotModule(rho)
            output = motion + rho[:, None, None] * 0.01
            capture.hook(module, (motion,), output)
        return capture.result()

    def test_canonical_trace_is_exact_and_separates_raw_attenuated(self):
        result = self._capture(unit=False)
        self.assertEqual(result["rho_mode"], "canonical")
        self.assertLessEqual(result["rho_canonical_max_abs"], 1.0e-7)
        self.assertEqual(result["sqrt_alpha_bar_sha256"], SQRT_ALPHA_BAR_SHA256)
        self.assertEqual(
            result["by_timestep"]["timesteps"],
            list(reversed(range(500))),
        )
        for timestep, expected in SQRT_ALPHA_BAR_SENTINELS.items():
            self.assertEqual(result["rho_sentinels"][str(timestep)], expected)
        self.assertIn(
            "raw_writeback_variance_by_anchor",
            result["by_timestep"]["values"],
        )
        self.assertIn(
            "attenuated_writeback_variance_by_anchor",
            result["by_timestep"]["values"],
        )

    def test_unit_rho_trace_is_independent_counterfactual(self):
        result = self._capture(unit=True)
        self.assertEqual(result["rho_mode"], "unit")
        self.assertEqual(result["rho_unit_max_abs"], 0.0)
        self.assertTrue(all(
            value == 1.0 for value in result["rho_sentinels"].values()
        ))


class D2AFEvaluationRunnerContractTests(unittest.TestCase):
    def test_run_id_architecture_and_resolved_internal_identity(self):
        actual_date = datetime.now().astimezone().strftime("%Y%m%d")
        internal_id = (
            "p1-hoi-d2af-sqrt-alpha-bar-reliability-internal-"
            f"s42-{actual_date}"
        )
        native_id = f"p1-hoi-d2af-native-eval-s42-{actual_date}"
        training_id = (
            f"p1-hoi-d2af-sqrt-alpha-bar-reliability-s42-{actual_date}"
        )
        self.assertIsNotNone(internal_runner.RUN_ID_RE.fullmatch(internal_id))
        self.assertIsNotNone(native_runner.RUN_ID_RE.fullmatch(native_id))
        self.assertIsNotNone(
            native_runner.TRAINING_RUN_ID_RE.fullmatch(training_id)
        )
        args = argparse.Namespace(
            run_id=internal_id,
            target_checkpoint=ROOT / "final.pth",
            target_sha256="a" * 64,
            training_run_id=training_id,
            formal_manifest=ROOT / "formal/manifest.json",
            formal_manifest_sha256="b" * 64,
            training_metrics=ROOT / "formal/metrics.json",
            training_metrics_sha256="c" * 64,
            training_state=ROOT / "formal/training_state.json",
            training_state_sha256="d" * 64,
            resume_contract=ROOT / "formal/resume_contract.json",
            resume_contract_sha256="e" * 64,
            output_dir=ROOT / "results/d2af-internal",
            metrics=ROOT / "results/d2af-internal.json",
            resolved_config=ROOT / "results/d2af-internal-resolved.json",
            batch_size=8,
            device="cuda:0",
        )
        resolved = internal_runner.resolved_config(args)
        self.assertEqual(resolved["subphase"], "1B-D2-AF0-internal")
        self.assertEqual(resolved["variants"], list(VARIANTS))
        self.assertEqual(
            resolved["target_checkpoint"]["architecture_variant"],
            HOI_ARCHITECTURE_D2AF,
        )
        self.assertEqual(
            resolved["assets"]["sqrt_alpha_bar_sha256"],
            SQRT_ALPHA_BAR_SHA256,
        )
        self.assertEqual(
            resolved["formal_lineage"]["execution_target_commit"],
            internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
        )

    def test_resumed_formal_payload_contract_and_batch_size_are_fixed(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = _formal_cadence_fixture(Path(directory))
            with mock.patch.object(
                internal_runner,
                "FORMAL_FINAL_CHECKPOINT_SHA256",
                fixture["final_sha256"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_RESUME_CHECKPOINT_SHA256",
                fixture["resume_sha256"],
            ):
                manifest, metrics, state, resume, checkpoint = _formal_payloads(
                    final_sha256=fixture["final_sha256"],
                    resume_sha256=fixture["resume_sha256"],
                    resume_checkpoint_bytes=fixture[
                        "resume_checkpoint_bytes"
                    ],
                    resume_rng_sidecars=fixture["resume_rng_sidecars"],
                )
                cadence = internal_runner.validate_formal_completion_verification(
                    payload=fixture["completion"],
                    checkpoint_directory=fixture["checkpoint_dir"],
                    resume_contract=resume,
                    checkpoint_inspector=lambda *args, **kwargs: {
                        "checks": {"fixture": True}
                    },
                    rng_inspector=lambda path: {
                        "checks": {"fixture": True}
                    },
                )
                result = internal_runner.validate_formal_lineage_payloads(
                    training_run_id=internal_runner.FORMAL_TRAINING_RUN_ID,
                    target_checkpoint=Path(
                        f"{internal_runner.FORMAL_TRAINING_RUN_ID}_"
                        "windows061440000.pth"
                    ),
                    target_sha256=fixture["final_sha256"],
                    manifest=manifest,
                    metrics=metrics,
                    training_state=state,
                    resume_contract=resume,
                    checkpoint=checkpoint,
                    completion_verification=fixture["completion"],
                    cadence_contract=cadence,
                )
                self.assertTrue(all(result["checks"].values()))
                self.assertEqual(len(
                    result["cadence_contract"]["artifacts"]
                ), 100)

                for field, bad_value in (
                    ("current_git_commit", "0" * 40),
                    ("diff_sha256", "0" * 64),
                ):
                    bad_state = copy.deepcopy(state)
                    bad_state["resume_commit_provenance"][field] = bad_value
                    with self.subTest(field=field):
                        with self.assertRaisesRegex(
                            ValueError, "transition_provenance_exact"
                        ):
                            internal_runner.validate_formal_lineage_payloads(
                                training_run_id=(
                                    internal_runner.FORMAL_TRAINING_RUN_ID
                                ),
                                target_checkpoint=Path(
                                    f"{internal_runner.FORMAL_TRAINING_RUN_ID}_"
                                    "windows061440000.pth"
                                ),
                                target_sha256=fixture["final_sha256"],
                                manifest=manifest,
                                metrics=metrics,
                                training_state=bad_state,
                                resume_contract=resume,
                                checkpoint=checkpoint,
                                completion_verification=fixture["completion"],
                                cadence_contract=cadence,
                            )
        internal_runner.validate_internal_batch_size(8)
        for invalid in (1, 4, 16, 64):
            with self.subTest(batch_size=invalid):
                with self.assertRaisesRegex(ValueError, "exactly 8"):
                    internal_runner.validate_internal_batch_size(invalid)

    def test_formal_artifact_contract_hash_binds_recovered_jsons(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = _formal_cadence_fixture(root)
            checkpoint_dir = fixture["checkpoint_dir"]
            target = checkpoint_dir / (
                f"{internal_runner.FORMAL_TRAINING_RUN_ID}_"
                "windows061440000.pth"
            )
            _, metrics, state, resume, checkpoint = _formal_payloads(
                final_sha256=fixture["final_sha256"],
                resume_sha256=fixture["resume_sha256"],
                resume_checkpoint_bytes=fixture["resume_checkpoint_bytes"],
                resume_rng_sidecars=fixture["resume_rng_sidecars"],
            )
            metrics_path = root / "metrics.json"
            metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
            metrics_sha = hashlib.sha256(metrics_path.read_bytes()).hexdigest()
            manifest, _, _, _, _ = _formal_payloads(
                metrics_sha,
                final_sha256=fixture["final_sha256"],
                resume_sha256=fixture["resume_sha256"],
                resume_checkpoint_bytes=fixture["resume_checkpoint_bytes"],
                resume_rng_sidecars=fixture["resume_rng_sidecars"],
            )
            paths = {
                "formal_manifest": root / "manifest.json",
                "training_metrics": metrics_path,
                "training_state": root / "training_state.json",
                "resume_contract": root / "resume_contract.json",
                "completion_verification": (
                    root / "formal_completion_verification.json"
                ),
            }
            for path, value in (
                (paths["formal_manifest"], manifest),
                (paths["training_state"], state),
                (paths["resume_contract"], resume),
                (
                    paths["completion_verification"],
                    fixture["completion"],
                ),
            ):
                path.write_text(json.dumps(value), encoding="utf-8")
            hashes = {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in paths.items()
            }
            args = argparse.Namespace(
                training_run_id=internal_runner.FORMAL_TRAINING_RUN_ID,
                target_checkpoint=target,
                target_sha256=fixture["final_sha256"],
                formal_manifest=paths["formal_manifest"],
                formal_manifest_sha256=hashes["formal_manifest"],
                training_metrics=paths["training_metrics"],
                training_metrics_sha256=hashes["training_metrics"],
                training_state=paths["training_state"],
                training_state_sha256=hashes["training_state"],
                resume_contract=paths["resume_contract"],
                resume_contract_sha256=hashes["resume_contract"],
            )
            with mock.patch.object(
                internal_runner,
                "FORMAL_FINAL_CHECKPOINT_SHA256",
                fixture["final_sha256"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_RESUME_CHECKPOINT_SHA256",
                fixture["resume_sha256"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_MANIFEST_SHA256",
                hashes["formal_manifest"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_METRICS_SHA256",
                hashes["training_metrics"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_TRAINING_STATE_SHA256",
                hashes["training_state"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_RESUME_CONTRACT_SHA256",
                hashes["resume_contract"],
            ), mock.patch.object(
                internal_runner,
                "FORMAL_COMPLETION_VERIFICATION_SHA256",
                hashes["completion_verification"],
            ), mock.patch.object(
                internal_runner,
                "checkpoint_contract",
                return_value=checkpoint,
            ), mock.patch.object(
                internal_runner,
                "inspect_formal_cadence_checkpoint",
                side_effect=lambda *args, **kwargs: {
                    "checks": {"fixture": True}
                },
            ), mock.patch.object(
                internal_runner,
                "inspect_formal_rng_sidecar",
                side_effect=lambda path: {"checks": {"fixture": True}},
            ):
                result = internal_runner.formal_artifact_contract(args)
                self.assertTrue(all(result["checks"].values()))
                self.assertEqual(
                    result["artifacts"]["completion_verification"]["sha256"],
                    hashes["completion_verification"],
                )
                self.assertEqual(
                    len(result["cadence_contract"]["artifacts"]), 100
                )
                paths["training_state"].write_text(
                    json.dumps({**state, "optimizer_updates": 29_999}),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(
                    ValueError, "training_state SHA-256 mismatch"
                ):
                    internal_runner.formal_artifact_contract(args)

    def test_formal_completion_rejects_incomplete_corrupt_and_symlink_cadence(self):
        for case in ("99-empty", "corrupt", "symlink"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = _formal_cadence_fixture(root)
                _, _, _, resume, _ = _formal_payloads(
                    final_sha256=fixture["final_sha256"],
                    resume_sha256=fixture["resume_sha256"],
                    resume_checkpoint_bytes=fixture[
                        "resume_checkpoint_bytes"
                    ],
                    resume_rng_sidecars=fixture["resume_rng_sidecars"],
                )
                names = internal_runner.expected_formal_cadence_names(
                    internal_runner.FORMAL_TRAINING_RUN_ID
                )
                if case == "99-empty":
                    (fixture["checkpoint_dir"] / names[-1]).unlink()
                    for name in names[:-1]:
                        (fixture["checkpoint_dir"] / name).write_bytes(b"")
                elif case == "corrupt":
                    path = fixture["checkpoint_dir"] / names[0]
                    path.write_bytes(path.read_bytes() + b"corrupt")
                else:
                    path = fixture["checkpoint_dir"] / names[0]
                    path.unlink()
                    path.symlink_to(fixture["checkpoint_dir"] / names[1])
                with mock.patch.object(
                    internal_runner,
                    "FORMAL_FINAL_CHECKPOINT_SHA256",
                    fixture["final_sha256"],
                ), self.assertRaisesRegex(
                    ValueError, "formal completion verification"
                ):
                    internal_runner.validate_formal_completion_verification(
                        payload=fixture["completion"],
                        checkpoint_directory=fixture["checkpoint_dir"],
                        resume_contract=resume,
                        checkpoint_inspector=lambda *args, **kwargs: {
                            "checks": {"fixture": True}
                        },
                        rng_inspector=lambda path: {
                            "checks": {"fixture": True}
                        },
                    )

    def test_formal_cadence_checkpoint_and_rng_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed_windows = 9_216_000
            optimizer_updates = 4_500
            stem = (
                f"{internal_runner.FORMAL_TRAINING_RUN_ID}_windows"
                f"{processed_windows:09d}"
            )
            checkpoint_path = root / f"{stem}.pth"
            checkpoint = {
                "schema_version": 2,
                "checkpoint_type": "hoi_prior_phase1b",
                "run_id": internal_runner.FORMAL_TRAINING_RUN_ID,
                "seed": 42,
                "processed_windows": processed_windows,
                "processed_frames": processed_windows * 16,
                "optimizer_updates": optimizer_updates,
                "architecture_variant": HOI_ARCHITECTURE_D2AF,
                "model_config": {
                    "architecture_variant": HOI_ARCHITECTURE_D2AF,
                },
                "rng_pattern": f"{stem}.rank{{rank}}.rng.pth",
                "git_commit": internal_runner.FORMAL_EXECUTION_TARGET_COMMIT,
            }
            torch.save(checkpoint, checkpoint_path)
            inspected = internal_runner.inspect_formal_cadence_checkpoint(
                checkpoint_path,
                processed_windows=processed_windows,
                optimizer_updates=optimizer_updates,
            )
            self.assertTrue(all(inspected["checks"].values()))
            checkpoint["git_commit"] = "0" * 40
            torch.save(checkpoint, checkpoint_path)
            with self.assertRaisesRegex(ValueError, "git_commit"):
                internal_runner.inspect_formal_cadence_checkpoint(
                    checkpoint_path,
                    processed_windows=processed_windows,
                    optimizer_updates=optimizer_updates,
                )

            rng_path = root / f"{stem}.rank0.rng.pth"
            torch.save({
                "cuda": torch.ones(8, dtype=torch.uint8),
                "numpy": ("MT19937",),
                "python": (3,),
                "torch": torch.ones(8, dtype=torch.uint8),
            }, rng_path)
            rng = internal_runner.inspect_formal_rng_sidecar(rng_path)
            self.assertTrue(all(rng["checks"].values()))
            torch.save({"torch": torch.ones(1)}, rng_path)
            with self.assertRaisesRegex(ValueError, "exact_keys"):
                internal_runner.inspect_formal_rng_sidecar(rng_path)

    def test_first_window_full_model_input_identity_is_strict(self):
        conditioning = _conditioning_variants()
        self.assertTrue(
            internal_runner.first_window_model_input_identity(conditioning)
        )
        later = copy.deepcopy(conditioning)
        later["unit_rho"][1]["path_local_model_inputs"]["sha256"][
            "global_bps"
        ] = "f" * 64
        self.assertTrue(
            internal_runner.first_window_model_input_identity(later)
        )
        mismatched = copy.deepcopy(conditioning)
        mismatched["unit_rho"][0]["path_local_model_inputs"]["sha256"][
            "global_bps"
        ] = "f" * 64
        self.assertFalse(
            internal_runner.first_window_model_input_identity(mismatched)
        )
        missing = copy.deepcopy(conditioning)
        missing["unit_rho"] = [
            row for row in missing["unit_rho"]
            if row["window_index"] != 2
        ]
        self.assertFalse(
            internal_runner.first_window_model_input_identity(missing)
        )
        reshaped = copy.deepcopy(conditioning)
        reshaped["unit_rho"][0]["path_local_model_inputs"]["shapes"][
            "global_bps"
        ] = [8, 512, 3]
        self.assertFalse(
            internal_runner.first_window_model_input_identity(reshaped)
        )

    def test_native_recomputes_valid_internal_and_allows_mechanism_negative(self):
        with tempfile.TemporaryDirectory() as directory:
            args, formal, _, _ = _write_internal_fixture(
                Path(directory), mechanism_negative=True,
            )
            validated = native_runner._validate_internal(args, formal)
        self.assertEqual(validated["internal_status"], "schedule-negative")
        self.assertFalse(validated["mechanism_passed"])
        self.assertTrue(validated["contract_passed"])
        self.assertTrue(
            validated["decision"]["native_evaluation_authorized"]
        )

    def test_native_rejects_missing_tampered_hash_and_gate_artifacts(self):
        for case in ("missing", "tampered", "closure-hash", "gate"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                args, formal, summary, paths = _write_internal_fixture(
                    Path(directory)
                )
                if case == "missing":
                    paths["full_rho"].unlink()
                elif case == "tampered":
                    paths["unit_rho"].write_text(
                        paths["unit_rho"].read_text(encoding="utf-8") + "\n",
                        encoding="utf-8",
                    )
                elif case == "closure-hash":
                    summary["artifact_closure"]["artifacts"][
                        "paired_noise"
                    ]["sha256"] = "0" * 64
                    args.internal_diagnostic.write_text(
                        json.dumps(summary), encoding="utf-8"
                    )
                    args.internal_diagnostic_sha256 = hashlib.sha256(
                        args.internal_diagnostic.read_bytes()
                    ).hexdigest()
                else:
                    check = next(iter(summary["decision"]["checks"]))
                    summary["decision"]["checks"][check] = not summary[
                        "decision"
                    ]["checks"][check]
                    args.internal_diagnostic.write_text(
                        json.dumps(summary), encoding="utf-8"
                    )
                    args.internal_diagnostic_sha256 = hashlib.sha256(
                        args.internal_diagnostic.read_bytes()
                    ).hexdigest()
                with self.assertRaisesRegex(
                    ValueError, "internal diagnostic contract mismatch"
                ):
                    native_runner._validate_internal(args, formal)

    def test_native_rejects_self_consistent_raw_protocol_forgery(self):
        small_trace_shapes = {
            key: (
                [500, 1]
                if key == "rho"
                else [1, 4, 3]
                if key in {"pooled_block_norm", "pooled_block_variance"}
                else [1, 1]
                if key == "gate"
                else [1, 4]
            )
            for key in native_runner.RELATION_TRACE_SHAPES
        }
        small_mean_shapes = {
            key: shape[1:] for key, shape in small_trace_shapes.items()
        }
        cases = (
            "different-noise",
            "future-gt-conditioning",
            "missing-window-two",
            "forged-cohort",
            "empty-relation",
            "zero-schedule",
            "nonfinite-relation-scalar",
            "nonfinite-relation-sentinel",
            "nonfinite-summary",
        )
        for case in cases:
            with self.subTest(case=case):
                with tempfile.TemporaryDirectory() as directory, mock.patch.object(
                    native_runner, "RELATION_TRACE_SHAPES", small_trace_shapes,
                ), mock.patch.object(
                    native_runner, "RELATION_MEAN_SHAPES", small_mean_shapes,
                ):
                    args, formal, summary, paths = _write_internal_fixture(
                        Path(directory)
                    )

                    def load(artifact_id):
                        return json.loads(
                            paths[artifact_id].read_text(encoding="utf-8")
                        )

                    def save(artifact_id, payload):
                        paths[artifact_id].write_text(
                            json.dumps(payload), encoding="utf-8"
                        )

                    if case == "different-noise":
                        raw = load("unit_rho")
                        raw["noise_streams"][0][
                            "generator_final_state_sha256"
                        ] = "f" * 64
                        save("unit_rho", raw)
                        support = load("paired_noise")
                        support["variants"]["unit_rho"] = raw[
                            "noise_streams"
                        ]
                        save("paired_noise", support)
                    elif case == "future-gt-conditioning":
                        raw = load("unit_rho")
                        row = raw["conditioning_streams"][0]
                        row["exogenous"]["sha256"]["text"] = "f" * 64
                        row["path_local_provenance"][
                            "future_gt_source"
                        ] = "future_gt"
                        save("unit_rho", raw)
                        support = load("paired_conditioning")
                        support["variants"]["unit_rho"] = raw[
                            "conditioning_streams"
                        ]
                        save("paired_conditioning", support)
                    elif case == "missing-window-two":
                        support = load("paired_conditioning")
                        for variant in VARIANTS:
                            raw = load(variant)
                            raw["conditioning_streams"] = [
                                row for row in raw["conditioning_streams"]
                                if row["window_index"] != 2
                            ]
                            save(variant, raw)
                            support["variants"][variant] = raw[
                                "conditioning_streams"
                            ]
                        save("paired_conditioning", support)
                    elif case == "forged-cohort":
                        forged = [
                            f"forged-{index:02d}" for index in range(64)
                        ]
                        for variant in VARIANTS:
                            raw = load(variant)
                            for record, name in zip(
                                raw["per_sequence"], forged,
                            ):
                                record["sequence"] = name
                            save(variant, raw)
                        overlap = load("causal_window_overlap")
                        for row, name in zip(overlap["rows"], forged):
                            row["sequence"] = name
                        save("causal_window_overlap", overlap)
                    elif case == "empty-relation":
                        appendix = load("diffusion_reliability_appendix")
                        for variant in VARIANTS:
                            raw = load(variant)
                            raw["relation_windows"] = []
                            raw["aggregate"]["diffusion_reliability"] = {}
                            save(variant, raw)
                            appendix["variants"][variant] = {
                                "aggregate": {},
                                "per_window": [],
                            }
                        save("diffusion_reliability_appendix", appendix)
                    elif case == "zero-schedule":
                        raw = load("full_rho")
                        for window in raw["relation_windows"]:
                            window["by_timestep"]["values"]["rho"] = [
                                [0.0] for _ in range(500)
                            ]
                            window["by_timestep_sha256"] = (
                                internal_runner.sha256_json(
                                    window["by_timestep"]
                                )
                            )
                            window["rho_sentinels"] = {
                                str(timestep): 0.0
                                for timestep in SQRT_ALPHA_BAR_SENTINELS
                            }
                            window["rho_canonical_max_abs"] = 0.0
                        aggregate = (
                            internal_runner.aggregate_relation_windows(
                                raw["relation_windows"]
                            )
                        )
                        raw["aggregate"][
                            "diffusion_reliability"
                        ] = aggregate
                        save("full_rho", raw)
                        appendix = load("diffusion_reliability_appendix")
                        appendix["variants"]["full_rho"] = {
                            "aggregate": aggregate,
                            "per_window": [
                                {
                                    key: item
                                    for key, item in window.items()
                                    if key != "by_timestep"
                                }
                                for window in raw["relation_windows"]
                            ],
                        }
                        save("diffusion_reliability_appendix", appendix)
                    elif case in {
                        "nonfinite-relation-scalar",
                        "nonfinite-relation-sentinel",
                    }:
                        raw = load("full_rho")
                        appendix = load("diffusion_reliability_appendix")
                        for window, appendix_window in zip(
                            raw["relation_windows"],
                            appendix["variants"]["full_rho"]["per_window"],
                        ):
                            if case == "nonfinite-relation-scalar":
                                window["rho_canonical_max_abs"] = float("nan")
                                appendix_window[
                                    "rho_canonical_max_abs"
                                ] = float("nan")
                            else:
                                timestep = next(iter(
                                    SQRT_ALPHA_BAR_SENTINELS
                                ))
                                window["rho_sentinels"][
                                    str(timestep)
                                ] = float("nan")
                                appendix_window["rho_sentinels"][
                                    str(timestep)
                                ] = float("nan")
                        save("full_rho", raw)
                        save("diffusion_reliability_appendix", appendix)
                    else:
                        raw = load("full_rho")
                        raw["finite"] = False
                        raw["all_fields_reported"] = False
                        raw["history_max_abs"] = 999.0
                        save("full_rho", raw)

                    _refresh_internal_fixture(
                        args, summary, paths,
                    )
                    with self.assertRaisesRegex(
                        ValueError, "internal diagnostic contract mismatch"
                    ):
                        native_runner._validate_internal(args, formal)

    def test_native_binds_formal_target_and_transition(self):
        with tempfile.TemporaryDirectory() as directory:
            args, formal, _, _ = _write_internal_fixture(Path(directory))
            validated = native_runner._validate_internal(args, formal)
            self.assertTrue(all(validated["checks"].values()))
            for key, bad_value in (
                ("execution_target_commit", "0" * 40),
                ("execution_diff_sha256", "0" * 64),
                ("final_checkpoint_sha256", "0" * 64),
            ):
                bad_formal = copy.deepcopy(formal)
                bad_formal[key] = bad_value
                with self.subTest(key=key):
                    with self.assertRaisesRegex(
                        ValueError, "formal_lineage"
                    ):
                        native_runner._validate_internal(args, bad_formal)

    def test_training_and_checkpoint_contracts_are_d2af_specific(self):
        internal_source = inspect.getsource(internal_runner.checkpoint_contract)
        native_source = inspect.getsource(native_runner.validate_training_result)
        for source in (internal_source, native_source):
            self.assertIn("HOI_ARCHITECTURE_D2AF", source)
            self.assertIn("diffusion_reliability", source)
        self.assertNotIn("HOI_ARCHITECTURE_D2AE", internal_source)
        self.assertNotIn("HOI_ARCHITECTURE_D2AE", native_source)
        self.assertEqual(
            native_runner.SEALED_D2AE_AGGREGATE_SHA256,
            "157acda463036bdf787618c217262c14c77a09a3f409cbeada03de06e9b902a1",
        )
        self.assertEqual(
            native_runner.SEALED_D2AE_PER_SEQUENCE_SHA256,
            "8533b66ea3c1fb0928b8a7581bb79c0cc14d594970314a3b7619659daddfb95c",
        )


if __name__ == "__main__":
    unittest.main()
