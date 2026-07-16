#!/usr/bin/env python3
"""Evaluate the preregistered Phase 1B D2-M0 paired smoke checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import numpy as np
import torch


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress, prepare_clean_x0  # noqa: E402
from priors.exposure import (  # noqa: E402
    CONDITION_VARIANTS,
    deterministic_condition_variants,
    fieldwise_mse_per_sample,
)
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.optimizer_reset import (  # noqa: E402
    CANDIDATES,
    NATIVE_SELECTION_SHA256,
    SOURCE_CHECKPOINT_SHA256,
    TEACHER_SELECTION_SHA256,
    TEACHER_TIMESTEPS,
    compact_statistic,
    paired_difference,
    paired_mean_ratio,
    select_native_holdout,
    select_teacher_holdout,
    stable_seed,
)
from priors.representation import REPRESENTATION  # noqa: E402
from priors.window_codec import WindowFrame, rotation_geodesic  # noqa: E402
from tools.evaluate_hoi_remediation import (  # noqa: E402
    global_goals,
    load_rest_vertices,
    rollout,
    stack_frames,
)


MODELS = ("source", *CANDIDATES)
PHYSICAL_METRICS = (
    "object_goal_error_cm",
    "pelvis_goal_error_cm",
    "mpjpe_cm",
    "object_translation_mae_cm",
    "pelvis_translation_mae_cm",
    "object_rotation_geodesic_deg",
)
NATIVE_PAIRED_METRICS = (
    "object_goal_error_cm",
    "pelvis_goal_error_cm",
    "mpjpe_cm",
    "joint_position_mae_cm",
    "pelvis_translation_mae_cm",
    "object_translation_mae_cm",
    "object_rotation_geodesic_deg",
    "contact_channel_mse",
    "foot_sliding",
    "physical_contact_f1",
    "physical_contact_precision",
    "physical_contact_recall",
)
NATIVE_AGGREGATE_METRICS = (
    "object_goal_error_cm",
    "pelvis_goal_error_cm",
    "mpjpe_cm",
    "pelvis_translation_mae_cm",
    "object_translation_mae_cm",
    "object_rotation_geodesic_deg",
    "foot_sliding",
    "physical_contact_f1",
    "physical_contact_precision",
    "physical_contact_recall",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _append(target: Dict[str, List[float]], values: Mapping[str, torch.Tensor]) -> None:
    for name, value in values.items():
        target[name].extend(value.detach().cpu().double().tolist())


def _teacher_batch(
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
):
    items = [dataset[position] for position in positions]
    keys = ("x", "text_embedding", "object_bps", "goals", "progress")
    batch = {key: torch.stack([item[key] for item in items]).to(device) for key in keys}
    return batch, items, stack_frames(items, device)


@torch.no_grad()
def physical_errors_per_sample(
    dataset: PriorWindowDataset,
    prediction: torch.Tensor,
    target: torch.Tensor,
    items: Sequence[Mapping[str, torch.Tensor]],
    frames: WindowFrame,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    decoded = dataset.codec.decode(prediction, frames)
    truth = dataset.codec.decode(target, frames)
    pelvis_goal, object_goal = global_goals(dataset, items, frames, device)
    active = slice(REPRESENTATION.history_frames, None)
    relative_prediction = decoded["joints"][:, active] - decoded["joints"][:, active, :1]
    relative_truth = truth["joints"][:, active] - truth["joints"][:, active, :1]
    rotation_error = rotation_geodesic(
        decoded["object_rotation"][:, active], truth["object_rotation"][:, active],
    )
    return {
        "object_goal_error_cm": torch.linalg.vector_norm(
            decoded["object_translation"][:, -1] - object_goal, dim=-1,
        ) * 100.0,
        "pelvis_goal_error_cm": torch.linalg.vector_norm(
            decoded["joints"][:, -1, 0, (0, 2)] - pelvis_goal[:, (0, 2)], dim=-1,
        ) * 100.0,
        "mpjpe_cm": torch.linalg.vector_norm(
            relative_prediction - relative_truth, dim=-1,
        ).mean(dim=(1, 2)) * 100.0,
        "object_translation_mae_cm": torch.linalg.vector_norm(
            decoded["object_translation"][:, active] - truth["object_translation"][:, active],
            dim=-1,
        ).mean(dim=1) * 100.0,
        "pelvis_translation_mae_cm": torch.linalg.vector_norm(
            decoded["joints"][:, active, 0] - truth["joints"][:, active, 0], dim=-1,
        ).mean(dim=1) * 100.0,
        "object_rotation_geodesic_deg": rotation_error.mean(dim=1) * (180.0 / math.pi),
    }


def _empty_teacher_values() -> Dict[str, object]:
    return {
        model: {
            variant: {
                "fields": {field.name: [] for field in REPRESENTATION.fields},
                "physical": {name: [] for name in PHYSICAL_METRICS},
            }
            for variant in CONDITION_VARIANTS
        }
        for model in MODELS
    }


def per_sequence_native_error(metrics, key):
    if key in {
        "foot_sliding",
        "physical_contact_f1",
        "physical_contact_precision",
        "physical_contact_recall",
    }:
        return np.asarray([
            float(value[key]) for value in sorted(
                metrics["per_sequence"], key=lambda item: item["sequence"],
            )
        ], dtype=np.float64)
    grouped: Dict[str, List[float]] = {}
    for value in metrics["per_sequence_window"]:
        if key == "object_goal_error_cm" and int(value["window"]) != 3:
            continue
        grouped.setdefault(value["sequence"], []).append(float(value[key]))
    return np.asarray(
        [np.mean(grouped[name]) for name in sorted(grouped)],
        dtype=np.float64,
    )


def _summarize_teacher_values(values: Mapping[str, object]) -> Dict[str, object]:
    result = {}
    for model in MODELS:
        variants = {}
        matched_fields = values[model]["matched"]["fields"]
        matched_physical = values[model]["matched"]["physical"]
        for variant in CONDITION_VARIANTS:
            variant_fields = values[model][variant]["fields"]
            variant_physical = values[model][variant]["physical"]
            record = {
                "fieldwise_mse": {
                    name: {
                        "mean": float(np.mean(field_values)),
                        "per_window": list(field_values),
                    }
                    for name, field_values in variant_fields.items()
                },
                "physical": {
                    name: {
                        "mean": float(np.mean(metric_values)),
                        "per_window": list(metric_values),
                    }
                    for name, metric_values in variant_physical.items()
                },
            }
            if variant != "matched":
                record["permuted_minus_matched"] = {
                    "fields": {
                        name: compact_statistic(paired_difference(
                            variant_fields[name], matched_fields[name],
                        ))
                        for name in variant_fields
                    },
                    "physical": {
                        name: compact_statistic(paired_difference(
                            variant_physical[name], matched_physical[name],
                        ))
                        for name in variant_physical
                    },
                }
            variants[variant] = record
        result[model] = variants
    return result


@torch.no_grad()
def teacher_evaluation(
    models: Mapping[str, torch.nn.Module],
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    positions: Sequence[int],
    device: torch.device,
    batch_size: int,
) -> Dict[str, object]:
    output = {"timesteps": {}, "finite": True, "history_max_abs": 0.0}
    for timestep in TEACHER_TIMESTEPS:
        values = _empty_teacher_values()
        q_noise_hash = hashlib.sha256()
        finite = True
        history_max_abs = 0.0
        for offset in range(0, len(positions), batch_size):
            selected = positions[offset:offset + batch_size]
            batch, items, frames = _teacher_batch(dataset, selected, device)
            clean = batch["x"]
            fixed = clean[:, :REPRESENTATION.history_frames]
            times = torch.full((len(selected),), timestep, dtype=torch.long, device=device)
            generator = torch.Generator(device=device)
            generator.manual_seed(stable_seed(f"D2M:teacher-q:{timestep}:{offset}"))
            noise = torch.randn(clean.shape, device=device, generator=generator)
            q_noise_hash.update(noise.detach().cpu().numpy().tobytes())
            noisy = diffusion.q_sample(clean, times, noise)
            variants, _ = deterministic_condition_variants(batch)
            repeat_count = len(CONDITION_VARIANTS)
            expanded_noisy = noisy.repeat(repeat_count, 1, 1)
            expanded_times = times.repeat(repeat_count)
            text = torch.cat([
                variants[name]["text_embedding"] for name in CONDITION_VARIANTS
            ])
            bps = torch.cat([
                variants[name]["object_bps"] for name in CONDITION_VARIANTS
            ])
            goals = torch.cat([
                variants[name]["goals"] for name in CONDITION_VARIANTS
            ])
            progress = normalize_progress(torch.cat([
                variants[name]["progress"] for name in CONDITION_VARIANTS
            ]))
            for model_name in MODELS:
                prediction = models[model_name](
                    expanded_noisy, expanded_times, text, bps, goals, progress,
                )
                expanded_fixed = fixed.repeat(repeat_count, 1, 1)
                prediction = prepare_clean_x0(
                    prediction, expanded_fixed, object_so3_x0=False,
                )
                finite = bool(finite and torch.isfinite(prediction).all())
                history_max_abs = max(
                    history_max_abs,
                    float(
                        (
                            prediction[:, :REPRESENTATION.history_frames] - expanded_fixed
                        ).abs().max()
                    ),
                )
                width = len(selected)
                for variant_index, variant in enumerate(CONDITION_VARIANTS):
                    rows = slice(variant_index * width, (variant_index + 1) * width)
                    selected_prediction = prediction[rows]
                    _append(
                        values[model_name][variant]["fields"],
                        fieldwise_mse_per_sample(selected_prediction, clean),
                    )
                    _append(
                        values[model_name][variant]["physical"],
                        physical_errors_per_sample(
                            dataset, selected_prediction, clean, items, frames, device,
                        ),
                    )
        summarized = _summarize_teacher_values(values)
        comparisons = {
            "current_minus_balanced": {
                field.name: paired_difference(
                    values["current"]["matched"]["fields"][field.name],
                    values["balanced"]["matched"]["fields"][field.name],
                )
                for field in REPRESENTATION.fields
            },
            "balanced_over_current": {
                field.name: paired_mean_ratio(
                    values["balanced"]["matched"]["fields"][field.name],
                    values["current"]["matched"]["fields"][field.name],
                )
                for field in REPRESENTATION.fields
            },
            "balanced_over_source": {
                field.name: paired_mean_ratio(
                    values["balanced"]["matched"]["fields"][field.name],
                    values["source"]["matched"]["fields"][field.name],
                )
                for field in REPRESENTATION.fields
            },
        }
        physical_comparisons = {
            "current_minus_balanced": {
                name: paired_difference(
                    values["current"]["matched"]["physical"][name],
                    values["balanced"]["matched"]["physical"][name],
                )
                for name in PHYSICAL_METRICS
            },
            "balanced_over_current": {
                name: paired_mean_ratio(
                    values["balanced"]["matched"]["physical"][name],
                    values["current"]["matched"]["physical"][name],
                )
                for name in PHYSICAL_METRICS
            },
            "balanced_over_source": {
                name: paired_mean_ratio(
                    values["balanced"]["matched"]["physical"][name],
                    values["source"]["matched"]["physical"][name],
                )
                for name in PHYSICAL_METRICS
            },
        }
        output["timesteps"][str(timestep)] = {
            "timestep": timestep,
            "q_noise_sha256": q_noise_hash.hexdigest(),
            "finite": finite,
            "history_max_abs": history_max_abs,
            "models": summarized,
            **comparisons,
            "physical_comparisons": physical_comparisons,
        }
        output["finite"] = bool(output["finite"] and finite)
        output["history_max_abs"] = max(float(output["history_max_abs"]), history_max_abs)
    output["all_fields_conditions_and_physical_reported"] = all(
        set(record["models"]) == set(MODELS)
        and all(
            set(record["models"][model]) == set(CONDITION_VARIANTS)
            for model in MODELS
        )
        and all(
            set(record["models"][model][variant]["fieldwise_mse"])
            == {field.name for field in REPRESENTATION.fields}
            and set(record["models"][model][variant]["physical"]) == set(PHYSICAL_METRICS)
            for model in MODELS for variant in CONDITION_VARIANTS
        )
        for record in output["timesteps"].values()
    )
    return output


@torch.no_grad()
def native_evaluation(
    models: Mapping[str, torch.nn.Module],
    diffusion: GaussianDiffusion,
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
) -> Dict[str, object]:
    rest_vertices = load_rest_vertices(dataset, triples, device)
    metrics = {
        model_name: rollout(
            models[model_name],
            diffusion,
            dataset,
            triples,
            device,
            "d2m-shared",
            "matched",
            rest_vertices,
        )
        for model_name in MODELS
    }
    per_sequence = {
        model_name: {
            metric: per_sequence_native_error(metrics[model_name], metric)
            for metric in NATIVE_PAIRED_METRICS
        }
        for model_name in MODELS
    }
    current_minus_balanced = {
        metric: paired_difference(
            per_sequence["current"][metric], per_sequence["balanced"][metric],
        )
        for metric in NATIVE_PAIRED_METRICS
    }
    balanced_over_current = {
        metric: paired_mean_ratio(
            per_sequence["balanced"][metric], per_sequence["current"][metric],
        )
        for metric in NATIVE_PAIRED_METRICS
        if float(np.mean(per_sequence["current"][metric])) > 0.0
    }
    balanced_over_source = {
        metric: paired_mean_ratio(
            per_sequence["balanced"][metric], per_sequence["source"][metric],
        )
        for metric in NATIVE_PAIRED_METRICS
        if float(np.mean(per_sequence["source"][metric])) > 0.0
    }
    finite = all(
        np.isfinite(per_sequence[model_name][metric]).all()
        for model_name in MODELS for metric in NATIVE_PAIRED_METRICS
    )
    return {
        "finite": bool(finite),
        "models": metrics,
        "current_minus_balanced": current_minus_balanced,
        "balanced_over_current": balanced_over_current,
        "balanced_over_source": balanced_over_source,
        "all_native_metrics_reported": all(
            set(per_sequence[model_name]) == set(NATIVE_PAIRED_METRICS)
            and set(NATIVE_AGGREGATE_METRICS).issubset(
                metrics[model_name].get("aggregate", {})
            )
            for model_name in MODELS
        ),
        "sampler_future_gt": False,
        "sampler_stored_per_frame_bps": False,
        "sampler_noise_label": "D2:d2m-shared:paired:{step}",
    }


def evaluate(
    checkpoints: Mapping[str, Path],
    checkpoint_hashes: Mapping[str, str],
    *,
    device: torch.device,
    teacher_batch_size: int,
) -> Dict[str, object]:
    if set(checkpoints) != set(MODELS) or set(checkpoint_hashes) != set(MODELS):
        raise ValueError(f"D2-M requires checkpoint roles {MODELS}")
    if checkpoint_hashes["source"] != SOURCE_CHECKPOINT_SHA256:
        raise ValueError("D2-M source checkpoint configured hash mismatch")
    for name in MODELS:
        actual = sha256_file(checkpoints[name].resolve())
        if actual != checkpoint_hashes[name]:
            raise ValueError(f"D2-M {name} checkpoint hash mismatch: {actual}")
    dataset = PriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    teacher_selection = select_teacher_holdout(dataset)
    native_selection = select_native_holdout(dataset)
    if (
        teacher_selection["sha256"] != TEACHER_SELECTION_SHA256
        or native_selection["sha256"] != NATIVE_SELECTION_SHA256
    ):
        raise ValueError("D2-M fresh selection mismatch")
    models = {}
    metadata = {}
    for name in MODELS:
        models[name], metadata[name] = load_trained_hoi_prior(
            str(checkpoints[name].resolve()), device, weight_variant="online",
        )
    diffusion = GaussianDiffusion(500).to(device)
    teacher = teacher_evaluation(
        models,
        diffusion,
        dataset,
        teacher_selection["positions"],
        device,
        teacher_batch_size,
    )
    native = native_evaluation(
        models,
        diffusion,
        dataset,
        native_selection["triples"],
        device,
    )
    return {
        "checkpoints": {
            name: {
                "path": str(checkpoints[name].resolve()),
                "sha256": checkpoint_hashes[name],
                "metadata": metadata[name],
            }
            for name in MODELS
        },
        "teacher_selection": {
            key: value for key, value in teacher_selection.items()
            if key not in {"positions", "global_indices"}
        },
        "native_selection": {
            key: value for key, value in native_selection.items()
            if key not in {"triples", "global_indices"}
        },
        "teacher": teacher,
        "native": native,
        "official_test_used": False,
        "chois_used": False,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--current-checkpoint", type=Path, required=True)
    parser.add_argument("--current-sha256", required=True)
    parser.add_argument("--balanced-checkpoint", type=Path, required=True)
    parser.add_argument("--balanced-sha256", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--teacher-batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-M evaluation requires the four-GPU HOI worker CUDA context")
    result = evaluate(
        {
            "source": args.source_checkpoint,
            "current": args.current_checkpoint,
            "balanced": args.balanced_checkpoint,
        },
        {
            "source": args.source_sha256,
            "current": args.current_sha256,
            "balanced": args.balanced_sha256,
        },
        device=device,
        teacher_batch_size=args.teacher_batch_size,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
