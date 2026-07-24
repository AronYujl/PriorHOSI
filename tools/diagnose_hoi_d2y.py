#!/usr/bin/env python3
"""Run the preregistered D2-X/D2-Y routed-foot gradient and residual audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import numpy as np
import torch
from pytorch3d import transforms
from torch.utils.data import default_collate


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
sys.path.insert(0, str(REPO))

from datasets.utils import get_smpl_parents  # noqa: E402
from priors.data import PriorWindowDataset  # noqa: E402
from priors.diffusion import GaussianDiffusion, normalize_progress  # noqa: E402
from priors.losses import (  # noqa: E402
    D2X_FOOT_XZ_VELOCITY_SLOTS,
    _fk_positions,
    _velocity_residuals,
    hoi_training_losses,
)
from priors.models import load_trained_hoi_prior  # noqa: E402
from priors.optimizer_reset import (  # noqa: E402
    NATIVE_SELECTION_SHA256,
    paired_difference,
    select_native_holdout,
    stable_seed,
)
from priors.representation import REPRESENTATION  # noqa: E402


RUN_ID = "p1-hoi-d2y-routed-foot-amplification-internal-s42-20260724"
SUBPHASE = "1B-D2-Y0-internal"
EXPECTED_PYTHON = "/home/yujinlun/data/envs/infbagel/bin/python"
EXPECTED_DATA_CONTRACT_SHA256 = (
    "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
)
CHECKPOINT_WINDOWS = {
    "early": 3_072_000,
    "mid": 30_720_000,
    "final": 61_440_000,
}
D2X_HASHES = {
    "early": "cf62327e97683ca670bb714f63dd14cb83a8aed4f737289bbd28b6a8bd16713d",
    "mid": "da0c253b2aa33042984ec6ea95eb6b02eef7d8b32a9896ad32fca3148998947e",
    "final": "b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51",
}
TIMESTEPS = (0, 249, 499)
OBJECTIVE_WEIGHTS = {
    "reconstruction": 1.0,
    "fk": 0.3569973401779424,
    "object_surface": 0.4772322188400037,
    "object_goal": 1.0,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tensor_sha256(value: torch.Tensor) -> str:
    return hashlib.sha256(
        value.detach().contiguous().cpu().numpy().tobytes()
    ).hexdigest()


def exclusive_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _checkpoint_arguments(parser: argparse.ArgumentParser) -> None:
    for expert in ("d2x", "d2y"):
        for stratum in CHECKPOINT_WINDOWS:
            parser.add_argument(
                f"--{expert}-{stratum}-checkpoint",
                type=Path,
                required=True,
            )
            parser.add_argument(
                f"--{expert}-{stratum}-sha256",
                required=True,
            )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    _checkpoint_arguments(parser)
    args = parser.parse_args()
    for expert in ("d2x", "d2y"):
        for stratum in CHECKPOINT_WINDOWS:
            value = getattr(args, f"{expert}_{stratum}_sha256")
            if not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError(f"{expert}/{stratum} SHA-256 must be lowercase hexadecimal")
    return args


def _checkpoint_contract(
    path: Path,
    expected_sha256: str,
    *,
    expert: str,
    stratum: str,
) -> Dict[str, object]:
    actual = sha256_file(path)
    expected_windows = CHECKPOINT_WINDOWS[stratum]
    expected_run = (
        "p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723"
        if expert == "d2x"
        else "p1-hoi-d2y-routed-foot-amplification-s42-20260723"
    )
    expected_basename = f"{expected_run}_windows{expected_windows:09d}.pth"
    if actual != expected_sha256:
        raise ValueError(f"{expert}/{stratum} checkpoint hash mismatch")
    if path.name != expected_basename:
        raise ValueError(f"{expert}/{stratum} checkpoint basename mismatch")
    raw = torch.load(path, map_location="cpu")
    initialization = raw.get("weight_initialization", {})
    checks = {
        "checkpoint_type": raw.get("checkpoint_type") == "hoi_prior_phase1b",
        "run_id": raw.get("run_id") == expected_run,
        "seed": raw.get("seed") == 42,
        "processed_windows": raw.get("processed_windows") == expected_windows,
        "optimizer_updates": raw.get("optimizer_updates") == expected_windows // 2048,
        "world_size": raw.get("world_size") == 4,
        "effective_batch_size": raw.get("effective_batch_size") == 2048,
        "data_contract": (
            raw.get("data_contract_sha256") == EXPECTED_DATA_CONTRACT_SHA256
        ),
        "random_initialization": (
            initialization.get("mode") == "random"
            and initialization.get("source_checkpoint") is None
            and initialization.get("restored_components") == []
        ),
        "model_only_audit": isinstance(raw.get("model"), dict),
    }
    failed = sorted(name for name, passed in checks.items() if not passed)
    if failed:
        raise ValueError(f"{expert}/{stratum} checkpoint contract mismatch: {failed}")
    return {
        "path": str(path),
        "sha256": actual,
        "processed_windows": expected_windows,
        "optimizer_updates": expected_windows // 2048,
        "git_commit": raw.get("git_commit"),
        "checks": checks,
    }


def _move_batch(batch: Mapping[str, torch.Tensor], device: torch.device):
    return {
        key: batch[key].to(device)
        for key in (
            "x",
            "text_embedding",
            "object_bps",
            "goals",
            "progress",
            "rest_human_offsets",
            "terminal_window",
            "rest_object_points",
            "world_to_local_rotation",
            "object_rotation_reference",
        )
    }


def _gradient_norm(
    named_gradients: Sequence[Tuple[str, torch.Tensor | None]],
    prefixes: Sequence[str] | None = None,
) -> float:
    values = []
    for name, gradient in named_gradients:
        if gradient is None:
            continue
        if prefixes is None or any(name.startswith(prefix) for prefix in prefixes):
            values.append(gradient.detach().double().square().sum())
    if not values:
        return 0.0
    return float(torch.stack(values).sum().sqrt().item())


def gradient_cosine(
    first: Sequence[torch.Tensor | None],
    second: Sequence[torch.Tensor | None],
) -> float | None:
    dot = None
    left = None
    right = None
    for a, b in zip(first, second):
        if a is None or b is None:
            continue
        product = (a.detach().double() * b.detach().double()).sum()
        a2 = a.detach().double().square().sum()
        b2 = b.detach().double().square().sum()
        dot = product if dot is None else dot + product
        left = a2 if left is None else left + a2
        right = b2 if right is None else right + b2
    if dot is None or left is None or right is None:
        return None
    denominator = left.sqrt() * right.sqrt()
    if not math.isfinite(float(denominator)) or float(denominator) == 0.0:
        return None
    return float((dot / denominator).item())


def _output_slice_norm(
    named_gradients: Mapping[str, torch.Tensor | None],
    start: int,
    stop: int,
) -> float:
    values = []
    for suffix in ("weight", "bias"):
        value = named_gradients.get(f"network.output.{suffix}")
        if value is not None:
            values.append(value[start:stop].detach().double().square().sum())
    return float(torch.stack(values).sum().sqrt().item()) if values else 0.0


def _predicted_fk(
    prediction: torch.Tensor,
    rest_offsets: torch.Tensor,
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
) -> torch.Tensor:
    scale = (maximum - minimum).reshape(1, 1, 1, 3)
    base = minimum.reshape(1, 1, 1, 3)
    positions = (
        (prediction[..., :84].reshape(*prediction.shape[:2], 28, 3) + 1.0)
        * scale / 2.0 + base
    )
    rotations = transforms.rotation_6d_to_matrix(
        prediction[..., 84:216].reshape(*prediction.shape[:2], 22, 6)
    )
    return _fk_positions(positions[..., 0, :], rotations, rest_offsets, parents)


def evaluate_model_timestep(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    *,
    timestep: int,
    residual_multiplier: float,
    seed_label: str,
) -> Dict[str, object]:
    model.train()
    clean = batch["x"]
    generator = torch.Generator(device=clean.device)
    generator.manual_seed(stable_seed(seed_label + ":noise"))
    noise = torch.randn(clean.shape, device=clean.device, generator=generator)
    timesteps = torch.full(
        (clean.shape[0],), timestep, device=clean.device, dtype=torch.long,
    )
    noisy = diffusion.q_sample(clean, timesteps, noise)
    dropout_seed = stable_seed(seed_label + ":dropout")
    torch.manual_seed(dropout_seed)
    torch.cuda.manual_seed_all(dropout_seed)
    prediction = model(
        noisy,
        timesteps,
        batch["text_embedding"],
        batch["object_bps"],
        batch["goals"],
        normalize_progress(batch["progress"]),
    )
    fk = _predicted_fk(
        prediction, batch["rest_human_offsets"], parents, minimum, maximum,
    )
    predicted_residual, target_residual = _velocity_residuals(
        prediction,
        clean,
        fk,
        minimum,
        maximum,
        fk_foot_temporal_routing=True,
    )
    routed_error = (
        predicted_residual[..., list(D2X_FOOT_XZ_VELOCITY_SLOTS)]
        - target_residual[..., list(D2X_FOOT_XZ_VELOCITY_SLOTS)]
    ).square()
    per_window_mse = routed_error.mean(dim=(1, 2))
    if per_window_mse.shape[0] != 96:
        raise ValueError("D2-Y internal diagnostic requires exactly 96 selected windows")
    per_sequence_mse = per_window_mse.reshape(32, 3).mean(dim=1)
    losses = hoi_training_losses(
        prediction,
        clean,
        batch["goals"],
        batch["rest_human_offsets"],
        parents,
        minimum,
        maximum,
        object_minimum,
        object_maximum,
        batch["terminal_window"],
        batch["rest_object_points"],
        batch["world_to_local_rotation"],
        batch["object_rotation_reference"],
        fk_weight=0.3569973401779424,
        object_surface_weight=0.4772322188400037,
        velocity_weight=0.1,
        goal_weight=1.0,
        fk_foot_temporal_routing=True,
        routed_foot_residual_multiplier=1.0,
    )
    routed_contribution = (
        0.1 * float(residual_multiplier) * routed_error.sum()
        / predicted_residual.numel()
    )
    named_parameters = [
        (name, parameter) for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    parameters = [parameter for _, parameter in named_parameters]
    routed_gradients = torch.autograd.grad(
        routed_contribution,
        parameters,
        retain_graph=True,
        allow_unused=True,
    )
    named_routed = list(zip((name for name, _ in named_parameters), routed_gradients))
    named_routed_map = dict(named_routed)
    cosines = {}
    objective_norms = {}
    objective_items = list(OBJECTIVE_WEIGHTS.items())
    for index, (name, weight) in enumerate(objective_items):
        gradients = torch.autograd.grad(
            float(weight) * losses[name],
            parameters,
            retain_graph=index < len(objective_items) - 1,
            allow_unused=True,
        )
        cosines[name] = gradient_cosine(routed_gradients, gradients)
        objective_norms[name] = _gradient_norm(
            list(zip((item[0] for item in named_parameters), gradients))
        )
    values = per_sequence_mse.detach().double().cpu().numpy()
    if not np.isfinite(values).all():
        raise ValueError("D2-Y internal routed residual contains nonfinite values")
    return {
        "timestep": timestep,
        "noise_sha256": tensor_sha256(noise),
        "dropout_seed": dropout_seed,
        "routed_residual_rms": float(routed_error.mean().sqrt().item()),
        "routed_residual_mse_by_sequence": values.tolist(),
        "routed_training_contribution": float(routed_contribution.detach().item()),
        "gradient_norms": {
            "all_parameters": _gradient_norm(named_routed),
            "input_projection": _gradient_norm(
                named_routed, ("network.motion_input.",)
            ),
            "output_projection": _gradient_norm(
                named_routed, ("network.output.",)
            ),
            "transformer": _gradient_norm(
                named_routed, ("network.transformer.",)
            ),
            "root_output": _output_slice_norm(named_routed_map, 0, 3),
            "rotation_output": _output_slice_norm(named_routed_map, 84, 216),
        },
        "objective_gradient_norms": objective_norms,
        "gradient_cosines": cosines,
    }


def _build_batch(
    dataset: PriorWindowDataset,
    triples: Sequence[Sequence[int]],
    device: torch.device,
) -> Mapping[str, torch.Tensor]:
    items = [dataset[position] for triple in triples for position in triple]
    return _move_batch(default_collate(items), device)


def _load_and_evaluate(
    path: Path,
    device: torch.device,
    diffusion: GaussianDiffusion,
    batch: Mapping[str, torch.Tensor],
    parents: torch.Tensor,
    minimum: torch.Tensor,
    maximum: torch.Tensor,
    object_minimum: torch.Tensor,
    object_maximum: torch.Tensor,
    *,
    expert: str,
    stratum: str,
) -> Dict[str, object]:
    model, metadata = load_trained_hoi_prior(
        str(path), device, weight_variant="online",
    )
    if metadata["data_contract_sha256"] != EXPECTED_DATA_CONTRACT_SHA256:
        raise ValueError(f"{expert}/{stratum} loaded model data contract mismatch")
    multiplier = 1.0 if expert == "d2x" else 1024.0
    results = {}
    for timestep in TIMESTEPS:
        results[str(timestep)] = evaluate_model_timestep(
            model,
            diffusion,
            batch,
            parents,
            minimum,
            maximum,
            object_minimum,
            object_maximum,
            timestep=timestep,
            residual_multiplier=multiplier,
            seed_label=f"D2Y:paired:t{timestep}",
        )
    del model
    torch.cuda.empty_cache()
    return {"metadata": metadata, "timesteps": results}


def mechanism_decision(results: Mapping[str, object]) -> Dict[str, object]:
    timestep_checks = {}
    for timestep in ("249", "499"):
        control = results["d2x"]["final"]["timesteps"][timestep][
            "routed_residual_mse_by_sequence"
        ]
        target = results["d2y"]["final"]["timesteps"][timestep][
            "routed_residual_mse_by_sequence"
        ]
        comparison = paired_difference(control, target)
        comparison["passed"] = comparison["bootstrap_95_ci"][0] > 0.0
        timestep_checks[timestep] = comparison
    passed = all(item["passed"] for item in timestep_checks.values())
    return {
        "mechanism_passed": passed,
        "timestep_checks": timestep_checks,
        "classification": (
            "routed-foot-amplification-internal-positive"
            if passed else "routed-foot-amplification-optimization-negative"
        ),
        "checkpoint_selected": False,
        "consistency_authorized": False,
        "consistency_started": False,
    }


def main() -> None:
    args = parse_args()
    if args.run_id != RUN_ID:
        raise ValueError(f"D2-Y internal run id must be {RUN_ID}")
    if args.python.resolve() != Path(os.environ.get("INFBAGEL_PYTHON", "")).resolve():
        raise ValueError("D2-Y internal diagnostic requires absolute INFBAGEL_PYTHON")
    if args.python.resolve() != Path(EXPECTED_PYTHON).resolve():
        raise ValueError(f"D2-Y internal diagnostic requires {EXPECTED_PYTHON}")
    if os.environ.get("INFBAGEL_WORKER_EXPERT") != "hoi" or socket.gethostname() != "node01":
        raise RuntimeError("D2-Y internal diagnostic is restricted to the HOI worker")
    if subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=REPO,
        text=True,
    ).strip():
        raise RuntimeError("D2-Y internal diagnostic refuses a dirty worker checkout")
    output_path = args.output.resolve()
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("D2-Y internal diagnostic requires worker CUDA")
    checkpoint_contracts = {"d2x": {}, "d2y": {}}
    checkpoint_paths = {"d2x": {}, "d2y": {}}
    for expert in ("d2x", "d2y"):
        for stratum in CHECKPOINT_WINDOWS:
            path = getattr(args, f"{expert}_{stratum}_checkpoint").resolve()
            expected_hash = getattr(args, f"{expert}_{stratum}_sha256")
            if expert == "d2x" and expected_hash != D2X_HASHES[stratum]:
                raise ValueError(f"D2-X {stratum} hash is not the sealed control")
            checkpoint_paths[expert][stratum] = path
            checkpoint_contracts[expert][stratum] = _checkpoint_contract(
                path,
                expected_hash,
                expert=expert,
                stratum=stratum,
            )
    dataset = PriorWindowDataset(
        str(REPO),
        "hoi",
        partition="internal_validation",
        split_manifest="experiments/splits/omomo_hoi_train_validation_seed42.json",
    )
    selection = select_native_holdout(dataset)
    if selection["sha256"] != NATIVE_SELECTION_SHA256 or selection["sequences"] != 32:
        raise ValueError("D2-Y internal selection mismatch")
    batch = _build_batch(dataset, selection["triples"], device)
    parents = torch.as_tensor(
        get_smpl_parents(use_joints24=True), device=device, dtype=torch.long,
    )
    minimum = torch.as_tensor(dataset.minimum, device=device)
    maximum = torch.as_tensor(dataset.maximum, device=device)
    object_minimum = torch.as_tensor(dataset.object_minimum, device=device)
    object_maximum = torch.as_tensor(dataset.object_maximum, device=device)
    diffusion = GaussianDiffusion(REPRESENTATION.diffusion_steps).to(device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    results = {"d2x": {}, "d2y": {}}
    for expert in ("d2x", "d2y"):
        for stratum in CHECKPOINT_WINDOWS:
            results[expert][stratum] = _load_and_evaluate(
                checkpoint_paths[expert][stratum],
                device,
                diffusion,
                batch,
                parents,
                minimum,
                maximum,
                object_minimum,
                object_maximum,
                expert=expert,
                stratum=stratum,
            )
    torch.cuda.synchronize(device)
    decision = mechanism_decision(results)
    output = {
        "schema_version": 1,
        "run_id": RUN_ID,
        "phase": "p1",
        "subphase": SUBPHASE,
        "status": "completed",
        "seed": 42,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True,
        ).strip(),
        "selection": {
            "partition": "internal_validation",
            "eligible_sequence_ranks": [128, 159],
            "sequences": 32,
            "windows": 96,
            "sha256": selection["sha256"],
            "global_indices": selection["global_indices"],
            "official_test_sequences": 0,
        },
        "checkpoints": {
            expert: {
                "early_sha256": checkpoint_contracts[expert]["early"]["sha256"],
                "mid_sha256": checkpoint_contracts[expert]["mid"]["sha256"],
                "final_sha256": checkpoint_contracts[expert]["final"]["sha256"],
            }
            for expert in ("d2x", "d2y")
        },
        "checkpoint_contracts": checkpoint_contracts,
        "timesteps": list(TIMESTEPS),
        "pairing": {
            "same_clean_windows": True,
            "same_timestep": True,
            "same_noise": True,
            "same_dropout_seed": True,
            "model_mode": "train",
            "condition_dropout": "model-native dropout replayed with identical seed",
        },
        "results": results,
        "decision": decision,
        "runtime": {
            "seconds": time.perf_counter() - started,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device),
            "cuda_synchronized_for_timing": True,
        },
        "optimizer_created": False,
        "training_updates": 0,
        "checkpoint_write": False,
        "checkpoint_selected": False,
        "released_checkpoint_loaded": False,
        "author_checkpoint_loaded": False,
        "official_test_used": False,
        "consistency_started": False,
    }
    exclusive_json(output_path, output)


if __name__ == "__main__":
    main()
