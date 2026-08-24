#!/usr/bin/env python3
"""Measure HOIPrior geometry forward scale and raw component gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
for entry in (str(ROOT), str(CODE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)
os.environ.setdefault("ROOT_DIR", str(ROOT))

from tools.chois_evaluator import atomic_output


class GeometryGradientError(RuntimeError):
    """Raised when the geometry-gradient probe cannot establish its contract."""


EXPECTED_CHECKPOINT_SHA256 = "722d83ee7755b051e2095ccd01d4094bacce99589e679f89379f54661fb43704"
EXPECTED_MANIFEST_SHA256 = "4ba8abf789c00d1f1cf9eb7a22c92c4836781fb52e6872bc487f08763db79c03"
EXPECTED_SPLIT_SHA256 = "019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e"
EXPECTED_NORM_SHA256 = "6969c0c05ac3e03d9b014380118bee78ce8999e5b9adeeb8e700f4eba8baa969"
CHECKPOINT_RELATIVE = Path(
    "results/experiments/p1-hoi-p12-frame-repair-baseline-s42-20260819/checkpoints/"
    "p1-hoi-p12-frame-repair-baseline-s42-20260819_windows299520000.pth"
)
MANIFEST_DEFAULT = Path(".claude/scratch/maskfix_stagea/B1a_manifest_sealed_a.json")
SPLIT_RELATIVE = Path("experiments/splits/omomo_hoi_train_validation_seed42.json")
NORM_RELATIVE = Path("data/train/norm.npy")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_fixed_inputs(args: argparse.Namespace) -> None:
    expected_checkpoint = (ROOT / CHECKPOINT_RELATIVE).resolve()
    for checkpoint in args.checkpoint:
        actual = checkpoint.resolve()
        if actual != expected_checkpoint:
            raise GeometryGradientError(f"E_CHECKPOINT_PATH_MISMATCH: expected {expected_checkpoint}, got {actual}")
        if _sha256(actual) != EXPECTED_CHECKPOINT_SHA256:
            raise GeometryGradientError("E_CHECKPOINT_SHA_MISMATCH")
    manifest = args.manifest.resolve()
    split = (ROOT / SPLIT_RELATIVE).resolve()
    norm = (ROOT / NORM_RELATIVE).resolve()
    for path, expected, code in ((manifest, EXPECTED_MANIFEST_SHA256, "E_MANIFEST_SHA_MISMATCH"),
        (split, EXPECTED_SPLIT_SHA256, "E_SPLIT_MANIFEST_SHA_MISMATCH"),
        (norm, EXPECTED_NORM_SHA256, "E_NORM_SHA_MISMATCH")):
        if _sha256(path) != expected:
            raise GeometryGradientError(code)
    if args.window_count != 256:
        raise GeometryGradientError("E_WINDOW_COUNT_MISMATCH: weight derivation requires 256")
    if args.batch_size != 16:
        raise GeometryGradientError("E_BATCH_SIZE_MISMATCH: paired measurement requires 16")
    if args.device != "cpu":
        raise GeometryGradientError("E_DEVICE_MISMATCH: weight derivation is CPU-only")
    if args.mask_mode is not None and (
        args.shard != [0] or args.timesteps != [250]
    ):
        raise GeometryGradientError(
            "E_L3_CELL_MISMATCH: single_mode_l3 is fixed to shard 0 / timestep 250")
def _build_manifest_loaders(
    config_name: str, manifest_path: Path, shard_ids: Sequence[int], batch_size: int
):
    import torch
    from hydra import compose, initialize_config_dir
    from priors.hoi.data import PriorWindowDataset
    from torch.utils.data import DataLoader, Subset
    from datasets.utils import get_smpl_parents
    with initialize_config_dir(config_dir=str(CODE / "config"), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[
                "dataset_limit=0", f"batch_size={int(batch_size)}", "num_workers=0"
            ],
        )
    dataset = PriorWindowDataset(
        str(ROOT), "hoi", partition="train", limit=0,
        split_manifest=str((ROOT / SPLIT_RELATIVE).resolve()),
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    loaders = {}
    rc1_loader = None
    for record in manifest["shards"]:
        shard = int(record["shard_id"])
        if shard not in shard_ids:
            continue
        indices = [int(value) for value in record["window_indices"]]
        subset_positions = _manifest_subset_positions(dataset.indices, indices, shard)
        subset = Subset(dataset, subset_positions)
        loaders[shard] = DataLoader(
            subset, batch_size=int(batch_size), shuffle=False,
            drop_last=False, num_workers=0,
        )
        if shard == 0:
            rc1_loader = DataLoader(
                subset, batch_size=32, shuffle=False, drop_last=False, num_workers=0
            )
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long)
    return (
        cfg, loaders, rc1_loader, dataset, parents,
        torch.as_tensor(dataset.minimum, dtype=torch.float32),
        torch.as_tensor(dataset.maximum, dtype=torch.float32),
        torch.as_tensor(dataset.object_minimum, dtype=torch.float32),
        torch.as_tensor(dataset.object_maximum, dtype=torch.float32),
    )
def _manifest_subset_positions(dataset_indices, window_indices, shard_id=0):
    positions = {int(value): index for index, value in enumerate(dataset_indices)}
    if list(window_indices) != sorted(
        int(value) for value in window_indices
    ):
        raise GeometryGradientError(
            f"E_MANIFEST_INDEX_NOT_IN_DATASET: shard {shard_id} is not ascending")
    missing = [int(value) for value in window_indices if int(value) not in positions]
    if missing:
        raise GeometryGradientError(
            f"E_MANIFEST_INDEX_NOT_IN_DATASET: shard {shard_id} missing {missing[0]}")
    return [positions[int(value)] for value in window_indices]
def provenance() -> Dict[str, Any]:
    import numpy as np
    import torch
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=ROOT, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "tool_path": str(Path(__file__).resolve()),
        "tool_sha256": _sha256(ROOT / "tools/build_hoi_gradient_manifest.py"),
        "python_version": sys.version.split()[0],
        "torch_version": str(torch.__version__),
        "numpy_version": np.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    subparsers = parser.add_subparsers(dest="command", required=True)
    forward = subparsers.add_parser(
        "forward-scale", help="measure the GT floor and perturbation response on CPU"
    )
    forward.add_argument("--output", type=Path, required=True)
    forward.add_argument("--window-count", type=int, default=256)
    forward.add_argument("--batch-size", type=int, default=16)
    forward.add_argument("--config-name", default="config_train_hoi_prior_p12")
    decomposition = subparsers.add_parser(
        "palm-decomposition", help="decompose the GT floor by semantic palm role on CPU"
    )
    decomposition.add_argument("--output", type=Path, required=True)
    decomposition.add_argument("--window-count", type=int, default=256)
    decomposition.add_argument("--batch-size", type=int, default=16)
    decomposition.add_argument("--config-name", default="config_train_hoi_prior_p12")
    mask_fix = subparsers.add_parser(
        "mask-fix-floor", help="compare geometry contact-mask reducers on CPU"
    )
    mask_fix.add_argument("--output", type=Path, required=True)
    mask_fix.add_argument("--window-count", type=int, default=256)
    mask_fix.add_argument("--batch-size", type=int, default=16)
    mask_fix.add_argument("--config-name", default="config_train_hoi_prior_p12")
    derivation = subparsers.add_parser(
        "weight-derivation", help="measure raw component gradients at fixed timesteps"
    )
    derivation.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    derivation.add_argument("--output", type=Path, nargs="+", required=True)
    derivation.add_argument("--manifest", type=Path, default=MANIFEST_DEFAULT)
    derivation.add_argument("--shard", type=int, nargs="+", default=[0, 1, 2, 3])
    derivation.add_argument("--window-count", type=int, default=256)
    derivation.add_argument("--batch-size", type=int, default=16)
    derivation.add_argument("--timesteps", type=int, nargs="+", default=[0, 125, 250, 375, 499])
    derivation.add_argument("--config-name", default="config_train_hoi_prior_p12")
    derivation.add_argument(
        "--weight-variant",
        choices=("online", "ema_0.999", "ema_0.9999"),
        default="online",
    )
    derivation.add_argument("--device", default="cpu")
    derivation.add_argument(
        "--mask-mode", choices=("sealed", "per_hand_per_frame"), default=None,
        help="run one single_mode_l3 invocation; omitted for paired_joint",
    )
    return parser
def _build_loader(
    config_name: str, window_count: int, batch_size: int
) -> Tuple[Any, Any, Any, Any, Any, Any, Any]:
    import torch
    from hydra import compose, initialize_config_dir
    from priors.hoi.data import PriorWindowDataset
    from torch.utils.data import DataLoader
    from datasets.utils import get_smpl_parents
    with initialize_config_dir(config_dir=str(CODE / "config"), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=[
                f"dataset_limit={int(window_count)}",
                f"batch_size={int(batch_size)}",
                "num_workers=0",
            ],
        )
    dataset = PriorWindowDataset(
        str(ROOT),
        "hoi",
        partition="train",
        limit=int(window_count),
        split_manifest=str(Path(str(cfg.split_manifest)).resolve()),
    )
    loader = DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long)
    return (
        cfg,
        loader,
        parents,
        torch.as_tensor(dataset.minimum, dtype=torch.float32),
        torch.as_tensor(dataset.maximum, dtype=torch.float32),
        torch.as_tensor(dataset.object_minimum, dtype=torch.float32),
        torch.as_tensor(dataset.object_maximum, dtype=torch.float32),
    )
def _load_model(checkpoint_path: Path, weight_variant: str, device: str):
    import torch
    from priors.hoi.diffusion import GaussianDiffusion
    from priors.hoi.models import build_expert
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model_config = checkpoint["model_config"]
    model = build_expert("hoi", init_checkpoint=None, **model_config)
    if weight_variant == "online":
        state = checkpoint["model"]
    else:
        state = checkpoint["ema_models"][weight_variant.removeprefix("ema_")]
    model.load_state_dict(state, strict=True)
    target = torch.device(device)
    model = model.to(target)
    diffusion = GaussianDiffusion(500).to(target)
    return model, diffusion, target
def _scratch_output(final_output: Path) -> Path:
    scratch = ROOT / ".claude" / "scratch" / "w3_stagea"
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch / f".{final_output.name}.{os.getpid()}.json"
def _finalize(result: Dict[str, Any], temporary: Path, output: Path) -> Dict[str, Any]:
    temporary.unlink()
    result["provenance"] = {**result.get("provenance", {}), **provenance()}
    result["probe_sha256"] = _sha256(Path(__file__).resolve())
    atomic_output(output, result)
    sidecar = output.with_name(output.name + ".sha256")
    if sidecar.exists():
        raise GeometryGradientError(f"refusing to overwrite output hash sidecar: {sidecar}")
    sidecar.write_text(f"{_sha256(output)}  {output.name}\n", encoding="utf-8")
    return result
def _l3_verdict(l3_results: Sequence[Dict[str, Any]], paired: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
    """Compare two single-mode L3 arms against the paired cell (shard 0, t=250)."""
    paired_cell = paired["pairing"]["input_sha256"]["0"]["250"]
    paired_seed = paired["pairing"]["cell_seed"]["0"]["250"]
    input_equal = (l3_results[0]["pairing"]["input_sha256"]["0"]["250"]
        == l3_results[1]["pairing"]["input_sha256"]["0"]["250"] == paired_cell)
    seed_equal = (
        l3_results[0]["pairing"]["cell_seed"]["0"]["250"]
        == l3_results[1]["pairing"]["cell_seed"]["0"]["250"] == paired_seed)
    cell_timestep_equal = bool(
        l3_results[0]["timesteps"] == l3_results[1]["timesteps"] == [250]
        and 250 in paired["timesteps"])
    cell_global_seed_equal = (
        l3_results[0]["pairing"]["cell_global_seed"]["0"]["250"]
        == l3_results[1]["pairing"]["cell_global_seed"]["0"]["250"]
        == paired["pairing"]["cell_global_seed"]["0"]["250"])
    shared_values = [l3["shared"]["per_shard"]["0"]["gradient_l2_nongeometry"]["250"] for l3 in l3_results]
    nongeometry_equal = (
        shared_values[0] == shared_values[1] == paired["shared"]["per_shard"]["0"]["gradient_l2_nongeometry"]["250"])
    geometry_equal = {mode: all(
        l3["modes"][mode]["per_shard"]["0"]["geometry_by_channel"]["250"]
        == paired["modes"][mode]["per_shard"]["0"]["geometry_by_channel"]["250"]
        for l3 in l3_results if mode in l3["modes"]) for mode in ("sealed", "per_hand_per_frame")}
    passed = (input_equal and seed_equal and cell_timestep_equal
        and cell_global_seed_equal and nongeometry_equal
        and all(geometry_equal.values()))
    crosscheck = {"performed": True, "cell": {"shard": 0, "timestep": 250},
        "artifacts": [], "input_sha256_equal": input_equal,
        "cell_seed_equal": seed_equal, "cell_timestep_equal": cell_timestep_equal,
        "cell_global_seed_equal": cell_global_seed_equal,
        "nongeometry_norms_bitwise_equal": nongeometry_equal,
        "geometry_matches_paired_joint": geometry_equal}
    return passed, crosscheck
def _run_l3(args, checkpoint_path: Path, paired: Dict[str, Any]) -> None:
    scratch = ROOT / ".claude" / "scratch" / "maskfix_stagea"
    scratch.mkdir(parents=True, exist_ok=True)
    artifacts = []
    started = time.perf_counter()
    for mode in ("sealed", "per_hand_per_frame"):
        artifact = scratch / f".l3_{mode}_{os.getpid()}.json"
        log_path = scratch / f"l3_{mode}_{os.getpid()}.log"
        command = ["/data/yujinlun/anaconda3/envs/infbagel/bin/python", str(Path(__file__).resolve()),
            "weight-derivation", "--checkpoint", str(checkpoint_path.resolve()), "--output", str(artifact),
            "--manifest", str(args.manifest.resolve()), "--shard", "0", "--window-count", "256",
            "--batch-size", "16", "--timesteps", "250", "--config-name", args.config_name,
            "--weight-variant", args.weight_variant, "--device", "cpu", "--mask-mode", mode,]
        with log_path.open("w", encoding="utf-8") as log:
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        artifacts.append(artifact)
    l3_results = [json.loads(path.read_text(encoding="utf-8")) for path in artifacts]
    passed, crosscheck = _l3_verdict(l3_results, paired)
    crosscheck["artifacts"] = [str(path.resolve()) for path in artifacts]
    paired["pairing"]["L3_independent_invocation_crosscheck"].update(crosscheck)
    paired["gates_shared"]["G9_pairing"] = passed
    if passed:
        paired["candidate"]["blocked_by"] = [
            value for value in paired["candidate"]["blocked_by"] if value != "G9_pairing"
        ]
        if not paired["candidate"]["blocked_by"] and not paired["candidate"]["sealed_gate_divergence"]:
            paired["candidate"]["produced"] = True
            paired["candidate"]["hand_object_contact_weight"] = (
                paired["modes"]["per_hand_per_frame"]["aggregate"]["w_geom_star"])
    else:
        if "G9_pairing" not in paired["candidate"]["blocked_by"]:
            paired["candidate"]["blocked_by"].append("G9_pairing")
        paired["candidate"]["produced"] = False
        paired["candidate"]["hand_object_contact_weight"] = None
    elapsed = time.perf_counter() - started
    paired["timing"]["t_l3_total_seconds"] = elapsed
    paired["timing"]["projected_total_seconds"] = (
        paired["timing"]["t_setup_seconds"] + 20.0 * paired["timing"]["t_cell_plain_seconds"]
        + float(paired["timing"]["t_rc1_delta_seconds"] or 0.0) + float(paired["timing"]["t_rc2_delta_seconds"] or 0.0)
        + elapsed)
def run(args: argparse.Namespace) -> Any:
    if args.command == "weight-derivation":
        _validate_fixed_inputs(args)
    import torch
    from priors.hoi.diagnostics import (
        geometry_term_forward_scale_probe,
        geometry_term_palm_decomposition_probe,
        geometry_mask_fix_floor_probe,
        geometry_weight_derivation_probe,
    )
    cpu_commands = ("forward-scale", "palm-decomposition", "mask-fix-floor")
    outputs = [args.output] if args.command in cpu_commands else list(args.output)
    for output in outputs:
        if output.resolve().exists():
            raise GeometryGradientError(
                f"refusing to overwrite geometry-gradient output: {output.resolve()}"
            )
    if args.command == "weight-derivation" and len(args.checkpoint) != len(outputs):
        raise GeometryGradientError(
            "weight-derivation requires one --output path per --checkpoint path"
        )
    if args.command == "weight-derivation":
        setup_started = time.perf_counter()
        cfg, loaders, rc1_loader, _dataset, parents, minimum, maximum, object_minimum, object_maximum = (
            _build_manifest_loaders(
            args.config_name, args.manifest.resolve(), args.shard, args.batch_size
            ))
    else:
        cfg, loader, parents, minimum, maximum, object_minimum, object_maximum = _build_loader(
            args.config_name, args.window_count, args.batch_size
        )
    if args.command == "forward-scale":
        temporary = _scratch_output(outputs[0])
        result = geometry_term_forward_scale_probe(
            loader,
            parents,
            minimum,
            maximum,
            object_minimum,
            object_maximum,
            cfg,
            output_path=temporary,
            window_count=args.window_count,
        )
        return _finalize(result, temporary, outputs[0].resolve())
    if args.command == "palm-decomposition":
        temporary = _scratch_output(outputs[0])
        result = geometry_term_palm_decomposition_probe(
            loader,
            parents,
            minimum,
            maximum,
            object_minimum,
            object_maximum,
            cfg,
            output_path=temporary,
            window_count=args.window_count,
        )
        return _finalize(result, temporary, outputs[0].resolve())
    if args.command == "mask-fix-floor":
        temporary = _scratch_output(outputs[0])
        result = geometry_mask_fix_floor_probe(
            loader, parents, minimum, maximum, object_minimum, object_maximum, cfg,
            output_path=temporary, window_count=args.window_count,
            batch_size=args.batch_size,
        )
        return _finalize(result, temporary, outputs[0].resolve())
    records = []
    for checkpoint_path, output in zip(args.checkpoint, outputs):
        model, diffusion, device = _load_model(
            checkpoint_path.resolve(), args.weight_variant, args.device
        )
        setup_seconds = time.perf_counter() - setup_started
        temporary = _scratch_output(output)
        result = geometry_weight_derivation_probe(
            model,
            diffusion,
            loaders,
            parents.to(device),
            minimum.to(device),
            maximum.to(device),
            object_minimum.to(device),
            object_maximum.to(device),
            cfg,
            checkpoint_path=checkpoint_path,
            output_path=temporary,
            window_count=args.window_count,
            timesteps=tuple(args.timesteps),
            device=device,
            manifest_path=args.manifest.resolve(),
            shard_ids=tuple(args.shard),
            measurement_mode="single_mode_l3" if args.mask_mode else "paired_joint",
            mask_mode=args.mask_mode,
            rc1_loader=rc1_loader,
            timing={"t_setup_seconds": setup_seconds},
        )
        if args.mask_mode is None:
            _run_l3(args, checkpoint_path, result)
        records.append(_finalize(result, temporary, output.resolve()))
        del model, diffusion
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return records
def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        results = run(args)
    except (GeometryGradientError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    outputs = (
        [args.output]
        if args.command in ("forward-scale", "palm-decomposition", "mask-fix-floor")
        else args.output
    )
    print(json.dumps({"outputs": [str(path.resolve()) for path in outputs]}, indent=2))
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
