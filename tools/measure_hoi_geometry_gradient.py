#!/usr/bin/env python3
"""Measure HOIPrior geometry forward scale and raw component gradients."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        "tool_sha256": _sha256(Path(__file__).resolve()),
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

    derivation = subparsers.add_parser(
        "weight-derivation", help="measure raw component gradients at fixed timesteps"
    )
    derivation.add_argument("--checkpoint", type=Path, nargs="+", required=True)
    derivation.add_argument("--output", type=Path, nargs="+", required=True)
    derivation.add_argument("--window-count", type=int, default=64)
    derivation.add_argument("--batch-size", type=int, default=8)
    derivation.add_argument("--timesteps", type=int, nargs="+", default=[250, 499])
    derivation.add_argument("--config-name", default="config_train_hoi_prior_p12")
    derivation.add_argument(
        "--weight-variant",
        choices=("online", "ema_0.999", "ema_0.9999"),
        default="online",
    )
    derivation.add_argument("--device", default="cpu")
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
    result["provenance"] = provenance()
    result["probe_sha256"] = _sha256(Path(__file__).resolve())
    atomic_output(output, result)
    return result


def run(args: argparse.Namespace) -> Any:
    import torch
    from priors.hoi.diagnostics import (
        geometry_term_forward_scale_probe,
        geometry_weight_derivation_probe,
    )

    outputs = [args.output] if args.command == "forward-scale" else list(args.output)
    for output in outputs:
        if output.resolve().exists():
            raise GeometryGradientError(
                f"refusing to overwrite geometry-gradient output: {output.resolve()}"
            )

    if args.command == "weight-derivation" and len(args.checkpoint) != len(outputs):
        raise GeometryGradientError(
            "weight-derivation requires one --output path per --checkpoint path"
        )

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

    records = []
    for checkpoint_path, output in zip(args.checkpoint, outputs):
        model, diffusion, device = _load_model(
            checkpoint_path.resolve(), args.weight_variant, args.device
        )
        temporary = _scratch_output(output)
        result = geometry_weight_derivation_probe(
            model,
            diffusion,
            loader,
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
        )
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
    outputs = [args.output] if args.command == "forward-scale" else args.output
    print(json.dumps({"outputs": [str(path.resolve()) for path in outputs]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
