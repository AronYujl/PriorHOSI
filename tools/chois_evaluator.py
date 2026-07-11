#!/usr/bin/env python3
"""Audit the pinned CHOIS evaluator and validate InfBaGel OMOMO exports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "experiments" / "evaluators" / "chois_omomo.json"


class EvaluatorError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(relative + b"\0" + sha256_file(child).encode("ascii") + b"\0")
    return digest.hexdigest()


def load_config(path: Path) -> Dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise EvaluatorError("evaluator config must be a JSON object")
    return value


def git_output(root: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), *args], text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if completed.returncode:
        raise EvaluatorError(completed.stderr.strip() or "git command failed")
    return completed.stdout.strip()


def verify_upstream(root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    if not (root / ".git").exists():
        raise EvaluatorError(f"not a CHOIS Git checkout: {root}")
    commit = git_output(root, "rev-parse", "HEAD")
    expected_commit = config["upstream_commit"]
    if commit != expected_commit:
        raise EvaluatorError(f"CHOIS commit mismatch: expected {expected_commit}, got {commit}")
    dirty = git_output(root, "status", "--porcelain")
    if dirty:
        raise EvaluatorError("CHOIS checkout is dirty")
    hashes = {}
    for relative, expected in config["files"].items():
        path = root / relative
        if not path.is_file():
            raise EvaluatorError(f"missing upstream evaluator file: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise EvaluatorError(f"upstream hash mismatch for {relative}: {actual}")
        hashes[relative] = actual
    return {"commit": commit, "files": hashes}


def read_npz_directory(path: Path) -> Tuple[Dict[str, Dict[str, Any]], str]:
    try:
        import numpy as np
    except ImportError as exc:
        raise EvaluatorError("numpy is required to validate evaluator inputs") from exc
    if not path.is_dir():
        raise EvaluatorError(f"NPZ directory does not exist: {path}")
    files = sorted(path.glob("*.npz"))
    if not files:
        raise EvaluatorError(f"no .npz files found in {path}")
    sequences: Dict[str, Dict[str, Any]] = {}
    for file in files:
        try:
            with np.load(file, allow_pickle=False) as value:
                missing = {"seq_name", "global_jpos"} - set(value.files)
                if missing:
                    raise EvaluatorError(f"{file.name} missing keys {sorted(missing)}")
                sequence_name = str(value["seq_name"].item())
                joints = value["global_jpos"]
        except (OSError, ValueError) as exc:
            raise EvaluatorError(f"cannot read {file}: {exc}") from exc
        if joints.ndim != 3 or joints.shape[1:] != (24, 3) or joints.shape[0] < 2:
            raise EvaluatorError(
                f"{file.name} global_jpos must have shape [T,24,3] with T>=2; got {joints.shape}"
            )
        if not np.issubdtype(joints.dtype, np.floating) or not np.isfinite(joints).all():
            raise EvaluatorError(f"{file.name} global_jpos must be finite floating point")
        if sequence_name in sequences:
            raise EvaluatorError(f"duplicate seq_name {sequence_name!r}")
        sequences[sequence_name] = {
            "file": file.name,
            "frames": int(joints.shape[0]),
            "dtype": str(joints.dtype),
            "sha256": sha256_file(file),
        }
    return sequences, sha256_tree(path)


def validate_pair(predictions: Path, ground_truth: Path) -> Dict[str, Any]:
    predicted, predicted_hash = read_npz_directory(predictions)
    truth, truth_hash = read_npz_directory(ground_truth)
    predicted_ids = set(predicted)
    truth_ids = set(truth)
    if predicted_ids != truth_ids:
        missing = sorted(truth_ids - predicted_ids)[:20]
        extra = sorted(predicted_ids - truth_ids)[:20]
        raise EvaluatorError(f"sequence mismatch; missing={missing}, extra={extra}")
    return {
        "sequence_count": len(predicted),
        "predictions": {"path": str(predictions.resolve()), "sha256": predicted_hash},
        "ground_truth": {"path": str(ground_truth.resolve()), "sha256": truth_hash},
        "sequences": predicted,
    }


def require_assets(data_root: Path, glove_root: Path, checkpoint: Path) -> Dict[str, Any]:
    required = [
        data_root / "t2m_mean_std_jpos.p",
        data_root / "omomo_text_anno_txt_data",
        glove_root / "our_vab_data.npy",
        glove_root / "our_vab_words.pkl",
        glove_root / "our_vab_idx.pkl",
        checkpoint,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise EvaluatorError("missing official evaluator assets: " + ", ".join(missing))
    if not checkpoint.is_file():
        raise EvaluatorError(f"checkpoint is not a file: {checkpoint}")
    return {
        "processed_omomo": {
            "path": str(data_root.resolve()),
            "mean_std_sha256": sha256_file(data_root / "t2m_mean_std_jpos.p"),
            "annotations_sha256": sha256_tree(data_root / "omomo_text_anno_txt_data"),
        },
        "glove": {
            "path": str(glove_root.resolve()),
            "files": {
                name: sha256_file(glove_root / name)
                for name in ("our_vab_data.npy", "our_vab_words.pkl", "our_vab_idx.pkl")
            },
        },
        "checkpoint": {"path": str(checkpoint.resolve()), "sha256": sha256_file(checkpoint)},
    }


def atomic_output(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise EvaluatorError(f"refusing to overwrite run specification: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def command_verify(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    print(json.dumps(verify_upstream(args.upstream.resolve(), config), indent=2, sort_keys=True))


def command_validate(args: argparse.Namespace) -> None:
    print(json.dumps(validate_pair(args.predictions, args.ground_truth), indent=2, sort_keys=True))


def command_prepare(args: argparse.Namespace) -> None:
    config = load_config(args.config)
    upstream = verify_upstream(args.upstream.resolve(), config)
    inputs = validate_pair(args.predictions.resolve(), args.ground_truth.resolve())
    assets = require_assets(args.data_root.resolve(), args.glove_root.resolve(), args.checkpoint.resolve())
    if not args.options_module.is_file():
        raise EvaluatorError(f"official options/train_options.py is missing: {args.options_module}")
    assets["options_module"] = {
        "path": str(args.options_module.resolve()),
        "sha256": sha256_file(args.options_module.resolve()),
    }
    value = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "evaluator_config_sha256": sha256_file(args.config),
        "upstream": upstream,
        "inputs": inputs,
        "assets": assets,
        "conversion": {
            "producer": "code/test_infbagel_hoi.py",
            "coordinate_transform": "yup_to_zup",
            "joint_layout": "24 SMPL joints",
        },
        "command_template": "cd <CHOIS>/t2m_eval && python final_evaluations.py",
    }
    atomic_output(args.output.resolve(), value)
    print(args.output.resolve())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify-upstream")
    verify.add_argument("--upstream", type=Path, required=True)
    verify.set_defaults(func=command_verify)
    validate = subparsers.add_parser("validate-inputs")
    validate.add_argument("--predictions", type=Path, required=True)
    validate.add_argument("--ground-truth", type=Path, required=True)
    validate.set_defaults(func=command_validate)
    prepare = subparsers.add_parser("prepare-run")
    prepare.add_argument("--upstream", type=Path, required=True)
    prepare.add_argument("--predictions", type=Path, required=True)
    prepare.add_argument("--ground-truth", type=Path, required=True)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--glove-root", type=Path, required=True)
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument(
        "--options-module", type=Path, required=True,
        help="official options/train_options.py omitted from the pinned Git release",
    )
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(func=command_prepare)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (EvaluatorError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
