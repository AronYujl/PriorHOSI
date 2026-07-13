#!/usr/bin/env python3
"""Create the preregistered Phase 1B sequence-disjoint OMOMO split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle
from pathlib import Path

import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_split(repo: Path, seed: int = 42, validation_ratio: float = 0.05) -> dict:
    if seed != 42 or validation_ratio != 0.05:
        raise ValueError("Phase 1B split is locked to seed 42 and validation ratio 0.05")
    root = repo / "data/train"
    names_path = root / "scene_name.pkl"
    language_path = root / "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl"
    names = pickle.loads(names_path.read_bytes())
    language = pickle.loads(language_path.read_bytes())
    if len(names) != 4304 or len(set(map(str, names))) != len(names):
        raise ValueError("expected 4,304 unique OMOMO train sequence names")
    ordered = sorted(
        range(len(names)),
        key=lambda index: (
            hashlib.sha256(f"{seed}:{names[index]}".encode("utf-8")).hexdigest(),
            str(names[index]),
            index,
        ),
    )
    validation_count = math.ceil(len(names) * validation_ratio)
    validation_indices = sorted(ordered[:validation_count])
    validation_set = set(validation_indices)
    train_indices = [index for index in range(len(names)) if index not in validation_set]
    window_sequences = np.asarray(language["ori_sequence_idx"], dtype=np.int64)

    def partition(indices: list[int]) -> dict:
        selected = set(indices)
        windows = int(np.count_nonzero(np.isin(window_sequences, list(selected))))
        return {
            "sequence_count": len(indices),
            "window_count": windows,
            "sequence_indices": indices,
            "sequence_names": [str(names[index]) for index in indices],
        }

    result = {
        "schema_version": 1,
        "algorithm": "omomo-sequence-sha256-seed42-v1",
        "seed": seed,
        "validation_ratio": validation_ratio,
        "rounding": "ceil",
        "source_contract_sha256": "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf",
        "source_audit_sha256": "1deea6a724a3319d4c5654da682d7f51af7e5c93b119d159bd2b37ad258f627f",
        "source_hashes": {
            str(names_path.relative_to(repo)): sha256_file(names_path),
            str(language_path.relative_to(repo)): sha256_file(language_path),
        },
        "train": partition(train_indices),
        "internal_validation": partition(validation_indices),
        "official_test_used_for_selection": False,
    }
    if result["train"]["window_count"] + result["internal_validation"]["window_count"] != len(window_sequences):
        raise AssertionError("not every train window was assigned exactly once")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    value = build_split(args.repo_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
