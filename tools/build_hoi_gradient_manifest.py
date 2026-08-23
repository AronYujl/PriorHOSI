#!/usr/bin/env python3
"""Build the deterministic Stage B HOI gradient-window manifest."""
import argparse
import hashlib
import math
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "code"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from tools.chois_evaluator import atomic_output, sha256_file
from priors.core.representation import REPRESENTATION
from priors.hoi.data import PriorWindowDataset

STRATA = ("S0", "S1", "S2", "S3", "S4")
BOTH_REFERENCE = 0.578568060199017
ENGAGED_REFERENCE = 0.8113585910646877
COVERAGE_TOLERANCE = 0.02


def classify_stratum(both_frames, engaged_frames):
    if engaged_frames == 0:
        return "S0"
    if both_frames == 0:
        return "S1"
    if both_frames * 2 < engaged_frames:
        return "S2"
    if both_frames < engaged_frames:
        return "S3"
    return "S4"


def largest_remainder(counts, total):
    corpus_total = sum(counts[name] for name in STRATA)
    if not corpus_total:
        raise ValueError("cannot allocate from an empty corpus")
    quotas = {name: total * counts[name] / corpus_total for name in STRATA}
    result = {name: math.floor(quotas[name]) for name in STRATA}
    order = sorted(STRATA, key=lambda name: (-(quotas[name] - result[name]), name))
    for name in order[:total - sum(result.values())]:
        result[name] += 1
    return result


def _coverage(records, both_reference, engaged_reference, tolerance):
    both = sum(record[2] for record in records)
    engaged = sum(record[3] for record in records)
    if not engaged:
        raise ValueError("E_COVERAGE_ZERO_ENGAGED_FRAMES: engaged-frame denominator is zero")
    both_fraction = both / engaged
    engaged_fraction = sum(record[3] > 0 for record in records) / len(records)
    both_deviation = abs(both_fraction - both_reference)
    engaged_deviation = abs(engaged_fraction - engaged_reference)
    return {
        "coverage": {
            "both_frame_fraction_of_engaged": both_fraction,
            "corpus_reference": both_reference,
            "absolute_deviation": both_deviation,
            "tolerance": tolerance,
            "accepted": both_deviation <= tolerance,
        },
        "allocation_quantization_check": {
            "engaged_window_fraction": engaged_fraction,
            "corpus_reference": engaged_reference,
            "absolute_deviation": engaged_deviation,
            "tolerance": tolerance,
            "accepted": engaged_deviation <= tolerance,
            "note": "Determined by the S0 stratum allocation; not a draw-representativeness check.",
        },
    }


def build_sampling_manifest(records, shards=4, windows_per_shard=256, cap_fraction=0.05,
                            both_reference=BOTH_REFERENCE,
                            engaged_reference=ENGAGED_REFERENCE,
                            tolerance=COVERAGE_TOLERANCE):
    """Pure sampling, allocation, intersection, and coverage logic."""
    records = [(int(i), str(s), int(b), int(e)) for i, s, b, e in records]
    if len({record[0] for record in records}) != len(records):
        raise ValueError("global window indices must be unique")
    by_stratum = {name: [] for name in STRATA}
    for record in records:
        by_stratum[classify_stratum(record[2], record[3])].append(record)
    counts = {name: len(by_stratum[name]) for name in STRATA}
    corpus_strata = {
        name: {"count": counts[name], "share": counts[name] / len(records)} for name in STRATA
    }
    target = largest_remainder(counts, windows_per_shard)
    cap = math.floor(windows_per_shard * cap_fraction)
    if cap < 1:
        raise ValueError("per-sequence cap rounds below one window")
    used = set()
    shard_results = []
    selected_records = []
    for shard_id in range(shards):
        per_sequence = Counter()
        actual = Counter()
        chosen = []
        for name in STRATA:
            ordered = sorted(
                by_stratum[name],
                key=lambda record: hashlib.sha256(
                    f"{shard_id}:{record[1]}:{record[0]}".encode()
                ).hexdigest(),
            )
            available_sequences = {record[1] for record in ordered if record[0] not in used}
            for record in ordered:
                if record[0] in used or per_sequence[record[1]] >= cap:
                    continue
                chosen.append(record)
                per_sequence[record[1]] += 1
                actual[name] += 1
                if actual[name] == target[name]:
                    break
            if actual[name] != target[name]:
                raise ValueError(
                    f"stratum {name} allocation {target[name]} selected {actual[name]}; "
                    f"distinct sequences available {len(available_sequences)}"
                )
        used.update(record[0] for record in chosen)
        selected_records.extend(chosen)
        shard_results.append({
            "shard_id": shard_id,
            "window_indices": sorted(record[0] for record in chosen),
            "sequence_count": len(per_sequence),
            "max_windows_from_one_sequence": max(per_sequence.values()),
            "windows_per_sequence_cap": cap,
            "strata": {name: {"target": target[name], "actual": actual[name]} for name in STRATA},
            **_coverage(chosen, both_reference, engaged_reference, tolerance),
        })
    pairs = {}
    index_sets = [set(shard["window_indices"]) for shard in shard_results]
    for left in range(shards):
        for right in range(left + 1, shards):
            pairs[f"{left}-{right}"] = len(index_sets[left] & index_sets[right])
    intersection = {
        "pairwise_intersection_sizes": pairs,
        "disjoint": all(size == 0 for size in pairs.values()),
        "union_size": len(set().union(*index_sets)),
    }
    checks = _coverage(selected_records, both_reference, engaged_reference, tolerance)
    def _gate(values, scope, suffix=""):
        for key, code, label in (
                ("coverage", "BOTH_FRACTION", "both-frame fraction"),
                ("allocation_quantization_check", "ENGAGED_WINDOW_QUANTIZATION",
                 "S0 allocation")):
            check = values[key]
            if not check["accepted"]:
                raise ValueError(
                    f"E_COVERAGE_{code}{suffix}: {scope} {label} deviation "
                    f"{check['absolute_deviation']} exceeds tolerance {check['tolerance']}"
                )
    _gate(checks, "union")
    for shard in shard_results:
        _gate(shard, f"shard {shard['shard_id']}", "_SHARD")
    return {"shards": shard_results, "corpus_strata": corpus_strata,
            "shard_intersection_check": intersection, **checks}


def _array_metadata(path):
    value = np.load(path, mmap_mode="r")
    return {"shape": list(value.shape), "file_size_bytes": path.stat().st_size,
            "array_nbytes": value.nbytes}


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, default=4)
    parser.add_argument("--windows-per-shard", type=int, default=256)
    parser.add_argument("--config-name", default="config_train_hoi_prior_p12",
                        help="recorded for provenance only; Hydra is not initialized")
    parser.add_argument("--split-manifest", type=Path,
                        default=Path("experiments/splits/omomo_hoi_train_validation_seed42.json"))
    return parser.parse_args()


def main():
    args = _parse_args()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite gradient manifest: {output}")
    split_path = args.split_manifest if args.split_manifest.is_absolute() else ROOT / args.split_manifest
    start = time.time()
    dataset = PriorWindowDataset(str(ROOT), "hoi", partition="train", limit=0,
                                 split_manifest=str(split_path))
    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False, num_workers=8)
    records = []
    history = REPRESENTATION.history_frames
    for batch in loader:
        contact = batch["x"][:, history:, 228:230] > 0.5
        left, right = contact[..., 0], contact[..., 1]
        left_only = (left & ~right).sum(dim=1)
        right_only = (right & ~left).sum(dim=1)
        both = (left & right).sum(dim=1).tolist()
        neither = (~left & ~right).sum(dim=1)
        if not torch.all(left_only + right_only + torch.as_tensor(both) + neither == left.shape[1]):
            raise RuntimeError("contact categories do not partition active frames")
        engaged = (left_only + right_only + torch.as_tensor(both)).tolist()
        records.extend(zip(batch["window_index"].tolist(), batch["sequence_index"].tolist(),
                           both, engaged))
    if len(records) != len(dataset):
        raise RuntimeError(f"consumed {len(records)} windows, dataset holds {len(dataset)}")
    sampled = build_sampling_manifest(records, args.shards, args.windows_per_shard)
    arrays = ("human_joints_aligned.npy", "human_orient.npy", "human_pose.npy")
    result = {
        "probe": "hoi_gradient_window_manifest",
        "window_index_space": "language_motion_dict_window_index",
        "sampling": {"shards": args.shards, "windows_per_shard": args.windows_per_shard,
                     "windows_total": args.shards * args.windows_per_shard,
                     "per_sequence_cap_fraction": 0.05},
        "dataset_config_fingerprint": {
            "split_manifest_sha256": sha256_file(split_path),
            "norm_sha256": sha256_file(ROOT / "data/train/norm.npy"),
            "history_frames": history, "contact_channels": [228, 229],
            "contact_threshold": 0.5, "total_windows": len(dataset),
            "total_sequences": len(set(dataset.sequence_ids[dataset.indices].tolist())),
            "config_name": args.config_name,
            "mmap_arrays": {name: _array_metadata(ROOT / "data/train" / name) for name in arrays},
        },
        **sampled,
        "provenance": {
            "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                                   text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"],
                                                       cwd=ROOT, text=True).strip()),
            "tool_sha256": sha256_file(Path(__file__).resolve()),
            "python_version": sys.version.split()[0], "numpy_version": np.__version__,
            "torch_version": torch.__version__,
        },
    }
    atomic_output(output, result)
    print(f"elapsed_seconds={time.time() - start:.2f}")


if __name__ == "__main__":
    main()
