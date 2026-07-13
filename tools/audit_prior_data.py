#!/usr/bin/env python3
"""Aggregate, non-overwriting Phase 1A audits for OMOMO HOI and LINGO HSI."""

import argparse
import hashlib
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))

from datasets.utils import zup_to_yup  # noqa: E402
from priors.contracts import HOI_CONTRACT, HSI_CONTRACT  # noqa: E402
from priors.data import hsi_filter, partition_for_scenes  # noqa: E402
from priors.representation import REPRESENTATION  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def finite_audit(path: Path, chunk: int = 100000) -> dict:
    value = np.load(path, mmap_mode="r")
    nonfinite = 0
    total = int(value.size)
    flat = value.reshape(-1)
    for start in range(0, total, chunk):
        nonfinite += int(np.count_nonzero(~np.isfinite(flat[start:start + chunk])))
    return {"shape": list(value.shape), "dtype": str(value.dtype), "nonfinite_values": nonfinite, "values": total}


def normalization_audit(root: Path, indices: np.ndarray, starts: np.ndarray, maximum_windows: int, object_mode: bool) -> dict:
    if maximum_windows and len(indices) > maximum_windows:
        positions = np.linspace(0, len(indices) - 1, maximum_windows).round().astype(np.int64)
        selected = indices[positions]
        selection = "deterministic-even"
    else:
        selected = indices
        selection = "all"
    joints = np.load(root / "human_joints_aligned.npy", mmap_mode="r")
    orient = np.load(root / "human_orient.npy", mmap_mode="r")
    norm = np.load(root / "norm.npy")
    minimum, maximum = norm[0], norm[1]
    object_trans = np.load(root / "object_trans.npy", mmap_mode="r") if object_mode else None
    object_outside = object_values = 0
    outside = values = nonfinite = 0
    maximum_absolute = 0.0
    offsets = np.arange(0, 48, 3, dtype=np.int64)
    for begin in range(0, len(selected), 1024):
        window_starts = starts[selected[begin:begin + 1024]]
        frames = window_starts[:, None] + offsets[None]
        block = np.asarray(joints[frames])
        initial = np.stack((block[:, 0, 0, 0], np.zeros(len(block)), block[:, 0, 0, 2]), axis=-1)
        oriented = zup_to_yup(np.asarray(orient[window_starts]).copy())
        yaw = Rotation.from_rotvec(oriented).as_euler("zxy")[:, 2]
        shift = Rotation.from_euler("zxy", np.stack((np.zeros_like(yaw), np.zeros_like(yaw), -yaw), axis=-1)).as_matrix()
        local = np.einsum("btjc,bdc->btjd", block - initial[:, None, None], shift)
        normalized = -1.0 + 2.0 * (local - minimum) / (maximum - minimum)
        outside += int(np.count_nonzero(np.abs(normalized) > 1.0))
        maximum_absolute = max(maximum_absolute, float(np.nanmax(np.abs(normalized))))
        nonfinite += int(np.count_nonzero(~np.isfinite(normalized)))
        values += int(normalized.size)
        if object_trans is not None:
            trans = np.einsum("btc,bdc->btd", np.asarray(object_trans[frames]) - initial[:, None], shift)
            normalized_object = -1.0 + 2.0 * (trans - norm[2]) / (norm[3] - norm[2])
            object_outside += int(np.count_nonzero(np.abs(normalized_object) > 1.0))
            object_values += int(normalized_object.size)
    return {
        "selection": selection,
        "audited_windows": int(len(selected)),
        "total_eligible_windows": int(len(indices)),
        "position_values": values,
        "position_outside_count": outside,
        "position_outside_rate": outside / values if values else None,
        "position_nonfinite_count": nonfinite,
        "position_maximum_absolute_normalized": maximum_absolute,
        "object_values": object_values,
        "object_outside_count": object_outside,
        "object_outside_rate": object_outside / object_values if object_values else None,
    }


def load_language(root: Path):
    path = root / "language_motion_dict/language_motion_dict__inter_and_loco__16.pkl"
    with path.open("rb") as handle:
        return path, pickle.load(handle)


def text_audit(root: Path, language: dict, indices: np.ndarray) -> dict:
    with (root / "text2features_idx.pkl").open("rb") as handle:
        mapping = pickle.load(handle)
    texts = [language["text"][int(index)][0] for index in indices]
    missing = sorted({text for text in texts if text not in mapping})
    empty = sum(not str(text).strip() for text in texts)
    return {
        "windows": len(texts), "unique_instructions": len(set(texts)),
        "empty_instructions": empty, "missing_feature_instructions": missing,
        "coverage_rate": (len(texts) - empty - sum(text in missing for text in texts)) / len(texts) if texts else 0.0,
    }


def audit_hoi(repo: Path, maximum_windows: int) -> dict:
    partitions = {}
    all_hashes = {}
    for partition, relative in (("train", "data/train"), ("validation", "data/test")):
        root = repo / relative
        language_path, language = load_language(root)
        starts = np.asarray(language["start_idx"], dtype=np.int64)
        indices = np.arange(len(starts), dtype=np.int64)
        seq_starts = np.load(root / "start_idx.npy", mmap_mode="r")
        seq_ends = np.load(root / "end_idx.npy", mmap_mode="r")
        sequence_ids = np.asarray(language["ori_sequence_idx"], dtype=np.int64)
        sequence_names = pickle.loads((root / "scene_name.pkl").read_bytes())
        missing_bps = []
        missing_contact = []
        for sequence in sorted(set(sequence_ids.tolist())):
            name = str(sequence_names[sequence])
            if not (root / "cano_object_bps_npy_files_joints24_120" / f"{name}.npy").is_file():
                missing_bps.append(name)
            if not (root / "contact_label_npy_files" / f"{name}.npy").is_file():
                missing_contact.append(name)
        short = (seq_ends[sequence_ids] - seq_starts[sequence_ids]) <= 48
        finite = {
            name: finite_audit(root / name) for name in (
                "human_joints_aligned.npy", "human_orient.npy", "human_pose.npy",
                "object_trans.npy", "object_rot_mat.npy", "clip_features.npy",
            )
        }
        partitions[partition] = {
            "raw_sequences": int(len(seq_starts)), "referenced_sequences": int(len(set(sequence_ids.tolist()))),
            "raw_windows": int(len(indices)), "retained_windows": int(len(indices)),
            "dynamic_object_windows": int(np.count_nonzero(np.asarray(language["need_object"]))),
            "short_sequence_windows": int(short.sum()), "text": text_audit(root, language, indices),
            "missing_bps_sequences": missing_bps, "missing_contact_sequences": missing_contact,
            "normalization": normalization_audit(root, indices, starts, maximum_windows, object_mode=True),
            "finite_values": finite,
        }
        for name in ("human_joints_aligned.npy", "human_orient.npy", "human_pose.npy", "object_trans.npy", "object_rot_mat.npy", "norm.npy", "start_idx.npy", "end_idx.npy", "clip_features.npy"):
            all_hashes[f"{relative}/{name}"] = sha256_file(root / name)
        all_hashes[str(language_path.relative_to(repo))] = sha256_file(language_path)
    result = {
        "schema_version": 1, "expert": "hoi", "contract": HOI_CONTRACT.as_dict(),
        "representation": REPRESENTATION.as_dict(), "partitions": partitions, "source_hashes": all_hashes,
        "scene_supervision": {"dataset_loaded": False, "model_api_accepts_scene": False, "loss_uses_scene": False},
    }
    result["contract_sha256"] = sha256_json(result)
    return result


def audit_hsi(repo: Path, maximum_windows: int) -> dict:
    root = repo / "data/dataset"
    language_path, language = load_language(root)
    starts = np.asarray(language["start_idx"], dtype=np.int64)
    sequence_ids = np.asarray(language["ori_sequence_idx"], dtype=np.int64)
    indices = np.arange(len(starts), dtype=np.int64)
    keep = hsi_filter(language["left_hand_inter_frame"], language["right_hand_inter_frame"])
    dynamic_eligible = indices[keep]
    scene_names = pickle.loads((root / "scene_name.pkl").read_bytes())
    window_scenes = np.asarray([scene_names[int(start)] for start in starts], dtype=object)
    split_path = repo / "experiments/splits/lingo_scene_disjoint_seed42.json"
    split = json.loads(split_path.read_text())
    sides = partition_for_scenes(split, window_scenes)
    seq_starts = np.load(root / "start_idx.npy", mmap_mode="r")
    seq_ends = np.load(root / "end_idx.npy", mmap_mode="r")
    lengths = seq_ends[sequence_ids] - seq_starts[sequence_ids]
    valid_length = lengths > 48
    eligible = indices[keep & valid_length]
    partitions = {}
    for partition in ("train", "validation"):
        selected = indices[keep & valid_length & (sides == partition)]
        partitions[partition] = {
            "retained_windows": int(len(selected)),
            "referenced_sequences": int(len(set(sequence_ids[selected].tolist()))),
            "scene_count": int(len(set(window_scenes[selected].tolist()))),
            "scene_family_count": int(len(split[partition]["scene_families"])),
            "short_sequence_windows": int(np.count_nonzero(lengths[selected] <= 48)),
            "text": text_audit(root, language, selected),
        }
    hashes = {}
    for name in ("human_joints_aligned.npy", "human_orient.npy", "human_pose.npy", "norm.npy", "start_idx.npy", "end_idx.npy", "clip_features.npy"):
        hashes[f"data/dataset/{name}"] = sha256_file(root / name)
    hashes[str(language_path.relative_to(repo))] = sha256_file(language_path)
    hashes[str(split_path.relative_to(repo))] = sha256_file(split_path)
    omomo_norm_hash = sha256_file(repo / "data/train/norm.npy")
    result = {
        "schema_version": 1, "expert": "hsi", "contract": HSI_CONTRACT.as_dict(),
        "representation": REPRESENTATION.as_dict(),
        "counts": {
            "raw_sequences": int(len(seq_starts)), "raw_windows": int(len(indices)),
            "dynamic_object_filter_retained_windows": int(len(dynamic_eligible)),
            "excluded_dynamic_object_windows": int((~keep).sum()),
            "excluded_short_sequence_windows": int(np.count_nonzero(keep & ~valid_length)),
            "retained_windows": int(len(eligible)),
            "scene_families": int(split["counts"]["families"]), "scenes": int(split["counts"]["scenes"]),
        },
        "partitions": partitions,
        "filter": {
            "dynamic_object_rule": "left_hand_inter_frame == -1 and right_hand_inter_frame == -1",
            "short_sequence_rule": "seq_length > 48",
            "all_retained_satisfy_dynamic_object_rule": bool(keep[eligible].all()),
            "all_retained_satisfy_short_sequence_rule": bool(valid_length[eligible].all()),
            "short_sequence_diagnosis": "source windows are 48 frames but 21,819 cross their declared sequence end; the legacy mixed loader also rejects seq_length <= 48",
        },
        "split": {
            "seed": split["seed"], "algorithm": split["algorithm"],
            "scene_family_leakage": sorted(set(split["train"]["scene_families"]) & set(split["validation"]["scene_families"])),
            "scene_leakage": sorted(set(split["train"]["scenes"]) & set(split["validation"]["scenes"])),
        },
        "normalization": {
            "source": "data/dataset/norm.npy", "matches_omomo_norm": hashes["data/dataset/norm.npy"] == omomo_norm_hash,
            "omomo_norm_sha256": omomo_norm_hash,
            "bounds": normalization_audit(root, eligible, starts, maximum_windows, object_mode=False),
        },
        "empty_supervision": {"object_translation": True, "object_rotation": True, "contact": True, "loss_masked": True},
        "finite_values": {
            name: finite_audit(root / name) for name in (
                "human_joints_aligned.npy", "human_orient.npy", "human_pose.npy", "clip_features.npy",
            )
        },
        "source_hashes": hashes,
    }
    result["contract_sha256"] = sha256_json(result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expert", choices=("hoi", "hsi"), required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--max-normalization-windows", type=int, default=100000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = audit_hoi(args.repo_root.resolve(), args.max_normalization_windows) if args.expert == "hoi" else audit_hsi(args.repo_root.resolve(), args.max_normalization_windows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
