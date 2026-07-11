#!/usr/bin/env python3
"""Build the fixed seed-42, scene-family-disjoint LINGO 80/20 split."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ALGORITHM = "scene-family-disjoint-v1"
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.20
VARIANT_SUFFIX = re.compile(
    r"(?:[_-](?:mirror(?:ed)?|flip|new[_-]?loco(?:motion)?|"
    r"action[_-]?(?:variant[_-]?)?\d+|variant[_-]?\d+|aug[_-]?\d+))$",
    re.IGNORECASE,
)


class SplitError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def atomic_write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise SplitError(f"refusing to overwrite existing split: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(path))
    finally:
        if temporary.exists():
            temporary.unlink()


def infer_scene_family(scene_name: str) -> str:
    """Conservatively remove only registered augmentation suffixes."""
    family = scene_name.strip()
    # LINGO scene ids use the leading numeric token for a physical scene; text
    # after it names locomotion/action captures of that same geometry.
    numeric_family = re.match(r"^(\d+)(?:$|[-_])", family)
    if numeric_family:
        return numeric_family.group(1)
    previous = None
    while family != previous:
        previous = family
        family = VARIANT_SUFFIX.sub("", family)
    if not family:
        raise SplitError(f"scene name collapses to an empty family: {scene_name!r}")
    return family


def load_family_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in value.items()
    ):
        raise SplitError("family map must be a JSON object of scene_name -> family")
    return value


def load_records_json(path: Path) -> List[Dict[str, str]]:
    if path.suffix == ".jsonl":
        raw: Iterable[Any] = (
            json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    else:
        value = json.loads(path.read_text(encoding="utf-8"))
        raw = value["records"] if isinstance(value, dict) and "records" in value else value
    records = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or "sequence_id" not in item or "scene_name" not in item:
            raise SplitError(f"record {index} must contain sequence_id and scene_name")
        records.append({"sequence_id": str(item["sequence_id"]), "scene_name": str(item["scene_name"])})
    return records


def load_dataset_records(root: Path) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Any]]:
    """Read sequence-to-scene metadata without importing the training dataset."""
    try:
        import numpy as np
    except ImportError as exc:
        raise SplitError("numpy is required for --dataset-root") from exc

    scene_path = root / "scene_name.pkl"
    start_path = root / "start_idx.npy"
    for required in (scene_path, start_path):
        if not required.is_file():
            raise SplitError(f"missing LINGO metadata: {required}")
    with scene_path.open("rb") as handle:
        scene_names = pickle.load(handle)
    starts = np.load(start_path).astype("int64")
    records: List[Dict[str, str]] = []
    for sequence_id, start in enumerate(starts.tolist()):
        if start < 0 or start >= len(scene_names):
            raise SplitError(f"sequence {sequence_id} start index {start} is out of bounds")
        records.append({"sequence_id": str(sequence_id), "scene_name": str(scene_names[start])})

    inputs = {
        str(scene_path.relative_to(root)): sha256_file(scene_path),
        str(start_path.relative_to(root)): sha256_file(start_path),
    }
    extra: Dict[str, Any] = {}
    language_path = root / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl"
    if language_path.is_file():
        with language_path.open("rb") as handle:
            language = pickle.load(handle)
        required_keys = {"ori_sequence_idx", "left_hand_inter_frame", "right_hand_inter_frame"}
        if required_keys.issubset(language):
            ori = np.asarray(language["ori_sequence_idx"])
            left = np.asarray(language["left_hand_inter_frame"])
            right = np.asarray(language["right_hand_inter_frame"])
            if not (len(ori) == len(left) == len(right)):
                raise SplitError("LINGO language arrays have inconsistent lengths")
            eligible = np.nonzero((left == -1) & (right == -1))[0].astype("int64").tolist()
            extra["dynamic_object_filter"] = {
                "rule": "left_hand_inter_frame == -1 and right_hand_inter_frame == -1",
                "eligible_window_count": int(len(eligible)),
                "excluded_window_count": int(len(ori) - len(eligible)),
                "window_count": int(len(ori)),
            }
            extra["_eligible_original_sequence_ids"] = [str(int(value)) for value in ori[eligible].tolist()]
        inputs[str(language_path.relative_to(root))] = sha256_file(language_path)
    return records, inputs, extra


def build_split(
    records: Sequence[Mapping[str, str]], seed: int, val_ratio: float,
    explicit_families: Mapping[str, str],
) -> Dict[str, Any]:
    if not 0.0 < val_ratio < 1.0:
        raise SplitError("validation ratio must be between 0 and 1")
    seen_sequences = set()
    scenes_to_sequences: Dict[str, List[str]] = {}
    scene_families: Dict[str, str] = {}
    for record in records:
        sequence_id = str(record["sequence_id"])
        scene = str(record["scene_name"])
        if sequence_id in seen_sequences:
            raise SplitError(f"duplicate sequence id: {sequence_id}")
        seen_sequences.add(sequence_id)
        family = explicit_families.get(scene, infer_scene_family(scene))
        scenes_to_sequences.setdefault(scene, []).append(sequence_id)
        if scene in scene_families and scene_families[scene] != family:
            raise SplitError(f"scene {scene} maps to multiple families")
        scene_families[scene] = family

    families = sorted(set(scene_families.values()))
    if len(families) < 2:
        raise SplitError("at least two scene families are required")
    shuffled = list(families)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(families) - 1, max(1, round(len(families) * val_ratio)))
    validation_families = set(shuffled[:validation_count])

    partitions: Dict[str, Dict[str, Any]] = {}
    for partition, selected in (
        ("train", set(families) - validation_families),
        ("validation", validation_families),
    ):
        scenes = sorted(scene for scene, family in scene_families.items() if family in selected)
        sequences = sorted(
            (sequence for scene in scenes for sequence in scenes_to_sequences[scene]),
            key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value),
        )
        partitions[partition] = {
            "scene_families": sorted(selected),
            "scenes": scenes,
            "sequence_ids": sequences,
        }

    train = partitions["train"]
    validation = partitions["validation"]
    if set(train["scene_families"]) & set(validation["scene_families"]):
        raise SplitError("scene-family leakage detected")
    if set(train["scenes"]) & set(validation["scenes"]):
        raise SplitError("scene leakage detected")
    if set(train["sequence_ids"]) & set(validation["sequence_ids"]):
        raise SplitError("sequence leakage detected")

    return {
        "algorithm": ALGORITHM,
        "seed": seed,
        "validation_ratio": val_ratio,
        "scene_to_family": dict(sorted(scene_families.items())),
        "counts": {
            "families": len(families),
            "scenes": len(scene_families),
            "sequences": len(records),
            "train_sequences": len(train["sequence_ids"]),
            "validation_sequences": len(validation["sequence_ids"]),
        },
        **partitions,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--dataset-root", type=Path)
    source.add_argument("--records", type=Path, help="JSON/JSONL sequence_id,scene_name records")
    parser.add_argument("--family-map", type=Path)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--validation-ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument(
        "--compact", action="store_true",
        help="track scene lists plus hashes/counts rather than every sequence id",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        family_map = load_family_map(args.family_map)
        input_hashes: Dict[str, str]
        extra: Dict[str, Any]
        if args.dataset_root:
            root = args.dataset_root.resolve()
            records, input_hashes, extra = load_dataset_records(root)
            source = {"type": "lingo_dataset_root", "path": str(args.dataset_root), "sha256": input_hashes}
        else:
            records = load_records_json(args.records)
            input_hashes = {args.records.name: sha256_file(args.records)}
            extra = {}
            source = {"type": "records", "path": str(args.records.resolve()), "sha256": input_hashes}
        split = build_split(records, args.seed, args.validation_ratio, family_map)
        eligible_original = extra.pop("_eligible_original_sequence_ids", None)
        if eligible_original is not None:
            train_ids = set(split["train"]["sequence_ids"])
            validation_ids = set(split["validation"]["sequence_ids"])
            extra["dynamic_object_filter"]["eligible_train_window_count"] = sum(
                value in train_ids for value in eligible_original
            )
            extra["dynamic_object_filter"]["eligible_validation_window_count"] = sum(
                value in validation_ids for value in eligible_original
            )
        if args.compact:
            for partition in ("train", "validation"):
                sequence_ids = split[partition].pop("sequence_ids")
                split[partition]["sequence_count"] = len(sequence_ids)
                split[partition]["sequence_ids_sha256"] = sha256_json(sequence_ids)
        manifest = {
            "schema_version": 1,
            "dataset": "LINGO",
            "source": source,
            "explicit_family_map": dict(sorted(family_map.items())),
            **split,
            **extra,
        }
        atomic_write(args.output.resolve(), manifest)
    except (SplitError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
