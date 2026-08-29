#!/usr/bin/env python3
"""Build the fixed seed-42, scene-family-disjoint LINGO splits.

Three algorithms live here.  ``scene-family-disjoint-v1`` is the Phase 1A split
and is frozen: ``experiments/splits/lingo_scene_disjoint_seed42.json`` must stay
reproducible from it byte-for-byte, so nothing on that path may change.

``scene-family-disjoint-v2`` is the Phase 1C rebuild preregistered in
``docs/plan/PHASE_1C_HSI.md`` (2026-08-12, section 4).  It repairs the released
LINGO scene labels -- every mirrored sequence ships labelled ``005_mirror``
regardless of which of the 110 rooms it was captured in -- before partitioning,
and it carves out a third ``test`` partition from the scene families the
released InfBaGel checkpoint never trained on.

``scene-family-disjoint-v3`` keeps the mirror repair and rebalances all scene
families by eligible-window counts without consulting the released checkpoint.
"""

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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple


ALGORITHM = "scene-family-disjoint-v1"
ALGORITHM_V2 = "scene-family-disjoint-v2"
ALGORITHM_V3 = "scene-family-disjoint-v3"
DEFAULT_SEED = 42
DEFAULT_VAL_RATIO = 0.20
DEFAULT_V3_VAL_RATIO = 0.10
V3_TRAIN_RATIO = 0.70
V3_TEST_RATIO = 0.20
MIRROR_SUFFIX = "_mirror"
# ``code/datasets/infbagel_mix.py:305`` gates scene filtering on this literal.
BASELINE_SCENE_FILTER_THRESHOLD = 111
DEFAULT_BASELINE_CONFIG = Path(__file__).resolve().parents[1] / "code" / "config" / "config_train_infbagel_mix.yaml"
BASELINE_SELECTION_RULE = (
    "replicates code/datasets/infbagel_mix.py:298-329 -- scene_flag = "
    "scene_dict[scene_name[start_ind[window]]], scene_dict = "
    "{file[:-4]: sid for sid, file in enumerate(sorted(os.listdir(<root>/Scene)))} "
    "(code/datasets/infbagel.py:153-170, vis=false); all_scenes is the "
    "first-appearance order of scene_flag over window index and the released "
    "checkpoint trains on all_scenes[:lingo_scene_num]"
)
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


def _require_numpy():
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SplitError("numpy is required for scene-family-disjoint-v2") from exc
    return np


def read_lingo_scene_num(config_path: Path) -> int:
    """Read ``lingo_scene_num`` from the released mix training config."""
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SplitError("PyYAML is required to read the baseline config") from exc
    if not config_path.is_file():
        raise SplitError(f"missing baseline config: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or "lingo_scene_num" not in config:
        raise SplitError(f"baseline config has no lingo_scene_num: {config_path}")
    value = config["lingo_scene_num"]
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise SplitError(f"invalid lingo_scene_num {value!r} in {config_path}")
    return value


def verify_mirror_pairs(starts, ends, translations, half: int) -> Dict[str, Any]:
    """Prove sequence ``i + half`` is the exact x-mirror of sequence ``i``.

    Checked for every pair, never a sample.  Any failure raises ``SplitError``;
    there is no silent downgrade, because a single unverified pair would make the
    ``<src>_mirror`` relabel a guess rather than a measurement.
    """
    np = _require_numpy()
    length_failures: List[int] = []
    x_failures: List[int] = []
    yz_failures: List[int] = []
    degenerate_x = 0
    for index in range(half):
        source_start, source_end = int(starts[index]), int(ends[index])
        mirror_start, mirror_end = int(starts[index + half]), int(ends[index + half])
        if (source_end - source_start) != (mirror_end - mirror_start):
            length_failures.append(index)
            continue
        source = np.asarray(translations[source_start:source_end], dtype=np.float64)
        mirror = np.asarray(translations[mirror_start:mirror_end], dtype=np.float64)
        if not np.array_equal(mirror[:, 0], -source[:, 0]):
            x_failures.append(index)
        if not np.array_equal(mirror[:, 1:], source[:, 1:]):
            yz_failures.append(index)
        if not np.any(source[:, 0]):
            degenerate_x += 1
    if length_failures or x_failures or yz_failures:
        raise SplitError(
            "mirror pairing is not exact: "
            f"{len(length_failures)} length, {len(x_failures)} x-negation and "
            f"{len(yz_failures)} y/z failures; first offenders "
            f"{length_failures[:5]} {x_failures[:5]} {yz_failures[:5]}"
        )
    return {
        "rule": "sequence i + N//2 is the exact x-mirror of sequence i",
        "pairs_checked": int(half),
        "pairs_sampled": False,
        "length_equal_failures": 0,
        "x_exactly_negated_failures": 0,
        "yz_exactly_equal_failures": 0,
        "pairs_with_all_zero_source_x": int(degenerate_x),
    }


def verify_mirror_grids(scene_dir: Path, sources: Sequence[str]) -> Dict[str, Any]:
    """Every relabelled scene needs its own grid, and the grids must be distinct.

    The released defect was one label (``005_mirror``) standing for 110 rooms, so
    the assertion that matters is injectivity of label -> occupancy grid.
    """
    np = _require_numpy()
    labels = list(sources) + [f"{name}{MIRROR_SUFFIX}" for name in sources]
    missing = [name for name in labels if not (scene_dir / f"{name}.npy").is_file()]
    if missing:
        raise SplitError(f"missing occupancy grids for relabelled scenes: {missing[:10]}")
    digests = {name: sha256_file(scene_dir / f"{name}.npy") for name in labels}
    if len(set(digests.values())) != len(labels):
        collisions: Dict[str, List[str]] = {}
        for name, digest in digests.items():
            collisions.setdefault(digest, []).append(name)
        duplicated = sorted(names for names in collisions.values() if len(names) > 1)
        raise SplitError(f"scene labels are not injective over occupancy grids: {duplicated[:5]}")
    reversal_failures: List[str] = []
    for name in sources:
        source = np.load(scene_dir / f"{name}.npy")
        mirror = np.load(scene_dir / f"{name}{MIRROR_SUFFIX}.npy")
        if source.shape != mirror.shape or not np.array_equal(mirror, source[::-1]):
            reversal_failures.append(name)
    if reversal_failures:
        raise SplitError(f"mirror grids are not the x-reversed source grids: {reversal_failures[:10]}")
    return {
        "scene_dir_entries": len(sorted(os.listdir(scene_dir))),
        "labels_checked": len(labels),
        "distinct_grid_sha256": len(set(digests.values())),
        "grid_shape_axis0_reversal_failures": 0,
        "label_grid_table_sha256": sha256_json(sorted(digests.items())),
    }


def recompute_baseline_scenes(
    scene_dir: Path, scene_names: Sequence[str], window_starts, lingo_scene_num: int,
) -> Tuple[List[str], Dict[str, Any]]:
    """Recompute the released checkpoint's LINGO training scenes from the data.

    The scene list is never hardcoded: it is derived here by replaying the
    selection in ``code/datasets/infbagel_mix.py``.
    """
    np = _require_numpy()
    scene_files = sorted(os.listdir(scene_dir))
    scene_dict = {name[:-4]: sid for sid, name in enumerate(scene_files) if name.endswith(".npy")}
    if len(scene_dict) != len(scene_files):
        raise SplitError(f"non-.npy entries in {scene_dir}")
    sid_to_name = {sid: name for name, sid in scene_dict.items()}
    frame_labels = np.asarray(scene_names, dtype=object)
    window_labels = frame_labels[np.asarray(window_starts, dtype=np.int64)]
    unique_labels, inverse = np.unique(window_labels, return_inverse=True)
    unknown = [str(name) for name in unique_labels if str(name) not in scene_dict]
    if unknown:
        raise SplitError(f"window scene labels without a grid file: {unknown[:10]}")
    flags = np.asarray([scene_dict[str(name)] for name in unique_labels], dtype=np.int64)[inverse]
    # ``dict.setdefault`` insertion order over ascending window index is exactly
    # "unique values ordered by index of first occurrence".
    values, first_occurrence = np.unique(flags, return_index=True)
    all_scenes = [int(value) for value in values[np.argsort(first_occurrence)].tolist()]
    if len(all_scenes) != BASELINE_SCENE_FILTER_THRESHOLD:
        raise SplitError(
            f"expected {BASELINE_SCENE_FILTER_THRESHOLD} referenced scenes to replicate the "
            f"baseline selection guard, found {len(all_scenes)}"
        )
    if lingo_scene_num >= BASELINE_SCENE_FILTER_THRESHOLD:
        selected_flags = all_scenes
    else:
        selected_flags = all_scenes[:lingo_scene_num]
    selected = [sid_to_name[flag] for flag in selected_flags]
    statistics = {
        "config_key": "lingo_scene_num",
        "lingo_scene_num": int(lingo_scene_num),
        "rule": BASELINE_SELECTION_RULE,
        "referenced_scene_count": len(all_scenes),
        "scene_dir_entry_count": len(scene_files),
        "selected_scene_count": len(selected),
        "selected_scenes_in_selection_order": list(selected),
        "selected_scenes": sorted(selected),
    }
    return selected, statistics


def load_v2_inputs(root: Path, baseline_config: Path) -> Dict[str, Any]:
    """Read every released artifact the v2 rebuild depends on, and hash it."""
    np = _require_numpy()
    scene_path = root / "scene_name.pkl"
    start_path = root / "start_idx.npy"
    end_path = root / "end_idx.npy"
    transl_path = root / "transl_aligned.npy"
    left_path = root / "left_hand_inter_frame.npy"
    right_path = root / "right_hand_inter_frame.npy"
    language_path = root / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl"
    scene_dir = root / "Scene"
    required = (scene_path, start_path, end_path, transl_path, left_path, right_path, language_path)
    for path in required:
        if not path.is_file():
            raise SplitError(f"missing LINGO metadata: {path}")
    if not scene_dir.is_dir():
        raise SplitError(f"missing LINGO occupancy directory: {scene_dir}")
    with scene_path.open("rb") as handle:
        scene_names = pickle.load(handle)
    with language_path.open("rb") as handle:
        language = pickle.load(handle)
    inputs = {
        str(path.relative_to(root)): sha256_file(path) for path in required
    }
    inputs[str(baseline_config)] = sha256_file(baseline_config)
    return {
        "scene_names": scene_names,
        "starts": np.load(start_path).astype("int64"),
        "ends": np.load(end_path).astype("int64"),
        "translations": np.load(transl_path, mmap_mode="r"),
        "sequence_left_hand": np.load(left_path),
        "sequence_right_hand": np.load(right_path),
        "language": language,
        "scene_dir": scene_dir,
        "sha256": inputs,
    }


def load_v3_inputs(root: Path) -> Dict[str, Any]:
    """Read every released artifact the v3 rebuild depends on, and hash it."""
    np = _require_numpy()
    scene_path = root / "scene_name.pkl"
    start_path = root / "start_idx.npy"
    end_path = root / "end_idx.npy"
    transl_path = root / "transl_aligned.npy"
    left_path = root / "left_hand_inter_frame.npy"
    right_path = root / "right_hand_inter_frame.npy"
    language_path = root / "language_motion_dict" / "language_motion_dict__inter_and_loco__16.pkl"
    scene_dir = root / "Scene"
    required = (scene_path, start_path, end_path, transl_path, left_path, right_path, language_path)
    for path in required:
        if not path.is_file():
            raise SplitError(f"missing LINGO metadata: {path}")
    if not scene_dir.is_dir():
        raise SplitError(f"missing LINGO occupancy directory: {scene_dir}")
    with scene_path.open("rb") as handle:
        scene_names = pickle.load(handle)
    with language_path.open("rb") as handle:
        language = pickle.load(handle)
    inputs = {
        str(path.relative_to(root)): sha256_file(path) for path in required
    }
    return {
        "scene_names": scene_names,
        "starts": np.load(start_path).astype("int64"),
        "ends": np.load(end_path).astype("int64"),
        "translations": np.load(transl_path, mmap_mode="r"),
        "sequence_left_hand": np.load(left_path),
        "sequence_right_hand": np.load(right_path),
        "language": language,
        "scene_dir": scene_dir,
        "sha256": inputs,
    }


def relabel_mirrored_scenes(scene_names, starts, ends, half: int) -> Tuple[List[str], Dict[str, Any]]:
    """``scene(i + H) := scene(i) + "_mirror"`` after checking the premises."""
    source_labels = [str(scene_names[int(starts[index])]) for index in range(half)]
    already_mirrored = sorted({name for name in source_labels if name.endswith(MIRROR_SUFFIX)})
    if already_mirrored:
        raise SplitError(f"first-half labels already end in {MIRROR_SUFFIX}: {already_mirrored[:10]}")
    inconsistent = [
        index for index in range(half)
        if len({str(value) for value in scene_names[int(starts[index]):int(ends[index])]}) != 1
    ]
    if inconsistent:
        raise SplitError(
            f"{len(inconsistent)} first-half sequences span more than one scene label; "
            f"the per-sequence scene is ambiguous: {inconsistent[:5]}"
        )
    labels = list(source_labels) + [f"{name}{MIRROR_SUFFIX}" for name in source_labels]
    released = sorted({str(scene_names[int(starts[index])]) for index in range(half, 2 * half)})
    statistics = {
        "rule": 'scene(i + N//2) := scene(i) + "_mirror"',
        "half": int(half),
        "source_labels": len(set(source_labels)),
        "source_labels_ending_in_mirror": 0,
        "relabelled_sequences": int(half),
        "relabelled_scene_labels": len({f"{name}{MIRROR_SUFFIX}" for name in source_labels}),
        "first_half_sequences_spanning_multiple_labels": 0,
        "released_second_half_labels": released,
    }
    return labels, statistics


def build_split_v2(
    inputs: Mapping[str, Any], seed: int, val_ratio: float,
    explicit_families: Mapping[str, str], lingo_scene_num: int,
) -> Dict[str, Any]:
    np = _require_numpy()
    if not 0.0 < val_ratio < 1.0:
        raise SplitError("validation ratio must be between 0 and 1")
    scene_names = inputs["scene_names"]
    starts, ends = inputs["starts"], inputs["ends"]
    total = len(starts)
    if total != len(ends):
        raise SplitError("start_idx.npy and end_idx.npy disagree on sequence count")
    if total % 2 != 0:
        raise SplitError(f"sequence count {total} is odd; the mirrored half cannot be derived")
    half = total // 2

    mirror_statistics = verify_mirror_pairs(starts, ends, inputs["translations"], half)
    labels, relabel_statistics = relabel_mirrored_scenes(scene_names, starts, ends, half)
    sources = sorted(set(labels[:half]))
    grid_statistics = verify_mirror_grids(inputs["scene_dir"], sources)

    scene_families: Dict[str, str] = {}
    scenes_to_sequences: Dict[str, List[int]] = {}
    for index, scene in enumerate(labels):
        family = explicit_families.get(scene, infer_scene_family(scene))
        if scene in scene_families and scene_families[scene] != family:
            raise SplitError(f"scene {scene} maps to multiple families")
        scene_families[scene] = family
        scenes_to_sequences.setdefault(scene, []).append(index)
    for source in sources:
        mirror = f"{source}{MIRROR_SUFFIX}"
        if scene_families[mirror] != scene_families[source]:
            raise SplitError(
                f"{mirror} lands in family {scene_families[mirror]} but {source} is in "
                f"{scene_families[source]}; the mirror-side rule would split one room"
            )

    families = sorted(set(scene_families.values()))
    selected_scenes, baseline_statistics = recompute_baseline_scenes(
        inputs["scene_dir"], scene_names, inputs["language"]["start_idx"], lingo_scene_num,
    )
    touched = {explicit_families.get(name, infer_scene_family(name)) for name in selected_scenes}
    unknown_touched = sorted(touched - set(families))
    if unknown_touched:
        raise SplitError(f"baseline scenes map to families outside the dataset: {unknown_touched}")
    test_families = set(families) - touched
    if not test_families:
        raise SplitError("the baseline touches every scene family; no zero-shot test set exists")

    remaining = sorted(touched)
    if len(remaining) < 2:
        raise SplitError("at least two touched scene families are required for train/validation")
    shuffled = list(remaining)
    random.Random(seed).shuffle(shuffled)
    validation_count = min(len(remaining) - 1, max(1, round(len(remaining) * val_ratio)))
    validation_families = set(shuffled[:validation_count])
    train_families = set(remaining) - validation_families

    assignment = {
        "train": train_families, "validation": validation_families, "test": test_families,
    }
    partitions: Dict[str, Dict[str, Any]] = {}
    discarded_scenes: List[str] = []
    for partition, selected in assignment.items():
        if partition == "train":
            scenes = sorted(scene for scene, family in scene_families.items() if family in selected)
        else:
            scenes = sorted(
                scene for scene, family in scene_families.items()
                if family in selected and not scene.endswith(MIRROR_SUFFIX)
            )
            discarded_scenes.extend(
                scene for scene, family in scene_families.items()
                if family in selected and scene.endswith(MIRROR_SUFFIX)
            )
        sequences = sorted(index for scene in scenes for index in scenes_to_sequences[scene])
        partitions[partition] = {
            "scene_families": sorted(selected),
            "scenes": scenes,
            "sequence_ids": [str(index) for index in sequences],
        }
    discarded_scenes = sorted(discarded_scenes)
    discarded_sequences = sorted(index for scene in discarded_scenes for index in scenes_to_sequences[scene])
    discarded = {
        "reason": (
            "mirror sequences of validation/test families are dropped entirely; moving them "
            "to train would recreate the released leakage the v2 rebuild exists to remove"
        ),
        "scene_families": sorted({scene_families[scene] for scene in discarded_scenes}),
        "scenes": discarded_scenes,
        "sequence_ids": [str(index) for index in discarded_sequences],
    }

    _assert_v2_invariants(
        partitions, discarded, scene_families, families, set(selected_scenes), total,
    )

    counts = {
        "families": len(families),
        "scenes": len(scene_families),
        "sequences": int(total),
        "source_scenes": len(sources),
        "mirror_scenes": len(sources),
        "baseline_touched_families": len(touched),
        "baseline_untouched_families": len(test_families),
    }
    for partition in ("train", "validation", "test"):
        counts[f"{partition}_families"] = len(partitions[partition]["scene_families"])
        counts[f"{partition}_scenes"] = len(partitions[partition]["scenes"])
        counts[f"{partition}_sequences"] = len(partitions[partition]["sequence_ids"])
    counts["discarded_mirror_scenes"] = len(discarded["scenes"])
    counts["discarded_mirror_sequences"] = len(discarded["sequence_ids"])

    return {
        "algorithm": ALGORITHM_V2,
        "seed": seed,
        "validation_ratio": val_ratio,
        "validation_ratio_scope": "the scene families the baseline touches, not all families",
        "scene_to_family": dict(sorted(scene_families.items())),
        "counts": counts,
        "mirror_verification": mirror_statistics,
        "mirror_relabel": relabel_statistics,
        "scene_grid_verification": grid_statistics,
        "baseline_reference": baseline_statistics,
        "discarded_mirror": discarded,
        **partitions,
    }


def _assert_v2_invariants(
    partitions: Mapping[str, Mapping[str, Any]], discarded: Mapping[str, Any],
    scene_families: Mapping[str, str], families: Sequence[str],
    baseline_scenes: Optional[Set[str]], total: int,
) -> None:
    names = ("train", "validation", "test")
    for key, label in (("scene_families", "scene-family"), ("scenes", "scene"), ("sequence_ids", "sequence")):
        for left in range(len(names)):
            for right in range(left + 1, len(names)):
                overlap = set(partitions[names[left]][key]) & set(partitions[names[right]][key])
                if overlap:
                    raise SplitError(
                        f"{label} leakage between {names[left]} and {names[right]}: {sorted(overlap)[:10]}"
                    )
    assigned = [family for name in names for family in partitions[name]["scene_families"]]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(families):
        raise SplitError("every scene family must be assigned to exactly one partition")
    for partition in ("validation", "test"):
        mirrored = [scene for scene in partitions[partition]["scenes"] if scene.endswith(MIRROR_SUFFIX)]
        if mirrored:
            raise SplitError(f"{partition} must contain only non-mirrored scenes: {mirrored[:10]}")
        twins = [
            scene for scene in partitions[partition]["scenes"]
            if f"{scene}{MIRROR_SUFFIX}" in set(partitions["train"]["scenes"])
        ]
        if twins:
            raise SplitError(f"{partition} scenes whose mirror twin is in train: {twins[:10]}")
    if baseline_scenes is not None:
        leaked = set(partitions["test"]["scenes"]) & baseline_scenes
        if leaked:
            raise SplitError(f"test scenes the released baseline trained on: {sorted(leaked)[:10]}")
    if not set(discarded["scenes"]).isdisjoint(
        set().union(*(set(partitions[name]["scenes"]) for name in names))
    ):
        raise SplitError("discarded mirror scenes also appear in a partition")
    if not all(scene.endswith(MIRROR_SUFFIX) for scene in discarded["scenes"]):
        raise SplitError("the discard bucket may only hold mirror scenes")
    discard_families = set(discarded["scene_families"])
    allowed = set(partitions["validation"]["scene_families"]) | set(partitions["test"]["scene_families"])
    if not discard_families <= allowed:
        raise SplitError(f"discarded families outside validation/test: {sorted(discard_families - allowed)}")
    every = [index for name in names for index in partitions[name]["sequence_ids"]]
    every.extend(discarded["sequence_ids"])
    if len(every) != total or len(set(every)) != total:
        raise SplitError(
            f"partitions plus discards cover {len(set(every))} of {total} sequences exactly once"
        )
    unmapped = set(scene_families) - set().union(
        *(set(partitions[name]["scenes"]) for name in names), set(discarded["scenes"])
    )
    if unmapped:
        raise SplitError(f"scenes assigned to neither a partition nor the discard bucket: {sorted(unmapped)[:10]}")


def build_split_v3(
    inputs: Mapping[str, Any], seed: int, explicit_families: Mapping[str, str],
) -> Dict[str, Any]:
    np = _require_numpy()
    scene_names = inputs["scene_names"]
    starts, ends = inputs["starts"], inputs["ends"]
    total = len(starts)
    if total != len(ends):
        raise SplitError("start_idx.npy and end_idx.npy disagree on sequence count")
    if total % 2 != 0:
        raise SplitError(f"sequence count {total} is odd; the mirrored half cannot be derived")
    half = total // 2

    mirror_statistics = verify_mirror_pairs(starts, ends, inputs["translations"], half)
    labels, relabel_statistics = relabel_mirrored_scenes(scene_names, starts, ends, half)
    sources = sorted(set(labels[:half]))
    grid_statistics = verify_mirror_grids(inputs["scene_dir"], sources)

    scene_families: Dict[str, str] = {}
    scenes_to_sequences: Dict[str, List[int]] = {}
    sequence_families: List[str] = []
    for index, scene in enumerate(labels):
        family = explicit_families.get(scene, infer_scene_family(scene))
        if scene in scene_families and scene_families[scene] != family:
            raise SplitError(f"scene {scene} maps to multiple families")
        scene_families[scene] = family
        scenes_to_sequences.setdefault(scene, []).append(index)
        sequence_families.append(family)
    for source in sources:
        mirror = f"{source}{MIRROR_SUFFIX}"
        if scene_families[mirror] != scene_families[source]:
            raise SplitError(
                f"{mirror} lands in family {scene_families[mirror]} but {source} is in "
                f"{scene_families[source]}; the mirror-side rule would split one room"
            )

    families = sorted(set(scene_families.values()))
    language = inputs["language"]
    eligible = np.nonzero(
        (np.asarray(language["left_hand_inter_frame"]) == -1)
        & (np.asarray(language["right_hand_inter_frame"]) == -1)
    )[0]
    origins = np.asarray(language["ori_sequence_idx"])[eligible].astype("int64")
    source_family_windows = {family: 0 for family in families}
    for origin in origins.tolist():
        if origin < half:
            source_family_windows[sequence_families[origin]] += 1

    ratios = {
        "train": V3_TRAIN_RATIO,
        "validation": DEFAULT_V3_VAL_RATIO,
        "test": V3_TEST_RATIO,
    }
    source_total = sum(source_family_windows.values())
    targets = {name: source_total * ratio for name, ratio in ratios.items()}
    assigned_windows = {name: 0 for name in ratios}
    assignment = {name: set() for name in ratios}
    shuffled = list(families)
    random.Random(seed).shuffle(shuffled)
    shuffled.sort(key=lambda family: source_family_windows[family], reverse=True)
    for family in shuffled:
        partition = max(ratios, key=lambda name: targets[name] - assigned_windows[name])
        assignment[partition].add(family)
        assigned_windows[partition] += source_family_windows[family]

    partitions: Dict[str, Dict[str, Any]] = {}
    discarded_scenes: List[str] = []
    for partition, selected in assignment.items():
        if partition == "train":
            scenes = sorted(scene for scene, family in scene_families.items() if family in selected)
        else:
            scenes = sorted(
                scene for scene, family in scene_families.items()
                if family in selected and not scene.endswith(MIRROR_SUFFIX)
            )
            discarded_scenes.extend(
                scene for scene, family in scene_families.items()
                if family in selected and scene.endswith(MIRROR_SUFFIX)
            )
        sequences = sorted(index for scene in scenes for index in scenes_to_sequences[scene])
        partitions[partition] = {
            "scene_families": sorted(selected),
            "scenes": scenes,
            "sequence_ids": [str(index) for index in sequences],
        }
    if len(partitions["validation"]["scenes"]) < 12:
        raise SplitError("validation must contain at least 12 distinct non-mirror scenes")
    if len(partitions["test"]["scenes"]) < 25:
        raise SplitError("test must contain at least 25 distinct non-mirror scenes")
    discarded_scenes = sorted(discarded_scenes)
    discarded_sequences = sorted(index for scene in discarded_scenes for index in scenes_to_sequences[scene])
    discarded = {
        "reason": (
            "mirror sequences of validation/test families are dropped entirely; moving them "
            "to train would recreate the released leakage the v3 rebuild exists to remove"
        ),
        "scene_families": sorted({scene_families[scene] for scene in discarded_scenes}),
        "scenes": discarded_scenes,
        "sequence_ids": [str(index) for index in discarded_sequences],
    }

    _assert_v2_invariants(partitions, discarded, scene_families, families, None, total)

    counts = {
        "families": len(families),
        "scenes": len(scene_families),
        "sequences": int(total),
        "source_scenes": len(sources),
        "mirror_scenes": len(sources),
    }
    for partition in ("train", "validation", "test"):
        counts[f"{partition}_families"] = len(partitions[partition]["scene_families"])
        counts[f"{partition}_scenes"] = len(partitions[partition]["scenes"])
        counts[f"{partition}_sequences"] = len(partitions[partition]["sequence_ids"])
    counts["discarded_mirror_scenes"] = len(discarded["scenes"])
    counts["discarded_mirror_sequences"] = len(discarded["sequence_ids"])

    return {
        "algorithm": ALGORITHM_V3,
        "seed": seed,
        "validation_ratio": DEFAULT_V3_VAL_RATIO,
        "validation_ratio_scope": "all scene families",
        "ratio_basis": (
            "eligible source (non-mirror) windows; train mirrors are augmentation and are not targeted"
        ),
        "scene_to_family": dict(sorted(scene_families.items())),
        "counts": counts,
        "mirror_verification": mirror_statistics,
        "mirror_relabel": relabel_statistics,
        "scene_grid_verification": grid_statistics,
        "discarded_mirror": discarded,
        **partitions,
    }


def summarize_hand_filters(
    inputs: Mapping[str, Any], split: Mapping[str, Any],
) -> Dict[str, Any]:
    """Record both no-hand tallies: the Phase 1A window rule and the sequence rule."""
    np = _require_numpy()
    starts, ends = inputs["starts"], inputs["ends"]
    language = inputs["language"]
    rule = "left_hand_inter_frame == -1 and right_hand_inter_frame == -1"
    buckets = {
        name: set(split[name]["sequence_ids"]) for name in ("train", "validation", "test")
    }
    buckets["discarded_mirror"] = set(split["discarded_mirror"]["sequence_ids"])

    window_left = np.asarray(language["left_hand_inter_frame"])
    window_right = np.asarray(language["right_hand_inter_frame"])
    origin = np.asarray(language["ori_sequence_idx"])
    if not (len(origin) == len(window_left) == len(window_right)):
        raise SplitError("LINGO language arrays have inconsistent lengths")
    eligible = np.nonzero((window_left == -1) & (window_right == -1))[0]
    eligible_ids = [str(int(value)) for value in origin[eligible].tolist()]
    window_filter = {
        "rule": rule,
        "granularity": "window (language_motion_dict), the Phase 1A rule",
        "window_count": int(len(origin)),
        "eligible_window_count": int(len(eligible)),
        "excluded_window_count": int(len(origin) - len(eligible)),
    }
    for name, members in buckets.items():
        window_filter[f"eligible_{name}_window_count"] = sum(value in members for value in eligible_ids)

    sequence_nohand = (inputs["sequence_left_hand"] == -1) & (inputs["sequence_right_hand"] == -1)
    if len(sequence_nohand) != len(starts):
        raise SplitError("per-sequence hand arrays do not match the sequence count")
    lengths = (np.asarray(ends) - np.asarray(starts)).astype("int64")
    sequence_filter = {
        "rule": rule,
        "granularity": "sequence (left/right_hand_inter_frame.npy at the dataset root)",
        "note": (
            "the preregistered Phase 1C test-pool size uses this per-sequence rule; "
            "frames_exclusive is end_idx - start_idx, frames_inclusive adds one per "
            "sequence and is the convention the preregistration's frame tally used"
        ),
        "eligible_sequence_count": int(sequence_nohand.sum()),
    }
    for name, members in buckets.items():
        selected = [index for index in range(len(starts)) if str(index) in members and sequence_nohand[index]]
        frames = int(lengths[selected].sum()) if selected else 0
        sequence_filter[f"{name}_sequence_count"] = len(selected)
        sequence_filter[f"{name}_frames_exclusive"] = frames
        sequence_filter[f"{name}_frames_inclusive"] = frames + len(selected)
        sequence_filter[f"{name}_scene_count"] = len(set(split[name]["scenes"]))
    return {"dynamic_object_filter": window_filter, "no_hand_sequence_pool": sequence_filter}


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
    parser.add_argument(
        "--algorithm", choices=(ALGORITHM, ALGORITHM_V2, ALGORITHM_V3), default=ALGORITHM,
        help="v1 is frozen; v2 and v3 are mirror-repaired three-way splits",
    )
    parser.add_argument(
        "--baseline-config", type=Path, default=DEFAULT_BASELINE_CONFIG,
        help="config whose lingo_scene_num defines the released baseline's LINGO exposure (v2 only)",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main_v2(args: argparse.Namespace, family_map: Mapping[str, str]) -> int:
    if not args.dataset_root:
        raise SplitError("scene-family-disjoint-v2 requires --dataset-root")
    root = args.dataset_root.resolve()
    baseline_config = args.baseline_config.resolve()
    inputs = load_v2_inputs(root, baseline_config)
    lingo_scene_num = read_lingo_scene_num(baseline_config)
    split = build_split_v2(inputs, args.seed, args.validation_ratio, family_map, lingo_scene_num)
    extra = summarize_hand_filters(inputs, split)
    if args.compact:
        for partition in ("train", "validation", "test"):
            sequence_ids = split[partition].pop("sequence_ids")
            split[partition]["sequence_count"] = len(sequence_ids)
            split[partition]["sequence_ids_sha256"] = sha256_json(sequence_ids)
        sequence_ids = split["discarded_mirror"].pop("sequence_ids")
        split["discarded_mirror"]["sequence_count"] = len(sequence_ids)
        split["discarded_mirror"]["sequence_ids_sha256"] = sha256_json(sequence_ids)
    manifest = {
        "schema_version": 1,
        "dataset": "LINGO",
        "source": {
            "type": "lingo_dataset_root",
            "path": str(args.dataset_root),
            "baseline_config": str(args.baseline_config),
            "sha256": inputs["sha256"],
        },
        "explicit_family_map": dict(sorted(family_map.items())),
        **split,
        **extra,
    }
    atomic_write(args.output.resolve(), manifest)
    print(args.output.resolve())
    return 0


def main_v3(args: argparse.Namespace, family_map: Mapping[str, str]) -> int:
    if not args.dataset_root:
        raise SplitError("scene-family-disjoint-v3 requires --dataset-root")
    root = args.dataset_root.resolve()
    inputs = load_v3_inputs(root)
    split = build_split_v3(inputs, args.seed, family_map)
    extra = summarize_hand_filters(inputs, split)
    if args.compact:
        for partition in ("train", "validation", "test"):
            sequence_ids = split[partition].pop("sequence_ids")
            split[partition]["sequence_count"] = len(sequence_ids)
            split[partition]["sequence_ids_sha256"] = sha256_json(sequence_ids)
        sequence_ids = split["discarded_mirror"].pop("sequence_ids")
        split["discarded_mirror"]["sequence_count"] = len(sequence_ids)
        split["discarded_mirror"]["sequence_ids_sha256"] = sha256_json(sequence_ids)
    manifest = {
        "schema_version": 1,
        "dataset": "LINGO",
        "source": {
            "type": "lingo_dataset_root",
            "path": str(args.dataset_root),
            "sha256": inputs["sha256"],
        },
        "explicit_family_map": dict(sorted(family_map.items())),
        **split,
        **extra,
    }
    atomic_write(args.output.resolve(), manifest)
    print(args.output.resolve())
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        family_map = load_family_map(args.family_map)
        if args.algorithm == ALGORITHM_V2:
            return main_v2(args, family_map)
        if args.algorithm == ALGORITHM_V3:
            return main_v3(args, family_map)
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
