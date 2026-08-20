#!/usr/bin/env python3
"""Run the pinned CHOIS text-motion metric without modifying its checkout.

CHOIS releases the metric implementation but omits ``options/train_options.py``
and hard-codes paths from the authors' workstation.  This adapter imports the
missing parser from a pinned public dependency and reproduces the released
dataset/metric logic with explicit paths.  It deliberately does not edit the
CHOIS checkout, so its provenance check remains meaningful.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import random
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence

import joblib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data._utils.collate import default_collate

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import chois_evaluator


DEFAULT_CHOIS_CONFIG = ROOT / "experiments" / "evaluators" / "chois_omomo.json"
DEFAULT_T2M_CONFIG = ROOT / "experiments" / "evaluators" / "text_to_motion.json"


class AdapterError(RuntimeError):
    """Raised for an unreproducible evaluator invocation."""


def _sha256(path: Path) -> str:
    return chois_evaluator.sha256_file(path)


def _git_output(root: Path, *args: str) -> str:
    return chois_evaluator.git_output(root, *args)


def verify_text_to_motion(root: Path, config: Mapping[str, Any]) -> Dict[str, Any]:
    """Verify the separately pinned source of CHOIS's omitted imports."""
    if not (root / ".git").exists():
        raise AdapterError(f"not a text-to-motion Git checkout: {root}")
    commit = _git_output(root, "rev-parse", "HEAD")
    if commit != config["upstream_commit"]:
        raise AdapterError(
            f"text-to-motion commit mismatch: expected {config['upstream_commit']}, got {commit}"
        )
    if _git_output(root, "status", "--porcelain"):
        raise AdapterError("text-to-motion checkout is dirty")
    files: Dict[str, str] = {}
    for relative, expected in config["files"].items():
        path = root / relative
        if not path.is_file():
            raise AdapterError(f"missing text-to-motion dependency file: {relative}")
        actual = _sha256(path)
        if actual != expected:
            raise AdapterError(f"text-to-motion hash mismatch for {relative}: {actual}")
        files[relative] = actual
    return {"commit": commit, "files": files}


@contextmanager
def import_paths(chois_root: Path, text_to_motion_root: Path) -> Iterator[None]:
    """Prioritise CHOIS modules while supplying only its missing dependency."""
    original = list(sys.path)
    try:
        sys.path.insert(0, str(text_to_motion_root))
        sys.path.insert(0, str(chois_root / "t2m_eval"))
        importlib.invalidate_caches()
        yield
    finally:
        sys.path[:] = original


class PathConfiguredCHOISEvaluationDataset(Dataset):
    """The released CHOIS evaluation dataset with paths passed explicitly.

    This mirrors ``t2m_eval/motion_loaders/chois_eval_dataset.py``.  The
    released implementation cannot be instantiated outside the authors'
    workstation because it embeds ``/move/u/.../processed_data``.  No samples,
    normalization, token processing, or metric computation are changed here.
    """

    def __init__(
        self,
        results_dir: Path,
        data_root: Path,
        word_vectorizer: Any,
        *,
        global_offset: Optional[np.ndarray] = None,
    ):
        self.results_dir = results_dir
        self.word_vectorizer = word_vectorizer
        if global_offset is None:
            self.global_offset = None
        else:
            value = np.asarray(global_offset, dtype=np.float64)
            if value.shape != (3,) or not np.isfinite(value).all():
                raise AdapterError("global offset diagnostic requires one finite 3-vector")
            self.global_offset = value
        self.language_annotations = data_root / "omomo_text_anno_txt_data"
        mean_std_path = data_root / "t2m_mean_std_jpos.p"
        if not mean_std_path.is_file():
            raise AdapterError(f"missing CHOIS mean/std asset: {mean_std_path}")
        values = joblib.load(mean_std_path)
        try:
            self.mean_jpos = torch.from_numpy(values["jpos_mean"]).float()
            self.std_jpos = torch.from_numpy(values["jpos_std"]).float()
        except (KeyError, TypeError) as exc:
            raise AdapterError(f"invalid CHOIS mean/std asset: {mean_std_path}") from exc
        self.samples = self._load_results()

    def _load_results(self) -> list[Dict[str, Any]]:
        sequences, _ = chois_evaluator.read_npz_directory(self.results_dir)
        samples: list[Dict[str, Any]] = []
        for seq_name, metadata in sorted(sequences.items()):
            path = self.results_dir / metadata["file"]
            with np.load(path, allow_pickle=False) as npz:
                samples.append({
                    "global_jpos": npz["global_jpos"],
                    "seq_name": str(npz["seq_name"].item()),
                })
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def sequence_ids(self) -> list[str]:
        return [str(sample["seq_name"]) for sample in self.samples]

    def _annotation(self, sequence_name: str) -> tuple[str, list[str]]:
        path = self.language_annotations / f"{sequence_name}.txt"
        if not path.is_file():
            raise AdapterError(f"missing CHOIS annotation for {sequence_name}: {path}")
        last: Optional[tuple[str, list[str]]] = None
        for line in path.read_text(encoding="utf-8").splitlines():
            fields = line.strip().split("#")
            if len(fields) < 2:
                raise AdapterError(f"invalid CHOIS annotation: {path}")
            last = (fields[0], fields[1].split(" "))
        if last is None:
            raise AdapterError(f"empty CHOIS annotation: {path}")
        return last

    def __getitem__(self, index: int) -> tuple[Any, ...]:
        sample = self.samples[index]
        global_jpos = sample["global_jpos"]
        if self.global_offset is not None:
            global_jpos = np.asarray(global_jpos, dtype=np.float64) - self.global_offset
        motion = torch.from_numpy(global_jpos.reshape(-1, 72)).float()
        motion = (motion - self.mean_jpos[None, :]) / self.std_jpos[None, :]
        caption, tokens = self._annotation(sample["seq_name"])
        if len(tokens) < 30:
            tokens = ["sos/OTHER"] + tokens + ["eos/OTHER"]
            sent_len = len(tokens)
            tokens = tokens + ["unk/OTHER"] * (32 - sent_len)
        else:
            tokens = ["sos/OTHER"] + tokens[:30] + ["eos/OTHER"]
            sent_len = len(tokens)
        word_embeddings, pos_one_hots = [], []
        for token in tokens:
            word, pos = self.word_vectorizer[token]
            word_embeddings.append(word[None, :])
            pos_one_hots.append(pos[None, :])
        return (
            np.concatenate(word_embeddings, axis=0),
            np.concatenate(pos_one_hots, axis=0),
            caption,
            sent_len,
            motion,
            motion.shape[0],
            "_".join(tokens),
            sample["seq_name"],
        )


def _collate(batch: list[tuple[Any, ...]]) -> Any:
    batch.sort(key=lambda item: item[3], reverse=True)
    return default_collate(batch)


def _loader(dataset: Dataset, batch_size: int, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=_collate,
        drop_last=True,
        num_workers=workers,
    )


def _make_wrapper_options(
    text_to_motion_root: Path, checkpoints_dir: Path, dataset_name: str, device: torch.device
) -> Any:
    with import_paths(Path("."), text_to_motion_root):
        from options.train_options import TrainTexMotMatchOptions

        previous_argv = list(sys.argv)
        try:
            sys.argv = [
                "chois-evaluator-adapter",
                "--checkpoints_dir", str(checkpoints_dir),
                "--dataset_name", dataset_name,
                "--gpu_id", "-1",
            ]
            options = TrainTexMotMatchOptions().parse()
        finally:
            sys.argv = previous_argv
    options.device = device
    return options


def _load_components(
    chois_root: Path, text_to_motion_root: Path, glove_root: Path,
    checkpoints_dir: Path, dataset_name: str, device: torch.device,
) -> tuple[Any, Any, Any]:
    with import_paths(chois_root, text_to_motion_root):
        from networks.evaluator_wrapper import EvaluatorModelWrapper
        from utils.metrics import (
            calculate_activation_statistics,
            calculate_diversity,
            calculate_frechet_distance,
            calculate_top_k,
            euclidean_distance_matrix,
        )
        from utils.word_vectorizer import WordVectorizer

        options = _make_wrapper_options(text_to_motion_root, checkpoints_dir, dataset_name, device)
        return (
            EvaluatorModelWrapper(options),
            WordVectorizer(str(glove_root), "our_vab"),
            {
                "activation_statistics": calculate_activation_statistics,
                "diversity": calculate_diversity,
                "frechet": calculate_frechet_distance,
                "top_k": calculate_top_k,
                "distance": euclidean_distance_matrix,
            },
        )


def _embeddings(
    loader: DataLoader, wrapper: Any, metrics: Mapping[str, Any], matching: bool,
    *, record_permutation: bool = False,
) -> Dict[str, Any]:
    all_motion_embeddings = []
    all_sequence_ids: list[str] = []
    row_permutations: list[list[int]] = []
    matching_distances = []
    r_precision_rows = []
    matching_score_sum = 0.0
    top_k_count = np.zeros(3)
    all_size = 0
    with torch.no_grad():
        for batch in loader:
            (
                word_embeddings, pos_one_hots, _, sent_lens, motions, motion_lengths,
                _, sequence_ids,
            ) = batch
            all_sequence_ids.extend(str(value) for value in sequence_ids)
            if record_permutation:
                row_permutations.append(_row_permutation_replay(motion_lengths))
            if matching:
                text_embeddings, motion_embeddings = wrapper.get_co_embeddings(
                    word_embs=word_embeddings,
                    pos_ohot=pos_one_hots,
                    cap_lens=sent_lens,
                    motions=motions,
                    m_lens=motion_lengths,
                )
                distance = metrics["distance"](
                    text_embeddings.cpu().numpy(), motion_embeddings.cpu().numpy()
                )
                batch_matching_distances = np.diag(distance)
                batch_r_precision = metrics["top_k"](
                    np.argsort(distance, axis=1), top_k=3,
                )
                matching_distances.append(batch_matching_distances)
                r_precision_rows.append(batch_r_precision)
                matching_score_sum += float(batch_matching_distances.sum())
                top_k_count += batch_r_precision.sum(axis=0)
                all_size += text_embeddings.shape[0]
            else:
                motion_embeddings = wrapper.get_motion_embeddings(motions=motions, m_lens=motion_lengths)
            all_motion_embeddings.append(motion_embeddings.cpu().numpy())
    if not all_motion_embeddings:
        raise AdapterError("empty evaluation loader after batching")
    result: Dict[str, Any] = {
        "motion_embeddings": np.concatenate(all_motion_embeddings, axis=0),
        "sequence_ids": all_sequence_ids,
    }
    if record_permutation:
        result["row_permutation"] = _row_permutation_summary(row_permutations)
    if matching:
        result["matching_score"] = matching_score_sum / all_size
        result["r_precision"] = (top_k_count / all_size).tolist()
        result["matching_distances"] = np.concatenate(matching_distances, axis=0)
        result["r_precision_rows"] = np.concatenate(r_precision_rows, axis=0)
    return result


def _sha256_ids(sequence_ids: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{sequence_id}\n" for sequence_id in sequence_ids).encode("utf-8")
    ).hexdigest()


def _global_mean_bias_vector(
    truth_dataset: PathConfiguredCHOISEvaluationDataset,
    cell_dataset: PathConfiguredCHOISEvaluationDataset,
    embedded_sequence_ids: Sequence[str],
) -> np.ndarray:
    """Measure one global cell-minus-GT offset over the embedded export subset."""
    if not embedded_sequence_ids:
        raise AdapterError("global offset diagnostic requires embedded sequences")
    truth_by_id = {
        str(sample["seq_name"]): sample["global_jpos"]
        for sample in truth_dataset.samples
    }
    cell_by_id = {
        str(sample["seq_name"]): sample["global_jpos"]
        for sample in cell_dataset.samples
    }
    missing_truth = [sequence_id for sequence_id in embedded_sequence_ids if sequence_id not in truth_by_id]
    missing_cell = [sequence_id for sequence_id in embedded_sequence_ids if sequence_id not in cell_by_id]
    if missing_truth or missing_cell:
        raise AdapterError(
            "global offset diagnostic cannot find the embedded sequence subset: "
            f"missing_truth={missing_truth[:20]}, missing_cell={missing_cell[:20]}"
        )
    try:
        truth_values = np.stack(
            [truth_by_id[sequence_id] for sequence_id in embedded_sequence_ids], axis=0,
        )
        cell_values = np.stack(
            [cell_by_id[sequence_id] for sequence_id in embedded_sequence_ids], axis=0,
        )
    except ValueError as exc:
        raise AdapterError("global offset diagnostic requires aligned [N,126,24,3] exports") from exc
    expected_shape = (len(embedded_sequence_ids), 126, 24, 3)
    if truth_values.shape != expected_shape or cell_values.shape != expected_shape:
        raise AdapterError(
            "global offset diagnostic requires aligned [N,126,24,3] exports; "
            f"got truth={truth_values.shape}, cell={cell_values.shape}"
        )
    return np.mean(
        np.asarray(cell_values, dtype=np.float64)
        - np.asarray(truth_values, dtype=np.float64),
        axis=(0, 1, 2),
        dtype=np.float64,
    )


def _sha256_float_series(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype=np.float64).tobytes()).hexdigest()


def _prefix_percentiles(samples: np.ndarray, prefixes: Sequence[int]) -> Dict[str, Any]:
    """Percentile intervals over leading prefixes of one replicate series.

    ``RandomState`` draws sequentially, so the first ``k`` replicates of an
    ``n``-replicate run are the draws a ``k``-replicate run would have made at
    the same seed.  Emitting the prefix interval therefore reproduces a shorter
    sealed interval without a second invocation and without a second embedding.
    """
    intervals: Dict[str, Any] = {}
    for prefix in prefixes:
        if prefix <= 1:
            raise AdapterError(f"bootstrap prefix must exceed one replicate: {prefix}")
        if prefix > int(samples.shape[0]):
            raise AdapterError(
                f"bootstrap prefix {prefix} exceeds the replicate count {int(samples.shape[0])}"
            )
        intervals[str(prefix)] = np.percentile(
            samples[:prefix], [2.5, 97.5],
        ).astype(float).tolist()
    return intervals


def _row_permutation_replay(motion_lengths: Any) -> list[int]:
    """Replay the upstream row permutation for one collated batch.

    ``EvaluatorModelWrapper.get_co_embeddings`` and ``get_motion_embeddings``
    both reorder the rows they return with
    ``np.argsort(m_lens.data.tolist())[::-1]`` and say so only in a comment, so
    the returned embedding rows are not in input order.  This replays the same
    expression on the same lengths; recording it makes a future change in
    ``np.argsort`` observable instead of silent.
    """
    return [int(value) for value in np.argsort(motion_lengths.data.tolist())[::-1].copy()]


def _row_permutation_summary(permutations: Sequence[Sequence[int]]) -> Dict[str, Any]:
    distinct = sorted({tuple(int(value) for value in batch) for batch in permutations})
    payload = "".join(
        ",".join(str(int(value)) for value in batch) + "\n" for batch in permutations
    )
    return {
        "batches": len(permutations),
        "distinct_permutations": len(distinct),
        "uniform_across_batches": len(distinct) == 1,
        "permutation": list(distinct[0]) if len(distinct) == 1 else None,
        "permutation_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _require_aligned_frames(
    frame_maps: Mapping[str, Mapping[str, int]],
    *,
    expected_frames: Optional[int] = None,
) -> Dict[str, Any]:
    """Gate G3: every compared set must carry identical per-sequence frame counts.

    The upstream wrapper reorders the rows it returns by
    ``np.argsort(m_lens)[::-1]``.  Identical lengths give identical permutations
    in every set, so embedding row ``i`` is the same sequence in every set and a
    paired difference is a difference over one sequence subset.  Unequal lengths
    turn that permutation into a genuinely set-dependent sort, and the pairing
    the whole comparison rests on silently disappears.  Fail closed instead.
    """
    names = list(frame_maps)
    if not names:
        raise AdapterError("frame-count gate requires at least one set")
    reference_name = names[0]
    reference = frame_maps[reference_name]
    for name in names[1:]:
        other = frame_maps[name]
        if set(other) != set(reference):
            missing = sorted(set(reference) - set(other))[:20]
            extra = sorted(set(other) - set(reference))[:20]
            raise AdapterError(
                f"G3 frame-count gate: set {name!r} does not cover the same sequences as "
                f"{reference_name!r}; missing={missing}, extra={extra}"
            )
        differing = sorted(key for key in reference if other[key] != reference[key])
        if differing:
            first = differing[0]
            raise AdapterError(
                f"G3 frame-count gate: set {name!r} disagrees with {reference_name!r} on "
                f"{len(differing)} of {len(reference)} sequences (first {first!r}: "
                f"{other[first]} frames against {reference[first]}). The upstream evaluator "
                "reorders returned rows by np.argsort(m_lens)[::-1], so unequal frame counts "
                "give the sets different row permutations and destroy the paired row "
                "alignment; refusing to compute a paired difference."
            )
    values = sorted({int(value) for value in reference.values()})
    if len(values) != 1:
        raise AdapterError(
            f"G3 frame-count gate: set {reference_name!r} is not frame-constant; "
            f"observed frame counts={values}. The paired evaluator requires one "
            "realised np.argsort(m_lens)[::-1] permutation across every sequence."
        )
    if expected_frames is not None and values[0] != int(expected_frames):
        raise AdapterError(
            f"G3 frame-count gate: expected {int(expected_frames)} frames per sequence, "
            f"but set {reference_name!r} has {values[0]}."
        )
    payload = "".join(f"{key}\t{reference[key]}\n" for key in sorted(reference))
    return {
        "gate": "G3",
        "aligned_across_sets": True,
        "sets_checked": names,
        "sequence_count": len(reference),
        "distinct_frame_counts": values,
        "frame_count_constant": True,
        "frame_count": values[0],
        "per_sequence_frames_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def _shared_resample_indices(count: int, replicates: int, seed: int) -> np.ndarray:
    """One resample index matrix shared by the ground truth and every cell.

    Drawn one replicate at a time with the same ``RandomState`` call shape as
    :func:`_bootstrap_fid_interval`, so row ``r`` is bitwise the ``r``-th
    selection that function makes at the same seed.  A longer matrix therefore
    extends a shorter one rather than replacing it.
    """
    if replicates <= 0:
        raise AdapterError("FID bootstrap replicates must be positive")
    if count < 2:
        raise AdapterError("bootstrap requires at least two embedded sequences")
    random_state = np.random.RandomState(seed)
    indices = np.empty((replicates, count), dtype=np.int64)
    for index in range(replicates):
        indices[index] = random_state.randint(0, count, size=count)
    return indices


def _paired_fid_bootstrap(
    truth_embeddings: np.ndarray,
    cell_embeddings: Mapping[str, np.ndarray],
    point_estimates: Mapping[str, float],
    *,
    frechet: Any,
    activation_statistics: Any,
    replicates: int,
    seed: int,
    prefixes: Sequence[int] = (),
    resample_indices: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """Every cell FID and every pairwise paired difference from one index matrix.

    One process, one ground-truth embedding, one replicate loop: replicate ``r``
    resamples once and applies that one selection to the ground truth and to
    every cell, so ``FID(cell_j) - FID(cell_i)`` is a difference over the same
    resampled sequence subset instead of over two independently drawn ones.
    Two separate invocations cannot be combined after the fact, because their
    ground-truth embeddings are not guaranteed to agree to the last bit.

    ``replicates <= 0`` emits the point-estimate differences only.
    """
    names = list(cell_embeddings)
    if not names:
        raise AdapterError("paired FID bootstrap requires at least one cell")
    if truth_embeddings.ndim != 2 or int(truth_embeddings.shape[0]) < 2:
        raise AdapterError("paired FID bootstrap requires [N,D] embeddings with N>=2")
    count = int(truth_embeddings.shape[0])
    for name in names:
        if cell_embeddings[name].shape != truth_embeddings.shape:
            raise AdapterError(
                f"paired FID bootstrap requires equal embedding shapes; cell {name!r} has "
                f"{tuple(cell_embeddings[name].shape)} against ground truth "
                f"{tuple(truth_embeddings.shape)}"
            )
        if name not in point_estimates:
            raise AdapterError(f"missing FID point estimate for cell {name!r}")
        if not math.isfinite(float(point_estimates[name])):
            raise AdapterError(f"nonfinite FID point estimate for cell {name!r}")
    indices: Optional[np.ndarray] = None
    series: Dict[str, np.ndarray] = {}
    if replicates > 0:
        if resample_indices is None:
            indices = _shared_resample_indices(count, replicates, seed)
        else:
            indices = np.ascontiguousarray(resample_indices, dtype=np.int64)
            if indices.shape != (replicates, count):
                raise AdapterError(
                    "shared FID resample indices have the wrong shape: "
                    f"expected {(replicates, count)}, got {indices.shape}"
                )
            if np.any(indices < 0) or np.any(indices >= count):
                raise AdapterError("shared FID resample indices contain an out-of-range row")
        series = {name: np.empty(replicates, dtype=np.float64) for name in names}
        for index in range(replicates):
            selection = indices[index]
            truth_mean, truth_covariance = activation_statistics(truth_embeddings[selection])
            for name in names:
                cell_mean, cell_covariance = activation_statistics(
                    cell_embeddings[name][selection],
                )
                value = float(
                    frechet(truth_mean, truth_covariance, cell_mean, cell_covariance)
                )
                if not math.isfinite(value):
                    raise AdapterError(
                        f"nonfinite FID bootstrap replicate {index} for cell {name!r}"
                    )
                series[name][index] = value
    cells: Dict[str, Any] = {}
    for name in names:
        entry: Dict[str, Any] = {"point_estimate": float(point_estimates[name])}
        if replicates > 0:
            entry["bootstrap_95_ci"] = np.percentile(
                series[name], [2.5, 97.5],
            ).astype(float).tolist()
            entry["replicate_series_sha256"] = _sha256_float_series(series[name])
            if prefixes:
                entry["prefix_bootstrap_95_ci"] = _prefix_percentiles(series[name], prefixes)
        cells[name] = entry
    differences: Dict[str, Any] = {}
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            subtrahend, minuend = names[left], names[right]
            entry = {
                "minuend": minuend,
                "subtrahend": subtrahend,
                "point_estimate": float(point_estimates[minuend])
                - float(point_estimates[subtrahend]),
            }
            if replicates > 0:
                delta = series[minuend] - series[subtrahend]
                entry["bootstrap_95_ci"] = np.percentile(
                    delta, [2.5, 97.5],
                ).astype(float).tolist()
                entry["replicate_series_sha256"] = _sha256_float_series(delta)
                if prefixes:
                    entry["prefix_bootstrap_95_ci"] = _prefix_percentiles(delta, prefixes)
            differences[f"{minuend}_minus_{subtrahend}"] = entry
    return {
        "unit": "paired_embedded_sequence",
        "replicates": int(replicates) if replicates > 0 else 0,
        "seed": seed,
        "count": count,
        "percentiles": [2.5, 97.5],
        "percentile_method": "linear",
        "shared_resample_index_matrix": replicates > 0,
        "single_process_single_ground_truth_embedding": True,
        "resample_index_shape": list(indices.shape) if indices is not None else None,
        "resample_index_dtype": str(indices.dtype) if indices is not None else None,
        "resample_index_sha256": (
            hashlib.sha256(np.ascontiguousarray(indices).tobytes()).hexdigest()
            if indices is not None else None
        ),
        "prefixes": [int(prefix) for prefix in prefixes],
        "cell_order": names,
        "cells": cells,
        "paired_differences": differences,
    }


def _bootstrap_mean_intervals(
    matching_distances: np.ndarray,
    r_precision_rows: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    if replicates <= 0:
        raise AdapterError("bootstrap replicates must be positive")
    if matching_distances.ndim != 1:
        raise AdapterError("matching distances must be one-dimensional")
    if r_precision_rows.shape != (matching_distances.shape[0], 3):
        raise AdapterError("R-Precision rows must have shape [N,3]")
    count = int(matching_distances.shape[0])
    if count < 2:
        raise AdapterError("bootstrap requires at least two embedded sequences")
    random_state = np.random.RandomState(seed)
    matching_samples = np.empty(replicates, dtype=np.float64)
    r_precision_samples = np.empty((replicates, 3), dtype=np.float64)
    chunk_size = min(512, replicates)
    for start in range(0, replicates, chunk_size):
        end = min(start + chunk_size, replicates)
        indices = random_state.randint(0, count, size=(end - start, count))
        matching_samples[start:end] = matching_distances[indices].mean(axis=1)
        r_precision_samples[start:end] = r_precision_rows[indices].mean(axis=1)
    return {
        "unit": "embedded_sequence",
        "replicates": replicates,
        "seed": seed,
        "MatchingScore": {
            "bootstrap_95_ci": np.percentile(
                matching_samples, [2.5, 97.5],
            ).astype(float).tolist(),
        },
        **{
            f"R-Precision@{rank}": {
                "bootstrap_95_ci": np.percentile(
                    r_precision_samples[:, rank - 1], [2.5, 97.5],
                ).astype(float).tolist(),
            }
            for rank in (1, 2, 3)
        },
    }


def _bootstrap_fid_interval(
    truth_embeddings: np.ndarray,
    prediction_embeddings: np.ndarray,
    *,
    frechet: Any,
    activation_statistics: Any,
    replicates: int,
    seed: int,
) -> Dict[str, Any]:
    if replicates <= 0:
        raise AdapterError("FID bootstrap replicates must be positive")
    if truth_embeddings.shape != prediction_embeddings.shape:
        raise AdapterError("paired FID bootstrap requires equal embedding shapes")
    if truth_embeddings.ndim != 2 or truth_embeddings.shape[0] < 2:
        raise AdapterError("paired FID bootstrap requires [N,D] embeddings with N>=2")
    count = int(truth_embeddings.shape[0])
    random_state = np.random.RandomState(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selection = random_state.randint(0, count, size=count)
        truth_mean, truth_covariance = activation_statistics(
            truth_embeddings[selection],
        )
        predicted_mean, predicted_covariance = activation_statistics(
            prediction_embeddings[selection],
        )
        samples[index] = float(
            frechet(
                truth_mean, truth_covariance,
                predicted_mean, predicted_covariance,
            )
        )
        if not math.isfinite(float(samples[index])):
            raise AdapterError(f"nonfinite FID bootstrap replicate {index}")
    return {
        "unit": "paired_embedded_sequence",
        "replicates": replicates,
        "seed": seed,
        "bootstrap_95_ci": np.percentile(
            samples, [2.5, 97.5],
        ).astype(float).tolist(),
    }


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    chois_config = chois_evaluator.load_config(args.chois_config)
    t2m_config = chois_evaluator.load_config(args.text_to_motion_config)
    chois_provenance = chois_evaluator.verify_upstream(args.chois_root.resolve(), chois_config)
    dependency_provenance = verify_text_to_motion(args.text_to_motion_root.resolve(), t2m_config)
    assets = chois_evaluator.require_assets(
        args.data_root.resolve(), args.glove_root.resolve(), args.checkpoint.resolve()
    )
    prediction_info, prediction_tree = chois_evaluator.read_npz_directory(args.predictions.resolve())
    truth_info, truth_tree = chois_evaluator.read_npz_directory(args.ground_truth.resolve())
    if args.require_matched_ids and sorted(prediction_info) != sorted(truth_info):
        missing = sorted(set(truth_info) - set(prediction_info))
        extra = sorted(set(prediction_info) - set(truth_info))
        raise AdapterError(
            f"matched input IDs differ: missing={missing[:20]}, extra={extra[:20]}"
        )
    comparison_paths = [path.resolve() for path in getattr(args, "compare_predictions", [])]
    comparison_inputs = []
    comparison_frame_gate: Optional[Dict[str, Any]] = None
    if args.emit_offset_corrected_fid and not comparison_paths:
        raise AdapterError(
            "--emit-offset-corrected-fid requires --compare-predictions so A and B are defined"
        )
    if comparison_paths:
        if not args.require_matched_ids:
            raise AdapterError(
                "--compare-predictions requires --require-matched-ids so every paired cell "
                "uses the same embedded sequence identities"
            )
        frame_maps: Dict[str, Mapping[str, int]] = {
            "ground_truth": {
                sequence_id: int(metadata["frames"])
                for sequence_id, metadata in truth_info.items()
            },
            "predictions": {
                sequence_id: int(metadata["frames"])
                for sequence_id, metadata in prediction_info.items()
            },
        }
        for index, path in enumerate(comparison_paths, start=1):
            info, tree = chois_evaluator.read_npz_directory(path)
            if sorted(info) != sorted(truth_info):
                missing = sorted(set(truth_info) - set(info))
                extra = sorted(set(info) - set(truth_info))
                raise AdapterError(
                    f"compare_predictions_{index} IDs differ from ground truth: "
                    f"missing={missing[:20]}, extra={extra[:20]}"
                )
            name = f"compare_predictions_{index}"
            comparison_inputs.append((name, path, info, tree))
            frame_maps[name] = {
                sequence_id: int(metadata["frames"])
                for sequence_id, metadata in info.items()
            }
        comparison_frame_gate = _require_aligned_frames(
            frame_maps,
            expected_frames=126,
        )
    if args.checkpoint.resolve() != (
        args.checkpoints_dir.resolve() / args.dataset_name / "text_motion_features" / "model" / "finest.tar"
    ):
        raise AdapterError("checkpoint must be checkpoints-dir/dataset-name/text_motion_features/model/finest.tar")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise AdapterError("CUDA requested but unavailable")
    device = torch.device("cuda:0" if args.device == "cuda" else args.device)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.set_device(device.index if device.index is not None else 0)
        torch.cuda.manual_seed_all(args.seed)

    wrapper, vectorizer, metrics = _load_components(
        args.chois_root.resolve(), args.text_to_motion_root.resolve(), args.glove_root.resolve(),
        args.checkpoints_dir.resolve(), args.dataset_name, device,
    )
    truth_dataset = PathConfiguredCHOISEvaluationDataset(
        args.ground_truth.resolve(), args.data_root.resolve(), vectorizer,
    )
    truth_loader = _loader(truth_dataset, args.batch_size, args.workers)
    comparison_output: Optional[Dict[str, Any]] = None
    offset_corrected_output: Optional[Dict[str, Any]] = None
    if comparison_paths:
        truth = _embeddings(
            truth_loader, wrapper, metrics, matching=False, record_permutation=True,
        )
        cell_specs = [
            ("predictions", args.predictions.resolve(), prediction_info, prediction_tree),
            *comparison_inputs,
        ]
        cell_datasets: Dict[str, PathConfiguredCHOISEvaluationDataset] = {}
        cell_embeddings: Dict[str, Dict[str, Any]] = {}
        for name, path, _, _ in cell_specs:
            dataset = PathConfiguredCHOISEvaluationDataset(
                path, args.data_root.resolve(), vectorizer,
            )
            cell_datasets[name] = dataset
            embedded = _embeddings(
                _loader(dataset, args.batch_size, args.workers),
                wrapper,
                metrics,
                matching=True,
                record_permutation=True,
            )
            if embedded["sequence_ids"] != truth["sequence_ids"]:
                raise AdapterError(
                    f"embedded matched sequence order differs for comparison cell {name!r}"
                )
            cell_embeddings[name] = embedded

        row_permutations = {
            "ground_truth": truth["row_permutation"],
            **{
                name: embedded["row_permutation"]
                for name, embedded in cell_embeddings.items()
            },
        }
        permutation_hashes = {
            value["permutation_sha256"] for value in row_permutations.values()
        }
        if len(permutation_hashes) != 1:
            raise AdapterError(
                "paired embedding row permutations differ across cells; refusing to "
                "compute a paired FID difference"
            )

        shared_fid_resample_indices: Optional[np.ndarray] = None
        offset_vectors: Dict[str, np.ndarray] = {}
        offset_corrected_embeddings: Dict[str, Dict[str, Any]] = {}
        if args.emit_offset_corrected_fid:
            offset_source_names = ("predictions", "compare_predictions_1")
            missing_offset_cells = [
                name for name in offset_source_names if name not in cell_datasets
            ]
            if missing_offset_cells:
                raise AdapterError(
                    "--emit-offset-corrected-fid requires primary A and B cells; "
                    f"missing={missing_offset_cells}"
                )
            if args.fid_bootstrap_replicates > 0:
                shared_fid_resample_indices = _shared_resample_indices(
                    len(truth["sequence_ids"]),
                    args.fid_bootstrap_replicates,
                    args.bootstrap_seed,
                )
            for name in offset_source_names:
                offset = _global_mean_bias_vector(
                    truth_dataset,
                    cell_datasets[name],
                    truth["sequence_ids"],
                )
                offset_vectors[name] = offset
                corrected_dataset = PathConfiguredCHOISEvaluationDataset(
                    cell_specs[
                        next(index for index, spec in enumerate(cell_specs) if spec[0] == name)
                    ][1],
                    args.data_root.resolve(),
                    vectorizer,
                    global_offset=offset,
                )
                corrected = _embeddings(
                    _loader(corrected_dataset, args.batch_size, args.workers),
                    wrapper,
                    metrics,
                    matching=False,
                    record_permutation=True,
                )
                if corrected["sequence_ids"] != truth["sequence_ids"]:
                    raise AdapterError(
                        f"embedded matched sequence order differs for offset cell {name!r}"
                    )
                if corrected["row_permutation"]["permutation_sha256"] != (
                    row_permutations["ground_truth"]["permutation_sha256"]
                ):
                    raise AdapterError(
                        f"row permutation differs for offset cell {name!r}; refusing to "
                        "compute its diagnostic FID"
                    )
                offset_corrected_embeddings[name] = corrected

        truth_mean, truth_covariance = metrics["activation_statistics"](
            truth["motion_embeddings"],
        )
        point_metrics_by_cell: Dict[str, Dict[str, float]] = {}
        additive_uncertainty_by_cell: Dict[str, Dict[str, Any]] = {}
        offset_corrected_fid_points: Dict[str, float] = {}
        if args.emit_offset_corrected_fid:
            for name, corrected in offset_corrected_embeddings.items():
                corrected_mean, corrected_covariance = metrics["activation_statistics"](
                    corrected["motion_embeddings"],
                )
                offset_corrected_fid_points[name] = float(metrics["frechet"](
                    truth_mean,
                    truth_covariance,
                    corrected_mean,
                    corrected_covariance,
                ))
                if not math.isfinite(offset_corrected_fid_points[name]):
                    raise AdapterError(f"nonfinite corrected FID point estimate for {name!r}")
        for name, _, _, _ in cell_specs:
            embedded = cell_embeddings[name]
            predicted_mean, predicted_covariance = metrics["activation_statistics"](
                embedded["motion_embeddings"],
            )
            point_metrics_by_cell[name] = {
                "FID": float(metrics["frechet"](
                    truth_mean, truth_covariance, predicted_mean, predicted_covariance,
                )),
                "MatchingScore": float(embedded["matching_score"]),
                "R-Precision@1": float(embedded["r_precision"][0]),
                "R-Precision@2": float(embedded["r_precision"][1]),
                "R-Precision@3": float(embedded["r_precision"][2]),
                "Diversity": float(metrics["diversity"](
                    embedded["motion_embeddings"], args.diversity_times,
                )),
            }
            if not all(math.isfinite(value) for value in point_metrics_by_cell[name].values()):
                raise AdapterError(f"nonfinite point metric in comparison cell {name!r}")
            if args.bootstrap_replicates:
                additive_uncertainty_by_cell[name] = _bootstrap_mean_intervals(
                    embedded["matching_distances"],
                    embedded["r_precision_rows"],
                    replicates=args.bootstrap_replicates,
                    seed=args.bootstrap_seed,
                )

        predicted = cell_embeddings["predictions"]
        point_metrics = point_metrics_by_cell["predictions"]
        embedded_ids = list(predicted["sequence_ids"])
        embedded_set = set(embedded_ids)
        all_prediction_ids = list(cell_datasets["predictions"].sequence_ids)
        dropped_prediction_ids = [
            sequence_id for sequence_id in all_prediction_ids
            if sequence_id not in embedded_set
        ]
        uncertainty: Dict[str, Any] = {}
        if args.bootstrap_replicates:
            uncertainty["additive_metrics"] = additive_uncertainty_by_cell["predictions"]

        bootstrap_cell_embeddings = {
            name: embedded["motion_embeddings"]
            for name, embedded in cell_embeddings.items()
        }
        bootstrap_point_estimates = {
            name: point_metrics_by_cell[name]["FID"]
            for name, _, _, _ in cell_specs
        }
        offset_prime_names = {
            "predictions": "A_prime",
            "compare_predictions_1": "B_prime",
        }
        if args.emit_offset_corrected_fid:
            bootstrap_cell_embeddings.update({
                prime_name: offset_corrected_embeddings[source_name]["motion_embeddings"]
                for source_name, prime_name in offset_prime_names.items()
            })
            bootstrap_point_estimates.update({
                prime_name: offset_corrected_fid_points[source_name]
                for source_name, prime_name in offset_prime_names.items()
            })

        all_fid_summary = _paired_fid_bootstrap(
            truth["motion_embeddings"],
            bootstrap_cell_embeddings,
            bootstrap_point_estimates,
            frechet=metrics["frechet"],
            activation_statistics=metrics["activation_statistics"],
            replicates=args.fid_bootstrap_replicates,
            seed=args.bootstrap_seed,
            prefixes=(200,) if args.fid_bootstrap_replicates >= 200 else (),
            resample_indices=shared_fid_resample_indices,
        )

        def _fid_summary_for(names: Sequence[str]) -> Dict[str, Any]:
            selected = set(names)
            return {
                **all_fid_summary,
                "cell_order": list(names),
                "cells": {
                    name: all_fid_summary["cells"][name]
                    for name in names
                },
                "paired_differences": {
                    key: value
                    for key, value in all_fid_summary["paired_differences"].items()
                    if value["minuend"] in selected and value["subtrahend"] in selected
                },
            }

        primary_cell_names = [name for name, _, _, _ in cell_specs]
        fid_summary = _fid_summary_for(primary_cell_names)
        if args.fid_bootstrap_replicates:
            primary_fid = fid_summary["cells"]["predictions"]
            uncertainty["FID"] = {
                "unit": fid_summary["unit"],
                "replicates": fid_summary["replicates"],
                "seed": fid_summary["seed"],
                "bootstrap_95_ci": primary_fid["bootstrap_95_ci"],
            }

        if args.emit_offset_corrected_fid:
            corrected_fid_summary = _fid_summary_for(list(offset_prime_names.values()))
            offset_cells: Dict[str, Any] = {}
            for source_name, prime_label in offset_prime_names.items():
                corrected_fid = corrected_fid_summary["cells"][prime_label]
                offset_cells[prime_label] = {
                    "label": "A′" if prime_label == "A_prime" else "B′",
                    "source_cell": source_name,
                    "global_mean_bias_vector": offset_vectors[source_name].astype(float).tolist(),
                    "vertical_component": float(offset_vectors[source_name][2]),
                    "vertical_axis": {"name": "z", "index": 2, "convention": "CHOIS Z-up"},
                    "embedded_sequence_count": len(truth["sequence_ids"]),
                    "frame_count": 126,
                    "joint_count": 24,
                    "corrected_fid_point_estimate": corrected_fid["point_estimate"],
                    "corrected_fid": corrected_fid,
                }
            delta_prime = corrected_fid_summary["paired_differences"][
                "B_prime_minus_A_prime"
            ]
            offset_corrected_output = {
                "diagnostic_label": "post_hoc_global_offset_diagnostic",
                "informational_only": True,
                "official_model_score": False,
                "correction": {
                    "type": "single_global_fixed_mean_bias_vector",
                    "computed_over": "the same 416 embedded sequences as the PRIMARY",
                    "frame_count": 126,
                    "joint_count": 24,
                    "subtract_each_cell_own_vector": True,
                    "common_correction": False,
                },
                "cells": offset_cells,
                "delta_prime": {
                    "formula": "FID(B′) - FID(A′)",
                    "minuend": "B_prime",
                    "subtrahend": "A_prime",
                    "point_estimate": delta_prime["point_estimate"],
                    **({"bootstrap_95_ci": delta_prime["bootstrap_95_ci"]}
                       if "bootstrap_95_ci" in delta_prime else {}),
                    "interpretation": (
                        "the residual budget effect after each cell's own best rigid correction"
                    ),
                    "correction_note": (
                        "A′ and B′ each subtract their own measured global vector; this is "
                        "not the budget effect under one common correction"
                    ),
                },
                "fid_bootstrap_replicates": int(args.fid_bootstrap_replicates),
                "shared_bootstrap_seed": int(args.bootstrap_seed),
                "resample_index_sha256": corrected_fid_summary["resample_index_sha256"],
                "primary_resample_index_sha256": fid_summary["resample_index_sha256"],
            }

        comparison_cells: Dict[str, Any] = {}
        cell_tree_hashes: Dict[str, str] = {}
        for name, path, info, tree in cell_specs:
            cell_tree_hashes[name] = tree
            dataset = cell_datasets[name]
            embedded = cell_embeddings[name]
            ids = list(embedded["sequence_ids"])
            id_set = set(ids)
            dropped = [
                sequence_id for sequence_id in dataset.sequence_ids
                if sequence_id not in id_set
            ]
            cell_value: Dict[str, Any] = {
                "input": {
                    "path": str(path),
                    "count": len(info),
                    "sha256": tree,
                },
                "metrics": point_metrics_by_cell[name],
                "FID": fid_summary["cells"][name],
                "embedding_protocol": {
                    "exported_prediction_count": len(info),
                    "embedded_count": len(ids),
                    "embedded_sequence_ids_sha256": _sha256_ids(ids),
                    "dropped_prediction_count": len(dropped),
                    "dropped_prediction_sequence_ids_sha256": _sha256_ids(dropped),
                },
                "row_permutation": embedded["row_permutation"],
            }
            if args.bootstrap_replicates:
                cell_value["additive_uncertainty"] = additive_uncertainty_by_cell[name]
            comparison_cells[name] = cell_value
        comparison_output = {
            "cell_order": [name for name, _, _, _ in cell_specs],
            "ground_truth": {
                "path": str(args.ground_truth.resolve()),
                "count": len(truth_info),
                "sha256": truth_tree,
                "embedded_count": len(truth["sequence_ids"]),
                "embedded_sequence_ids_sha256": _sha256_ids(truth["sequence_ids"]),
                "row_permutation": truth["row_permutation"],
            },
            "tree_sha256": {
                "ground_truth": truth_tree,
                **cell_tree_hashes,
            },
            "cells": comparison_cells,
            "paired_differences": fid_summary["paired_differences"],
            "fid": fid_summary,
            "fid_bootstrap_replicates": int(args.fid_bootstrap_replicates),
            "bootstrap_seed": int(args.bootstrap_seed),
            "shared_bootstrap_seed": int(args.bootstrap_seed),
            "resample_index_sha256": fid_summary["resample_index_sha256"],
            "resample_index_shape": fid_summary["resample_index_shape"],
            "shared_resample_index_matrix": fid_summary["shared_resample_index_matrix"],
            "single_process_single_ground_truth_embedding": True,
            "frame_count_guard": comparison_frame_gate,
            "row_permutations": row_permutations,
        }
    else:
        prediction_dataset = PathConfiguredCHOISEvaluationDataset(
            args.predictions.resolve(), args.data_root.resolve(), vectorizer,
        )
        prediction_loader = _loader(prediction_dataset, args.batch_size, args.workers)
        truth = _embeddings(truth_loader, wrapper, metrics, matching=False)
        predicted = _embeddings(prediction_loader, wrapper, metrics, matching=True)
        if args.require_matched_ids and truth["sequence_ids"] != predicted["sequence_ids"]:
            raise AdapterError("embedded matched sequence order differs")
        truth_mean, truth_covariance = metrics["activation_statistics"](truth["motion_embeddings"])
        predicted_mean, predicted_covariance = metrics["activation_statistics"](predicted["motion_embeddings"])
        point_metrics = {
            "FID": float(metrics["frechet"](truth_mean, truth_covariance, predicted_mean, predicted_covariance)),
            "MatchingScore": float(predicted["matching_score"]),
            "R-Precision@1": float(predicted["r_precision"][0]),
            "R-Precision@2": float(predicted["r_precision"][1]),
            "R-Precision@3": float(predicted["r_precision"][2]),
            "Diversity": float(metrics["diversity"](predicted["motion_embeddings"], args.diversity_times)),
        }
        embedded_ids = list(predicted["sequence_ids"])
        embedded_set = set(embedded_ids)
        all_prediction_ids = list(prediction_dataset.sequence_ids)
        dropped_prediction_ids = [
            sequence_id for sequence_id in all_prediction_ids
            if sequence_id not in embedded_set
        ]
        uncertainty = {}
        if args.bootstrap_replicates:
            uncertainty["additive_metrics"] = _bootstrap_mean_intervals(
                predicted["matching_distances"],
                predicted["r_precision_rows"],
                replicates=args.bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
        if args.fid_bootstrap_replicates:
            if not args.require_matched_ids:
                raise AdapterError("FID paired bootstrap requires --require-matched-ids")
            uncertainty["FID"] = _bootstrap_fid_interval(
                truth["motion_embeddings"],
                predicted["motion_embeddings"],
                frechet=metrics["frechet"],
                activation_statistics=metrics["activation_statistics"],
                replicates=args.fid_bootstrap_replicates,
                seed=args.bootstrap_seed,
            )
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "adapter": {
            "path": str(Path(__file__).resolve()),
            "sha256": _sha256(Path(__file__).resolve()),
            "semantics": "CHOIS released metric with explicit paths; prediction and GT sets may differ in count",
        },
        "upstream": chois_provenance,
        "text_to_motion_dependency": dependency_provenance,
        "assets": assets,
        "inputs": {
            "predictions": {"path": str(args.predictions.resolve()), "count": len(prediction_info), "sha256": prediction_tree},
            "ground_truth": {"path": str(args.ground_truth.resolve()), "count": len(truth_info), "sha256": truth_tree},
        },
        "embedding_protocol": {
            "batch_size": args.batch_size,
            "drop_last": True,
            "matched_ids_required": bool(args.require_matched_ids),
            "exported_prediction_count": len(prediction_info),
            "exported_ground_truth_count": len(truth_info),
            "embedded_count": len(embedded_ids),
            "embedded_sequence_ids": embedded_ids,
            "embedded_sequence_ids_sha256": _sha256_ids(embedded_ids),
            "dropped_prediction_count": len(dropped_prediction_ids),
            "dropped_prediction_sequence_ids": dropped_prediction_ids,
            "dropped_prediction_sequence_ids_sha256": _sha256_ids(dropped_prediction_ids),
        },
        "runtime": {
            "device": str(device), "batch_size": args.batch_size, "workers": args.workers, "seed": args.seed,
        },
        "metrics": point_metrics,
        "uncertainty": uncertainty,
    }
    if comparison_output is not None:
        result["comparison"] = comparison_output
    if offset_corrected_output is not None:
        result["offset_corrected_fid"] = offset_corrected_output
    if args.output.exists():
        raise AdapterError(f"refusing to overwrite evaluator output: {args.output}")
    chois_evaluator.atomic_output(args.output, result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chois-root", type=Path, required=True)
    parser.add_argument("--text-to-motion-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--glove-root", type=Path, required=True)
    parser.add_argument("--checkpoints-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-name", default="omomo")
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--diversity-times", type=int, default=300)
    parser.add_argument("--require-matched-ids", action="store_true")
    parser.add_argument("--compare-predictions", type=Path, action="append", default=[], metavar="DIR")
    parser.add_argument("--emit-offset-corrected-fid", action="store_true")
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--fid-bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--chois-config", type=Path, default=DEFAULT_CHOIS_CONFIG)
    parser.add_argument("--text-to-motion-config", type=Path, default=DEFAULT_T2M_CONFIG)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate(args)
    except (AdapterError, chois_evaluator.EvaluatorError, OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
