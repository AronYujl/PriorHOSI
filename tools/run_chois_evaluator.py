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

    def __init__(self, results_dir: Path, data_root: Path, word_vectorizer: Any):
        self.results_dir = results_dir
        self.word_vectorizer = word_vectorizer
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
        motion = torch.from_numpy(sample["global_jpos"].reshape(-1, 72)).float()
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


def _embeddings(loader: DataLoader, wrapper: Any, metrics: Mapping[str, Any], matching: bool) -> Dict[str, Any]:
    all_motion_embeddings = []
    all_sequence_ids: list[str] = []
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
    prediction_dataset = PathConfiguredCHOISEvaluationDataset(
        args.predictions.resolve(), args.data_root.resolve(), vectorizer,
    )
    truth_loader = _loader(truth_dataset, args.batch_size, args.workers)
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
    uncertainty: Dict[str, Any] = {}
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
