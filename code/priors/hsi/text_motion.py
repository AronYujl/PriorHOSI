"""Native Table 3 readout from sealed motions and the frozen LINGO encoder."""

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


GEOMETRY_KEYS = {
    "locomotion": ("pene_pct_scene", "pene_sum_mean_floorexcl",
                   "pene_sum_max_floorexcl", "fs_nemf"),
    "interactive": ("last_dist", "success_last_5cm", "pene_sum_mean_floorexcl",
                    "pene_sum_max_floorexcl", "interior_jerk", "boundary_jerk",
                    "contact_count", "contact_count_exterior"),
}


def frechet_samples(reference, prediction):
    """Empirical Gaussian FID via the sample-space covariance-factor product.

    For centered X,Y, tr(sqrt(Cx Cy)) = ||X Y^T||_* / (N-1).
    Leading dimensions are independent bootstrap replicates.
    """
    mean_x, mean_y = reference.mean(-2), prediction.mean(-2)
    x = reference - mean_x.unsqueeze(-2)
    y = prediction - mean_y.unsqueeze(-2)
    divisor = reference.shape[-2] - 1
    cross = x @ y.transpose(-1, -2) / divisor
    trace_root = torch.linalg.svdvals(cross).sum(-1)
    return ((mean_x - mean_y).square().sum(-1)
            + (x.square().sum((-2, -1)) + y.square().sum((-2, -1))) / divisor
            - 2 * trace_root)


def geometry_groups(metrics, captions):
    groups = {"locomotion": {}, "interactive": {}}
    for sequence_id, record in metrics.items():
        group = "locomotion" if captions[sequence_id] == "walk" else "interactive"
        values = dict(record)
        values["success_last_5cm"] = float(values["last_dist"] <= 0.05)
        groups[group][sequence_id] = {key: values[key] for key in GEOMETRY_KEYS[group]}
    return groups


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _legacy_encoder(cfg, device):
    source = Path(cfg.table3_encoder_source)
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("frozen_lingo_readout", source)
    legacy = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = legacy
    spec.loader.exec_module(legacy)
    # Qualified upstream imports keep its utils package separate from code/utils.py.
    sys.path.insert(0, str(cfg.table3_chois_root))
    from t2m_eval.networks.modules import (
        TextEncoderBiGRUCo, MotionEncoderBiGRUCo, MovementConvEncoder,
    )
    from t2m_eval.utils.word_vectorizer import POS_enumerator, WordVectorizer

    args = SimpleNamespace(
        mean=Path(cfg.table3_mean), std=Path(cfg.table3_std),
        glove_root=Path(cfg.table3_glove_root), embedding_batch=64,
    )
    models = (
        TextEncoderBiGRUCo(300, len(POS_enumerator), 512, 512, device=device).to(device),
        MotionEncoderBiGRUCo(512, 1024, 512, device=device).to(device),
        MovementConvEncoder(84, 512, 512).to(device),
    )
    state = torch.load(str(cfg.table3_encoder_checkpoint), map_location=device)
    for name, model in zip(("text_encoder", "motion_encoder", "movement_encoder"), models):
        model.load_state_dict(state[name], strict=True)
        model.eval()
        model.requires_grad_(False)
    modules = {"WordVectorizer": WordVectorizer}
    return legacy, args, modules, models


def _gallery_records(legacy, items, text, motion, distinct):
    captions = np.asarray([item.caption for item in items], dtype=object)
    selection = legacy.make_gallery_indices(np.arange(len(items)), captions, 32, 42, distinct)
    result = legacy.score_galleries(text, motion, captions, selection, 32, 42, 10000, 42)
    # Preserve the frozen occurrence-level retrieval protocol, including repeated IDs.
    rng = np.random.RandomState(42)
    records = []
    for begin in range(0, len(selection), 32):
        chosen = selection[begin:begin + 32]
        distances = np.linalg.norm(text[chosen, None, :] - motion[chosen][None, :, :], axis=2)
        for row in range(32):
            ranking = np.lexsort((rng.random(32), distances[row]))
            records.append({"sequence_id": items[chosen[row]].sequence_id,
                            "gallery": begin // 32, "query": row,
                            "r_at_3": float(row in ranking[:3])})
    result["unique_sequences_scored"] = len({r["sequence_id"] for r in records})
    return result, records


def table3_readout(cfg):
    """Evaluate existing R2 U/CG artifacts; never instantiate a motion generator."""
    output = Path(cfg.table3_output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(str(cfg.device))
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    legacy, args, modules, models = _legacy_encoder(cfg, device)
    truth_items = legacy.load_directory(Path(cfg.table3_truth_motion), 800)
    truth_text, truth_motion = legacy.embed(truth_items, modules, models, args, device)
    truth_tensor = torch.as_tensor(truth_motion, dtype=torch.float64, device=device)
    truth_ids = [item.sequence_id for item in truth_items]
    assert len(truth_ids) == 245

    # The historical C cell fixes preprocessing/encoder/gallery continuity.
    control_items = [item for item in legacy.load_directory(Path(cfg.table3_control_motion), 800)
                     if item.caption != "walk"]
    assert [item.sequence_id for item in control_items] == truth_ids
    control_text, control_motion = legacy.embed(control_items, modules, models, args, device)
    control_fid = float(frechet_samples(truth_tensor, torch.as_tensor(
        control_motion, dtype=torch.float64, device=device)))
    control_mm = float(np.linalg.norm(control_text - control_motion, axis=1).mean())
    control_ranking, _ = _gallery_records(legacy, control_items, control_text, control_motion, True)
    sealed = json.loads(Path(cfg.table3_control_result).read_text())
    expected_fid = sealed["distribution_table"]["Generated motions"]["FID (internal only, lower is better)"]
    expected_r = sealed["ranking"]["Generated motions"]["PRIMARY_deduplicated_gallery32"]["R-Precision@1/2/3"]
    assert abs(control_fid - expected_fid) <= 1e-4
    assert abs(control_mm - sealed["metrics"]["MM-Dist"]) <= 1e-5
    assert control_ranking["R-Precision@1/2/3"] == expected_r
    control_check = {"fid": control_fid, "fid_difference": control_fid - expected_fid,
                     "mm_dist": control_mm, "r_precision": expected_r}
    _write(output / "control_continuity.json", control_check)
    print("Frozen evaluator continuity passed", control_check, flush=True)

    truth_payload = json.loads(Path(cfg.table3_truth_metrics).read_text())
    summary = {"schema_version": 1, "seed": 42, "optimizer_updates": 0,
               "generated_windows": 0, "control_continuity": control_check,
               "metric_scope": "internal LINGO evaluator; published feature metrics are incompatible",
               "protocol": {"locomotion_episodes": 130, "interactive_episodes": 245,
                            "object_reaching": "final-frame minimum planar distance over 28 joints",
                            "fid_replicates": 2000, "other_replicates": 10000,
                            "r_precision_unit": "frozen gallery query occurrence; may repeat sequences",
                            "generation": "sealed 500-step diffusion, CFG 1, seed 42"},
               "arms": {}}
    bootstrap_indices = np.random.RandomState(42).randint(0, 245, size=(2000, 245))
    fid_samples = {}
    for arm, source in cfg.table3_inputs.items():
        payload = json.loads(Path(source).read_text())
        items = []
        for shard in payload["merged_from"]:
            items.extend(legacy.load_directory(Path(shard).parent.parent / "motion", 800))
        items.sort(key=lambda item: item.sequence_id)
        assert len(items) == 375
        assert {item.sequence_id for item in items} == set(payload["metrics"])
        captions = {item.sequence_id: item.caption for item in items}
        groups = geometry_groups(payload["metrics"], captions)
        assert [len(groups[k]) for k in ("locomotion", "interactive")] == [130, 245]
        selected = [item for item in items if item.caption != "walk"]
        assert [item.sequence_id for item in selected] == truth_ids
        assert [item.caption for item in selected] == [item.caption for item in truth_items]
        text, motion = legacy.embed(selected, modules, models, args, device)
        prediction = torch.as_tensor(motion, dtype=torch.float64, device=device)
        fid = float(frechet_samples(truth_tensor, prediction))
        mm_values = np.linalg.norm(text - motion, axis=1)
        mm_low, mm_high = legacy.bootstrap_mean_ci(mm_values, 10000, 42)
        ranking, occurrences = _gallery_records(legacy, selected, text, motion, True)
        literal, _ = _gallery_records(legacy, selected, text, motion, False)
        samples = []
        for begin in range(0, 2000, int(cfg.table3_bootstrap_batch)):
            index = torch.as_tensor(bootstrap_indices[begin:begin + int(cfg.table3_bootstrap_batch)],
                                    device=device)
            values = frechet_samples(truth_tensor[index], prediction[index])
            samples.extend(values.cpu().tolist())
            if begin % 200 == 0:
                print(f"{arm}: FID bootstrap {begin}/2000", flush=True)
        fid_samples[arm] = np.asarray(samples)
        arm_dir = output / arm
        arm_dir.mkdir()
        np.savez_compressed(arm_dir / "embeddings.npz", sequence_ids=np.asarray(truth_ids),
                            text=text, motion=motion, truth_text=truth_text, truth_motion=truth_motion,
                            fid_bootstrap=np.asarray(samples))
        _write(arm_dir / "retrieval_occurrences.json", occurrences)
        for item, distance in zip(selected, mm_values):
            groups["interactive"][item.sequence_id]["MM-Dist"] = float(distance)
        points = {}
        for group, records in groups.items():
            _write(arm_dir / f"{group}.json", {"sequence_count": len(records), "metrics": records})
            keys = list(next(iter(records.values())))
            data = np.asarray([[v[k] for k in keys] for v in records.values()])
            low, high = legacy.bootstrap_mean_ci(data, 10000, 42)
            points[group] = {k: {"mean": float(data[:, j].mean()), "ci95": [low[j], high[j]]}
                             for j, k in enumerate(keys)}
        points["interactive"]["FID"] = {"mean": fid, "ci95": np.percentile(samples, [2.5, 97.5]).tolist()}
        points["interactive"]["R-Precision@3"] = {
            "mean": ranking["R-Precision@1/2/3"][2],
            "ci95": ranking["bootstrap"]["95_ci"]["R-Precision@3"]}
        summary["arms"][arm] = {"source": str(source), "points": points,
                                "ranking": ranking, "literal_ranking": literal,
                                "MM-Dist_ci95": [mm_low[0], mm_high[0]]}
        _write(arm_dir / "summary.json", summary["arms"][arm])
        print(arm, "FID", fid, "R@3", ranking["R-Precision@1/2/3"][2],
              "MM-Dist", float(mm_values.mean()), flush=True)
        if arm == "unguided":
            gt_groups = geometry_groups(truth_payload["metrics"], captions)
            for group, records in gt_groups.items():
                _write(output / f"truth_{group}.json", {"sequence_count": len(records), "metrics": records})
    delta = fid_samples["guided"] - fid_samples["unguided"]
    summary["fid_guided_minus_unguided"] = {
        "mean_delta": summary["arms"]["guided"]["points"]["interactive"]["FID"]["mean"]
                      - summary["arms"]["unguided"]["points"]["interactive"]["FID"]["mean"],
        "ci95": np.percentile(delta, [2.5, 97.5]).tolist()}
    torch.cuda.synchronize(device)
    summary["seconds"] = time.perf_counter() - started
    summary["peak_cuda_bytes"] = torch.cuda.max_memory_allocated(device)
    _write(output / "summary.json", summary)
    return output / "summary.json"
