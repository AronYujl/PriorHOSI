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


def generation_protocol(payload):
    return {"sample_type": payload["sample_type"],
            "sampler_steps": payload["timing"]["sampler_steps_per_window"],
            "guided": payload["guided"], "seed": payload["seed"]}


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
    """Evaluate sealed motion artifacts with the frozen encoder and gallery."""
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
                            "generation": {}},
               "arms": {}}
    bootstrap_indices = np.random.RandomState(42).randint(0, 245, size=(2000, 245))
    fid_samples = {}
    for arm, source in cfg.table3_inputs.items():
        payload = json.loads(Path(source).read_text())
        summary["protocol"]["generation"][arm] = generation_protocol(payload)
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



def paired_mean_ratios(reference, student, replicates=10000, seed=42):
    """Paired ratios of cohort means, sharing resampled rows across all metrics."""
    generator = torch.Generator(device=reference.device).manual_seed(seed)
    index = torch.randint(reference.shape[0], (replicates, reference.shape[0]),
                          generator=generator, device=reference.device)
    samples = student[index].mean(1) / reference[index].mean(1)
    point = student.mean(0) / reference.mean(0)
    interval = torch.quantile(samples, reference.new_tensor([0.025, 0.975]), dim=0)
    return point, interval


def ratio_gate(point, interval, larger_is_better):
    low, high = (float(value) for value in interval)
    if larger_is_better:
        status = "PASS" if low >= 0.95 else "FAIL" if high < 0.95 else "INCONCLUSIVE"
    else:
        status = "PASS" if high <= 1.05 else "FAIL" if low > 1.05 else "INCONCLUSIVE"
    return {"student_over_teacher": float(point), "ci95": [low, high],
            "larger_is_better": larger_is_better, "status": status}


def cm_distillation_readout(cfg):
    """Apply the registered quality, safety and latency gates to sealed readouts."""
    output = Path(cfg.cm1_gate_output)
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device(str(cfg.device))
    teacher_dir, student_dir = Path(cfg.cm1_reference_readout), Path(cfg.table3_output)
    teacher_summary = json.loads((teacher_dir / "summary.json").read_text())
    student_summary = json.loads((student_dir / "summary.json").read_text())
    exclusions = {row["sequence_id"] for row in
                  json.loads(Path(cfg.cm1_safety_exclusions).read_text())["episodes"]}
    full_keys = ("pen_ratio", "pene_pct_scene", "pene_sum_mean_floorexcl",
                 "pene_sum_max_floorexcl", "fs_nemf", "boundary_jerk", "interior_jerk",
                 "goal_planar_err_m", "contact_count", "contact_count_exterior")
    higher = {"contact_count", "contact_count_exterior", "success_last_5cm", "R-Precision@3"}
    result = {"schema_version": 1, "seed": 42, "replicates": 10000,
              "fid_replicates": 2000, "quality_scope": "internal native evaluator",
              "primary_arm": "guided", "arms": {}, "latency": {}}
    for arm in ("unguided", "guided"):
        teacher_payload = json.loads(Path(cfg.cm1_reference_inputs[arm]).read_text())
        student_payload = json.loads(Path(cfg.table3_inputs[arm]).read_text())
        native = {}
        walk_ids = None
        for group in ("full375", "locomotion", "interactive"):
            if group == "full375":
                reference, candidate = teacher_payload["metrics"], student_payload["metrics"]
                keys = full_keys
            else:
                reference = json.loads((teacher_dir / arm / f"{group}.json").read_text())["metrics"]
                candidate = json.loads((student_dir / arm / f"{group}.json").read_text())["metrics"]
                keys = tuple(next(iter(reference.values())))
                if group == "locomotion":
                    walk_ids = set(reference)
            names = sorted(reference)
            assert set(names) == set(candidate)
            x = torch.tensor([[reference[name][key] for key in keys] for name in names],
                             dtype=torch.float64, device=device)
            y = torch.tensor([[candidate[name][key] for key in keys] for name in names],
                             dtype=torch.float64, device=device)
            point, interval = paired_mean_ratios(x, y)
            native[group] = {key: dict(ratio_gate(point[j], interval[:, j], key in higher),
                                       teacher_mean=float(x[:, j].mean()),
                                       student_mean=float(y[:, j].mean()), sequences=len(names))
                             for j, key in enumerate(keys)}

        # FID draws have the same frozen sequence order and NumPy seed42 indices.
        with np.load(teacher_dir / arm / "embeddings.npz") as reference, \
                np.load(student_dir / arm / "embeddings.npz") as candidate:
            assert np.array_equal(reference["sequence_ids"], candidate["sequence_ids"])
            x = torch.as_tensor(reference["fid_bootstrap"], dtype=torch.float64, device=device)
            y = torch.as_tensor(candidate["fid_bootstrap"], dtype=torch.float64, device=device)
            fid_interval = torch.quantile(y / x, x.new_tensor([0.025, 0.975]))
        tfid = teacher_summary["arms"][arm]["points"]["interactive"]["FID"]["mean"]
        sfid = student_summary["arms"][arm]["points"]["interactive"]["FID"]["mean"]
        native["interactive"]["FID"] = dict(ratio_gate(sfid / tfid, fid_interval, False),
                                                teacher_mean=tfid, student_mean=sfid, sequences=245)
        reference = json.loads((teacher_dir / arm / "retrieval_occurrences.json").read_text())
        candidate = json.loads((student_dir / arm / "retrieval_occurrences.json").read_text())
        identity = lambda rows: [(r["sequence_id"], r["gallery"], r["query"]) for r in rows]
        assert identity(reference) == identity(candidate)
        x = torch.tensor([[r["r_at_3"]] for r in reference], dtype=torch.float64, device=device)
        y = torch.tensor([[r["r_at_3"]] for r in candidate], dtype=torch.float64, device=device)
        point, interval = paired_mean_ratios(x, y)
        native["interactive"]["R-Precision@3"] = dict(
            ratio_gate(point[0], interval[:, 0], True), teacher_mean=float(x.mean()),
            student_mean=float(y.mean()), query_occurrences=len(reference),
            unique_sequences=len({r["sequence_id"] for r in reference}))

        safety = {}
        for cohort, names in (("full375", sorted(student_payload["metrics"])),
                              ("holdout355", sorted(set(student_payload["metrics"]) - exclusions))):
            rows = student_payload["metrics"]
            over5g = [name for name in names if rows[name]["frames_over_5g"] > 0]
            low_walk = [name for name in names if name in walk_ids and rows[name]["pelvis_h_min"] < 0.6]
            safety[cohort] = {"sequences": len(names),
                              "walk_sequences": len(set(names) & walk_ids),
                              "over5g_episodes": len(over5g),
                              "over5g_frames": int(sum(rows[name]["frames_over_5g"] for name in names)),
                              "low_walk_episodes": len(low_walk),
                              "over5g_ids": over5g, "low_walk_ids": low_walk}
        holdout = safety["holdout355"]
        assert holdout["sequences"] == 355 and holdout["walk_sequences"] == 126
        safety["passed"] = (holdout["over5g_episodes"] <= 8 and
                            holdout["over5g_frames"] <= 38 and holdout["low_walk_episodes"] <= 2)
        result["arms"][arm] = {"ratios": native, "safety": safety,
                               "generation_protocol": generation_protocol(student_payload)}

    for label, path in cfg.cm1_latency_inputs.items():
        payload = json.loads(Path(path).read_text())
        assert payload["timing"]["timing_valid"] and payload["latency_subset"]["enabled"]
        result["latency"][label] = {"source": str(path), "timing": payload["timing"],
                                    "generation_protocol": generation_protocol(payload)}
    for arm in ("unguided", "guided"):
        student = result["latency"][f"student_{arm}"]["timing"]
        teacher = result["latency"][f"teacher_{arm}"]["timing"]
        result["arms"][arm]["generation_speedup"] = (teacher["total_generation_seconds"] /
                                                     student["total_generation_seconds"])
    guided = result["arms"]["guided"]
    failures = [f"{group}/{key}:{value['status']}" for group, values in guided["ratios"].items()
                for key, value in values.items() if value["status"] != "PASS"]
    if not guided["safety"]["passed"]:
        failures.append("safety:FAIL")
    fps = result["latency"]["student_guided"]["timing"]["avg_fps"]
    if fps < 20:
        failures.append("guided_generation_fps:FAIL")
    result["gate"] = {"passed": not failures, "unmet_bounds": failures,
                      "guided_fps": fps, "required_guided_fps": 20,
                      "baseline": "R2+CG until every registered bound passes"}
    _write(output / "summary.json", result)
    return output / "summary.json"
