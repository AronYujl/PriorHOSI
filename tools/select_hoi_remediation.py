#!/usr/bin/env python3
"""Apply the preregistered D2 candidate/EMA selection order."""

import argparse
import json
import math
from pathlib import Path


def load_candidate(path, name, effective_batch):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload["selection"]["official_test_sequence_count"] != 0 or payload["selection"]["chois_sequence_count"] != 0:
        raise ValueError("official test/CHOIS may not participate in D2 selection")
    result = []
    for weight, value in payload["weights"].items():
        eligibility = value["eligibility"]
        result.append({
            "candidate": name,
            "effective_batch_size": effective_batch,
            "weight_variant": weight,
            "checkpoint_sha256": payload["checkpoint_sha256"],
            "eligible": bool(eligibility["eligible"]),
            "checks": eligibility["checks"],
            "ratios": eligibility["ratios"],
            "contact_f1_increase": eligibility["contact_f1_increase"],
            "geometry_score": math.sqrt(
                eligibility["ratios"]["object_goal"] * eligibility["ratios"]["pelvis_goal"]
            ),
            "physical_contact_f1": value["rollout"]["matched"]["aggregate"]["physical_contact_f1"],
        })
    return payload, result


def select(records):
    eligible = [value for value in records if value["eligible"]]
    if eligible:
        best_score = min(value["geometry_score"] for value in eligible)
        near = [value for value in eligible if value["geometry_score"] <= best_score * 1.02]
        best_contact = max(value["physical_contact_f1"] for value in near)
        near_contact = [value for value in near if value["physical_contact_f1"] >= best_contact * 0.98]
        selected = min(near_contact, key=lambda value: (
            value["effective_batch_size"] != 1024, value["effective_batch_size"], value["weight_variant"],
        ))
        return {"decision": "selected", "selected": selected, "d2_g_allowed": False}
    contact_only = []
    for value in records:
        failed = [name for name, passed in value["checks"].items() if not passed]
        if failed == ["contact_f1_increase_ge_0.10"]:
            contact_only.append(value)
    if contact_only:
        chosen = min(contact_only, key=lambda value: (
            value["geometry_score"], -value["physical_contact_f1"],
            value["effective_batch_size"] != 1024, value["effective_batch_size"],
        ))
        return {"decision": "D2-G-contact-only-fallback", "selected": chosen, "d2_g_allowed": True}
    return {"decision": "stop-no-eligible-candidate", "selected": None, "d2_g_allowed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r1024", required=True)
    parser.add_argument("--r3072", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    first, r1024 = load_candidate(args.r1024, "R-1024", 1024)
    second, r3072 = load_candidate(args.r3072, "R-3072", 3072)
    if first["selection"]["sequence_selection_sha256"] != second["selection"]["sequence_selection_sha256"]:
        raise ValueError("D2 candidates used different internal sequence selections")
    records = r1024 + r3072
    output = {
        "schema_version": 1,
        "selection_rule": [
            "eligible-only", "minimum object/pelvis ratio geometric mean",
            "maximum contact F1", "within 2 percent prefer EB1024",
        ],
        "official_test_used": False,
        "chois_used": False,
        "records": records,
        **select(records),
    }
    path = Path(args.output).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, sort_keys=True)
        handle.write("\n")


if __name__ == "__main__":
    main()
