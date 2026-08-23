import json

import pytest

from tools.build_hoi_gradient_manifest import (
    build_sampling_manifest, classify_stratum, largest_remainder,
)


def balanced_records(per_stratum=20):
    shapes = ((0, 0), (0, 10), (2, 10), (5, 10), (10, 10))
    return [
        (stratum * per_stratum + offset, f"q{stratum}-{offset}", both, engaged)
        for stratum, (both, engaged) in enumerate(shapes)
        for offset in range(per_stratum)
    ]


def build_balanced():
    return build_sampling_manifest(
        balanced_records(), shards=2, windows_per_shard=20,
        both_reference=0.425, engaged_reference=0.8, tolerance=0.0,
    )


def test_deterministic_serialization():
    first = json.dumps(build_balanced(), sort_keys=True)
    second = json.dumps(build_balanced(), sort_keys=True)
    assert first.encode() == second.encode()


def test_stratum_classification_boundaries():
    assert classify_stratum(0, 0) == "S0"
    assert classify_stratum(0, 10) == "S1"
    assert classify_stratum(4, 10) == "S2"
    assert classify_stratum(5, 10) == "S3"
    assert classify_stratum(10, 10) == "S4"


def test_per_sequence_cap_flooring_boundary():
    records = [(i, f"seq-{i % 20}", 0, 10) for i in range(40)]
    shard = build_sampling_manifest(records, shards=1, windows_per_shard=20,
                                    both_reference=0, engaged_reference=1)["shards"][0]
    assert shard["windows_per_sequence_cap"] == 1
    assert shard["max_windows_from_one_sequence"] == 1


def test_per_sequence_cap_enforced():
    records = [(i, f"seq-{i % 20}", 0, 10) for i in range(120)]
    result = build_sampling_manifest(records, shards=2, windows_per_shard=60,
                                     both_reference=0, engaged_reference=1, tolerance=0)
    cap = 3
    maxima = [shard["max_windows_from_one_sequence"] for shard in result["shards"]]
    assert cap in maxima
    assert all(maximum <= cap for maximum in maxima)


def test_cap_infeasibility_fails_closed():
    records = [(i, f"seq-{i % 2}", 0, 10) for i in range(20)]
    with pytest.raises(ValueError, match=r"stratum S1 allocation 20 selected 2;.* 2"):
        build_sampling_manifest(records, shards=1, windows_per_shard=20,
                                both_reference=0, engaged_reference=1)


def test_shards_disjoint_with_expected_union():
    result = build_balanced()
    check = result["shard_intersection_check"]
    assert check["disjoint"] is True
    assert check["union_size"] == 40
    assert check["pairwise_intersection_sizes"] == {"0-1": 0}


def test_both_coverage_gate_fails_closed():
    records = [(i, f"seq-{i}", 0, 10) for i in range(20)]
    with pytest.raises(ValueError, match=r"E_COVERAGE_BOTH_FRACTION:"):
        build_sampling_manifest(records, shards=1, windows_per_shard=20,
                                both_reference=0.5, engaged_reference=1, tolerance=0.02)


def test_per_shard_both_coverage_gate_fails_closed():
    reference, tight_tolerance = 0.7, 0.05
    uniform = [(i, f"seq-{i}", 7, 10) for i in range(80)]
    learned = build_sampling_manifest(uniform, shards=4, windows_per_shard=20,
                                      both_reference=reference, engaged_reference=1, tolerance=1)
    target = set(learned["shards"][2]["window_indices"])
    elevated = set(sorted(target)[:16])
    records = [(i, f"seq-{i}", 9 if i in elevated else 7, 10) for i in range(80)]
    permissive = build_sampling_manifest(records, shards=4, windows_per_shard=20,
                                         both_reference=reference, engaged_reference=1, tolerance=1)
    deviations = {shard["shard_id"]: shard["coverage"]["absolute_deviation"]
                  for shard in permissive["shards"]}
    assert [shard_id for shard_id, value in deviations.items() if value > tight_tolerance] == [2]
    assert permissive["coverage"]["absolute_deviation"] <= tight_tolerance
    with pytest.raises(ValueError, match=r"E_COVERAGE_BOTH_FRACTION_SHARD: shard 2 "):
        build_sampling_manifest(records, shards=4, windows_per_shard=20,
                                both_reference=reference, engaged_reference=1,
                                tolerance=tight_tolerance)


def test_engaged_window_coverage_gate_fails_closed():
    records = [(i, f"seq-{i}", 0, 0 if i < 10 else 10) for i in range(20)]
    with pytest.raises(ValueError, match=r"E_COVERAGE_ENGAGED_WINDOW_QUANTIZATION:"):
        build_sampling_manifest(records, shards=1, windows_per_shard=20,
                                both_reference=0, engaged_reference=1, tolerance=0.02)


def test_zero_engaged_frame_denominator_fails_closed():
    records = [(i, f"seq-{i}", 0, 0) for i in range(20)]
    with pytest.raises(ValueError, match="E_COVERAGE_ZERO_ENGAGED_FRAMES"):
        build_sampling_manifest(records, shards=1, windows_per_shard=20)


def test_largest_remainder_sums_exactly():
    allocation = largest_remainder({name: 1 for name in ("S0", "S1", "S2", "S3", "S4")}, 7)
    assert sum(allocation.values()) == 7
    assert allocation == {"S0": 2, "S1": 2, "S2": 1, "S3": 1, "S4": 1}
