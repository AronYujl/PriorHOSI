import pytest

from mixer.waypoint_protection import assess_waypoint_quality, accept_waypoint_pair


def _base():
    return dict(contact_anchor_m=0.02, contact_surface_distance_m=0.03,
                support_speed_m_per_s=0.10, human_scene_RMS_cm=0.20,
                object_scene_RMS_cm=0.20, human_occupied_fraction=0.10,
                object_occupied_fraction=0.10, contact_fraction=0.80,
                root_directed_m=0.02, object_directed_m=0.01,
                world_joint_speed_m_per_s=1.0)


def test_joint_gate_accepts_only_within_registered_margins():
    baseline = _base()
    assert assess_waypoint_quality(_base(), baseline)["accepted"]
    rejected = _base(); rejected["object_scene_RMS_cm"] += .1001
    result = assess_waypoint_quality(rejected, baseline)
    assert not result["accepted"] and "object_scene_RMS_cm" in result["failures"]


def test_pair_requires_both_signed_conditions():
    baseline = _base()
    good = assess_waypoint_quality(_base(), baseline)
    bad_metrics = _base(); bad_metrics["contact_fraction"] -= .0501
    bad = assess_waypoint_quality(bad_metrics, baseline, limits={**{
        "contact_anchor_m": .01, "contact_surface_distance_m": .01,
        "support_speed_m_per_s": .01, "human_scene_RMS_cm": .1,
        "object_scene_RMS_cm": .1, "human_occupied_fraction": .005,
        "object_occupied_fraction": .005}, "contact_fraction": .05})
    assert accept_waypoint_pair(good, good)
    assert not accept_waypoint_pair(good, bad)


def test_root_response_is_required():
    baseline = _base(); metrics = _base(); metrics["root_directed_m"] = 0.
    result = assess_waypoint_quality(metrics, baseline)
    assert "root_directed_m" in result["failures"]


def test_negative_object_response_is_rejected():
    baseline = _base(); metrics = _base(); metrics["object_directed_m"] = -1e-6
    result = assess_waypoint_quality(metrics, baseline)
    assert "object_directed_m" in result["failures"]
