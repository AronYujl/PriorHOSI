"""Deterministic acceptance of waypoint windows against joint quality margins."""


DEFAULT_LIMITS = {
    "contact_anchor_m": 0.01,
    "contact_surface_distance_m": 0.01,
    "support_speed_m_per_s": 0.01,
    "human_scene_RMS_cm": 0.1,
    "object_scene_RMS_cm": 0.1,
    "human_occupied_fraction": 0.005,
    "object_occupied_fraction": 0.005,
    "contact_fraction": 0.05,
}


def assess_waypoint_quality(metrics, baseline, limits=None, min_speed_ratio=0.95):
    """Return acceptance and named failures for one waypoint against its W0 baseline."""
    limits = DEFAULT_LIMITS if limits is None else limits
    failures = []
    for name, limit in limits.items():
        delta = metrics[name] - baseline[name]
        if name == "contact_fraction":
            if delta < -limit:
                failures.append(name)
        elif delta > limit:
            failures.append(name)
    if metrics["root_directed_m"] < 0.01:
        failures.append("root_directed_m")
    if metrics["object_directed_m"] < 0.0:
        failures.append("object_directed_m")
    if metrics["world_joint_speed_m_per_s"] < min_speed_ratio * baseline["world_joint_speed_m_per_s"]:
        failures.append("world_joint_speed_retention")
    return {"accepted": not failures, "failures": failures}


def accept_waypoint_pair(positive, negative):
    """Require both signed waypoint conditions to pass before retaining the pair."""
    return bool(positive["accepted"] and negative["accepted"])
