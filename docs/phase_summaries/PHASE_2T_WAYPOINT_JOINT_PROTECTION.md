# Phase 2.22: bounded waypoint joint protection (2026-09-07)

The approved question was whether the confirmed signed waypoint response can be
used while preserving grasp relations and scene quality. A deterministic gate was
implemented in `code/mixer/waypoint_protection.py`; it compares each signed first
window with its frozen W0 baseline and requires both signs to pass the registered
root/object response, contact, support, scene RMS, occupancy and speed margins.

The completed Phase 2.21 records were replayed for all 22 applicable signed pairs.
W+ passed 9/22 and W- passed 8/22; both signs passed for tasks 18, 334, 374 and
375, giving 4/22 (18.18%) pair coverage. Rejection reasons were led by contact
anchor drift (13), support speed (6), human scene RMS (6), object scene RMS (5)
and object occupancy (4). The result is the same protected coverage already
observed in Phase 2.21, so the gate preserves quality by rejecting most proposed
waypoints rather than making the response itself safer.

No new GPU rollout, HSI forward, training, route-pool expansion or 441/469 run was
started. The compact result is
`experiments/results/p2_mixer_waypoint_joint_protection_s42_20260907.json`; the
protocol is
`experiments/protocols/p2_waypoint_joint_protection_s42_20260907.json`.

Verification: `pytest tests/phase2/test_waypoint_protection.py tests/phase2/test_waypoint_control.py -q`
passed 20 tests. This closes the bounded diagnostic with NO-GO for route-pool
expansion. It establishes no native quality improvement, HSI value or superiority
to InfBaGel. The next mechanism would need to change the generated trajectory or
the condition proposal while preserving these constraints; this session does not
start that work.
