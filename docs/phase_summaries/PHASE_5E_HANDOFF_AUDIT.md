# Phase 5.5.1 - Actual handoff and frame-budget audit

Date: 2026-09-10
Branch: `phase/05e1-handoff-audit`
Run: `p5-inference-handoff-audit-s42-20260910`
Commit: `817fa1a1a3cb6d8d09e866cb5aa8cf015788b858`

This subphase checked the first edge of the published `multitask-hosi-027`
episode using the existing Phase 2.34 `correct_terminal_draw0` artifact. It did
not sample either expert or Kimodo. The source body, scene, object, data index,
betas, task row and native reconstruction were checked before measuring the
handoff.

## Implementation

`code/mixer/multitask_handoff.py` adds a reusable handoff stage to the existing
Hydra dispatcher. It measures the declared native frame budget, removes the two
legacy held padding frames, evaluates the achieved object transform and body
geometry, checks both prescribed bridge contexts, and records explicit successor
blocking records. The config fragment is
`code/config/config_sample_hosi_handoff_audit.yaml`. Component tests cover frame
accounting, padding removal, actual object-state inheritance and every handoff
guard.

Each 16-frame stride-3 window contributes 42 new 30-Hz samples after the initial
0.1-second history. The published 9.8 s HOI budget permits 298 observed frames.
The cached eight-window export contains 340 observed frames plus two held padding
frames, equivalent to 11.2 s of new motion and 42 observed frames beyond the
published budget.

## Results

All source identity checks passed. Native SMPL-X reconstruction error was
`2.6656007889869215e-7` m. Both prescribed target contexts pass their membership
geometry, foot-support, body-identity and finite reconstruction checks when
placed against the achieved object transform. Their position/velocity continuity
is retained as a later execution metric and does not turn a context witness into
generated motion.

At the published budget cutoff, pelvis goal error is 12.285 cm, object goal error
is 24.843 cm, object translation speed reaches 0.382 m/s, object angular speed
reaches 0.905 rad/s, and the foot marker support check fails. The cached terminal
does reach the original goal-only metrics more closely, but it remains outside
the published timing contract: pelvis/object errors are 2.566/2.668 cm while
object translation/angular speed reach 0.617 m/s and 0.520 rad/s. The original
terminal repair record says `attempted=false`, `steps=0`; its goal completion flag
therefore does not establish a release-ready state.

The execution entry gate is false because the cached source exceeds the declared
budget and neither cutoff satisfies the full handoff guards. The walk and sit
segments are recorded as `blocked_by_predecessor`, with null generated metrics.
The run makes no full-chain success claim and leaves benchmark membership
unchanged.

## Verification and artifacts

The complete authority suite passed **1154 tests** with 4 historical skips. The
first suite attempt is retained as an environment failure caused by restricting
CUDA visibility while a pre-existing test addressed `cuda:6`; the all-device
rerun is the authoritative verification. The resolved job config and machine
preflight are beside the sealed manifest. The run contains per-cutoff contexts,
the complete source trace, a time-series object handoff plot and scene context
preview under `results/experiments/p5-inference-handoff-audit-s42-20260910/`.
The compact tracked result is
`experiments/results/p5_inference_handoff_audit_s42_20260910.json`.

## Gate and next entry

The diagnostic gate is complete, but the full-chain execution gate is not met.
Do not relax the timing or support thresholds and do not reuse the over-budget
cache as a successful chain. The exact next entry is `phase/05e2-multitask-
execution`: preregister a compatible actual-history HOI release/settling contract,
then run the walk and sit experts plus Kimodo with all transition frames scored.
