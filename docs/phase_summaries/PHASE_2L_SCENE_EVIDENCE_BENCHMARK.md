# Phase 2.12 fixed full469 native benchmark — execution handoff

**Status: prepared for execution; scientific results pending.** The user approved
full469 native evaluation after the independent-development lambda26 calibration.
This handoff records the frozen protocol and exact recovery entry. It must be
updated with completed/failed results before integration or tagging this phase.

## Scope and configuration

Branch `phase/02l-scene-evidence-benchmark`. New config
`code/config/config_sample_hosi_evidence_benchmark.yaml` inherits the scene editor
and fixes lambda26,8 shards,469 tasks and native evaluate mode. Runtime, experts,
core and native metric code are unchanged from Phase2.11. Keep P15 online/Arm B,
R2 final EMA,500-step source generation,8 edit levels/iterations, beta_ref1,
original explicit terms/bounds/local Armijo and isolated teacher query RNG.

Generate three complete rows: full lambda26, lambda0 and reconstruct_only.
Reuse the sealed469 native HOI row by reference. That gives1407 new rollouts,
24 shard jobs, with all four local rows retained for analysis. The original
source identity has current native-chain test coverage and a previous exact
seven-episode comparison. No lambda or other parameter is selected on this test.
Previously observed28-task/four-scene results are exploratory history.

The dated Phase2.12 plan is authoritative for gates. Four primary HS/OS mean
contrasts (full versus lambda0, full versus native HOI) use98.75% Bonferroni paired
intervals at both469-episode and67-scene units. HSI evidence and total scene-quality
gates are separate. Protection margins at both units and against both references
are contact/completion loss at most2 percentage points and FS ratio at most1.10,
using nominal95% bounds. Six local pairwise comparisons retain all15 native
metrics plus completed and all nominal intervals. Directional counts, object
strata, completion discordance and ten largest HS/OS delta contributors are
reported without dropping tasks. Feet height has no registered preferred direction;
its increase/decrease/tie counts remain descriptive.

InfBaGel paper Table1 Hybrid1:0.5 is a separately labelled external reference:
success81.45%, FS.15, contact76.96%, HS mean3.17 and OS mean12.45. Its pre-repair
representation/reconstruction differs from current evaluation; it supplies no
paired significance or same-protocol superiority claim. Completion is endpoint
success, and s_mean remains frame-averaged summed penetrating-vertex depths.

## Verification and execution

Authority suite: **949 passed,4 skipped,166.94s**. Skips are two historical HSI
checkpoint-pair and two P8 evaluation artifact cases. All24 configurations resolve;
only registered lambda/mode, shard and output-directory differences occur.
GPU float64 adjusted-delta and paired-ratio intervals agree with the NumPy
reference in the archived mathematical check. Registry/diff checks precede the
clean-worktree manifest. No runtime source change, separate smoke or performance
workload is introduced.

Run directory:
`results/experiments/p2-mixer-evidence-benchmark-s42-20260907/`.
Run id `p2-mixer-evidence-benchmark-s42-20260907` is reserved for this execution.
Eight persistent GPU lanes each execute full, lambda0 and reconstruct in order,
followed by automatic native merge, complete per-episode trajectory/solver/domain
audit, saved-motion FS verification, original paired-bootstrap reports and GPU
adjusted/FS-ratio analysis. Source/config commits are recorded in the manifest.
Sharded timings describe the instrumented workload, not isolated production FPS.

Preserve execution_plan.json, resolved/, config_comparison.json,
analysis_math_verification.json, machine_preflight.json, manifest.json,
launch.sh, campaign.py, controller.log, per-lane/per-job exit records and every
per-episode audit/motion. Exact job commands are execution_plan.json's command
arrays. New rows merge to `<row>-merged08/`; complete analysis lives in analysis/.
The baseline and calibration/input manifests are referenced in the execution plan.
No unfinished shard can be merged as a complete row.

After initial stability (one full native episode per lane, finite metrics and
teacher gradients, exact history/contact, motion artifacts and peak allocation
below20GiB), report measured throughput/ETA and yield. The host-owned screen owns
all jobs and automatic analysis. No continuous polling or automatic restart is
required. Per-episode files persist; the native evaluator does not support
resuming a partially written shard into the same directory.

## Next entry: complete this phase

1. Inspect this handoff, OVERVIEW.md, the Phase2.12 plan and manifest.json.
2. Inspect campaign.exit.json / controller.log and all24 job exit codes. A success
   writes completion_metrics.json and seals the manifest. A technical failure
   writes failure_metrics.json, retains all artifacts and seals a failed manifest.
3. Preserve and report every native outcome, both primary gates, all protections,
   tail analyses, timing/call counts, representation/external-baseline limits and
   any failures. Unresolved or negative scientific gates complete the benchmark
   but do not authorize tuning or a new mechanism on the469 test tasks.
4. Write `experiments/results/p2_mixer_evidence_benchmark_s42_20260907.json`, update
   this summary and the plan, append the registry completion row, run completion
   authority/registry/diff checks, then make the completion commit and integrate/
   tag `exp/p2l-scene-evidence-benchmark-v1`. The tag seals the benchmark outcome;
   it does not override a failed wider Phase2 quality gate.

No subsequent phase starts in this session. Do not reuse this run id, overwrite
outputs, or alter source while this reportable campaign is active.
