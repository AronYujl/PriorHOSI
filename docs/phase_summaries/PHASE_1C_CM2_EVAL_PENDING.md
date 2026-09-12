# Phase1C CM2.3: quality readout complete, timing awaiting resources

2026-09-12. CM2 has a confirmed holdout safety failure and observed physical
regressions. Keep corrected R2+CG. Complete acceptance remains pending student
serial U/G latency and the registered combined ratio/latency gate. Phase1C is open.

## Completed scope

Training completed58678 updates/120172544 windows, process exit0. The final epoch089
student equals all218 tensors of the terminal resume model. All469424 gradient/cfg
records and46512 endpoint-loss records are finite;40 clipping events are retained.
Epoch004 remains the fixed diagnostic checkpoint, never a selection candidate.

Six successful GPU workloads completed: internal U/G each60 episodes/364 windows;
final U/G each375 episodes/2271 windows; frozen Table3; position/FK recovery covering
GT and final U/G,375 each. There are37 paired reports and4 cached paired FID
comparisons. Seed42, corrected interpolation and all registered data/metrics remain.

R2/CM1 reference views reuse feature embeddings/gallery records while rebuilding
physical groups from corrected native metrics. Old physical confidence intervals
are not copied into these views. GT representation replay is exactly equal on375
rows. Frozen encoder continuity and GT embedding equality pass; R@3 retains224
gallery query occurrences from148 unique sequences.

## Results

| Guided metric | R2 | CM1 | CM2 |
|---|---:|---:|---:|
| Penetration ratio |0.0218843|0.0281318|0.0293217|
| FS |0.272221|0.288150|0.291540|
| Boundary jerk |142.68556|145.74579|153.99359|
| Exterior contact |365.63099|319.96180|318.06604|
| Body21 direct-position/FK disagreement cm |0.887031|1.656058|1.692449|
| FID |40.04968|22.90380|17.45352|
| MM-Dist |8.90005|8.25482|7.53117|
| R@3 |0.433036|0.450893|0.473214|

Against CM1, guided penetration and boundary jerk regress with pointwise95% paired
intervals excluding0. FS/contact changes are uncertain. Body disagreement also
increases in both U/G. Guided FID and MM-Dist improve; R@3 improvement is uncertain.
Unguided FID17.44221 improves at the point estimate against CM1, but its interval
crosses0. Complete U/G, group, internal, Diversity/MultiModality and uncertainty
readouts are retained in the report/compact.

Guided holdout355 has9 >5g episodes/18 frames, exceeding the8-episode limit; two
low-pelvis walks meet the limit2. Full375 has10 >5g episodes/27 frames and two low
walks. Unguided has none of these failures. Safety alone rules out promotion;
unmeasured speed must still be completed under the registered protocol.

## Operational recovery and budget

The initial position-fk run failed before reconstruction because its caller omitted
the existing component's required GT cohort. All8 process exits1 and its manifest
remain. The new r1 id supplies GT and gives each shard one visible physical GPU,
matching the component's logical cuda:0 convention. All8 shards and merge exited0.
The GT reconstruction is retained as a control. No completed motion generation was
repeated and no runtime code changed.

Preparation/training157.224198GPU-h plus completed evaluation, including failure,
2.793511GPU-h totals160.017709. Actual already-started workload durations exceeded
the160 estimate by0.017709; this is recorded without raising the ceiling silently.
The asynchronous resource request is unanswered. The earlier conservative request
was168 to retain the original8GPU-h evaluation reservation; measured remaining work
is about0.1GPU-h, so a161 total ceiling should suffice. New GPU work is paused;
CPU comparisons have completed. Timing has no placeholder or borrowed student result.

## Source and artifacts

Execution source deef2d8; runtime matches the previously checked code. Authority487
passed/3 skipped and targeted66 are reused.49 initial configs and9 recovery configs
are archived; recovery is a corrected invocation, not an evaluator code change.
All7 terminal workloads (6 success,1 failure) are registered. This handoff records
an actual resource stop and failure recovery, not a completed scientific acceptance.

- experiments/results/p1_hsi_cm2_training_s42_20260912.json.
- experiments/results/p1_hsi_cm2_acceptance_partial_s42_20260912.{md,json}.
- results/cm2_evaluation_setup_20260912/resource_pending.json.
- Same setup: completed_workloads.json, jobs.json, references.json,
  representation_recovery.json, readout_scheduling.json, gt_representation_replay.json.
- Same setup: paired_native/ (23), paired_features/ (8 plus4 cached FID pairs),
  paired_representation/ (6). Existing outputs are immutable references.
- results/hsi_cm2_table3_s42_20260912/ and hsi_cm2_position_fk_r1_s42_20260912/.
- Each results/experiments/p1-hsi-cm2-*-s42-20260912/ contains its resolved config,
  preflight, logs and completed/failed manifest.

## Exact continuation

Read this summary, OVERVIEW.md and the CM2.3 section of PHASE_1C_HSI.md. Obtain the
pending resource ceiling decision before starting another GPU workload. Then run
only latency-unguided, latency-guided and gate from jobs.json through the existing
execute_jobs.sh and experiment.py lifecycle. Their manifests have not been created;
check that state before launch. Student timing is planned serially on physical GPU1,
with logical cuda:0 and fresh contention snapshots. Teacher latency is reused.

Both timing jobs create setup/latency_unguided.json and latency_guided.json links
required by the gate. Consolidated completed_workloads.json already contains all7
terminal jobs; the remaining launcher appends its results. Reuse all completed GPU
workloads and37 comparisons, then produce the final CM2.3 report and completion
summary. No new training, new loss or next phase is authorized by this continuation.
