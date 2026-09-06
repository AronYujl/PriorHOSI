# Phase 2.7 source stance displacement target — 2026-09-06

**Deliverable PASS; pilot quality FAIL.** Both adjusted object-scene depth
comparisons pass. The human-scene frame-prevalence primary remains unresolved
at episode level. A00-increment also raises its common objective and significantly
worsens native sliding against reconstruction. Retain reconstruction and the
fixed expert pair; the correction recipe remains unpromoted.

## Scope and implementation

The user approved A00-increment/A01-increment, 28 episodes each. Reuse Phase 2.6
A00-height/A01-height and Phase 2.3 reconstruction. Fix R2 final EMA+CG,
P15 online+Arm B, seed 42, bins 0/22/44/66, seven objects, 500 diffusion steps,
499 CG calls/window, corrections 10/1/0, 20 Adam steps at .05, all bounds/scales,
contact labels and source-height mask rules. Forced floor and the HSI target
factor remain off; A01 includes the existing human/object geometry objective.

One config enables `source_stance_velocity`. For detached source FK feet and
corrected feet, the horizontal correction is d=(p_corrected-p_source)_xz. The
new stance energy is mean_M ||d[t]-d[t-1]||^2/(2*.02^2) on future transitions
1→2 through 14→15 and joints 7/8/10/11. Source feet, estimated floor and mask
stay fixed through the 20 optimizer steps. The default target remains zero
velocity for sealed configurations. Raw `stance_displacement_cm` is retained;
`stance_increment_cm`, its before value, the optimized energy and target mode
are recorded. Expert/core code and the native evaluator stay fixed.

## Native results and gates

Contact and HS frames are percentages. HS/OS s_mean is the native frame-average
sum of penetrating vertex depths; FS uses the native weighted-displacement scale.
Completion requires both endpoint errors below 10 cm and measures the atomic task.

| Row | Complete | Contact % | FS | HS frames % | HS mean | OS mean |
|---|---:|---:|---:|---:|---:|---:|
| reconstruction | 22/28 | 68.466 | 0.129277 | 34.203 | 4.014550 | 30.533886 |
| a00_height | 22/28 | 69.183 | 0.140208 | 36.347 | 4.562122 | 30.920008 |
| a01_height | 22/28 | 69.038 | 0.140129 | 38.794 | 2.060411 | 20.910681 |
| a00_increment | 22/28 | 68.054 | 0.139963 | 33.820 | 3.940869 | 30.645320 |
| a01_increment | 22/28 | 68.094 | 0.160854 | 35.998 | 2.212554 | 21.285703 |

The primary family uses 10,000 seed-42 paired replicates and Bonferroni 98.3333%
percentile intervals. Both episode (28) and scene (4) units must pass.

| Primary contrast | Delta | Episode adjusted CI | Scene adjusted CI |
|---|---:|---|---|
| A01-increment − A01-height, HS frame ratio | -0.027963 | [-0.064085, +0.002678] | [-0.064548, -0.000209] |
| A01-increment − A00-increment, OS mean | -9.359617 | [-19.907682, -1.923776] | [-12.535319, -5.229397] |
| A01-increment − reconstruction, OS mean | -9.248183 | [-19.728026, -1.922425] | [-12.334323, -5.138535] |

HS prevalence falls 2.7963 percentage points against A01-height, with its
adjusted episode interval crossing zero. The nominal 95% episode interval
[-5.7625,-0.2615] points and scene interval [-6.0684,-0.3941] points describe
positive secondary evidence. The preregistered adjusted primary stays unresolved.
OS mean improves 30.29% against reconstruction, with both adjusted comparisons
passing at both units. Compared with A01-height, OS mean rises .375022 and
FS rises .020725; both nominal comparisons are unresolved at both units.

Candidate contact/completion point protections and the registered candidate
nominal FS/HS protections pass. Every episode's completion result matches
reconstruction, 22/28 throughout. A01 contact falls .9444 points from A01-height
and .3718 from reconstruction. The former decrease is significant only at the
scene unit. A01 FS exceeds reconstruction by .031576, with nominal episode CI
[-.034932,.099395] and scene CI [-.046519,.109672]. Passing a protection defined
by absence of significant harm gives no equivalence conclusion.

The A00 control's FS increase versus reconstruction is .010686 (8.27%), nominal
episode CI [.005608,.016591] and scene CI [.005296,.016077]. Its endpoint object
error also rises .093579 cm at both units. These adverse secondary outcomes stay
visible even though A00's FS was outside the registered candidate protection.
A00 versus A00-height improves HS frame ratio, mean and maximum at both units;
its contact falls 1.1283 points, significant only at scene level. A01 improves
HS depth mean/maximum and OS maximum versus reconstruction and new A00 at both
units. All 16 native outcomes and both-unit intervals are in the compact result,
including findings significant at just one unit and the source-target×geometry
factorial. The interaction for endpoint object error is -.165138 cm and for
hand-penetration ratio -.001336, significant at both units; its FS, HS-frame and
OS-mean interactions remain unresolved.

## Registered diagnostics and the common optimizer

All 744 initial stance energies and initial stance increments are exactly zero.
Saved motions independently reproduce the increment telemetry/optimized energy
and raw displacement. This verifies the objective definition and recording.

The complete A00 objective comprises residual, contact, stance and endpoint
terms. It increases in **336/372 corrections**, ties in 36 and decreases in zero.
Its episode-first initial mean is 3.9373e-13, arising from the contact term,
and its final mean is .0479565. Final terms are residual .0003516, contact
.0044081, stance .0429265 and endpoint .0002702. A00 therefore departs from an
already near-zero initial objective despite the new stance target. The logs
measure the optimizer departure; they do not separate numerical perturbation,
the finite Adam trajectory or another cause. This is the exact next review target.
A01 increases its common subtotal in 363/372 corrections, but it also optimizes
scene terms; its common subtotal alone is not its complete objective.

| Episode-first correction measure | A00-increment | A01-increment |
|---|---:|---:|
| Source selected displacement, cm | 3.056445 | 3.048475 |
| Corrected selected displacement, cm | 3.140412 | 3.119567 |
| Selected displacement increment RMS, cm | 0.533344 | 0.526288 |
| FS proxy change inside source mask | +0.026928 | +0.038096 |
| FS proxy change outside source mask | -0.000400 | +0.008209 |
| Mean signed toe vertical correction, cm | +0.049169 | -0.120935 |
| Mean absolute toe vertical correction, cm | 0.064604 | 0.399258 |
| Source-mask transition count / 56 | 39.9455 | 39.9334 |
| Corrected same-floor count / 56 | 39.7342 | 39.5990 |

The FS proxy uses height-weighted horizontal displacement. Source and corrected
motion use the same recorded episode floor for this diagnostic, while mask
membership uses the frozen source floor. Later source trajectories differ across
rows; averaging selected transitions across rows changes their membership.

A00 has 5/372 empty masks, with 14,717 source memberships, 102 lost and 27 added
under the same frozen floor. A01 has 7/372 empty masks, 14,657 memberships,
248 lost and 189 added. Its smalltable episode in scene a3df624b uses the
registered zero-floor fallback twice (window 30, steps 10 and 1) because it has
zero low-speed samples. Source floor means are 5.6820/5.6913 cm; episode floor
ranges average 3.9534/4.0905 cm and absolute differences from final native floors
average 2.3365/2.3778 cm. These are estimate discrepancies with distinct temporal
and representation support. Coverage and smaller corrections alone establish
neither physical support labels nor motion realism.

A01 human-scene objective decreases .003779→.001449 and object-scene objective
.049946→.027484; joint occupied fraction decreases .005390→.004225. The native
evaluation uses mesh vertices and SDF, so these sparse-joint quantities do not
locate or fully describe native penetrating frames. All 455 diagnostic means
per row, including steps 10/1/0 and initial/generated-history strata, are retained.

## Execution, verification and provenance

Eight GPU jobs and automatic analyses exited zero. All 56 episodes, 248 windows,
744 corrections, 14,880 optimizer steps and 123,752 CG calls completed. Native
metrics, optimizer scalars and saved correction states were finite, with exact
history/contact and complete native pairing. Saved evaluated joints reproduce
native FS in every episode. The 56 motion files total 49,128,600 bytes.

GPU job wall time was 1926 s (32m06s); manifest start through analysis completion
was 1959 s (32m39s). Each row has 5376 frames. Generation sums are 5260.34/5290.69 s;
correction sums 214.97/214.28 s, and peak allocated memory 804.43 MiB. Sharded
timing describes this execution; it cannot establish isolated production latency.
The formal run supplied the registered batch-1 functional and compute validation.

Implementation suite: **922 passed, 4 skipped in 163.29 seconds**.
Completion suite: **922 passed, 4 skipped in 163.87 seconds**. Registry valid
with 352 records; all native pairs complete, saved-native FS and increment audits
pass, and diff check passes.

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
export ROOT_DIR=/data/yujinlun/InfBaGel-mixer
"$INFBAGEL_PYTHON" -m pytest tests -q
"$INFBAGEL_PYTHON" tools/experiment.py validate
git diff --check
```

Preregistration: 2a56b34. Implementation and complete GPU/analysis source:
e76aec3; manifest start and finish record the same source. The completion commit
contains this handoff, compact result, plan and registry row. Integrate by
fast-forward into phase/02-mixer and tag exp/p2g-stance-increment-v1 after final
verification; the tag records the valid experiment and its failed quality gate.

- Compact: `experiments/results/p2_mixer_stance_increment_s42_20260906.json`.
- Run: `results/experiments/p2-mixer-stance-increment-s42-20260906/`.
- Preserve manifest, resolved configs, preflight, logs, raw motions, native and
  diagnostic projections, paired/factorial/adjusted reports, runtime and closure
  audits. Existing sealed input identities are reused by manifest reference.
- Baselines: Phase 2.6 height rows and Phase 2.3 reconstruction; earlier failures
  and negative results remain intact.

## Exact next entry

Retain reconstruction, R2+CG/P15+Arm B and forced-floor exclusion. Read this
handoff, `docs/plan/OVERVIEW.md` and the current Phase 2 plan. Begin with a
read-only review of A00's optimizer at its near-zero source objective and its
336 objective increases. Any further mechanism, changed optimizer or workload
needs its own concrete approval and dated preregistration. Full Phase 2, useful
HSI supervision, motion realism and learned training remain open. Close only 2.7.
