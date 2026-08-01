# Phase 1B protocol decomposition P1: what the released baseline's advantage was made of

## Scope and final outcome

This sub-phase is an inference-only measurement. It trained nothing, wrote no
checkpoint, selected no checkpoint, changed no model or evaluator code and did
not touch the native D2 protocol. Its single question had never been attributed
anywhere in the Phase 1B record: how much of the released InfBaGel row's
advantage over the autonomous D2-* rows is the model, and how much is the
inference protocol the released row was measured under.

The released row (`p0-hoi-table5-baseline-s42-20260712`) was produced at commit
`c358fa4` with the consistency sampler at `cm_timesteps: 16`, `guidance_weight:
1`, CFG `w: 1` and scene/object-voxel conditioning, and before the 2026-07-13
rewrite of the autoregressive rollout block in `code/test_infbagel_hoi.py`.
Every D2-* row is 500-step unguided ancestral DDPM, no CFG, `load_scene=false`,
after that rewrite. The two rows differ on at least seven axes at once.

Classification: `released-baseline-protocol-attribution-guidance-dominant`.
The preregistered decision rule landed in its first branch, `g >= 0.5`.

Plain conclusion, none of it softened:

1. **Guidance dominates the recorded gap.** Inference-time guidance accounts for
   **59.20%** of the released-minus-D2-X contact-recall gap and **65.49%** of the
   contact-F1 gap. Iteration count (16 steps vs 1) is a **null** effect on
   contact F1: its paired CI crosses zero under both rollouts. The rollout
   rewrite is a **null** effect on all three contact-quality metrics.
2. **HOIPrior's raw generative geometry is nevertheless genuinely worse.**
   Protocol-matched, with both models unguided, D2-X sits **0.8334 cm
   [0.4906, 1.1824]** further from the object at GT-contact frames. The released
   model's 1.6587 cm guided advantage therefore splits roughly half real prior
   quality, half inference-time guidance purchase.
3. **D2-AG's F1 gain was an operating point, not better geometry.** D2-AG is
   geometrically indistinguishable from D2-X (**+0.0351 cm**, CI crosses zero)
   even though its contact F1 is higher.
4. **The failure mode is gross misses, not near misses.** **57.94%** of D2-X's
   failed GT-contact frames are at **>= 8 cm**, against **16.93%** in the
   `[5,6)` cm bin. This is evidence **against** the earlier "the model knows when
   to make contact and misses by a few cm" framing in its literal form.

A separate scoring artifact was found and is recorded below: every contact-recall
number in this project is depressed by the constant factor `397/438 = 0.9064`.

Nothing here authorizes a new lever. Production guidance and stepwise object
requery for HOIPrior remain unauthorized and still require a separate dated plan
amendment and explicit user approval. Phase 1B HOIPrior search stays closed.

## Design and executed configuration

Released checkpoint as baseline only, never as an initializer; official 438
sequences x 3 windows; seed 42; 2 commits x 3 knob settings = six pure-inference
runs, plus two HOIPrior native evaluations added to obtain an absolute
GT-contact-frame distance, a quantity `code/eval_metrics.py` computes internally
but never returns.

| cell | commit | sampler | cm_timesteps | guidance_weight | role |
|---|---|---|---:|---:|---|
| A0-old | `c358fa4` | consistency | 16 | 1 | reproduction gate; the published protocol |
| A-old | `c358fa4` | consistency | 16 | 0 | guidance share |
| B-old | `c358fa4` | consistency | 1 | 0 | iteration + stepwise-requery share |
| A0-new | `5f7dde7` | consistency | 16 | 1 | rollout-rewrite share |
| A-new | `5f7dde7` | consistency | 16 | 0 | guidance robustness |
| B-new | `5f7dde7` | consistency | 1 | 0 | pure NFE share |
| D2-X probe | `5f7dde7` | 500-step unguided diffusion | — | — | sealed autonomous control |
| D2-AG probe | `5f7dde7` | 500-step unguided diffusion | — | — | D2-AG geometry |

`-old-` ran in the pinned worktree `/data/yujinlun/InfBaGel-c358fa4-baseline`,
`-new-` and both probes in `/data/yujinlun/InfBaGel-head-baseline` (the three
`-new-` runs without `save_motion_params` are in the main checkout). The
released `checkpoint.pth` is byte-identical in all three worktrees, SHA-256
`6853351b3f5468d21293d12a6dd2bccde62c1400fdb83934f7878136b131b198`, equal to the
sealed Phase-0 record. `code/eval_metrics.py` and `code/guidance_loss.py` are the
same Git blobs at `c358fa4` and at HEAD (`575602e`, `44fd497`; the guidance file
hashes `5747721bcc015911c7999692079666f7e5d6204912761d91b8b82d4ba4c4ab24`), so
neither the metric definitions nor the guidance formula is a confounded axis.
`code/test_infbagel_hoi.py` is not: 171 insertions / 59 deletions separate the
two commits, which is exactly the seventh axis cell A0-new measures.

Per the plan's authorized lifecycle simplification, this measurement allocated no
run id and produced no `tools/experiment.py start` manifest. Traceability rests
on the executed commit, the archived Hydra resolved config and overrides, the
fixed official test set and byte-identical metric code.

## Reproduction, determinism and independent verification

- **Reproduction gate, passed in its strongest form.** A0-old at `c358fa4`
  reproduced the sealed published row bit-exactly: all 18 aggregate metrics are
  exactly equal, of which 16 are the non-ratio subset and the remaining two are
  `hand_pen_ratio` and `human_pen_ratio`.
- **Determinism.** Each of the six cells was re-run once with
  `save_motion_params=true`. 108/108 metrics are bit-identical, across two
  different worktrees.
- **Independent extraction.** A per-sequence extractor was written that recomputes
  contact statistics from the exported prediction/GT npz and the dumped object
  motion parameters, without reading any aggregate. It reproduces 36/36 pooled
  metrics on the six released runs (`contact_f1`, `contact_recall`,
  `contact_precision`, `contact_percent`, `gt_contact_percent`, `foot_sliding`).
  The seventh pooled quantity, `feet_height`, agrees only to float32-vs-float64
  accumulation (largest absolute difference `3.9e-09`) and is deliberately not
  counted as a bit-exact reproduction. The independent GT-contact-distance
  pipeline additionally reproduces all eight runs' own published
  `contact_recall` exactly (8/8) and matches
  `evaluation/per_sequence_metrics.json` on every sequence for the five runs
  that have one (`a0-new`, `a-new`, `b-new`, `d2x`, `d2ag`), 0 mismatches.
- **Cross-host agreement.** Both HOIPrior probes were run on this single host
  against the same final-online checkpoints and reproduced the sealed
  worker-produced aggregates bit-exactly, 18/18 each. Single-host evaluation
  therefore matches the retired two-host worker results.
- **Uncertainty.** Paired sequence-level bootstrap, seed 42, 10,000 replicates,
  n = 438, using the `tools/summarize_hoi_phase1b.py:112` convention; the
  resampling index matrix was asserted bit-identical to the three existing
  implementations in this repository (`summarize_hoi_phase1b.py:112`,
  `run_hoi_d2n.py:383`, `run_hoi_d2af_eligibility.py:96`).

## Results: the decomposition

Point estimates with paired 95% bootstrap CIs.

| contrast | contact recall | contact F1 | foot sliding (lower better) |
|---|---|---|---|
| guidance (A0-old − A-old) | +0.0788 [+0.0649, +0.0931] | +0.0588 [+0.0475, +0.0707] | −0.0357 [−0.0505, −0.0209] |
| guidance (HEAD rollout) | +0.0836 [+0.0687, +0.0988] | +0.0612 [+0.0491, +0.0737] | −0.0496 [−0.0650, −0.0344] |
| iteration cm16 → cm1 | +0.0085 [−0.0046, +0.0213] | +0.0029 [−0.0079, +0.0137] | +0.0184 [+0.0046, +0.0319] |
| rollout rewrite | −0.0084 [−0.0190, +0.0022] | −0.0047 [−0.0130, +0.0035] | +0.0092 [−0.0026, +0.0210] |

Shares of the released-minus-D2-X gap:

| quantity | gap | guidance | iteration | genuine model |
|---|---:|---:|---:|---:|
| contact recall | 0.13313 | 0.07882 (59.20%) | 0.00848 (6.37%) | 0.04583 (34.43%) |
| contact F1 | 0.08983 | 0.05883 (65.49%) | 0.00290 (3.23%) | 0.02810 (31.28%) |

`g = 0.5920 >= 0.5` puts the decision in the preregistered first branch: the gap
is mainly inference-time guidance and past released-vs-D2 comparisons were not
fair comparisons. The branch permits *proposing* protocol-aligned guidance for
HOIPrior later at equal budget with foot sliding, object translation MAE and
pelvis goal reported alongside; it does not enable it.

Two further facts belong in the record because they cut against earlier claims:

- The rollout rewrite is null on contact precision, recall and F1, but it is
  **not** null on predicted `contact_percent`: −0.0122 [−0.0213, −0.0031]. The
  rewrite changes how much contact the model asserts, without changing how well
  that contact scores.
- Guidance **improves** foot sliding in both rollouts, the opposite direction to
  the D2-Q0 measurement (ratio 1.5350). The source explains it:
  `code/priors/contact_guidance.py` (D2-Q0) and `code/priors/routed_guidance.py`
  (D2-R0) reference `apply_feet_floor_contact_guidance` zero times. Both
  implemented only the author's hand-object x10 term and omitted the feet-floor
  x500 term. The historical D2-Q0/D2-R0 foot-sliding cost is a property of those
  two partial implementations, not of the author's guidance.
- Protocol-matched, D2-X contact **precision** is at or above the released model:
  0.78806 vs 0.78115.

## Results: HOIPrior's absolute contact geometry

The probes measure, per GT-contact frame, the minimum over the two SMPL hand
joints (22, 23) of the distance to the nearest vertex of the posed *predicted*
object mesh, restricted to frames where either GT hand is within the 5 cm
threshold — the mask at `code/eval_metrics.py:271`, and the accumulator at
`code/eval_metrics.py:264-283` that `compute_hand_object_interaction` computes
but never returns. It is measured from a skeleton joint, not the hand surface,
and to the nearest mesh vertex, not the nearest surface point; both the
prediction and the GT leg use the identical definition, and the GT-vs-GT floor of
the metric is 1.6981 cm frame-pooled / 1.7655 cm sequence-mean.

| paired contrast | Δ mean cm | 95% CI | crosses zero |
|---|---:|---|---|
| D2-X − A0-old (guided released) | +1.6587 | [+1.3050, +2.0182] | no |
| D2-X − A-old (both unguided) | +0.8334 | [+0.4906, +1.1824] | no |
| D2-X − B-old (unguided, one step) | +0.6887 | [+0.3331, +1.0360] | no |
| D2-AG − A-old | +0.8686 | [+0.5023, +1.2660] | no |
| D2-AG − D2-X | +0.0351 | [−0.3247, +0.4150] | **yes** |

Protocol matching removes about half of the apparent geometric deficit
(0.8334 / 1.6587 = 50.2%) and the remaining half survives with a CI excluding
zero. So the correct statement is neither "the released advantage was all
protocol" nor "HOIPrior is as good and only scored worse": HOIPrior's raw
generative geometry is genuinely worse, by 0.83 cm, and the other 0.83 cm was
bought at inference time.

D2-AG's higher contact F1 came from where its operating point sits relative to
the 5 cm threshold, not from putting hands nearer objects: its mean distance is
statistically indistinguishable from D2-X's.

## Results: the failure mode is gross misses

Conditional on a GT-contact frame failing the 5 cm threshold:

| run | [5,6) cm | [6,8) cm | >= 8 cm |
|---|---:|---:|---:|
| D2-X | 16.93% | 25.13% | **57.94%** |
| D2-AG | 18.44% | 26.20% | 55.36% |

The residual contact deficit is not a population of near misses that a small
geometric nudge would convert. The majority of failures are at or beyond 8 cm,
which is not a threshold problem. Record this explicitly: it is evidence
**against** the earlier "knows when, misses by a few cm" framing in its literal
form, and it means a mechanism whose whole benefit is sub-centimetre refinement
cannot recover most of the missing recall.

## Scoring artifact affecting every recall in the project

41 of the 438 official sequences contain no GT-contact frame at all.
`code/eval_metrics.py:316-320` scores `contact_recall = 0` when `TP + FN == 0`,
and the aggregate is an unweighted mean over all 438 sequences. Every
contact-recall value ever reported in this repository is therefore depressed by
the constant factor `397/438 = 0.9064`; absolute values and differences are
compressed by the same factor, so a true difference is the reported one times
`1.1033`. Shares and rankings are unaffected. The evaluator was not changed and
no existing number was rewritten.

## Errata caused by this decomposition

These are registered here and in the compact result. No sealed hash-bound
artifact was edited, and `docs/phase_summaries/PHASE_1B_D2AG.md` was not touched.

1. **The registered contact-F1 minimum `0.6598838781` is a cross-protocol
   constant.** It was `F1_X + 0.25 * (F1_released − F1_X)` against the *guided*
   released row. The F1 gap 0.08983 is guidance 0.05883 (65.5%), iteration
   0.00290 (3.2%) and genuine model difference 0.02810 (31.3%), so the
   protocol-aligned 25% closure threshold is about `0.64445`.
2. **Protocol-aligned gap closure.** Recomputed against the unguided one-step
   released row: D2-AG `0.451` (recorded `0.1409`) and D2-AC `0.376` (recorded
   `0.1176`) both clear 0.25; D2-AE `0.161` and D2-AF `0.129` still fail; D2-AD
   is negative under both conventions. **No run's overall decision changes** —
   D2-AG still fails on its paired F1 CI lower bound `−0.0082` and its
   foot-sliding ratio CI upper bound `1.184`. What changes is the recorded set
   of failure reasons.
3. **`released_95_percent_effectiveness` contains a protocol component.** For
   D2-AG the contact-F1 ratio `0.894` fails against the guided released row but
   is `0.973` and passes against the protocol-matched unguided row; recall moves
   `0.823 -> 0.923` and foot sliding `1.203 -> 1.086`, both still failing.
   "11 checks, 6 passed" is therefore not a pure model statement.
4. **`experiments/results/p1_hoi_phase1b_d2aa_table5_completion_s42_20260724.json`
   `.local_protocol.native_quality` is false for the released row that file
   contains.** It declares `sample_type: diffusion, diffusion_steps: 500,
   guidance: false, cfg: false, scene: false`, while the table includes the
   released row, which was produced with the consistency sampler at
   `cm_timesteps: 16`, `guidance_weight: 1`, CFG `w: 1` and scene/object-voxel
   conditioning. That file is hash-bound by three registry rows and four
   documents, so per existing precedent it is not edited in place; the fact is
   registered instead.

## Verification

- Eight `evaluation/aggregate_metrics.json` files read directly; no metric in
  this summary was retyped from prose.
- A0-old vs the sealed `p0-hoi-table5-baseline-s42-20260712` aggregate: 16/16
  non-ratio and 18/18 all metrics exactly equal.
- Six `-ps-` determinism replicates: 108/108 metrics bit-identical.
- Independent extractor: 36/36 pooled metrics, 8/8 recall identities, 5/5 runs
  with zero per-sequence mismatches against `per_sequence_metrics.json`.
- Both probes vs their sealed worker aggregates: 18/18 each.
- `code/eval_metrics.py` and `code/guidance_loss.py` Git blobs identical between
  `c358fa4` and HEAD; released checkpoint SHA-256 identical in all three
  worktrees.
- `python tools/experiment.py validate` passes with 239 registry records.

## Artifacts

- Compact result:
  `experiments/results/p1_hoi_protocol_decomp_s42_20260801.json`, SHA-256
  `e44538fa819c14ad6f407015118a331cbe8f8ca01890a8a8b6115301bb7f2d1b`.
  It carries the eight runs with their identifying config and full aggregates,
  every paired CI, the distance statistics and histograms, the decomposition
  shares, the decision rule and its branch, and the four errata.
- Registry completion row: `p1-hoi-protocol-decomp-completion-s42-20260801`.
- Plan section: `docs/EXPERIMENT_PLAN.md`,
  "2026-08-01 Phase 1B 基线协议分解 P1（released baseline 协议归因，用户批准）".
- Run outputs (untracked, on this host):
  `/data/yujinlun/InfBaGel-c358fa4-baseline/results/experiments/p0-hoi-protocol-decomp-{a0,a,b}-old[-ps]-s42-20260801`,
  `/data/yujinlun/InfBaGel-release/results/experiments/p0-hoi-protocol-decomp-{a0,a,b}-new-s42-20260801`,
  `/data/yujinlun/InfBaGel-head-baseline/results/experiments/p0-hoi-protocol-decomp-{a0,a,b}-new-ps-s42-20260801`,
  `/data/yujinlun/InfBaGel-head-baseline/results/experiments/p1-hoi-{d2x,d2ag}-distance-probe-s42-20260801`.
- No tag was created and no merge is authorized by this sub-phase.

## Unresolved risks

- The two pinned baseline worktrees hold the only copies of the six released-cell
  outputs and both probe trees. They are untracked working directories, not
  recovered staging trees with a checksum pass; the tracked record is this
  summary plus the compact result.
- The distance metric is joint-to-vertex, not surface-to-surface. It is
  internally consistent across all eight runs and against the GT floor, but its
  absolute scale is not a skin-contact distance and must not be compared to
  externally published surface distances.
- The `397/438` recall artifact is described but not corrected anywhere. Any
  future absolute-recall claim must state which convention it uses.
- The six released-checkpoint cells and both HOIPrior probes exist only in the
  two untracked external worktrees; there is no checksum-passed staging tree for
  them. Their provenance rests on the executed git commit, the recorded command
  line and the archived Hydra `resolved_config`.

## Exact next entry point

Unchanged: Phase 1B HOIPrior search remains closed, no checkpoint is selectable,
and the only next-session entry is a dated Phase 1C HSIPrior plan-only
preregistration on `phase/01c-hsi`, trained from random initialization and never
loading the released, author, D2-X, D2-AC, D2-AD, D2-AE, D2-AF or D2-AG
checkpoints.

Any later reuse of a "released minus D2" gap, a gap-closure fraction or a
released-95% ratio recorded elsewhere in this repository must cite this summary
and treat that number as a cross-protocol quantity that overstates the model
deficit.
