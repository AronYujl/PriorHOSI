# HSIPrior design priors

Status: active from 2026-08-10, for Phase 1C and any later expert training.

Phase 1B spent roughly one month and about thirty experiments on HOIPrior. Its
most valuable output is not a checkpoint — it is a set of measured negatives.
This file carries them forward as **binding defaults with their evidence and
their overturn condition**, so Phase 1C does not re-derive them at the same cost.

Two limits on how to read it:

- Every measurement below is on HOI data with an object-conditioned denoiser.
  Transfer to HSI is an *assumption*, stated as such. Where a prior is expensive
  to hold and cheap to test, the entry says how to test it.
- Adding to this file is cross-branch communication and needs the user's explicit
  approval (`AGENTS.md`, "Cross-branch communication").

Authoritative sources for every number: `docs/HOIPRIOR_EVIDENCE_INDEX.md`, the
phase summaries it names, and the compact JSONs under `experiments/results/`.

---

## 1. Do not search architectures. Budget is the one measured lever.

**Evidence.** Eleven model-side interventions were controlled negatives: D2-I/J/K/L
(gradient dominance, clipping, AdamW routing, auxiliary balancing), D2-AB (no-slip
objective), D2-AC/AD (local-object adapters), D2-AE (sparse relation field),
D2-AF (reliability routing), D2-AG (self-conditioned relation source), D2-AJ
(goal-token split). D2-AI, which changed **only** the budget — 299,520,000
windows against D2-X's 61,440,000, 4.875x, same recipe — is significantly better
on 9 of 18 metrics with 0 significantly worse on a 438-sequence paired bootstrap.

**Constraint on Phase 1C.** Do not open Phase 1C with a width/depth/token-count/
point-count/adapter-placement/LR/batch sweep. Reach a working from-random
HSIPrior at a preregistered budget first, then treat budget as the primary knob.
Capacity was never HOIPrior's binding constraint (D2-V/D2-X reached strong native
quality from random init at 61.44M) and there is no evidence it will be HSI's.

**What would overturn it.** An HSI-specific structural deficit that a diagnostic
localises *before* training — for example, if scene occupancy conditioning
provably cannot represent the geometry the metric scores. A hunch that "scene is
harder than objects" is not that diagnostic.

## 2. Never gate budget, early stopping or checkpoint choice on held-out denoising loss.

**Evidence.** All nine D2 configs show held-out `total` rising +5.6..+12.4% and
the `contact` term +25..+31% after a minimum at 21.5-24.6M windows. P4 tested
whether that reached metric space and falsified it in the opposite direction:
`contact_f1` at 21.504M is 0.108 *below* 61.44M, CI [-0.1340, -0.0827]. D2-AI
reproduces this far more strongly: validation `total` bottoms at 27,648,000
windows and rises **+22.7%** by 299,520,000 while native metrics improve on 9 of
18 with zero regressions. Gating that run on validation loss would have stopped
it near 27.65M and forfeited every gain.

**Why.** The validation loss is single-step teacher-forced denoising. The metric
is a 500-step reverse chain rolled across three windows on generated history.
They are different quantities and they anticorrelate here.

**Constraint on Phase 1C.** Select the formal configuration on an internal
**native rollout** at fixed processed-window budget, not on validation loss. This
is already written into the Phase 1C plan ("不得复制首次 Phase 1B 容量最大即正式
batch、teacher-forced loss 即模型选择的错误"); this entry is its measurement.

**What would overturn it.** An HSI measurement showing the two correlate — which
requires running the rollout anyway, so it costs nothing to check and never
justifies gating on loss in advance.

## 3. Put new mechanisms in the objective, not in the network.

**Evidence.** Nine loss/representation attempts plus D2-AJ point the same way:
model-side additions get absorbed as generic residuals. D2-AJ's own
preregistration predicted its null, because splitting a fused `Linear(12,512)`
goal token into three tokens is close to a first-layer reparameterization. The
one training-side intervention that *did* move a stuck metric was P8's
GT-contact-masked hand-object geometry term: it raised `contact_percent` from
0.49045 to 0.64230 against GT 0.66188 (~84% of the gap). It worked because it
acts on the loss, where there is no residual path to absorb it.

**Constraint on Phase 1C.** A proposal that adds a module, token, adapter or
conditioning path must state why it will not be absorbed the same way. Prefer an
objective term with an explicit target.

**What would overturn it.** Nothing cheap. Treat it as the strongest prior here.

## 4. Audit objective weights against metric scale before the first formal run.

**Evidence.** Ten 61.44M-window HOI configs inherited byte-identically an
`fk_weight`/`object_surface_weight` value chosen at one tenth of that budget.
D2-AH measured that choice as roughly 135x/52x under-priced relative to the
metric it was supposed to serve, and then showed the fix was unaffordable at
61.44M: the author's own recipe at 98.3% of D2-X's budget reads `xy_points_err`
5.7623 against 3.7402 (+42.3% worse on `end_obj_trans_err`). The diagnosis was
right; the remedy needed ~5x the budget.

**Constraint on Phase 1C.** Before the first formal HSI run, produce a one-page
table: each loss term, its weight, its typical magnitude on a real batch, and the
native metric it is supposed to move. Register it. This is a CPU-only check that
costs an hour and would have saved ten HOI runs.

**What would overturn it.** Nothing. This is a free check.

## 5. Pin the baseline protocol before the first comparison table.

**Evidence.** Every "released minus HOIPrior" number in this repository was for
weeks a cross-protocol quantity. The released InfBaGel row is guided 16-step
consistency sampling with CFG and scene/object-voxel conditioning; the D2-* rows
are unguided 500-step diffusion. Of the 0.1331 contact-recall gap, inference
guidance accounts for 0.0788 (59.2%), the 16-vs-1 iteration count 0.0085 (6.4%),
and the genuine model difference 0.0458 (34.4%).

**Constraint on Phase 1C.** Before the first HSI comparison, write down the
baseline's sampler, step count, guidance state, conditioning inputs and
evaluation split, and confirm the HSI row matches all of them. Record any
mismatch as a protocol difference in the table itself, not in a footnote.

## 6. Design the causal diagnostic before training, and do not rely on whole-gate ablation.

**Evidence.** Jointly trained models treat a new path as a generic residual, and
zeroing the gate is out of distribution, so a large gate-ablation effect proves
almost nothing. D2-AC showed strong adapter use under whole-gate ablation while
correspondence permutation had no significant effect — the adapter was a generic
high-leverage conditioning path, not a local relation mechanism. D2-AE's five
paired interventions (role swap, temporal permutation, source substitution) are
the form that did discriminate.

**Constraint on Phase 1C.** Any new HSI mechanism needs a preregistered
intervention that distinguishes *using the structure* from *using the extra
capacity*: permute the correspondence, swap the roles, substitute the source.
Share initial latent, posterior noise, conditions and ordering across paired
paths.

## 7. Read every penetration or smoothness gain against engagement first.

**Evidence.** D2-AH's epoch-100 row looked best-in-class on hand penetration and
foot sliding purely because `contact_percent` had collapsed to 0.3192 against GT
0.66188. The metric improves when the model stops interacting. The opposite
pattern also exists and is real: D2-AG reached hand penetration 0.18367 against
D2-X's 0.24536 at essentially identical engagement (0.47706 vs 0.47655), which is
a genuine geometry gain.

**Constraint on Phase 1C.** The HSI analogue is human-scene penetration and foot
sliding against how much the model actually contacts the scene. Report the
engagement quantity in the same table as every penetration and FS number, and
never claim a penetration win without it.

## 8. Teacher-forced movement is not rollout evidence.

**Evidence.** Phase 1B's first failure mode was exactly this shape: validation
kept falling while three-window autoregressive generation degraded across the
board (object/pelvis goal error 32.963/46.163 cm, contact F1 0.2561). D2-Y later
moved its teacher-forced surrogate while official foot sliding did not improve.

**Constraint on Phase 1C.** Every internal gate that claims a mechanism works
must be measured on generated history, not on a teacher-forced surrogate.

---

## Process lesson

Phase 1B produced one script, one test and one config per experiment: 66 one-off
tools, 35 one-off tests and 25 retired configs for about thirty experiments, plus
18 near-duplicate evaluation runners of ~1,400 lines each that had drifted apart
by hundreds of lines. That drift is an evaluation-consistency hazard, not just
clutter. See `docs/EXPERIMENT_CONVENTIONS.md` for the convention that replaces
it. HSIPrior starts with that convention in force.
