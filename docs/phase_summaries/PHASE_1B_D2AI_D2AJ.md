# Phase 1B D2-AI / D2-AJ: long-budget arms

Status: both arms terminated 2026-08-04. Classification `budget-positive-goal-pathway-null`.
Compact result: `experiments/results/p1_hoi_d2ai_d2aj_long_budget_arms_s42_20260804.json`.
Preregistration: `docs/EXPERIMENT_PLAN.md`, "2026-08-03 Phase 1B D2-AI 全预算与 D2-AJ 目标条件通路",
commit `22882b9`. Implementation commit `21eb5f5`.

## What was tested

Two concurrent four-GPU arms at 299,520,000 processed windows (146,250 updates, 4.875x the
formal budget), user-approved after P4 found nothing saturated at 61.44M.

| arm | run id | manipulated factor | outcome |
| --- | --- | --- | --- |
| D2-AI | `p1-hoi-d2ai-full-budget-s42-20260803` | `max_processed_windows` only | completed, broad significant gain |
| D2-AJ | `p1-hoi-d2aj-split-goal-tokens-s42-20260803` | split pelvis/object/progress goal tokens | stopped early at 50.3%, null |

## Arm 1 result: budget is the first broadly effective model-side lever

Paired sequence-level bootstrap, 438 sequences, both unguided, same tree, seed 42,
10,000 replicates. **9 significantly better, 0 significantly worse.**

| metric | D2-AI | D2-X | paired diff | 95% CI |
| --- | ---: | ---: | ---: | --- |
| `obj_trans_dist` | 14.81981 | 15.99405 | −1.17424 | [−1.59101, −0.76639] |
| `human_pen_loss_infbagel` | 2.76049 | 3.86908 | −1.10860 | [−1.86033, −0.43539] |
| `trans_dist` | 7.70140 | 8.17009 | −0.46869 | [−0.65694, −0.28444] |
| `mpjpe` | 11.74146 | 12.05085 | −0.30939 | [−0.51845, −0.09656] |
| `hand_pen_loss_omomo` | 0.17481 | 0.24536 | −0.07055 | [−0.11800, −0.02705] |
| `obj_rot_dist` | 0.99656 | 1.03094 | −0.03438 | [−0.06589, −0.00267] |
| `contact_precision` | 0.81474 | 0.78806 | +0.02668 | [+0.00980, +0.04528] |
| `hand_pen_ratio` | 0.11839 | 0.14387 | −0.02548 | [−0.04532, −0.00560] |
| `human_pen_ratio` | 0.12216 | 0.14619 | −0.02403 | [−0.04414, −0.00422] |

Null: `contact_f1` +0.01623, `contact_recall` +0.02035, `end_obj_trans_err` +0.07616,
`foot_sliding` −0.00866.

Against released, metrics beating it rise **4/17 → 8/17**, and the two largest real gaps
collapse: `hand_pen` +41.3% → **+2.7%**, `human_pen` +40.0% → **+1.8%**. This is training-side,
not a guidance artefact — unguided D2-AI already reads 0.17481 and 2.76049.

Training: 526.87 epochs, 21.74 h, `loss_finite` true, 0 AMP skips, terminal checkpoint sha256
`a190e56c249161c0b52f0aebb097d0d5b95cb0c3810abb664000fc3c2fdda224`, 29,673,448 params
(identical to sealed D2-X).

## The cost, stated plainly

Contact regressed against released: `contact_f1` deficit +0.1% → **+7.1%**, `contact_recall`
+2.7% → **+12.3%**, `contact_percent` deviation +45.2% → **+140.5%**.

The mechanism: D2-X + Arm B bought contact parity largely through guidance raising engagement
`0.47655 → 0.56956`. **D2-AI responds to the same guidance far less** (`0.49045 → 0.50899`), so
that parity does not survive a stronger base model. This is a second instance of
evidence-index conclusion 9 (contact levers do not stack).

## P4's direction was right, its magnitude was not

P4 extrapolated `contact_f1` to ~0.856 from the last four cadence points' log-log slope. The
measurement is **0.65366**. The "not saturated" conclusion holds; the extrapolated magnitudes
do not. Power-law fits on a short tail were over-confident, and I had already flagged the
`end_obj` −62% figure as untrustworthy for the same reason — the whole extrapolation deserved
that caveat, not just one row.

## Validation loss anticorrelation, now over 4.875x

Held-out `total` bottoms at **27,648,000** windows and rises **+22.7%** by 299,520,000 (versus
+8.4% at 61.44M), while native metrics improve on 9 of 18 with zero regressions over the same
interval. **Gating this run on validation loss would have stopped it near 27.65M and forfeited
every gain.** 98 validation records retained in `metrics.json`.

## Arm 2 result: the tenth failed model-side intervention

At the preregistered 61,440,000 go/no-go:

| criterion | D2-AJ | D2-X | paired diff | 95% CI | verdict |
| --- | ---: | ---: | ---: | --- | --- |
| `contact_f1` | 0.63753 | 0.63743 | +0.00010 | [−0.02222, +0.02209] | null |
| `end_obj_trans_err` | 3.71761 | 3.76611 | −0.04849 | [−0.29809, +0.19882] | null |

Informational only, cannot satisfy the rule: `pelvis_goal_error_cm` +0.19289
[+0.04281, +0.34125] — significant in the **unfavourable** direction.

Decision `stop_early`, classification `d2aj-goal-pathway-null-at-matched-budget-stop`,
terminated at 150,528,000 windows (50.3%), ~8 GPU hours saved, 49 cadence checkpoints retained,
manifest sealed `aborted` with `operational_failure: false`.

**The arm confirmed the prior its own preregistration stated**: the fused `Linear(12,512)` is
already an affine map of the same information, so splitting it into three tokens is close to a
first-layer reparameterization. Nine loss/representation attempts plus this one now point the
same way — model-side additions get absorbed as generic residuals.

## Preregistration defect, recorded not repaired

`xy_points_err` was named as a third go/no-go criterion but **does not exist per sequence** (the
analogue is `pelvis_goal_error_cm`; the mapping was already recorded at
`tools/run_hoi_d2ac_native_evaluation.py:63` and was not re-checked when the preregistration was
written). It is recorded as unevaluable and **not counted as favourable**; counting
`pelvis_goal_error_cm` after the fact would have relaxed a preregistered criterion. Because the
rule is "at least one favourable", dropping it makes the rule **stricter**, so the conclusion is
unaffected.

## Tree control

The D2-AJ variant does not exist at `5f7dde7`, so both evaluations ran on the main repo at HEAD.
A D2-X tree control (`p1-hoi-d2aj-gonogo-treecontrol-d2x-s42-20260804`) reproduced the sealed
per-sequence sha256 `69cc811c256345ba64c84e89c4b19ca1b4ff64113e6585ec89d88fdbe0438b4a`
**bit-exactly** first; the script was written to exit without producing any D2-AJ number if it
had not.

## Concurrency cost is ~zero

| | windows/s |
| --- | ---: |
| Arm 1 concurrent with Arm 2 (48 cadences) | 3816.8 |
| Arm 1 solo after Arm 2 stopped (26 cadences) | 3843.4 |
| solo speedup | **1.007x** |

Both arms ran above sealed D2-X's 3243.04 throughout and far above the 2757 contention floor, so
**no contention was recorded** per the preregistered rule. Layout: Arm 1 → GPU4-7 (NUMA1),
Arm 2 → GPU0-3 (NUMA0), `taskset`-pinned, dataset `mmap`-shared through one page cache. The
before/after comparison is direct evidence, not a pre-run benchmark.

8-GPU merging was rejected in preregistration and remains rejected:
`code/priors/losses.py:177` self-normalizes `object_goal` per micro-batch, and halving the
micro-batch would reprice that term ~13.6% — the term supervising both goal-recall metrics.

## Operational notes

- Both training logs end with DataLoader worker shutdown tracebacks
  (`RuntimeError: DataLoader worker ... killed by signal: Aborted`). These occur **after**
  training completes and the terminal checkpoint is written. For Arm 1: `status: completed`,
  recorded and recomputed checkpoint sha256 agree, all weights finite.
- `training_state.json` is written **only at the end of a run**, not per cadence. Progress must
  be monitored from checkpoint files.
- DDP ranks are `multiprocessing.spawn` children whose command line is `spawn_main`, so
  `pgrep -f train_hoi_prior.py` matches only the two parents (and the monitoring shell itself).
  Count GPU compute processes instead.
- A single `nvidia-smi` utilization snapshot can read 0% mid-all-reduce and does not indicate a
  hang.

## Next entry point: contact engagement

It is now the single mechanism explaining all three remaining contact-side deficits
(`contact_percent` +140.5%, `contact_recall` +12.3%, `contact_f1` +7.1%). **D2-AI made it more
visible, not less**: the model is under-engaged at every budget (0.49045 unguided vs GT 0.66188)
and has become *less* responsive to inference guidance. Any next proposal must target engagement
directly and state why it will not be absorbed the way the ten previous model-side interventions
were.

`end_obj_trans_err` is not a candidate: released reaches 3.03724 via 16-step consistency
sampling, and the author's own diffusion recipe at matched budget does not close it either.

## What this does not show

- Single seed, single lineage.
- Budget is confounded with revisit count (526.87 epochs); this does not isolate *why* longer
  budget helps.
- The released row remains protocol-incomparable (16-step CM, guidance, CFG, scene/object-voxel
  conditioning, 50,014,184 params vs 29,673,448). Gap percentages against it are point estimates
  with no CI, since it has no per-sequence output.
