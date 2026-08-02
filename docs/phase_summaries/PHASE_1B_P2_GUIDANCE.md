# Phase 1B P2 inference contact guidance: protocol alignment, not a model improvement

## Scope and final outcome

This sub-phase is an inference-only measurement. It trained nothing, wrote no
checkpoint, selected no checkpoint, changed no model, training or evaluator code,
and did not touch the fixed native D2 protocol. It changes **how HOIPrior is
sampled, not the quality of the prior.**

Its question follows directly from the 2026-08-01 protocol decomposition: every
recorded HOIPrior-vs-released contact comparison put an unguided HOIPrior against
a *guided* baseline, and guidance alone accounted for 59.2% of the recall gap and
65.5% of the F1 gap. P2 removes that asymmetry by applying the author's complete
`apply_hoi_guidance_loss` (hand-object x10 **plus** feet-floor x500) to the sealed
D2-X checkpoint at inference, on the unchanged official 438 sequences.

Classification:
`inference-protocol-alignment-contact-parity-reached-cost-arm-dependent`.

Plain conclusions, none of them softened:

1. **Guided HOIPrior reaches statistical contact parity with the guided released
   model.** Arm A minus `a0-old`: contact recall `−0.0043`, precision `+0.0113`,
   F1 `+0.0035`, `contact_percent` `−0.0120` — **all four CIs cross zero**. Arm B's
   F1 difference `−0.0007` also crosses zero. Arm A and Arm B are mutually
   indistinguishable on recall, precision and F1.
2. **Guidance moved the geometry, not just the operating point.** The paired
   GT-contact-frame distance shifts **−1.4371 cm [−1.7204, −1.1532]**, the median
   goes **3.1835 → 1.7178 cm**, and the `[0,2)` cm mass rises `0.359 → 0.547`. No
   rethresholding can produce a shift of the whole distribution. On the
   **continuous, unthresholded residual** Arm A is indistinguishable from the
   released guided model: **+0.2216 cm [−0.0882, +0.5310]**, CI crosses zero.
3. **Arm A is the preregistered primary and has the highest contact F1
   (`0.73071`), but Arm B is the configuration that survives cost scrutiny.**
   Arm B keeps 88% of Arm A's recall gain, 96% of its F1 gain, 85% of its
   predicted-contact gain and 85% of its distance gain — the plan's "~90% of the
   contact gain and ~85% of the distance gain" — at essentially zero
   object-trajectory cost (ratio `1.0063`) and with foot sliding **improved**
   (ratio `0.9797`).
4. **If D2-Q0's `<= 1.10` bar were applied**, Arm A would **fail** foot sliding
   (CI upper `1.1661`) and clear object translation by `1.7e-04`; Arm B would clear
   both. **P2 preregistered mandatory cost reporting but no cost threshold.** The
   `<= 1.10` line was preregistered for D2-Q0, is not P2's gate, and the above is
   recorded for comparability with the earlier sub-phases, **not as a P2 verdict.**
5. **D2-Q0's negative is overturned.** It stopped on a foot-sliding ratio of
   `1.5350`, but its implementation references `apply_feet_floor_contact_guidance`
   **zero times** — it omitted the feet-floor term entirely — and its checkpoint had
   a **95.3 cm** object-goal error against D2-X's `3.7402 cm`. With the author's
   complete loss the ratio is `0.9797` (Arm B) and `1.0992` (Arm A).
6. **What guidance does not fix.** Arm A's `>= 8 cm` tail stays **fatter** than the
   released model's, `0.1272` vs `0.1052`, and object-trajectory and penetration
   costs remain against `a0-old` (`obj_trans_dist +1.5122`, `end_obj +1.1783`,
   `hand_pen +0.0802`, `human_pen +1.2612` cm/point estimates). This is consistent
   with the **0.83 cm [0.49, 1.18]** genuine generative-geometry gap measured on
   2026-08-01: the contact failure was a measurement artifact, the remaining gaps
   are real.

Nothing here enables a lever. Guidance stays default-off in production. Production
guidance, stepwise object requery and any use of these runs as a
checkpoint-selection signal still require a separate dated plan amendment and
explicit user approval. Phase 1B HOIPrior search stays closed.

## Design and executed configuration

Sealed D2-X checkpoint
(`p1-hoi-d2x-fk-foot-temporal-routing-r1-s42-20260723` final-online, SHA-256
`b0fa6bdddc280b2f561344d26046fff7c89eae50842073a52e49d5c39e2a3d51`), official 438
sequences x 3 windows, 500-step diffusion, `load_scene=false`, no CFG, seed 42.
The arms differ **only** in inference guidance. D2-AG was deliberately not used:
its self-conditioned relation source consumes `current`
(`code/priors/diffusion.py:234-236`) and guidance modifies `current`, which would
be an uncontrolled distribution shift.

| run | arm | guidance_scale | last_steps | clamp | clamp_target | guided steps | role |
|---|---|---:|---:|---:|---|---:|---|
| `p1-hoi-p2-guidance-arma-s42-20260801` | a | 1.0 | 10 | 1.0 | update | 1497 | preregistered primary; InfBaGel-faithful port |
| `p1-hoi-p2-guidance-armb-s42-20260801` | b | 1000.0 | 10 | 1.0 | update | 27 | declared second arm; CHOIS-style DDPM analogue |
| `p1-hoi-d2x-distance-probe-s42-20260801` | — | — | — | — | — | 0 | unguided control |

`last_steps` and `clamp` are read for both arms but only take effect for arm b
(`code/priors/inference_guidance.py:133-140,371-381`); arm a guides **every**
reverse step except the last, hence 1497 = 3 windows x 499 steps, against arm b's
27 = 3 x 9. Arm B's `guidance_scale` is `1000.0` because that is the cited CHOIS
`classifier_scale = 1e3`; at `1.0` the `posterior_variance[t]` scaling would shrink
the update by about `3.8e-4` and Arm B would have been an accidental null arm. Both
values were fixed in the plan before any GPU run.

Executed commit `c40dc00b2ad315f194a01d034413d80c493cf220` for all five runs;
preregistration `9a3a351`; the dated plan section landed in `2b7fa9e`. The official
438 runs are in the pinned worktree `/data/yujinlun/InfBaGel-p2`, the functional
smokes in the main checkout, the unguided control in
`/data/yujinlun/InfBaGel-head-baseline`, the released reference in
`/data/yujinlun/InfBaGel-c358fa4-baseline`. `code/eval_metrics.py`
(`445e681f...`) and `code/guidance_loss.py` (`5747721b...`) are unchanged from
`c358fa4`, so neither the metric definitions nor the guidance formula is a
confounded axis. `code/priors/contact_guidance.py` — the sealed D2-Q0 partial
implementation — was left untouched on purpose so that the historical diagnostic
is not retroactively rewritten.

Per the authorized lean lifecycle this measurement allocated no run id and produced
no `tools/experiment.py start` manifest. Traceability rests on the executed commit,
the archived Hydra config and overrides, the fixed official test set and
byte-identical metric code.

**No test-time GT leakage.** Every guidance input comes from a model prediction or
a given asset: `contact_labels = x_start[:,:,228:232]` with
`x_start = pred_x_0.detach().requires_grad_(True)`
(`code/models/infbagel.py:721,677`); the floor height is the hard constant `0.02`
(`code/guidance_loss.py:82`); GT contact annotation enters only as the two window-0
seed history frames, identically in the unguided path. The author's own
implementation shadows and never reads its GT `contact_labels` argument
(`chois_release/manip/model/trainer_chois.py:2024`).

## The pre-run prediction was falsified, and that is recorded as such

Before any GPU run the plan predicted that Arm A would raise
`position_outside_rate` and possibly produce degenerate motion, because with the
real `norm.npy` one guidance step adds about `2.06` to the `[-1,1]` normalized
pelvis-y channel at 5 cm foot error and about `19` at 30 cm, while HOIPrior applies
the raw gradient on 499 steps where the author applies it on 15.

**It did not happen.** All three functional smokes and both full runs report
`nonfinite_values = 0` and `position_outside_rate = 0.0`; `object_outside_rate` is
also `0.0`, and `guidance_nonfinite_steps` is `0` in both arms. The most likely
reason is that the loss falls towards zero once contact is achieved, so the
guidance is self-limiting: Arm A's mean gradient RMS is `0.1525` while its maximum
absolute value is `437.9` — large updates are rare and transient, not sustained.
The inherited NaN risk (the temporal term computes `v/||v||` with no epsilon,
`code/guidance_loss.py:58-64`) also never fired.

The functional smokes (4 sequences, `hoi_sequence_limit=4`) checked **runnability
and finiteness only**, as preregistered; they were not used to choose between arms,
and their 4-sequence contact numbers are not comparable to the official 438.

## Results: contact parity with the guided released model

Aggregates on the official 438:

| metric | unguided control (D2-X) | Arm A | Arm B | released, guided (`a0-old`) |
|---|---:|---:|---:|---:|
| contact F1 | 0.63743 | **0.73071** | 0.72653 | 0.72726 |
| contact recall | 0.59445 | 0.72332 | 0.70783 | 0.72759 |
| contact precision | 0.78806 | 0.80216 | 0.81223 | 0.79081 |
| contact_percent | 0.47655 | 0.58632 | 0.56956 | 0.59832 |
| foot sliding | 0.36301 | 0.39903 | 0.35565 | 0.33336 |
| obj_trans_dist | 15.99405 | 17.23781 | 16.09432 | 15.72565 |

Paired sequence-level bootstrap, seed 42, 10,000 replicates, n = 438
(`*` = CI crosses zero):

| contrast | recall | precision | F1 | contact_percent | foot sliding |
|---|---|---|---|---|---|
| Arm A − control | +0.1289 [+0.1046, +0.1537] | +0.0141 [−0.0032, +0.0321]* | +0.0933 [+0.0730, +0.1146] | +0.1098 [+0.0903, +0.1295] | +0.0360 [+0.0142, +0.0577] |
| Arm B − control | +0.1134 [+0.0981, +0.1286] | +0.0242 [+0.0116, +0.0381] | +0.0891 [+0.0759, +0.1022] | +0.0930 [+0.0808, +0.1049] | −0.0074 [−0.0242, +0.0096]* |
| **Arm A − `a0-old`** | **−0.0043 [−0.0267, +0.0181]\*** | **+0.0113 [−0.0050, +0.0269]\*** | **+0.0035 [−0.0148, +0.0218]\*** | **−0.0120 [−0.0305, +0.0068]\*** | +0.0657 [+0.0478, +0.0837] |
| Arm B − `a0-old` | −0.0198 [−0.0419, +0.0018]* | +0.0214 [+0.0066, +0.0363] | −0.0007 [−0.0188, +0.0168]* | −0.0288 [−0.0468, −0.0109] | +0.0223 [+0.0036, +0.0407] |
| Arm A − Arm B | +0.0155 [−0.0035, +0.0348]* | −0.0101 [−0.0248, +0.0040]* | +0.0042 [−0.0113, +0.0200]* | +0.0168 [+0.0013, +0.0324] | +0.0434 [+0.0255, +0.0613] |

State it plainly: **protocol-aligned, the guided HOIPrior and the guided released
model are statistically indistinguishable on all four contact quantities.** The
recorded contact deficit of every earlier D2-* row was, on this axis, a protocol
artifact.

## Results: the geometry moved, not only the operating point

Per GT-contact frame, minimum over the two SMPL hand joints (22, 23) of the
distance to the nearest vertex of the posed *predicted* object mesh, restricted to
the `code/eval_metrics.py:271` mask — the same joint-to-vertex definition and the
same 397-sequence / 36,528-frame pool as the 2026-08-01 decomposition, whose
GT-vs-GT floor is `1.6981 cm` frame-pooled.

| run | mean cm (frame-pooled) | median cm | `[0,2)` | `[2,4)` | `[4,5)` | `[5,6)` | `[6,8)` | `>= 8` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| control D2-X | 5.3886 | 3.1835 | 0.3586 | 0.2190 | 0.0797 | 0.0580 | 0.0861 | 0.1986 |
| Arm A | 3.9759 | 1.7178 | 0.5473 | 0.2013 | 0.0470 | 0.0342 | 0.0430 | **0.1272** |
| Arm B | 4.1725 | 2.1396 | 0.4744 | 0.2421 | 0.0647 | 0.0445 | 0.0527 | 0.1216 |
| released, guided | 3.6836 | 1.8012 | 0.5333 | 0.2135 | 0.0557 | 0.0393 | 0.0529 | **0.1052** |

| paired contrast | Δ mean cm | 95% CI | crosses zero |
|---|---:|---|---|
| Arm A − control | **−1.4371** | [−1.7204, −1.1532] | no |
| Arm B − control | −1.2144 | [−1.3700, −1.0619] | no |
| Arm A − Arm B | −0.2227 | [−0.4343, −0.0084] | no |
| **Arm A − `a0-old`** | **+0.2216** | **[−0.0882, +0.5310]** | **yes** |
| Arm B − `a0-old` | +0.4443 | [+0.1238, +0.7647] | no |

The distribution itself moved: the median more than halves and the `[0,2)` cm mass
rises by 19 points. Rethresholding cannot do that. And on the continuous residual,
which has no threshold in it at all, Arm A is statistically indistinguishable from
the guided released model — so the parity in the previous section is not an
artifact of where the 5 cm line falls.

## Results: costs, and the criteria question

Paired differences against the unguided control (`*` = CI crosses zero;
penetration on 181 sequences):

| cost | Arm A − control | Arm B − control |
|---|---|---|
| MPJPE cm | +0.2958 [+0.1287, +0.4727] | +0.0285 [−0.0210, +0.0795]* |
| obj_trans_dist | +1.2438 [+0.9168, +1.5868] | +0.1003 [+0.0003, +0.2036] |
| end_obj_trans_err cm | +0.4720 [+0.3014, +0.6450] | +0.0992 [+0.0331, +0.1658] |
| hand penetration | −0.0027 [−0.0407, +0.0319]* | −0.0159 [−0.0349, +0.0006]* |
| human penetration | −0.0186 [−0.6159, +0.5228]* | −0.2430 [−0.5404, +0.0145]* |

Both arms *improve* both penetration point estimates relative to the control, and
neither improvement is significant.

D2-Q0-form mean ratios against the unguided control
(`tools/run_hoi_d2n.py:407`):

| ratio | Arm A | Arm B |
|---|---|---|
| obj_trans_dist | 1.0778 [1.0571, **1.099831**] | 1.0063 [1.0000, 1.0128] |
| foot_sliding | 1.0992 [1.0374, **1.1661**] | **0.9797** [0.9354, 1.0276] |

**The criteria statement, so that no criterion is invented after the fact.** P2
preregistered mandatory cost *reporting* — foot sliding, object translation MAE,
end-object, penetration, MPJPE, `position_outside_rate` and the GT-contact-frame
distance distribution — and preregistered **no cost threshold**. The `<= 1.10`
protection bar belongs to D2-Q0 and **is not P2's gate**. No pass or fail against
that line is declared here. Reported for comparability only: **if it were applied,
Arm A fails foot sliding at CI upper `1.1661` and clears object translation by
`1.685e-04`; Arm B clears both.**

Therefore: **Arm A is the preregistered primary and carries the highest contact F1,
but Arm B is the configuration that survives cost scrutiny.** Arm B retains 88% of
Arm A's recall gain, 96% of its F1 gain, 85% of its predicted-contact gain and 85%
of its distance gain, while its object-trajectory cost is near zero and its foot
sliding is better than the unguided control's. Both arms were declared before any
run and both are reported; the primary-arm declaration is **not** rewritten because
Arm B looks better on cost.

## D2-Q0's negative is overturned

D2-Q0 stopped on a foot-sliding ratio of `1.5350`. Two facts remove that result's
authority over the author's guidance:

1. `code/priors/contact_guidance.py` (D2-Q0) and `code/priors/routed_guidance.py`
   (D2-R0) reference `apply_feet_floor_contact_guidance` **zero times**. Both
   implemented only the hand-object x10 term and **omitted the feet-floor x500
   term entirely** — the term whose whole purpose is protecting the feet.
2. Its gate checkpoint had a **95.3 cm** object-goal error, against D2-X's
   `3.7402 cm`. The measurement was taken on a substantially weaker model.

With the author's complete loss on D2-X the foot-sliding ratio is `0.9797` for
Arm B (an improvement) and `1.0992` for Arm A. The historical D2-Q0/D2-R0
foot-sliding cost is a property of those two partial implementations, not of the
author's guidance.

## What guidance does not fix

Guidance closes the contact gap; it does not make HOIPrior the released model.

- **The far tail is still fatter.** Arm A leaves `0.1272` of GT-contact frames at
  `>= 8 cm` against the released model's `0.1052`. Guidance shifts the bulk, not
  the gross-miss population.
- **Object trajectory and penetration remain worse than `a0-old`.** Arm A point
  estimates: `obj_trans_dist +1.5122`, `end_obj_trans_err +1.1783 cm`,
  `mpjpe +0.3491 cm`, `hand_pen +0.0802`, `human_pen +1.2612`. Arm B:
  `+0.3687`, `+0.8028 cm`, `+0.0818 cm`, `+0.0670`, `+1.0368`. These are point
  estimates only: `a0-old` has no `per_sequence_metrics.json`, so no CI can be
  formed for these contrasts.
- This is exactly what the 2026-08-01 decomposition predicted. Protocol-matched
  and both unguided, D2-X sat **0.83 cm [0.49, 1.18]** further from the object.
  **The contact failure was a measurement artifact; the remaining gaps are real.**

## Measurement caveats and declared confounds

- **`end_obj_trans_err` per-sequence and aggregate are different quantities.** The
  aggregate uses the pre-interpolation window endpoint; the per-sequence value uses
  the final frame of the interpolated trajectory. Relative difference `0.533%`
  (Arm A), `0.658%` (Arm B), `0.692%` (control) — about `0.7%`. Every
  `end_obj_trans_err` CI in this record is therefore a CI on the per-sequence
  construction, not on the published aggregate.
- **`mpjpe`** differs between pooling orders only at `6.2e-08` relative;
  `obj_trans_dist` and both penetration terms agree exactly.
- **Penetration CIs rest on 181 of 438 sequences**, because the frozen metric code
  does not compute those terms for six object classes.
- **Confound 1: guidance consumes geometry the model never sees.** The loss uses
  the full posed object mesh (13,086-38,353 vertices) while HOIPrior conditions
  only on a 1024-point BPS. The evaluator already uses that mesh, but it is extra
  test-time geometry the prior itself never receives, so part of the gain is bought
  with information outside the model.
- **Confound 2: the foot-sliding improvement may be partly metric-shaped.** The FS
  metric estimates the floor from the **predicted** joints
  (`code/test_infbagel_hoi.py:249`, `code/eval_metrics.py:101-113`) while the
  guidance pins the support foot to a fixed `0.02 m`. The two can be mutually
  self-consistent, so Arm B's `0.9797` must not be read as a pure
  physical-plausibility gain.
- **Confound 3: determinism convention.** A deterministic vertex subset (the D2-Q0
  convention) is used instead of the author's
  `torch.randperm(...)[:10000]` (`code/models/infbagel.py:736`), so a configuration
  reproduces across runs. This differs from the author's implementation.
- The `397/438 = 0.9064` contact-recall scoring artifact recorded on 2026-08-01 is
  unchanged and untouched. It depresses every recall in this project equally and
  affects no P2 contrast, all of which are paired on the same 438 sequences.

## Verification

- Five `evaluation/aggregate_metrics.json` files plus the two arms' and the
  control's `per_sequence_metrics.json` read directly. No metric in this summary
  was retyped from prose.
- Independent per-sequence extraction gate before any contrast: for Arm A, Arm B
  and the control, all six pooled quantities (`contact_f1`, `contact_recall`,
  `contact_precision`, `contact_percent`, `gt_contact_percent`, `foot_sliding`)
  reproduce the aggregates exactly, the pooled recall identity holds, and there are
  **0** per-sequence mismatches against `per_sequence_metrics.json`.
- The unguided control reproduces the sealed worker evaluation
  `p1-hoi-d2x-native-eval-r1-s42-20260723` bit-exactly, 18/18 metrics.
- Finiteness gate: `nonfinite_values = 0` and `position_outside_rate = 0.0` in all
  three smokes and both full runs.
- Uncertainty: paired sequence-level bootstrap, seed 42, 10,000 replicates, using
  the `tools/summarize_hoi_phase1b.py:112` convention; ratios use the
  `tools/run_hoi_d2n.py:407` expression.
- `python -m unittest discover -s tests -q` — 598 tests OK.
- `python tools/experiment.py validate` — 241 registry records.

## Artifacts

- Compact result:
  `experiments/results/p1_hoi_p2_inference_contact_guidance_s42_20260801.json`.
  It carries both arms' full guidance configuration and telemetry, the four runs'
  complete aggregates, every paired CI, the D2-Q0-form ratios, the distance
  statistics and histograms, the smoke results, the falsified pre-run prediction
  and the criteria statement.
- Registry completion row:
  `p1-hoi-p2-inference-contact-guidance-completion-s42-20260801`; the
  preregistration row is
  `p1-hoi-p2-inference-contact-guidance-preregister-s42-20260801`.
- Plan section: `docs/EXPERIMENT_PLAN.md`,
  "2026-08-01 Phase 1B 推理期接触引导 P2（协议对齐，用户批准）".
- Implementation: `code/priors/inference_guidance.py`, the guidance hook in
  `code/priors/diffusion.py` and the keys in `code/config/sampler/hoi_prior.yaml`,
  committed as `c40dc00`. Guidance defaults to `None`.
- Run outputs (untracked, on this host):
  `/data/yujinlun/InfBaGel-p2/results/experiments/p1-hoi-p2-guidance-{arma,armb}-s42-20260801`,
  `/data/yujinlun/InfBaGel-release/results/experiments/p2-smoke-{arma,armb,ctrl}-s42-20260801`,
  `/data/yujinlun/InfBaGel-head-baseline/results/experiments/p1-hoi-d2x-distance-probe-s42-20260801`.
- No tag was created and no merge is authorized by this sub-phase.

## Unresolved risks

- The pinned worktree `/data/yujinlun/InfBaGel-p2` holds the only copies of both
  official 438-sequence arm outputs. It is an untracked working directory, not a
  recovered staging tree with a checksum pass; the tracked record is this summary
  plus the compact result and their hashes.
- Neither arm is a validated production configuration. The measurement authorizes
  no default change, and Arm B's advantage on cost was **not** a preregistered
  selection criterion — reading it as one would be exactly the post-hoc criterion
  this record refuses to create.
- The foot-sliding confound is unresolved. Distinguishing a real physical gain from
  a self-consistent metric would need a floor estimate that does not come from the
  predicted joints; no such measurement exists here.
- The distance metric is joint-to-vertex, not surface-to-surface. It is internally
  consistent across all runs and against the GT floor, but its absolute scale is not
  a skin-contact distance.
- `a0-old` has no `per_sequence_metrics.json`, so every Arm-vs-released **cost**
  contrast is a point estimate without a CI. Only the contact and distance
  contrasts against `a0-old` are CI-backed.

## Exact next entry point

Unchanged. Phase 1B HOIPrior search remains closed, no checkpoint is selectable,
guidance remains default-off, and the only next-session entry is a dated Phase 1C
HSIPrior plan-only preregistration on `phase/01c-hsi`, trained from random
initialization and never loading the released, author, D2-X, D2-AC, D2-AD, D2-AE,
D2-AF or D2-AG checkpoints.

Any future comparison of a HOIPrior contact number against a guided baseline must
cite this summary: unguided-vs-guided is not a model comparison, and the
protocol-aligned answer on contact is parity.
