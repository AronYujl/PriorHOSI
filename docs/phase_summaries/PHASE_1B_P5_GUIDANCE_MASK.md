# Phase 1B P5: the guidance contact mask is not the bottleneck

Status: closed 2026-08-04. Classification
`guidance-contact-mask-dose-response-null-prediction-falsified-direction-dropped`.
Compact result: `experiments/results/p1_hoi_p5_guidance_mask_dose_response_s42_20260804.json`.
Preregistration: `docs/EXPERIMENT_PLAN.md`, "2026-08-04 Phase 1B P5 推理期接触 mask
剂量-响应与 GT 上界探针", registry row
`p1-hoi-p5-guidance-mask-dose-response-preregister-s42-20260804`, commit `3c8869e`.

**The preregistered prediction was falsified. The direction is dropped.**

## What was tested

Inference only. No training, no checkpoint written, no checkpoint selected, no change to
`code/eval_metrics.py` or to the official 438-sequence protocol.

The preregistration localized guidance's weak effect on D2-AI to one line,
`code/guidance_loss.py:31`: `contact_labels = pred_contact_semantic > 0.95`. Guidance pulls a
palm toward the object only on frames the model itself already calls contact at >0.95
confidence, so it amplifies committed engagement rather than creating it. P5 swept that
threshold with the mask as the **only** manipulated factor, on the sealed D2-AI final
checkpoint `a190e56c249161c0b52f0aebb097d0d5b95cb0c3810abb664000fc3c2fdda224` under sealed
Arm B (`guidance_scale=1000`, `last_steps=10`, `clamp=1.0`, `clamp_target=update`,
`load_scene=false`, seed 42, 500-step ancestral DDPM).

| cell | threshold | run id |
| --- | ---: | --- |
| A0 | 0.95 | `p1-hoi-d2ai-guidance-armb-s42-20260804` (sealed, **reused, not re-run**) |
| A1 | 0.90 | `p1-hoi-p5-mask-a1-s42-20260804` |
| A2 | 0.75 | `p1-hoi-p5-mask-a2-s42-20260804` |
| A3 | 0.50 | `p1-hoi-p5-mask-a3-s42-20260804` |
| A4 | 0.25 | `p1-hoi-p5-mask-a4-s42-20260804` |
| U | GT mask | **NOT RUN** |

## Result: flat

| metric | A0 (0.95) | A1 (0.90) | A2 (0.75) | A3 (0.50) | A4 (0.25) |
| --- | ---: | ---: | ---: | ---: | ---: |
| `contact_percent` | 0.508987 | 0.509024 | 0.509024 | 0.509024 | 0.509024 |
| `contact_recall` | 0.638188 | 0.638212 | 0.638212 | 0.638212 | 0.638212 |
| `contact_precision` | 0.815790 | 0.815753 | 0.815753 | 0.815753 | 0.815753 |
| `contact_f1` | 0.675702 | 0.675709 | 0.675709 | 0.675709 | 0.675709 |
| `contact_acc` | 0.762448 | 0.762448 | 0.762448 | 0.762448 | 0.762448 |
| `hand_pen_loss_omomo` | 0.166761 | 0.166761 | 0.166761 | 0.166761 | 0.166761 |
| `hand_pen_ratio` | 0.116680 | 0.116680 | 0.116680 | 0.116680 | 0.116680 |
| `human_pen_loss_infbagel` | 2.635192 | 2.635188 | 2.635187 | 2.635186 | 2.635187 |
| `human_pen_ratio` | 0.120977 | 0.120977 | 0.120977 | 0.120977 | 0.120977 |
| `mpjpe` | 11.769875 | 11.769947 | 11.770029 | 11.770003 | 11.769950 |
| `end_obj_trans_err` | 3.896488 | 3.896294 | 3.896271 | 3.896293 | 3.896360 |
| `foot_sliding` | 0.341770 | 0.341788 | 0.341767 | 0.341764 | 0.341767 |
| `obj_trans_dist` | 14.817251 | 14.817183 | 14.817205 | 14.817268 | 14.817284 |

GT `contact_percent` is **0.661883** throughout; the engagement deficit at A4 is **0.152859**,
exactly where D2-AI left it.

Across a 3.8x reduction of the threshold: `contact_recall` **+0.0000240**, `contact_f1`
**+0.0000073**, `contact_percent` **+0.0000362**, `hand_pen_loss_omomo` **−0.00000034**.

The preregistration says verbatim: *"若降低阈值对 `contact_recall` 无效果，则'mask 门限制了引导
作用面'这一推断被证伪"* — if lowering the threshold does not move `contact_recall`, the
inference that the mask gate limits guidance's action surface is **FALSIFIED**. It is
falsified. Recorded as such, not softened.

It is worse than weak — but not in the way an earlier draft of this summary claimed. Two figures
in that draft were wrong and are corrected here.

**The aggregates are not bit-identical.** A1-A4 agree with each other to full float precision on
the contact aggregates, but all four differ from A0 at the ~1e-5 level: `contact_recall`
**+0.0000240**, `contact_f1` **+0.0000073**, `contact_percent` **+0.0000362**,
`contact_precision` **−0.0000372**, `contact_acc` **0.0000000**. The non-contact aggregates
(`mpjpe`, `trans_dist`, `obj_trans_dist`, `obj_rot_dist`, `end_obj_trans_err`, `foot_sliding`,
both penetration losses) separate all four thresholds. The earlier claim that the cells were
"identical to full float precision" overstated the flatness.

**The motion changed on 22-33 sequences, not 2.** The earlier "2 of 438" figure counted *contact
fields only*. Counting **all** metric fields, the sequences whose motion changed against A0 are:

| threshold | sequences changed vs A0 |
| ---: | ---: |
| 0.90 | 22 / 438 |
| 0.75 | 26 / 438 |
| 0.50 | 32 / 438 |
| 0.25 | 33 / 438 |

Monotone in mask width, as a real dose-response on the trajectory should be. On those sequences
`end_obj_trans_err`, `foot_sliding`, `mpjpe`, `obj_rot_dist`, `obj_trans_dist`,
`pelvis_goal_error_cm` and `trans_dist` **all** differ; `hand_pen_loss_omomo` differs on 7 and
`human_pen_loss_infbagel` on 9 (at 0.25). Only the contact fields are near-degenerate:
`contact_f1` on 2 sequences, recall on 1, precision on 1.

So the framing "the gate opened; nothing came through" is **wrong and is withdrawn**. The gate
opened, the **motion changed** on a growing set of sequences, and **contact did not follow**.
That is a sharper null than a no-op would be: the perturbation propagated into the trajectory and
still failed to move a single contact decision on 435 of 438 sequences.

A descriptive paired sequence bootstrap against A0 (438 sequences, seed 42, 10,000 replicates)
is null on `contact_f1`, `contact_recall`, `contact_precision`, `hand_pen_loss_omomo`, `mpjpe`
and `end_obj_trans_err` for every cell; the largest effect is `contact_recall` +0.0000240
[0.0000000, +0.0000720]. This is **not** the preregistered PRIMARY — see the split section.

## The dilution objection: checked, and it fails

The obvious objection to the paragraph above is that the aggregate averages 438 sequences while
only 33 changed, so a real per-sequence effect would be diluted 13× into invisibility. I tested
this rather than argue it. Restricted to the **33 sequences that changed at threshold 0.25**,
comparing 0.25 against 0.95 on that subset only:

| metric | 0.95 | 0.25 | delta |
| --- | ---: | ---: | ---: |
| `contact_recall` | 0.65883 | 0.65915 | +0.000319 |
| `contact_f1` | 0.69396 | 0.69406 | +0.000096 |
| `contact_precision` | 0.85961 | 0.85912 | −0.000494 |
| `mpjpe` | 13.06671 | 13.06770 | +0.000984 |
| `trans_dist` | 8.82020 | 8.82249 | +0.002289 |
| `pelvis_goal_error_cm` | 3.54819 | 3.54702 | −0.001173 |
| `hand_pen_loss_omomo` | 0.02773 | 0.02773 | −0.000006 |

The subset deltas are **~13× the aggregate deltas — exactly the 438/33 dilution factor**. Undiluting
therefore recovers no new information; it reproduces the same effect at the same size, rescaled.
And the per-sequence effect is itself negligible: `contact_recall` moves +0.000319 on the sequences
that actually changed, against a `contact_percent` gap to GT of ~0.15. That is three parts in ten
thousand against a deficit of three parts in twenty. `contact_precision` and `pelvis_goal_error_cm`
move slightly the *wrong* way even on the changed subset, so the restriction reveals no suppressed
benefit either.

**Recorded as a checked-and-rejected alternative explanation.** The null is a property of the mask
lever, not of averaging over unaffected sequences.

## The mask plumbing verifiably worked

This is the scientific content of the null, and it is why the result is informative rather than
a bug report.

| threshold | `guidance_loss_mean` | `guidance_hand_loss_mean` |
| ---: | ---: | ---: |
| 0.95 | 5063.738426 | 484.505195 |
| 0.90 | 5060.665509 | 484.196867 |
| 0.75 | 5059.053819 | 484.035608 |
| 0.50 | 5058.300347 | 483.959969 |
| 0.25 | 5056.710069 | 483.799491 |

Each cell's **own** `normalization_audit.inference_guidance` records a monotonically lower
guidance loss at a lower threshold. `guidance_applied_steps` is 27 in every cell,
`guidance_nonfinite_steps` is 0 in every cell. The widened mask was read by the author's
`apply_hoi_guidance_loss` exactly as intended.

## Retraction: the mask barely opens at the guided steps

**I claimed earlier in this phase that the author-visible mask expands from 1000 frames at 0.95 to
1368 frames at 0.25 (+36.8%), with 18% of probed frames in the newly opened `(0.25, 0.95]` band.
That probe did not measure the tensor guidance actually consumes, and those figures are RETRACTED.**
It measured decoded contact outside the guided reverse loop, not the contact channels present at
the guided steps.

A verified decomposition on the **real guided path** — reconciliation against the unpatched path
exact, 72/72 checks, max abs err 0.0 — shows the 0.95 → 0.25 sweep opens only

> **6 of 7200 channel elements — 0.083%.**

The reason is saturation. At the guided steps the predicted contact logits are near-saturated and
bimodal: **43.67%** of elements are already above 0.95 and **43.75%** are above 0.25, so almost
nothing lies in the band between. Arm B guides only the **last 10 reverse steps**, where `x0_hat`
is nearly clean and the contact predictions have already binarized. There is essentially no mass
left for a threshold to reclassify.

This makes the null over-determined: the mask does not merely fail to help — at the guided steps it
barely changes at all.

## Mechanism of the null

Measured on 8 sequences via a monkeypatched real evaluator run, same reconciliation as above.

**The two hand sub-terms move in opposite directions and the wrong one wins.** Across 0.95 → 0.25:

| sub-term | 0.95 | 0.25 | delta | |
| --- | ---: | ---: | ---: | --- |
| `loss_contact` | 0.02117535 | 0.02125364 | +0.000078 | **UP** |
| `loss_consistency` | 1.39546636 | 1.39422995 | −0.001236 | **DOWN** |

Net: **DOWN**. On the **6 of 36** steps where the mask actually changed the ratio is **15.79×**,
against a closed-form prediction of **15.58×** — so the cancellation is understood analytically,
not merely observed.

**The cause is structural, and no threshold can fix it.** `loss_contact` is an L1 mean over
`bs*T*2` slots, so one newly admitted frame adds `(d − 0.02) / (2*bs*T)` — **linear** in the number
of admitted frames `k`. `loss_consistency` masks a `T × T` outer product, so one newly admitted
frame lights ~`2k` pairs and subtracts `2k*s_bar / (bs*T*T)` — **quadratic** in `k`. The ratio is
`4k*s_bar / (T*(d − 0.02))`, which **grows with `k`**: every frame the threshold admits adds a
little contact pull and subtracts proportionally more consistency penalty. **A wider mask
over-cancels harder.** The pathology is in the relative scaling of the two sub-terms, not in where
the cut is placed.

Supporting facts, all from the same decomposition:

- `loss_consistency` is **98.5%** of the hand term; `loss_contact` only **1.5%** — 1.86% of the
  total loss value, but **29.2% of the total gradient norm**. `loss_contact` is not negligible in
  the update; it is outvoted in the loss by a term that moves the other way.
- **The clamp is not binding**: 449 / 835,200 = **0.0538%** of update elements saturate at
  `clamp=1.0`, so the clamp cannot be blamed for the null.
- **The direction is unchanged**: `cos(total grad @0.95, total grad @0.25)` is **0.9977** on
  average and **exactly 1.0 in 30 of 36 steps**. The threshold does not redirect guidance; it
  barely rescales it.

Three independent measurements therefore converge: only 0.083% of channel elements change, that
change is over-cancelled ~15.8× inside the hand term, and the resulting gradient is
direction-identical with a non-binding clamp. **The mask threshold is not a lever on this path at
all, and the failure is structural rather than a matter of tuning.**

## Retraction: I was wrong about the feet term

**I claimed earlier in this phase that the hand-object term is drowned by the feet term** —
`code/guidance_loss.py:93` weights hand ×10 and feet ×500 — **and I put the hand term's share of
the guidance loss at 13.1%. That is wrong and is retracted here, not quietly dropped.**

The sealed runs' own audits refute it:

| quantity | value | share |
| --- | ---: | ---: |
| `guidance_loss_mean` | 5063.738426 | 100% |
| `guidance_hand_loss_mean` × 10 | 4845.051951 | **95.68%** |
| `guidance_feet_weighted_mean` | 218.686475 | **4.32%** |

The two add to the total exactly. The ×500 feet weight multiplies a feet loss whose unweighted
mean is only **0.437373**, so its weighted contribution is 218.69 against 5063.74. **The hand
term is not drowned; it dominates.** My 13.1% came from an unrepresentative ad-hoc probe whose
feet loss was ~2000× the real value. The sealed audits were already in the tree and would have
refuted the claim before I made it; I did not read them first.

This matters for the conclusion, not just for the record. It removes the "wrong term is being
weighted" escape hatch. The mask perturbs a loss the hand term already dominates by 22:1, and
although the motion does move on 22-33 sequences, contact does not follow. What remains is the
decomposition above: at the guided steps the mask barely opens (0.083%), and what little it opens is
over-cancelled ~15.8× inside the hand term itself. That is also consistent with the recorded failure
mode — contact failures are gross misses of ≥8 cm, not near-misses that a 2 cm-deadband L1 pull
could close in 10 late reverse steps.

## Preregistered obligations, recorded honestly

**Engagement.** `contact_percent` is reported beside every contact and penetration number above.
GT 0.661883, A0 0.508987, A4 0.509024. Engagement did **not** rise, so no contact number here is
bought by engagement and no penetration number is flattered by under-engagement.
`hand_pen_loss_omomo` is unchanged at 0.16676 across all five cells.

**The select/confirm split was not exercised.** The preregistration fixed a
`sha256('42:' + name)[0] & 1` split — 209 select / 229 confirm — before any result existed, to
guard threshold selection. Because the dose-response is numerically flat, **no threshold was
selected**, and the PRIMARY test on the confirm half was **never run**. The split validated
nothing. It must not be presented as if it had. Its only value here is that it was fixed in
advance and therefore could not have been reshaped by the outcome.

**Cell U (GT mask) was not run.** It was the non-deployable upper-bound probe answering "how much
contact could guidance recover if the engagement decision were perfect". It was not executed as part
of P5. The preregistration's branch — *a null here means the mask is not the bottleneck and the
direction is dropped immediately* — was reached from the five predicted-mask cells: the mask
demonstrably reached the loss, the motion demonstrably moved, and contact demonstrably did not
follow. **Consequence, stated plainly: the ceiling on a perfect engagement decision is UNMEASURED by
P5.** This result refutes the predicted-mask threshold as a lever; it does not prove that a GT mask
would also be inert. Cell U has since been **separately authorized as an upper-bound probe under a
new preregistration**, so P5 records U as not run without claiming the direction is permanently
closed.

**Cost.** Four evaluation cells, launched together on 4 GPUs, 22:01:26 → 22:15:32 local — 14.1 min
of wall clock for all four, ≈0.94 GPU-hours. Each cell's own `end_to_end_seconds` is 841.9-846.4 s
under 4-way contention; the sealed A0 cell ran solo in 222.3 s (~3.7 min). Wall clock here is not
a performance record.

**Gates.** `nonfinite_values` 0, `position_outside_rate` 0.0, `object_outside_rate` 0.0,
`guidance_nonfinite_steps` 0, 438 sequences, `scene_condition_loaded` false — on every cell.

## Governance

Zero training, zero checkpoints, no training run id allocated, no sealed row replaced, no existing
classification changed. The source change is confined to `code/priors/inference_guidance.py` and
`code/config/sampler/hoi_prior.yaml`: the mask source and threshold become configurable with
defaults (`predicted` / `0.95`) that keep the guided path bit-identical to every sealed P2/P3/D2-AI
result. `code/guidance_loss.py` was not edited — the two contact channels it consumes are rewritten
to hard 1.0/0.0 so the author's own `> 0.95` reproduces the selected mask exactly.
`code/eval_metrics.py` and the official 438 protocol are untouched. Guidance remains default-off.

## What this does not show

- Cell U was not run as part of P5; the perfect-engagement ceiling is unmeasured here. It has since
  been separately authorized as an upper-bound probe under a new preregistration.
- Only the threshold was varied. `guidance_scale`, `last_steps`, `clamp` and the 2 cm contact
  deadband (`code/guidance_loss.py:36`) were held at sealed Arm B values.
- Single seed, single checkpoint, single lineage (D2-AI final online).
- The earlier mask-expansion probe (1000 → 1368 frames, +36.8%, 18% in the opened band) is
  **retracted**; the correct figure at the guided steps is 0.083% (6/7200 channel elements). The
  retracted numbers must not be cited.
- The guided-path decomposition was measured on 8 sequences, not 438; its reconciliation is exact
  (72/72 checks, max abs err 0.0) but its sample is small.
- Nothing here revisits whether guidance helps at all; A0 versus unguided D2-AI is the sealed
  D2-AI result and is unchanged.

## Next entry point

The mask direction is closed. Contact engagement remains the single open deficit —
`contact_percent` 0.509 against GT 0.662, unmoved by this lever — and the next proposal must
attack it without going through the guidance contact mask, which is now measured to be inert.
Eleven model-side or inference-side interventions have now been absorbed or measured null; any new
direction requires a dated plan amendment and a registry entry before code changes.
