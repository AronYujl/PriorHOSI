# Phase 1B HOIPrior evidence index

Status: compact research handoff through D2-AG0, P1 protocol decomposition,
P2 inference guidance, the D2-AH negative preflight, the P3 relation-field
lineage under guidance, the P4 budget-metric curve, the D2-AI/D2-AJ
long-budget arms, the P8-P9c hand-object geometry weight sweep, the P10
geometry-term repair, the P11 root-gradient detach and the P12 representation-frame
repair, 2026-08-20.

**Every model-side geometric number from the sealed D2-* rows is void and not
recomputable** after P12: the released code split the rotation and joint channels
into two worlds 90 deg apart, so the channel those checkpoints fitted no longer
exists. Methodology conclusions still hold; data assets and input hashes are unchanged.

Use this file as the first research-context entry point. It summarizes conclusions;
the named phase summaries and compact JSON files remain authoritative for exact
protocols, confidence intervals and hashes.

## 1. Locked comparison points

All D2-* values are from the unchanged official 438-sequence, three-window, 500-step
unguided native protocol, except the P2 and P3 rows, which are the same sealed
checkpoints under the same protocol with inference-time contact guidance added and are
marked as such. The released InfBaGel row is measured on the same 438
sequences with a byte-identical `code/eval_metrics.py`, but **not** under that protocol:
per `results/experiments/p0-hoi-table5-baseline-s42-20260712/resolved_config.yaml` it was
produced at commit `c358fa4` with the consistency sampler (`sample_type: consistency`,
`cm_timesteps: 16`), inference guidance (`guidance_weight: 1`), CFG (`w: 1`) and
scene/object-voxel conditioning (`load_scene: true`, `add_object_voxel: true`); the
autoregressive rollout block of `code/test_infbagel_hoi.py` was also rewritten after
`ffc548a` (2026-07-13), so the released row predates the rollout used by every D2-* row.
Do not read the released row against the D2-* rows as a protocol-matched comparison.

| model | end-object cm | FS | contact P/R/F1 | hand penetration | MPJPE cm | status |
|---|---:|---:|---:|---:|---:|---|
| released InfBaGel | 3.0372 | 0.33336 | 0.79081 / 0.72759 / 0.72726 | 0.16240 | 11.9976 | baseline only; guided 16-step consistency, not the D2 protocol |
| released InfBaGel, guidance off | 3.1251 | 0.36903 | 0.78115 / 0.64877 / 0.66842 | 0.18591 | 12.0328 | 2026-08-01 A-old; same checkpoint at `cm_timesteps: 16` with `guidance_weight: 0`; the closest protocol-matched reference the released model has |
| D2-V long budget | 3.6807 | 0.37828 | 0.78911 / 0.58529 / 0.62859 | 0.26405 | 12.1224 | strong but FS/penetration negative |
| D2-X FK-foot routing | 3.7402 | 0.36301 | 0.78806 / 0.59445 / 0.63743 | 0.24536 | 12.0508 | sealed autonomous control |
| **D2-AI full budget (299.52M, 4.875x)** | 3.8375 | 0.35435 | 0.81474 / 0.61480 / 0.65366 | **0.17481** | **11.7415** | 2026-08-04; budget the ONLY manipulated factor vs D2-X. **9 of 18 metrics significantly better on a 438-sequence paired bootstrap, 0 significantly worse.** `human_pen` 2.76049 (−1.109), `obj_trans_dist` 14.8198 (−1.174), `trans_dist` 7.7014 (−0.469). With Arm B: 3.8965 / 0.34177 / 0.81579 / 0.63819 / 0.67570 / 0.16676 / 11.7699, and **8 of 17 metrics beat released** (up from 4/17 for D2-X + Arm B). Contact is the cost: `contact_percent` 0.49045 unguided, deviation from GT grows +45.2% → +140.5% |
| D2-AJ split goal tokens | — | — | — / — / 0.63753 | — | — | 2026-08-04; **stopped early at the preregistered 61.44M go/no-go**, both criteria null (`contact_f1` +0.00010 [−0.02222, +0.02209]; `end_obj` −0.04849 [−0.29809, +0.19882]) and `pelvis_goal_error_cm` significantly *worse*. Tenth failed model-side intervention; its own preregistration predicted this |
| D2-X + P2 inference guidance, Arm A | 4.2156 | 0.39903 | 0.80216 / 0.72332 / 0.73071 | 0.24265 | 12.3466 | 2026-08-02 P2; **guided**, not the unguided native protocol. Protocol alignment, not a model change: same sealed D2-X weights sampled with the author's full `apply_hoi_guidance_loss`. Statistical contact parity with the guided released row (recall, precision, F1, contact_percent all cross zero). Arm B (late-steps-only, variance-scaled, clamped) is 3.8401 / 0.35565 / 0.81223 / 0.70783 / 0.72653 / 0.22942 / 12.0794 and is the arm that survives cost scrutiny |
| D2-AB no-slip objective | 3.6840 | 0.36606 | 0.78957 / 0.59533 / 0.63831 | 0.23832 | 12.0639 | mechanism/FS negative |
| D2-AC local adapter | 5.6473 | 0.39861 | 0.78758 / 0.60416 / 0.64799 | 0.25179 | 12.4268 | locality and protection negative |
| D2-AD local-frame adapter | 4.2373 | 0.42539 | 0.76795 / 0.53300 / 0.58687 | 0.21656 | 12.3847 | locality and transfer negative |
| D2-AE sparse relation field | 4.2990 | 0.39896 | 0.80363 / 0.59614 / 0.64194 | 0.17938 | 12.1558 | internal positive, transfer negative |
| D2-AF reliability routing | 5.5735 | 0.35958 | 0.79093 / 0.59904 / 0.64106 | 0.22689 | 12.4221 | mechanism and repair negative |
| D2-AG self-conditioned relation source | 3.6922 | 0.40092 | 0.81120 / 0.59850 / 0.65009 | 0.18367 | 12.0129 | source-provenance and transfer negative; **best D2 contact F1, precision and MPJPE**, and `hand_pen_ratio` `0.11287` beats the released guided row's `0.13286` at D2-X-level engagement |
| D2-AG + P3 inference guidance, Arm B | 3.7448 | 0.39543 | 0.82237 / 0.65271 / 0.69343 | 0.17214 | 12.0498 | 2026-08-02 P3; **guided**, not the unguided native protocol. Penetration closes to `+6.0%` / `+6.1%` of the released guided row but contact parity fails and the protection bound against D2-X + Arm B is violated: preregistered **cost failure**. D2-AE + Arm B is 4.2940 / 0.39675 / 0.80585 / 0.68278 / 0.71027 / 0.18113 / 12.1231, keeps contact intact and pays on end-object and FS instead. Neither arm is selectable |

Released InfBaGel is not a valid initializer. D2-X is the sealed autonomous-diffusion
control; no D2 checkpoint is selectable as a new prior initializer.

The 2026-08-01 protocol decomposition (`docs/plan/PHASE_1B_HOI/05_INFERENCE_GUIDANCE.md`,
section "2026-08-01 Phase 1B 基线协议分解 P1") re-evaluated the released checkpoint on the same
438 sequences with iteration count and inference guidance ablated. Of the `0.1331`
contact-recall difference between the released row (`0.72759`) and D2-X (`0.59445`),
inference-time guidance accounts for `0.0788` (59.2%), the 16-vs-1 iteration count for
`0.0085` (6.4%), and the genuine model difference is `0.0458` (34.4%): released recall
is `0.64877` with guidance off at `cm_timesteps=16` and `0.64029` at one unguided step,
with contact F1 `0.66842` and `0.66552` and FS `0.36903` and `0.35059`. Every
"released minus D2" gap, gap-closure fraction and released-95% ratio in this repository
is therefore a cross-protocol quantity that overstates the model deficit.

## 2. Experiment map

### Representation, sampler and optimization diagnosis

- D2-B through D2-E isolated the BPS backend and geometry-equivalence questions.
  The linear geometric contract was ultimately validated; backend differences alone
  did not explain the rollout deficit.
- D2-F and D2-H tested reverse-manifold/current-state exposure explanations. Their
  registered diagnostics were negative; simple reverse-state exposure did not repair
  the model.
- D2-I through D2-L tested weighted-objective gradient dominance, clipping, AdamW
  routing and auxiliary balancing. All were controlled negatives. The evidence does
  not support another optimizer, clip or scalar-loss-balance retry without a new
  mechanism.
- D2-M/N tested fresh-optimizer and author-native transfer behavior. Balanced variants
  could improve one kinematic quantity while sharply damaging contact; historical
  checkpoints are not suitable initializers.
- D2-O/P/P5/Q/R/S examined contact alignment, coordinate defects, author contact
  guidance, state-routed guidance and denoiser-response frontiers. They exposed mixed
  contact deficits and coordinate sensitivity but did not produce an authorized
  training route that improved the native Pareto frontier.

### Strong from-random training lineage

- D2-T showed that the author update rule was insufficient.
- D2-U's balanced objective was a major positive training mechanism, but the short
  budget remained far below baseline contact and object quality.
- D2-V's tenfold 61.44M-window budget established that the 232-D autonomous denoiser
  can learn strong kinematic, object and contact behavior from random initialization.
  Its remaining deficits were FS, penetration and contact recall, not basic capacity.
- D2-W rejected late constant-LR degradation as the primary cause: the final D2-V
  checkpoint was better than the midpoint on the registered FK-foot diagnostic.

### FK-foot and no-slip family

- D2-X routed evaluator-aligned FK-foot temporal residuals. It preserved D2-V and is
  the current sealed control, but its paired FS improvement CI crossed zero.
- D2-Y amplified routed foot residuals. The teacher-forced surrogate moved, while
  official FS did not improve significantly and end-object/contact protection failed.
- D2-Z added immutable-GT near-ground gating. It did not improve official FS and also
  harmed end-object/contact protection.
- D2-AB used predicted-support no-slip supervision. Support sanity passed, but the
  optimized supported-velocity mechanism moved in the wrong direction and native FS
  did not improve. A new no-slip scalar loss is therefore not supported by this line.

### Interaction-representation family

- D2-AC added local object tokens and role queries. Whole-gate ablation showed strong
  adapter use, but correspondence permutation had no significant effect. The adapter
  behaved as a generic high-leverage conditioning path, not a local relation mechanism.
- D2-AD repaired the coordinate frame with human-local full-mesh BPS. The repair was
  geometrically correct but did not make local correspondence causal; native contact
  and several protection metrics worsened. It also exposed the unacceptable CPU
  full-mesh/KD-tree bottleneck.
- D2-AE replaced that path with a GPU-native 100-point current-state sparse field,
  structurally bound to left hand, right hand and pelvis with fixed temporal routing.
  All five internal causal gates passed, including gate use, temporal correspondence
  and role binding. Native contact F1 improved only `+0.00452` over D2-X with CI
  crossing zero; recall barely moved, while end-object and FS protection failed.
- D2-AF multiplied the D2-AE writeback by canonical `sqrt(alpha_bar[t])`. All seven
  internal reliability/path/temporal/role gates failed. Native F1 was `0.64106`,
  statistically indistinguishable from D2-AE and D2-X, and end-object worsened to
  `5.5735 cm`.
- D2-AG kept the D2-AE field structurally unchanged and moved only the tensor the
  variable temporal anchors `5/10/15` read from: the model's own detached `x0_hat`
  instead of the current noisy `x_t`, symmetrically in training (per-sample
  Bernoulli `p=0.5`) and sampling (`prev_x0`, `x_t` at the first reverse step), with
  `s[:, :2]` pinned to `x_t` on both sides. Three of five internal gates failed and
  the nulls are tight, not underpowered: substituting the source back to `x_t` moves
  union 5-cm F1 by `-0.00411` `[-0.01317, +0.00524]` while role swap and temporal
  permutation move it by `0.305` and `0.184`. Native transfer failed all four
  registered checks (F1 `0.65009` against the registered minimum `0.65988`; released
  gap closure `0.1409` against `>=0.25`), and foot sliding regressed significantly
  (`0.40092` against D2-X `0.36301`, ratio CI upper `1.1838`). Classification
  `selfcond-relation-source-transfer-negative-stop`; the checkpoint is not selectable.
  **But its accuracy-side numbers are the best of the D2 family and they are not an
  engagement artefact**: at essentially identical engagement to D2-X
  (`contact_percent` `0.47706` vs `0.47655`, recall `0.59850` vs `0.59445`) it reaches
  hand penetration `0.18367` vs `0.24536` and precision `0.81120` vs `0.78806`, and its
  `hand_pen_ratio` `0.11287` is already better than the released guided row's `0.13286`.
  This is the opposite pattern to the D2-AH epoch100 artefact, where penetration only
  looked good because `contact_percent` collapsed to `0.3192`. See
  `docs/phase_summaries/PHASE_1B_D2AG.md`.

### Objective-weighting family

- D2-AH (2026-08-02) proposed restoring the author's `fk_weight`/`object_surface_weight`
  of `50.0`, an uncontrolled constant inherited byte-identically by ten 61.44M-window
  runs from a decision made at one tenth of that budget. Its preregistered pre-flight
  diagnostic **failed its fixed abort rule and no formal training was ever started**:
  the author's own recipe at `60,384,668` windows (98.3% of D2-X) reads `xy_points_err`
  `5.7623` against D2-X `4.0505` (+42.3%) and `end_obj_trans_err` `5.4176` against
  `3.7402` (+44.8%). The preregistered tree-effect control confirmed rather than
  excused it: seven of eight metrics reproduce the earlier `1e982bc` epoch500 row within
  1.62% relative, but `end_obj_trans_err` moves `+0.3642` (+11.0%), a real
  metric-specific tree effect on a gate metric that is still ~99x too small on
  `xy_points_err` and insufficient on end-object. At `299,531,868` windows (4.875x the
  formal budget) the same recipe does beat D2-X on `xy_points_err` (`−0.8198 cm`),
  contact F1 (`+0.0286`), recall (`+0.0493`) and hand penetration (`−0.0517`), so what
  is refuted is the affordability of the remedy at 61.44M windows, not the 135x/52x
  under-weighting diagnosis. The epoch100 penetration and FS advantages are engagement
  artefacts (`contact_percent` `0.3192` against GT `0.66188`). Classification
  `metric-geometry-weight-restoration-preflight-negative-stop`; see
  `experiments/results/p1_hoi_d2ah_negative_preflight_s42_20260802.json`.
- The same diagnostic produced a finding that reprioritises the remaining gaps: on
  `end_obj_trans_err`, D2-X (`3.7402`) is statistically indistinguishable from the
  author's from-scratch diffusion recipe at 4.875x budget (`3.6866`; paired sequence
  bootstrap mean `−0.0679`, CI `[−0.3798, +0.2521]`), while the released checkpoint
  reads `3.0372` and the in-house CM reproduction `3.5553`. **The end-object gap to the
  released model is therefore not attributable to the diffusion training recipe**;
  consistency distillation explains `0.1313` of the `0.6493` residual and `0.5181`
  remains unexplained. Future effort on this metric should move off diffusion-training
  changes.
- P8/P9/P9b/P9c then established the objective-side geometry lineage: an eight-point
  weight dose-response produced its one positive and selected W3 (`weight=3`). P10's
  hinge × object-detach formula 2×2 was negative; P11's root-gradient detach was also
  negative. P11 removed a gradient path from an existing loss rather than adding content
  to the network, so it is the **second consecutive objective-side negative after P10**
  and does not increment the model-side count. See conclusions 14–18.

### Relation field x inference guidance

- P3 (2026-08-02) completed a 3x2 `{D2-X, D2-AG, D2-AE} x {500-step unguided, 500-step
  + P2 Arm B}` in one execution tree, five new cells plus the sealed D2-X + Arm B cell
  reused by reference, with frozen Arm B hyperparameters and no sweep. Its preregistered
  verdict is a **cost failure**. Penetration closure passed: D2-AG + Arm B reads
  `hand_pen_loss_omomo` `0.17214` against the released `0.16240` (`+6.0%`, previously
  `+41.3%`) and `human_pen_loss_infbagel` `2.74657` against `2.58927` (`+6.1%`,
  previously `+40.0%`), both inside the preregistered `+/-10%`. But contact parity
  failed (`contact_f1` `0.69343` against the `0.72726 - 0.02` floor) and PROTECTION (i)
  was violated (`-0.0331` `[-0.0517, -0.0145]` against D2-X + Arm B). PROTECTION (ii)
  held (`end_obj_trans_err` `+0.0515` `[-0.0091, +0.1178]` against the `+0.25 cm` bar),
  and every finiteness gate passed. Classification
  `relation-field-guidance-contact-redundancy-cost-negative-stop`; see
  `experiments/results/p1_hoi_p3_relation_field_guidance_s42_20260802.json`.
- **All three preregistered same-tree unguided hash gates passed bit-exactly**
  (D2-AG `eb701cf4...`, D2-X `69cc811c...`, D2-AE `8533b66e...`). The
  `5f7dde7 -> c40dc00` guidance implementation is therefore **empirically inert on the
  unguided path for three different architectures**, which is a methodological result in
  its own right: it is the first measurement of that boundary rather than a source-level
  guard argument. Consequence: no sealed unguided cell needed substitution, and the main
  effects are interpretable, not only the tree-immune interaction term.
- The pre-run additive prediction (`contact_f1` `0.73919`) was **falsified by `-0.0458`,
  exactly where the plan said an interaction would show**. The interaction
  `(AG_g - AG_u) - (X_g - X_u)` is `-0.0458` `[-0.0591, -0.0329]` on `contact_f1` and
  `-0.0592` `[-0.0759, -0.0428]` on `contact_recall`, while the interactions on hand
  penetration, body penetration, foot sliding and end-object all cross zero. The
  interaction is on contact and only on contact.
- The mechanism is a **monotone dose-response** that the declared second arm was built to
  expose: the guidance gain on `contact_f1` is `+0.0891` on D2-X, `+0.0683` on D2-AE and
  `+0.0433` on D2-AG, and D2-AE's interaction against D2-X (`-0.0208`
  `[-0.0359, -0.0057]`) is about half D2-AG's. The sparse relation field damps guidance's
  contact effect; making its source self-conditioned damps it about twice as much. This
  retroactively vindicates part of P2's reason for excluding D2-AG - **not on safety**
  (non-finite values and out-of-range positions are zero everywhere) **but on efficacy**.
  P3's overturn of that exclusion was correct on safety and wrong on efficacy; neither
  record is rewritten.
- The pre-committed engagement-artefact narrative **did not materialise**. In the
  unguided cells engagement is essentially identical (`contact_percent` `0.47655` /
  `0.47706` / `0.47663`, both relation-field-minus-D2-X contrasts crossing zero on both
  `contact_percent` and `contact_recall`) while `hand_pen_loss_omomo` is `0.24536` /
  `0.18367` / `0.17938`. The relation field's penetration advantage is **not
  engagement-bought**.
- Synthesis: inference guidance and the sparse relation field are **partially redundant
  on contact** - both push the hand toward the object, so their contact gains do not
  stack - and **complementary on penetration**, which guidance barely moves (D2-X
  `0.24536 -> 0.22942`, D2-AG `0.18367 -> 0.17214`; all three guidance main effects on
  hand penetration cross zero). Neither arm dominates: D2-AG closes penetration further
  but loses contact significantly, D2-AE keeps contact statistically intact
  (`contact_f1` `-0.0163` `[-0.0342, +0.0016]` against D2-X + Arm B) but pays on
  end-object (`+0.4521` `[+0.2309, +0.6774]`) and foot sliding (`+0.0411`
  `[+0.0218, +0.0604]`) and its penetration gaps sit at `+11.5%` / `+11.1%`, just
  outside the band D2-AG cleared. No checkpoint was selected and both remain sealed
  negatives.

- **P12 (2026-08-20) repaired the coordinate representation frame and re-established the
  D2-AI baseline on it; classification `eval-consistency-null`.** Gate (ii) failed on
  `end_obj_trans_err` (`+0.12419` `[-0.02538, +0.28236]`, not significant), which is the
  preregistered trigger for that label -- but the label's rationale is contradicted by
  measurement: the step-0 window frame rule alone moves **7 of 14 metrics significantly,
  all favouring the repaired rule, none favouring the historical one** (`mpjpe` `-4.42681`
  `[-5.03786, -3.82698]`, `obj_trans_dist` `-6.92076`, `trans_dist` `-2.94284`,
  `contact_f1` `+0.06789`). The PRIMARY row is a **new baseline, not an improvement over
  D2-AI**: the frame factor's effects are 7-14x the observed PRIMARY-minus-D2AI deltas, so
  that comparison is confounded beyond decomposition. The 438-sequence ground-truth
  reference row is sealed with the correction that GT `foot_sliding` is **0.26346**, not the
  0.31654 written into P12's own gate (`foot_sliding` never passes through
  `interpolate_joints`), and because 0.26346 is below the released row's 0.33336 the metric
  is **not** demoted. `docs/phase_summaries/PHASE_1B_P12_REPRESENTATION_FRAME.md`,
  `experiments/results/p1_hoi_p12_frame_repair_baseline_s42_20260820.json`.

## 3. Strongest current conclusions

1. The scene-free 232-D denoiser has sufficient capacity: D2-V/D2-X reach strong
   native quality from random initialization.
2. Teacher-forced denoising or auxiliary-loss movement is not reliable evidence of a
   500-step rollout improvement.
3. Whole-gate ablation alone is weak causal evidence because jointly trained models
   can treat the path as a generic residual and gate-zero is out of distribution.
4. D2-AE proves that explicit current-state role/temporal relation structure can be
   learned causally and efficiently, but that structure did not improve native contact
   recall or goal protection under the existing training exposure.
5. D2-AF shows that a fixed signal-reliability attenuation is not enough and may cause
   the relation path to become unused.
6. Contact precision is already close to the released model. The largest interaction
   gap is recall/coverage, while end-object, FS and penetration must remain protected.
   The 2026-08-01 decomposition rescales this: against a protocol-matched unguided
   released row, D2 precision is at or above it (`0.78806` vs `0.78115`) and the recall
   gap is `0.0458`, not `0.1331`.
7. Width, depth, token count, point count, adapter placement, LR and batch sweeps have no
   positive evidence and are poor uses of the remaining budget. **Longer budget is the one
   exception, and it is now measured rather than extrapolated.** D2-AI (2026-08-04) trained the
   sealed D2-X recipe at 299,520,000 windows (4.875x, budget the only manipulated factor) and is
   **significantly better on 9 of 18 metrics with 0 significantly worse** on a 438-sequence
   paired bootstrap: `obj_trans_dist` −1.174, `human_pen` −1.109, `trans_dist` −0.469, `mpjpe`
   −0.309, `hand_pen` −0.071, `obj_rot_dist` −0.034, `contact_precision` +0.027, and both
   penetration ratios. Metrics beating released rise 4/17 → 8/17, and the two largest real gaps
   (hand/human penetration, +41.3%/+40.0%) collapse to **+2.7%/+1.8%** — a training-side gain,
   since unguided D2-AI already reaches 0.17481 and 2.76049. P4's *direction* was right and its
   *magnitude* was not: the log-log fit predicted `contact_f1` ~0.856, the measurement is
   0.65366. The earlier pessimism was never tested — D2-V's "long budget" was itself 61.44M, a
   tenfold step up from the 6.144M screening budget that *established* 61.44M as formal.
   See `experiments/results/p1_hoi_d2ai_d2aj_long_budget_arms_s42_20260804.json`.
8. The author's dynamic occupancy offers direct temporal spatial routing but mixes
   scene supervision and train/sample relation sources. Copying it would violate the
   independent scene-free HOIPrior objective.
9. Contact and penetration respond to different levers, and contact levers do not
   stack. P3 measured a significant negative interaction between the sparse relation
   field and inference contact guidance on contact F1/recall/percent, with no
   interaction on any penetration, foot-sliding or end-object term. Any future proposal
   that expects to add a contact mechanism on top of an existing one must state why it
   would not be absorbed the same way. **D2-AI adds a second form of this**: it responds to
   Arm B guidance far less than D2-X did (engagement 0.49045 → 0.50899 versus D2-X's
   0.47655 → 0.56956), so the contact parity D2-X + Arm B enjoyed was substantially
   *bought by guidance* and does not survive a stronger base model.
10. **The held-out denoising validation loss anticorrelates with the native rollout
    metrics and must not gate budget, early-stopping or checkpoint decisions.** All nine
    D2 configs show `total` rising +5.6..+12.4% and the `contact` term +25..+31% after a
    minimum at 21.5-24.6M windows. P4 tested whether that reached metric space and
    falsified it in the opposite direction: `contact_f1` at 21.504M is **0.108 below**
    61.44M, CI [-0.1340, -0.0827], 438 paired sequences. **D2-AI reproduces this far more
    strongly over 4.875x**: validation `total` bottoms at 27,648,000 windows and rises
    **+22.7%** by 299,520,000 while the native metrics improve on 9 of 18 with zero
    regressions. Gating that run on validation loss would have stopped it near 27.65M and
    forfeited every gain. The validation loss is single-step denoising; the metric is a
    500-step reverse diffusion chained across three windows on generated history.
11. **HOIPrior is under-engaged at every budget, so any penetration improvement must be
    read against `contact_percent` first, and engagement is now the primary remaining
    deficit.** It climbs monotonically 0.28358 → 0.47655 across the P4 curve against GT
    0.66188, and D2-AI at 4.875x still reads only 0.49045 unguided. **D2-AI made this more
    visible, not less**: `contact_percent` deviation from GT grows +45.2% → +140.5% and
    `contact_recall` +2.7% → +12.3% against released, and it is the single mechanism
    explaining all three remaining contact-side deficits. Penetration and foot sliding can
    *worsen* with budget purely because engagement rises — same confound as the D2-AH
    epoch100 row.
12. **`end_obj_trans_err` and `xy_points_err` are goal-recall metrics, not forecasting
    metrics.** The object goal handed to the model *is* GT object translation at
    `end_range-4`, which is the exact frame the metric scores (verified across all 438
    sequences, agreeing to 0.0574 cm — float round-trip only); `pelvis_goal` equals GT
    pelvis at frame 15 to 0.0000 cm while the metric scores frame 14. Both HOIPrior and
    released/e500 receive both goals, so the comparison is commensurable and no "gap vs
    released" number has a goal-leakage problem — but a gap on these two metrics is a
    constraint-satisfaction deficit, not a dynamics-prediction deficit. Prose elsewhere that
    describes these two metrics as prediction quality is wrong and should be qualified.
    Note `xy_points_err` does **not** exist per sequence; the per-sequence analogue is
    `pelvis_goal_error_cm`, which cost the D2-AJ preregistration one unevaluable criterion.
13. **Restructuring the goal conditioning pathway is inert — the tenth failed model-side
    intervention.** D2-AJ split the fused `Linear(12,512)` goal/progress token into separate
    pelvis (xz), object and progress tokens (+525,312 params, 4 → 6 condition tokens),
    mirroring released InfBaGel's three-module structure, at budget matched to D2-AI. At the
    preregistered 61.44M go/no-go both criteria were null (`contact_f1` +0.00010
    [−0.02222, +0.02209]; `end_obj_trans_err` −0.04849 [−0.29809, +0.19882]) and the one
    significant informational metric moved the *wrong* way (`pelvis_goal_error_cm` +0.19289
    [+0.04281, +0.34125]). Stopped early at 50.3% of budget. Its own preregistration had
    stated the arm was more likely inert than not, because the fused layer is already an
    affine map of the same information and the split is close to a first-layer
    reparameterization. Nine loss/representation attempts plus this now point the same way:
    **model-side additions get absorbed as generic residuals.** Do not revisit without a
    mechanism that addresses absorption directly.

14. **A GT-contact-masked hand-object relative geometry training term is the first training-side
    intervention to close the engagement gap — selected weight is 3.** P8-P9-P9b-P9c swept
    `hand_object_contact_weight` ∈ {1,3,5,8,10,15,50} at fixed 299.52M budget, all evaluated with
    sealed guidance (`contact_weight=3`, `object_goal_weight=1`). Winner **W3 (weight=3)**:
    `contact_percent` 0.64230 (gap +0.020 to GT 0.66188, ~84% closed), `contact_f1` 0.77886,
    `mpjpe` 12.313 (+0.569 vs no-geometry baseline). This directly contrasts with the D2-AJ-style
    "model-side additions get absorbed" pattern: the geometry term acts on the loss, not the
    network, so it cannot be absorbed as a residual. It also settles the P5/P6/U trilogy —
    inference guidance provably cannot create engagement (Cell U showed perfect labels make
    contact *worse*), and the deficit was confirmed training-side. See
    `experiments/results/p1_hoi_p8_p9_geometry_weight_sweep_sealed_s42_20260809.json` and
    `docs/phase_summaries/PHASE_1B_P8_P9_GEOMETRY_WEIGHT_SWEEP.md`.

15. **Contact gain saturates by weight ≤ 5; motion cost is nearly linear in weight.** Across the
    eight-point dose-response at 299.52M: `contact_percent` rises sharply 0→1 (+0.083) then
    decelerates (1→3 +0.024, 3→5 +0.009, 5→8 +0.002, 8→50 +0.010), while `mpjpe` delta grows
    ~linearly with weight (+0.28/0.57/0.78/0.88/1.15/1.42/2.50 at w=1/3/5/8/10/15/50). Higher
    weights (>5) trade three times the motion cost for ~1.7% more engagement — pure waste. The
    Pareto frontier is W3-W5; W3 selected for lower `mpjpe` cost. W15 is an outlier (contact 0.644
    below W10 0.657, hand_pen 0.571 worst) and not treated as signal.

16. **The geometry term moved contact metrics ahead of the released model, at a motion cost.**
    W3 vs released: `contact_percent` 0.64230 vs 0.59832 (ahead), `contact_f1` 0.77886 vs 0.72726
    (ahead), `mpjpe` 12.313 vs 11.998 (worse by +0.31 cm), `hand_pen_loss_omomo` 0.2587 vs 0.16240
    (worse). The hand is pulled toward the object, increasing penetration. Remaining shortfalls are
    `hand_pen` (penetration) and `end_obj_trans_err` (**4.6820** vs released 3.0372, a **+1.645 cm** gap).
    Phase 1B's budget lever (4.875×) remains the known complement: budget mitigates the geometry term's
    motion damage (H1−L1 `mpjpe` −1.55 cm) while improving native motion (H0−L0 −0.31 cm).
    *Corrected at P10 closure (2026-08-10):* this line previously read `end_obj_trans_err` `3.99`, which is
    the w=50 (H1) arm's `3.9898`, not W3's. The sealed sweep JSON's `dose_response` rows were always
    correct and only the prose misattributed them; the true gap is +1.645 cm, so the shortfall had been
    understated by about 73%.

17. **Reshaping the geometry term's target is ruled out as a penetration fix: the zero-distance target is
    not the cause of the penetration deficit.** P10 (2026-08-10) tested a 2x2 factorial repair of the P8
    hand-object geometry term — hinge ∈ {0, 0.02 m} × detach-object ∈ {false, true} — at 299.52M windows
    with `hand_object_contact_weight` **fixed at 3**, the sealed W3 arm reused as cell A00, all four cells
    evaluated under the byte-identical sealed P7 guidance on the official 438 sequences, and a
    10,000-replicate paired sequence bootstrap sharing one resample index across every cell and metric.
    **No cell was selectable**; classification `geometry-term-repair-negative-stop`, no checkpoint
    selected, and W3 (weight 3, hinge 0, detach false) remains the sealed contact configuration.
    - The load-bearing negative: **no cell puts `hand_pen_loss_omomo` or `human_pen_loss_infbagel`
      significantly below W3.** Hinge alone moves them in the predicted direction but not significantly
      (`hand_pen` −0.01795 [−0.04243, +0.00757], `human_pen` −0.27048 [−0.65888, +0.13716], n=181);
      detach alone makes both significantly *worse*; hinge+detach is null on both. The mechanism this
      sub-phase was built on — that driving palm-joint-to-predicted-surface distance to zero forces the
      surrounding hand vertices into the object — is **refuted**. All three new cells additionally lose
      `contact_f1` significantly against W3, so criterion (iii) fails as well.
    - **The hinge × detach interaction is significant and large on every object/motion metric and absent
      on every contact and penetration metric, so neither main effect is interpretable alone**:
      `end_obj_trans_err` −3.599 [−3.989, −3.217], `obj_trans_dist` −3.000 [−3.711, −2.300], `trans_dist`
      −0.539 [−0.834, −0.241], `mpjpe` −0.523 [−0.838, −0.209], `pelvis_goal_error_cm` −0.467
      [−0.745, −0.180] and `obj_rot_dist` −0.058 [−0.097, −0.020], while all three contact metrics, both
      penetration losses, both penetration ratios and foot sliding cross zero.
    - Concretely: **detach alone is catastrophic** for object placement (`end_obj_trans_err` +3.228
      [+2.811, +3.631], `obj_trans_dist` +3.774, `trans_dist` +3.327, `mpjpe` +1.020, all significant),
      **while detach+hinge recovers `end_obj_trans_err` to −0.246 [−0.630, +0.112]** — slightly better
      than W3 and not significant. The object gradient in the geometry term is therefore load-bearing
      rather than parasitic, but only because the target is zero-distance; hinging the target is what
      makes detaching the object survivable.
    - **P8/P9's aggregate-mean framing understated the W3-vs-H0 trade.** Re-reading the two sealed
      per-sequence files under P10's shared resample index gives, for W3 against the no-geometry
      baseline H0/D2-AI, **2 significant gains — both contact (`contact_f1` +0.0743
      [+0.0559, +0.0932], `contact_recall` +0.1185 [+0.0942, +0.1428]) — against 9 significant
      degradations** (`end_obj_trans_err`, `hand_pen_loss_omomo`, `hand_pen_ratio`,
      `human_pen_loss_infbagel`, `human_pen_ratio`, `mpjpe`, `obj_trans_dist`, `pelvis_goal_error_cm`,
      `trans_dist`), with 3 null. This is retroactive uncertainty on an already-sealed decision: it
      rewrites no P8/P9 value and does not unseal W3, but conclusions 14-16 should be read against it.
      The full per-metric table, its two input per-sequence hashes and the shared resample index are
      recorded under `results.retroactive_w3_vs_h0` of the P10 outcome registry row.
    - Caveat carried forward: `contact_percent` is a point estimate only (the evaluator computes it per
      sequence but does not persist it, and the official 438-sequence entry point was off-limits per the
      preregistration). The protection floor `contact_percent ≥ 0.60` held in every cell
      (A00 0.64230, A10 0.62639, A01 0.63809, A11 0.61981).

    See `experiments/results/p1_hoi_p10_geometry_term_repair_s42_20260810.json` (decision) and
    `experiments/results/p1_hoi_p10_geometry_repair_2x2_s42_20260810.json` (full factorial bootstrap),
    plus the registry rows `p1-hoi-p10-geom-{hinge,detach,both}-s42-20260809`,
    `p1-hoi-p10-eval-{hinge,detach,both}-guided-s42-20260810` and the outcome row
    `p1-hoi-p10-geometry-term-repair-s42-20260810`.

18. **Detaching the geometry term's root-translation gradient is ruled out: that gradient was
    load-bearing for global placement under the sealed W3 objective.** P11 (2026-08-15) changed only
    `hand_object_contact_detach_root=true`, leaving the forward value and every non-geometry FK consumer
    unchanged. The pre-GPU probe measured `root_gradient_share=1.011` and geometry-vs-non-geometry root
    gradient cosine `−0.319`; the hypothesis read that dominance as pathological. Native evaluation
    falsified that reading: `trans_dist` **8.43721016280387 → 17.52702829738458**, delta
    **+9.089818134580709** [**+8.02119313337368, +10.220037403018914**], and
    `pelvis_goal_error_cm` **4.650959228569446 → 12.904819034754414**, delta
    **+8.253859806184968** [**+7.431480166806359, +9.113061047471703**]. Both PRIMARY metrics were
    significantly worse, not better. Engagement also fell (`contact_percent`
    0.642295426542002 → 0.6169638327172573; `contact_f1` significantly worse), while protection failed
    on both `end_obj_trans_err` and `mpjpe`. Classification `root-coupling-negative-stop`; no checkpoint
    selected; next entry returns to P10's “objective has attractors but no repulsor” pointer.
    - Ten of 14 metrics were significant, all in the worse direction; zero were significantly better.
      The two significant penetration-ratio regressions, `hand_pen_ratio` and `human_pen_ratio`, are
      **n=181**, not 438. All four penetration metrics use 181/438 finite pairs and drop 257; every other
      metric uses all 438 pairs.
    - Model-side count arithmetic is unchanged: D2-AJ remains the **tenth failed model-side intervention**
      (nine loss/representation attempts plus it), and the README's tally remains eleven interventions
      through P6 when D2-AH's pre-diagnosis negative is included. P11 is objective-side by its own
      preregistration: P8/P9/P9b/P9c dose-response is the geometry lineage's one positive (W3 selected),
      P10 formula repair is its first subsequent objective-side negative, and P11 root detach is the
      **second consecutive objective-side negative**. Do not add P11 to the model-side tally.
    - Resolved 2026-08-16, previously open: sealed W3 records
      `sampler.pelvis._target_=priors.diffusion.HOIPriorSampler`, whereas P11 records
      `priors.hoi.diffusion.HOIPriorSampler` after structural refactor `9259d3a`. Numerical
      equivalence across that refactor is now proven on the evaluation path with a residual of exactly
      zero; see conclusion 19 and
      `experiments/results/p1_hoi_p8_w3_eval_replication_s42_20260816.json`. No P11 number moves.
      P11's own compact result is append-only and its `open_caveat` field is left as written.

19. **Structural refactor `9259d3a` is numerically inert on the HOI evaluation path, and so is the
    NVIDIA driver difference between 570.133.20 and 580.126.09 — the residual is exactly zero, not
    merely small.** Re-evaluating the sealed W3 checkpoint
    (`bac2d1add9a164db3c1763427da078cba7759720758604d9d270993e52414761`) at HEAD `5828b35` on both
    hosts reproduces `per_sequence_metrics.json` with sha256
    `bbcd9e1b550d42bf4ac19f9a55db4b9eebb896a8ddb2d562b5226a11b297f6b2`, byte for byte identical to the
    sealed run: `p1-hoi-p8-eval-w3-guided-replication-authority-s42-20260816` (driver 570.133.20) and
    `p1-hoi-p8-eval-w3-guided-replication-worker-s42-20260816` (driver 580.126.09), both RTX 3090 /
    torch 1.13.1+cu117 / CUDA 11.7 / cuDNN 8500. All 438 sequences, all 14 per-sequence numeric
    metrics, all 18 aggregate metrics and all 32 guidance audit scalars are bitwise identical,
    including the integer `guidance_clamp_saturated_elements = 74131`; aggregate `trans_dist`
    **8.437210321426392** and `contact_f1` **0.7788589083944136** agree to the last digit in all
    three. The refactored path genuinely executed — the composed configs differ at line 107
    `_target_`, `priors.diffusion.HOIPriorSampler` → `priors.hoi.diffusion.HOIPriorSampler` — and the
    runs are independent executions rather than copies, with `generation_seconds` 63.35301164817065 /
    62.89035706873983 / 62.112682808889076. Because the residual is zero, no P11 conclusion moves
    against its deltas `trans_dist` +9.089818134580709 and `pelvis_goal_error_cm` +8.253859806184968.
    - **Scope limits, of equal standing with the result. This is not a general guarantee that
      `9259d3a` is numerically inert.** Proven only for: this checkpoint, guided Arm B, 500-step
      diffusion, the `base` architecture variant, drivers 570.133.20 and 580.126.09, and the
      evaluation path. **Not covered:** unguided sampling, every other guidance arm, the
      `sparse_relation` and `interaction_adapter` variants, any future driver version, and the entire
      training path.
    - Commit `9259d3a`'s own message already claimed 52/52 bit-equivalence checks against pre-split
      code at `c77b9d8`, including the full 500-step `sample()` byte-equal with a real HOIPrior. The
      sampling loop was therefore verified at the time; this audit **confirms** that claim end-to-end
      on real data and the full native metric suite rather than overturning an unverified one. What it
      adds is that the claim had never been exercised through the evaluator on the 438-sequence test
      set.
    - **The sealed W3 baseline's evaluation-time code identity, previously unrecorded, is now pinned
      empirically.** That evaluation ran 12:24:25-12:27:43 on 2026-08-09 while the commit holding its
      code, `5e89644`, landed at 12:43:32 the same day — 15 minutes 49 seconds later — so the branch's
      most-referenced baseline was produced from an uncommitted worktree with no recorded code
      identity. `aggregate_metrics.json` recorded no execution environment and no evaluation-time
      commit at all; its only commit was `checkpoint.git_commit`, the *training* commit. From schema
      v2 the file carries an `execution_provenance` block with host, GPU, NVIDIA driver, torch, CUDA,
      cuDNN and the evaluation-time commit with a dirty-worktree flag, so no future question of this
      kind needs a two-host re-execution to answer. The schema change was itself proven inert:
      `p1-hoi-p8-eval-w3-guided-provenance-schema-verify-r2-s42-20260816`, run against the exact
      committed code, reproduces the same `per_sequence_metrics.json` hash and all 18 aggregate
      metrics bitwise. Five runs now share that one hash while reporting five distinct
      `generation_seconds`.
    - `per_sequence_metrics.json` contains no absolute paths, so its file hash is a valid cross-host
      readout; the absolute paths live in `aggregate_metrics.json`. The sealed overrides file has 27
      entries, not 26. See
      `experiments/results/p1_hoi_p8_w3_eval_replication_s42_20260816.json`.
    - This closure authorizes no new mechanism and no new direction. The next entry point is unchanged.

20. **`contact_precision`, `contact_recall` and `contact_f1` carry a hard analytic cap of
    0.906392694063927 on the 438-sequence protocol, so the ground-truth reference row's 1.000 is
    wrong for those three, and every "gap to 1.0" reading of those columns anywhere in this
    repository was overstated by about 10 percentage points.** 41 of the 438 official test sequences
    contain no ground-truth hand-object contact frame at all, and `code/eval_metrics.py:311-323`
    returns `contact_precision = 0` when `TP + FP == 0`, `contact_recall = 0` when `TP + FN == 0`,
    and `contact_f1 = 0` when both are zero — **including for a ground-truth self-comparison**,
    where `FP = FN = 0` and `TP = gt_contact_cnt = 0`. `code/test_infbagel_hoi.py:340` aggregates by
    per-sequence mean, so those 41 enter every model's three contact columns as exact zeros and the
    attainable maximum is 397/438 = **0.906392694063927**. No training or inference intervention can
    move them; the number is a property of the metric definition, not of any model.
    - **`contact_acc` = 1.0 is correct and is NOT corrected.** A zero-contact sequence gives
      `TP = FP = FN = 0` and `TN = 126`, so its accuracy is genuinely 1.0. `contact_percent` for the
      ground-truth row is likewise correct: it equals `gt_contact_percent` = 0.6618830180474017.
      Only three of the four contact columns are affected.
    - **The 41 are not threshold-marginal, so this is a protocol property and not a tolerance
      choice.** The nearest of them has a minimum hand-object distance of **0.07432 m**, i.e. 2.43 cm
      *past* the 0.05 m contact threshold; the farthest is **0.62411 m**; and only **1** of all 438
      sequences sits within 1 cm of the threshold. Re-deriving the channel from the dataset plus the
      rest object meshes with the evaluator's own rule (hand joints 22/23, 0.05 m, the 126-frame
      span) reproduces the count bitwise and returns mean `gt_contact_percent`
      0.6618830180474017 — delta 0.0 against the value that is bitwise identical across 47 sealed
      438-sequence runs. An independent cross-model bound agrees and is tighter than a four-model
      one (≤ 44): intersecting the all-three-exactly-zero sets of all 47 sealed 438-sequence runs
      (24 distinct sets) gives **42**, a superset of the 41 by exactly one sequence,
      `sub17_woodchair_029` — which has a single GT contact frame in 126 at minimum distance
      0.04549 m, i.e. it is the one threshold-marginal sequence in the protocol, and no model in
      the branch's history has ever hit that frame.
    - **Restricted to the 397 sequences where the metric is defined**, a ground-truth round-trip
      reference scores precision **0.9968411711880822**, recall **0.9954630506006943**, f1
      **0.9960322331956971**. The residual is FK round-trip noise, not a modelling gap, so the
      defined-subset floor is the one to quote when a true ceiling is needed.
    - **A coverage fact of the same protocol, measured in the same pass.** The four penetration
      terms are computed on only **181 of 438** sequences, because `code/test_infbagel_hoi.py:276`
      hard-codes exclusion of woodchair, whitechair, largebox, largetable, plasticbox and trashcan.
      **32 of the 41 zero-contact sequences are also penetration-excluded**, so those 32 carry no
      interaction reading of any kind — neither contact nor penetration — while still occupying the
      438 denominator; only 9 zero-contact sequences retain a penetration reading.
    - `experiments/results/p1_hoi_p12_frame_repair_baseline_s42_20260820.json` is **left
      byte-identical** and still carries the wrong 1.000. Its sha256
      `08bae281fa15576d7f5bdc14eb1eaa865b8498f3b5eb5b4013170c16d4c02fab` is pinned by the P12
      completion row and by `docs/phase_summaries/PHASE_1B_P12_REPRESENTATION_FRAME.md:6` and `:240`,
      and sealed compact results are append-only — the same treatment the 2026-08-20 sampling-caliber
      amendment gave the same file (`config.not_rewritten`). This entry plus the dated appended
      sections of that phase summary and of `docs/plan/PHASE_1B_HOI/07_REPRESENTATION_FRAME.md` are
      the correction of record; `/data/yujinlun/report/baseline.md` is outside the repository and was
      corrected in place. Recorded 2026-08-21.

21. **The one clean budget contrast on the distribution-level metric column is a null: a 4.875x
    processed-window increase moves CHOIS FID by -0.311 with a 2000-replicate paired CI of
    [-0.780, +0.058], which contains zero.** D2-AI at 299,520,000 windows against sealed D2-X at
    61,440,000 is the lineage's only single-factor budget pair whose evaluation configs are
    key-for-key identical, and both cells share one ground-truth tree
    (`d439a98ea32f5d67...`, shared by 42 of the 45 complete exports). Classification
    `budget-fid-null-stop`. **Conclusions 7, 10 and 11 are unchanged**, and by the preregistered
    interpretation rule conclusion 10 receives no scope qualification: the
    validation-loss-anticorrelation finding stays stated for the 500-step native geometric and
    contact metric class only, untested at the distribution level.
    - **The preregistration's own ex-ante prediction was right and my post-smoke revision was
      wrong.** The Stage C functional smoke used the three real cells, so the point estimate was
      known before the preregistration was committed (disclosed in the amendment). From four
      replicates all falling below zero I wrote that classification 3 was the likely outcome.
      The 4-replicate interval was about 3.7x narrower than the 2000-replicate one and pointed at
      the wrong branch. Every citation of this result must also cite that disclosure section.
    - **Guidance, not budget, is what moves the distribution.** The informational cell C
      (D2-AI + the sealed P2 Arm B guidance) gives Delta(C-B) = -0.336 with CI [-0.475, -0.236],
      excluding zero -- the first distribution-level measurement of inference guidance in this
      branch, which P2/P3/P5/P6 never made. Delta(C-A) = -0.648, CI [-1.143, -0.284], is exactly
      the sum of the null budget term and the significant guidance term. C's interval is far
      narrower than the PRIMARY's because C and B share a checkpoint and differ only in guidance,
      so their embeddings co-vary across resamples; A and B are different checkpoints.
    - **One rigid vector is worth more than the entire budget effect.** The informational
      offset-corrected cells drop FID from 1.7755 to 0.6332 on A and from 1.4641 to 0.4100 on B by
      subtracting a single global three-number mean bias vector per cell, whose vertical components
      are 2.79 cm and 2.30 cm. That is 1.142 and 1.054 FID for one rigid offset, against a budget
      effect of 0.311 that does not separate from zero. After the removal the residual difference is
      -0.223 with CI [-0.362, -0.105], excluding zero: the rigid offset both dominates the raw
      metric and inflates the estimator's spread enough to mask a real remainder. A' and B' are
      diagnostics, never official scores, and never comparable to the released 0.9334244584430564.
    - **G1 landed on its tier 2 branch.** Recomputing sealed cell A gives 1.7754769074 against the
      sealed 1.7754768927, residual 1.47e-08, and the 200-replicate prefix percentiles differ by
      1.23e-07 / 3.33e-07 -- so bitwise cross-host reproduction fails while the 1e-3 tolerance
      passes, recorded as cross-host environment drift exactly as written before any number
      existed. Conclusion 19's bitwise-reproducibility result covers our own native evaluation
      path and explicitly not this one; that limitation is now measured rather than assumed.
    - G2 (416 embedded / 22 dropped, identical id hashes across all four sets), G3 (126 frames
      throughout, one uniform upstream row permutation), G4 (all four tree hashes plus both
      third-party commits and the feature checkpoint) and G5 (every float finite) all pass.
    - Scope: no cell crosses the 2026-08-19 representation repair, and P12's FID rows use different
      ground-truth trees, so no P12 row may be set against these numbers. FID is a 416-sequence
      quantity. `embedded_sequence_ids` is valid as a set but not as a row label, so no
      per-sequence attribution of FID is permitted. Recorded 2026-08-21.

22. **The four penetration terms have a non-zero ground-truth floor, and it is about half the
    model's value, so every reading of a penetration number in this repository as a distance from
    zero was wrong.** Measured 2026-08-21 by `tools/measure_hoi_repr_ceiling.py` on the official
    438-sequence protocol (`p1-hoi-p12-repr-ceiling-row-s42-20260821`, classification
    `repr-ceiling-row-established`, read-only CPU, 220.7 s). Ground-truth floor: `hand_pen_ratio`
    0.064150, `human_pen_ratio` 0.065202, `hand_pen_loss_omomo` 0.083060,
    `human_pen_loss_infbagel` 1.299681. P12 therefore penetrates **2.0705x / 2.0941x / 2.0709x /
    2.0992x** the achievable floor, not infinitely more than zero. The ground truth penetrates
    because it is itself a SMPL-X fit scored against object SDFs. This closes limitation #4 of the
    P12 phase summary.
    - **The representation contributes essentially nothing to penetration**: the ceiling row
      matches the floor to within 0.3% on all four terms.
    - The same run establishes the full representation-ceiling row. `foot_sliding` at **58.1%** is
      the only metric the representation dominates; `feet_height` is **-0.2%**, i.e. none, which is
      what proves the frame-0 vertical bias belongs to the model rather than to the
      keyframe/interpolation/FK round-trip; `mpjpe` 1.4%, `trans_dist` 1.2%, `obj_trans_dist` 0.8%,
      `obj_rot_dist` 0.6%; the three contact metrics 3.0% / 1.5% / 1.6% **against the 397/438
      analytic cap of conclusion 20, not against 1.0**. CHOIS FID is explicitly out of scope: the
      round-trip FID is 0.0014, three orders of magnitude below the model gap.
    - All seven correctness gates pass and every carried-in anchor reproduces (`gt_foot_sliding`
      delta 2.55e-08, `gt_feet_height` 6.55e-10, `gt_contact_percent` exactly 0.0, 41 zero-contact
      sequences, 181 penetration-covered), so there is no `repr-ceiling-contradicts-exploration`
      event and the 2026-08-20 exploration's numbers stand.
    - **Discount gates A6 and A7**: they are pilot-calibrated, not blind. A 12-sequence pilot showed
      a zero penetration ratio, so A7 was rekeyed onto the penetration loss terms and a zero ratio
      became a warning rather than a failure. Recorded in the plan section. Recorded 2026-08-21.

23. **Teacher forcing PARTIALLY restores height tracking -- about half of window 2's deficit and a
    third of window 3's -- and the per-window contraction it leaves behind is the larger share and
    the main problem.** The 3-window rollout's cross-sequence slope of model pelvis height on
    ground truth decays 1.025 (first generated frame) -> 0.899 (w1) -> 0.505 (w2) -> 0.186 (w3),
    faster than geometric. P14 replaced all five history channels for windows 2 and 3 with that
    window's world-frame ground truth, holding checkpoint, 438 sequences, seed, noise and guidance
    fixed (`p1-hoi-p14-teacher-forcing-s42-20260821`, classification `tf-partial`,
    `experiments/results/p1_hoi_p14_teacher_forcing_s42_20260821.json`). A complete ground-truth
    history recovers rho = 0.515 / 0.320 (cell G, windows 2 / 3) and 0.506 / 0.314 (cell N) of the
    gap to the ground-truth-conditioned window w1 -- **+0.199 to +0.229 slope units, all four
    paired-bootstrap 95% CIs excluding zero**. So compounding of a degraded history is a real,
    measured contributor and rho is its size. The residual is the larger share: **0.485 of the
    window-2 gap and 0.680 of the window-3 gap survive a perfect history.**
    - **What the secondary discriminator adds.** lambda -- where b_TF sits between the model's own
      response-to-its-own-history slope and w1's slope -- is **0.018 to 0.045**, i.e. b_TF lands
      essentially ON the own-history anchor (0.7082 / 0.7077 against an anchor of 0.7041). The
      model's response slope is nearly the same whether the history level is ground truth or its own
      drifted output, so a clean history does not repair the per-window contraction. lambda **bounds
      the exposure-bias mechanism's share; it does not establish that the mechanism is absent**, and
      per the preregistration it cannot overturn the PRIMARY `tf-partial` classification.
    - **The remedy reading is a bound, not an exclusion.** Scheduled sampling or rollout fine-tuning
      may capture the rho fraction, but on this evidence their ceiling is the teacher-forced slope
      itself -- b = 0.708 / 0.414 against b_ref = 0.899 -- so they cannot be the whole remedy and
      must not be the only next training direction.
    - **Two facts that need no E-versus-M partition.** Window 1 already contracts 1.025 -> 0.899
      over its own 42 frames from a ground-truth start with nothing accumulated; and b_TF(w3) = 0.414
      is far below b_TF(w2) = 0.708 although **both** windows consumed a complete ground-truth
      history, so part of the depth dependence is not history quality at all.
    - **The authorized conclusion form is only** whether a COMPLETE ground-truth history restores
      height keeping. It does not. It may NOT be attributed to the root-y channel: all five channels
      were substituted, and the vertical-only arm that could have isolated it was withdrawn after a
      drift injection showed it converts horizontal pelvis drift into object-translation error at
      full magnitude. This does not reopen the D0 root-y repricing abort.
    - **Load-bearing for HOSI.** The LLM state machine chains many more than 3 windows over the same
      `[B,16,232]` contract, and tracking is already at b=0.19 by window 3 -- inside the released
      protocol's own horizon. A complete ground-truth history raises that to 0.41, not to 0.90, so a
      longer chain needs the contraction itself addressed, not just cleaner conditioning.
    - Ex-ante integrity: all four frozen point predictions hit (b_TF(w2) in [0.70,0.75], b_TF(w3) in
      [0.40,0.52]) and so did the verdict prediction. Revision 7 disclosed before the run that pure
      contraction predicts rho(w2)=0.497; measured 0.515.
    - A prerequisite measurement, G4, established that the step-0 (dataset) and step-N
      (`window_codec.encode`) conditioning paths are equivalent over the full protocol -- 876
      comparisons, every channel block <= 5.96e-07 against a 4.768e-07 float32 reference, with the
      joints vertical channel and the frame origin bit-identical -- so a non-recovery implicates the
      model rather than the frame construction. It does NOT cover `obj_bps_data`,
      `object_goal_batch`, `pelvis_goal_batch` or `object_points_batch`.
    - The teacher-forced arms' 18 protocol metrics are **not model scores** and must never enter
      `baseline.md`, this index's headline table, or any model comparison: windows 2 and 3 are
      anchored at the ground-truth location by construction. Recorded 2026-08-21; **interpretation
      amended 2026-08-22 on the user's ruling** -- the original headline recorded exposure bias as
      ruled out, which overstated a secondary discriminator. No measured value changed; the sealed
      compact result and registry row 288 carry the amendment as an appended sibling key.

24. **The P6 cell-U ground-truth-contact-mask upper bound is VOID: its mask was degenerate in 2 of
    its 3 windows, so the inference-time mask direction is NOT closed on that evidence.**
    `gt_contact_label_batch` holds ONE 16-frame window (`code/datasets/infbagel.py:626-629`), while
    `_gt_contact_window` (`code/test_infbagel_hoi.py:371-393`) indexes it as a whole sequence at
    stride 14; at step 2 `start=28` exceeds the 16 available frames and the short-sequence branch
    returns window 0's LAST frame repeated 16 times. cell U consumes the full 16-frame mask, so its
    window 2 was 2/16 frames correct and its window 3 was 0/16. The sealed bound rests on an
    engagement fraction of **0.7891457382039574** (16591/21024) where the correct figure is
    **0.6612442922374430** (13902/21024) -- a **+19.35% relative inflation of the very quantity the
    probe exists to bound**. Cross-checked against 482 annotation files (52470/83559 = 0.62794,
    matching the `code/priors/hoi/diffusion.py` docstring) and against per-window self-read across
    all 1314 windows.
    - **Void, not merely weakened.** The degenerate mask is OVER-BROAD, and an over-broad contact
      mask applies contact guidance on frames ground truth says are not in contact -- a sufficient
      mechanism for damaging precision. cell U's sealed finding was that the GT mask made
      `contact_f1` significantly WORSE (-0.0028307, engagement -0.19% of the gap). Had it found the
      mask helped, over-broadness would only understate the help and the finding would survive a
      fortiori; because it found harm, and over-broadness causes harm, the two are confounded. The
      probe never implemented a perfect engagement decision on 2 of its 3 windows.
    - **What reopens**: the inference-time contact-mask direction, and the attribution of the
      residual ~82.66% engagement gap to a TRAINING-side geometry property. Both need re-measuring.
    - **Unaffected, verified not asserted**: the predicted-mask P5/P6 sub-term sweep. All 9 sibling
      arms use `contact_mask_source=predicted`, and both `gt_contact_label_batch` read points sit
      inside the `_hoi_guidance_uses_ground_truth` gate (`code/test_infbagel_hoi.py:395`).
      `deployable: false` remains correct.
    - Sealed values are preserved unchanged; the disclosure is appended as a sibling key in both the
      compact result and registry row 264. Found by P14's gate G4. P14's teacher-forcing path uses a
      separate per-window accumulator and does not touch the cell-U path. A governance gap worth
      closing: the audit records `contact_mask_source` as `None`, so a sealed run does not say which
      mask it used. Recorded 2026-08-21.

25. **A PERFECT inference-time contact engagement mask buys nothing on the current model, and the
    voided cell-U "significant harm" was the degenerate mask's own signature.** Re-measured
    2026-08-22 on the corrected sequence-length mask (`p1-hoi-p6-cellu-corrected-mask-s42-20260822`,
    classification `cellu-null`,
    `experiments/results/p1_hoi_p6_cellu_corrected_mask_s42_20260822.json`): replacing the author's
    predicted-contact gate with ground-truth engagement changes `contact_f1` by **-0.000401, 95% CI
    [-0.001426, +0.000617]** (438 sequences paired by name, 10,000 replicates, seed 42, one shared
    resample-index matrix), and closes **-0.69%** of the engagement gap (denominator 0.14134). This
    is a **tight** null, not an underpowered one: the effect is bounded to 0.0014 `contact_f1` units,
    2.0x smaller than the voided point estimate -0.0028307 and 100x smaller than the 0.1413 absolute
    engagement gap the probe exists to bound. `contact_precision` and `contact_recall` are null too.
    - **The degeneracy signature, measured.** In the sealed mask the mean number of DISTINCT frames
      per window was 1.703 at step 0, 1.078 at step 1 and exactly **1.000** at step 2 -- one repeated
      row for every sequence -- and it claimed **5616** engaged frames at step 2 where the annotation
      has **2797**, a 2.01x inflation. The corrected mask equals a direct per-window read on all
      **1314** windows with zero mismatches.
    - **So the abort's practical content is restored on valid evidence, with its sign corrected from
      harm to null.** Inference-time mask selection is not the lever, which is consistent with P5's
      flat dose-response. What is NOT re-established is the sealed number itself.
    - **The preregistration was mis-specified, and running it is what found that.** It pinned the
      pre-repair D2-AI checkpoint for comparability with the sealed cell U. Under repaired code that
      checkpoint's own predicted-mask baseline collapses: `contact_f1` 0.675702 -> **0.097631**,
      `mpjpe` 11.7699 -> **59.4331** cm, `feet_height` 0.046125 -> **0.699227**, all 438 sequences
      differing, while `gt_contact_percent` is unchanged at 0.661883 (ratio exactly 1.000), which
      places the change on the model side and not in the data path. Conclusion 23's scope rule at
      this file's line 593 -- "no cell crosses the 2026-08-19 representation repair" -- already
      forbade that pairing. Recorded as a new stop, `cellu-checkpoint-representation-mismatch`, which
      was not among the preregistered stops. **Cell U can therefore never be re-measured in its own
      configuration**; only the practical question about the current model is answerable, and the
      verdict above is that question's answer, taken on the post-repair P12 checkpoint with both arms
      sharing it.
    - **Gates.** All six pass. G2, the user-named scoping gate, passes **bitwise** against the sealed
      post-repair baseline `p1-hoi-p12-guidance-armb-s42-20260820`
      (`d55f6b74bc02e7d798350e3b3a3b6d821cda13b0ad7292a2ffa6c49ce49b582d`), so the repaired source
      provably cannot reach the deployable predicted-mask path. G1 reproduces 13902/21024 =
      0.661244292237443. G6 is the closed governance gap: `GuidanceAudit.as_dict()` emitted neither
      `contact_mask_source` nor `contact_mask_threshold` -- it did not emit them as `None`, as the
      2026-08-21 note said -- and now emits both, with an unbound audit reporting null rather than the
      `predicted` default.
    - Four metrics move significantly and every one is trivial: `end_obj_trans_err` +0.0162 cm on a
      2.761 cm baseline (0.6%, worse), `mpjpe` -0.0106 cm, `trans_dist` -0.0072 cm and
      `human_pen_ratio` -0.00079 (all better). None is a contact metric.
    - **W3's dependency is released**: the geometry-term training run was held only on this bound.
      Both arms remain NON-DEPLOYABLE probes whose 18 protocol metrics must never enter
      `baseline.md`, this index's headline table, or any model comparison. 4 rollouts including the
      2 aborted ones, 6.3 min against a 12 min ceiling. Recorded 2026-08-22.

## 4. Open research question for the next review

The preregistered next entry returns to P10's “objective has attractors but no repulsor”
pointer. A useful proposal must make one falsifiable change while retaining:

- scene-free current-state provenance;
- the clean `[B,16,232]` expert interface;
- random initialization and the fixed formal budget;
- D2-X-class object-goal, FS, penetration and kinematic protection;
- a causal diagnostic fixed before the formal run.

The reviewer may recommend a previously untried training or inference mechanism, but
must identify which old prohibition or protocol it changes and obtain explicit user
approval before implementation. This closure does not itself authorize a repulsor or any
other new mechanism.

## 5. Authoritative files

- Plan and registry: `docs/plan/` (navigation page `docs/EXPERIMENT_PLAN.md`; Phase 1B
  index `docs/plan/PHASE_1B_HOI/README.md`), `experiments/registry.jsonl`
- D2-X/Y/Z/AA/AB/AC/AD/AE/AF summaries:
  `docs/phase_summaries/PHASE_1B_D2*.md`
- Compact results: `experiments/results/p1_hoi_phase1b_*.json`
- 2026-08-01 protocol decomposition:
  `docs/phase_summaries/PHASE_1B_PROTOCOL_DECOMP.md` and
  `experiments/results/p1_hoi_protocol_decomp_s42_20260801.json`
- 2026-08-02 P2 inference contact guidance:
  `docs/phase_summaries/PHASE_1B_P2_GUIDANCE.md` and
  `experiments/results/p1_hoi_p2_inference_contact_guidance_s42_20260801.json`.
  Protocol alignment only: guidance stays default-off, no checkpoint was selected,
  and the 0.83 cm genuine generative-geometry gap is unchanged.
- 2026-08-02 P3 relation field x inference guidance:
  `experiments/results/p1_hoi_p3_relation_field_guidance_s42_20260802.json` and the
  dated closure section of `docs/plan/PHASE_1B_HOI/05_INFERENCE_GUIDANCE.md`. Preregistered
  cost failure; guidance stays default-off, no checkpoint was selected, and the D2-AG and
  D2-AE negative classifications stand.
- 2026-08-15 P11 root-gradient detach:
  `docs/phase_summaries/PHASE_1B_P11_ROOT_DETACH.md` and
  `experiments/results/p1_hoi_p11_root_detach_s42_20260815.json`; full paired bootstrap at
  `results/experiments/p1-hoi-p11-geom-rootdetach-r1-s42-20260814/chain/bootstrap_p1-hoi-p11-geom-rootdetach-r1-eval-guided-s42-20260815.json`.
  Classification `root-coupling-negative-stop`, no checkpoint selected.
- 2026-08-16 W3 evaluation replication and provenance schema:
  `experiments/results/p1_hoi_p8_w3_eval_replication_s42_20260816.json`, plus the run trees
  `results/experiments/p1-hoi-p8-eval-w3-guided-replication-{authority,worker}-s42-20260816` and
  `results/experiments/p1-hoi-p8-eval-w3-guided-provenance-schema-verify{,-r2}-s42-20260816`.
  Classification `refactor-numerical-equivalence-null`, no checkpoint selected, no registry row —
  evaluation runs live under their parent training run's `results` field and this audit has none.
  `aggregate_metrics.json` is at `schema_version` 2 from this commit; `per_sequence_metrics.json`
  stays at 1 by design, because its hash is pinned by several sealed records and the provenance block
  is host-dependent.
- Early targeted diagnostics: `docs/D2H_EXPOSURE_DIAGNOSTIC.md`,
  `docs/D2I_GRADIENT_ROUTING_DIAGNOSTIC.md`,
  `docs/D2J_GRADIENT_CLIP_ROUTING_DIAGNOSTIC.md`,
  `docs/D2K_ADAMW_ROUTING_DIAGNOSTIC.md`,
  `docs/D2L_AUXILIARY_BALANCE_DIAGNOSTIC.md`

Read raw logs and large artifact trees only when a concrete evidence discrepancy
cannot be resolved from these tracked sources.
