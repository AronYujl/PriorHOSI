# Phase 1B HOIPrior evidence index

Status: compact research handoff through D2-AG0, P1 protocol decomposition,
P2 inference guidance, the D2-AH negative preflight, the P3 relation-field
lineage under guidance and the P4 budget-metric curve, 2026-08-02.

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

The 2026-08-01 protocol decomposition (`docs/EXPERIMENT_PLAN.md`, section
"2026-08-01 Phase 1B 基线协议分解 P1") re-evaluated the released checkpoint on the same
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
   exception, and this item previously said otherwise.** P4 (2026-08-02) measured six D2-X
   cadence checkpoints under the frozen protocol: `contact_f1`, `contact_recall`,
   `contact_acc`, `mpjpe`, `xy_points_err` and `obj_rot_dist` are strictly monotone in
   budget, and the final 43.01M->61.44M segment carries the *largest* contact increments of
   the whole curve (`contact_f1` +0.0367, `contact_recall` +0.0568). Nothing has saturated at
   the formal budget. The earlier "longer budget" pessimism rested on D2-V and on the
   held-out loss series, and item 10 explains why the latter was misread.
8. The author's dynamic occupancy offers direct temporal spatial routing but mixes
   scene supervision and train/sample relation sources. Copying it would violate the
   independent scene-free HOIPrior objective.
9. Contact and penetration respond to different levers, and contact levers do not
   stack. P3 measured a significant negative interaction between the sparse relation
   field and inference contact guidance on contact F1/recall/percent, with no
   interaction on any penetration, foot-sliding or end-object term. Any future proposal
   that expects to add a contact mechanism on top of an existing one must state why it
   would not be absorbed the same way.
10. **The held-out denoising validation loss anticorrelates with the native rollout
    metrics and must not gate budget, early-stopping or checkpoint decisions.** All nine
    D2 configs show `total` rising +5.6..+12.4% and the `contact` term +25..+31% after a
    minimum at 21.5-24.6M windows. P4 tested whether that reached metric space and
    falsified it in the opposite direction: `contact_f1` at 21.504M is **0.108 below**
    61.44M, CI [-0.1340, -0.0827], 438 paired sequences. The validation loss is
    single-step denoising; the metric is a 500-step reverse diffusion chained across three
    windows on generated history. See
    `experiments/results/p1_hoi_p4_budget_metric_curve_s42_20260802.json`.
11. **HOIPrior is under-engaged at every budget, so any penetration improvement must be
    read against `contact_percent` first.** It climbs monotonically 0.28358 -> 0.47655
    across the P4 curve against GT 0.66188. Penetration and foot sliding *worsen* with
    budget purely because engagement rises; at 3.07M, `hand_pen_loss_omomo` is 0.13559
    only because `contact_percent` is 0.28358. Same confound as the D2-AH epoch100 row.
12. **`end_obj_trans_err` and `xy_points_err` are goal-recall metrics, not forecasting
    metrics.** The object goal handed to the model *is* GT object translation at
    `end_range-4`, which is the exact frame the metric scores (verified across all 438
    sequences, agreeing to 0.0574 cm — float round-trip only); `pelvis_goal` equals GT
    pelvis at frame 15 to 0.0000 cm while the metric scores frame 14. Both HOIPrior and
    released/e500 receive both goals, so the comparison is commensurable and no "gap vs
    released" number has a goal-leakage problem — but a gap on these two metrics is a
    constraint-satisfaction deficit, not a dynamics-prediction deficit. HOIPrior routes
    both goals through one shared `Linear(12,512)` (`code/priors/models.py:302`) as a
    single conditioning token, of which `goals[3:6]` is never written and pelvis y is
    zeroed; released uses three separate embedding modules. Prose elsewhere that describes
    these two metrics as prediction quality is wrong and should be qualified.

## 4. Open research question for the next review

The next candidate should explain why a relation path that is internally causal under
paired interventions does not improve production rollout. A useful proposal must make
one falsifiable change that targets this train-to-rollout transfer gap while retaining:

- scene-free current-state provenance;
- the clean `[B,16,232]` expert interface;
- random initialization and the fixed formal budget;
- D2-X-class object-goal, FS, penetration and kinematic protection;
- a diagnostic that distinguishes useful relation use from generic residual reliance.

The reviewer may recommend a previously untried training or inference mechanism, but
must identify which old prohibition or protocol it changes and obtain explicit user
approval before implementation. One remaining full-budget experiment does not permit a
sweep or several loosely coupled interventions.

## 5. Authoritative files

- Plan and registry: `docs/EXPERIMENT_PLAN.md`, `experiments/registry.jsonl`
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
  dated closure section of `docs/EXPERIMENT_PLAN.md`. Preregistered cost failure;
  guidance stays default-off, no checkpoint was selected, and the D2-AG and D2-AE
  negative classifications stand.
- Early targeted diagnostics: `docs/D2H_EXPOSURE_DIAGNOSTIC.md`,
  `docs/D2I_GRADIENT_ROUTING_DIAGNOSTIC.md`,
  `docs/D2J_GRADIENT_CLIP_ROUTING_DIAGNOSTIC.md`,
  `docs/D2K_ADAMW_ROUTING_DIAGNOSTIC.md`,
  `docs/D2L_AUXILIARY_BALANCE_DIAGNOSTIC.md`

Read raw logs and large artifact trees only when a concrete evidence discrepancy
cannot be resolved from these tracked sources.
