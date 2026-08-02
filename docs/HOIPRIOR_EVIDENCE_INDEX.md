# Phase 1B HOIPrior evidence index

Status: compact research handoff through D2-AF0, 2026-07-30.

Use this file as the first research-context entry point. It summarizes conclusions;
the named phase summaries and compact JSON files remain authoritative for exact
protocols, confidence intervals and hashes.

## 1. Locked comparison points

All D2-* values are from the unchanged official 438-sequence, three-window, 500-step
unguided native protocol, except the P2 row, which is the same sealed D2-X checkpoint
under the same protocol with inference-time contact guidance added and is marked as
such. The released InfBaGel row is measured on the same 438
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
7. Width, depth, token count, point count, adapter placement, LR, batch and longer
   budget sweeps have no positive evidence and are poor uses of the remaining budget.
8. The author's dynamic occupancy offers direct temporal spatial routing but mixes
   scene supervision and train/sample relation sources. Copying it would violate the
   independent scene-free HOIPrior objective.

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
- Early targeted diagnostics: `docs/D2H_EXPOSURE_DIAGNOSTIC.md`,
  `docs/D2I_GRADIENT_ROUTING_DIAGNOSTIC.md`,
  `docs/D2J_GRADIENT_CLIP_ROUTING_DIAGNOSTIC.md`,
  `docs/D2K_ADAMW_ROUTING_DIAGNOSTIC.md`,
  `docs/D2L_AUXILIARY_BALANCE_DIAGNOSTIC.md`

Read raw logs and large artifact trees only when a concrete evidence discrepancy
cannot be resolved from these tracked sources.
