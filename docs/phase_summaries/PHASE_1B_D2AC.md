# Phase 1B D2-AC：part-aware local-object interaction adapter

## Scope and outcome

D2-AC0 tested one fixed representation/routing hypothesis relative to D2-X: the existing
232-D, 16-frame HOIPrior retained its global BPS token and D2-X FK-foot temporal routing,
while a 349,697-parameter part-aware local-object cross-attention adapter was inserted
between trunk layers 4 and 5. The adapter used 16 deterministic local BPS clusters and
three role queries (`left_hand`, `right_hand`, `object_motion`) with a scalar
`tanh(alpha)` ReZero writeback gate.

The complete approved lifecycle finished: implementation, CPU contract, registered GPU
smoke, one 61.44M-window from-random training, the fixed three-way internal diagnostic,
the fixed official 438-sequence native evaluation, and artifact recovery. The final
classification is:

`interaction-adapter-locality-negative-stop`

The adapter was clearly used: full strongly outperformed gate ablation on both registered
internal quantities. However, the fixed local-correspondence permutation did not
significantly degrade either quantity, so the preregistered locality mechanism gate
failed. Native contact F1/recall point estimates improved only slightly and their paired
confidence intervals included zero; end-object, foot sliding and both penetration
protection checks also failed.

The fixed final-online checkpoint is therefore not selectable. D2-AC1 is ineligible
because it required the exact
`interaction-adapter-positive-but-not-effective-stop` classification. D2-AC1,
checkpoint selection, consistency distillation, HSIPrior, Mixer and any new HOIPrior
search were not started.

## Commits and fixed mechanism

- Plan-only handoff:
  `61a989adab2f3053230bfcd0ebb702601fcdaab2`.
- Model/config/CPU-contract implementation:
  `4debaff90defca921daec0d141a83094da97caf1`.
- Registered smoke implementation:
  `88133ab397e45fcb26b15dc4d39a100878e67450`.
- Evaluator/lifecycle implementation:
  `a32707047014abb2618b0b2c0ca5a23f55bfcc69`.
- CPU completion record:
  `273e6d7e693f6664b3cd9d0c45b31b6b20c58496`.
- Evaluation date binding:
  `655930f0d9b6bb47fbe116c1d779650cfd3dff63`.
- Internal SDF filename-key parity:
  `7481c5ee2465725a857fd961876d8f1b997a0eed`.
- Internal zero-denominator plan/implementation:
  `ef39e62c2d30c9dd0d2575121a7806375d53e23b` /
  `cc931b8b6272e323e25be6cc6c6a6e3a49076558`.
- Native serialized-field parity plan/implementation:
  `376950ea03652306e448bd8c7e7f27362860dd54` /
  `e6ee3fd9611ede9ee8e0cad20b94bd81e9c13366`.
- Completion documentation/registry/result commit: the final commit that adds this
  summary and
  `experiments/results/p1_hoi_phase1b_d2ac_interaction_adapter_s42_20260727.json`;
  identify it with
  `git log --format=%H -- docs/phase_summaries/PHASE_1B_D2AC.md`.

The fixed architecture contract was:

- base/adapter/total parameters:
  `29,673,448 / 349,697 / 30,023,145`;
- parameter increase: `1.1784845%`, below the registered `1.25%` cap;
- BPS SHA-256:
  `fdff7204b4697e105457cb7e39267b9555bc0d8d854dbc92cd67e2d8c3e77042`;
- assignment SHA-256:
  `b62f91f4eb6c4bf2a9211f0187cd1eb97c25394ee45de155f33607959fddeecd`;
- local tokens/features: `16 × 10-D`;
- adapter attention: width 128, four heads, dropout 0;
- writeback placement: after original trunk layer 4 and before layers 5--8;
- scalar `alpha` initialized exactly to zero;
- 232-D API, global BPS token, 4 condition tokens, 16 motion tokens, D2-X loss
  routing, optimizer, split, sampler and evaluator unchanged;
- D2-AB predicted-support objective disabled;
- released, author, D2-V/X/Y/Z/AB, prior, EMA and consistency checkpoints forbidden
  as initialization.

No HSIPrior parameter, storage, architecture, checkpoint schema or forward path was
changed. The future Mixer contract remains clean `[B,16,232]` expert predictions only.

## Authority CPU contract

The original lifecycle
`p1-hoi-d2ac-cpu-contract-s42-20260726` was aborted before workload because its
resolved-config helper derived `/data/yujinlun/code` instead of the authority checkout
root. It created no optimizer, CUDA call, checkpoint load/write or scientific result.
The failure remains immutable.

The registered retry
`p1-hoi-d2ac-cpu-contract-r1-s42-20260726` completed on
`a32707047014abb2618b0b2c0ca5a23f55bfcc69`:

- 329 authority tests passed;
- BPS file, centers, cluster sizes and assignment hashes matched;
- feature shape/dtype/finiteness: `[2,16,10] / float32 / true`;
- exact total/adapter parameters:
  `30,023,145 / 349,697`;
- shared-trunk `eval()`, `alpha=0` base parity max-abs difference: `0.0`;
- initial alpha gradient finite/nonzero;
- test-only `tanh(alpha)=0.1` gradients finite/nonzero for object encoder,
  object identity, part embedding, query projection, Q/K/V/out attention projections
  and writeback;
- fixed locality permutation effect max-abs:
  `0.00024044513702392578`;
- zero/constant/extreme BPS, role separation, dtype/device/batch propagation,
  checkpoint rejection, HSIPrior independence, Mixer clean-output API and static
  forbidden-path scans passed;
- optimizer/CUDA/checkpoint load/checkpoint write:
  `false/0/0/0`.

Metrics/manifest/CPU-log/resolved-config SHA-256:

`b152ff16...01ec` / `151b49d0...ddea` /
`a48090c1...6458` / `eb4381f7...efb4`.

The authority CPU/final-test snapshot is:

`/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-authority-artifacts-s42-20260727`

with 10 files / 162,389 bytes and tree SHA-256
`67a2cd8485f1f2b9cd408d9c9a58075b0b52458ec23b52d9986c3db19f0a69bd`.

## Registered GPU smoke

The original no-update smoke
`p1-hoi-d2ac-gpu-smoke-s42-20260726` was stable and remains preserved. It preceded
the final authority suite closure and its metadata conflated the batch-8 measured
attention tensor with the formal micro-batch-512 estimate, so it was not the final
formal-launch gate. Its tree is 7 files / 56,047 bytes, SHA-256
`98beade0ac72aaa25bcbbec68d50c0106139d09628ea6b5602c19ac0cba40b63`.

The registered final smoke was
`p1-hoi-d2ac-gpu-smoke-r1-s42-20260726`:

- exact Git object:
  `273e6d7e693f6664b3cd9d0c45b31b6b20c58496`;
- host/device: `infbagel-4gpu/node01`, `cuda:0`;
- four RTX 3090 devices visible, no pre-workload compute contention;
- real-data batch 8 and registered timesteps `0/249/499`;
- random initialization; initial model-state SHA-256:
  `dea5461e225b04cb0aae25601048a1383ca7b68bbdce3e02ee4ac918378f273c`;
- losses finite; alpha gradient and all test-only adapter gradient groups
  finite/nonzero;
- measured smoke attention shape/elements:
  `[8,16,3,4,16] / 24,576`;
- registered formal shape/element estimate:
  `[512,16,3,4,16] / 1,572,864`;
- synchronized peak allocated/reserved:
  `252,510,720 / 304,087,040 bytes`;
- memory headroom:
  `24,991,956,992 bytes`;
- optimizer created/updates, checkpoint loads/writes, selection, consistency:
  `false/0/0/0/false/false`.

Smoke manifest/metrics/resolved/preflight/run-local-registry SHA-256:

`15a08b61...2eb` / `d591a617...38e` / `3d18b60c...82d` /
`f3d6629f...d58` / `e62348e7...665`.

The complete tree is 9 files / 70,470 bytes, SHA-256
`72b39d1d1e9b2cbb1197f141fddee50fac48e477e784ab373e4ef82c41983b1f`.

## Formal from-random training

Run:
`p1-hoi-d2ac-interaction-adapter-s42-20260726`.

- exact worker Git object:
  `273e6d7e693f6664b3cd9d0c45b31b6b20c58496`;
- random initialization, no source checkpoint;
- initial model-state SHA-256:
  `260d12ed7e60774e9eb4280f3abc580a7af8b9d8a36d3c75b83a9bbf0021a1bc`;
- fixed split SHA-256:
  `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`;
- `61,440,000` processed windows / `983,040,000` frames /
  `30,000` optimizer updates;
- 4×RTX 3090, per-GPU batch 512, effective batch 2,048, accumulation 1;
- FP32 Adam, LR `1e-4`, betas `(0.9,0.999)`, no warmup/scheduler/
  weight decay/gradient clipping/AMP/EMA;
- loss and required gradients finite; AMP overflow skips 0;
- mean training total loss:
  `0.0448770182`;
- final validation total/reconstruction/FK/object-surface/velocity:
  `0.0482425985 / 0.0402186446 / 0.0106323917 / 0.0084568356 /
  0.000110568023`;
- learned alpha/gate:
  `0.0907876045 / 0.0905389935`;
- wall time:
  `19,157.121 s`;
- throughput:
  `3,207.162 windows/s` / `51,314.599 frames/s`;
- minimum per-rank memory headroom:
  `21,074,477,056 bytes`;
- 20 cadence checkpoints and 80 per-rank RNG sidecars;
- first resumable checkpoint SHA-256:
  `0f7d56a0e2c6e3a8cd6e394b00fcb002f10474a1804196da27f3bfceee609809`;
- fixed final-online checkpoint SHA-256:
  `fede1c2b2f331407ceba7db16e3a4b30ccc6ffb6c8fc252861662bdcc96c7b96`.

Manifest/metrics/resolved/preflight/run-local-registry SHA-256:

`a0079817...d775` / `ee310732...5f4f` / `3ffbf770...20b` /
`379dc3f0...03a` / `37297ea2...a54`.

The recovered tree is 115 files / 7,211,816,400 bytes, SHA-256
`d3784f0b01b8762ab1e6dcc7b0343ef2aa2147c1ca9672f516ae2f672cd92d98`.

## Fixed internal causal diagnostic

Final run:
`p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727`.

The run loaded only the fixed final-online checkpoint and used the sealed D2-O
64-sequence / 192-window cohort, phase offsets `(14,56,98)`, selection SHA-256
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`.
Full, gate-ablated and local-correspondence-permuted 500-step rollouts shared the
initial latent, every posterior-noise draw, conditions, ordering and history
restoration. Statistics used sequence units, seed 42 and 10,000 paired bootstrap
replicates. No official-test sequence, optimizer/update, checkpoint write/selection
or consistency path was used.

Primary registered comparisons:

| comparison | paired mean | sequence-bootstrap 95% CI | gate |
|---|---:|---:|---|
| full − gate-ablated direct-hand union 5-cm F1 | 0.6215448 | [0.5397641, 0.7003120] | pass |
| full − locality-permuted direct-hand union 5-cm F1 | 0.0103921 | [-0.0177716, 0.0375935] | fail |
| gate-ablated − full GT-contact distance (cm) | 90.9780050 | [81.0602569, 100.8264305] | pass |
| locality-permuted − full GT-contact distance (cm) | 0.0138198 | [-0.3039546, 0.3092714] | fail |

Full/gate-ablated/permuted direct-hand union 5-cm F1 point estimates were
`0.7679229 / 0.0370542 / 0.7565714` in aggregate-frame reporting and
`0.6504507 / 0.0289058 / 0.6400586` under the registered sequence-level paired
gate reduction. The distinction is retained explicitly; the sequence-level values
alone determine the causal gate.

Representative aggregate diagnostics:

| variant | semantic union F1 | direct 5-cm union F1 | FK-palm 5-cm union F1 | MPJPE cm | object goal cm | pelvis goal cm | FS |
|---|---:|---:|---:|---:|---:|---:|---:|
| full | 0.95796 | 0.76792 | 0.75466 | 12.0097 | 95.2952 | 4.3278 | 0.81160 |
| gate ablated | 0.88826 | 0.03705 | 0.00100 | 208.0364 | 335.0759 | 63.5029 | 0.99891 |
| locality permuted | 0.95718 | 0.75657 | 0.74170 | 12.4420 | 94.9122 | 4.5503 | 0.82031 |

The full artifact reports left/right/union semantic contact P/R/F1/coverage,
direct-hand indices `24/26` and FK-palm indices `22/23` at 2/5/7.5/10 cm,
contact-run lengths, GT-contact-frame distance, penetration, MPJPE, object/pelvis
goals, FS and per-role attention entropy. Full normalized attention entropy for
left/right/object-motion was
`0.488900 / 0.489855 / 0.488040`; attention remained descriptive only.

### Legal zero-denominator repair

The r1 internal run completed all three raw variants but failed while trying to
compute:

`full_mean / gate_ablated_mean`

for descriptive hand penetration, because the gate-ablated mean was exactly zero.
Zero penetration is valid and lower-is-better; only the ratio is mathematically
undefined. The scoped repair in `cc931b8...`:

- rejects mismatched, empty, non-finite or negative vectors;
- uses the unchanged paired-ratio function whenever denominator mean is positive;
- for a zero denominator records
  `ratio_defined=false`, `mean_ratio=null`, `bootstrap_95_ci=null`,
  `undefined_reason=zero_denominator_mean`;
- adds no epsilon, pseudocount, clamp or infinity encoding;
- retains the exact paired values and reports the existing paired-difference
  statistic;
- is called only by the D2-AC internal penetration summary, not by the native
  evaluator.

The final r2 hand-penetration comparison records numerator/denominator means
`9.308420447e-07 / 0`, undefined ratio, and paired difference
`9.308420447e-07`, CI `[0, 2.792526134e-06]`. Penetration is descriptive and does
not participate in the primary internal gate.

The adapter-use checks passed, but both locality checks failed. The internal
classification is therefore
`interaction-adapter-locality-negative-stop`.

Internal metrics/manifest/resolved/preflight/run-local-registry SHA-256:

`4c46b366...1109` / `0f738c87...c357` / `65d69c05...90f` /
`62dc7908...4ff` / `e69172cb...66a`.

Full/gate-ablated/permuted raw SHA-256:

`f47ba461...3bcb` / `90267554...83c` / `6245dc6d...ff0`.

Paired-noise/attention-appendix SHA-256:

`d0850bb7...998` / `25392bf8...a99`.

The recovered tree is 11 files / 26,236,078 bytes, SHA-256
`62225323d8a5d3d252d34587165bd2da0ade4ed469ddae1c644e848cd391e753`.

## Native evaluator repair and parity

The first native lifecycle
`p1-hoi-d2ac-native-eval-s42-20260727` completed the official 438×3 target
generation, but the D2-AC post-evaluator paired summary requested nonexistent
serialized aliases for Troot/Tobj/Oobj and penetration. The raw official records
contained the valid short keys and finite values. This was a D2-AC wrapper
field-routing defect after official evaluation, not an evaluator metric, checkpoint,
mask or sampler defect.

The final patch changed exactly five D2-AC wrapper mappings:

- `trans_dist -> trans_dist`;
- `obj_trans_dist -> obj_trans_dist`;
- `obj_rot_dist -> obj_rot_dist`;
- `hand_pen_loss_omomo -> hand_pen_loss_omomo`;
- `human_pen_loss_infbagel -> human_pen_loss_infbagel`.

It did not change metric formulas, aggregate reduction, finite handling, penetration
mask membership, bootstrap functions, thresholds, gates or classification precedence.
The fixed r1 lifecycle reran the official target evaluator from scratch and did not
reuse the failed attempt's aggregate, per-sequence or partial output.

Locked source hashes remained:

- official `code/test_infbagel_hoi.py`:
  `22886f8797ceb04a892487393dea9f80e19877bc02dd7a6f39127e7319119524`;
- `code/eval_metrics.py`:
  `445e681fb618e5f4c89b407a89f152e539a8819f4e8ec1588ae83f6cb062c547`;
- fixed eval config:
  `89c702d96b98289924225c4b163d3b29eb22efe27c50ac799ddd0c71c515aa73`;
- shared D2-X wrapper:
  `b6753a66207492e6ee4addb8f450cb38c5d021401d43430faa9e5c9ed77c6e31`.

The final D2-AC wrapper SHA-256 is
`04b49c17602d13da2f45f2ae47dba191c4a21a5e914ada560994cdde3c0c827c`.

A structural comparison of sealed D2-X and D2-AC resolved target configs found no
semantic difference after excluding only run/output/checkpoint identity fields.
Both used the same official 438 sequences, three windows/sequence, 500-step
unguided production sampler, data, normalization, history, batch 1, CUDA device and
all disabled CFG/dynamic-perception/guidance/scene/consistency settings.

The D2-X control was reused without regeneration with locked checkpoint/aggregate/
per-sequence hashes:

`b0fa6bdd...3d51` / `3bfe1b62...42b` / `69cc811c...b4a`.

The two independent from-scratch target evaluator executions have very small normal
GPU numerical differences. They share the same checkpoint, seed, protocol, evaluator
hashes, data, normalization and mask; no bitwise-identical-rerun claim is made.

## Fixed native evaluation

Final run:
`p1-hoi-d2ac-native-eval-r1-s42-20260727`.

- official 438 sequences × 3 windows;
- fixed final-online checkpoint;
- 500-step unguided production diffusion;
- CFG, dynamic perception, guidance, scene conditioning and consistency off;
- paired sequence unit, seed 42, 10,000 bootstrap replicates;
- penetration finite mask:
  181/438 sequences, ID SHA-256
  `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`;
- normalization audit:
  zero non-finite and zero out-of-range object/position values.

D2-AC target point estimates:

| Te | Txy | FS | Cprec | Crec | Cf1 | C% | Pbody | Phand | MPJPE | Troot | Tobj | Oobj |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5.6473 | 4.2379 | 0.3986 | 0.7876 | 0.6042 | 0.6480 | 0.4913 | 4.0121 | 0.2518 | 12.4268 | 8.7229 | 16.8110 | 1.0306 |

Native transfer:

- D2-AC minus D2-X contact F1:
  `+0.0105639`, CI `[-0.0088320, 0.0303036]` — fail;
- D2-AC minus D2-X contact recall:
  `+0.0097080`, CI `[-0.0124421, 0.0322983]` — fail;
- D2-AC minus D2-X contact precision:
  `-0.0004817`, CI `[-0.0191720, 0.0185576]` — protection pass;
- released contact-F1 gap closure:
  `0.1175963 < 0.25` — fail.

Protection ratios use the registered paired per-sequence means, not the official
aggregate point-estimate reduction:

| target/control ratio | paired 95% CI | `upper <= 1.10` |
|---|---:|---|
| end-object | [1.42382, 1.58993] | fail |
| Txy | [1.00592, 1.08852] | pass |
| FS | [1.03049, 1.17165] | fail |
| Pbody | [0.90233, 1.19812] | fail |
| hand penetration | [0.89232, 1.18712] | fail |
| MPJPE | [1.01158, 1.05130] | pass |
| Troot | [1.04499, 1.09026] | pass |
| Tobj | [1.02260, 1.08024] | pass |
| Oobj | [0.97003, 1.03063] | pass |

The released-baseline 95% effectiveness checks passed only for MPJPE, Oobj and
contact precision among the registered monotone metrics; the overall effectiveness
gate failed. Contact coverage was reported but not used as a separate monotone
selection metric.

Runtime was `353.847 s`; synchronized generation/end-to-end time was
`64.499 / 347.193 s` for 55,188 frames, or `855.645 FPS`. The evaluator did not
generate FID, Matching, R-Precision or Diversity. Their absence is retained honestly,
`fid_rprecision_used=false`, and no value was deleted, substituted or used for
selection.

Native manifest/metrics/aggregate/per-sequence/resolved/preflight/run-local-registry
SHA-256:

`31cbbbc5...159b` / `16d1ec9e...cc81` / `3c996f0b...438c` /
`7cc9ab76...b46` / `5de930fe...104` / `2ccdc755...fca` /
`7be4278c...1e1`.

The recovered tree is 16 files / 1,373,268 bytes, SHA-256
`83b6a811eab7e519f5f15ce2cfeb36d12bb8814625905ac7f2378caeb8fefa34`.

## Preserved operational failures

All operational and scientific failures remain retained:

1. CPU initial path-resolution abort, before CUDA/optimizer/checkpoint workload.
2. Original stable smoke retained but superseded as the final launch gate by the
   registered r1 smoke.
3. Internal initial SDF key failure:
   `floorlamp.ply.npy` was keyed as `floorlamp.ply`; tree
   `e9eaea94...1d7ce`.
4. Internal r1 legal zero-denominator summary failure after all raw variants completed;
   tree `2f7c0138...9dd8`.
5. Native initial post-evaluator serialized-field routing failure; tree
   `ae7f0e3d...e1b27`.
6. A control-only tmux quoting invocation for native r1 created no pane, process,
   log, metric, evaluation output or GPU allocation. The first actual workload then
   completed under the same still-unused manifest. The event is retained in
   `operational_launch.log` SHA-256
   `5deef8d76f2e39cbd6d6ef262bd4468df400db3c323c7923afb8299d3a593fec`.

No event caused result overwrite, partial-output promotion, checkpoint selection,
training restart, run-id reuse after workload, evaluator omission or suppressed
negative evidence.

## Verification

Authority and worker both ended on branch `phase/01b-hoi`, exact source object
`e6ee3fd9611ede9ee8e0cad20b94bd81e9c13366`, with clean worktrees before completion
documentation. Project commands used the verified authority/worker Python 3.8.20
interpreters.

Final implementation verification before workload publication recorded:

- authority targeted D2-AC tests: 25 passed;
- authority full suite: 335 tests, all passed;
- worker full suite: 335 tests, all role-applicable tests passed with the two
  approved real-LINGO skips;
- final worker-suite log SHA-256:
  `d589f01d59af425945dc1edaaf10f6ffca1e69a5df27eb3a2adabd9dfd0e9cb3`;
- authority full-suite log SHA-256:
  `528c707aa2413a23e007dc92580c954cf4832cf5dba7dc0d5bd452ae49264619`;
- registry validation, `py_compile` and `git diff --check` passed.

Completion documentation reran:

- `"$INFBAGEL_PYTHON" -m unittest -v tests.test_hoi_d2ac`;
- `"$INFBAGEL_PYTHON" -m unittest -v tests.test_research_governance`;
- `"$INFBAGEL_PYTHON" -m unittest discover -s tests -v`;
- `"$INFBAGEL_PYTHON" tools/experiment.py validate`;
- compact-result JSON parse, registry JSONL parse and `git diff --check`;
- deterministic `sha256_path` verification of all recovered CPU/smoke/training/
  internal/native/failure trees;
- a resolved-target semantic comparison against sealed D2-X after excluding only
  run/output/checkpoint identity fields.

The worker preflight-log snapshot is 4 files / 67,558 bytes, tree SHA-256
`0d8c12f2530ae785bb8cce4b5fc4ff9a84fa33ff905136056f426d670dffff6b`.

## Tracked and external artifacts

- Compact tracked result:
  `experiments/results/p1_hoi_phase1b_d2ac_interaction_adapter_s42_20260727.json`.
- Authority CPU/final-test snapshot:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-authority-artifacts-s42-20260727`.
- Smoke r1:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-gpu-smoke-r1-s42-20260726`.
- Training:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-interaction-adapter-s42-20260726`.
- Internal r2:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-interaction-adapter-internal-r2-s42-20260727`.
- Native r1:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-native-eval-r1-s42-20260727`.
- Preserved internal failures:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-interaction-adapter-internal-s42-20260727`
  and
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-interaction-adapter-internal-r1-s42-20260727`.
- Preserved native failure:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ac-native-eval-s42-20260727`.

No merge commit or immutable result tag was created because the D2-AC scientific
gate failed.

## Exact next entry point

D2-AC is closed as a controlled locality-negative result. The checkpoint is
non-selectable, must not initialize another prior, and must not be resumed.
D2-AC1 is ineligible and remains unauthorized.

Any future action must start in a new session by reading this summary and
`docs/EXPERIMENT_PLAN.md`, executing the real date/path/branch/HEAD/status preflight,
adding a new dated plan and append-only registry hypothesis, and obtaining explicit
user authorization. It must not silently turn D2-AC1 into a fallback, perform an
adapter/token/width/depth/placement/role sweep, add a new loss or weighting method,
select a checkpoint, start consistency, HSIPrior or Mixer, or reopen unrestricted
HOIPrior search.
