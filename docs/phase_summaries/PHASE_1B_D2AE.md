# Phase 1B D2-AE：GPU-native sparse current-state role-relative object-field routing

## Scope and final outcome

D2-AE0 tested one fixed alternative to the sealed D2-AC/D2-AD interaction adapters.
It did not add more adapter tokens, copy the author's scene occupancy path, use clean
future state, or query a full mesh on CPU. At every diffusion step it rebuilt a
100-point object surface from the current noisy/generated state, formed explicit
left-hand/right-hand/pelvis relative point features, mean/max pooled the three roles,
and wrote four fixed relation vectors into the D2-X motion stream before all eight
trunk layers.

The complete authorized lifecycle is closed: identifier/source/provenance audit,
plan-only registration, implementation, authority CPU contracts, registered
functional smoke, registered four-GPU full-micro-batch performance gate, one complete
61.44M-window training from random initialization, the fixed four-path internal
diagnostic, the fixed official 438-sequence native evaluation, artifact recovery,
hash verification and compact reporting.

The final preregistered classification is:

`sparse-relation-field-transfer-negative-stop`

The learned relation path was genuinely used, its fixed temporal correspondence was
causal, and its left/right role binding was structural. The implementation also met
the performance gate and trained slightly faster than sealed D2-X. However, the
official native contact F1 and recall gains over D2-X were small and statistically
uncertain, the contact-F1 point estimate missed the registered minimum, and released
gap closure was only 5.03%. End-object and foot-sliding protection checks also
failed. The fixed final-online checkpoint is not selectable. D2-AE1, a longer budget,
consistency, checkpoint selection, sweeps, HSIPrior and Mixer were not started.

## Commits and fixed mechanism

- D2-AE0 plan-only registration:
  `eded185f7e5ba075ba83fde97282cb1464ddb08f`.
- Source/config/tests/lifecycle implementation:
  `20d989b59cc894ce48cd65b33abb207a7399d099`.
- Authority CPU completion record:
  `eda6db650343f971215cc8df0401cdf6c07e90fe`.
- Functional smoke record:
  `17c56505f3257adc80ccd9a7af46c2f5fbe42c43`.
- Performance gate record:
  `5ff7a4d98c50db0c10d0c6fe51fb85574b6d40be`.
- Formal preflight retry binding:
  `f61d0f29de6702bcc99120ddb2218c342c60b558`.
- Formal training record:
  `993934cb1d27a2fb406b4d3640eda90d8737767a`.
- Evaluation date-transition record:
  `190d95d1c634299407b398946b2a01d5737b45d7`.
- Fixed internal diagnostic record:
  `5a167347ec4761ec8427b518a36da9157b8fe033`.
- Completion documentation/result/registry commit: identify the commit that adds
  this file with
  `git log --format=%H -- docs/phase_summaries/PHASE_1B_D2AE.md`.

The exact module contract was:

- base/relation/total parameters:
  `29,673,448 / 413,953 / 30,087,401`;
- parameter increase `1.3950283%`, below the fixed `1.50%` limit;
- current diffusion state `x_t [B,16,232]` only;
- immutable rest-object points `[B,100,3]`, with no full-mesh runtime query;
- temporal anchors `(0,5,10,15)`;
- structurally ordered roles
  `(left joint 24, right joint 26, pelvis joint 0)`;
- shared `4→128→128` SiLU point encoder;
- mean/max set pooling, three-role concatenation, `768→512` projection, four learned
  temporal embeddings and LayerNorm;
- one scalar `tanh(alpha)` gate initialized exactly to zero;
- segment writeback `0..4→0`, `5..9→5`, `10..14→10`, `15→15`, before condition
  concatenation, position embeddings and all eight trunk layers;
- global BPS, 20-token sequence, 232-D clean output, D2-X FK-foot routing, losses,
  optimizer, sampler and evaluator unchanged.

Sparse asset SHA-256:

- mapping:
  `1af35119c1dd54e2ad44c99f3cb91b62c1b88f62ca80cddcc96f4b201ffe0f5b`;
- manifest:
  `e88d74a7ee434f3e6320c95d1ebb74efdc8fe4740b70ff596e502666a096f7a7`;
- stacked `[13,100,3]` tensor:
  `793dad6a805d0a908087b273590bf171e7bce4c026297cf94d40f8c651fe4cab`.

No Scene asset, scene label, static occupancy, clean `x_start`, future GT, previous
predicted clean state, stored relation, contact label, category embedding, KD-tree,
SciPy, dense voxel, full-mesh `cdist`, new loss, HSIPrior parameter or Mixer input was
introduced.

## Authority CPU contract

Authority run `p1-hoi-d2ae-cpu-contract-s42-20260728` completed as
`cpu-contract-passed`.

- 26 targeted D2-AE tests and the 378-test authority suite passed at implementation
  closure.
- Exact base/relation/total parameter counts and `[B,16,232]` output passed.
- Shared D2-X trunk `eval()`, zero gate parity had max-abs difference `0.0`.
- Object-surface construction had exact parity with the existing training loss.
- Common global-yaw relation/surface max-abs differences were
  `7.15e-7 / 9.54e-7`.
- Left/right pooled block exchange was exact; point-order permutation remained
  invariant within `9.54e-7`.
- Relative rotation and translation changed the relation by `0.38050 / 0.19566`;
  temporal permutation changed the routed output by `0.29134`.
- Initial alpha gradient and all test-only activated point/projection/temporal/trunk
  gradients were finite and nonzero.
- Train/sampler relation surface and feature parity were exact.
- D2-X, D2-AC, D2-AD and released checkpoint variants were rejected.
- SO(3), zero/constant/extreme state, dtype/device/batch, HSIPrior storage,
  Mixer clean-output and static forbidden-source contracts passed.

Metrics/resolved/tree SHA-256:

`b1bfd61e...d605` / `b8708528...8f14` /
`662cf1fa37121d24b660334fa22c5fec1d5114e980271d6e1df58aa67973fae5`.

The recovered tree has 3 files / 81,347 bytes.

## Functional smoke and performance gate

The first functional identity
`p1-hoi-d2ae-gpu-functional-smoke-s42-20260728` stopped before workload because
its preflight inherited `CUDA_VISIBLE_DEVICES=0` and saw one rather than four visible
devices. The 4-file / 57,187-byte failure tree remains unmodified with SHA-256
`d2bd049d7688c8f5493c0698066f79dcfceeb90f8ff34530da4b4035db4170b5`.

Retry `p1-hoi-d2ae-gpu-functional-smoke-r1-s42-20260728` ran real-data batch 8 at
timesteps `0/249/499` on worker GPU 0. Random-init losses, alpha gradient and the
test-only activated relation/trunk gradients were finite. Peak allocated/reserved
memory and headroom were `281,199,616 / 322,961,408 / 24,973,082,624` bytes. It
created no optimizer, update, checkpoint load or checkpoint write. Its 5-file /
58,333-byte tree matches worker and authority:
`c2eed8eef78c720db46fd4064d78bad07fb85f1462e25d113a99a69cea474259`.

The original full-micro-batch performance run passed at
`3,199.622 windows/s`, but it was bound to the formal identity that later stopped in
preflight. The unchanged sacrificial retry
`p1-hoi-d2ae-performance-benchmark-r1-s42-20260728` was therefore rerun and bound
to formal r1. It used 4×RTX 3090, per-GPU batch 512, 64 warm-up plus 256 measured
updates and 524,288 measured windows.

- Throughput: `3,500.338 windows/s`.
- Fraction of sealed D2-X: `1.07934`.
- Full-budget ETA: `4.87572 h`.
- Required throughput / ETA: `≥2,756.580 windows/s / ≤6.20 h`.
- Minimum headroom: `18,989,383,680` bytes, versus required
  `2,529,604,403`.
- Mean relation geometry/module time across ranks over the measured interval:
  `0.5383 / 1.9570 s`.
- Mean loader/H2D/forward/backward/optimizer times:
  `35.6687 / 0.3882 / 15.9231 / 89.9209 / 3.0833 s`.
- All rank losses/gradients and fixed relation shapes were finite; relation building
  was CUDA-only, CPU dynamic geometry and external contention were absent, and
  checkpoint loads/writes were zero.

The 10-file / 1,242,098-byte recovered benchmark tree matches both hosts with
SHA-256
`b7042d965a8483afd8b1306e7a81d2a30d067f54f1094dfc8910d88fcb4882c7`.
The benchmark summary SHA-256 is
`975a9b6dca4d1a4613af604dfca6420be7e056ea055566d6b3573397e3d914d9`.

## Formal from-random training

The first formal identity
`p1-hoi-d2ae-sparse-relation-field-s42-20260728` stopped before GPU workload:
the registered worker preflight and resolved Hydra config had passed, but an
auxiliary inspection used invalid import path `code.train_hoi_prior`. The 4-file /
22,853-byte failure tree remains preserved with SHA-256
`620c4cd5d6361036d15e0adac58a40adb503e7196a2946c7c25ebc4cd43c0136`.

Retry run:
`p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728`.

- worker source commit:
  `f61d0f29de6702bcc99120ddb2218c342c60b558`;
- random initialization; no source, weight-init, resume, released, prior, EMA,
  scheduler, scaler or old RNG state;
- initial model-state SHA-256:
  `b549358a847205ca7cf6376fd5125a60f87295c455a95fb72d245a4249b7bc8c`;
- terminal model-state SHA-256:
  `2f80498d3fea1adcbea244ee58dbbe466dd6df631bb7ba4a738e1275cac87e61`;
- split SHA-256:
  `019b01ddd6d98cf1e22f1a5a87051d43908e76886d4682c105271c7c91fcac9e`;
- 4×RTX 3090, per-GPU batch 512, effective batch 2,048, accumulation 1;
- exactly `61,440,000` windows / `983,040,000` frames / `30,000` FP32 Adam
  updates;
- wall time `18,356.507 s`, throughput `3,347.042 windows/s` /
  `53,552.671 frames/s`;
- mean training total loss `0.0434231`;
- final validation total/reconstruction/FK/object-surface/velocity:
  `0.0486456 / 0.0409278 / 0.0101676 / 0.00825529 / 0.000102699`;
- learned alpha/gate:
  `-0.14936647 / -0.14826550`;
- 20 cadence checkpoints and 80 rank RNG sidecars;
- first resumable checkpoint SHA-256:
  `dd3540aa01931354dd52ed79410c456cafa3823b55350549c0354d7eadfd869c`;
- fixed final-online checkpoint SHA-256:
  `b7d49046504e9f8367bfd2bce0aeefb1c8590bf9c542b6eed637f05bdfcdd840`.

Metrics/manifest/resolved/preflight/training-state/gradient-audit SHA-256:

`f8b9ca00...8590` / `da09e26e...8d0d` / `1b113e81...150` /
`6f02903d...958` / `ea42a2fc...df69` / `83044e8b...9068`.

The complete recovered tree has 119 files / 7,226,999,632 bytes and matching
worker/authority SHA-256
`3c8a987d54dfb63e89d7ec243fb065dc4f84c95808d92eee13b46ab621959428`.
An earlier 118-file snapshot is also retained without overwrite at SHA-256
`420e2f89d8059e4d9b5d0249001fbb9dbaffd5e591990f8ba7d6fbcdf6e44ae6`.

## Fixed internal causal diagnostic

The intended internal identity and retry r1 each stopped before manifest/workload
because instantaneous idle preflight observed one GPU in P5 despite zero utilization
and no compute process. Both two-file trees are preserved:

- base: 10,311 bytes,
  `015e180d5aa21f093fe7f712d576150f12d47203aac26269f28f56c0015336e3`;
- r1: 10,320 bytes,
  `88f20c8ba3f0c013ba475e04551706ce2194c1904d33db2738dde497175de8bd`.

Successful run
`p1-hoi-d2ae-sparse-relation-field-internal-r2-s42-20260729` loaded only the
fixed final-online checkpoint and used the sealed D2-O 64-sequence/192-window cohort,
selection SHA-256
`1db59afabe7983e6cf370cb609597e14134a487e01135aa466bbdd477e7b4b6a`.
The full, gate-ablated, temporal-permuted and left/right-swapped 500-step paths
shared initial latent, every posterior-noise draw, exogenous condition, window
ordering and history restoration. Statistics used sequence units, seed 42 and
10,000 paired bootstraps.

All five mechanism gates passed:

| comparison | paired point | sequence-bootstrap 95% CI |
|---|---:|---:|
| full − gate-ablated direct-hand union 5-cm F1 | +0.236691 | [0.148411, 0.326983] |
| full − temporal-permuted direct-hand union 5-cm F1 | +0.153893 | [0.081493, 0.226123] |
| full − left/right-swapped direct-hand macro-F1 | +0.178708 | [0.122784, 0.232256] |
| gate-ablated − full GT-contact-frame distance (cm) | +3.509101 | [2.090270, 4.957889] |
| temporal-permuted − full GT-contact-frame distance (cm) | +4.010482 | [2.072867, 6.222641] |

The internal classification was
`sparse-relation-field-internal-positive-continue`. Whole-path dependence alone
was not used as proof of added value; the temporal and role perturbations supplied
the causal correspondence evidence. Full descriptive direct-union 5-cm F1 was
`0.778771`; internal MPJPE/object goal/pelvis goal/FS were
`11.9857 cm / 94.1255 cm / 5.23849 cm / 0.799586`.

The 17-file / 37,798,242-byte tree matches worker and authority with SHA-256
`044f98f78d52347af0c3120a1a5ca4df25c5e4773256c89c2fd5e6bd77fd0b21`.
Metrics/manifest/paired-noise/paired-conditioning/relation-appendix hashes are:

`0d1e4223...1ba9` / `811cb1be...80b2` / `1f412394...988c` /
`1eaa2380...4b9d` / `a38693f5...0815`.

## Fixed official native evaluation

Run:
`p1-hoi-d2ae-native-eval-s42-20260729`.

The unchanged official evaluator ran 438 sequences × 3 windows with 500 unguided
steps, final-online weights, seed 42 and 10,000 paired sequence bootstraps. CFG,
dynamic perception, guidance, scene conditioning and consistency were off. The
sealed D2-X aggregate/per-sequence artifacts were reused without regeneration.
D2-AC and D2-AD were sealed descriptive evidence only and never entered selection.
Runtime was `383.201 s`; synchronized generation was `71.086 s` for 55,188 frames.

Target and sealed D2-X point estimates:

| metric | D2-X control | D2-AE0 | D2-AE − D2-X / ratio |
|---|---:|---:|---:|
| end-object (cm) ↓ | 3.7402 | 4.2990 | 1.1477× |
| Txy (cm) ↓ | 4.0505 | 3.9894 | 0.9849× |
| FS ↓ | 0.36301 | 0.39896 | 1.0990× |
| contact precision ↑ | 0.78806 | 0.80363 | +0.01557 |
| contact recall ↑ | 0.59445 | 0.59614 | +0.00168 |
| contact F1 ↑ | 0.63743 | 0.64194 | +0.00452 |
| contact coverage | 0.47655 | 0.47663 | +0.00007 |
| Pbody ↓ | 3.8691 | 2.8603 | 0.7393× |
| hand penetration ↓ | 0.24536 | 0.17938 | 0.7311× |
| MPJPE (cm) ↓ | 12.0508 | 12.1558 | 1.0087× |
| Troot (cm) ↓ | 8.1701 | 8.1651 | 0.9994× |
| Tobj (cm) ↓ | 15.9940 | 15.9553 | 0.9976× |
| Oobj ↓ | 1.03094 | 1.00324 | 0.9731× |

Native gate evidence:

- contact F1 difference 95% CI:
  `[-0.0180941, 0.0268459]`;
- contact recall difference 95% CI:
  `[-0.0231387, 0.0263812]`;
- contact precision difference 95% CI:
  `[-0.00519356, 0.0367914]`;
- target contact F1 `0.641944`, below required `0.659884`;
- released contact-F1 gap closure `0.050293`, below required `0.25`;
- end-object ratio CI `[1.08425, 1.21382]` and FS ratio CI
  `[1.03233, 1.17347]` violate the protection upper bound;
- native transfer, protection and released-95% effectiveness gates failed;
- the sealed 181-sequence penetration finite-mask contract passed, sequence-ID
  SHA-256
  `2c47612e69e8f5f5a6fa5906fd6c2593d2ed021101933433be4cb641513439ec`;
- no local-BPS CPU build occurred; sparse metadata was forwarded three times and
  normalization had zero nonfinite generated values.

Classification precedence stops at native transfer:
`sparse-relation-field-transfer-negative-stop`. Later protection failures are
reported but do not replace that earlier class. The checkpoint is not selectable.

The detached wrapper's `exit_code` file contains preserved raw bytes `0n` rather
than `0` plus newline because of shell escaping. It was not overwritten. A
postflight verifier parsed the leading return code as zero and bound completed
metrics, raw artifact hashes, process/GPU postflight and gate precedence. Its first
schema check incorrectly required aggregate-only aliases on per-sequence rows; that
failed audit file remains preserved. An append-only r1 verifier corrected only the
serialized field mapping and passed without changing any metric, mask, reduction,
threshold or workload.

Metrics/manifest/aggregate/per-sequence/resolved/preflight/run-local-registry SHA-256:

`55927deb...60f8` / `419be60f...d8db` / `157acda4...2a1` /
`8533b66e...f95c` / `d747b549...a378` / `5572d7f5...487e` /
`8fb26313...e972`.

Failed/corrected postflight verifier SHA-256:

`05133fc6afc981ec8b28d7b3ede5c938da9110fc0146a704858045289ed50e15` /
`c4f2f86ccc341e835fdfe6f87f11fb9ec3d7dfa5db8c1bb4d4abba073ba28d18`.

The complete 18-file / 3,474,559-byte native tree matches worker and authority with
SHA-256
`4f31bb8f61bd40eb4604a25a0802a970686092306faf86efa0b289c856cd34b5`;
the checksum dry-run reported zero differences. The evaluator did not generate FID,
Matching, R-Precision or Diversity, and none was synthesized or backfilled.

## Interpretation

D2-AE0 resolves the two clearest D2-AC/D2-AD mechanism defects:

1. The new roles are structurally tied to joints 24/26/0 rather than three additive
   embeddings of one shared motion summary.
2. The new signal is an explicit current-state human/object relation rather than
   repeated global BPS tokens.
3. Fixed temporal correspondence and left/right binding both matter causally.
4. Pure GPU 100-point geometry avoids D2-AD's CPU full-mesh/KD-tree bottleneck and
   restores D2-X-class throughput.

That success is internal and architectural, not a native-quality success. Under the
official rollout, mean contact F1 and recall moved only marginally and their paired
confidence intervals crossed zero. The mechanism also traded better penetration,
Txy, Troot, Tobj and Oobj points for worse end-object and foot sliding. The supported
conclusion is therefore narrow: explicit sparse role-relative routing is learnable
and causally organized, but this fixed teacher-forced training setup did not turn it
into reliable 500-step native contact transfer.

This result does not justify tuning point count, width, depth, roles, anchors,
placement, LR, batch, thresholds or budget; none was tested. It also does not
authorize new loss terms, timestep weighting, rollout exposure, CFG, consistency,
D2-AE1, HSIPrior or Mixer.

## Verification and artifact entry points

Final authority verification uses:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" -m unittest tests.test_hoi_d2ae -v
"$INFBAGEL_PYTHON" -m unittest discover -s tests -v
"$INFBAGEL_PYTHON" tools/experiment.py validate
git diff --check
```

Primary recovered roots:

- CPU:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-cpu-contract-s42-20260728`;
- functional base/r1:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-gpu-functional-smoke[-r1]-s42-20260728`;
- performance base/r1:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-performance-benchmark[-r1]-s42-20260728`;
- formal failure and complete r1:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-s42-20260728`
  and
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-r1-s42-20260728-recovery-r1`;
- internal base/r1/r2:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-sparse-relation-field-internal[-r1|-r2]-s42-20260729`;
- native:
  `/data/yujinlun/InfBaGel-p1b-staging/p1-hoi-d2ae-native-eval-s42-20260729`.

Tracked compact result:

`experiments/results/p1_hoi_phase1b_d2ae_sparse_relation_field_s42_20260729.json`.

## Exact next entry point

Stop Phase 1B D2-AE0. Do not select or resume its checkpoint and do not start
D2-AE1, consistency, a longer budget, any sweep, HSIPrior or Mixer in this session.
Any later HOIPrior direction requires a new dated plan and registry hypothesis; the
sealed D2-AE0 result remains a transfer-negative architectural diagnostic.
