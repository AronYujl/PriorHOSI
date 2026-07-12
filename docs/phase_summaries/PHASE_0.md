# Phase 0 Handoff: Governance, Data, and Evaluation Closure

## Scope and gate decision

Phase 0 established the reproducibility and evaluation boundary for the
state-compositional-priors study. The gate **passed**: the 469-case Atomic-HOSI
baseline is within the locked reproduction tolerance, the native HOI and pinned
CHOIS metrics run end to end, the LINGO scene-family split is fixed, timing is
separated by protocol, and the 8-GPU training micro-batch decision is complete.

Phase 0 does not train HOIPrior, HSIPrior, or a mixer. The released InfBaGel
checkpoint remains a baseline only.

## Implemented features and configuration changes

- Added repository governance in `AGENTS.md`, the preregistered full plan, an
  append-only registry, immutable manifests, asset/config/dependency/hardware
  hashing, dirty-worktree refusal, split/evaluator validation, and tests.
- Generated the seed-42 LINGO scene-family-disjoint split: 76 families, 111
  scenes, and 19,450 sequences; train has 61 families/91 scenes/17,555
  sequences, validation has 15 families/20 scenes/1,895 sequences. The no-hand-
  interaction HSI filter retains 1,939,898 of 2,275,973 windows.
- Pinned CHOIS at commit `8ec585aa0200fd2a890ffb12897bcf69ae719463`
  and its separately pinned text-to-motion parser dependency at commit
  `72df96ec453edea2fbe9603b1d58a955eaf71636`. Added strict input/checkpoint/
  source verification and an adapter for the released evaluator.
- Made Atomic-HOSI and HOI evaluation deterministic, non-overwriting, resolved-
  config aware, CUDA-synchronized, and capable of aggregate JSON output. Added
  strict matched prediction/GT NPZ export for the CHOIS evaluator.
- Added a separate warm-start batch-1 HOI timing protocol so batched throughput
  is never reported as single-sample latency.
- Added real DDP micro-batch auditing and training gradient accumulation. The
  trainer rejects any configuration where
  `micro_batch × world_size × accumulation != effective_batch_size`, supports a
  bounded optimizer-update budget, records per-rank peak memory, and does not
  emit audit checkpoints.
- Locked Phase 1 execution into subphases 1A data/scaffolding, 1B HOIPrior, 1C
  HSIPrior, and 1D joint gate. Each has an independent branch, gate, summary,
  and tag; no Phase 1 implementation was started in this session.

## Experiments and results

### Atomic-HOSI baseline

Registry id: `p0-atomic-hosi-baseline-r2-s42-20260712`.

- 469/469 cases across 67 scenes; 383 completed.
- Completion 81.663%; pelvis/object goal error 4.686/8.129 cm.
- Foot sliding 0.1384; contact 0.7805.
- Warm generation 25.647 FPS by mean and 25.379 FPS by aggregate frames/time;
  end-to-end episode latency 8.194 s.
- This is within the locked reference (82.09%, 4.57/8.17 cm, 0.14, 0.781,
  23.34 FPS).

### Native HOI and CHOIS metrics

Registry ids: `p0-hoi-table5-baseline-s42-20260712` and
`p0-hoi-chois-matched-s42-20260712`.

- Native protocol: 438 sequences × 3 windows; object/pelvis goal error
  3.037/3.923 cm; foot sliding 0.3334.
- Contact precision/recall/F1 0.7908/0.7276/0.7273; MPJPE 11.998;
  human-object penetration 2.5893 with ratio 0.1376.
- Matched CHOIS evaluation over 438 prediction/GT pairs: FID 0.93342,
  R-Precision@1/2/3 0.17308/0.31010/0.43510, Matching Score 3.82295, and
  Diversity 9.14892.
- Batch-438 throughput is 322.56 FPS and is labeled throughput only.

The separate official supplied-result regression produced FID 0.68816 and
R-Precision@1/2/3 0.15208/0.29792/0.42500, verifying the adapter against the
released assets before using InfBaGel exports.

### Batch-1 timing

Registry id: `p0-hoi-batch1-timing-r1-s42-20260712`.

After one excluded warm-up batch, one deterministic three-window sequence
generated 126 frames in 7.0871 s: 17.779 FPS. End-to-end time including metrics
was 23.820 s. This baseline is below the Phase 6 Fast target of 20 FPS; that is a
recorded baseline limitation, not a Phase 0 gate failure.

### 8-GPU micro-batch selection

Registry id: `p0-train-microbatch-audit-r1-s42-20260712`.

All candidates completed a full forward/backward and one optimizer update on
real OMOMO training data at global effective batch 1024:

| Per-GPU micro-batch | Accumulation | Last loss | Max allocated | Max reserved | Result |
|---:|---:|---:|---:|---:|:---|
| 32 | 4 | 505.850 | 2.486 GB | 2.881 GB | stable |
| 64 | 2 | 526.642 | 3.396 GB | 4.855 GB | stable |
| 128 | 1 | 542.069 | 5.062 GB | 6.713 GB | stable |

The preregistered rule selects the maximum stable value: **128 per GPU**, eight
GPUs, accumulation **1**, global effective batch **1024**. Later methods must
keep effective batch and optimizer-update budgets fixed.

## Failed and negative runs retained

- `p0-atomic-hosi-baseline-s42-20260712` aborted before GPU work because its
  manifest omitted the required `code/` working directory.
- `p0-atomic-hosi-baseline-r1-s42-20260712` aborted during resolved-config
  preflight because of stale `${dataset.seq_len}` interpolation.
- `p0-hoi-batch1-timing-s42-20260712` generated its subset but failed metrics
  because post-processing reloaded the full 438-sequence count.
- `p0-train-microbatch-audit-s42-20260712` reached the real loss path for all
  three candidates but failed on stale `occ_goal/occ_temp` cleanup references.
  The narrow cleanup fix was tested and rerun under the immutable `r1` id.

No failed id or artifact was overwritten, and all four outcomes remain in the
registry.

## Verification

Final pre-merge checks use the existing environment:

```bash
/data/yujinlun/anaconda3/envs/infbagel/bin/python -m unittest discover -s tests -v
/data/yujinlun/anaconda3/envs/infbagel/bin/python tools/experiment.py validate
/data/yujinlun/anaconda3/envs/infbagel/bin/python -m py_compile \
  code/train_infbagel.py code/models/infbagel.py tools/benchmark_train_microbatch.py
git diff --check
```

Expected result at closure: 16 tests pass; 11 registry records, one split, and
two evaluator definitions validate; the worktree is clean before integration.

## Artifacts and hashes

Tracked aggregate results live in `experiments/results/`; raw manifests, logs,
matched NPZ exports, and per-sample outputs remain ignored under
`results/experiments/`.

- Atomic checkpoint SHA-256:
  `6853351b3f5468d21293d12a6dd2bccde62c1400fdb83934f7878136b131b198`.
- Matched HOI prediction/GT trees:
  `dbc43eb78ac633b769d6bdd32f5e3fdf024b19eb2afa766ae50b5a8be6585386` /
  `d439a98ea32f5d67964bc98431fe25bdffc24b63e00b42601c5355445d01742c`.
- OMOMO train tree:
  `bfa7f32259ab524b06c395c2b622fd122ce156d9cf3d94ef193f1028dd9b7fdf`.
- Successful micro-batch manifest/metrics/resolved-config archive:
  `f84303242976c3790a67b78f8a52284ce17da6dee2a70ca8771c415bd2b537a6` /
  `ba61dc0d089cf4ca2f4bdefc3646f3dc5994f8b72b3289c9e3806e32a98211eb` /
  `0e4756dec679eb44abe6e89999308b900778afd46fd3fb507ad304abfec59e97`.

## Commits, integration, and tag

Phase work spans `16640fb` through the closing Phase 0 results/summary commit.
Key checkpoints are `2f495bc`/`f0c3885` (CHOIS adapter/regression), `2b46d47`
(Atomic baseline), `38d0741`/`361d921` (HOI/CHOIS), `7baa1d4` (batch-1 timing),
`07edf6d`/`da0b301` (micro-batch audit), `9907afe` (negative audit), and
`7f89b63` (training-loss cleanup fix).

The phase branch is fast-forwarded into `research/state-compositional-priors`;
there is therefore no separate merge commit. The immutable closure tag is
`exp/p0-baseline-v1`.

## Unresolved limitations and risks

- The selected micro-batch passed a real optimizer update but not a long-run
  fragmentation/stability soak. Full Phase 1 runs must still record peak memory
  and fail without silently lowering effective batch.
- HOI batch-1 baseline speed is 17.779 FPS; realtime improvement is deferred to
  the preregistered consistency/optimization phase.
- The current work is kinematics-based and uses upstream/oracle goals; it does
  not solve semantic localization or low-level physics control.
- Phase 0 proves the baseline/evaluator closure, not the independent-prior
  quality claim. That claim cannot be made until all Phase 1 subphase gates pass.

## Exact next-session entry point

1. Read this file and the complete `docs/EXPERIMENT_PLAN.md`.
2. Verify `research/state-compositional-priors` contains tag
   `exp/p0-baseline-v1` and has a clean worktree.
3. Create `phase/01a-data` from the tagged research branch; do not start 1B, 1C,
   or 1D in the same session.
4. Before implementation, append the Phase 1A hypothesis/config contract to the
   plan and registry workflow. Implement only the data/filter/representation/
   from-scratch assertions and smoke gates specified for Phase 1A.
5. Close that session with `docs/phase_summaries/PHASE_1A.md` and tag
   `exp/p1a-data-v1` only if its gate passes.
