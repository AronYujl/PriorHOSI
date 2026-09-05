# Phase 1C R3-ND: rebase numerical diagnosis

R3-ND completed on 2026-09-05. The numerical mechanism is supported; the R3-AR
checkpoint remains a negative result. Phase 1C remains open and R2 final EMA
with posterior-coefficient guidance remains the working baseline.

## Scope and implementation

- Added one config, `config_sample_hsi_rebase_numerics.yaml`, and a named probe
  in `priors/hsi/diagnostics.py`, dispatched by the existing LINGO evaluator.
- The probe calls production `p_losses` on sealed R2/R3 final EMA weights.
  It changes precision and output bias locally through temporary forward hooks.
  The trunk is frozen; output-head gradients are measured without any update.
- Production `p_losses` additionally exposes its existing position and rotation
  loss tensors. Production training and sampling arithmetic are unchanged.
- The input adapter uses the formal mixed-dataset entry with float32 geometry
  and integer progress indices. The first two attempts per checkpoint failed
  before a completed prediction because these input semantics were initially
  incomplete. All four failures are retained in manifests and registry.

## Fixed protocol and evidence

352 exact-valid windows from the existing frozen 60 episodes; 12 terminal padded
windows excluded; timesteps 0/50/250/498; 11 batches of 32; seed 42; train-mode
dropout with paired RNG; no guidance, rollout, optimizer update or new checkpoint.
Each model has 36 cells: four precision paths times two bias states at each
timestep, plus a full-fp32 rebase-off descriptor. All cells completed.

R3 t=0:

| arithmetic | base | bias-intervention output RMS | bias-gradient RMS |
|---|---:|---:|---:|
| bf16 | 2.021580 | 7.200033 | 4.0020e-5 |
| rebase fp32 | 2.021308 | 7.198837 | 3.3547e-5 |
| head and rebase fp32 | 2.016813 | 1.3873e-6 | 6.4259e-10 |
| full fp32 | 2.015641 | 1.3854e-6 | 6.1491e-10 |

All four timesteps on both checkpoints meet the preregistered forward-invariance,
backward-leakage and head-localization criteria. The three bf16 trunk paths have
identical head inputs. R3 raw rotation output RMS is approximately 2771 versus
R2 0.575. Casting after the bf16 output projection preserves its quantization
and does not solve the problem.

R3 has substantial remaining learned error: full-fp32 t=0 rotation L1 is 2.0143
versus R2+c3 0.04985. The base-loss difference is +1.9654, paired 95% CI
[1.89846, 2.03351]. The other timesteps agree. Precision correction alone does
not recover this candidate. The hard first-frame constraint remains separate:
its parameter derivative is zero in exact arithmetic.

Forward readouts use equal-weight episode means and 10,000 seed-42 paired
bootstrap replicates. Gradients are batch quantities; no episode CI is assigned
to them. This is a single-step numerical diagnosis, not a reconstruction of the
historical optimization trajectory or a rollout improvement claim.

## Verification and artifacts

- Authority suite before first workload: `python -m pytest tests -q`:
  433 passed, 3 skipped. Commands used the canonical `infbagel` interpreter.
- Final input-adapter/component checks: 40 passed.
- Maximum c3 production-vs-readout discrepancy: R2 2.98e-8, R3 4.77e-7.
  All predictions and numeric batch readouts are finite.
- Registry validation passed before execution and after completion.
- Compact result: `experiments/results/p1_hsi_rebase_numerics_s42_20260905.json`.
- Per-window/batch records: `results/hsi_rebase_numerics/{r2_r2,r3_r2}/`.
- All 32 bootstrap reports: `results/hsi_rebase_numerics/paired_statistics/`.
- Manifests: `results/experiments/p1-hsi-rebase-numerics-{r2,r3}[-r1|-r2]-s42-20260905/`.
  Existing manifest/checkpoint identities are retained by reference.
- Total formal cost including the four failed entries: 0.087781 GPU-h, within 1.0.
  No production release tag or Phase 1C merge was made.

## Next entry point

Read this summary, `docs/plan/OVERVIEW.md`, and the R3-ND section of
`docs/plan/PHASE_1C_HSI.md`. Preserve R2+CG and the R3 failure. A proposed training
successor should keep the first future frame learnable, control output scale
before low-precision projection, and avoid subtracting large unconstrained
outputs. Its rollout protocol and cost must be reviewed before training; this
diagnostic does not authorize another R3 run, a precision sweep, distillation,
or changes to the frozen expert contract.
