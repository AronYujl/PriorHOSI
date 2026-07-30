# Prompt for Claude Code Opus 5: final Phase 1B HOIPrior review and experiment

You are working as the primary research engineer for the State-Compositional Priors
project. Your task has two stages. First, independently analyze the full HOIPrior
evidence and recommend exactly one highest-value experiment. Stop for my approval.
Only after I explicitly approve that recommendation may you implement it, train it on
the four-GPU worker and complete its internal and native evaluation.

This is a one-experiment budget. Do not turn it into a sweep, several candidates, or a
sequence of exploratory full trainings.

## 1. Repository and instruction bootstrap

Claude Code does not automatically inherit Codex's `AGENTS.md` instructions. Before
reasoning about the research or touching any file, explicitly run and inspect:

```bash
cd /data/yujinlun/InfBaGel-release
pwd -P
readlink -f .
git branch --show-current
git rev-parse HEAD
git log -1 --pretty=%s
git status --short
date --iso-8601=seconds
sed -n '1,320p' CLAUDE.md
sed -n '1,360p' AGENTS.md
```

Expected branch: `phase/01b-hoi`. The checkout must be clean and its HEAD must be a
descendant of historical D2-AF closure commit
`51ffd4806f12c79d0571af140c20a72485ed414d`. If path, branch, ancestry or clean status
does not match, stop and report; do not reset, merge, cherry-pick, stash or clean.

Set and use the authority interpreter for every project Python command:

```bash
export INFBAGEL_PYTHON=/data/yujinlun/anaconda3/envs/infbagel/bin/python
"$INFBAGEL_PYTHON" --version
```

Never silently use system Python.

Then read completely:

1. `docs/HOIPRIOR_ITERATION_WORKFLOW.md`
2. `docs/HOIPRIOR_EVIDENCE_INDEX.md`
3. the Phase 1B portion and latest dated amendments in `docs/EXPERIMENT_PLAN.md`
4. the decisive summaries `PHASE_1B_D2X.md`, `PHASE_1B_D2AB.md`,
   `PHASE_1B_D2AC.md`, `PHASE_1B_D2AD.md`, `PHASE_1B_D2AE.md` and
   `PHASE_1B_D2AF.md`
5. all `experiments/results/p1_hoi_phase1b_*.json` through a small parser that
   extracts every mechanism, classification, metric and confidence interval; open
   additional long summaries only when the compact evidence exposes a real gap
6. `docs/MULTI_SERVER_TRAINING.md`

For the source audit, locate the active D2-X/D2-AE/D2-AF model, training-loss,
diffusion, sampler, data, evaluator and test paths with `rg`. Also inspect the author's
contrast path in `code/models/infbagel.py`,
`code/config/config_train_infbagel.yaml` and its dataset occupancy builder so you do
not accidentally reintroduce its scene or clean-future leakage.

Do not begin by reading every raw log or multi-gigabyte artifact tree. Use the compact
evidence index and summaries first; open raw artifacts only to resolve a concrete
discrepancy. If you use subagents, give each one a bounded read-only audit and require
a compact evidence report rather than copying logs into the main context.

## 2. User intent and current phase authority

I am not satisfied with the current HOIPrior and I have explicitly reopened Phase 1B.
The historical D2-AF negative result remains immutable, but its earlier statement that
Phase 1B was closed is superseded prospectively by the latest user-directed plan
amendment. Do not delete or rewrite the historical closure record.

I will fund exactly one more full HOIPrior experiment before moving to HSIPrior. I want
Opus 5 to use the accumulated evidence, not merely continue the most recent adapter or
make a cosmetic parameter change. Your recommendation must be the single direction
with the highest expected scientific value and probability of improving the official
native result.

Stage A is analysis only. No edits, plan/registry append, identifiers, manifests,
checkpoint loads or GPU work are authorized before I approve your recommendation.

After approval, the authorized scope is one end-to-end lifecycle: preregistration,
implementation, tests, required CPU/GPU preflight, one from-random formal training,
one fixed internal diagnostic, one fixed native evaluation, recovery and concise
closure. Within that approved scope, proceed autonomously without asking for approval
at every normal step. Stop only for a failed hard gate, repository mismatch, an action
outside the approved experiment, or a genuinely material choice not covered by the
approved design.

Do not start HSIPrior, Mixer, consistency distillation or another HOIPrior experiment.

## 3. Locked scientific background

Released InfBaGel is a baseline only and cannot initialize any new prior. Its native
metrics include:

- end-object `3.03724 cm`;
- FS `0.33336`;
- contact precision/recall/F1 `0.79081 / 0.72759 / 0.72726`;
- hand penetration `0.16240`;
- MPJPE `11.99759 cm`.

The sealed autonomous control is D2-X, trained from random initialization for
61,440,000 windows / 30,000 updates on 4x RTX 3090:

- throughput `3243.036 windows/s`;
- end-object `3.7402 cm`;
- FS `0.36301`;
- contact precision/recall/F1 `0.78806 / 0.59445 / 0.63743`;
- hand penetration `0.24536`;
- MPJPE `12.0508 cm`.

The main verified research history is:

- D2-U/D2-V: balanced objective plus the long budget proves the autonomous 232-D
  denoiser has enough basic capacity. Remaining deficiencies are rollout contact
  recall/coverage, FS and penetration, not simple undertraining.
- D2-W/X/Y/Z: evaluator-aligned FK-foot routing is safe but not statistically
  sufficient; amplification and immutable-GT gating harm goal/contact protection.
- D2-AB: predicted-support no-slip supervision failed its internal optimization
  direction and did not improve native FS.
- D2-AC: the adapter was strongly used, but locality permutation barely mattered;
  it became a generic residual/conditioning path. Native F1 `0.64799` came with
  end-object `5.6473 cm` and FS `0.39861`.
- D2-AD: repairing the local coordinate frame did not make correspondence causal and
  reduced native F1 to `0.58687`; its CPU full-mesh/KD-tree path was also structurally
  too slow.
- D2-AE: a GPU-native 100-point current-state sparse role-relative field produced
  strong causal internal gate, temporal and left/right-role effects. Yet native F1
  was only `0.64194`, recall `0.59614`, end-object `4.2990 cm`, FS `0.39896`; gains over
  D2-X were statistically uncertain.
- D2-AF: multiplying the D2-AE relation residual by `sqrt(alpha_bar[t])` failed all
  seven internal causal gates. Native F1 was `0.64106`, recall `0.59904`, end-object
  `5.5735 cm`; it did not repair D2-AE.
- D2-F/H/I/J/K/L/M/N/O/P/Q/R/S/T and related diagnostics rejected simple sampler
  exposure, gradient dominance, clipping, AdamW, auxiliary balancing, guidance and
  author-update explanations. Read their compact records before resurrecting one.

The strongest unresolved problem is not how to make a high-leverage path nonzero. It
is how to convert structurally meaningful current-state interaction information into
better 500-step native contact recall while preserving object goals, FS, penetration
and kinematics. Teacher-forced loss improvements and whole-gate ablations are
insufficient evidence.

The author's dynamic occupancy path is not a clean solution: it uses scene assets and
different train/sample relation sources, including clean-future or previous-x0 anchors.
HOIPrior must remain independent and scene-free. No Scene asset, static occupancy or
future clean/GT relation leakage is permitted.

## 4. Stage A deliverable: recommend exactly one experiment

Audit the current model, diffusion, loss, data, sampler and evaluator paths relevant to
your diagnosis. Then give me a concise but technically complete proposal containing:

1. **Evidence diagnosis.** Identify the dominant bottleneck supported by the full D2
   history. Separate verified facts from inference.
2. **One manipulated factor.** Define exactly one new mechanism. State every tensor
   source, training/sampling symmetry, placement, parameterization and changed file.
3. **Why this is the best remaining bet.** Compare it against at least three plausible
   alternatives and reject them using existing evidence rather than preference.
4. **Compatibility.** Explain how it preserves scene-free independence, random origin,
   the clean `[B,16,232]` contract and later HSIPrior/Mixer composition.
5. **Leakage and shortcut audit.** Prove it does not use future clean `x0`, future GT,
   Scene assets, cached future relations, an old checkpoint or a sampler-only signal
   unavailable during training.
6. **Causal diagnostic.** Design a paired internal intervention that distinguishes the
   intended information from generic residual reliance. Do not rely only on whole-gate
   ablation, validation loss or descriptive attention norms.
7. **Native success and protection gates.** Use D2-X and released InfBaGel as sealed
   controls. By default require statistically positive D2-X contact F1 and recall,
   at least 25% released-gap closure (`F1 >= approximately 0.6598838781`), and the
   existing end-object/Txy/FS/Pbody/penetration/MPJPE/Troot/Tobj/Oobj protection rules.
   If you propose different gates, justify them before approval.
8. **Compute contract.** Estimate added parameters, memory and throughput. Use the
   sealed D2-X `3243.036 windows/s` profile; default performance floor is 85% of D2-X
   (`2756.58 windows/s`) when the runtime path changes.
9. **Failure modes.** State what a negative internal or native result would teach us
   and the exact stop classification. No fallback experiment or sweep may be hidden in
   the proposal.
10. **Execution plan.** List the minimal preregistration, implementation, tests,
    smoke/benchmark conditions, formal run, internal/native evaluation and artifacts
    required by `docs/HOIPRIOR_ITERATION_WORKFLOW.md`.

Choose one direction, not a menu. End Stage A with a direct request for my approval and
wait. Do not modify the repository while waiting.

You may recommend a mechanism that was outside an older D2-specific authorization, but
you must flag that explicitly. My approval of your Stage A proposal is what authorizes
that exact new scope; do not infer authorization for adjacent mechanisms.

In particular, SNR/timestep loss weighting, gradient projection, rollout exposure,
CFG/guidance, a new contact/no-slip/penetration loss, learned reliability schedules or
other previously forbidden D2 variants are not implicitly authorized. You may argue
for exactly one of them during Stage A only if the accumulated evidence makes it the
best remaining experiment; implementation requires my explicit approval of that
specific mechanism. Scene conditioning, HSIPrior, Mixer, consistency distillation,
old-checkpoint initialization and multi-candidate sweeps remain outside this budget.

## 5. Stage B contract after explicit approval

Once I approve the recommendation:

1. Recheck path/branch/HEAD/clean/date and audit the new identifier.
2. Append one dated plan amendment and one registry hypothesis, then make one
   preregistration commit.
3. Implement source, config, tests and concise documentation in one logical commit.
   Do not create separate binding/hardening commits that only repeat manifest hashes.
4. Run targeted contracts and exactly one full authority suite if shared runtime code
   changed. Validate registry/config and keep the worktree clean.
5. Publish the committed object to `/home/yujinlun/data/work/InfBaGel-release` by the
   worker-initiated fast-forward workflow. The worker must not edit source.
6. Use worker Python verified on site and `INFBAGEL_WORKER_EXPERT=hoi`. Generate the
   fully resolved config and reportable manifest in the same GPU execution context.
7. Run one real-data functional smoke. Run one full-micro-batch performance benchmark
   only when the approved mechanism affects compute, data, communication, shapes or
   memory. If its registered hard gate fails, retain it and stop before formal training.
8. If every gate passes, run one formal seed-42 training from random initialization.
   Preserve the D2-X comparison contract unless the approved proposal explicitly
   changes one factor: 4x RTX 3090, per-GPU batch 512, effective batch 2048,
   accumulation 1, 61,440,000 windows, 30,000 FP32 Adam updates, no old checkpoint,
   no EMA and no checkpoint selection.
9. After finite initial behavior, memory headroom, throughput and one resumable
   checkpoint are established, report the ETA and stop continuous polling. Let the
   worker-owned persistent session finish.
10. Evaluate only the fixed final-online checkpoint with the approved paired internal
    diagnostic and the fixed 438x3, 500-step unguided native protocol. Reuse sealed
    D2-X/released artifacts without regeneration.
11. Recover each workload once by non-destructive worker-initiated transfer, perform
    one checksum pass, write one compact result, one concise phase summary and one
    registry completion record, then make one closure commit.

Critical artifacts and identities must remain verifiable, but do not repeat hashes or
full-suite tests after documentation-only appends. Do not paste long logs into chat;
report decisions, metrics, failures, artifact roots and essential hashes compactly.

## 6. Infrastructure constraints

- Authority repository: `/data/yujinlun/InfBaGel-release`
- Authority Python: `/data/yujinlun/anaconda3/envs/infbagel/bin/python`
- HOI worker: `infbagel-4gpu/node01`, 4x RTX 3090
- Worker repository: `/home/yujinlun/data/work/InfBaGel-release`
- Worker expert mode: `INFBAGEL_WORKER_EXPERT=hoi`
- Interactive control endpoint on authority: `127.0.0.1:22214`
- Bulk Git/data/artifact transfer is worker-initiated; do not make long workloads
  depend on the interactive tunnel.
- Integration baseline remains
  `b9a158f75ab0740c91c9cfc8863a65fa381b014c`.
- Never merge or cherry-pick `feature/independent-hoi-hsi-priors`.

Start with Stage A only.
