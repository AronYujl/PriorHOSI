# State-Compositional Priors: Repository Rules

These rules apply to every file in this repository.

## Locked provenance

- The integration baseline is commit `b9a158f75ab0740c91c9cfc8863a65fa381b014c`.
- Development belongs on `research/state-compositional-priors` and short-lived
  `phase/00-eval`, `phase/01-priors`, ... branches.
- Do not inherit, merge, or cherry-pick `feature/independent-hoi-hsi-priors`.
- The released InfBaGel checkpoint is a baseline only. It must never initialize
  HOIPrior, HSIPrior, or the mixer.

## Concurrent expert branches

From 2026-08-10 the two experts and the mixer advance concurrently at different
rates. `phase/01b-hoi` is not frozen and no expert is sealed before the others
start; the mixer may later require retraining an expert.

- `phase/01b-hoi` owns `code/priors/hoi/`, `tests/hoi/`, the `*_hoi_*` tools and
  the `config_train_hoi_prior_*` configs. It iterates on the 4-GPU worker
  `10.181.9.214`.
- `phase/01c-hsi` owns `code/priors/hsi/`, `tests/hsi/`, `*_hsi_*` tools and
  `config_train_hsi_prior_*` configs. It iterates on the 8-GPU authority host.
- Each branch deletes the other expert's files rather than carrying them, so the
  two working trees stay context-clean and their tracked paths stay disjoint.
- `code/priors/hsi/` on `phase/01b-hoi` is a read-only mirror kept only so the
  parameter-independence and expert-contract tests can run there. Never edit it
  on the HOI branch.

### The frozen contract

`code/priors/core/` is the only code both branches and the future mixer share.
Changing any file under it is by definition cross-branch communication and
requires the user's explicit approval plus a matching change on the other expert
branch in the same session. `tests/core/test_contract_freeze.py` enforces this
mechanically: it pins the SHA256 of every `core/` file and asserts that nothing
in `core/` imports from `priors.hoi` or `priors.hsi`.

Everything outside `core/` — datasets, objectives, architectures, samplers,
diagnostics — is per-expert and needs no approval to diverge.

### Cross-branch communication

Carrying a result, failure lesson, tuned value or document from one expert
branch to the other requires the user's explicit approval first. This includes
writing an HOI conclusion into an HSI plan file and vice versa. The transferable
Phase 1B lessons that are already approved for HSIPrior are recorded once, in
`docs/HSIPRIOR_DESIGN_PRIORS.md`; adding to that file is itself a cross-branch
action.

### Final integration is a graft, not a merge

Because the branches are path-disjoint by construction, recombine them by
checking the HSI-owned paths out of the HSI branch onto an integration branch
cut from the HOI branch:

```
git checkout -b integration/p1-priors phase/01b-hoi
git checkout phase/01c-hsi -- code/priors/hsi/ tests/hsi/ \
    docs/plan/PHASE_1C_HSI.md <hsi tools and configs>
```

A `git merge` would instead raise modify/delete conflicts on every `hoi/` file
the HSI branch deleted, at the worst possible moment. Verify after the graft that
`core/` is byte-identical on both branches — if it is not, the mixer's premise
was violated somewhere and that must be resolved before composition.


## Experiment lifecycle

- Before adding a new direction, update the phase file under `docs/plan/`
  (navigation page: `docs/EXPERIMENT_PLAN.md`) and append a hypothesis to
  `experiments/registry.jsonl`.
- Follow `docs/EXPERIMENT_CONVENTIONS.md`: an experiment adds one config override
  fragment, one registry row and one dated plan section, and no new script. The
  lifecycle path is `tools/experiment.py start` -> trainer/evaluator under its
  Hydra config -> `tools/paired_bootstrap.py`. Tests are organised by component,
  never by experiment id.
- Before proposing or implementing HSIPrior work, read
  `docs/HSIPRIOR_DESIGN_PRIORS.md`. It carries the measured Phase 1B negatives
  forward as binding defaults with their evidence and their overturn condition.
- Training must refuse a dirty worktree. Use `tools/experiment.py start`; do not
  bypass the check for reportable runs.
- Name runs `<phase>-<component>-<variant>-s<seed>-<YYYYMMDD>`. Never reuse a run
  id and never overwrite results.
- Local JSONL manifests are authoritative. TensorBoard is visualization only.
- Every manifest records commit/diff state, resolved config and hashes for data,
  evaluator, checkpoints, dependencies, seed, hardware, command working directory,
  timestamps, and metrics.
- Keep large data, checkpoints, generated motion, and per-sample results out of
  Git. Track split/task manifests, aggregate tables, failure analyses, and hashes.

### Lean HOIPrior iteration profile

For Phase 1B HOIPrior work after 2026-07-30, follow
`docs/HOIPRIOR_ITERATION_WORKFLOW.md`. Its purpose is to reduce orchestration and
context overhead without changing model training or scientific evaluation.

- A proposal/review turn is read-only. Do not edit source, allocate a run id, or
  start a GPU workload until the user explicitly approves one concrete experiment.
- After approval, use one preregistration commit, one logical implementation
  commit, and one completion commit. Add another governance-only commit only when
  an actual source transition, reportable failure, or resume contract requires it;
  never add a binding commit merely to restate hashes already present in a manifest.
- Reuse sealed input hashes and baseline artifacts by reference. Recompute a hash
  only when the file was created, changed, transferred, or is the exact target of
  the current workload. Do not repeatedly hash unchanged checkpoint cadences or
  recovered trees.
- Run targeted tests for every changed component. Run the full authority suite once
  before the first GPU workload when shared model, diffusion, training, data, or
  evaluator code changed; do not rerun it after documentation-only lifecycle appends.
- A real-data functional smoke is required for runtime-code changes. A full-micro-
  batch performance benchmark is required only when the change can affect per-step
  compute, communication, data loading, tensor shapes, or memory. Record why it is
  skipped when the executed path is unchanged.
- Preserve one formal from-random training, the preregistered internal diagnostic,
  and the fixed native evaluation. Workflow slimming must never alter their data,
  budget, sampler, metrics, uncertainty, checkpoint-selection, or failure rules.
- Recover each completed worker workload once with non-destructive transfer and one
  checksum pass. Keep the manifest, resolved config, logs, metrics, final checkpoint
  identity, aggregate/per-sequence outputs, compact result, and completion record;
  avoid duplicate archival wrappers and per-file narrative in Git.
- A local command mistake before `tools/experiment.py start` is implementation work,
  not a reportable experiment. Once a run id or manifest exists, retain every
  operational or scientific failure and never reuse or overwrite it.

## Execution environment

- Run project Python commands with the machine-local, verified `infbagel`
  environment. Set `INFBAGEL_PYTHON` to its absolute interpreter path and invoke
  `"$INFBAGEL_PYTHON"`; on the 8-GPU development server the canonical value is
  `/data/yujinlun/anaconda3/envs/infbagel/bin/python`. Do not silently fall back to
  the system Python or create an unverified replacement environment.
- The 8-GPU server `10.184.17.253` is the authoritative development/integration
  host and the Phase 1C HSIPrior host. The 4-GPU server `10.181.9.214` is a
  Phase 1B HOIPrior execution worker. Host-local repository, environment, data,
  and output paths may differ, but every reportable run must use an identical
  committed Git object and verified input hashes.
- Publish committed code from the authoritative host to the worker with Git and
  transfer immutable data snapshots separately. Never bidirectionally `rsync` a
  live worktree, registry, checkpoint directory, or running result directory.
  The worker must not edit source while a reportable workload is running.
- Because inbound TCP/22 from the 8-GPU host to the 4-GPU worker is blocked, the
  worker initiates all server-to-server Git/rsync transfers over a dedicated,
  source-restricted SSH key to `10.184.17.253`. Its machine-local repository,
  environment, datasets, staging, and results live below
  `/home/yujinlun/data`; Windows is only a public-key/bootstrap client, not a
  bulk-data relay.
- Interactive control from the authority/Codex side uses the worker-initiated,
  loopback-only reverse SSH endpoint `127.0.0.1:22214` on the 8-GPU host. Keep
  the tunnel key restricted to that `permitlisten`, verify the worker host key,
  and never enable `GatewayPorts` or expose the endpoint on a LAN interface.
  This control channel does not change transfer ownership: Git/data/artifact
  bulk transfers remain worker-initiated, and remote commands must not edit the
  worker source checkout during a reportable run.
- Do not make a long reportable workload depend on an interactive SSH channel.
  Launch it in a worker-owned persistent session, keep its manifest and logs on
  the worker, and use the reverse channel only to start, inspect, or capture
  output. A tunnel interruption is an access event, not permission to restart,
  reuse a run id, or overwrite an existing result.
- Once a detached long run has passed resolved-config/preflight checks, produced
  finite losses and required gradients through its initial stability interval,
  stayed within the registered memory headroom, and demonstrated a resumable
  checkpoint, continuous Codex polling is not required. Report the measured
  throughput and remaining-time estimate, yield to the user, and inspect the
  persistent run again only after the user asks to continue or a separately
  configured failure notification arrives. This changes monitoring cadence, not
  manifest, failure-retention, checkpoint, or completion requirements.
- Keep `ROOT_DIR` equal to the current checkout root. Provide the worker's
  OMOMO-only snapshot through the checkout-local `data` link (or a subsequently
  tested explicit data-root configuration). Do not copy LINGO `data/dataset` or
  synthesized OMOMO `Scene*` assets into the HOI worker snapshot.
- Set `INFBAGEL_WORKER_EXPERT=hoi` when validating the HOI-only worker. This may
  skip only tests that load real LINGO files; representation, HSI mask/model API,
  HOI real-data, governance, and checkpoint-rejection tests must still run.
  `smpl_models` is a required code-independent kinematic asset for HOI loading,
  not scene supervision, and must be hash-verified on the worker.
- Follow `docs/MULTI_SERVER_TRAINING.md` for provisioning, immutable data
  verification, code publication, artifact recovery, and concurrent-run
  ownership. A reportable remote run may start only after its machine preflight
  has been archived beside the manifest.
- GPU/driver access can be hidden by the Codex sandbox. If `nvidia-smi` fails or
  `torch.cuda.is_available()` is false in a sandboxed command, rerun the relevant
  check or workload with escalated permissions before diagnosing missing GPUs.
- Reportable GPU manifests must be created from the same escalated execution
  context as the workload so their hardware snapshot reflects the actual GPUs.
- Before starting a reportable Hydra GPU workload, generate the fully resolved
  job config with the exact overrides and archive it beside the manifest. Treat
  any unresolved interpolation as a preflight failure; do not start the workload.
- The reproducible target is eight RTX 3090 24GB GPUs. Record any reduced GPU set,
  contention, or hardware substitution in the run manifest and registry.

## Reproducibility and reporting

- Use the fixed scene-disjoint LINGO split generated with seed 42. Keep mirror,
  new-locomotion, and action variants in the same scene family and split side.
- Select training resources independently for HOIPrior and HSIPrior: use the
  largest stable per-GPU micro-batch compatible with the selected conventional
  effective-batch tier on the assigned server, while leaving documented memory
  headroom. HOIPrior is assigned to 4×RTX 3090; HSIPrior is assigned to 8×RTX
  3090. Do not require the experts to use the same micro-batch or effective batch.
- Formal expert-training effective batch must use a registered conventional
  tier. The default candidates are `{512, 1024, 2048, 3072}`; values such as
  `1536` are forbidden. Any additional tier requires a dated plan/registry update.
- Keep effective batch and processed-window/frame budget fixed within a given
  expert's controlled comparisons. Across HOI and HSI, use processed windows or
  frames as the primary budget and record optimizer updates as a derived count.
- Prefer gradient accumulation 1 for throughput. Use accumulation only to reach
  the selected conventional effective-batch tier or when a preregistered
  optimization study justifies it. Jointly preregister learning rate and warmup
  when changing effective batch.
- CUDA timing must synchronize before and after measured regions. Report warm
  generation, planning, and end-to-end latency separately.
- All screening, training, main-table, and evaluation experiments use seed 42
  only. Report point estimates and registered sample/sequence-level uncertainty;
  do not require additional training seeds or claim cross-seed confidence intervals.
- Never select best-of-N for one method only. If multi-sampling is used, give all
  methods the same budget and report both mean and best.
- Never overwrite a result, omit a preregistered metric, cherry-pick favorable
  subsets, or suppress a failed/negative run.

## Merge gates

- A logical change includes its config, tests, and documentation in one commit.
- Run the relevant tests and registry validation before merging.
- Scope one working session to one phase. Do not begin the next phase in the same
  session after closing the current phase.
- If a planned phase cannot reasonably be implemented, verified, and summarized
  in one session, split it into numbered subphases before implementation. Update
  the relevant `docs/plan/` phase file, branch names, gates, and registry
  phase/component labels so every subphase has a concrete deliverable and gate.
- Merge a phase only after its gate in its `docs/plan/` phase file is met; then
  tag the immutable result (for example `exp/p0-baseline-v1`).
- Before merging, write `docs/phase_summaries/PHASE_<N>.md` (or one summary per
  preregistered subphase). It must record scope, implementation and configuration
  changes, experiments and results including failures, verification commands,
  artifacts/hashes, commits/tag, unresolved risks, and the exact next-phase entry
  point. A new session must read this summary, `docs/plan/OVERVIEW.md`, and the
  `docs/plan/` file for the phase it is working on first.
- A failed gate permits only its preregistered diagnostics/fallback. Any new
  direction requires a dated plan and registry update before code changes.
