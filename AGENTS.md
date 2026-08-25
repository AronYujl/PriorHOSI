# State-Compositional Priors: Repository Rules

These rules apply to every file in this repository.

## Motion-parameter generation workflow

When the user asks to generate or export motion params, use the established
worker2 inference-and-return workflow by default.

- This `visualization/renderer` worktree owns inference orchestration and
  visualization. Do not edit or add planning/context files to `phase/01b-hoi`
  or `phase/01c-hsi` for a motion-parameter generation request.
- The user's normal input is one absolute checkpoint path on the authority
  host. Start the workflow with:

  ```bash
  /data/yujinlun/anaconda3/envs/infbagel/bin/python \
    tools/worker2_inference.py start /absolute/authority/checkpoint.pth
  ```

- Read `docs/visualization/WORKER2_INFERENCE.md` and
  `tools/visualization/worker2_profiles.json` before execution. They are the
  canonical, updateable operational specification. Fail closed when no tested
  profile matches the checkpoint/model family. Do not derive an inference
  commit or evaluator configuration from a checkpoint's training commit.
- The currently validated HOI profile is `hoi-p12-armb`: P12 Arm B, guidance
  scale 1000, 10 last steps, clamp 1, clamp target `update`, `y_up` frame, and
  438 expected sequences. Its validated inference commit is
  `8742d1a3b88800161324a8e45c597ffafdcbb607`; the P12 checkpoint training
  commit `25931627a7a5668598e3120f57762546306c13a7` is not a substitute.
- Control worker2 through the authority host's loopback reverse endpoint
  `127.0.0.1:22216`. Worker2 data belongs below
  `/data2/yujinlun/infbagel-inference`; never use `/home/yujinlun/data` for
  worker2 data.
- Git, checkpoint, and artifact bulk transfers are worker2-initiated directly
  over the campus LAN to `10.184.17.253` with `ssh -F /dev/null`. Never route
  them through Windows or the Windows proxy.
- Keep inference in the worker-owned persistent systemd unit. Return results
  through `.incoming`, verify the motion artifact SHA256, and atomically
  promote it. Never overwrite or reuse a run ID or artifact destination.
- If generation succeeded but return failed, use
  `tools/worker2_inference.py retry-return RUN_ID`; do not rerun generation.
  Use `tools/worker2_inference.py status RUN_ID` for inspection.
- HOI architectures other than the registered P12 profile, plus HSIPrior and
  PriorHOSI, require a tested registered inference profile before the
  one-checkpoint workflow may run.

## Locked provenance

- The integration baseline is commit `b9a158f75ab0740c91c9cfc8863a65fa381b014c`.
- Development belongs on `research/state-compositional-priors` and short-lived
  `phase/00-eval`, `phase/01-priors`, ... branches.
- Do not inherit, merge, or cherry-pick `feature/independent-hoi-hsi-priors`.
- The released InfBaGel checkpoint is a baseline only. It must never initialize
  HOIPrior, HSIPrior, or the mixer.

## Experiment lifecycle

- Before adding a new direction, update `docs/EXPERIMENT_PLAN.md` and append a
  hypothesis to `experiments/registry.jsonl`.
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
- Screening may use one seed; main-table configurations require at least three
  training seeds and the registered statistical protocol.
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
  `docs/EXPERIMENT_PLAN.md`, branch names, gates, and registry phase/component
  labels so every subphase has a concrete deliverable and gate.
- Merge a phase only after its gate in `docs/EXPERIMENT_PLAN.md` is met; then tag
  the immutable result (for example `exp/p0-baseline-v1`).
- Before merging, write `docs/phase_summaries/PHASE_<N>.md` (or one summary per
  preregistered subphase). It must record scope, implementation and configuration
  changes, experiments and results including failures, verification commands,
  artifacts/hashes, commits/tag, unresolved risks, and the exact next-phase entry
  point. A new session must read this summary and `docs/EXPERIMENT_PLAN.md` first.
- A failed gate permits only its preregistered diagnostics/fallback. Any new
  direction requires a dated plan and registry update before code changes.
