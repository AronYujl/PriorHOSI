# Lean Phase 1B HOIPrior iteration workflow

Status: active from 2026-07-30 for user-authorized Phase 1B continuation work.

This workflow removes repeated governance work while preserving the scientific
training and evaluation contract. It does not retroactively alter any D2 result,
artifact, classification, or checkpoint decision.

## 1. Invariants that are not slimmed

- HOIPrior remains independent, scene-free and composable, with the shared clean
  `[B,16,232]` output contract unless a user-approved hypothesis explicitly changes
  the common interface.
- Formal training uses seed 42, random initialization and the registered OMOMO split.
  Released, author and prior experimental checkpoints are baselines only.
- The HOI worker remains `infbagel-4gpu/node01` with 4x RTX 3090, committed clean
  source, verified input assets and `INFBAGEL_WORKER_EXPERT=hoi`.
- Reportable workloads use `tools/experiment.py start`, a fully resolved config and
  a same-context manifest. Run ids are never reused and failed runs are retained.
- The approved experiment owns one fixed scientific configuration. No unregistered
  sweep, checkpoint selection, best-of-N asymmetry or post-hoc metric change is
  allowed.
- Formal training, the registered causal diagnostic, fixed native evaluation,
  sequence-level uncertainty and protection gates remain scientifically complete.

## 2. Fixed lifecycle

### Stage A: read-only evidence review

1. Verify checkout path, branch, HEAD, clean status, date and authority Python.
2. Read `AGENTS.md`, this workflow, `docs/HOIPRIOR_EVIDENCE_INDEX.md`, the latest
   relevant summaries and compact results.
3. Inspect only the code paths needed to understand the candidate mechanism.
4. Produce exactly one recommended experiment with its manipulated factor, expected
   benefit, principal risk, causal diagnostic, training/evaluation contract and
   alternatives rejected by existing evidence.
5. Stop for explicit user approval. Do not edit files, allocate identifiers, load a
   checkpoint or start a GPU workload during Stage A.

### Stage B: one preregistration

After approval:

1. Audit the proposed identifier and actual-date run stem once.
2. Append one dated amendment to `docs/EXPERIMENT_PLAN.md` and one hypothesis to
   `experiments/registry.jsonl`.
3. Lock the single manipulated factor, fixed comparisons, performance condition,
   internal diagnostic, native gates, stop classifications and allowed file scope.
4. Commit this preregistration once. Do not add a second binding record that merely
   repeats the same plan or future commit hash.

### Stage C: one logical implementation

Implement source, config, tests, lifecycle integration and short documentation in one
logical commit. Keep unrelated user changes untouched.

Required checks are proportional to the changed path:

- always: targeted tests, config resolution, registry validation, `git diff --check`,
  output/checkpoint provenance contracts and forbidden-source scans relevant to the
  hypothesis;
- shared model/diffusion/training/data/evaluator changes: one full authority test
  suite before GPU publication;
- documentation/registry-only append: JSON/registry validation and diff check, with
  no repeated full suite;
- runtime-code change: one real-data functional smoke with finite forward/backward,
  gradients, memory and output API checks;
- compute/data/communication/tensor-shape change: one registered full-micro-batch
  performance benchmark; otherwise reuse the sealed execution profile and record the
  reason the benchmark is not applicable.

Publish the resulting clean commit to the worker by Git fast-forward. Do not create a
governance-only implementation-binding commit unless the worker must execute a
different source object from the logical implementation commit.

### Stage D: one formal training lineage

Run the approved full budget once from random initialization. The manifest records
the exact Git object, config, immutable input bundle, seed, hardware and initial model
identity. Training records finite initial behavior, throughput/ETA, memory headroom,
cadence counts, resumable state and the fixed final-online identity.

After the stable interval and first resumable checkpoint, stop active polling and let
the worker-owned persistent session finish. Resume is allowed only for a genuine
operational interruption and must continue the same scientific lineage; it is not a
second experiment or checkpoint selection.

### Stage E: fixed evaluation

1. Load only the fixed final-online checkpoint.
2. Run one preregistered internal causal diagnostic whose perturbation directly tests
   the new mechanism. Share initial latent, posterior noise, conditions and ordering
   across paired paths where required.
3. Run one fixed official native evaluation, even when the internal mechanism is
   negative if the preregistration requires it.
4. Reuse sealed D2-X and released baseline outputs; never regenerate them merely to
   repeat a comparison.
5. Compute the registered paired sequence-level uncertainty and every native
   transfer/protection metric. Do not add exploratory metrics to the decision after
   seeing the result.

Evaluation preflight verifies only the critical upstream identities: final checkpoint,
formal manifest/state, code/config, cohort or official split, evaluator and paired
noise contract. It does not re-audit every cadence checkpoint or re-run the full code
suite when source has not changed.

### Stage F: one recovery and closure

For each completed worker workload, perform one worker-initiated non-destructive
recovery and one checksum pass. Retain:

- manifest, resolved config, preflight, log and metrics;
- final model/checkpoint identity plus cadence/RNG counts;
- internal raw paired variants and uncertainty output;
- native aggregate, per-sequence and bootstrap output;
- dependency, hardware, data and evaluator identities by reference;
- one recovered-tree index, one compact tracked result and one concise phase summary.

Append one completion record and make one closure commit. The summary should contain
the scientific conclusion, failures, critical artifact hashes and next entry point,
not a chronological transcript of every shell wrapper.

## 3. Work that is intentionally removed

- no separate date-transition record when the actual-date run id can be generated at
  workload start;
- no repeated plan, implementation-binding, evaluation-hardening and recovery-binding
  commits when no scientific/source transition occurred;
- no full test-suite rerun after registry/summary-only edits;
- no repeated hashing of unchanged sealed baselines, data snapshots, evaluator assets,
  checkpoint cadences or already recovered trees;
- no registration of a local typo or wrapper mistake that occurred before a run id or
  manifest existed;
- no continuous polling after a detached run has proved stable and resumable;
- no large raw-log excerpts in plan, registry, prompt or phase summary.

These removals affect orchestration only. They do not reduce model checks, formal
budget, causal rollouts, native sequence coverage, metrics, bootstrap uncertainty or
failure retention after reportable start.

## 4. Minimal tracked record per experiment

| Stage | Required tracked record |
|---|---|
| approval | dated plan amendment and one registry hypothesis |
| implementation | one logical source/config/tests/docs commit |
| pre-GPU | concise CPU/smoke/performance outcome, only where applicable |
| training | run id, manifest/config hashes, final checkpoint hash, budget, throughput and failure state |
| evaluation | internal and native run ids, aggregate/per-sequence/bootstrap hashes and decision |
| closure | compact result, concise phase summary and one registry completion row |

Unchanged immutable inputs are referenced by their existing sealed hashes instead of
being copied into every later record.
