# Claude Code repository entry point

Claude Code does not implicitly receive the repository rules from another agent.
Before taking any action in this checkout, read `AGENTS.md` completely and obey it.

This checkout is **`phase/01c-hsi`: the HSIPrior branch.** It runs on the 8-GPU
authority host and owns `code/priors/hsi/`, `tests/hsi/`, the `*_hsi_*` tools and
the `config_train_hsi_prior_*` configs. `code/priors/core/` is the frozen contract
shared with HOIPrior and the mixer; changing it requires the user's explicit
approval. HOIPrior lives on `phase/01b-hoi` and iterates on the 4-GPU worker.

Read before proposing or implementing an experiment:

1. `docs/plan/OVERVIEW.md` and `docs/plan/PHASE_1C_HSI.md`
2. `docs/HSIPRIOR_DESIGN_PRIORS.md` — the measured Phase 1B negatives that bind
   Phase 1C, each with its evidence and its overturn condition
3. `docs/EXPERIMENT_CONVENTIONS.md`
4. `AGENTS.md`, section "Concurrent expert branches"

The Phase 1B HOI iteration history, `docs/HOIPRIOR_EVIDENCE_INDEX.md`,
`docs/HOIPRIOR_ITERATION_WORKFLOW.md` and `docs/phase_summaries/PHASE_1B_*.md` do
not exist on this branch by design. Do not fetch them from `phase/01b-hoi`:
everything transferable is already in `docs/HSIPRIOR_DESIGN_PRIORS.md`, and
pulling anything else across is cross-branch communication that needs the user's
approval first.

Phase 1C's gate is not yet defined. Do not start a formal HSIPrior training run
before the user has approved it.

The proposal stage is read-only. Model/source changes and GPU workloads require the
user's explicit approval of one concrete experiment.

## Standing subagent authorization

Dispatching subagents is pre-authorized; no per-turn request is required. Keep the
main session on dispatch, progress tracking, and aggregation, and delegate concrete
work to the `.claude/agents/` types by difficulty:

- `worker-medium` — the spec and pass criteria are already fixed.
- `worker-high` — cross-file implementation, refactoring, or non-trivial diagnosis.
- `worker-max` — high-risk, hard to reverse, or requiring adversarial self-doubt.

Subagents run with the same tools and permissions as the main session, so every rule
above and in `AGENTS.md` binds them identically. In particular, no agent may create a
commit or tag, allocate a run id, run `tools/experiment.py start`, or launch a GPU
workload without the user's explicit approval of one concrete experiment.

Subagents write long logs, tables, and diffs to `.claude/scratch/` (git-ignored) and
return bounded structured summaries instead of pasted file contents. They must not
create an untracked file anywhere else in the checkout: `tools/experiment.py` counts
one as a dirty worktree and refuses to start a reportable run.
