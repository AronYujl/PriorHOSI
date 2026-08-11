# Claude Code repository entry point

Claude Code does not implicitly receive the repository rules from another agent.
Before taking any action in this checkout, read `AGENTS.md` completely and obey it.

For Phase 1B HOIPrior work, also read these files before proposing or implementing
an experiment:

1. `docs/HOIPRIOR_ITERATION_WORKFLOW.md`
2. `docs/HOIPRIOR_EVIDENCE_INDEX.md`
3. the relevant dated section under `docs/plan/PHASE_1B_HOI/` (start from its
   `README.md` index; `docs/plan/OVERVIEW.md` holds the cross-phase protocol)
4. the phase summaries and compact results named by the evidence index

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
