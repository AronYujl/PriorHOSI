# Phase Summary Contract

Every completed phase, or preregistered subphase, must add an immutable handoff
summary in this directory before merge. Use `PHASE_<N>.md`; a split phase uses
the identifier declared in `docs/EXPERIMENT_PLAN.md`.

Each summary must contain:

1. scope and gate decision;
2. implemented features and configuration/API changes;
3. completed and failed experiments, with metrics and registry ids;
4. verification commands and outcomes;
5. tracked results plus external artifact locations and hashes;
6. relevant commits, merge commit, and immutable tag;
7. unresolved limitations or risks;
8. exact prerequisites and first action for the next phase/session.

Write facts that can be checked from the repository. Do not copy large raw logs
or untracked per-sample outputs into Git.
