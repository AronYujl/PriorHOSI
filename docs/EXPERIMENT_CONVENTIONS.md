# Experiment conventions

Status: active from 2026-08-10 on every branch and for both experts.

Phase 1B produced, for roughly thirty experiments, 66 one-off `tools/` scripts,
35 one-off `tests/` modules, 25 retired configs and 18 near-duplicate evaluation
runners of about 1,400 lines each. Two of those runners differed by 839 lines.
The clutter was the visible cost; the drift between eighteen copies of the
evaluation protocol was the real one, because evaluation consistency is what
every comparison in the paper rests on.

These conventions exist so that effort and context go into experiment design
instead of file generation, without weakening accuracy or comparability.

## 1. One experiment adds no new scripts

An experiment is:

- **one config override fragment** — `defaults: [<base>, _self_]` carrying only
  the delta, which is how `code/config/config_train_hoi_prior_p9w3.yaml` already
  works;
- **one registry hypothesis row** in `experiments/registry.jsonl`, and later one
  completion row;
- **one dated section** in the phase file under `docs/plan/`.

Adding a file under `tools/` needs a reason recorded in the preregistration. The
default is that no new file is needed, because the generic path already exists.

## 2. The generic lifecycle path is the only path

`tools/experiment.py start` → the trainer or `code/test_infbagel_hoi.py` under
its Hydra config → `tools/paired_bootstrap.py`.

All sixteen P8/P9/P10 registry rows already use exactly this and reference no
per-experiment wrapper. That is the pattern; the eighteen deleted wrappers were
what it replaced. A new arm changes the config and the run id, not the entry
point — which is precisely what makes two arms comparable.

## 3. Tests are organised by component, never by experiment id

`tests/core/` holds the frozen-contract tests. `tests/hoi/` and `tests/hsi/` hold
per-expert component tests named after the component they cover
(`test_losses.py`, `test_guidance.py`), not after the experiment that motivated
them. `tests/` itself holds the project-wide governance and statistics tests.

When an experiment needs a new assertion, add it to the component's test module.
A file named after an experiment id becomes dead the day that experiment closes,
and 35 of them did.

## 4. Diagnostics are registered probes, not new files

A causal diagnostic is a named function in the expert's diagnostic module,
invoked by name. Do not create `diagnose_<experiment>.py`. The probe name goes in
the preregistration so the diagnostic is fixed before the result is seen.

## 5. Conclusions are not deletable; scripts are

The durable record of an experiment is its registry rows, its compact JSON under
`experiments/results/`, its phase summary, and `docs/HOIPRIOR_EVIDENCE_INDEX.md`.
Those are never rewritten and never deleted.

A script that produced a sealed result is recoverable from git history and from
the recovery tag, so once its experiment is closed and its numbers are sealed,
keeping it in the working tree buys nothing and costs every future reader's
context. Retire it in the same commit that closes the experiment, and record the
retirement in the closure note.

## 6. Retiring a config is not retiring a comparison point

Keep the config of any arm that is still cited as a comparison baseline — for
HOIPrior that is D2-X, D2-AE, D2-AG, D2-AI and W3, plus the live geometry
lineage. Retire the rest. If a retired arm is ever needed again, its resolved
config is archived beside its manifest, which is what provenance actually
depends on.
