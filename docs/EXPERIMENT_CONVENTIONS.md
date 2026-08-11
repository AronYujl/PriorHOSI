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

`code/train_hoi_prior.py` or `code/test_infbagel_hoi.py` under its Hydra config,
then `tools/paired_bootstrap.py`. On the HOI worker, `tools/hoi_chain.py` runs
all three in sequence so an arm completes without the authority host.

All sixteen P8/P9/P10 registry rows already use exactly these entry points and
reference no per-experiment wrapper. That is the pattern; the eighteen deleted
wrappers were what it replaced. A new arm changes the config and the run id, not
the entry point — which is precisely what makes two arms comparable.

> **Correction, 2026-08-11.** This section previously described the path as
> beginning with `tools/experiment.py start` and claimed the sixteen rows used
> it. The no-wrapper half is correct and verified; the manifest half was not.
> Only 2 of the 16 rows that carry a manifest field have one, and the other 14 —
> every P8, P9 and P10 arm — record `"no tools/experiment.py start manifest
> exists for this arm"`. The rule said A while practice did B for five weeks.
>
> The gate that rule existed for, refusing a dirty worktree, now lives in the
> trainer's own preflight, on the path that actually executes, and
> `tools/hoi_chain.py` applies it again before consuming a run id. A check
> nobody runs is not a check.

## 2b. Provenance is resolved at run start

`metrics.json` is written once, at completion. Resolving `git rev-parse HEAD`
there records whatever commit is checked out hours later, which is how P10's
A10 and A01 arms started at `91232ad` and were recorded as `5d39ac3`. Their
internal validity survived only because `AGENTS.md` forbids editing worker
source mid-run — insulation by rule, not by design.

The trainer now resolves the commit once in `main()` and carries it in the
config to every rank; the live HEAD at completion is recorded beside it as
`git_commit_at_completion`. Any new expert trainer must do the same. With two
experts advancing concurrently on two hosts, the rule that saved P10 no longer
covers the case where the *other* branch commits during a run.

## 2c. Formal recipes are composed, never restated

A formal arm is `defaults: [<base>, recipe: <sealed recipe>, _self_]` plus its
run id and its manipulated factor. Before 2026-08-11 each arm restated the whole
recipe inline: ~52 keys per file, 22 of them byte-identical restatements of the
base config. "Same recipe as the sealed baseline" was maintained by copy-paste
across eleven files, and one missed line silently produces an uncontrolled
comparison. `code/config/recipe/d2ai.yaml` states it once and a test pins every
key, so the guarantee is mechanical.

A recipe file is frozen: changing a value in place retroactively changes what
every future arm is compared against. Add a new recipe file instead, with a
dated plan amendment.


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

Each of the four carries a different thing, and repeating the same prose in all
four costs a re-edit at every closure. The registry row carries the
classification, the selected checkpoint, and pointers to the compact result and
the phase summary, with a judgement of at most two sentences — P10's closing row
carries about 1,800 characters of narrative that also exists in its summary.
Numbers live in the compact JSON. Narrative lives in the phase summary, once.
The evidence index carries one line and a pointer.

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

## 7. A new optimized term is recorded in the same commit that adds it

If a term enters the objective, it enters the training metrics in the same
commit. P8 added the hand-object geometry term without a matching `LOSS_KEYS`
entry, so the term was unobservable in its own sealed run: P10's geometry curve
has three arms instead of four, and W3's cannot be recovered after the fact. An
optimized quantity that is not logged is a quantity you cannot later explain.

## 8. Every reported metric is persisted per sequence

Aggregate means cannot carry a confidence interval. The evaluator computes
`contact_percent` per sequence but writes only the aggregate, so every decision
resting on it — including two of P10's protection criteria — has a point
estimate and no uncertainty, and the fixed 438-sequence entry point cannot be
changed after preregistration to recover it.

Whatever a new expert's evaluator reports, it writes to
`per_sequence_metrics.json` from its first run. Adding a metric later means it
is missing from exactly the baselines you want to compare against.

