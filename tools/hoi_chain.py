#!/usr/bin/env python3
"""Run one HOIPrior arm end to end on the execution worker.

The 8-GPU authority host owns HSIPrior and is expected to be busy with it, so a
HOIPrior arm must not need it: no checkpoint leaves the worker, and evaluation
runs where the weights already are. The authority host dispatches an arm and
collects the compact results.

    train -> evaluate -> paired bootstrap

Every stage uses the generic lifecycle path -- the trainer and
``code/test_infbagel_hoi.py`` under their Hydra configs, then
``tools/paired_bootstrap.py`` -- so two arms stay comparable by construction.
This tool adds orchestration and no science: it selects no checkpoint by
quality, computes no metric, and changes no protocol.

What it deliberately does not do:

* it never commits, tags, or appends to ``experiments/registry.jsonl``;
* it writes only under the run directory, which is git-ignored, so a chain in
  flight can never dirty the worktree out from under the trainer's own gate;
* it never reuses a run id or overwrites a stage that already succeeded.

Usage on the worker, detached, after the user has approved the arm::

    cd ~/data/work/InfBaGel-release
    nohup "$INFBAGEL_PYTHON" tools/hoi_chain.py --arm p9w3 \\
        --baseline-eval p1-hoi-p8-eval-h0-guided-s42-20260806 \\
        > /dev/null 2>&1 &

Progress, per-stage status and logs land under
``results/experiments/<train-run-id>/chain/``.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

RUN_ID_RE = re.compile(r"^p[0-6]-[a-z0-9][a-z0-9._-]*-s[0-9]+-[0-9]{8}$")
STAGES = ("train", "evaluate", "bootstrap")
DEFAULT_HOST = "node01"
CHECKPOINT_RE = re.compile(r"_windows(\d{9})\.pth$")


class ChainError(RuntimeError):
    """A refusal that must stop the chain before it consumes anything."""


# --------------------------------------------------------------------------- #
# identities
# --------------------------------------------------------------------------- #

def arm_config_name(arm: str) -> str:
    return arm if arm.startswith("config_train_hoi_prior") else f"config_train_hoi_prior_{arm}"


def read_run_id(repo: Path, config_name: str) -> str:
    """The arm's run id, taken from its resolved config -- never invented here."""
    path = repo / "code" / "config" / f"{config_name}.yaml"
    if not path.is_file():
        raise ChainError(f"no such arm config: {path}")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("run_id:"):
            value = line.split(":", 1)[1].strip().strip('"').strip("'")
            if value and value != "null":
                return value
    raise ChainError(
        f"{path.name} does not state a run_id; a reportable arm must name its run"
    )


def derive_eval_run_id(train_run_id: str, today: str, tag: str = "guided") -> str:
    """``...-<variant>-s42-<date>`` -> ``...-<variant>-eval-<tag>-s42-<today>``.

    Follows the P8/P9/P10 naming, where every evaluation run id carries its own
    date and the sampler condition it was produced under.
    """
    match = re.fullmatch(r"(?P<stem>.+)-s(?P<seed>[0-9]+)-(?P<date>[0-9]{8})", train_run_id)
    if match is None:
        raise ChainError(f"cannot derive an evaluation run id from {train_run_id!r}")
    return f"{match.group('stem')}-eval-{tag}-s{match.group('seed')}-{today}"


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ChainError(
            f"run id {run_id!r} must match <phase>-<component>-<variant>-s<seed>-<YYYYMMDD>"
        )


# --------------------------------------------------------------------------- #
# preflight
# --------------------------------------------------------------------------- #

def git_status_porcelain(repo: Path) -> List[str]:
    output = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=repo, text=True
    )
    return output.splitlines()


def git_head(repo: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()


def free_gpu_count() -> Optional[int]:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return sum(1 for line in output.splitlines() if line.strip() and int(line) <= 128)


def preflight(repo: Path, expected_host: Optional[str], required_gpus: int) -> Dict[str, object]:
    """Fail before a run id is consumed, not eight hours into it."""
    host = socket.gethostname()
    if expected_host and host != expected_host:
        raise ChainError(
            f"this chain is meant for {expected_host}; running on {host}. Pass "
            "--expected-host to override deliberately."
        )
    status = git_status_porcelain(repo)
    if status:
        listing = "\n".join(status[:20])
        raise ChainError(
            "the chain requires a clean worktree; the trainer would refuse anyway "
            f"and the evaluator has no such gate:\n{listing}"
        )
    free = free_gpu_count()
    if free is not None and free < required_gpus:
        raise ChainError(
            f"{required_gpus} idle GPUs required, {free} idle. Another workload owns "
            "this host; a shared GPU invalidates the recorded throughput."
        )
    return {
        "host": host,
        "git_head": git_head(repo),
        "worktree_clean": True,
        "idle_gpus": free,
        "required_gpus": required_gpus,
    }


# --------------------------------------------------------------------------- #
# stage bookkeeping
# --------------------------------------------------------------------------- #

def chain_dir(repo: Path, train_run_id: str) -> Path:
    return repo / "results" / "experiments" / train_run_id / "chain"


def stage_status_path(repo: Path, train_run_id: str, stage: str) -> Path:
    return chain_dir(repo, train_run_id) / f"{stage}.json"


def stage_completed(repo: Path, train_run_id: str, stage: str) -> bool:
    path = stage_status_path(repo, train_run_id, stage)
    if not path.is_file():
        return False
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("status") == "completed"
    except (ValueError, OSError):
        return False


def write_stage_status(repo: Path, train_run_id: str, stage: str, value: Dict[str, object]) -> Path:
    path = stage_status_path(repo, train_run_id, stage)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    return path


def utc_now() -> str:
    return _datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def final_checkpoint(repo: Path, train_run_id: str) -> Path:
    """The highest-window checkpoint: the fixed final identity, not a selection."""
    directory = repo / "results" / "experiments" / train_run_id / "checkpoints"
    if not directory.is_dir():
        raise ChainError(f"no checkpoint directory for {train_run_id}: {directory}")
    best: Optional[Path] = None
    best_windows = -1
    for path in directory.glob(f"{train_run_id}_windows*.pth"):
        match = CHECKPOINT_RE.search(path.name)
        if match is None:
            continue
        windows = int(match.group(1))
        if windows > best_windows:
            best, best_windows = path, windows
    if best is None:
        raise ChainError(f"no {train_run_id}_windows*.pth checkpoint under {directory}")
    return best


def train_command(python: str, config_name: str) -> List[str]:
    return [python, "code/train_hoi_prior.py", f"--config-name={config_name}"]


def evaluate_command(python: str, eval_run_id: str, checkpoint: Path) -> List[str]:
    return [
        python,
        "code/test_infbagel_hoi.py",
        "--config-name=config_eval_hoi_prior",
        f"exp_name={eval_run_id}",
        f"ckpt_path={checkpoint}",
    ]


def bootstrap_command(python: str, baseline: str, arm: str, output: Path) -> List[str]:
    return [
        python,
        "tools/paired_bootstrap.py",
        "--a", baseline,
        "--b", arm,
        "--output", str(output),
    ]


def run_stage(
    repo: Path,
    train_run_id: str,
    stage: str,
    command: Sequence[str],
    extra: Optional[Dict[str, object]] = None,
    dry_run: bool = False,
) -> Dict[str, object]:
    record: Dict[str, object] = {
        "stage": stage,
        "command": list(command),
        "started_at": utc_now(),
        "status": "running",
    }
    record.update(extra or {})
    log_path = chain_dir(repo, train_run_id) / f"{stage}.log"
    record["log"] = str(log_path.relative_to(repo))
    write_stage_status(repo, train_run_id, stage, record)
    print(f"[chain] {stage}: {' '.join(shlex.quote(part) for part in command)}", flush=True)
    if dry_run:
        record.update(status="skipped", reason="dry-run", ended_at=utc_now())
        write_stage_status(repo, train_run_id, stage, record)
        return record

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as handle:
        completed = subprocess.run(
            list(command), cwd=repo, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    record["returncode"] = completed.returncode
    record["ended_at"] = utc_now()
    record["status"] = "completed" if completed.returncode == 0 else "failed"
    write_stage_status(repo, train_run_id, stage, record)
    if completed.returncode != 0:
        raise ChainError(
            f"stage {stage} failed with return code {completed.returncode}; see {log_path}. "
            "The failure is retained: do not reuse this run id."
        )
    return record


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--arm", required=True,
                        help="arm name, e.g. p9w3, or a full config_train_hoi_prior_* name")
    parser.add_argument("--baseline-eval", default=None,
                        help="evaluation run id to bootstrap this arm against; "
                             "omit to stop after evaluation")
    parser.add_argument("--eval-run-id", default=None,
                        help="override the derived evaluation run id")
    parser.add_argument("--eval-tag", default="guided",
                        help="sampler condition recorded in the evaluation run id")
    parser.add_argument("--stages", default=",".join(STAGES),
                        help=f"comma-separated subset of {','.join(STAGES)}")
    parser.add_argument("--python", default=os.environ.get("INFBAGEL_PYTHON", sys.executable))
    parser.add_argument("--expected-host", default=DEFAULT_HOST,
                        help="refuse to run elsewhere; pass an empty value to disable")
    parser.add_argument("--required-gpus", type=int, default=4)
    parser.add_argument("--date", default=None,
                        help="YYYYMMDD stamped into the evaluation run id (default: today)")
    parser.add_argument("--dry-run", action="store_true",
                        help="resolve every identity and print the commands without running")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    repo = REPO
    stages = [stage.strip() for stage in args.stages.split(",") if stage.strip()]
    unknown = [stage for stage in stages if stage not in STAGES]
    if unknown:
        raise ChainError(f"unknown stage(s): {unknown}; valid stages are {list(STAGES)}")

    config_name = arm_config_name(args.arm)
    train_run_id = read_run_id(repo, config_name)
    validate_run_id(train_run_id)
    today = args.date or _datetime.date.today().strftime("%Y%m%d")
    eval_run_id = args.eval_run_id or derive_eval_run_id(train_run_id, today, args.eval_tag)
    validate_run_id(eval_run_id)

    state = preflight(
        repo,
        args.expected_host or None,
        args.required_gpus if "train" in stages else 1,
    )
    header = {
        "train_run_id": train_run_id,
        "eval_run_id": eval_run_id,
        "config": f"code/config/{config_name}.yaml",
        "baseline_eval": args.baseline_eval,
        "stages": stages,
        "preflight": state,
        "created_at": utc_now(),
    }
    write_stage_status(repo, train_run_id, "chain", header)
    print(json.dumps(header, indent=2, sort_keys=True), flush=True)

    if "train" in stages:
        if stage_completed(repo, train_run_id, "train"):
            print("[chain] train already completed; not rerunning", flush=True)
        else:
            run_stage(repo, train_run_id, "train",
                      train_command(args.python, config_name), dry_run=args.dry_run)

    checkpoint = None
    if "evaluate" in stages:
        if stage_completed(repo, train_run_id, "evaluate"):
            print("[chain] evaluate already completed; not rerunning", flush=True)
        else:
            checkpoint = (
                repo / "results" / "experiments" / train_run_id / "checkpoints" / "<final>.pth"
                if args.dry_run and not (
                    repo / "results" / "experiments" / train_run_id / "checkpoints"
                ).is_dir()
                else final_checkpoint(repo, train_run_id)
            )
            run_stage(repo, train_run_id, "evaluate",
                      evaluate_command(args.python, eval_run_id, checkpoint),
                      extra={"eval_run_id": eval_run_id, "checkpoint": str(checkpoint)},
                      dry_run=args.dry_run)

    if "bootstrap" in stages:
        if not args.baseline_eval:
            print("[chain] no --baseline-eval given; stopping after evaluation", flush=True)
        elif stage_completed(repo, train_run_id, "bootstrap"):
            print("[chain] bootstrap already completed; not rerunning", flush=True)
        else:
            output = chain_dir(repo, train_run_id) / f"bootstrap_{eval_run_id}.json"
            run_stage(repo, train_run_id, "bootstrap",
                      bootstrap_command(args.python, args.baseline_eval, eval_run_id, output),
                      extra={"baseline_eval": args.baseline_eval, "output": str(output)},
                      dry_run=args.dry_run)

    print(f"[chain] done; status under {chain_dir(repo, train_run_id)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ChainError as error:
        print(f"[chain] REFUSED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2)
