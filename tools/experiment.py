#!/usr/bin/env python3
"""Create immutable run manifests and maintain the tracked experiment registry."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


SCHEMA_VERSION = 1
PHASES = {f"p{i}" for i in range(7)}
TERMINAL_STATUSES = {"completed", "failed", "aborted"}
RUN_ID_RE = re.compile(
    r"^p[0-6]-[a-z0-9][a-z0-9._-]*-s[0-9]+-[0-9]{8}$"
)


class ManifestError(RuntimeError):
    """Raised for a reproducibility or schema violation."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def run_command(args: Sequence[str], cwd: Path, check: bool = True) -> str:
    completed = subprocess.run(
        list(args), cwd=str(cwd), text=True, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, check=False,
    )
    if check and completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise ManifestError(f"command failed ({' '.join(args)}): {detail}")
    return completed.stdout.strip()


def find_repo_root(start: Path) -> Path:
    root = run_command(["git", "rev-parse", "--show-toplevel"], start)
    return Path(root).resolve()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> Dict[str, Any]:
    """Hash a file or directory deterministically, including relative names."""
    path = path.resolve()
    if not path.exists():
        raise ManifestError(f"asset does not exist: {path}")
    if path.is_symlink():
        raise ManifestError(f"top-level assets may not be symlinks: {path}")
    if path.is_file():
        return {"kind": "file", "sha256": sha256_file(path), "bytes": path.stat().st_size}

    digest = hashlib.sha256()
    files = 0
    total_bytes = 0
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        if child.is_symlink():
            raise ManifestError(f"asset tree contains a symlink: {child}")
        relative = child.relative_to(path).as_posix().encode("utf-8")
        file_hash = sha256_file(child).encode("ascii")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(file_hash)
        files += 1
        total_bytes += child.stat().st_size
    return {
        "kind": "directory",
        "sha256": digest.hexdigest(),
        "files": files,
        "bytes": total_bytes,
    }


def git_state(repo: Path) -> Dict[str, Any]:
    status = run_command(["git", "status", "--porcelain=v1", "--untracked-files=all"], repo)
    unstaged = run_command(["git", "diff", "--binary", "HEAD"], repo)
    staged = run_command(["git", "diff", "--binary", "--cached", "HEAD"], repo)
    diff_digest = hashlib.sha256((staged + "\0" + unstaged).encode("utf-8")).hexdigest()
    return {
        "commit": run_command(["git", "rev-parse", "HEAD"], repo),
        "branch": run_command(["git", "branch", "--show-current"], repo),
        "dirty": bool(status),
        "status_porcelain": status.splitlines() if status else [],
        "tracked_diff_sha256": diff_digest,
    }


def dependency_snapshot(repo: Path) -> Dict[str, Any]:
    packages: List[str] = []
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            packages.append(f"{name}=={distribution.version}")
    packages.sort(key=str.lower)
    encoded = "\n".join(packages).encode("utf-8")
    result: Dict[str, Any] = {
        "python": sys.version,
        "packages": packages,
        "packages_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    requirements = repo / "requirements.txt"
    if requirements.exists():
        result["requirements_txt_sha256"] = sha256_file(requirements)
    return result


def hardware_snapshot(repo: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_count": os.cpu_count(),
    }
    disk = shutil.disk_usage(repo)
    info["repository_filesystem"] = {
        "path": str(repo), "total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free,
    }
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = run_command(query, repo)
    except (ManifestError, FileNotFoundError):
        info["gpus"] = []
    else:
        info["gpus"] = [line.strip() for line in output.splitlines() if line.strip()]
    process_query = [
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    try:
        processes = run_command(process_query, repo)
    except (ManifestError, FileNotFoundError):
        info["gpu_compute_processes"] = []
    else:
        info["gpu_compute_processes"] = [line.strip() for line in processes.splitlines() if line.strip()]
    try:
        clock = run_command(
            ["timedatectl", "show", "-p", "Timezone", "-p", "NTPSynchronized", "-p", "LocalRTC"], repo,
        )
    except (ManifestError, FileNotFoundError):
        info["clock"] = None
    else:
        info["clock"] = clock.splitlines()
    return info


def atomic_write_json(path: Path, value: Mapping[str, Any], overwrite: bool = False) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise ManifestError(f"refusing to overwrite existing file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def load_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ManifestError(f"expected a JSON object: {path}")
    return value


def parse_assets(values: Iterable[str], repo: Path) -> Dict[str, Any]:
    assets: Dict[str, Any] = {}
    for value in values:
        if "=" not in value:
            raise ManifestError(f"asset must use ROLE=PATH: {value}")
        role, raw_path = value.split("=", 1)
        if not re.fullmatch(r"[a-z][a-z0-9_-]*", role):
            raise ManifestError(f"invalid asset role: {role}")
        if role in assets:
            raise ManifestError(f"duplicate asset role: {role}")
        path = Path(raw_path)
        if not path.is_absolute():
            path = repo / path
        record = sha256_path(path)
        record["path"] = os.path.relpath(path.resolve(), repo)
        assets[role] = record
    return assets


def validate_run_id(run_id: str, phase: str, seed: int) -> None:
    if not RUN_ID_RE.fullmatch(run_id):
        raise ManifestError(
            "run id must match <phase>-<component>-<variant>-s<seed>-<YYYYMMDD>"
        )
    if not run_id.startswith(phase + "-") or f"-s{seed}-" not in run_id:
        raise ManifestError("run id phase/seed does not match --phase/--seed")


def command_start(args: argparse.Namespace) -> None:
    repo = find_repo_root(Path.cwd())
    validate_run_id(args.id, args.phase, args.seed)
    state = git_state(repo)
    if state["dirty"]:
        lines = "\n".join(state["status_porcelain"][:20])
        raise ManifestError(f"training/evaluation requires a clean worktree:\n{lines}")

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = repo / config_path
    if not config_path.is_file():
        raise ManifestError(f"config is not a file: {config_path}")

    output = Path(args.output) if args.output else (
        repo / "results" / "experiments" / args.id / "manifest.json"
    )
    if not output.is_absolute():
        output = repo / output
    run_workdir = Path(args.workdir) if args.workdir else repo
    if not run_workdir.is_absolute():
        run_workdir = repo / run_workdir
    if not run_workdir.is_dir():
        raise ManifestError(f"run working directory does not exist: {run_workdir}")
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": args.id,
        "phase": args.phase,
        "status": "running",
        "seed": args.seed,
        "started_at": utc_now(),
        "ended_at": None,
        "git": state,
        "config": {
            "path": os.path.relpath(config_path.resolve(), repo),
            "sha256": sha256_file(config_path),
            "content": config_path.read_text(encoding="utf-8"),
            "overrides": list(args.override),
        },
        "command": args.run_command,
        "working_directory": os.path.relpath(run_workdir.resolve(), repo),
        "assets": parse_assets(args.asset, repo),
        "hardware": hardware_snapshot(repo),
        "dependencies": dependency_snapshot(repo),
        "metrics": None,
    }
    atomic_write_json(output, manifest)
    print(output)


def _finish_commit_transition(
    repo: Path,
    manifest: Mapping[str, Any],
    current: Mapping[str, Any],
    metrics: Mapping[str, Any],
    args: argparse.Namespace,
) -> Optional[Dict[str, Any]]:
    """Validate and record one explicitly hash-bound completion transition.

    A reportable run normally requires the finishing HEAD to equal the
    manifest's starting commit.  The D2-AB continuation is the narrow
    exception: its already-running manifest began before a governance-only
    resume guard commit, while the completed metrics bind the exact source,
    target, binary diff, and changed-file allowlist.  Keep the normal
    exact-HEAD behavior fail-closed for every other case.
    """
    starting_commit = str(manifest.get("git", {}).get("commit", ""))
    transition_values = (
        args.commit_transition_source,
        args.commit_transition_target,
        args.commit_transition_diff_sha256,
    )
    allow_paths = sorted(set(args.commit_transition_allow_path))
    if current["commit"] == starting_commit:
        if any(value is not None for value in transition_values) or allow_paths:
            raise ManifestError(
                "commit-transition arguments are forbidden when HEAD matches the manifest"
            )
        return None

    if (
        any(value is None for value in transition_values)
        or not allow_paths
    ):
        raise ManifestError(
            "HEAD changed during the run; an explicit hash-bound commit transition is required"
        )
    source = str(args.commit_transition_source)
    target = str(args.commit_transition_target)
    diff_sha256 = str(args.commit_transition_diff_sha256)
    if source != starting_commit:
        raise ManifestError("commit-transition source does not match manifest start commit")
    if target != current["commit"]:
        raise ManifestError("commit-transition target does not match current HEAD")
    if not re.fullmatch(r"[0-9a-f]{40}", source) or not re.fullmatch(r"[0-9a-f]{40}", target):
        raise ManifestError("commit-transition source/target must be lowercase Git object ids")
    if not re.fullmatch(r"[0-9a-f]{64}", diff_sha256):
        raise ManifestError("commit-transition diff must be a lowercase SHA-256")

    diff = subprocess.run(
        ["git", "diff", "--binary", source, target],
        cwd=str(repo),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if diff.returncode:
        detail = diff.stderr.decode("utf-8", errors="replace").strip()
        raise ManifestError(f"cannot inspect commit transition: {detail}")
    actual_diff_sha256 = hashlib.sha256(diff.stdout).hexdigest()
    if actual_diff_sha256 != diff_sha256:
        raise ManifestError("commit-transition binary diff hash mismatch")
    changed = run_command(["git", "diff", "--name-only", source, target], repo).splitlines()
    if changed != allow_paths:
        raise ManifestError(
            f"commit-transition changed-file allowlist mismatch: expected {allow_paths}, got {changed}"
        )

    if metrics.get("git_commit") != target:
        raise ManifestError("metrics git_commit does not match commit-transition target")
    provenance = metrics.get("resume_commit_provenance")
    expected_provenance = {
        "mode": "explicit_bound_transition",
        "checkpoint_git_commit": source,
        "current_git_commit": target,
        "diff_sha256": diff_sha256,
        "changed_paths": changed,
    }
    if not isinstance(provenance, Mapping):
        raise ManifestError("metrics lack resume commit provenance for the HEAD transition")
    for key, expected in expected_provenance.items():
        if provenance.get(key) != expected:
            raise ManifestError(f"metrics resume provenance mismatch for {key}")
    return {
        "mode": "explicit_bound_transition",
        "source_commit": source,
        "target_commit": target,
        "diff_sha256": diff_sha256,
        "changed_paths": changed,
        "metrics_git_commit": metrics.get("git_commit"),
    }


def command_finish(args: argparse.Namespace) -> None:
    repo = find_repo_root(Path.cwd())
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("status") != "running" or manifest.get("ended_at") is not None:
        raise ManifestError("only a running manifest may be finished")
    current = git_state(repo)
    if current["dirty"]:
        raise ManifestError("worktree became dirty during the run")
    metrics_path = Path(args.metrics).resolve()
    metrics = load_json(metrics_path)
    transition = _finish_commit_transition(repo, manifest, current, metrics, args)
    if args.resolved_config:
        resolved_config = Path(args.resolved_config).resolve()
        if not resolved_config.is_file():
            raise ManifestError(f"resolved config is not a file: {resolved_config}")
        manifest["config"]["resolved_path"] = os.path.relpath(resolved_config, repo)
        manifest["config"]["resolved_sha256"] = sha256_file(resolved_config)
        manifest["config"]["resolved_content"] = resolved_config.read_text(encoding="utf-8")
    manifest["status"] = args.status
    manifest["ended_at"] = utc_now()
    manifest["metrics"] = metrics
    manifest["metrics_file"] = {
        "path": os.path.relpath(metrics_path, repo),
        "sha256": sha256_file(metrics_path),
        "bytes": metrics_path.stat().st_size,
    }
    if transition is not None:
        manifest["commit_transition"] = transition
    manifest["final_git"] = current
    atomic_write_json(manifest_path, manifest, overwrite=True)
    print(manifest_path)


def iter_registry(path: Path) -> Iterable[Dict[str, Any]]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"{path}:{line_number}: {exc}") from exc
            if not isinstance(value, dict):
                raise ManifestError(f"{path}:{line_number}: expected object")
            yield value


def validate_registry(path: Path) -> int:
    seen = set()
    count = 0
    for record in iter_registry(path):
        missing = {
            "schema_version", "experiment_id", "phase", "status", "hypothesis",
            "config", "results", "conclusion", "next_action", "created_at",
        } - set(record)
        if missing:
            raise ManifestError(f"{record.get('experiment_id', '<unknown>')} missing {sorted(missing)}")
        experiment_id = record["experiment_id"]
        if experiment_id in seen:
            raise ManifestError(f"duplicate experiment id: {experiment_id}")
        if record["phase"] not in PHASES:
            raise ManifestError(f"invalid phase for {experiment_id}: {record['phase']}")
        seen.add(experiment_id)
        count += 1
    return count


def validate_split(path: Path) -> None:
    split = load_json(path)
    algorithm = split.get("algorithm")
    if algorithm == "omomo-sequence-sha256-seed42-v1":
        if split.get("seed") != 42 or split.get("validation_ratio") != 0.05 or split.get("rounding") != "ceil":
            raise ManifestError(f"HOI split must use seed 42, ratio 0.05, and ceil: {path}")
        train = split.get("train", {})
        validation = split.get("internal_validation", {})
        train_indices = train.get("sequence_indices", [])
        validation_indices = validation.get("sequence_indices", [])
        if set(train_indices) & set(validation_indices):
            raise ManifestError(f"HOI sequence leakage: {path}")
        if len(train_indices) != train.get("sequence_count") or len(validation_indices) != 216:
            raise ManifestError(f"invalid HOI split counts: {path}")
        if len(train_indices) + len(validation_indices) != 4304:
            raise ManifestError(f"HOI split must assign all 4,304 sequences: {path}")
        if split.get("official_test_used_for_selection") is not False:
            raise ManifestError(f"official HOI test may not be used for selection: {path}")
        hashes = split.get("source_hashes", {})
        if not hashes or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in hashes.values()):
            raise ManifestError(f"invalid HOI split source hashes: {path}")
        return
    if algorithm != "scene-family-disjoint-v1":
        raise ManifestError(f"unexpected split algorithm: {path}")
    if split.get("seed") != 42 or split.get("validation_ratio") != 0.2:
        raise ManifestError(f"LINGO split must use seed 42 and ratio 0.2: {path}")
    train = split.get("train", {})
    validation = split.get("validation", {})
    if set(train.get("scene_families", [])) & set(validation.get("scene_families", [])):
        raise ManifestError(f"scene-family leakage: {path}")
    if set(train.get("scenes", [])) & set(validation.get("scenes", [])):
        raise ManifestError(f"scene leakage: {path}")
    mapped_families = set(split.get("scene_to_family", {}).values())
    partitioned = set(train.get("scene_families", [])) | set(validation.get("scene_families", []))
    if mapped_families != partitioned:
        raise ManifestError(f"not every scene family is assigned exactly once: {path}")


def validate_evaluator_config(path: Path) -> None:
    config = load_json(path)
    commit = config.get("upstream_commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ManifestError(f"invalid evaluator commit: {path}")
    files = config.get("files", {})
    if not files or any(not re.fullmatch(r"[0-9a-f]{64}", value) for value in files.values()):
        raise ManifestError(f"invalid evaluator file hashes: {path}")


def validate_effective_batch(value: int, protocol: Mapping[str, Any], allow_extended: bool = False) -> None:
    batch = protocol.get("effective_batch", {})
    candidates = batch.get("default_candidates", [])
    forbidden = batch.get("forbidden_examples", [])
    if value in forbidden:
        raise ManifestError(f"forbidden non-power-of-two effective batch: {value}")
    if value in candidates:
        return
    is_power_of_two = isinstance(value, int) and value > 0 and value & (value - 1) == 0
    if not allow_extended or not is_power_of_two:
        raise ManifestError(
            f"effective batch {value} is not a default tier {candidates}; "
            "extended tiers require a power of two plus a dated amendment"
        )


def validate_training_resource_protocol(path: Path) -> None:
    protocol = load_json(path)
    if protocol.get("schema_version") != 1:
        raise ManifestError(f"unexpected training-resource schema: {path}")
    candidates = protocol.get("effective_batch", {}).get("default_candidates")
    if candidates != [512, 1024, 2048, 3072]:
        raise ManifestError(f"default effective-batch tiers must be 512/1024/2048/3072: {path}")
    for value in candidates:
        validate_effective_batch(value, protocol)
    if 1536 not in protocol.get("effective_batch", {}).get("forbidden_examples", []):
        raise ManifestError(f"protocol must explicitly forbid effective batch 1536: {path}")
    if protocol.get("selection_scope") != "independent_per_expert":
        raise ManifestError(f"training resources must be selected independently per expert: {path}")
    assignment = protocol.get("hardware_assignment", {})
    if assignment.get("hoi", {}).get("gpu_count") != 4 or assignment.get("hsi", {}).get("gpu_count") != 8:
        raise ManifestError(f"HOI/HSI hardware assignment must be 4/8 GPUs: {path}")
    if protocol.get("memory_audit_phase") != {"hoi": "1B", "hsi": "1C"}:
        raise ManifestError(f"memory audits must remain in Phase 1B/1C: {path}")
    if protocol.get("primary_budget_units") != ["processed_windows", "processed_frames"]:
        raise ManifestError(f"unexpected primary training budget units: {path}")
    phase_1b = protocol.get("phase_1b_preregistration", {})
    if phase_1b.get("status") != "locked_before_implementation_and_gpu_workload":
        raise ManifestError(f"Phase 1B resource protocol is not locked: {path}")
    model = phase_1b.get("model", {})
    required_model = {
        "prediction": "clean_motion_x0", "dimension": 232, "window_frames": 16,
        "history_frames": 2, "diffusion_steps": 500, "dim_model": 512,
        "num_heads": 16, "num_layers": 8, "scene_condition": False,
        "initialization": "random_only",
    }
    if model != required_model:
        raise ManifestError(f"unexpected Phase 1B model preregistration: {path}")
    audit = phase_1b.get("memory_audit", {})
    if audit.get("micro_batch_candidates") != [128, 256, 512, 768]:
        raise ManifestError(f"unexpected Phase 1B memory candidates: {path}")
    if audit.get("gradient_accumulation_steps") != 1:
        raise ManifestError(f"Phase 1B memory audit must prefer accumulation one: {path}")
    if audit.get("effective_batch_by_micro_batch") != {
        "128": 512, "256": 1024, "512": 2048, "768": 3072,
    }:
        raise ManifestError(f"invalid Phase 1B batch mapping: {path}")
    if (
        audit.get("selected_micro_batch_per_gpu") != 768
        or audit.get("selected_effective_batch_size") != 3072
        or audit.get("selected_gradient_accumulation_steps") != 1
    ):
        raise ManifestError(f"unexpected Phase 1B selected training resources: {path}")
    screening = phase_1b.get("screening", {})
    if screening.get("processed_windows_per_candidate") != 3145728:
        raise ManifestError(f"unexpected Phase 1B screening budget: {path}")
    if screening.get("validation_windows") != 32768 or screening.get("seed") != 42:
        raise ManifestError(f"unexpected Phase 1B screening validation protocol: {path}")
    if screening.get("optimizer_updates_per_candidate") != 1024:
        raise ManifestError(f"unexpected Phase 1B screening update budget: {path}")
    if (
        screening.get("selected_candidate") != "lr3e-4-warm1572864"
        or screening.get("selected_run") != "p1-hoi-screen-lr3e4-s42-20260713"
    ):
        raise ManifestError(f"unexpected Phase 1B screening selection: {path}")
    if (
        screening.get("validation_cadence_windows") != 3145728
        or screening.get("validation_cadence_updates") != 1024
        or screening.get("checkpoint_cadence_windows") != 3145728
        or screening.get("checkpoint_cadence_updates") != 1024
    ):
        raise ManifestError(f"unexpected Phase 1B screening cadence: {path}")
    formal = phase_1b.get("formal_training", {})
    if formal.get("seeds") != [42]:
        raise ManifestError(f"Phase 1B requires the global single seed 42: {path}")
    if formal.get("processed_windows_per_seed") != 61440000:
        raise ManifestError(f"unexpected Phase 1B formal window budget: {path}")
    if formal.get("processed_frames_per_seed") != 983040000:
        raise ManifestError(f"unexpected Phase 1B formal frame budget: {path}")
    if formal.get("optimizer_updates_per_seed") != 20000:
        raise ManifestError(f"unexpected Phase 1B formal update budget: {path}")
    if (
        formal.get("learning_rate") != 0.0003
        or formal.get("warmup_windows") != 1572864
        or formal.get("warmup_updates") != 512
    ):
        raise ManifestError(f"unexpected Phase 1B selected optimization: {path}")

    remediation = protocol.get("phase_1b_remediation_preregistration", {})
    if remediation.get("status") != "locked_before_remediation_implementation_or_gpu_workload":
        raise ManifestError(f"Phase 1B remediation protocol is not locked: {path}")
    if remediation.get("date") != "2026-07-14" or remediation.get("seed") != 42:
        raise ManifestError(f"unexpected Phase 1B remediation date/seed: {path}")
    trigger = remediation.get("trigger", {})
    if trigger.get("failed_gate_sha256") != (
        "77fda0bff85a4c5d44cf3e47cd9e3f6951554bae4cde38a01ebbf56c5f38b251"
    ):
        raise ManifestError(f"Phase 1B remediation must reference the immutable failed gate: {path}")
    planning_evidence = remediation.get("planning_evidence", {})
    if (
        planning_evidence.get("hoi_train_windows") != 597868
        or planning_evidence.get("hoi_train_need_pelvis_windows") != 597868
        or planning_evidence.get("current_pelvis_goal_slot_nonzero_windows") != 0
        or planning_evidence.get("released_baseline_object_surface_weight") != 50.0
        or planning_evidence.get("current_failed_prior_object_surface_weight") != 0.0
    ):
        raise ManifestError(f"unexpected Phase 1B planning evidence: {path}")
    scope = remediation.get("scope", {})
    expected_architecture = {
        "dimension": 232, "window_frames": 16, "history_frames": 2,
        "diffusion_steps": 500, "dim_model": 512, "num_heads": 16,
        "num_layers": 8,
    }
    if scope.get("architecture_unchanged") != expected_architecture:
        raise ManifestError(f"Phase 1B remediation may not change the architecture: {path}")
    if scope.get("scene_condition") is not False or scope.get("initialization") != "random_only":
        raise ManifestError(f"Phase 1B remediation violated scene/init scope: {path}")
    diagnostics = remediation.get("diagnostics", {})
    if diagnostics.get("official_test_used_for_selection") is not False:
        raise ManifestError(f"official HOI test may not select remediation changes: {path}")
    if diagnostics.get("rollout_sequences") != 128 or diagnostics.get("rollout_windows_per_sequence") != 3:
        raise ManifestError(f"unexpected Phase 1B remediation rollout protocol: {path}")
    if diagnostics.get("teacher_forced_timesteps") != [0, 1, 10, 50, 100, 250, 499]:
        raise ManifestError(f"unexpected Phase 1B timestep diagnostic: {path}")
    if diagnostics.get("existing_checkpoint_weight_variants") != ["online", "ema_0.9999"]:
        raise ManifestError(f"unexpected Phase 1B existing checkpoint variants: {path}")
    repairs = remediation.get("representation_repairs", {})
    if (
        repairs.get("pelvis_goal_legacy_replay_max_abs") != 0.00001
        or "existing goals[:3]" not in repairs.get("pelvis_goal_condition", "")
    ):
        raise ManifestError(f"Phase 1B must restore the existing pelvis-goal slot: {path}")
    if repairs.get("object_goal_loss_mask") != "end_pi == seq_length":
        raise ManifestError(f"Phase 1B goal loss must be terminal-only: {path}")
    if repairs.get("bps_gt_replay_max_abs") != 0.0001:
        raise ManifestError(f"unexpected Phase 1B BPS replay tolerance: {path}")
    if repairs.get("loss_weights") != {
        "fk": 50.0, "object_surface": 50.0, "velocity": 0.1,
        "terminal_object_goal": 1.0,
    }:
        raise ManifestError(f"unexpected Phase 1B remediation loss weights: {path}")
    remediation_screen = remediation.get("screening", {})
    if (
        remediation_screen.get("processed_windows_per_candidate") != 6144000
        or remediation_screen.get("processed_frames_per_candidate") != 98304000
        or remediation_screen.get("official_test_and_chois_forbidden_for_selection") is not True
        or remediation_screen.get("no_eligible_candidate_action") != (
            "apply the single registered D2-G fallback only when its exact trigger holds; "
            "otherwise stop Phase 1B and require a new dated amendment; never promote the "
            "least-bad ineligible candidate"
        )
    ):
        raise ManifestError(f"unexpected Phase 1B remediation screening budget: {path}")
    if remediation_screen.get("checkpoint_weight_variants") != [
        "online", "ema_0.999", "ema_0.9999",
    ]:
        raise ManifestError(f"unexpected Phase 1B screening checkpoint variants: {path}")
    expected_candidates = [
        {
            "name": "R-1024", "micro_batch_per_gpu": 256, "gpu_count": 4,
            "gradient_accumulation_steps": 1, "effective_batch_size": 1024,
            "peak_learning_rate": 0.0001, "warmup_windows": 1572864,
            "warmup_updates": 1536, "optimizer_updates": 6000,
        },
        {
            "name": "R-3072", "micro_batch_per_gpu": 768, "gpu_count": 4,
            "gradient_accumulation_steps": 1, "effective_batch_size": 3072,
            "peak_learning_rate": 0.0003, "warmup_windows": 1572864,
            "warmup_updates": 512, "optimizer_updates": 2000,
        },
    ]
    if remediation_screen.get("candidates") != expected_candidates:
        raise ManifestError(f"unexpected Phase 1B remediation candidates: {path}")
    for candidate in expected_candidates:
        validate_effective_batch(candidate["effective_batch_size"], protocol)
        effective = (
            candidate["micro_batch_per_gpu"] * candidate["gpu_count"]
            * candidate["gradient_accumulation_steps"]
        )
        if effective != candidate["effective_batch_size"]:
            raise ManifestError(f"invalid remediation effective batch: {candidate}")
        if remediation_screen["processed_windows_per_candidate"] // effective != candidate["optimizer_updates"]:
            raise ManifestError(f"invalid remediation optimizer-update budget: {candidate}")
    fallback = remediation.get("conditional_geometry_fallback", {})
    if (
        fallback.get("maximum_candidates") != 1
        or fallback.get("external_interface_unchanged") is not True
        or fallback.get("must_pass_full_d2_eligibility") is not True
        or fallback.get("object_surface_inherited_from_core") is not True
        or fallback.get("contact_geometry_channels") != [0, 1]
        or fallback.get("trigger") != (
            "all D2 eligibility conditions pass, including object and pelvis ratios both "
            "<=0.70, except contact F1"
        )
    ):
        raise ManifestError(f"Phase 1B geometry fallback must be single and interface-preserving: {path}")
    remediation_formal = remediation.get("formal_training", {})
    if (
        remediation_formal.get("seeds") != [42]
        or remediation_formal.get("processed_windows") != 61440000
        or remediation_formal.get("processed_frames") != 983040000
        or remediation_formal.get("screening_checkpoint_initialization") is not False
        or remediation_formal.get("official_test_runs_after_lock") != {"native": 1, "chois": 1}
    ):
        raise ManifestError(f"unexpected Phase 1B remediation formal protocol: {path}")


def command_register(args: argparse.Namespace) -> None:
    repo = find_repo_root(Path.cwd())
    manifest = load_json(Path(args.manifest).resolve())
    if manifest.get("status") not in TERMINAL_STATUSES:
        raise ManifestError("only terminal manifests may be registered")
    registry = Path(args.registry) if args.registry else repo / "experiments" / "registry.jsonl"
    if not registry.is_absolute():
        registry = repo / registry
    existing = {record["experiment_id"] for record in iter_registry(registry)}
    if manifest.get("experiment_id") in existing:
        raise ManifestError(f"experiment already registered: {manifest.get('experiment_id')}")
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": manifest["experiment_id"],
        "phase": manifest["phase"],
        "status": manifest["status"],
        "hypothesis": args.hypothesis,
        "config": manifest["config"],
        "results": manifest["metrics"],
        "conclusion": args.conclusion,
        "next_action": args.next_action,
        "manifest_sha256": sha256_file(Path(args.manifest).resolve()),
        "git_commit": (
            manifest.get("commit_transition", {}).get(
                "target_commit", manifest["git"]["commit"]
            )
        ),
        "created_at": utc_now(),
    }
    if manifest.get("commit_transition") is not None:
        record["manifest_start_git_commit"] = manifest["git"]["commit"]
        record["commit_transition"] = manifest["commit_transition"]
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    validate_registry(registry)
    print(registry)


def command_validate(args: argparse.Namespace) -> None:
    repo = find_repo_root(Path.cwd())
    registry = Path(args.registry) if args.registry else repo / "experiments" / "registry.jsonl"
    if not registry.is_absolute():
        registry = repo / registry
    count = validate_registry(registry)
    split_paths = sorted((repo / "experiments" / "splits").glob("*.json"))
    for path in split_paths:
        validate_split(path)
    evaluator_paths = sorted((repo / "experiments" / "evaluators").glob("*.json"))
    for path in evaluator_paths:
        validate_evaluator_config(path)
    training_protocol_paths = sorted((repo / "experiments").glob("training_resource_protocol.json"))
    for path in training_protocol_paths:
        validate_training_resource_protocol(path)
    print(
        f"valid research metadata: {count} registry records, "
        f"{len(split_paths)} splits, {len(evaluator_paths)} evaluators, "
        f"{len(training_protocol_paths)} training protocols"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="create a manifest; requires clean Git")
    start.add_argument("--id", required=True)
    start.add_argument("--phase", choices=sorted(PHASES), required=True)
    start.add_argument("--seed", type=int, required=True)
    start.add_argument("--config", required=True)
    start.add_argument("--asset", action="append", default=[], metavar="ROLE=PATH")
    start.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    start.add_argument("--command", dest="run_command")
    start.add_argument("--workdir", default=".")
    start.add_argument("--output")
    start.set_defaults(func=command_start)

    finish = subparsers.add_parser("finish", help="seal a running manifest")
    finish.add_argument("--manifest", required=True)
    finish.add_argument("--metrics", required=True)
    finish.add_argument("--status", choices=sorted(TERMINAL_STATUSES), required=True)
    finish.add_argument("--resolved-config")
    finish.add_argument("--commit-transition-source")
    finish.add_argument("--commit-transition-target")
    finish.add_argument("--commit-transition-diff-sha256")
    finish.add_argument("--commit-transition-allow-path", action="append", default=[])
    finish.set_defaults(func=command_finish)

    register = subparsers.add_parser("register", help="append a sealed run to registry")
    register.add_argument("--manifest", required=True)
    register.add_argument("--registry")
    register.add_argument("--hypothesis", required=True)
    register.add_argument("--conclusion", required=True)
    register.add_argument("--next-action", required=True)
    register.set_defaults(func=command_register)

    validate = subparsers.add_parser("validate", help="validate the append-only registry")
    validate.add_argument("--registry")
    validate.set_defaults(func=command_validate)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except ManifestError as exc:
        parser.exit(2, f"error: {exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
