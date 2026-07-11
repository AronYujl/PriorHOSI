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
    query = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = run_command(query, repo)
    except (ManifestError, FileNotFoundError):
        info["gpus"] = []
    else:
        info["gpus"] = [line.strip() for line in output.splitlines() if line.strip()]
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
        },
        "assets": parse_assets(args.asset, repo),
        "hardware": hardware_snapshot(repo),
        "dependencies": dependency_snapshot(repo),
        "metrics": None,
    }
    atomic_write_json(output, manifest)
    print(output)


def command_finish(args: argparse.Namespace) -> None:
    repo = find_repo_root(Path.cwd())
    manifest_path = Path(args.manifest).resolve()
    manifest = load_json(manifest_path)
    if manifest.get("status") != "running" or manifest.get("ended_at") is not None:
        raise ManifestError("only a running manifest may be finished")
    current = git_state(repo)
    if current["commit"] != manifest.get("git", {}).get("commit"):
        raise ManifestError("HEAD changed during the run")
    if current["dirty"]:
        raise ManifestError("worktree became dirty during the run")
    metrics = load_json(Path(args.metrics).resolve())
    manifest["status"] = args.status
    manifest["ended_at"] = utc_now()
    manifest["metrics"] = metrics
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
    if split.get("algorithm") != "scene-family-disjoint-v1":
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
        "git_commit": manifest["git"]["commit"],
        "created_at": utc_now(),
    }
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
    print(
        f"valid research metadata: {count} registry records, "
        f"{len(split_paths)} splits, {len(evaluator_paths)} evaluators"
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
    start.add_argument("--output")
    start.set_defaults(func=command_start)

    finish = subparsers.add_parser("finish", help="seal a running manifest")
    finish.add_argument("--manifest", required=True)
    finish.add_argument("--metrics", required=True)
    finish.add_argument("--status", choices=sorted(TERMINAL_STATUSES), required=True)
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
