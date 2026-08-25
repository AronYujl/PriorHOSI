#!/usr/bin/env python3
"""Dispatch a pinned checkpoint-to-motion job to the dedicated worker2 host.

The only required scientific input to ``start`` is a trusted authority-side
checkpoint.  All bulk transfers are initiated by worker2 over the campus LAN;
the authority uses the loopback reverse tunnel for control only.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
PROFILE_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")
SAFE_RELATIVE_ASSET = re.compile(r"[A-Za-z0-9._/-]+")
PROTECTED_OVERRIDES = (
    "exp_name=",
    "ckpt_path=",
    "checkpoint_sha256=",
    "checkpoint_weight_variant=",
    "save_motion_params=",
    "hoi_sequence_limit=",
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROFILES = REPO_ROOT / "tools/visualization/worker2_profiles.json"
DEFAULT_ARTIFACT_ROOT = Path(
    "/data/yujinlun/InfBaGel-visualization-artifacts/worker2-inference"
)

TOPOLOGY: Dict[str, Any] = {
    "authority_host": "10.184.17.253",
    "authority_repo": "/data/yujinlun/InfBaGel-release",
    "authority_user": "yujinlun",
    "control_host": "127.0.0.1",
    "control_port": 22216,
    "control_user": "yujinlun",
    "control_identity": "/home/yujinlun/.ssh/id_ed25519",
    "control_known_hosts": "/home/yujinlun/.ssh/known_hosts_infbagel_worker2",
    "worker_root": "/data2/yujinlun/infbagel-inference",
    "worker_base_repo": "/data2/yujinlun/infbagel-inference/work/InfBaGel-release",
    "worker_python": "/data2/yujinlun/infbagel-inference/env/infbagel/bin/python",
    "worker_transfer_identity": (
        "/home/yujinlun/.ssh/id_ed25519_infbagel_authority_transfer"
    ),
    "worker_data": "/data2/yujinlun/infbagel-inference/data/InfBaGel",
    "worker_smpl_models": "/data2/yujinlun/infbagel-inference/data/smpl_models",
}


class WorkflowError(RuntimeError):
    """Raised when the workflow cannot preserve its fail-closed contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_sha256(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> Dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise WorkflowError("cannot read JSON %s: %s" % (path, exc)) from exc
    if not isinstance(value, dict):
        raise WorkflowError("expected a JSON object: %s" % path)
    return value


def _require_relative_asset(value: Any, profile_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise WorkflowError("profile %s has an invalid required asset" % profile_name)
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or SAFE_RELATIVE_ASSET.fullmatch(value) is None:
        raise WorkflowError("profile %s asset must be checkout-relative: %s" % (profile_name, value))
    return value


def validate_profile(raw: Mapping[str, Any]) -> Dict[str, Any]:
    profile = dict(raw)
    name = profile.get("name")
    if not isinstance(name, str) or not PROFILE_NAME.fullmatch(name):
        raise WorkflowError("invalid profile name: %r" % name)
    commit = profile.get("inference_commit")
    if not isinstance(commit, str) or not HEX40.fullmatch(commit):
        raise WorkflowError("profile %s has an invalid inference commit" % name)
    entrypoint = profile.get("entrypoint")
    if not isinstance(entrypoint, str) or not entrypoint.startswith("code/"):
        raise WorkflowError("profile %s entrypoint must be under code/" % name)
    if Path(entrypoint).is_absolute() or ".." in Path(entrypoint).parts:
        raise WorkflowError("profile %s has an unsafe entrypoint" % name)
    config_name = profile.get("config_name")
    if not isinstance(config_name, str) or not re.fullmatch(r"[a-zA-Z0-9_-]+", config_name):
        raise WorkflowError("profile %s has an invalid config name" % name)
    run_variant = profile.get("run_variant")
    if not isinstance(run_variant, str) or not RUN_ID.fullmatch(run_variant):
        raise WorkflowError("profile %s has an invalid run variant" % name)
    expected = profile.get("expected_sequences")
    if not isinstance(expected, int) or isinstance(expected, bool) or expected <= 0:
        raise WorkflowError("profile %s expected_sequences must be positive" % name)
    if profile.get("legacy_human_frame") not in ("y_up", "z_up"):
        raise WorkflowError("profile %s has an invalid legacy human frame" % name)
    match = profile.get("match")
    if not isinstance(match, dict) or not match:
        raise WorkflowError("profile %s must define match rules" % name)
    if "run_id_regex" in match:
        try:
            re.compile(str(match["run_id_regex"]))
        except re.error as exc:
            raise WorkflowError("profile %s has an invalid run_id_regex" % name) from exc
    overrides = profile.get("hydra_overrides")
    if not isinstance(overrides, list) or not all(isinstance(item, str) for item in overrides):
        raise WorkflowError("profile %s hydra_overrides must be strings" % name)
    for override in overrides:
        if not override or any(override.startswith(prefix) for prefix in PROTECTED_OVERRIDES):
            raise WorkflowError("profile %s overrides protected field: %s" % (name, override))
    assets = profile.get("required_assets")
    if not isinstance(assets, list) or not assets:
        raise WorkflowError("profile %s must declare required assets" % name)
    profile["required_assets"] = [_require_relative_asset(item, name) for item in assets]
    return profile


def load_profiles(path: Path = DEFAULT_PROFILES) -> Tuple[List[Dict[str, Any]], str]:
    registry = read_json(path)
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise WorkflowError("unsupported profile registry schema")
    raw_profiles = registry.get("profiles")
    if not isinstance(raw_profiles, list) or not raw_profiles:
        raise WorkflowError("profile registry must contain profiles")
    profiles = [validate_profile(item) for item in raw_profiles if isinstance(item, dict)]
    if len(profiles) != len(raw_profiles):
        raise WorkflowError("every profile registry entry must be an object")
    names = [profile["name"] for profile in profiles]
    if len(names) != len(set(names)):
        raise WorkflowError("duplicate profile name")
    return profiles, sha256_file(path)


def profile_matches(profile: Mapping[str, Any], metadata: Mapping[str, Any]) -> bool:
    for key, expected in profile["match"].items():
        if key == "run_id_regex":
            value = metadata.get("run_id")
            if not isinstance(value, str) or re.search(str(expected), value) is None:
                return False
        elif metadata.get(key) != expected:
            return False
    return True


def select_profile(
    profiles: Sequence[Mapping[str, Any]],
    metadata: Mapping[str, Any],
    requested: str = "auto",
) -> Dict[str, Any]:
    if requested != "auto":
        candidates = [profile for profile in profiles if profile["name"] == requested]
        if not candidates:
            raise WorkflowError("unknown inference profile: %s" % requested)
        if not profile_matches(candidates[0], metadata):
            raise WorkflowError("checkpoint does not satisfy profile %s" % requested)
        return dict(candidates[0])
    matches = [profile for profile in profiles if profile_matches(profile, metadata)]
    if not matches:
        raise WorkflowError(
            "no registered inference profile matches checkpoint type/run_id; "
            "register and test one instead of guessing an inference commit"
        )
    if len(matches) != 1:
        raise WorkflowError("checkpoint matches multiple inference profiles")
    return dict(matches[0])


def load_checkpoint_metadata(path: Path) -> Dict[str, Any]:
    if not path.is_absolute():
        raise WorkflowError("checkpoint path must be absolute")
    if not path.is_file():
        raise WorkflowError("checkpoint does not exist: %s" % path)
    try:
        import torch
    except ImportError as exc:
        raise WorkflowError("trusted checkpoint inspection requires the verified infbagel environment") from exc
    try:
        payload = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise WorkflowError("cannot inspect trusted checkpoint: %s" % exc) from exc
    if not isinstance(payload, dict):
        raise WorkflowError("checkpoint root must be a mapping")
    keys = (
        "schema_version",
        "checkpoint_type",
        "expert",
        "run_id",
        "seed",
        "git_commit",
        "primary_weight_variant",
        "data_contract_sha256",
        "processed_windows",
        "processed_frames",
    )
    metadata = {key: payload.get(key) for key in keys}
    required = ("schema_version", "checkpoint_type", "expert", "run_id", "seed", "git_commit")
    missing = [key for key in required if metadata.get(key) is None]
    if missing:
        raise WorkflowError("checkpoint metadata is missing: %s" % ", ".join(missing))
    if not isinstance(metadata["git_commit"], str) or not HEX40.fullmatch(metadata["git_commit"]):
        raise WorkflowError("checkpoint training git_commit is invalid")
    if not isinstance(metadata["seed"], int) or isinstance(metadata["seed"], bool):
        raise WorkflowError("checkpoint seed is invalid")
    return metadata


def control_ssh_argv(topology: Mapping[str, Any] = TOPOLOGY) -> List[str]:
    return [
        "/usr/bin/ssh",
        "-F", "/dev/null",
        "-p", str(topology["control_port"]),
        "-i", str(topology["control_identity"]),
        "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
        "-o", "UserKnownHostsFile=%s" % topology["control_known_hosts"],
        "-o", "ConnectTimeout=10",
        "%s@%s" % (topology["control_user"], topology["control_host"]),
    ]


def transfer_ssh_words(topology: Mapping[str, Any] = TOPOLOGY) -> List[str]:
    return [
        "/usr/bin/ssh",
        "-F", "/dev/null",
        "-i", str(topology["worker_transfer_identity"]),
        "-o", "IdentitiesOnly=yes",
        "-o", "BatchMode=yes",
        "-o", "StrictHostKeyChecking=yes",
    ]


def _shell_join(parts: Iterable[Any]) -> str:
    return " ".join(shlex.quote(str(part)) for part in parts)


def _validate_run_id(value: str) -> str:
    if not RUN_ID.fullmatch(value) or len(value) > 180:
        raise WorkflowError("invalid run id: %s" % value)
    return value


def build_request(
    *,
    checkpoint: Path,
    checkpoint_sha256: str,
    metadata: Mapping[str, Any],
    profile: Mapping[str, Any],
    profile_registry: Path,
    profile_registry_sha256: str,
    artifact_root: Path = DEFAULT_ARTIFACT_ROOT,
    gpu: int = 0,
    run_id: Optional[str] = None,
    created_at: Optional[datetime] = None,
    topology: Mapping[str, Any] = TOPOLOGY,
) -> Dict[str, Any]:
    if gpu < 0 or gpu > 3:
        raise WorkflowError("worker2 GPU index must be in [0,3]")
    if not HEX64.fullmatch(checkpoint_sha256):
        raise WorkflowError("checkpoint SHA256 is invalid")
    when = created_at or datetime.now().astimezone()
    seed = int(metadata["seed"])
    expert = str(metadata["expert"])
    if run_id is None:
        run_id = "viz-%s-%s-%s-s%d-%s" % (
            expert,
            profile["run_variant"],
            checkpoint_sha256[:12],
            seed,
            when.strftime("%Y%m%d"),
        )
    run_id = _validate_run_id(run_id)
    worker_root = Path(str(topology["worker_root"]))
    commit = str(profile["inference_commit"])
    checkout = worker_root / "work/checkouts" / commit
    worker_checkpoint = worker_root / "checkpoints/by-sha256" / checkpoint_sha256 / checkpoint.name
    worker_job = worker_root / "artifacts/jobs" / run_id
    worker_results = checkout / "results/experiments" / run_id
    motion_dir = checkout / "code/motion_params" / run_id / checkpoint.stem
    artifact_parent = artifact_root.resolve() / expert
    staging = artifact_parent / (".%s.incoming" % run_id)
    final = artifact_parent / run_id
    unit = "infbagel-infer-%s" % hashlib.sha256(run_id.encode("utf-8")).hexdigest()[:16]
    weight_variant = metadata.get("primary_weight_variant") or "online"
    inference_args = [
        str(topology["worker_python"]),
        Path(str(profile["entrypoint"])).name,
        "--config-name=%s" % profile["config_name"],
        "exp_name=%s" % run_id,
        "ckpt_path=%s" % worker_checkpoint,
        "checkpoint_sha256=%s" % checkpoint_sha256,
        "checkpoint_weight_variant=%s" % weight_variant,
    ]
    inference_args.extend(profile["hydra_overrides"])
    inference_args.append("save_motion_params=true")
    return {
        "schema_version": SCHEMA_VERSION,
        "workflow": "worker2-checkpoint-to-motion-v1",
        "status": "planned",
        "created_at": when.astimezone(timezone.utc).isoformat(),
        "run_id": run_id,
        "checkpoint": {
            "authority_path": str(checkpoint.resolve()),
            "worker_path": str(worker_checkpoint),
            "sha256": checkpoint_sha256,
            "metadata": dict(metadata),
        },
        "profile": {
            **dict(profile),
            "registry_path": str(profile_registry.resolve()),
            "registry_sha256": profile_registry_sha256,
        },
        "authority": {
            "hostname": socket.gethostname(),
            "user": getpass.getuser(),
            "address": topology["authority_host"],
            "source_repo": topology["authority_repo"],
            "artifact_staging": str(staging),
            "artifact_final": str(final),
        },
        "worker": {
            "control_endpoint": "%s:%s" % (topology["control_host"], topology["control_port"]),
            "root": str(worker_root),
            "base_repo": topology["worker_base_repo"],
            "checkout": str(checkout),
            "job_dir": str(worker_job),
            "result_dir": str(worker_results),
            "motion_dir": str(motion_dir),
            "log_path": str(worker_job / "provenance/workflow.log"),
            "unit": unit,
            "gpu": gpu,
        },
        "inference": {
            "working_directory": str(checkout / "code"),
            "argv": inference_args,
            "expected_sequences": int(profile["expected_sequences"]),
            "legacy_human_frame": profile["legacy_human_frame"],
        },
        "network": {
            "control": "authority -> loopback reverse SSH only",
            "bulk": "worker2 -> 10.184.17.253 direct campus SSH/rsync",
            "ssh_config": "/dev/null",
            "windows_proxy_used": False,
        },
    }


def build_completion_script(request: Mapping[str, Any], topology: Mapping[str, Any] = TOPOLOGY) -> str:
    worker = request["worker"]
    checkpoint = request["checkpoint"]
    authority = request["authority"]
    inference = request["inference"]
    profile = request["profile"]
    transfer_words = transfer_ssh_words(topology)
    authority_target = "%s@%s" % (topology["authority_user"], topology["authority_host"])
    verify_script = "\n".join(
        [
            "set -euo pipefail",
            "test ! -e %s" % shlex.quote(authority["artifact_final"]),
            "cd %s" % shlex.quote(authority["artifact_staging"] + "/raw"),
            "/usr/bin/sha256sum -c ../provenance/motion_params.sha256 >/dev/null",
            "/usr/bin/mv %s %s" % (
                shlex.quote(authority["artifact_staging"]),
                shlex.quote(authority["artifact_final"]),
            ),
        ]
    )
    verify_command = "/bin/bash -lc %s" % shlex.quote(verify_script)
    prepare_script = "/usr/bin/install -d %s %s %s" % (
        shlex.quote(authority["artifact_staging"] + "/raw"),
        shlex.quote(authority["artifact_staging"] + "/provenance"),
        shlex.quote(authority["artifact_staging"] + "/inference-record"),
    )
    prepare_command = "/bin/bash -lc %s" % shlex.quote(prepare_script)
    command = _shell_join(inference["argv"])
    return "\n".join(
        [
            "set -euo pipefail",
            "RUN_ID=%s" % shlex.quote(request["run_id"]),
            "CHECKOUT=%s" % shlex.quote(worker["checkout"]),
            "CHECKPOINT=%s" % shlex.quote(checkpoint["worker_path"]),
            "JOB_DIR=%s" % shlex.quote(worker["job_dir"]),
            "RESULT_DIR=%s" % shlex.quote(worker["result_dir"]),
            "MOTION_DIR=%s" % shlex.quote(worker["motion_dir"]),
            "EXPECTED=%d" % int(inference["expected_sequences"]),
            "STAGING=%s" % shlex.quote(authority["artifact_staging"]),
            "AUTH_TARGET=%s" % shlex.quote(authority_target),
            "AUTH_SSH=(%s)" % _shell_join(transfer_words),
            'AUTH_RSH="${AUTH_SSH[*]}"',
            "PREPARE_COMMAND=%s" % shlex.quote(prepare_command),
            "VERIFY_COMMAND=%s" % shlex.quote(verify_command),
            'if [ ! -f "$JOB_DIR/provenance/generation_complete.json" ]; then',
            '  test ! -e "$RESULT_DIR"',
            '  test ! -e "$MOTION_DIR"',
            '  cd "$CHECKOUT/code"',
            "  ROOT_DIR=%s CUDA_VISIBLE_DEVICES=%d PYTHONUNBUFFERED=1 %s" % (
                shlex.quote(worker["checkout"]), int(worker["gpu"]), command
            ),
            '  MOTION_COUNT=$(/usr/bin/find "$MOTION_DIR" -maxdepth 1 -type f -name "*_motion_params.pkl" | /usr/bin/wc -l)',
            '  PREDICTION_COUNT=$(/usr/bin/find "$RESULT_DIR/chois/predictions" -maxdepth 1 -type f -name "*.npz" | /usr/bin/wc -l)',
            '  GROUND_TRUTH_COUNT=$(/usr/bin/find "$RESULT_DIR/chois/ground_truth" -maxdepth 1 -type f -name "*.npz" | /usr/bin/wc -l)',
            '  test "$MOTION_COUNT" -eq "$EXPECTED"',
            '  test "$PREDICTION_COUNT" -eq "$EXPECTED"',
            '  test "$GROUND_TRUTH_COUNT" -eq "$EXPECTED"',
            '  (cd "$MOTION_DIR" && /usr/bin/find . -maxdepth 1 -type f -name "*_motion_params.pkl" -print0 | /usr/bin/sort -z | /usr/bin/xargs -0 /usr/bin/sha256sum) > "$JOB_DIR/provenance/motion_params.sha256"',
            '  /usr/bin/sha256sum "$CHECKPOINT" > "$JOB_DIR/provenance/checkpoint.sha256"',
            '  /usr/bin/git -C "$CHECKOUT" status --porcelain --untracked-files=all > "$JOB_DIR/provenance/git_status.txt"',
            '  test ! -s "$JOB_DIR/provenance/git_status.txt"',
            '  COMPLETED_AT=$(/usr/bin/date --iso-8601=seconds)',
            '  /usr/bin/printf \'{"schema_version":1,"status":"generation-complete","run_id":"%%s","profile":"%%s","checkpoint_sha256":"%%s","motion_count":%%s,"completed_at":"%%s"}\\n\' "$RUN_ID" %s %s "$MOTION_COUNT" "$COMPLETED_AT" > "$JOB_DIR/provenance/generation_complete.json"' % (
                shlex.quote(profile["name"]), shlex.quote(checkpoint["sha256"])
            ),
            "fi",
            '"${AUTH_SSH[@]}" "$AUTH_TARGET" "$PREPARE_COMMAND"',
            '/usr/bin/rsync -a --protect-args -e "$AUTH_RSH" "$MOTION_DIR/" "$AUTH_TARGET:$STAGING/raw/"',
            '/usr/bin/rsync -a --protect-args -e "$AUTH_RSH" "$JOB_DIR/provenance/" "$AUTH_TARGET:$STAGING/provenance/"',
            '/usr/bin/rsync -a --protect-args -e "$AUTH_RSH" "$RESULT_DIR/" "$AUTH_TARGET:$STAGING/inference-record/"',
            '"${AUTH_SSH[@]}" "$AUTH_TARGET" "$VERIFY_COMMAND"',
        ]
    )


def build_preflight_script(request: Mapping[str, Any], topology: Mapping[str, Any] = TOPOLOGY) -> str:
    worker = request["worker"]
    checkpoint = request["checkpoint"]
    profile = request["profile"]
    authority = request["authority"]
    inference = request["inference"]
    transfer_words = transfer_ssh_words(topology)
    authority_target = "%s@%s" % (topology["authority_user"], topology["authority_host"])
    checkpoint_source = "%s:%s" % (authority_target, checkpoint["authority_path"])
    repo_source = "%s:%s" % (authority_target, topology["authority_repo"])
    request_source = "%s:%s/request.json" % (authority_target, authority["artifact_staging"])
    authority_check_script = "test -d %s && test ! -e %s" % (
        shlex.quote(authority["artifact_staging"]),
        shlex.quote(authority["artifact_final"]),
    )
    authority_check_command = "/bin/bash -lc %s" % shlex.quote(authority_check_script)
    completion = build_completion_script(request, topology)
    resolved_args = list(inference["argv"])
    resolved_args.insert(2, "--cfg")
    resolved_args.insert(3, "job")
    resolved_args.insert(4, "--resolve")
    required_checks = [
        'test -e "$CHECKOUT/%s"' % asset for asset in profile["required_assets"]
    ]
    return "\n".join(
        [
            "set -euo pipefail",
            "BASE_REPO=%s" % shlex.quote(worker["base_repo"]),
            "CHECKOUT=%s" % shlex.quote(worker["checkout"]),
            "CHECKPOINT=%s" % shlex.quote(checkpoint["worker_path"]),
            "JOB_DIR=%s" % shlex.quote(worker["job_dir"]),
            "UNIT=%s" % shlex.quote(worker["unit"]),
            "AUTH_TARGET=%s" % shlex.quote(authority_target),
            "AUTH_SSH=(%s)" % _shell_join(transfer_words),
            'AUTH_RSH="${AUTH_SSH[*]}"',
            "AUTHORITY_CHECK_COMMAND=%s" % shlex.quote(authority_check_command),
            'test ! -e "$JOB_DIR"',
            'test ! -e %s' % shlex.quote(worker["result_dir"]),
            'test ! -e %s' % shlex.quote(worker["motion_dir"]),
            'test -d "$BASE_REPO/.git"',
            'test -z "$(/usr/bin/git -C "$BASE_REPO" status --porcelain --untracked-files=all)"',
            '"${AUTH_SSH[@]}" "$AUTH_TARGET" "$AUTHORITY_CHECK_COMMAND"',
            'if [ ! -f "$CHECKPOINT" ]; then',
            '  /usr/bin/install -d "$(/usr/bin/dirname "$CHECKPOINT")"',
            '  /usr/bin/rsync -a --protect-args -e "$AUTH_RSH" %s "$(/usr/bin/dirname "$CHECKPOINT")/"' % shlex.quote(checkpoint_source),
            "fi",
            '/usr/bin/printf \'%%s  %%s\\n\' %s "$CHECKPOINT" | /usr/bin/sha256sum -c -' % shlex.quote(checkpoint["sha256"]),
            'if ! /usr/bin/git -C "$BASE_REPO" cat-file -e %s^{commit}; then' % shlex.quote(profile["inference_commit"]),
            '  GIT_SSH_COMMAND="$AUTH_RSH" /usr/bin/git -C "$BASE_REPO" fetch --no-tags %s %s' % (
                shlex.quote(repo_source), shlex.quote(profile["inference_commit"])
            ),
            "fi",
            'if [ ! -d "$CHECKOUT" ]; then',
            '  /usr/bin/install -d "$(/usr/bin/dirname "$CHECKOUT")"',
            '  /usr/bin/git -C "$BASE_REPO" worktree add --detach "$CHECKOUT" %s' % shlex.quote(profile["inference_commit"]),
            "fi",
            'test "$(/usr/bin/git -C "$CHECKOUT" rev-parse HEAD)" = %s' % shlex.quote(profile["inference_commit"]),
            'if [ ! -e "$CHECKOUT/data" ]; then /usr/bin/ln -s %s "$CHECKOUT/data"; fi' % shlex.quote(topology["worker_data"]),
            'if [ ! -e "$CHECKOUT/smpl_models" ]; then /usr/bin/ln -s %s "$CHECKOUT/smpl_models"; fi' % shlex.quote(topology["worker_smpl_models"]),
            'test "$(/usr/bin/readlink -f "$CHECKOUT/data")" = %s' % shlex.quote(topology["worker_data"]),
            'test "$(/usr/bin/readlink -f "$CHECKOUT/smpl_models")" = %s' % shlex.quote(topology["worker_smpl_models"]),
            *required_checks,
            'test -z "$(/usr/bin/git -C "$CHECKOUT" status --porcelain --untracked-files=all)"',
            'GPU_PROCESSES=$(/usr/bin/nvidia-smi -i %d --query-compute-apps=pid --format=csv,noheader,nounits)' % int(worker["gpu"]),
            'test -z "$GPU_PROCESSES"',
            'test "$(/bin/systemctl --user show "$UNIT" --property=LoadState --value)" = "not-found"',
            'mkdir -p "$JOB_DIR/provenance"',
            '/usr/bin/rsync -a --protect-args -e "$AUTH_RSH" %s "$JOB_DIR/provenance/request.json"' % shlex.quote(request_source),
            'cd "$CHECKOUT/code"',
            'ROOT_DIR="$CHECKOUT" CUDA_VISIBLE_DEVICES=%d %s > "$JOB_DIR/provenance/resolved_config.yaml"' % (
                int(worker["gpu"]), _shell_join(resolved_args)
            ),
            'if /usr/bin/grep -F \'${\' "$JOB_DIR/provenance/resolved_config.yaml"; then exit 1; fi',
            '/usr/bin/sha256sum "$JOB_DIR/provenance/resolved_config.yaml" > "$JOB_DIR/provenance/resolved_config.sha256"',
            '/bin/systemd-run --user --unit="$UNIT" --working-directory="$CHECKOUT/code" --setenv=PYTHONUNBUFFERED=1 --property=StandardOutput=append:%s --property=StandardError=append:%s /bin/bash -lc %s' % (
                shlex.quote(worker["log_path"]),
                shlex.quote(worker["log_path"]),
                shlex.quote(completion),
            ),
        ]
    )


def _git_clean(repo: Path) -> None:
    result = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=all"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.stdout.strip():
        raise WorkflowError("visualization workflow must start from a clean worktree")


def _assert_source_commit(commit: str, source_repo: str) -> None:
    result = subprocess.run(
        ["git", "-C", source_repo, "cat-file", "-e", "%s^{commit}" % commit],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        raise WorkflowError("inference commit is unavailable from authority source repo: %s" % commit)


def _write_json_exclusive(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError as exc:
        raise WorkflowError("refusing to overwrite %s" % path) from exc


def _run_control_script(script: str, topology: Mapping[str, Any] = TOPOLOGY) -> subprocess.CompletedProcess:
    remote_command = "/bin/bash -lc %s" % shlex.quote(script)
    return subprocess.run(
        control_ssh_argv(topology) + [remote_command],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def command_start(args: argparse.Namespace) -> None:
    checkpoint = Path(args.checkpoint)
    metadata = load_checkpoint_metadata(checkpoint)
    profiles_path = Path(args.profiles)
    profiles, profiles_sha = load_profiles(profiles_path)
    profile = select_profile(profiles, metadata, args.profile)
    checkpoint_sha = sha256_file(checkpoint)
    request = build_request(
        checkpoint=checkpoint,
        checkpoint_sha256=checkpoint_sha,
        metadata=metadata,
        profile=profile,
        profile_registry=profiles_path,
        profile_registry_sha256=profiles_sha,
        artifact_root=Path(args.artifact_root),
        gpu=args.gpu,
        run_id=args.run_id,
    )
    script = build_preflight_script(request)
    if args.dry_run:
        print(json.dumps({"request": request, "remote_preflight_sha256": hashlib.sha256(script.encode()).hexdigest()}, indent=2, sort_keys=True))
        return
    _git_clean(REPO_ROOT)
    _assert_source_commit(profile["inference_commit"], TOPOLOGY["authority_repo"])
    staging = Path(request["authority"]["artifact_staging"])
    final = Path(request["authority"]["artifact_final"])
    if staging.exists() or final.exists():
        raise WorkflowError("run/artifact destination already exists; never reuse a run id")
    staging.mkdir(parents=True, exist_ok=False)
    request["status"] = "dispatching"
    request["request_sha256"] = json_sha256(request)
    _write_json_exclusive(staging / "request.json", request)
    try:
        result = _run_control_script(script)
    except subprocess.CalledProcessError as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "dispatch-failed",
            "created_at": utc_now(),
            "returncode": exc.returncode,
            "stdout": exc.stdout,
            "stderr": exc.stderr,
        }
        _write_json_exclusive(staging / "dispatch_failed.json", failure)
        raise WorkflowError("worker2 preflight/dispatch failed; retained %s" % staging) from exc
    print(json.dumps({
        "status": "started",
        "run_id": request["run_id"],
        "profile": profile["name"],
        "checkpoint_sha256": checkpoint_sha,
        "worker_unit": request["worker"]["unit"],
        "artifact_staging": str(staging),
        "artifact_final": str(final),
        "dispatch_stdout": result.stdout.strip(),
        "status_command": "%s tools/worker2_inference.py status %s" % (sys.executable, request["run_id"]),
    }, indent=2, sort_keys=True))


def _find_request(run_id: str, artifact_root: Path) -> Tuple[Path, Dict[str, Any], bool]:
    _validate_run_id(run_id)
    matches: List[Tuple[Path, bool]] = []
    if artifact_root.is_dir():
        for expert in artifact_root.iterdir():
            if not expert.is_dir():
                continue
            final = expert / run_id
            staging = expert / (".%s.incoming" % run_id)
            if (final / "request.json").is_file():
                matches.append((final, True))
            if (staging / "request.json").is_file():
                matches.append((staging, False))
    if len(matches) != 1:
        raise WorkflowError("expected exactly one artifact request for %s, found %d" % (run_id, len(matches)))
    root, promoted = matches[0]
    return root, read_json(root / "request.json"), promoted


def command_status(args: argparse.Namespace) -> None:
    root, request, promoted = _find_request(args.run_id, Path(args.artifact_root))
    if promoted:
        completed_path = root / "provenance/generation_complete.json"
        completed = read_json(completed_path) if completed_path.is_file() else None
        print(json.dumps({
            "status": "complete",
            "run_id": args.run_id,
            "artifact_root": str(root),
            "generation": completed,
        }, indent=2, sort_keys=True))
        return
    worker = request["worker"]
    inspect_script = "\n".join([
        "set -euo pipefail",
        "/bin/systemctl --user show %s --property=LoadState,ActiveState,SubState,Result,ExecMainStatus --no-pager" % shlex.quote(worker["unit"]),
        "/usr/bin/printf -- '--- log tail ---\\n'",
        "/usr/bin/tail -n %d %s 2>/dev/null || true" % (args.log_lines, shlex.quote(worker["log_path"])),
    ])
    try:
        result = _run_control_script(inspect_script)
        remote = result.stdout
    except subprocess.CalledProcessError as exc:
        remote = "control inspection failed: %s\n%s" % (exc.returncode, exc.stderr)
    print(json.dumps({
        "status": "running-or-failed",
        "run_id": args.run_id,
        "artifact_staging": str(root),
        "worker_unit": worker["unit"],
        "remote_status": remote,
    }, indent=2, sort_keys=True))


def command_retry_return(args: argparse.Namespace) -> None:
    root, request, promoted = _find_request(args.run_id, Path(args.artifact_root))
    if promoted:
        raise WorkflowError("artifact is already complete: %s" % root)
    worker = request["worker"]
    script = "\n".join([
        "set -euo pipefail",
        "test -f %s" % shlex.quote(worker["job_dir"] + "/provenance/generation_complete.json"),
        "! /bin/systemctl --user is-active --quiet %s" % shlex.quote(worker["unit"]),
        "/bin/systemctl --user restart %s" % shlex.quote(worker["unit"]),
    ])
    result = _run_control_script(script)
    print(json.dumps({
        "status": "return-restarted",
        "run_id": args.run_id,
        "artifact_staging": str(root),
        "worker_unit": worker["unit"],
        "stdout": result.stdout.strip(),
    }, indent=2, sort_keys=True))


def command_profiles(args: argparse.Namespace) -> None:
    profiles, digest = load_profiles(Path(args.profiles))
    print(json.dumps({
        "registry": str(Path(args.profiles).resolve()),
        "sha256": digest,
        "profiles": profiles,
    }, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dispatch checkpoint inference to worker2")
    parser.add_argument("--profiles", default=str(DEFAULT_PROFILES))
    parser.add_argument("--artifact-root", default=str(DEFAULT_ARTIFACT_ROOT))
    subparsers = parser.add_subparsers(dest="command", required=True)

    start = subparsers.add_parser("start", help="validate and start one persistent inference job")
    start.add_argument("checkpoint", help="trusted absolute authority-side .pth path")
    start.add_argument("--profile", default="auto")
    start.add_argument("--gpu", type=int, default=0)
    start.add_argument("--run-id")
    start.add_argument("--dry-run", action="store_true")
    start.set_defaults(func=command_start)

    status = subparsers.add_parser("status", help="inspect an existing workflow job")
    status.add_argument("run_id")
    status.add_argument("--log-lines", type=int, default=40)
    status.set_defaults(func=command_status)

    retry = subparsers.add_parser("retry-return", help="retry artifact return after generation completed")
    retry.add_argument("run_id")
    retry.set_defaults(func=command_retry_return)

    profiles = subparsers.add_parser("profiles", help="show registered fail-closed inference profiles")
    profiles.set_defaults(func=command_profiles)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except (WorkflowError, OSError, subprocess.SubprocessError) as exc:
        print("ERROR: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
