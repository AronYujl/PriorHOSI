#!/usr/bin/env python3
"""Capture the live four-GPU Phase 1B worker preflight beside a run manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Sequence


EXPECTED_AUDIT_SHA256 = "1deea6a724a3319d4c5654da682d7f51af7e5c93b119d159bd2b37ad258f627f"
EXPECTED_CONTRACT_SHA256 = "a908994bef58a21798af605f01df25582743e1066dd7d0211315c3f0c88951cf"
EXPECTED_T2M_COMMIT = "72df96ec453edea2fbe9603b1d58a955eaf71636"
EXPECTED_CHOIS_COMMIT = "8ec585aa0200fd2a890ffb12897bcf69ae719463"
EXPECTED_CHOIS_CHECKPOINT_SHA256 = "a125bc15ffd9772686737111c7501ecee0a2d8571d9aca348ec1195ddef78775"
IDLE_MEMORY_USED_MIB_MAX = 128
DISPLAY_ONLY_UTILIZATION_PERCENT_MAX = 1
IDLE_PSTATE = "P8"
IDLE_SAMPLE_COUNT = 3
IDLE_SAMPLE_INTERVAL_SECONDS = 1.0


def run(command: Sequence[str], cwd: Optional[Path] = None, check: bool = True) -> str:
    completed = subprocess.run(
        list(command), cwd=str(cwd) if cwd else None, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False,
    )
    if check and completed.returncode:
        raise RuntimeError(f"command failed ({' '.join(command)}): {completed.stdout.strip()}")
    return completed.stdout.strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gpu_snapshot() -> list[dict]:
    output = run([
        "nvidia-smi",
        "--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate,driver_version",
        "--format=csv,noheader,nounits",
    ])
    keys = (
        "index", "uuid", "name", "memory_total_mib", "memory_used_mib", "memory_free_mib",
        "utilization_percent", "temperature_c", "pstate", "driver_version",
    )
    records = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        record = dict(zip(keys, fields))
        for key in (
            "index", "memory_total_mib", "memory_used_mib", "memory_free_mib",
            "utilization_percent", "temperature_c",
        ):
            record[key] = int(record[key])
        records.append(record)
    return records


def compute_processes() -> list[str]:
    output = run([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ], check=False)
    return [line.strip() for line in output.splitlines() if line.strip()]


def four_gpu_idle(gpus: list[dict], processes: list[str]) -> bool:
    """Accept bounded display-only load; P-state is descriptive, not binding."""
    return (
        len(gpus) == 4
        and not processes
        and all(
            gpu["memory_used_mib"] <= IDLE_MEMORY_USED_MIB_MAX
            and gpu["utilization_percent"] <= DISPLAY_ONLY_UTILIZATION_PERCENT_MAX
            for gpu in gpus
        )
    )


def four_gpu_evaluation_idle(gpus: list[dict], processes: list[str]) -> bool:
    """Ignore only GPU 0 display utilization; P-state remains descriptive."""
    return (
        len(gpus) == 4
        and not processes
        and all(
            gpu["memory_used_mib"] <= IDLE_MEMORY_USED_MIB_MAX
            and (
                gpu["index"] == 0
                or gpu["utilization_percent"] <= DISPLAY_ONLY_UTILIZATION_PERCENT_MAX
            )
            for gpu in gpus
        )
    )


def forbidden_snapshot_entries(data: Path) -> list[str]:
    forbidden = []
    for relative in ("dataset", "hosi_test"):
        if (data / relative).exists():
            forbidden.append(relative)
    for partition in ("train", "test"):
        root = data / partition
        if not root.is_dir():
            continue
        forbidden.extend(
            str(path.relative_to(data)) for path in root.iterdir() if path.name.startswith("Scene")
        )
    return sorted(forbidden)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--audit-recheck", type=Path, required=True)
    parser.add_argument("--chois-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-idle", action="store_true")
    parser.add_argument("--allow-gpu0-display-utilization", action="store_true")
    args = parser.parse_args()
    repo = args.repo_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    status = run(["git", "status", "--porcelain=v1", "--untracked-files=all"], repo)
    data_link = repo / "data"
    if not data_link.is_symlink():
        raise RuntimeError(f"worker data must be a checkout-local symlink: {data_link}")
    data = data_link.resolve()
    gpu_samples = []
    process_samples = []
    for sample_index in range(IDLE_SAMPLE_COUNT):
        gpu_samples.append(gpu_snapshot())
        process_samples.append(compute_processes())
        if sample_index + 1 < IDLE_SAMPLE_COUNT:
            time.sleep(IDLE_SAMPLE_INTERVAL_SECONDS)
    gpus = gpu_samples[-1]
    processes = process_samples[-1]
    idle_function = (
        four_gpu_evaluation_idle
        if args.allow_gpu0_display_utilization
        else four_gpu_idle
    )
    idle = all(
        idle_function(sample_gpus, sample_processes)
        for sample_gpus, sample_processes in zip(gpu_samples, process_samples)
    )
    disk = shutil.disk_usage(repo)
    contract = json.loads((
        repo / "experiments/results/p1_data_hoi_contract_s42_20260713.json"
    ).read_text(encoding="utf-8"))
    audit_hash = sha256(args.audit_recheck.resolve())
    t2m_root = repo / "third_party/text-to-motion"
    chois_checkpoint = (
        repo / "third_party/chois_omomo_evaluator_assets/for_t2m_eval/checkpoints/omomo/"
        "text_motion_features/model/finest.tar"
    )
    python_details = json.loads(run([
        str(args.python.resolve()), "-c",
        "import hashlib,json,sys,torch,pytorch3d; p=sys.executable; "
        "h=hashlib.sha256(open(p,'rb').read()).hexdigest(); "
        "print(json.dumps({'executable':p,'sha256':h,'version':sys.version,'torch':torch.__version__,"
        "'torch_cuda':torch.version.cuda,'pytorch3d':pytorch3d.__version__,"
        "'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count()}))",
    ]))
    configured_python = os.environ.get("INFBAGEL_PYTHON")
    checks = {
        "hostname_node01": socket.gethostname() == "node01",
        "branch_phase_01b_hoi": run(["git", "branch", "--show-current"], repo) == "phase/01b-hoi",
        "worktree_clean": not status,
        "data_symlink": data_link.is_symlink(),
        "omomo_only_snapshot": not forbidden_snapshot_entries(data),
        "audit_hash_exact": audit_hash == EXPECTED_AUDIT_SHA256,
        "contract_hash_exact": contract.get("contract_sha256") == EXPECTED_CONTRACT_SHA256,
        "four_rtx_3090": len(gpus) == 4 and all(gpu["name"] == "NVIDIA GeForce RTX 3090" for gpu in gpus),
        "python_exact": (
            bool(configured_python)
            and Path(configured_python).is_absolute()
            and args.python.is_absolute()
            and args.python.resolve() == Path(configured_python).resolve()
            and Path(python_details["executable"]).resolve() == args.python.resolve()
        ),
        "cuda_four_visible": python_details["cuda_available"] and python_details["cuda_device_count"] == 4,
        "ntp_synchronized": "NTPSynchronized=yes" in run([
            "timedatectl", "show", "-p", "Timezone", "-p", "NTPSynchronized", "-p", "LocalRTC",
        ]).splitlines(),
        "reverse_tunnel_active": run(["systemctl", "--user", "is-active", "infbagel-reverse-ssh.service"]) == "active",
        "reverse_tunnel_enabled": run(["systemctl", "--user", "is-enabled", "infbagel-reverse-ssh.service"]) == "enabled",
        "text_to_motion_pinned": run(["git", "rev-parse", "HEAD"], t2m_root) == EXPECTED_T2M_COMMIT,
        "chois_pinned": run(["git", "rev-parse", "HEAD"], args.chois_root.resolve()) == EXPECTED_CHOIS_COMMIT,
        "chois_checkpoint_pinned": sha256(chois_checkpoint) == EXPECTED_CHOIS_CHECKPOINT_SHA256,
    }
    result = {
        "schema_version": 1,
        "captured_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "role": "phase_1b_hoi_worker",
        "reportable_workload_started": False,
        "repo": {
            "path": str(repo), "branch": run(["git", "branch", "--show-current"], repo),
            "head": run(["git", "rev-parse", "HEAD"], repo), "status": status.splitlines(),
        },
        "data": {
            "link": str(data_link), "target": str(data),
            "forbidden_entries": forbidden_snapshot_entries(data),
            "audit_recheck": {"path": str(args.audit_recheck.resolve()), "sha256": audit_hash},
            "contract_sha256": contract.get("contract_sha256"),
        },
        "python": python_details,
        "gpus": gpus,
        "compute_processes": processes,
        "gpu_samples": gpu_samples,
        "compute_process_samples": process_samples,
        "four_gpu_idle": idle,
        "idle_contract": {
            "gpu_count": 4,
            "compute_processes_must_be_empty": True,
            "memory_used_mib_max_per_gpu": IDLE_MEMORY_USED_MIB_MAX,
            "instantaneous_utilization_percent_max_per_gpu": (
                DISPLAY_ONLY_UTILIZATION_PERCENT_MAX
            ),
            "gpu0_display_utilization_ignored": bool(
                args.allow_gpu0_display_utilization
            ),
            "pstate_binding": False,
            "preferred_idle_pstate": IDLE_PSTATE,
            "sample_count": IDLE_SAMPLE_COUNT,
            "sample_interval_seconds": IDLE_SAMPLE_INTERVAL_SECONDS,
            "scope": "display-only Xorg driver floor; CUDA compute contention remains forbidden",
        },
        "filesystem": {"total_bytes": disk.total, "used_bytes": disk.used, "free_bytes": disk.free},
        "clock": run(["timedatectl", "status"]),
        "reverse_tunnel": run(["systemctl", "--user", "status", "infbagel-reverse-ssh.service"], check=False),
        "assets": {
            "text_to_motion_commit": run(["git", "rev-parse", "HEAD"], t2m_root),
            "chois_commit": run(["git", "rev-parse", "HEAD"], args.chois_root.resolve()),
            "chois_checkpoint": {"path": str(chois_checkpoint), "sha256": sha256(chois_checkpoint)},
        },
        "checks": checks,
        "passed": all(checks.values()) and (idle if args.require_idle else True),
        "idle_required": bool(args.require_idle),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps({"passed": result["passed"], "four_gpu_idle": idle, "checks": checks}, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
