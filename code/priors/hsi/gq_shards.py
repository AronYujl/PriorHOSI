"""Independent P16-GQ wrapper and fail-closed execution-attestation gate.

The LINGO evaluator is a sealed baseline.  This module owns the orchestration
around it: resolving and checking the formal config, running the CPU preflight,
creating an attestation before the sealed child starts, and writing a receipt
after the child has produced its *raw* payload.  The raw evaluator JSON is
never decorated or rewritten.  In particular, there is no API that can turn a
caller-supplied JSON object into a reportable treatment identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

# Make the checked-out ``code`` root available when this HSI-owned wrapper is
# invoked by file path (``python code/priors/hsi/gq_shards.py``), as opposed to
# module form with an already configured PYTHONPATH.  This does not import or
# modify the sealed evaluator.
_CODE_ROOT = Path(__file__).resolve().parents[2]
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from priors.hsi.body_proxy import (  # noqa: E402
    AREA512_INDEX_RAW_INT64_SHA256,
    AREA512_INDEX_SHA256,
    BODY_PROXY_ASSET_SHA256,
    BODY_PROXY_ASSET_SIZE_BYTES,
    SMPLX_SOURCE_SHA256,
    SMPLX_SOURCE_SIZE_BYTES,
)
from priors.hsi.preflight import (  # noqa: E402
    SEALED_CHECKPOINT_SHA256,
    run_formal_preflight,
)
from priors.hsi.scene_field import sdf_cache_protocol_identity  # noqa: E402


SEALED_EVALUATOR_RELATIVE_PATH = Path("code/test_infbagel_lingo_hsi.py")
FORMAL_WRAPPER_RELATIVE_PATH = "code/priors/hsi/gq_shards.py"
FORMAL_CONFIG_NAME = "config_sample_infbagel_lingo_hsi_p16gq"
SEALED_BASE = "fc033a9"
# This is the sealed evaluator's known digest at the protected baseline.  The
# wrapper also checks the actual bytes against ``SEALED_BASE`` before execution
# and before merge; retaining the digest in the attestation makes a copied or
# altered receipt fail closed even when its source checkout is unavailable.
SEALED_EVALUATOR_SHA256 = (
    "4f25a6e67ab5104f2b10b41acbafa7ef257814751e0c402f0e28581b7b9eac0f"
)

ATTESTATION_SCHEMA_VERSION = 1
ATTESTATION_PROTOCOL = "p16-gq-preflight-attestation-v1"
RECEIPT_SCHEMA_VERSION = 1
RECEIPT_PROTOCOL = "p16-gq-execution-receipt-v1"
PREFLIGHT_ATTESTATION_SUFFIX = ".gq_preflight.json"
EXECUTION_RECEIPT_SUFFIX = ".gq_receipt.json"
PAYLOAD_FILENAME = "per_sequence_metrics.json"

GQ_GUIDANCE_MODE = "mesh_sdf_gq"
GQ_GUIDANCE_VERSION = "p16-gq-mesh-sdf-v1"
GQ_SDF_WEIGHT = 4879
GQ_SDF_MARGIN_M = 0.0
GQ_FLOOR_THRESHOLD_M = 0.02
GQ_SHARD_COUNT = 8
GQ_EPISODES = 375
GQ_WINDOWS = 2271

# This value was explicitly excluded from P16-GQ.  Rejecting it recursively in
# every runtime claim prevents it from reappearing under an unrecognised field,
# a legacy identity block, an episode id, or a synthesized config fragment.
_FORBIDDEN_30573 = 30573
_FORBIDDEN_30573_TEXT = re.compile(r"(?<![0-9])30573(?![0-9])")

# These are the treatment flags whose observed values must come from the
# resolved Hydra config.  ``shard_index`` is deliberately absent: it is an
# invocation binding, not a caller-controlled config fragment.
P16_GQ_CONFIG_FLAGS: Dict[str, Any] = {
    "lingo_hsi_mode": "sample",
    "sample_type": "diffusion",
    "use_guidance": True,
    "export_motion": True,
    "hsi_progress_fix": True,
    "hsi_guidance_sdf_proxy": "area512",
    "hsi_guidance_sdf_weight": GQ_SDF_WEIGHT,
    "formal_preflight": True,
    "formal_attestation": True,
    "formal_attestation_protocol": ATTESTATION_PROTOCOL,
    "formal_wrapper": FORMAL_WRAPPER_RELATIVE_PATH,
    "load_scene": True,
    "load_scene_goal": True,
    "load_pelvis_goal": True,
    "dataset_hsi_mesh_root": "lingo_mesh_root",
    "seed": 42,
    "shard_count": GQ_SHARD_COUNT,
}

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _canonical_digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _hash(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError("%s must be a lowercase SHA-256 string" % label)
    return value


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _reject_forbidden_30573(value: Any, path: str = "value") -> None:
    """Reject the explicitly forbidden legacy weight/episode token anywhere."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            if isinstance(key, str) and _FORBIDDEN_30573_TEXT.search(key):
                raise ValueError("forbidden 30573 occurrence at %s" % path)
            _reject_forbidden_30573(child, "%s.%s" % (path, key))
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            _reject_forbidden_30573(child, "%s[%d]" % (path, index))
        return
    if isinstance(value, str):
        if _FORBIDDEN_30573_TEXT.search(value):
            raise ValueError("forbidden 30573 occurrence at %s" % path)
        return
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == _FORBIDDEN_30573:
            raise ValueError("forbidden 30573 occurrence at %s" % path)


def _exact_difference(expected: Any, actual: Any, path: str = "") -> Optional[str]:
    """Return the first type-sensitive difference between JSON values."""
    if isinstance(expected, Mapping) or isinstance(actual, Mapping):
        if not isinstance(expected, Mapping) or not isinstance(actual, Mapping):
            return "%s type %s != %s" % (
                path or "attestation",
                type(actual).__name__,
                type(expected).__name__,
            )
        expected_keys = set(expected)
        actual_keys = set(actual)
        if any(not isinstance(key, str) for key in expected_keys | actual_keys):
            return "%s keys must be strings" % (path or "attestation")
        missing = sorted(expected_keys - actual_keys)
        if missing:
            return "%s missing key %r" % (path or "attestation", missing[0])
        extra = sorted(actual_keys - expected_keys)
        if extra:
            return "%s has unexpected key %r" % (path or "attestation", extra[0])
        for key in sorted(expected_keys):
            difference = _exact_difference(
                expected[key],
                actual[key],
                "%s.%s" % (path, key) if path else str(key),
            )
            if difference:
                return difference
        return None
    if isinstance(expected, list) or isinstance(actual, list):
        if not isinstance(expected, list) or not isinstance(actual, list):
            return "%s type %s != %s" % (
                path or "attestation",
                type(actual).__name__,
                type(expected).__name__,
            )
        if len(expected) != len(actual):
            return "%s length %d != %d" % (path, len(actual), len(expected))
        for index, (left, right) in enumerate(zip(expected, actual)):
            difference = _exact_difference(left, right, "%s[%d]" % (path, index))
            if difference:
                return difference
        return None
    if type(expected) is not type(actual):
        return "%s type %s != %s" % (
            path or "attestation",
            type(actual).__name__,
            type(expected).__name__,
        )
    if expected != actual:
        return "%s: %r != %r" % (path, actual, expected)
    return None


def _require_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("%s must be an object" % label)
    return value


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read JSON %s: %s" % (path, error)) from error
    if not isinstance(value, dict):
        raise ValueError("JSON value is not an object: %s" % path)
    _reject_forbidden_30573(value, str(path))
    return value


def load_payload(path: Path) -> Dict[str, Any]:
    """Load raw evaluator JSON without adding wrapper-owned claims."""
    try:
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cannot read payload %s: %s" % (path, error)) from error
    if not isinstance(payload, dict):
        raise ValueError("payload is not a JSON object: %s" % path)
    _reject_forbidden_30573(payload, str(path))
    return payload


def _write_json_atomic(
    path: Path, value: Mapping[str, Any], *, overwrite: bool = False
) -> None:
    """Write a JSON sidecar, refusing an existing attestation or receipt."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError("refusing to overwrite existing sidecar: %s" % path)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=".%s." % path.name,
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
    try:
        if overwrite:
            os.replace(str(temporary), str(path))
        else:
            # A same-directory hard-link is atomic and cannot replace an
            # attestation/receipt that appeared after the initial existence check.
            os.link(str(temporary), str(path))
    except FileExistsError:
        raise
    finally:
        temporary.unlink(missing_ok=True)


def _canonical_payload_path(path: Path) -> Path:
    path = Path(path).resolve()
    if path.name != PAYLOAD_FILENAME or path.parent.name != "evaluation":
        raise ValueError(
            "P16-GQ payload must be evaluation/%s, got %s" % (PAYLOAD_FILENAME, path)
        )
    return path


def preflight_attestation_path(payload_path: Path) -> Path:
    """Return the deterministic sibling path created before a sealed run."""
    payload_path = _canonical_payload_path(payload_path)
    output_dir = payload_path.parent.parent
    return output_dir.parent / (output_dir.name + PREFLIGHT_ATTESTATION_SUFFIX)


def execution_receipt_path(payload_path: Path) -> Path:
    """Return the deterministic sibling path created after a sealed run."""
    payload_path = _canonical_payload_path(payload_path)
    output_dir = payload_path.parent.parent
    return output_dir.parent / (output_dir.name + EXECUTION_RECEIPT_SUFFIX)


def _load_episode_manifest(episode_dir: Path) -> Dict[str, Any]:
    """Hash every episode file and return the exact ordered shard input."""
    root = Path(episode_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError("episode manifest directory is missing: %s" % root)
    entries: List[Dict[str, Any]] = []
    file_records: List[Dict[str, Any]] = []
    for path in sorted(root.glob("*.json")):
        raw = path.read_bytes()
        try:
            episodes = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("episode manifest is unreadable: %s" % path) from error
        if not isinstance(episodes, list):
            raise ValueError("episode manifest file is not a list: %s" % path)
        _reject_forbidden_30573(path.name, str(path))
        for index, episode in enumerate(episodes):
            if not isinstance(episode, Mapping):
                raise ValueError("episode is not an object in %s" % path)
            _reject_forbidden_30573(episode, "%s[%d]" % (path, index))
            if episode.get("scene_name") != path.stem:
                raise ValueError("episode scene mismatch in %s" % path)
            if episode.get("object_name") is not None:
                raise ValueError("P16-GQ requires scene-only episodes")
            episode_num = episode.get("episode_num")
            if type(episode_num) is not int or episode_num < 1:
                raise ValueError("episode_num must be a positive integer in %s" % path)
            entries.append(
                {
                    "scene_name": path.stem,
                    "scene_episode_index": index,
                    "episode_num": episode_num,
                }
            )
        file_records.append(
            {
                "path": path.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw),
                "episode_count": len(episodes),
            }
        )
    if not entries:
        raise ValueError("no episodes found under %s" % root)
    descriptor = {"files": file_records}
    return {
        "root": str(root),
        "files": file_records,
        "sha256": _canonical_digest(descriptor),
        "entries": entries,
        "entries_sha256": _canonical_digest(entries),
        "episode_count": len(entries),
        "window_count": int(sum(item["episode_num"] for item in entries)),
    }


def _plan_episode_shards(window_counts: Sequence[int]) -> Tuple[Tuple[int, ...], ...]:
    if len(window_counts) < GQ_SHARD_COUNT:
        raise ValueError(
            "P16-GQ shard_count=8 exceeds episode count %d" % len(window_counts)
        )
    loads = [0] * GQ_SHARD_COUNT
    bins: List[List[int]] = [[] for _ in range(GQ_SHARD_COUNT)]
    order = sorted(
        range(len(window_counts)),
        key=lambda index: (-int(window_counts[index]), index),
    )
    for index in order:
        target = min(range(GQ_SHARD_COUNT), key=lambda shard: (loads[shard], shard))
        bins[target].append(index)
        loads[target] += int(window_counts[index])
    return tuple(tuple(sorted(item)) for item in bins)


def _resolve_formal_config(
    repo_root: Path, config_name: str, shard_index: int
) -> Dict[str, Any]:
    if config_name != FORMAL_CONFIG_NAME:
        raise RuntimeError("P16-GQ accepts only its committed formal config")
    # Import Hydra only for resolving the already-committed wrapper config.  The
    # exact resolved object is stored in the preflight attestation.
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    old_root = os.environ.get("ROOT_DIR")
    os.environ["ROOT_DIR"] = str(repo_root)
    try:
        with initialize_config_dir(
            config_dir=str(repo_root / "code" / "config"), version_base=None
        ):
            cfg = compose(
                config_name=config_name,
                overrides=["shard_index=%d" % int(shard_index)],
            )
            resolved = OmegaConf.to_container(cfg, resolve=True)
    finally:
        if old_root is None:
            os.environ.pop("ROOT_DIR", None)
        else:
            os.environ["ROOT_DIR"] = old_root
    if not isinstance(resolved, dict):
        raise RuntimeError("resolved P16-GQ config is not a mapping")
    return resolved


def _resolved_config_flags(resolved: Mapping[str, Any]) -> Dict[str, Any]:
    dataset = resolved.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeError("resolved P16-GQ config has no dataset mapping")
    flags: Dict[str, Any] = {}
    for key in P16_GQ_CONFIG_FLAGS:
        if key == "dataset_hsi_mesh_root":
            value = dataset.get("hsi_mesh_root")
        else:
            value = resolved.get(key)
        if value is None:
            raise RuntimeError("resolved P16-GQ config is missing flag %s" % key)
        flags[key] = str(value) if key == "dataset_hsi_mesh_root" else copy.deepcopy(value)
    return flags


def _validate_resolved_config_contract(
    resolved: Mapping[str, Any], shard_index: int
) -> Dict[str, Any]:
    _reject_forbidden_30573(resolved, "resolved_config")
    if resolved.get("formal_wrapper") != FORMAL_WRAPPER_RELATIVE_PATH:
        raise RuntimeError("resolved P16-GQ config does not require the formal wrapper")
    if resolved.get("shard_count") != GQ_SHARD_COUNT:
        raise RuntimeError("resolved P16-GQ config does not declare shard_count=8")
    if resolved.get("shard_index") != int(shard_index):
        raise RuntimeError("resolved P16-GQ shard_index does not match the invocation")
    if resolved.get("expected_checkpoint_sha256") != SEALED_CHECKPOINT_SHA256:
        raise RuntimeError("resolved P16-GQ checkpoint expectation is not sealed")
    dataset = resolved.get("dataset")
    if not isinstance(dataset, Mapping):
        raise RuntimeError("resolved P16-GQ config has no dataset mapping")
    if dataset.get("hsi_mesh_root") != resolved.get("lingo_mesh_root"):
        raise RuntimeError("resolved dataset mesh root does not match lingo_mesh_root")
    flags = _resolved_config_flags(resolved)
    expected_flags = copy.deepcopy(P16_GQ_CONFIG_FLAGS)
    expected_flags["dataset_hsi_mesh_root"] = str(resolved["lingo_mesh_root"])
    difference = _exact_difference(expected_flags, flags, "config_flags")
    if difference:
        raise RuntimeError(
            "resolved P16-GQ treatment flags do not match the registered contract: %s"
            % difference
        )
    for key in ("ckpt_path", "lingo_episode_dir", "lingo_mesh_root", "lingo_output_dir"):
        if not isinstance(resolved.get(key), str) or not resolved[key]:
            raise RuntimeError("resolved P16-GQ config has no usable %s" % key)
    return flags


def _canonical_shard_output_path(
    repo_root: Path,
    resolved: Mapping[str, Any],
    shard_index: int,
    checkpoint_observed_sha256: str,
) -> Path:
    checkpoint_observed_sha256 = _hash(
        checkpoint_observed_sha256, "checkpoint_observed_sha256"
    )
    if checkpoint_observed_sha256 != SEALED_CHECKPOINT_SHA256:
        raise RuntimeError("P16-GQ observed checkpoint is not sealed")
    output_root = Path(str(resolved["lingo_output_dir"])).resolve()
    repo_root = Path(repo_root).resolve()
    try:
        output_root.relative_to(repo_root)
    except ValueError as error:
        raise RuntimeError(
            "P16-GQ output root must be checkout-local: %s" % output_root
        ) from error
    if output_root == repo_root:
        raise RuntimeError("P16-GQ output root must not be the checkout root")
    checkpoint_stem = Path(str(resolved["ckpt_path"])).stem
    output_dir = output_root / (
        "%s-%s-shard%02dof%02d"
        % (checkpoint_stem, checkpoint_observed_sha256[:12], int(shard_index), GQ_SHARD_COUNT)
    )
    return output_dir / "evaluation" / PAYLOAD_FILENAME


def _file_asset_record(
    path: Path, expected_sha256: str, label: str, **extra: Any
) -> Dict[str, Any]:
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError("P16-GQ asset is missing: %s" % path)
    observed = sha256_file(path)
    if observed != expected_sha256:
        raise RuntimeError(
            "%s hash mismatch: expected %s, got %s" % (label, expected_sha256, observed)
        )
    record: Dict[str, Any] = {
        "path": str(path),
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "size_bytes": path.stat().st_size,
    }
    record.update(extra)
    return record


def _verified_asset_records(
    preflight: Mapping[str, Any], resolved: Mapping[str, Any]
) -> Dict[str, Any]:
    asset_dir = Path(__file__).resolve().parent / "assets"
    index = _file_asset_record(
        asset_dir / "idx_area512.npy",
        AREA512_INDEX_SHA256,
        "P16-GQ area512 index",
        raw_int64_sha256=AREA512_INDEX_RAW_INT64_SHA256,
    )
    proxy = _file_asset_record(
        asset_dir / "body_proxy_area512.npz",
        BODY_PROXY_ASSET_SHA256,
        "P16-GQ derived proxy",
    )
    source = {
        "path": "SMPLX_MALE.npz",
        "expected_sha256": SMPLX_SOURCE_SHA256,
        "observed_sha256": preflight["proxy"]["source_sha256"],
        "size_bytes": SMPLX_SOURCE_SIZE_BYTES,
        "runtime_dependency": False,
    }
    if source["observed_sha256"] != SMPLX_SOURCE_SHA256:
        raise RuntimeError("P16-GQ preflight source hash is not frozen")
    checkpoint = {
        "path": str(Path(preflight["checkpoint"]["path"]).resolve()),
        "expected_sha256": str(resolved["expected_checkpoint_sha256"]),
        "observed_sha256": preflight["checkpoint"]["sha256"],
        "size_bytes": int(preflight["checkpoint"]["size_bytes"]),
    }
    if checkpoint["expected_sha256"] != SEALED_CHECKPOINT_SHA256:
        raise RuntimeError("P16-GQ checkpoint expected hash is not frozen")
    if checkpoint["observed_sha256"] != SEALED_CHECKPOINT_SHA256:
        raise RuntimeError("P16-GQ checkpoint observed hash is not frozen")
    if proxy["size_bytes"] != BODY_PROXY_ASSET_SIZE_BYTES:
        raise RuntimeError("P16-GQ derived proxy size is not frozen")
    return {
        "area512_index": index,
        "proxy_tables": proxy,
        "source_smplx": source,
        "checkpoint": checkpoint,
    }


def _validate_preflight_observation(
    preflight: Mapping[str, Any],
    resolved: Mapping[str, Any],
    scene_names: Sequence[str],
) -> None:
    _reject_forbidden_30573(preflight, "preflight")
    checkpoint = _require_mapping(preflight.get("checkpoint"), "preflight.checkpoint")
    expected_checkpoint = str(resolved["expected_checkpoint_sha256"])
    _hash(checkpoint.get("sha256"), "preflight checkpoint sha256")
    if checkpoint.get("sha256") != expected_checkpoint:
        raise RuntimeError("preflight checkpoint observed hash does not match config")
    if Path(str(checkpoint.get("path"))).resolve() != Path(
        str(resolved["ckpt_path"])
    ).resolve():
        raise RuntimeError("preflight checkpoint path does not match resolved config")
    proxy = _require_mapping(preflight.get("proxy"), "preflight.proxy")
    proxy_expectations = {
        "asset_sha256": BODY_PROXY_ASSET_SHA256,
        "asset_size_bytes": BODY_PROXY_ASSET_SIZE_BYTES,
        "source_sha256": SMPLX_SOURCE_SHA256,
        "weights_shape": [512, 22],
        "offsets_shape": [512, 22, 3],
        "posedirs_shape": [512, 3, 189],
    }
    for key, expected in proxy_expectations.items():
        if proxy.get(key) != expected:
            raise RuntimeError(
                "preflight proxy %s mismatch: expected %r, got %r"
                % (key, expected, proxy.get(key))
            )
    sdf_cache = _require_mapping(preflight.get("sdf_cache"), "preflight.sdf_cache")
    protocol_difference = _exact_difference(
        sdf_cache_protocol_identity(), sdf_cache.get("protocol"), "sdf_cache.protocol"
    )
    if protocol_difference:
        raise RuntimeError("preflight SDF protocol mismatch: %s" % protocol_difference)
    cache_env = os.environ.get("INFBAGEL_SDF_CACHE")
    if not cache_env:
        raise RuntimeError("P16-GQ requires INFBAGEL_SDF_CACHE")
    if Path(str(sdf_cache.get("root"))).resolve() != Path(cache_env).resolve():
        raise RuntimeError("preflight SDF root does not match INFBAGEL_SDF_CACHE")
    scenes = sdf_cache.get("scenes")
    if not isinstance(scenes, list):
        raise RuntimeError("preflight SDF scene records are missing")
    observed_names = sorted(
        str(item.get("scene_name")) for item in scenes if isinstance(item, Mapping)
    )
    if observed_names != sorted(str(name) for name in scene_names):
        raise RuntimeError(
            "preflight scenes do not match the planned shard: %r != %r"
            % (observed_names, sorted(scene_names))
        )
    for record in scenes:
        if not isinstance(record, Mapping):
            raise RuntimeError("preflight SDF scene record is not an object")
        _hash(record.get("sha256"), "preflight SDF cache sha256")
        _hash(record.get("mesh_sha256"), "preflight mesh sha256")
        if record.get("protocol_id") != sdf_cache_protocol_identity()["id"]:
            raise RuntimeError("preflight SDF scene protocol id mismatch")


def _episode_shard_block(
    manifest: Mapping[str, Any], shard_index: int
) -> Dict[str, Any]:
    if manifest.get("episode_count") != GQ_EPISODES:
        raise RuntimeError(
            "P16-GQ episode manifest has %d episodes, expected %d"
            % (manifest.get("episode_count"), GQ_EPISODES)
        )
    if manifest.get("window_count") != GQ_WINDOWS:
        raise RuntimeError(
            "P16-GQ episode manifest has %d windows, expected %d"
            % (manifest.get("window_count"), GQ_WINDOWS)
        )
    entries = manifest.get("entries")
    if not isinstance(entries, list) or len(entries) != GQ_EPISODES:
        raise RuntimeError("P16-GQ episode manifest entries are incomplete")
    shards = _plan_episode_shards([int(item["episode_num"]) for item in entries])
    selected = list(shards[int(shard_index)])
    selected_entries = [copy.deepcopy(entries[ordinal]) for ordinal in selected]
    return {
        "manifest": copy.deepcopy(dict(manifest)),
        "shard_index": int(shard_index),
        "shard_count": GQ_SHARD_COUNT,
        "canonical_episode_total": GQ_EPISODES,
        "canonical_window_total": GQ_WINDOWS,
        "selected_episode_ordinals": selected,
        "selected_episodes": selected_entries,
        "selected_window_total": int(
            sum(entries[ordinal]["episode_num"] for ordinal in selected)
        ),
        "partition_rule": "greedy_longest_first_bin_packing_by_window_count",
        "per_episode_seeding": "seed_everything(seed + canonical_ordinal)",
    }


def _build_preflight_attestation(
    *,
    repo_root: Path,
    config_name: str,
    resolved: Mapping[str, Any],
    manifest: Mapping[str, Any],
    shard_index: int,
    scene_names: Sequence[str],
    preflight: Mapping[str, Any],
    command: Sequence[str],
    sealed_evaluator_sha256: str,
) -> Tuple[Dict[str, Any], Path]:
    flags = _validate_resolved_config_contract(resolved, shard_index)
    _validate_preflight_observation(preflight, resolved, scene_names)
    assets = _verified_asset_records(preflight, resolved)
    checkpoint_observed = assets["checkpoint"]["observed_sha256"]
    output_path = _canonical_shard_output_path(
        repo_root, resolved, shard_index, checkpoint_observed
    )
    attestation_path = preflight_attestation_path(output_path)
    receipt_path = execution_receipt_path(output_path)
    episode_shard = _episode_shard_block(manifest, shard_index)
    treatment = {
        "guidance_mode": GQ_GUIDANCE_MODE,
        "guidance_version": GQ_GUIDANCE_VERSION,
        "sdf_weight": flags["hsi_guidance_sdf_weight"],
        "sdf_margin_m": GQ_SDF_MARGIN_M,
        "floor_threshold_m": GQ_FLOOR_THRESHOLD_M,
        "area512_index_sha256": assets["area512_index"]["observed_sha256"],
        "proxy_tables_sha256": assets["proxy_tables"]["observed_sha256"],
        "source_smplx_sha256": assets["source_smplx"]["observed_sha256"],
        "checkpoint_expected_sha256": assets["checkpoint"]["expected_sha256"],
        "checkpoint_observed_sha256": checkpoint_observed,
        "scene_mesh_sdf_cache_protocol": copy.deepcopy(
            preflight["sdf_cache"]["protocol"]
        ),
        "config_flags": copy.deepcopy(flags),
    }
    attestation: Dict[str, Any] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "protocol": ATTESTATION_PROTOCOL,
        "wrapper_used": True,
        "preflight_passed": True,
        "wrapper": {
            "path": FORMAL_WRAPPER_RELATIVE_PATH,
            "sha256": sha256_file(Path(__file__).resolve()),
        },
        "sealed_evaluator": {
            "path": str(SEALED_EVALUATOR_RELATIVE_PATH),
            "sha256": sealed_evaluator_sha256,
        },
        "resolved_config": {
            "config_name": config_name,
            "sha256": _canonical_digest(resolved),
            "values": copy.deepcopy(dict(resolved)),
        },
        "treatment": treatment,
        "assets": assets,
        "preflight": copy.deepcopy(dict(preflight)),
        "episode_shard": episode_shard,
        "output_binding": {
            "canonical_output_path": str(output_path),
            "canonical_output_dir": str(output_path.parent.parent),
            "preflight_attestation_path": str(attestation_path),
            "execution_receipt_path": str(receipt_path),
        },
        "execution": {
            "working_directory": str(Path(repo_root).resolve()),
            "root_dir": str(Path(repo_root).resolve()),
            "sdf_cache_root": str(Path(os.environ["INFBAGEL_SDF_CACHE"]).resolve()),
            "evaluator_command": [str(item) for item in command],
        },
    }
    _reject_forbidden_30573(attestation, "attestation")
    attestation["attestation_digest"] = _canonical_digest(attestation)
    return attestation, output_path


def _attestation_without_digest(attestation: Mapping[str, Any]) -> Dict[str, Any]:
    body = copy.deepcopy(dict(attestation))
    body.pop("attestation_digest", None)
    return body


def _validate_asset_records(
    attestation: Mapping[str, Any], *, verify_files: bool
) -> None:
    assets = _require_mapping(attestation.get("assets"), "attestation.assets")
    expected_asset_names = {"area512_index", "proxy_tables", "source_smplx", "checkpoint"}
    if set(assets) != expected_asset_names:
        raise ValueError("attestation assets are incomplete or contain extra claims")
    index = _require_mapping(assets.get("area512_index"), "assets.area512_index")
    if set(index) != {"path", "expected_sha256", "observed_sha256", "size_bytes", "raw_int64_sha256"}:
        raise ValueError("area512 index asset record is altered")
    proxy = _require_mapping(assets.get("proxy_tables"), "assets.proxy_tables")
    if set(proxy) != {"path", "expected_sha256", "observed_sha256", "size_bytes"}:
        raise ValueError("proxy asset record is altered")
    source = _require_mapping(assets.get("source_smplx"), "assets.source_smplx")
    if set(source) != {"path", "expected_sha256", "observed_sha256", "size_bytes", "runtime_dependency"}:
        raise ValueError("source asset record is altered")
    checkpoint = _require_mapping(assets.get("checkpoint"), "assets.checkpoint")
    if set(checkpoint) != {"path", "expected_sha256", "observed_sha256", "size_bytes"}:
        raise ValueError("checkpoint asset record is altered")
    expected_hashes = {
        "area512_index": (index, AREA512_INDEX_SHA256, AREA512_INDEX_SHA256),
        "proxy_tables": (proxy, BODY_PROXY_ASSET_SHA256, BODY_PROXY_ASSET_SHA256),
        "source_smplx": (source, SMPLX_SOURCE_SHA256, SMPLX_SOURCE_SHA256),
        "checkpoint": (checkpoint, SEALED_CHECKPOINT_SHA256, SEALED_CHECKPOINT_SHA256),
    }
    for name, (record, expected, observed) in expected_hashes.items():
        if record.get("expected_sha256") != expected or record.get("observed_sha256") != observed:
            raise ValueError("%s asset hash mismatch" % name)
        _hash(record.get("expected_sha256"), "%s expected hash" % name)
        _hash(record.get("observed_sha256"), "%s observed hash" % name)
    if index.get("raw_int64_sha256") != AREA512_INDEX_RAW_INT64_SHA256:
        raise ValueError("area512 index raw hash mismatch")
    if proxy.get("size_bytes") != BODY_PROXY_ASSET_SIZE_BYTES:
        raise ValueError("proxy asset size mismatch")
    if source.get("size_bytes") != SMPLX_SOURCE_SIZE_BYTES or source.get("runtime_dependency") is not False:
        raise ValueError("source asset record mismatch")
    if checkpoint.get("path") != str(Path(attestation["preflight"]["checkpoint"]["path"]).resolve()):
        raise ValueError("checkpoint asset path mismatch")
    if verify_files:
        expected_paths = {
            "area512_index": Path(__file__).resolve().parent / "assets" / "idx_area512.npy",
            "proxy_tables": Path(__file__).resolve().parent / "assets" / "body_proxy_area512.npz",
        }
        for name, expected_path in expected_paths.items():
            record = assets[name]
            if Path(str(record["path"])).resolve() != expected_path.resolve():
                raise ValueError("%s asset path is not the tracked asset" % name)
            if sha256_file(expected_path) != record["observed_sha256"]:
                raise ValueError("%s asset changed after preflight" % name)
        checkpoint_path = Path(str(checkpoint["path"]))
        if not checkpoint_path.is_file() or sha256_file(checkpoint_path) != checkpoint["observed_sha256"]:
            raise ValueError("checkpoint changed after preflight")


def _validate_episode_manifest_files(manifest: Mapping[str, Any]) -> None:
    actual = _load_episode_manifest(Path(str(manifest["root"])))
    difference = _exact_difference(actual, manifest, "episode_shard.manifest")
    if difference:
        raise ValueError("episode manifest changed after preflight: %s" % difference)


def _validate_cache_files(attestation: Mapping[str, Any]) -> None:
    preflight = _require_mapping(attestation.get("preflight"), "attestation.preflight")
    sdf_cache = _require_mapping(preflight.get("sdf_cache"), "preflight.sdf_cache")
    for record in sdf_cache.get("scenes", []):
        if not isinstance(record, Mapping):
            raise ValueError("preflight cache scene record is not an object")
        cache_path = Path(str(record["path"]))
        mesh_path = Path(str(record["mesh_path"]))
        if not cache_path.is_file() or sha256_file(cache_path) != record["sha256"]:
            raise ValueError("SDF cache changed after preflight: %s" % cache_path)
        if not mesh_path.is_file() or sha256_file(mesh_path) != record["mesh_sha256"]:
            raise ValueError("scene mesh changed after preflight: %s" % mesh_path)


def _expected_attestation_treatment(
    attestation: Mapping[str, Any], flags: Mapping[str, Any]
) -> Dict[str, Any]:
    assets = attestation["assets"]
    preflight = attestation["preflight"]
    return {
        "guidance_mode": GQ_GUIDANCE_MODE,
        "guidance_version": GQ_GUIDANCE_VERSION,
        "sdf_weight": flags["hsi_guidance_sdf_weight"],
        "sdf_margin_m": GQ_SDF_MARGIN_M,
        "floor_threshold_m": GQ_FLOOR_THRESHOLD_M,
        "area512_index_sha256": assets["area512_index"]["observed_sha256"],
        "proxy_tables_sha256": assets["proxy_tables"]["observed_sha256"],
        "source_smplx_sha256": assets["source_smplx"]["observed_sha256"],
        "checkpoint_expected_sha256": assets["checkpoint"]["expected_sha256"],
        "checkpoint_observed_sha256": assets["checkpoint"]["observed_sha256"],
        "scene_mesh_sdf_cache_protocol": copy.deepcopy(
            preflight["sdf_cache"]["protocol"]
        ),
        "config_flags": copy.deepcopy(dict(flags)),
    }


def _require_command_shape(
    command: Any, shard_index: int, config_name: str
) -> List[str]:
    if not isinstance(command, list) or len(command) != 5:
        raise ValueError("formal wrapper evaluator command must have no extra overrides")
    values = [str(item) for item in command]
    if not values[0] or not Path(values[1]).is_absolute():
        raise ValueError("formal wrapper evaluator command path is invalid")
    if Path(values[1]).as_posix().split("/")[-2:] != ["code", "test_infbagel_lingo_hsi.py"]:
        raise ValueError("formal wrapper evaluator path is altered")
    return [
        values[0],
        values[1],
        "--config-name",
        str(config_name),
        "shard_index=%d" % int(shard_index),
    ]


def validate_preflight_attestation(
    attestation: Mapping[str, Any], *, verify_files: bool = True
) -> Dict[str, Any]:
    """Verify an already-created attestation; never create one from claims."""
    if not isinstance(attestation, Mapping):
        raise ValueError("P16-GQ preflight attestation is not an object")
    _reject_forbidden_30573(attestation, "attestation")
    required = {
        "schema_version",
        "protocol",
        "wrapper_used",
        "preflight_passed",
        "wrapper",
        "sealed_evaluator",
        "resolved_config",
        "treatment",
        "assets",
        "preflight",
        "episode_shard",
        "output_binding",
        "execution",
        "attestation_digest",
    }
    if set(attestation) != required:
        raise ValueError("P16-GQ preflight attestation has missing or extra fields")
    if attestation.get("schema_version") != ATTESTATION_SCHEMA_VERSION:
        raise ValueError("unsupported P16-GQ attestation schema")
    if attestation.get("protocol") != ATTESTATION_PROTOCOL:
        raise ValueError("unsupported P16-GQ attestation protocol")
    if attestation.get("wrapper_used") is not True or attestation.get("preflight_passed") is not True:
        raise ValueError("P16-GQ attestation is not a successful wrapper preflight")
    digest = _hash(attestation.get("attestation_digest"), "attestation_digest")
    if digest != _canonical_digest(_attestation_without_digest(attestation)):
        raise ValueError("P16-GQ preflight attestation digest mismatch")

    wrapper = _require_mapping(attestation.get("wrapper"), "attestation.wrapper")
    if set(wrapper) != {"path", "sha256"} or wrapper.get("path") != FORMAL_WRAPPER_RELATIVE_PATH:
        raise ValueError("P16-GQ wrapper binding is altered")
    _hash(wrapper.get("sha256"), "wrapper sha256")
    if wrapper.get("sha256") != sha256_file(Path(__file__).resolve()):
        raise ValueError("P16-GQ wrapper source changed after preflight")
    sealed = _require_mapping(attestation.get("sealed_evaluator"), "attestation.sealed_evaluator")
    if set(sealed) != {"path", "sha256"} or sealed.get("path") != str(SEALED_EVALUATOR_RELATIVE_PATH):
        raise ValueError("sealed evaluator binding is altered")
    if sealed.get("sha256") != SEALED_EVALUATOR_SHA256:
        raise ValueError("sealed evaluator digest is not the protected digest")

    resolved_block = _require_mapping(attestation.get("resolved_config"), "attestation.resolved_config")
    if set(resolved_block) != {"config_name", "sha256", "values"}:
        raise ValueError("resolved config attestation is altered")
    if resolved_block.get("config_name") != FORMAL_CONFIG_NAME:
        raise ValueError("P16-GQ attestation config name is not formal")
    resolved = _require_mapping(resolved_block.get("values"), "resolved_config.values")
    _hash(resolved_block.get("sha256"), "resolved config sha256")
    if resolved_block.get("sha256") != _canonical_digest(resolved):
        raise ValueError("resolved config digest mismatch")
    shard = _require_mapping(attestation.get("episode_shard"), "attestation.episode_shard")
    shard_index = shard.get("shard_index")
    if type(shard_index) is not int or not 0 <= shard_index < GQ_SHARD_COUNT:
        raise ValueError("attestation shard index is invalid")
    flags = _validate_resolved_config_contract(resolved, shard_index)

    preflight = _require_mapping(attestation.get("preflight"), "attestation.preflight")
    selected_names = sorted(
        str(item["scene_name"])
        for item in shard.get("selected_episodes", [])
        if isinstance(item, Mapping)
    )
    _validate_preflight_observation(preflight, resolved, selected_names)
    _validate_asset_records(attestation, verify_files=verify_files)
    treatment_expected = _expected_attestation_treatment(attestation, flags)
    difference = _exact_difference(
        treatment_expected, attestation.get("treatment"), "treatment"
    )
    if difference:
        raise ValueError("P16-GQ treatment attestation mismatch: %s" % difference)

    manifest = _require_mapping(shard.get("manifest"), "episode_shard.manifest")
    _validate_episode_manifest_files(manifest)
    expected_shard = _episode_shard_block(manifest, shard_index)
    difference = _exact_difference(expected_shard, shard, "episode_shard")
    if difference:
        raise ValueError("episode/shard manifest mismatch: %s" % difference)

    output = _require_mapping(attestation.get("output_binding"), "attestation.output_binding")
    expected_output_keys = {
        "canonical_output_path",
        "canonical_output_dir",
        "preflight_attestation_path",
        "execution_receipt_path",
    }
    if set(output) != expected_output_keys:
        raise ValueError("output binding is incomplete or altered")
    output_path = _canonical_payload_path(Path(str(output["canonical_output_path"])))
    if str(output_path.parent.parent) != output.get("canonical_output_dir"):
        raise ValueError("canonical output directory binding is altered")
    if preflight_attestation_path(output_path).as_posix() != str(output["preflight_attestation_path"]):
        raise ValueError("preflight attestation path binding is altered")
    if execution_receipt_path(output_path).as_posix() != str(output["execution_receipt_path"]):
        raise ValueError("execution receipt path binding is altered")

    execution = _require_mapping(attestation.get("execution"), "attestation.execution")
    if set(execution) != {"working_directory", "root_dir", "sdf_cache_root", "evaluator_command"}:
        raise ValueError("execution binding is incomplete or altered")
    if execution.get("working_directory") != execution.get("root_dir"):
        raise ValueError("execution root binding is inconsistent")
    if Path(str(execution["sdf_cache_root"])).resolve() != Path(
        str(preflight["sdf_cache"]["root"])
    ).resolve():
        raise ValueError("execution SDF root binding is altered")
    command = execution.get("evaluator_command")
    expected_command = _require_command_shape(command, shard_index, resolved_block["config_name"])
    if command != expected_command:
        raise ValueError("sealed evaluator command binding is altered")
    if verify_files:
        _validate_cache_files(attestation)
    return copy.deepcopy(dict(attestation))


def _validate_raw_payload(
    payload: Mapping[str, Any], payload_path: Path, attestation: Mapping[str, Any]
) -> str:
    """Check only raw claims that the sealed evaluator itself emitted."""
    if not isinstance(payload, Mapping):
        raise ValueError("raw P16-GQ payload is not an object")
    _reject_forbidden_30573(payload, "raw_payload")
    for forbidden in (
        "treatment_identity",
        "treatment_attestation",
        "execution_receipt",
        "preflight_attestation",
    ):
        if forbidden in payload:
            raise ValueError("raw payload contains wrapper-owned claim %s" % forbidden)
    treatment = _require_mapping(attestation["treatment"], "attestation.treatment")
    flags = treatment["config_flags"]
    if payload.get("guided") is not True:
        raise ValueError("raw payload is not a guided payload")
    if payload.get("sample_type") != flags["sample_type"]:
        raise ValueError("raw payload sample_type does not match attestation")
    if payload.get("seed") != flags["seed"]:
        raise ValueError("raw payload seed does not match attestation")
    expected_dir = payload_path.parent.parent
    if not isinstance(payload.get("output_dir"), str) or Path(payload["output_dir"]).resolve() != expected_dir:
        raise ValueError("raw payload output_dir is not the canonical output")
    observed = _payload_checkpoint_sha256(payload)
    if observed != treatment["checkpoint_observed_sha256"]:
        raise ValueError("raw payload observed checkpoint does not match attestation")
    checkpoint = payload["checkpoint"]
    if "checkpoint_path" in checkpoint and Path(str(checkpoint["checkpoint_path"])).resolve() != Path(
        str(attestation["assets"]["checkpoint"]["path"])
    ).resolve():
        raise ValueError("raw payload checkpoint path does not match attestation")
    sharding = _require_mapping(payload.get("sharding"), "raw_payload.sharding")
    expected_shard = attestation["episode_shard"]
    required_sharding = {
        "shard_index": expected_shard["shard_index"],
        "shard_count": expected_shard["shard_count"],
        "canonical_episode_total": expected_shard["canonical_episode_total"],
        "canonical_window_total": expected_shard["canonical_window_total"],
        "shard_episode_ordinals": expected_shard["selected_episode_ordinals"],
        "shard_window_total": expected_shard["selected_window_total"],
        "partition_rule": expected_shard["partition_rule"],
        "per_episode_seeding": expected_shard["per_episode_seeding"],
    }
    for key, expected in required_sharding.items():
        if sharding.get(key) != expected:
            raise ValueError("raw payload sharding.%s does not match attestation" % key)
    if payload.get("sequence_count") != len(expected_shard["selected_episode_ordinals"]):
        raise ValueError("raw payload sequence_count does not match shard manifest")
    timing = _require_mapping(payload.get("timing"), "raw_payload.timing")
    if timing.get("window_count") != expected_shard["selected_window_total"]:
        raise ValueError("raw payload window_count does not match shard manifest")
    return observed


def _payload_checkpoint_sha256(payload: Mapping[str, Any]) -> str:
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, Mapping):
        raise ValueError("raw P16-GQ shard payload has no checkpoint block")
    return _hash(checkpoint.get("checkpoint_sha256"), "payload checkpoint_sha256")


def _receipt_body(
    *,
    attestation: Mapping[str, Any],
    payload_path: Path,
    raw_payload_sha256: str,
    checkpoint_observed_sha256: str,
) -> Dict[str, Any]:
    return {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "protocol": RECEIPT_PROTOCOL,
        "wrapper_used": True,
        "preflight_passed": True,
        "attestation_digest": attestation["attestation_digest"],
        "preflight_attestation_path": str(preflight_attestation_path(payload_path)),
        "raw_payload_sha256": raw_payload_sha256,
        "shard_index": attestation["episode_shard"]["shard_index"],
        "checkpoint_observed_sha256": checkpoint_observed_sha256,
        "canonical_output_path": str(payload_path),
        "resolved_config_sha256": attestation["resolved_config"]["sha256"],
    }


def _validate_execution_receipt(
    receipt: Mapping[str, Any],
    *,
    attestation: Mapping[str, Any],
    payload_path: Path,
    raw_payload_sha256: str,
    checkpoint_observed_sha256: str,
) -> Dict[str, Any]:
    if not isinstance(receipt, Mapping):
        raise ValueError("P16-GQ execution receipt is not an object")
    _reject_forbidden_30573(receipt, "execution_receipt")
    expected_body = _receipt_body(
        attestation=attestation,
        payload_path=payload_path,
        raw_payload_sha256=raw_payload_sha256,
        checkpoint_observed_sha256=checkpoint_observed_sha256,
    )
    required = set(expected_body) | {"receipt_digest"}
    if set(receipt) != required:
        raise ValueError("execution receipt is missing or has extra fields")
    difference = _exact_difference(
        expected_body,
        {key: receipt[key] for key in expected_body},
        "receipt",
    )
    if difference:
        raise ValueError("execution receipt binding mismatch: %s" % difference)
    receipt_digest = _hash(receipt.get("receipt_digest"), "receipt_digest")
    if receipt_digest != _canonical_digest(expected_body):
        raise ValueError("execution receipt digest mismatch")
    return copy.deepcopy(dict(receipt))


def _attach_execution_receipt(payload_path: Path) -> Path:
    """Attach only a receipt to a raw output after a verified wrapper run."""
    payload_path = _canonical_payload_path(payload_path)
    attestation_path = preflight_attestation_path(payload_path)
    receipt_path = execution_receipt_path(payload_path)
    if not attestation_path.is_file():
        raise ValueError("missing pre-created P16-GQ preflight attestation")
    if receipt_path.exists():
        raise FileExistsError("refusing to overwrite execution receipt: %s" % receipt_path)
    attestation = validate_preflight_attestation(_load_json(attestation_path), verify_files=True)
    payload = load_payload(payload_path)
    checkpoint_observed = _validate_raw_payload(payload, payload_path, attestation)
    raw_digest = sha256_file(payload_path)
    receipt_body = _receipt_body(
        attestation=attestation,
        payload_path=payload_path,
        raw_payload_sha256=raw_digest,
        checkpoint_observed_sha256=checkpoint_observed,
    )
    receipt = copy.deepcopy(receipt_body)
    receipt["receipt_digest"] = _canonical_digest(receipt_body)
    _validate_execution_receipt(
        receipt,
        attestation=attestation,
        payload_path=payload_path,
        raw_payload_sha256=raw_digest,
        checkpoint_observed_sha256=checkpoint_observed,
    )
    _write_json_atomic(receipt_path, receipt)
    return receipt_path


def validate_attested_shard_file(path: Path) -> Dict[str, Any]:
    """Validate one raw payload through its attestation and execution receipt."""
    payload_path = _canonical_payload_path(path)
    attestation_path = preflight_attestation_path(payload_path)
    receipt_path = execution_receipt_path(payload_path)
    if not attestation_path.is_file():
        raise ValueError("missing P16-GQ preflight attestation")
    if not receipt_path.is_file():
        raise ValueError("missing P16-GQ execution receipt")
    attestation = validate_preflight_attestation(_load_json(attestation_path), verify_files=True)
    payload = load_payload(payload_path)
    checkpoint_observed = _validate_raw_payload(payload, payload_path, attestation)
    raw_digest = sha256_file(payload_path)
    receipt = _validate_execution_receipt(
        _load_json(receipt_path),
        attestation=attestation,
        payload_path=payload_path,
        raw_payload_sha256=raw_digest,
        checkpoint_observed_sha256=checkpoint_observed,
    )
    return {
        "payload_path": str(payload_path),
        "payload": payload,
        "attestation": attestation,
        "receipt": receipt,
    }


def _merge_attestation_common(attestation: Mapping[str, Any]) -> Dict[str, Any]:
    resolved = copy.deepcopy(dict(attestation["resolved_config"]["values"]))
    resolved.pop("shard_index", None)
    shard = attestation["episode_shard"]
    return {
        "protocol": attestation["protocol"],
        "wrapper": copy.deepcopy(attestation["wrapper"]),
        "sealed_evaluator": copy.deepcopy(attestation["sealed_evaluator"]),
        "resolved_config_name": attestation["resolved_config"]["config_name"],
        "resolved_config_values_without_shard_index": resolved,
        "treatment": copy.deepcopy(attestation["treatment"]),
        "manifest": copy.deepcopy(shard["manifest"]),
        "shard_count": shard["shard_count"],
        "canonical_episode_total": shard["canonical_episode_total"],
        "canonical_window_total": shard["canonical_window_total"],
    }


def merge_gq_shard_payloads(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_episodes: int = GQ_EPISODES,
    expected_windows: int = GQ_WINDOWS,
) -> Dict[str, Any]:
    """Disabled mapping-only route: a raw mapping has no execution receipt."""
    raise ValueError(
        "P16-GQ merge requires canonical shard files with preflight attestations "
        "and execution receipts; in-memory payloads are not reportable"
    )


def merge_gq_shard_files(
    paths: Iterable[Path],
    *,
    expected_episodes: int = GQ_EPISODES,
    expected_windows: int = GQ_WINDOWS,
) -> Dict[str, Any]:
    """Validate receipt-backed shards, then delegate structural merge only."""
    paths = [Path(path) for path in paths]
    if not paths:
        raise ValueError("P16-GQ merge received no shard files")
    verified = [validate_attested_shard_file(path) for path in paths]
    indices = [int(item["attestation"]["episode_shard"]["shard_index"]) for item in verified]
    if len(indices) != GQ_SHARD_COUNT:
        raise ValueError("P16-GQ merge requires exactly eight receipt-backed shards")
    if sorted(indices) != list(range(GQ_SHARD_COUNT)):
        raise ValueError("P16-GQ merge has replayed or missing shard indices: %s" % indices)
    reference_common = _merge_attestation_common(verified[0]["attestation"])
    for item in verified[1:]:
        difference = _exact_difference(
            reference_common,
            _merge_attestation_common(item["attestation"]),
            "merge.attestation",
        )
        if difference:
            raise ValueError("P16-GQ shard attestations disagree: %s" % difference)

    repo_root = Path(__file__).resolve().parents[3]
    verify_sealed_evaluator_unchanged(repo_root)
    import test_infbagel_lingo_hsi as sealed_evaluator

    ordered = [
        item for _index, item in sorted(zip(indices, verified), key=lambda pair: pair[0])
    ]
    payloads = [item["payload"] for item in ordered]
    merged = sealed_evaluator.merge_shard_payloads(
        payloads,
        expected_episodes=int(expected_episodes),
        expected_windows=int(expected_windows),
        expected_shard_count=GQ_SHARD_COUNT,
    )
    merged["treatment_attestation"] = {
        "schema_version": ATTESTATION_SCHEMA_VERSION,
        "protocol": ATTESTATION_PROTOCOL,
        "wrapper_used": True,
        "preflight_passed": True,
        "attestation_digests": [
            item["attestation"]["attestation_digest"] for item in ordered
        ],
        "execution_receipt_digests": [
            item["receipt"]["receipt_digest"] for item in ordered
        ],
        "episode_manifest_sha256": reference_common["manifest"]["sha256"],
        "shard_count": GQ_SHARD_COUNT,
    }
    return merged


def verify_sealed_evaluator_unchanged(
    repo_root: Path, *, base: str = SEALED_BASE
) -> str:
    """Mechanically prove the wrapper did not edit the sealed evaluator."""
    repo_root = Path(repo_root).resolve()
    path = repo_root / SEALED_EVALUATOR_RELATIVE_PATH
    expected = subprocess.check_output(
        ["git", "show", "%s:%s" % (base, SEALED_EVALUATOR_RELATIVE_PATH)],
        cwd=str(repo_root),
    )
    actual = path.read_bytes()
    if actual != expected:
        raise RuntimeError(
            "sealed evaluator differs from %s: %s" % (base, SEALED_EVALUATOR_RELATIVE_PATH)
        )
    digest = hashlib.sha256(actual).hexdigest()
    if digest != SEALED_EVALUATOR_SHA256:
        raise RuntimeError("sealed evaluator digest is not the protected digest")
    return digest


def verify_complete_wrapper_delta(
    repo_root: Path, *, base: str = SEALED_BASE
) -> Dict[str, Any]:
    """Check the final delta has the HSI wrapper and no sealed/core edits."""
    repo_root = Path(repo_root).resolve()
    sealed_hash = verify_sealed_evaluator_unchanged(repo_root, base=base)
    changed = subprocess.check_output(
        ["git", "diff", "--name-only", "%s..HEAD" % base],
        cwd=str(repo_root),
        text=True,
    ).splitlines()
    if str(SEALED_EVALUATOR_RELATIVE_PATH) in changed:
        raise RuntimeError("sealed evaluator is present in the wrapper delta")
    if any(path.startswith("code/priors/core/") for path in changed):
        raise RuntimeError("frozen priors/core is present in the wrapper delta")
    if FORMAL_WRAPPER_RELATIVE_PATH not in changed:
        raise RuntimeError("HSI GQ wrapper is absent from the final delta")
    config_path = repo_root / "code" / "config" / (FORMAL_CONFIG_NAME + ".yaml")
    if str(config_path.relative_to(repo_root)) not in changed:
        raise RuntimeError("formal P16-GQ wrapper config is absent from the final delta")
    config = config_path.read_text(encoding="utf-8")
    if "formal_wrapper: %s" % FORMAL_WRAPPER_RELATIVE_PATH not in config:
        raise RuntimeError("formal P16-GQ config does not require the wrapper")
    if "formal_attestation: true" not in config:
        raise RuntimeError("formal P16-GQ config does not require attestation")
    return {"sealed_evaluator_sha256": sealed_hash, "changed_paths": changed}


def build_evaluator_command(
    repo_root: Path,
    python: str,
    shard_index: int,
    *,
    config_name: str = FORMAL_CONFIG_NAME,
    extra_overrides: Sequence[str] = (),
) -> List[str]:
    """Build the only formal invocation: a plain shard-index override."""
    if config_name != FORMAL_CONFIG_NAME:
        raise ValueError("P16-GQ accepts only its committed formal config")
    if type(shard_index) is not int or shard_index < 0 or shard_index >= GQ_SHARD_COUNT:
        raise ValueError("shard_index must be in 0..7")
    if extra_overrides:
        raise ValueError("P16-GQ wrapper does not accept caller config overrides")
    return [
        str(python),
        str(Path(repo_root).resolve() / SEALED_EVALUATOR_RELATIVE_PATH),
        "--config-name",
        FORMAL_CONFIG_NAME,
        "shard_index=%d" % shard_index,
    ]


def run_gq_shard(
    repo_root: Path,
    *,
    python: str,
    shard_index: int,
    config_name: str = FORMAL_CONFIG_NAME,
) -> Path:
    """Preflight, attest, run one sealed shard, and write its receipt."""
    repo_root = Path(repo_root).resolve()
    if config_name != FORMAL_CONFIG_NAME:
        raise RuntimeError("P16-GQ accepts only its committed formal config")
    if not bool(os.environ.get("INFBAGEL_SDF_CACHE")):
        raise RuntimeError("INFBAGEL_SDF_CACHE must be set before a P16-GQ shard")
    sealed_hash = verify_sealed_evaluator_unchanged(repo_root)
    resolved = _resolve_formal_config(repo_root, config_name, int(shard_index))
    _validate_resolved_config_contract(resolved, int(shard_index))
    manifest = _load_episode_manifest(Path(str(resolved["lingo_episode_dir"])))
    if manifest["episode_count"] != GQ_EPISODES or manifest["window_count"] != GQ_WINDOWS:
        raise RuntimeError("formal P16-GQ requires the fixed 375-episode/2271-window manifest")
    shard_block = _episode_shard_block(manifest, int(shard_index))
    scene_names = sorted({item["scene_name"] for item in shard_block["selected_episodes"]})
    preflight = run_formal_preflight(
        repo_root=repo_root,
        checkpoint_path=Path(str(resolved["ckpt_path"])),
        dataset_root=repo_root / "data" / "dataset",
        mesh_root=Path(str(resolved["lingo_mesh_root"])),
        scene_names=scene_names,
        expected_checkpoint_sha256=str(resolved["expected_checkpoint_sha256"]),
    )
    command = build_evaluator_command(
        repo_root, python, int(shard_index), config_name=config_name
    )
    attestation, output_path = _build_preflight_attestation(
        repo_root=repo_root,
        config_name=config_name,
        resolved=resolved,
        manifest=manifest,
        shard_index=int(shard_index),
        scene_names=scene_names,
        preflight=preflight,
        command=command,
        sealed_evaluator_sha256=sealed_hash,
    )
    attestation_path = preflight_attestation_path(output_path)
    receipt_path = execution_receipt_path(output_path)
    if output_path.parent.parent.exists():
        raise FileExistsError(
            "refusing to overwrite P16-GQ output: %s" % output_path.parent.parent
        )
    if attestation_path.exists() or receipt_path.exists():
        raise FileExistsError("refusing to reuse P16-GQ attestation/receipt")
    _write_json_atomic(attestation_path, attestation)

    env = dict(os.environ)
    env["ROOT_DIR"] = str(repo_root)
    code_root = str(repo_root / "code")
    env["PYTHONPATH"] = code_root + os.pathsep + env.get("PYTHONPATH", "")
    completed = subprocess.run(
        command,
        cwd=str(repo_root),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("sealed P16-GQ shard failed:\n%s" % completed.stdout[-12000:])
    reported_path = None
    for line in reversed(completed.stdout.splitlines()):
        if line.startswith("Wrote "):
            reported_path = Path(line[6:].strip())
            break
    if reported_path is None:
        raise RuntimeError("sealed P16-GQ shard did not report its payload path")
    if not reported_path.is_absolute():
        reported_path = (repo_root / reported_path).resolve()
    if reported_path.resolve() != output_path.resolve():
        raise RuntimeError(
            "sealed P16-GQ shard output is not the pre-attested canonical path: %s != %s"
            % (reported_path, output_path)
        )
    if not output_path.is_file():
        raise RuntimeError("sealed P16-GQ shard did not create its canonical payload")
    _attach_execution_receipt(output_path)
    return output_path


def canonical_treatment_identity(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Disabled legacy API; canonical constants cannot bless a raw payload."""
    raise RuntimeError(
        "canonical treatment decoration is disabled; use the formal wrapper run"
    )


def attach_treatment_identity(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Disabled legacy API retained only to fail closed for old callers."""
    raise RuntimeError(
        "attach_treatment_identity is disabled; raw payloads require a wrapper receipt"
    )


def decorate_payload_file(*args: Any, **kwargs: Any) -> Path:
    """Disabled legacy file route; it never reads or rewrites a payload."""
    raise RuntimeError(
        "decorate_payload_file is disabled; arbitrary payload decoration is forbidden"
    )


def validate_treatment_identity(*args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Disabled legacy validator; identity validation is receipt/path based."""
    raise RuntimeError(
        "legacy treatment_identity validation is disabled; use validate_attested_shard_file"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    merge = subparsers.add_parser("merge")
    merge.add_argument("payloads", nargs="+", type=Path)
    merge.add_argument("--expected-episodes", type=int, default=GQ_EPISODES)
    merge.add_argument("--expected-windows", type=int, default=GQ_WINDOWS)
    run = subparsers.add_parser("run")
    run.add_argument("--repo-root", type=Path, default=Path.cwd())
    run.add_argument("--python", required=True)
    run.add_argument("--shard-index", type=int, required=True)
    run.add_argument("--config-name", default=FORMAL_CONFIG_NAME)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "merge":
        merged = merge_gq_shard_files(
            args.payloads,
            expected_episodes=args.expected_episodes,
            expected_windows=args.expected_windows,
        )
        json.dump(merged, sys.stdout, indent=2, sort_keys=True, allow_nan=False)
        sys.stdout.write("\n")
        return 0
    print(
        run_gq_shard(
            args.repo_root,
            python=args.python,
            shard_index=args.shard_index,
            config_name=args.config_name,
        )
    )
    return 0


__all__ = [
    "ATTESTATION_PROTOCOL",
    "ATTESTATION_SCHEMA_VERSION",
    "EXECUTION_RECEIPT_SUFFIX",
    "FORMAL_CONFIG_NAME",
    "FORMAL_WRAPPER_RELATIVE_PATH",
    "GQ_SHARD_COUNT",
    "P16_GQ_CONFIG_FLAGS",
    "SEALED_CHECKPOINT_SHA256",
    "build_evaluator_command",
    "canonical_treatment_identity",
    "decorate_payload_file",
    "execution_receipt_path",
    "load_payload",
    "merge_gq_shard_files",
    "merge_gq_shard_payloads",
    "preflight_attestation_path",
    "run_gq_shard",
    "validate_attested_shard_file",
    "validate_preflight_attestation",
    "verify_complete_wrapper_delta",
    "verify_sealed_evaluator_unchanged",
]


if __name__ == "__main__":
    raise SystemExit(main())
