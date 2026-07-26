#!/usr/bin/env python3
"""Run the authority-only D2-AC0 CPU implementation contract.

This diagnostic is intentionally evaluator-independent and does not create an
optimizer, update a parameter, load a checkpoint, or access motion data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict

import torch

REPO = Path(__file__).resolve().parents[1]
import sys

sys.path.insert(0, str(REPO / "code"))

from priors.diffusion import HOIPriorSampler  # noqa: E402
from priors.interaction_adapter import (  # noqa: E402
    ADAPTER_PARAMETER_COUNT,
    ASSIGNMENT_SHA256,
    BPS_SHA256,
    cluster_bps_features,
    load_bps_partition,
)
from priors.models import (  # noqa: E402
    HOI_ARCHITECTURE_D2AC,
    assert_parameter_independence,
    build_expert,
    load_trained_hoi_prior,
)
from train_hoi_prior import _d2ac_gradient_audit  # noqa: E402


def _state_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().contiguous().cpu()
        digest.update(name.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _inputs(batch: int = 2):
    generator = torch.Generator().manual_seed(42)
    return (
        torch.randn(batch, 16, 232, generator=generator),
        torch.arange(batch, dtype=torch.long) * 249 % 500,
        torch.randn(batch, 768, generator=generator),
        torch.randn(batch, 1024, 3, generator=generator),
        torch.randn(batch, 9, generator=generator),
        torch.randn(batch, 3, generator=generator),
    )


def run_contract(repo: Path) -> Dict[str, object]:
    torch.manual_seed(42)
    basis, assignment, basis_means, sizes, partition = load_bps_partition(
        repo / "code/bps.pt"
    )
    values = _inputs()
    features = cluster_bps_features(
        values[3], basis, assignment, basis_means, sizes
    )
    if tuple(features.shape) != (2, 16, 10) or not torch.isfinite(features).all():
        raise RuntimeError("interaction-adapter-contract-failure-stop: invalid local features")

    torch.manual_seed(42)
    base = build_expert("hoi", dim_model=512, num_heads=16, num_layers=8)
    torch.manual_seed(99)
    model = build_expert(
        "hoi", dim_model=512, num_heads=16, num_layers=8,
        architecture_variant=HOI_ARCHITECTURE_D2AC,
    )
    base_state = base.state_dict()
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if not missing or unexpected:
        raise RuntimeError("interaction-adapter-contract-failure-stop: trunk sharing failed")
    base.eval()
    model.eval()
    with torch.no_grad():
        base_output = base(*_inputs(batch=1))
        model_output = model(*_inputs(batch=1))
    parity = float((base_output - model_output).abs().max())
    if parity > 1.0e-6:
        raise RuntimeError(
            f"interaction-adapter-contract-failure-stop: base parity {parity}"
        )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    adapter_count = sum(
        parameter.numel()
        for parameter in model.network.interaction_adapter.parameters()
    )
    if parameter_count != 30_023_145 or adapter_count != ADAPTER_PARAMETER_COUNT:
        raise RuntimeError("interaction-adapter-contract-failure-stop: parameter count mismatch")

    model.train()
    prediction = model(*values)
    (prediction - values[0]).square().mean().backward()
    initial_audit = _d2ac_gradient_audit(
        model, require_adapter_paths=False
    )
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        model.network.interaction_adapter.alpha.copy_(
            torch.atanh(torch.tensor(0.1))
        )
    prediction = model(*values)
    (prediction - values[0]).square().mean().backward()
    activated_audit = _d2ac_gradient_audit(
        model, require_adapter_paths=True
    )

    model.eval()
    with torch.no_grad():
        model.network.interaction_adapter.set_diagnostic_variant("full")
        model.network.interaction_adapter.set_gate_override(0.1)
        full = model(*_inputs(batch=1))
        model.network.interaction_adapter.set_diagnostic_variant(
            "local_correspondence_permuted"
        )
        model.network.interaction_adapter.set_gate_override(0.1)
        permuted = model(*_inputs(batch=1))
    permutation_effect = float((full - permuted).abs().max())
    if permutation_effect <= 1.0e-8:
        raise RuntimeError("interaction-adapter-contract-failure-stop: locality permutation is inert")

    hsi = build_expert("hsi", dim_model=32, num_heads=4, num_layers=1)
    assert_parameter_independence(model, hsi)
    sampler = HOIPriorSampler(device="cpu", auto_regre_num=2, timesteps=500)
    del sampler

    source = (
        (repo / "code/priors/interaction_adapter.py").read_text(encoding="utf-8")
        + (repo / "code/priors/models.py").read_text(encoding="utf-8")
    )
    forbidden = (
        "eval_metrics", "near_ground", "contact_guidance",
        "stored_per_frame_bps", "future_gt", "contact_label",
    )
    static_passed = not any(value in source for value in forbidden)
    if not static_passed:
        raise RuntimeError("interaction-adapter-contract-failure-stop: forbidden model-path input")

    # A malformed D2-AC checkpoint must be rejected before any state is loaded.
    with tempfile.TemporaryDirectory() as temporary:
        malformed = Path(temporary) / "malformed.pth"
        torch.save({
            "checkpoint_type": "hoi_prior_phase1b",
            "expert": "hoi",
            "initialization": "random",
            "model_config": {
                "dim_model": 512, "num_heads": 16, "num_layers": 8,
                "architecture_variant": HOI_ARCHITECTURE_D2AC,
            },
        }, malformed)
        try:
            load_trained_hoi_prior(str(malformed), torch.device("cpu"))
        except ValueError as error:
            provenance_rejected = "provenance" in str(error)
        else:
            provenance_rejected = False
    if not provenance_rejected:
        raise RuntimeError("interaction-adapter-contract-failure-stop: malformed checkpoint accepted")

    return {
        "schema_version": 1,
        "run_id": "p1-hoi-d2ac-cpu-contract-s42-20260726",
        "classification": "cpu-contract-passed",
        "bps": partition,
        "feature_shape": list(features.shape),
        "feature_dtype": str(features.dtype),
        "feature_finite": bool(torch.isfinite(features).all()),
        "parameter_count": parameter_count,
        "adapter_parameter_count": adapter_count,
        "base_parity_max_abs": parity,
        "initial_alpha_gradient": initial_audit,
        "activated_adapter_gradients": activated_audit,
        "local_permutation_max_abs": permutation_effect,
        "hsiprior_parameter_storage_independent": True,
        "mixer_clean_output_contract": [2, 16, 232],
        "checkpoint_variant_rejection": provenance_rejected,
        "static_model_path_scan_passed": static_passed,
        "optimizer_created": False,
        "optimizer_updates": 0,
        "checkpoint_loads": 0,
        "checkpoint_writes": 0,
        "bps_sha256": BPS_SHA256,
        "assignment_sha256": ASSIGNMENT_SHA256,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    try:
        result = run_contract(args.repo_root.resolve())
    except Exception as error:
        print(f"interaction-adapter-contract-failure-stop: {error}", flush=True)
        raise
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
