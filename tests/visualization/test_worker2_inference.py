import copy
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest

from tools.worker2_inference import (
    DEFAULT_PROFILES,
    TOPOLOGY,
    WorkflowError,
    build_completion_script,
    build_preflight_script,
    build_request,
    control_ssh_argv,
    load_profiles,
    select_profile,
    validate_profile,
)


def _metadata(run_id="p1-hoi-p12-frame-repair-baseline-s42-20260819"):
    return {
        "schema_version": 2,
        "checkpoint_type": "hoi_prior_phase1b",
        "expert": "hoi",
        "run_id": run_id,
        "seed": 42,
        "git_commit": "2" * 40,
        "primary_weight_variant": "online",
        "data_contract_sha256": "3" * 64,
    }


def _request(tmp_path, checkpoint_name="p12 weight.pth"):
    profiles, registry_sha = load_profiles()
    metadata = _metadata()
    profile = select_profile(profiles, metadata)
    return build_request(
        checkpoint=tmp_path / checkpoint_name,
        checkpoint_sha256="a" * 64,
        metadata=metadata,
        profile=profile,
        profile_registry=DEFAULT_PROFILES,
        profile_registry_sha256=registry_sha,
        artifact_root=tmp_path / "returned artifacts",
        created_at=datetime(2026, 8, 25, tzinfo=timezone.utc),
    )


def test_registered_p12_profile_pins_inference_commit_separately_from_training():
    profiles, digest = load_profiles()
    selected = select_profile(profiles, _metadata())

    assert len(digest) == 64
    assert selected["name"] == "hoi-p12-armb"
    assert selected["inference_commit"] == "8742d1a3b88800161324a8e45c597ffafdcbb607"
    assert selected["inference_commit"] != _metadata()["git_commit"]
    assert selected["legacy_human_frame"] == "y_up"


def test_registered_p15_profile_pins_matched_guided_evaluator():
    profiles, _ = load_profiles()
    metadata = _metadata("p1-hoi-p15-geom-perhand-equalized-s42-20260825")
    metadata["git_commit"] = "b85c7459b5a6fef1975e9ec5013fbb7849cb2aa0"
    selected = select_profile(profiles, metadata)

    assert selected["name"] == "hoi-p15-armb"
    assert selected["inference_commit"] == metadata["git_commit"]
    assert selected["run_variant"] == "p15-armb-motion-export"
    assert selected["expected_sequences"] == 438
    assert selected["legacy_human_frame"] == "y_up"
    assert selected["hydra_overrides"] == [
        "sampler.pelvis.guidance.enabled=true",
        "sampler.pelvis.guidance.arm=b",
        "sampler.pelvis.guidance.guidance_scale=1000.0",
        "sampler.pelvis.guidance.last_steps=10",
        "sampler.pelvis.guidance.clamp=1.0",
        "sampler.pelvis.guidance.clamp_target=update",
    ]


def test_auto_profile_fails_closed_for_an_unregistered_checkpoint():
    profiles, _ = load_profiles()

    with pytest.raises(WorkflowError, match="no registered inference profile"):
        select_profile(profiles, _metadata("p1-hoi-p13-future-s42-20260825"))


def test_profile_cannot_override_workflow_owned_fields():
    profiles, _ = load_profiles()
    profile = copy.deepcopy(profiles[0])
    profile["hydra_overrides"].append("save_motion_params=false")

    with pytest.raises(WorkflowError, match="protected field"):
        validate_profile(profile)


def test_request_needs_only_checkpoint_derived_metadata_and_is_deterministic(tmp_path):
    request = _request(tmp_path)

    assert request["run_id"] == "viz-hoi-p12-armb-motion-export-aaaaaaaaaaaa-s42-20260825"
    assert request["checkpoint"]["authority_path"].endswith("p12 weight.pth")
    assert request["checkpoint"]["worker_path"].endswith("/" + "a" * 64 + "/p12 weight.pth")
    assert request["worker"]["checkout"].endswith(
        "/work/checkouts/8742d1a3b88800161324a8e45c597ffafdcbb607"
    )
    assert request["authority"]["artifact_staging"].endswith(
        "/hoi/.%s.incoming" % request["run_id"]
    )
    assert request["network"]["windows_proxy_used"] is False
    assert request["inference"]["argv"][-1] == "save_motion_params=true"


def test_control_plane_is_loopback_only_and_ignores_ssh_config():
    argv = control_ssh_argv()

    assert argv[0] == "/usr/bin/ssh"
    assert argv[argv.index("-F") + 1] == "/dev/null"
    assert argv[-1] == "yujinlun@127.0.0.1"
    assert str(TOPOLOGY["control_port"]) in argv
    assert not any("Proxy" in item for item in argv)


def test_preflight_uses_worker_initiated_direct_transfers_and_detached_checkout(tmp_path):
    request = _request(tmp_path)
    script = build_preflight_script(request)

    assert "10.184.17.253" in script
    assert "id_ed25519_infbagel_authority_transfer" in script
    assert "-F /dev/null" in script
    assert "rsync -a --protect-args" in script
    assert "AUTHORITY_CHECK_COMMAND=" in script
    assert '"$AUTH_TARGET" "$AUTHORITY_CHECK_COMMAND"' in script
    assert '"$AUTH_TARGET" /bin/bash -lc' not in script
    assert "worktree add --detach" in script
    assert request["profile"]["inference_commit"] in script
    assert request["checkpoint"]["metadata"]["git_commit"] not in script
    assert "nvidia-smi -i 0" in script
    assert "--cfg job --resolve" in script
    assert "systemd-run --user" in script
    assert "--collect" not in script
    assert "save_motion_params=true" in script
    assert "http_proxy" not in script.lower()
    assert "https_proxy" not in script.lower()


def test_completion_is_retryable_after_generation_and_promotes_atomically(tmp_path):
    request = _request(tmp_path)
    script = build_completion_script(request)

    assert 'if [ ! -f "$JOB_DIR/provenance/generation_complete.json" ]' in script
    assert "motion_params.sha256" in script
    assert "PREPARE_COMMAND=" in script
    assert "VERIFY_COMMAND=" in script
    assert script.count("rsync -a --protect-args") == 3
    assert '"$AUTH_TARGET" "$VERIFY_COMMAND"' in script
    assert '"$AUTH_TARGET" /bin/bash -lc' not in script
    assert "sha256sum -c ../provenance/motion_params.sha256" in script
    assert request["authority"]["artifact_staging"] in script
    assert request["authority"]["artifact_final"] in script
    assert "/usr/bin/mv" in script
    assert script.index("sha256sum -c") < script.index("/usr/bin/mv")


def test_generated_remote_shell_is_valid_with_spaces_in_paths(tmp_path):
    script = build_preflight_script(_request(tmp_path))

    result = subprocess.run(
        ["/bin/bash", "-n"],
        input=script,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert result.returncode == 0, result.stderr


def test_inference_script_never_mutates_authority_expert_worktrees(tmp_path):
    script = build_preflight_script(_request(tmp_path))

    assert "git -C /data/yujinlun/InfBaGel-release checkout" not in script
    assert "/data/yujinlun/InfBaGel-hsi" not in script
    assert "phase/01b-hoi" not in script
    assert "phase/01c-hsi" not in script
