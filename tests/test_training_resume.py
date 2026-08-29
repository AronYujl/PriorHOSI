#!/usr/bin/env python3
"""Kill/resume continuity checks for ``code/train_infbagel.py``.

Run this module as a script, like the other project-wide test modules.  The
driver starts a straight-through reference, hard-kills a second process group
after an atomic epoch-boundary save, resumes it, and requires bitwise-identical
parameters and Adam moments.  The default one-rank run covers 72 optimizer
updates; ``RESUME_CHECK_WORLD_SIZE=2`` covers the odd epoch boundary and 36
updates on two GPUs.
"""

import argparse
import json
import os
import random
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "code"))
os.environ.setdefault("ROOT_DIR", str(REPO))

import train_infbagel as trainer  # noqa: E402


SEED = 42
WARMUP_UPDATES = 10
ACCUMULATION_STEPS = 2
MICRO_BATCH = 4
DATASET_SIZE = 25
BASE_LR = 2e-4
EPOCHS = 24
KILL_AFTER_EPOCH = 10
PRECISION = "bf16_tf32"


class DeterministicDataset(Dataset):
    def __len__(self):
        return DATASET_SIZE

    def __getitem__(self, index):
        base = torch.arange(8, dtype=torch.float32)
        return (
            torch.sin(base * (index + 1) * 0.37),
            torch.cos(base * (index + 1) * 0.11),
        )


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.input = nn.Linear(8, 32)
        self.output = nn.Linear(32, 8)
        self.unused = nn.Parameter(torch.zeros(4))

    def forward(self, value):
        return self.output(torch.relu(self.input(value)))


class Config(dict):
    """Minimal attribute-access config for the real resume helpers."""

    def __getattr__(self, name):
        return self[name]


def _worker(rank, world_size, args):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    random.seed(SEED + rank)
    np.random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    torch.cuda.manual_seed_all(SEED + rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    torch.manual_seed(1234)
    model = TinyModel().to(device)
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        broadcast_buffers=False,
        find_unused_parameters=True,
    )
    torch.manual_seed(SEED + rank)
    optimizer = Adam(model.parameters(), lr=BASE_LR)

    dataset = DeterministicDataset()
    sampler = DistributedSampler(dataset, seed=SEED)
    dataloader = DataLoader(
        dataset,
        batch_size=MICRO_BATCH,
        drop_last=True,
        num_workers=0,
        sampler=sampler,
    )
    config = Config(
        batch_size=MICRO_BATCH,
        gradient_accumulation_steps=ACCUMULATION_STEPS,
        effective_batch_size=MICRO_BATCH * world_size * ACCUMULATION_STEPS,
        lr=BASE_LR,
        seed=SEED,
        sample_type="diffusion",
        precision=PRECISION,
        exp_dir=args.exp_dir,
        exp_name="resume_check",
        start_epoch=0,
    )
    geometry = trainer.resume_geometry(
        config, world_size, len(dataloader), WARMUP_UPDATES
    )

    micro_steps = 0
    optimizer_updates = 0
    start_epoch = 0
    resume_state = None
    scaler = None

    if args.mode == "phase2":
        resume_state = torch.load(args.resume, map_location="cpu")
        trainer.check_resume_compatibility(resume_state, geometry)
        model.module.load_state_dict(resume_state["model"], strict=True)
        optimizer.load_state_dict(resume_state["optimizer"])
        if resume_state["grad_scaler"] is not None:
            raise AssertionError("bf16 resume state unexpectedly contains a GradScaler")
        trainer.restore_rng_state(resume_state["rng_states"][rank], device)
        start_epoch = int(resume_state["next_epoch"])
        micro_steps = int(resume_state["micro_steps"])
        optimizer_updates = int(resume_state["optimizer_updates"])

    lr_scheduler = trainer.build_lr_scheduler(
        optimizer, BASE_LR, WARMUP_UPDATES, optimizer_updates
    )
    if resume_state is not None:
        assert int(resume_state["lr_scheduler"]["last_epoch"]) == int(
            lr_scheduler.last_epoch
        )

    optimizer.zero_grad(set_to_none=True)
    restored_gradients = 0
    if resume_state is not None:
        restored_gradients = trainer.restore_pending_gradients(
            model, resume_state.get("pending_gradients")
        )
        resume_state = None

    lr_trace = []
    gradient_digest_at_save = None
    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        for inputs, targets in dataloader:
            micro_steps += 1
            inputs = inputs.to(device)
            targets = targets.to(device)
            noise = torch.randn_like(inputs) * 0.05
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                loss = ((model(inputs + noise) - targets) ** 2).mean()
            (loss / ACCUMULATION_STEPS).backward()
            if micro_steps % ACCUMULATION_STEPS == 0:
                lr_used = float(optimizer.param_groups[0]["lr"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                lr_scheduler.step()
                lr_trace.append([optimizer_updates, lr_used])

        if args.mode == "phase1" and epoch == args.kill_after_epoch:
            digest = torch.zeros(1, dtype=torch.float64, device=device)
            for parameter in model.module.parameters():
                if parameter.grad is not None:
                    digest += parameter.grad.double().abs().sum()
            gathered_digests = [torch.zeros_like(digest) for _ in range(world_size)]
            dist.all_gather(gathered_digests, digest)
            gradient_digest_at_save = [
                float(value.item()) for value in gathered_digests
            ]

            rng_states = [None] * world_size
            dist.all_gather_object(rng_states, trainer.collect_rng_state(device))
            if rank == 0:
                pending_gradients = (
                    trainer.collect_pending_gradients(model)
                    if micro_steps % ACCUMULATION_STEPS != 0
                    else None
                )
                trainer.atomic_torch_save(
                    trainer.build_resume_state(
                        geometry,
                        epoch,
                        True,
                        micro_steps,
                        optimizer_updates,
                        model,
                        optimizer,
                        lr_scheduler,
                        scaler,
                        rng_states,
                        pending_gradients,
                    ),
                    trainer.resume_state_path(args.exp_dir, "resume_check"),
                )
            dist.barrier()
            break

    if rank == 0:
        summary = {
            "mode": args.mode,
            "steps_per_epoch": len(dataloader),
            "micro_steps": micro_steps,
            "optimizer_updates": optimizer_updates,
            "lr_trace": lr_trace,
            "restored_pending_gradients": restored_gradients,
            "grad_digest_by_rank_at_save": gradient_digest_at_save,
        }
        Path(args.out).write_text(json.dumps(summary), encoding="utf-8")
        torch.save(
            {
                name: parameter.detach().cpu()
                for name, parameter in model.module.named_parameters()
            },
            args.out + ".params",
        )
        torch.save(trainer.detach_to_cpu(optimizer.state_dict()), args.out + ".optim")

    dist.barrier()
    if args.mode == "phase1":
        if rank == 0:
            Path(args.ready).touch()
        while True:
            torch.cuda.synchronize()
    dist.destroy_process_group()


def _worker_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--world-size", type=int, required=True)
    parser.add_argument("--kill-after-epoch", type=int, default=-1)
    parser.add_argument("--exp-dir", required=True)
    parser.add_argument("--resume", default="")
    parser.add_argument("--out", required=True)
    parser.add_argument("--ready", default="")
    args = parser.parse_args()
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ.setdefault("MASTER_PORT", "29577")
    torch.multiprocessing.spawn(
        _worker, args=(args.world_size, args), nprocs=args.world_size, join=True
    )


def _tensors_identical(left, right):
    return set(left) == set(right) and all(
        torch.equal(left[key], right[key]) for key in left
    )


def _adam_state_identical(left, right):
    left_state = left["state"]
    right_state = right["state"]
    if set(left_state) != set(right_state):
        return False
    for key in left_state:
        for field in ("step", "exp_avg", "exp_avg_sq"):
            left_value = left_state[key].get(field)
            right_value = right_state[key].get(field)
            if torch.is_tensor(left_value) or torch.is_tensor(right_value):
                if not torch.equal(
                    torch.as_tensor(left_value), torch.as_tensor(right_value)
                ):
                    return False
            elif left_value != right_value:
                return False
    return True


class TrainingResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA is required for the bf16 resume test")
        cls.world_size = int(os.environ.get("RESUME_CHECK_WORLD_SIZE", "1"))
        if cls.world_size not in (1, 2):
            raise ValueError("RESUME_CHECK_WORLD_SIZE must be 1 or 2")
        if torch.cuda.device_count() < cls.world_size:
            raise unittest.SkipTest(
                f"requested {cls.world_size} GPUs, found {torch.cuda.device_count()}"
            )
        cls.python = os.environ.get("INFBAGEL_PYTHON")
        if not cls.python:
            raise RuntimeError("INFBAGEL_PYTHON must name the verified infbagel interpreter")

    def _launch(self, work, extra, expect_kill=False, ready=None):
        command = [
            self.python,
            str(Path(__file__).resolve()),
            "--worker",
            "--exp-dir",
            str(work),
            "--world-size",
            str(self.world_size),
        ] + extra
        if ready is not None:
            command += ["--ready", str(ready)]
        if not expect_kill:
            subprocess.run(command, check=True, cwd=REPO / "code")
            return

        process = subprocess.Popen(
            command, cwd=REPO / "code", start_new_session=True
        )
        deadline = time.time() + 300
        while not ready.exists():
            if time.time() > deadline:
                process.kill()
                self.fail("phase1 never reported a completed checkpoint")
            if process.poll() is not None:
                self.fail(f"phase1 exited early with {process.returncode}")
            time.sleep(0.2)
        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
        process.wait()
        self.assertLess(process.returncode, 0)

    def test_bf16_autocast_kill_resume_is_bitwise_continuous(self):
        with tempfile.TemporaryDirectory(prefix="infbagel-resume-test-") as temp:
            work = Path(temp)
            resume_path = Path(trainer.resume_state_path(work, "resume_check"))

            straight_out = work / "straight.json"
            self._launch(
                work,
                ["--mode", "straight", "--epochs", str(EPOCHS), "--out", str(straight_out)],
            )

            phase1_out = work / "phase1.json"
            ready = work / "ready"
            self._launch(
                work,
                [
                    "--mode", "phase1", "--epochs", str(EPOCHS),
                    "--out", str(phase1_out), "--kill-after-epoch",
                    str(KILL_AFTER_EPOCH),
                ],
                expect_kill=True,
                ready=ready,
            )
            self.assertTrue(resume_path.exists())
            self.assertFalse(Path(str(resume_path) + ".tmp").exists())

            phase2_out = work / "phase2.json"
            self._launch(
                work,
                [
                    "--mode", "phase2", "--epochs", str(EPOCHS),
                    "--out", str(phase2_out), "--resume", str(resume_path),
                ],
            )

            straight = json.loads(straight_out.read_text(encoding="utf-8"))
            phase1 = json.loads(phase1_out.read_text(encoding="utf-8"))
            phase2 = json.loads(phase2_out.read_text(encoding="utf-8"))
            joined_lr = phase1["lr_trace"] + phase2["lr_trace"]
            self.assertEqual(joined_lr, straight["lr_trace"])
            expected_updates = 72 if self.world_size == 1 else 36
            self.assertEqual(len(joined_lr), expected_updates)

            straddles = straight["steps_per_epoch"] % ACCUMULATION_STEPS != 0
            digests = phase1["grad_digest_by_rank_at_save"]
            self.assertEqual(len(set(digests)), 1)
            if straddles:
                self.assertGreater(phase2["restored_pending_gradients"], 0)
            else:
                self.assertEqual(phase2["restored_pending_gradients"], 0)

            self.assertTrue(
                _tensors_identical(
                    torch.load(str(straight_out) + ".params"),
                    torch.load(str(phase2_out) + ".params"),
                )
            )
            self.assertTrue(
                _adam_state_identical(
                    torch.load(str(straight_out) + ".optim"),
                    torch.load(str(phase2_out) + ".optim"),
                )
            )

            state = torch.load(resume_path, map_location="cpu")
            self.assertIsNone(state["grad_scaler"])
            self.assertEqual(state["geometry"]["precision"], PRECISION)
            for field, value in (
                ("world_size", self.world_size + 1),
                ("gradient_accumulation_steps", 4),
                ("steps_per_epoch", 99),
                ("lr", 1e-3),
                ("warmup_updates", 500),
                ("precision", "fp32"),
            ):
                changed = dict(state["geometry"])
                changed[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    trainer.check_resume_compatibility(state, changed)

    def test_atomic_save_preserves_previous_checkpoint_on_write_failure(self):
        with tempfile.TemporaryDirectory(prefix="infbagel-atomic-test-") as temp:
            path = Path(temp) / "checkpoint.pth"
            trainer.atomic_torch_save({"good": torch.ones(3)}, path)
            real_save = torch.save

            def failing_save(payload, handle, *args, **kwargs):
                handle.write(b"\x00" * 4096)
                raise RuntimeError("simulated interrupted write")

            torch.save = failing_save
            try:
                with self.assertRaises(RuntimeError):
                    trainer.atomic_torch_save({"bad": torch.zeros(3)}, path)
            finally:
                torch.save = real_save
            self.assertTrue(torch.equal(torch.load(path)["good"], torch.ones(3)))


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker_main()
    else:
        unittest.main()
