import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import json
import math
import random
import time
from pathlib import Path

import numpy as np

os.environ['ROOT_DIR'] = '..'
os.environ['HYDRA_FULL_ERROR'] = '1'
os.environ['CURRENT_TIME'] = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'
os.environ['NCCL_IB_DISABLE'] = '0'

import sys
sys.path.append(os.path.join(os.environ['ROOT_DIR'], 'code'))

# batch fields consumed by the training step (the only tensors moved to GPU each iteration;
# GT / metadata fields returned by the dataset are intentionally left on CPU)
TRAIN_BATCH_KEYS = (
    'joints', 'mat', 'object_trans', 'object_rot_mat', 'scene_flag',
    'text_clip_embedding', 'pelvis_goal', 'scene_goal', 'object_goal',
    'need_scene', 'need_pelvis_dir', 'pi', 'need_pi', 'is_loco', 'is_object',
    'obj_bps_data', 'obj_rot_mat_ref', 'rest_pose_obj_nn_pts', 'transformed_obj_verts', 'object_points',
    'global_rot_6d', 'contact_label', 'rest_human_offsets', 'seg_len', 'end_pi',
)

# ---------------------------------------------------------------------------
# Resumable training state.
#
# `{exp_name}_epoch{NNN}.pth` keeps its historical meaning exactly: it is
# `model.module.state_dict()` and nothing else, under the same name, so every
# downstream consumer (evaluation, distillation, checkpoint audits) is
# unaffected.  Everything a restart additionally needs -- optimizer moments, the
# epoch and optimizer-update counters, the warmup position, an AMP scaler if one
# ever exists, the per-rank RNG, and any gradients pending inside an
# accumulation group the epoch boundary cut in half -- lives in one separate,
# rolling artifact `{exp_name}_resume.pth`.
#
# Resume granularity is the EPOCH BOUNDARY, matching the existing checkpoint
# cadence.  `DistributedSampler.set_epoch(e)` seeds its permutation from
# `seed + e`, so a resumed epoch replays exactly the permutation an
# uninterrupted run would have used: no window is re-shown and none is skipped.
# Mid-epoch resume is not supported and is not silently approximated.
# ---------------------------------------------------------------------------

RESUME_STATE_VERSION = 1

# Fields whose value silently changes the optimization trajectory if it differs
# between the process that saved and the process that resumes.
RESUME_GEOMETRY_FIELDS = (
    'world_size',
    'micro_batch_per_gpu',
    'gradient_accumulation_steps',
    'effective_batch_size',
    'steps_per_epoch',
    'warmup_updates',
    'lr',
    'grad_clip_max_norm',
    'seed',
    'sample_type',
    'precision',
)


def flush_grad_norms(buffer, path, rank):
    norm_values = torch.stack([entry[3] for entry in buffer]).cpu().tolist()
    with open(path, 'a', encoding='utf-8') as handle:
        for (update, t_wall, t_mono, _), grad_norm in zip(buffer, norm_values):
            handle.write(json.dumps({
                'rank': rank,
                'update': update,
                'grad_norm': grad_norm,
                't_wall': t_wall,
                't_mono': t_mono,
            }) + '\n')
    buffer.clear()


def resume_state_path(exp_dir, exp_name):
    return os.path.join(str(exp_dir), 'checkpoints', f'{exp_name}_resume.pth')


def atomic_torch_save(payload, path):
    """Write to a temporary file, fsync, then rename over `path`.

    A process killed mid-write therefore never leaves a truncated file under
    `path`: the previously completed checkpoint stays intact and remains the one
    a resume picks up.
    """
    path = str(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = f'{path}.tmp'
    with open(temporary_path, 'wb') as handle:
        torch.save(payload, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary_path, path)
    directory_fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def build_lr_scheduler(optimizer, base_lr, warmup_updates, completed_updates):
    """Linear warmup measured in OPTIMIZER UPDATES, never in micro-steps.

    `completed_updates` is the number of `optimizer.step()` calls already made.
    The same value always maps to the same learning rate, so the schedule is
    continuous whether the process reached that update in one go or across a
    restart.
    """
    for param_group in optimizer.param_groups:
        param_group['initial_lr'] = base_lr
    # (update + 1) because the loop calls optimizer.step() before
    # lr_scheduler.step(): the u-th update must already run at the warmed lr,
    # not at the value for u-1.  Without the offset the first update runs at lr
    # exactly 0 and is a no-op.
    return LambdaLR(
        optimizer,
        lr_lambda=lambda update: 1.0 if warmup_updates == 0 else min((update + 1) / warmup_updates, 1.0),
        last_epoch=int(completed_updates) - 1,
    )


# Per-update diagnostics for the unfrozen cfg_scale_embedding.  Ordered, and the
# order is the on-disk field order, because the flush stacks one tensor per field
# per update into a single device-to-host copy.
CFG_DIAGNOSTIC_FIELDS = (
    'grad_norm_preclip',
    'grad_norm_trunk',
    'grad_norm_cfg',
    'clip_coefficient',
    'loss',
    'cfg_weight_rms',
    'cfg_bias_rms',
    'cfg_param_norm',
    'cfg_param_delta_rel',
    'cfg_student_target_dist',
    'cfg_student_target_rel',
)


def flush_cfg_diagnostics(buffer, path, rank):
    """One device-to-host copy for the whole buffer, same contract as
    `flush_grad_norms`: nothing in the update loop may call `.item()`.

    `grad_norm_*` and `clip_coefficient` are measured BEFORE `optimizer.step()`;
    every `cfg_*` parameter statistic is measured AFTER it.  A non-finite
    gradient anywhere makes `grad_norm_preclip` non-finite, so that field is
    itself the non-finite detector and no extra pass over the gradients is
    needed.
    """
    width = len(CFG_DIAGNOSTIC_FIELDS)
    values = torch.stack([value for entry in buffer for value in entry[2]]).double().cpu().tolist()
    with open(path, 'a', encoding='utf-8') as handle:
        for position, entry in enumerate(buffer):
            record = {
                'rank': rank,
                'update': entry[0],
                'lr': entry[1],
            }
            record.update(zip(CFG_DIAGNOSTIC_FIELDS, values[position * width:(position + 1) * width]))
            handle.write(json.dumps(record) + '\n')
    buffer.clear()


def detach_to_cpu(value):
    """Deep-copy every tensor in a state dict onto the CPU.

    Saving CUDA tensors would tag them with the saving rank's device, and
    `torch.load` would then materialize the whole payload on `cuda:0` on every
    rank at once.  CPU tensors load anywhere; `Optimizer.load_state_dict` and
    `Module.load_state_dict` move them back to the right device themselves.
    """
    if torch.is_tensor(value):
        return value.detach().to('cpu', copy=True)
    if isinstance(value, dict):
        return {key: detach_to_cpu(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(detach_to_cpu(item) for item in value)
    return value


def collect_rng_state(device):
    """This rank's python / numpy / torch-CPU / torch-CUDA generator state."""
    return {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch_cpu': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
    }


def restore_rng_state(state, device):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch_cpu'].cpu())
    if state['torch_cuda'] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(state['torch_cuda'].cpu(), device)


def collect_pending_gradients(model):
    """Gradients of an accumulation group that an epoch boundary cut in half.

    `is_accumulation_boundary` counts micro-steps globally, so with an odd
    number of steps per epoch an accumulation group straddles the boundary and
    real gradients are sitting in `.grad` when the checkpoint is written.
    Dropping them would make one update of the resumed run half-size.

    DDP synchronizes on every micro-step here (the loop deliberately does not
    use `no_sync`), so every rank holds the same reduced `.grad` and rank 0's
    copy is authoritative for all of them.
    """
    return {
        name: parameter.grad.detach().to('cpu', copy=True)
        for name, parameter in model.module.named_parameters()
        if parameter.grad is not None
    }


def restore_pending_gradients(model, pending_gradients):
    if not pending_gradients:
        return 0
    restored = 0
    for name, parameter in model.module.named_parameters():
        saved = pending_gradients.get(name)
        if saved is not None:
            parameter.grad = saved.to(device=parameter.device, dtype=parameter.dtype, copy=True)
            restored += 1
    return restored


def resume_geometry(cfg, world_size, steps_per_epoch, warmup_updates):
    return {
        'world_size': int(world_size),
        'micro_batch_per_gpu': int(cfg.batch_size),
        'gradient_accumulation_steps': int(cfg.gradient_accumulation_steps),
        'effective_batch_size': int(cfg.effective_batch_size),
        'steps_per_epoch': int(steps_per_epoch),
        'warmup_updates': int(warmup_updates),
        'lr': float(cfg.lr),
        'grad_clip_max_norm': (
            None if cfg.get('grad_clip_max_norm', None) is None
            else float(cfg.get('grad_clip_max_norm'))
        ),
        'seed': int(cfg.seed),
        'sample_type': str(cfg.sample_type),
        'precision': str(cfg.get('precision', 'fp32')),
    }


def check_resume_compatibility(state, geometry):
    """Reject, loudly, any resume that would silently change the schedule.

    Resuming at a different world size (or micro-batch, or accumulation) keeps
    the optimizer moments but changes the effective batch and the data sharding,
    so the run would no longer be the preregistered one.  It is rejected rather
    than adapted.  A resume is therefore supported at any world size, including
    4 and 8, provided it is the world size that wrote the state.
    """
    version = int(state.get('schema_version', -1))
    if version != RESUME_STATE_VERSION:
        raise ValueError(
            f'resume state schema version {version} is not {RESUME_STATE_VERSION}; '
            f'this state was written by a different trainer and is not resumable'
        )
    if not bool(state.get('epoch_completed', False)):
        raise ValueError(
            f'resume state was written after epoch {state.get("epoch")} stopped mid-epoch at the '
            f'optimizer-update budget; the run is finished and there is nothing to resume'
        )
    saved = state['geometry']
    mismatches = [
        f'{field}: saved {saved.get(field)!r} != current {geometry[field]!r}'
        for field in RESUME_GEOMETRY_FIELDS
        if saved.get(field) != geometry[field]
    ]
    if mismatches:
        raise ValueError(
            'refusing to resume: the training geometry changed, which would silently alter the '
            'preregistered schedule.\n  ' + '\n  '.join(mismatches)
        )


def resolve_resume_path(cfg):
    """`''` starts fresh, `'auto'` resumes the default path when it exists, and
    an explicit path must exist."""
    requested = str(cfg.get('resume_from', '') or '').strip()
    if requested == '':
        return None
    default_path = resume_state_path(cfg.exp_dir, cfg.exp_name)
    if requested == 'auto':
        return default_path if os.path.exists(default_path) else None
    if not os.path.exists(requested):
        raise FileNotFoundError(f'resume_from={requested!r} does not exist')
    return requested


def build_resume_state(geometry, epoch, epoch_completed, micro_steps, optimizer_updates,
                       model, optimizer, lr_scheduler, scaler, rng_states, pending_gradients,
                       target_model=None):
    return {
        'schema_version': RESUME_STATE_VERSION,
        'geometry': geometry,
        'epoch': int(epoch),
        'epoch_completed': bool(epoch_completed),
        'next_epoch': int(epoch) + 1,
        'micro_steps': int(micro_steps),
        'optimizer_updates': int(optimizer_updates),
        'model': detach_to_cpu(model.module.state_dict()),
        # The consistency objective EMA-updates target_model (infbagel.py:262),
        # so it is trained state, not a reconstructible constant.  The teacher is
        # frozen and is rebuilt identically from ckpt_path.
        'target_model': None if target_model is None else detach_to_cpu(target_model.module.state_dict()),
        'optimizer': detach_to_cpu(optimizer.state_dict()),
        'lr_scheduler': lr_scheduler.state_dict(),
        'grad_scaler': scaler.state_dict() if scaler is not None else None,
        'rng_states': rng_states,
        'pending_gradients': pending_gradients,
    }


@hydra.main(version_base=None, config_path="config", config_name="config_train_infbagel")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        # Required before any NCCL object collective (the checkpoint gathers the
        # per-rank RNG state); without it every rank would stage its payload on
        # cuda:0.
        torch.cuda.set_device(rank)
    precision = str(cfg.get('precision', 'fp32'))
    if precision == 'bf16_tf32':
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        use_bf16_autocast = True
    elif precision == 'fp32':
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        use_bf16_autocast = False
    else:
        raise ValueError(
            f'unsupported precision {precision!r}; expected "bf16_tf32" or "fp32"'
        )
    cfg.device = f"cuda:{rank}"
    random.seed(int(cfg.seed) + rank)
    np.random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)
    torch.cuda.manual_seed_all(int(cfg.seed) + rank)
    print(f'Training on {device}', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # cfg.sample_type selects the training objective:
    #   diffusion   -> standard diffusion training via trainer.p_losses
    #   consistency -> consistency-model distillation via trainer.consistency_loss
    is_consistency = cfg.sample_type == 'consistency'

    # Absent, this defaults to no clipping at all, which is what every run before it did.
    grad_clip_max_norm = cfg.get('grad_clip_max_norm', None)
    grad_clip_max_norm = None if grad_clip_max_norm is None else float(grad_clip_max_norm)
    if grad_clip_max_norm is not None and not math.isfinite(grad_clip_max_norm):
        raise ValueError(
            'grad_clip_max_norm must be finite; an infinite max_norm turns every gradient '
            'into NaN (see the clipping comment in the update loop)'
        )

    if is_consistency:
        teacher_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        # NOT .eval().  init_model leaves teacher and target in train() mode, and that
        # dropout is load-bearing: with update_ema(rate=0.95) the target is nearly the
        # student itself, so the consistency condition is near-degenerate and the noise
        # is what breaks the degeneracy.  Calling .eval() here collapsed the student
        # inside 656 updates -- measured, see the ablation in the commit message.
        teacher_model.requires_grad_(False)

        student_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        student_model.requires_grad_(False)
        student_model.module.embedding_input.requires_grad_(True)
        student_model.module.embedding_output.requires_grad_(True)
        student_model.module.transformer.requires_grad_(True)
        student_model.module.out.requires_grad_(True)
        # This sibling module is outside the four blocks above; without it, w conditioning cannot learn.
        student_model.module.cfg_scale_embedding.requires_grad_(True)

        target_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        target_model.requires_grad_(False)

        model = student_model
        optimizer = Adam(student_model.parameters(), lr=cfg.lr)
    else:
        model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        optimizer = Adam(model.parameters(), lr=cfg.lr)

    infbagel_dataset = hydra.utils.instantiate(cfg.dataset)

    expected_effective_batch = int(cfg.batch_size) * world_size * int(cfg.gradient_accumulation_steps)
    if expected_effective_batch != int(cfg.effective_batch_size):
        raise ValueError(
            f'effective batch mismatch: {cfg.batch_size} x {world_size} x '
            f'{cfg.gradient_accumulation_steps} = {expected_effective_batch}, '
            f'configured {cfg.effective_batch_size}'
        )

    sampler = DistributedSampler(infbagel_dataset, seed=int(cfg.seed))
    dataloader = DataLoader(infbagel_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True, persistent_workers=cfg.num_workers > 0)

    micro_steps = int(cfg.start_epoch) * len(dataloader)
    optimizer_updates = micro_steps // int(cfg.gradient_accumulation_steps)
    warmup_updates = int(cfg.get('warmup_updates', 0))
    steps_per_epoch = len(dataloader)
    geometry = resume_geometry(cfg, world_size, steps_per_epoch, warmup_updates)
    log_grad_norm = bool(cfg.get('log_grad_norm', False))
    grad_norm_buffer = []
    if log_grad_norm:
        grad_norm_dir = os.path.join(cfg.exp_dir, 'grad_norms')
        os.makedirs(grad_norm_dir, exist_ok=True)
        grad_norm_path = os.path.join(grad_norm_dir, f'grad_norms_rank{rank}.jsonl')

    log_cfg_diagnostics = is_consistency and bool(cfg.get('log_cfg_embedding_diagnostics', False))
    cfg_diagnostic_buffer = []
    if log_cfg_diagnostics:
        cfg_diagnostic_dir = os.path.join(cfg.exp_dir, 'cfg_diagnostics')
        os.makedirs(cfg_diagnostic_dir, exist_ok=True)
        cfg_diagnostic_path = os.path.join(cfg_diagnostic_dir, f'cfg_diagnostics_rank{rank}.jsonl')
        # Plain lists for the gradient-norm decomposition only.  The optimizer has a
        # single parameter group; these never touch it.
        cfg_embedding_parameters = list(student_model.module.cfg_scale_embedding.parameters())
        cfg_embedding_parameter_ids = {id(parameter) for parameter in cfg_embedding_parameters}
        trunk_parameters = [
            parameter for parameter in student_model.parameters()
            if id(parameter) not in cfg_embedding_parameter_ids
        ]
        cfg_embedding_named = dict(student_model.module.cfg_scale_embedding.named_parameters())
        cfg_weight_parameter = cfg_embedding_named['proj.weight']
        cfg_bias_parameter = cfg_embedding_named['proj.bias']
        target_cfg_embedding_parameters = list(target_model.module.cfg_scale_embedding.parameters())

    # bf16 has fp32-like exponent range and does not use GradScaler.  Keeping
    # this explicitly None also preserves the resume contract: a checkpoint
    # carrying an fp16 scaler state is rejected rather than silently ignored.
    scaler = None

    resume_path = resolve_resume_path(cfg)
    resume_state = None
    if resume_path is None:
        start_epoch = int(cfg.start_epoch)
        if rank == 0:
            print(f'No resume state in use; starting at epoch {start_epoch}', flush=True)
    else:
        resume_state = torch.load(resume_path, map_location='cpu')
        check_resume_compatibility(resume_state, geometry)
        model.module.load_state_dict(resume_state['model'], strict=True)
        if is_consistency:
            if resume_state.get('target_model') is None:
                raise ValueError('resume state has no EMA target_model but this is a consistency run')
            target_model.module.load_state_dict(resume_state['target_model'], strict=True)
        optimizer.load_state_dict(resume_state['optimizer'])
        if resume_state['grad_scaler'] is not None and scaler is None:
            raise ValueError('resume state carries an AMP GradScaler but this run has no scaler')
        if resume_state['grad_scaler'] is not None:
            scaler.load_state_dict(resume_state['grad_scaler'])
        restore_rng_state(resume_state['rng_states'][rank], device)
        start_epoch = int(resume_state['next_epoch'])
        micro_steps = int(resume_state['micro_steps'])
        optimizer_updates = int(resume_state['optimizer_updates'])
        if int(cfg.start_epoch) not in (0, start_epoch):
            raise ValueError(
                f'start_epoch={int(cfg.start_epoch)} contradicts the resume state, which continues '
                f'at epoch {start_epoch}; leave start_epoch at 0 when resuming'
            )
        if rank == 0:
            print(
                f'Resuming from {resume_path}: epoch {start_epoch}, '
                f'optimizer update {optimizer_updates}, micro step {micro_steps}',
                flush=True,
            )

    lr_scheduler = build_lr_scheduler(optimizer, float(cfg.lr), warmup_updates, optimizer_updates)
    if resume_state is not None:
        saved_position = int(resume_state['lr_scheduler']['last_epoch'])
        if saved_position != int(lr_scheduler.last_epoch):
            raise ValueError(
                f'lr schedule position mismatch on resume: saved {saved_position}, '
                f'rebuilt {int(lr_scheduler.last_epoch)}'
            )
        # Host RAM, not VRAM, is the binding constraint for this run, so the CPU
        # copies are consumed and released here rather than held until the loop.
        optimizer.zero_grad(set_to_none=True)
        restored_grads = restore_pending_gradients(model, resume_state.pop('pending_gradients', None))
        if rank == 0 and restored_grads:
            print(f'Restored {restored_grads} pending gradients from an unfinished '
                  f'accumulation group', flush=True)
        resume_state = None

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    if is_consistency:
        trainer.set_dataset_and_model(infbagel_dataset, student_model, teacher_model, target_model)
    else:
        trainer.set_dataset_and_model(infbagel_dataset, model)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    last_loss = None
    stop_training = False
    torch.cuda.reset_peak_memory_stats(device)
    if resume_path is None:
        # The resume branch above already zeroed the gradients and then restored
        # the pending accumulation group; do not wipe it here.
        optimizer.zero_grad(set_to_none=True)

    for epoch in range(start_epoch, cfg.epochs):
        print(f'Start epoch {epoch}', flush=True)
        sampler.set_epoch(epoch)

        step = 0
        for batch in dataloader:
            step += 1
            micro_steps += 1

            # async H2D copy for the training tensors (DataLoader sets pin_memory=True)
            b = {k: batch[k].to(device, non_blocking=True) for k in TRAIN_BATCH_KEYS}

            joints, mat, object_trans, object_rot_mat, scene_flag = \
                b['joints'], b['mat'], b['object_trans'], b['object_rot_mat'], b['scene_flag']
            text_clip_embedding, pelvis_goal, scene_goal, object_goal = \
                b['text_clip_embedding'], b['pelvis_goal'], b['scene_goal'], b['object_goal']
            need_scene, need_pelvis_dir, pi, need_pi, is_loco, is_object = \
                b['need_scene'], b['need_pelvis_dir'], b['pi'], b['need_pi'], b['is_loco'], b['is_object']
            obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, object_points = \
                b['obj_bps_data'], b['obj_rot_mat_ref'], b['rest_pose_obj_nn_pts'], b['transformed_obj_verts'], b['object_points']
            contact_label, rest_human_offsets, seg_len, end_pi = \
                b['contact_label'], b['rest_human_offsets'], b['seg_len'], b['end_pi']

            global_rot_6d = b['global_rot_6d'].reshape(b['global_rot_6d'].shape[0], b['global_rot_6d'].shape[1], -1)

            local_batch_size = joints.shape[0]
            t = torch.randint(0, trainer.timesteps, (local_batch_size,), device=device).long()
            x_start = torch.cat([joints, global_rot_6d, object_trans, object_rot_mat.reshape(object_rot_mat.shape[0], object_rot_mat.shape[1], -1), contact_label], dim=-1) # 84 + 132 + 3 + 9 + 4
            with torch.no_grad():
                mask, _, _ = get_mask(x_start, -1, p=1., fixed_frame=cfg.auto_regre_num)

            with torch.autocast(
                device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16_autocast
            ):
                if is_consistency:
                    loss_dict = trainer.consistency_loss(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
                        need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

                    loss_consistency, loss_object, loss_fk = \
                        loss_dict['loss_consistency'], loss_dict['loss_object'], loss_dict['loss_fk']

                    loss = loss_consistency
                    if loss_object is not None:
                        loss = loss + cfg.loss_w_obj_pts * loss_object
                    if loss_fk is not None:
                        loss = loss + cfg.loss_w_fk * loss_fk
                else:
                    loss_dict = trainer.p_losses(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
                        need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

                    loss, loss_object, loss_fk = \
                        loss_dict['loss'], loss_dict['loss_object'], loss_dict['loss_fk']

                    if loss_object is not None:
                        loss = loss + cfg.loss_w_obj_pts * loss_object
                    if loss_fk is not None:
                        loss = loss + cfg.loss_w_fk * loss_fk

            if step % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, LR: {current_lr:.6f}", flush=True)
                if cfg.use_tensorboard and rank == 0:
                    writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)
                    if is_consistency:
                        writer.add_scalar('Loss_consistency', loss_consistency.item(), epoch * len(dataloader) + step)
                    if loss_object is not None:
                        writer.add_scalar('Loss_object', loss_object.item(), epoch * len(dataloader) + step)
                    if loss_fk is not None:
                        writer.add_scalar('Loss_fk', loss_fk.item(), epoch * len(dataloader) + step)

            last_loss = float(loss.detach().item())
            is_accumulation_boundary = micro_steps % int(cfg.gradient_accumulation_steps) == 0
            # Keep DDP synchronization on every micro-step. This is slightly
            # slower than wrapping the entire forward/backward in no_sync(),
            # but preserves simple, identical semantics for both objectives.
            (loss / int(cfg.gradient_accumulation_steps)).backward()

            if is_accumulation_boundary:
                if log_grad_norm or grad_clip_max_norm is not None or log_cfg_diagnostics:
                    # Do not use clip_grad_norm_ with max_norm=inf.  Torch 1.13.1
                    # computes max_norm / (total_norm + 1e-6), clamps that value
                    # to at most 1.0, and multiplies every gradient in place.  A
                    # finite norm gives a bitwise-neutral multiplier of 1.0, but
                    # an infinite norm makes inf / inf NaN and silently overwrites
                    # every gradient with NaN.  It also adds a redundant write
                    # pass over all gradients.
                    # The comprehension is inlined deliberately.  Binding the
                    # gradient list to a local name keeps every gradient tensor
                    # alive past the zero_grad(set_to_none=True) below, so the
                    # old set would still be resident while the next backward
                    # allocates a fresh one: a measured +49.5 MiB per rank that
                    # the instrument would then be reporting as the run's own.
                    total_norm = torch.norm(torch.stack([
                        torch.norm(parameter.grad.detach(), 2.0)
                        for parameter in model.parameters()
                        if parameter.grad is not None
                    ]), 2.0)
                    # This loop deliberately does not use no_sync, so DDP has
                    # already averaged every gradient across ranks.  This is the
                    # global averaged-gradient norm that a future max_norm would
                    # use, and every rank logs it so equality is checkable.
                    # Keeping the scalar on the GPU avoids a per-update item(),
                    # which would wait for backward before the CPU could launch
                    # the optimizer step, next H2D copy, and next forward pass,
                    # removing overlap that the C-budget timing depends on.
                if log_cfg_diagnostics:
                    # PRE-clip, and it has to stay above the clip block: the clip
                    # multiplies every gradient in place, so computing these after it
                    # would silently report coefficient-scaled norms under a name that
                    # says otherwise.  Same inlining rule as the total-norm
                    # comprehension above -- no local name may keep the gradient set
                    # alive past zero_grad(set_to_none=True).
                    trunk_grad_norm = torch.norm(torch.stack([
                        torch.norm(parameter.grad.detach(), 2.0)
                        for parameter in trunk_parameters
                        if parameter.grad is not None
                    ]), 2.0)
                    cfg_grad_norm = torch.norm(torch.stack([
                        torch.norm(parameter.grad.detach(), 2.0)
                        for parameter in cfg_embedding_parameters
                        if parameter.grad is not None
                    ]), 2.0)
                clip_coefficient = None
                if grad_clip_max_norm is not None:
                    # Reproduces clip_grad_norm_ exactly -- max_norm /
                    # (total_norm + 1e-6), clamped to at most 1.0, multiplied in
                    # place -- but reuses the norm computed just above, so the
                    # coefficient recorded is the one that acted and there is no
                    # second norm pass.  The scalar stays on the device for the
                    # same reason total_norm does.
                    clip_coefficient = torch.clamp(grad_clip_max_norm / (total_norm + 1e-6), max=1.0)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.detach().mul_(clip_coefficient)
                if log_grad_norm:
                    # The recorded norm is the PRE-clip one: it is the quantity
                    # whose distribution decides whether a clip threshold is a
                    # no-op or is load-bearing, and it stays comparable with runs
                    # that had no clipping at all.
                    grad_norm_buffer.append((
                        optimizer_updates + 1, time.time(), time.perf_counter(), total_norm
                    ))
                    if len(grad_norm_buffer) == 128:
                        flush_grad_norms(grad_norm_buffer, grad_norm_path, rank)
                if log_cfg_diagnostics:
                    cfg_parameters_before_step = [
                        parameter.detach().clone() for parameter in cfg_embedding_parameters
                    ]
                optimizer.step()
                if log_cfg_diagnostics:
                    cfg_parameter_norm = torch.norm(torch.stack([
                        torch.norm(parameter.detach(), 2.0) for parameter in cfg_embedding_parameters
                    ]), 2.0)
                    cfg_parameter_delta = torch.norm(torch.stack([
                        torch.norm(parameter.detach() - previous, 2.0)
                        for parameter, previous in zip(cfg_embedding_parameters, cfg_parameters_before_step)
                    ]), 2.0)
                    del cfg_parameters_before_step
                    # The EMA target trails the student by update_ema(rate=0.95);
                    # this distance is how far the regression target is from the
                    # weights currently producing it, for this module alone.
                    cfg_student_target_distance = torch.norm(torch.stack([
                        torch.norm(student_parameter.detach() - target_parameter.detach(), 2.0)
                        for student_parameter, target_parameter
                        in zip(cfg_embedding_parameters, target_cfg_embedding_parameters)
                    ]), 2.0)
                    cfg_diagnostic_buffer.append((
                        optimizer_updates + 1,
                        float(optimizer.param_groups[0]['lr']),
                        [
                            total_norm.float(),
                            trunk_grad_norm.float(),
                            cfg_grad_norm.float(),
                            (torch.ones_like(total_norm) if clip_coefficient is None
                             else clip_coefficient).float(),
                            loss.detach().float(),
                            cfg_weight_parameter.detach().pow(2).mean().sqrt().float(),
                            cfg_bias_parameter.detach().pow(2).mean().sqrt().float(),
                            cfg_parameter_norm.float(),
                            (cfg_parameter_delta / (cfg_parameter_norm + 1e-12)).float(),
                            cfg_student_target_distance.float(),
                            (cfg_student_target_distance / (cfg_parameter_norm + 1e-12)).float(),
                        ],
                    ))
                    if len(cfg_diagnostic_buffer) == 128:
                        flush_cfg_diagnostics(cfg_diagnostic_buffer, cfg_diagnostic_path, rank)
                optimizer.zero_grad(set_to_none=True)
                optimizer_updates += 1
                lr_scheduler.step()
                if cfg.max_optimizer_updates is not None and optimizer_updates >= int(cfg.max_optimizer_updates):
                    stop_training = True
                    break

        # Unchanged cadence and unchanged meaning of save_checkpoints /
        # ckpt_interval.  What is new is that every rank now takes part, because
        # the resume state has to gather the per-rank RNG before rank 0 writes.
        if cfg.save_checkpoints and (
            epoch % cfg.ckpt_interval == 0 or epoch == cfg.epochs - 1
        ):
            rng_states = [None] * world_size
            torch.distributed.all_gather_object(rng_states, collect_rng_state(device))
            if rank == 0:
                print(f'Saving checkpoint', flush=True)
                ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
                os.makedirs(ckpt_folder, exist_ok=True)
                atomic_torch_save(
                    model.module.state_dict(),
                    os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"),
                )
                pending_gradients = (
                    collect_pending_gradients(model)
                    if micro_steps % int(cfg.gradient_accumulation_steps) != 0
                    else None
                )
                atomic_torch_save(
                    build_resume_state(
                        geometry, epoch, not stop_training, micro_steps, optimizer_updates,
                        model, optimizer, lr_scheduler, scaler, rng_states, pending_gradients,
                        target_model=target_model if is_consistency else None,
                    ),
                    resume_state_path(cfg.exp_dir, cfg.exp_name),
                )

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()
        if stop_training:
            break

    torch.cuda.synchronize(device)
    local_peaks = torch.tensor(
        [torch.cuda.max_memory_allocated(device), torch.cuda.max_memory_reserved(device)],
        dtype=torch.float64,
        device=device,
    )
    gathered_peaks = [torch.zeros_like(local_peaks) for _ in range(world_size)]
    torch.distributed.all_gather(gathered_peaks, local_peaks)

    # Flush after peak collection so its temporary stack cannot enter the
    # training peak-memory measurement.
    if grad_norm_buffer:
        flush_grad_norms(grad_norm_buffer, grad_norm_path, rank)
    if cfg_diagnostic_buffer:
        flush_cfg_diagnostics(cfg_diagnostic_buffer, cfg_diagnostic_path, rank)

    if rank == 0 and cfg.benchmark_metrics_path is not None:
        metrics_path = Path(str(cfg.benchmark_metrics_path))
        if not metrics_path.is_absolute():
            metrics_path = Path.cwd() / metrics_path
        if metrics_path.exists():
            raise FileExistsError(f'refusing to overwrite benchmark metrics: {metrics_path}')
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        peak_values = [value.cpu().tolist() for value in gathered_peaks]
        metrics = {
            'schema_version': 1,
            'status': 'stable',
            'seed': int(cfg.seed),
            'world_size': world_size,
            'micro_batch_per_gpu': int(cfg.batch_size),
            'gradient_accumulation_steps': int(cfg.gradient_accumulation_steps),
            'effective_batch_size': expected_effective_batch,
            'optimizer_updates': optimizer_updates,
            'micro_steps': micro_steps,
            'last_loss': last_loss,
            'peak_memory_allocated_bytes_by_rank': [int(value[0]) for value in peak_values],
            'peak_memory_reserved_bytes_by_rank': [int(value[1]) for value in peak_values],
            'max_peak_memory_allocated_bytes': int(max(value[0] for value in peak_values)),
            'max_peak_memory_reserved_bytes': int(max(value[1] for value in peak_values)),
        }
        temporary_path = metrics_path.with_suffix(metrics_path.suffix + '.tmp')
        temporary_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + '\n', encoding='utf-8')
        os.replace(temporary_path, metrics_path)

    if cfg.use_tensorboard and rank == 0:
        writer.close()
    torch.distributed.destroy_process_group()


def get_mask(x_start, ind, p, fixed_frame=0, mask_y=True):
    '''
    get mask for the input sequence of pre frames and final goal frame
    '''
    mask_frame = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)
    mask_goal = torch.zeros_like(x_start).to(dtype=torch.bool, device=x_start.device)

    # goal mask
    if ind != -1:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_goal[rand_batch, -1, ind * 3: ind * 3 + 3] = True
        if not mask_y:
            mask_goal[rand_batch, -1, ind * 3 + 1] = False

    # prefix frame mask
    if fixed_frame > 0:
        rand_batch = torch.rand(x_start.shape[0]).to(x_start.device) < p
        mask_frame[rand_batch, :fixed_frame, :] = True
    mask = torch.logical_or(mask_frame, mask_goal)
    return mask, mask_frame, mask_goal


if __name__ == '__main__':
    train()
