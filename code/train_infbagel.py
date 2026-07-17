import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim import Adam
from utils import *
from constants import *
import os
from torch.utils.tensorboard import SummaryWriter
import datetime
import random
import json
import time
import gc

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


def shutdown_dataloader(dataloader):
    """Stop persistent workers before a spawned DDP rank exits.

    A bounded smoke breaks out of an active iterator. Letting the rank process
    terminate with that iterator alive can race the pin-memory thread and
    produce a misleading ``ConnectionResetError`` during otherwise successful
    checkpoint cleanup. This uses the PyTorch iterator shutdown hook when it
    exists and intentionally leaves normal epoch-to-epoch persistence intact.
    """
    iterator = getattr(dataloader, '_iterator', None)
    shutdown = getattr(iterator, '_shutdown_workers', None)
    if shutdown is not None:
        shutdown()
    if hasattr(dataloader, '_iterator'):
        dataloader._iterator = None


def synchronized_time(enabled, device):
    """Return a wall-clock timestamp after pending CUDA work completes."""
    if not enabled:
        return None
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    return time.perf_counter()

@hydra.main(version_base=None, config_path="config", config_name="config_train_infbagel")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = find_free_port()
    world_size = cfg.num_gpus
    print('Usable GPUS: ', torch.cuda.device_count(), flush=True)
    if torch.cuda.device_count() < world_size:
        raise RuntimeError(f'configured num_gpus={world_size}, but only {torch.cuda.device_count()} CUDA devices are visible')
    torch.multiprocessing.spawn(train_ddp,
                                args=(world_size, cfg),
                                nprocs=world_size,
                                join=True)

def train_ddp(rank, world_size, cfg):

    OmegaConf.register_new_resolver("times", lambda x, y: int(x) * int(y))

    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    cfg.device = f"cuda:{rank}"
    profile_timing = bool(getattr(cfg, 'profile_timing', False))
    timing_warmup_updates = int(getattr(cfg, 'timing_warmup_updates', 2))
    timing_init = {'checkpoint_sec': 0.0}
    random.seed(int(cfg.seed) + rank)
    np.random.seed(int(cfg.seed) + rank)
    torch.manual_seed(int(cfg.seed) + rank)
    torch.cuda.manual_seed_all(int(cfg.seed) + rank)
    precision = str(getattr(cfg, 'precision', 'fp32')).lower()
    if precision not in ('fp32', 'amp'):
        raise ValueError(f"Unsupported precision={precision!r}; use 'fp32' or 'amp'")
    amp_enabled = precision == 'amp' and device.type == 'cuda'
    if amp_enabled:
        # RTX 3090 is Ampere. Autocast uses the FP16 Tensor Core path and
        # TF32 accelerates the remaining FP32 matmuls. This is opt-in because
        # it is a numerical/performance variant of the FP32 baseline.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        if hasattr(torch, 'set_float32_matmul_precision'):
            torch.set_float32_matmul_precision('high')
    print(f'Training on {device}', flush=True)
    if rank == 0:
        print(f'Precision: {precision} (autocast={amp_enabled})', flush=True)
    print('Initializing Distributed', flush=True)
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # cfg.sample_type selects the training objective:
    #   diffusion   -> standard diffusion training via trainer.p_losses
    #   consistency -> consistency-model distillation via trainer.consistency_loss
    is_consistency = cfg.sample_type == 'consistency'

    model_init_started = synchronized_time(profile_timing, device)
    if is_consistency:
        teacher_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        teacher_model.requires_grad_(False)

        student_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        student_model.requires_grad_(False)
        student_model.module.embedding_input.requires_grad_(True)
        student_model.module.embedding_output.requires_grad_(True)
        student_model.module.transformer.requires_grad_(True)
        student_model.module.out.requires_grad_(True)

        target_model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        target_model.requires_grad_(False)

        model = student_model
        optimizer = Adam(student_model.parameters(), lr=cfg.lr)
    else:
        model = init_model(list(cfg.model.values())[0], device=rank, eval=False, load_state_dict=cfg.load_state_dict)
        optimizer = Adam(model.parameters(), lr=cfg.lr)

    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    if profile_timing:
        timing_init['model_init_sec'] = (
            synchronized_time(True, device) - model_init_started
        )

    dataset_init_started = synchronized_time(profile_timing, device)
    infbagel_dataset = hydra.utils.instantiate(cfg.dataset)
    if profile_timing:
        timing_init['dataset_init_sec'] = (
            synchronized_time(True, device) - dataset_init_started
        )

    loader_init_started = synchronized_time(profile_timing, device)
    dataloader_dataset = infbagel_dataset
    dataloader_kwargs = {}
    if cfg.num_workers > 0 and hasattr(infbagel_dataset, 'cpu_worker_view'):
        dataloader_dataset = infbagel_dataset.cpu_worker_view(TRAIN_BATCH_KEYS)
        # The default Linux ``fork`` context inherits the parent rank's CUDA
        # address space even when the worker dataset is CPU-only.  Spawn starts
        # workers cleanly and unpickles only the CPU view.
        dataloader_kwargs['multiprocessing_context'] = 'spawn'
        if rank == 0:
            print('DataLoader workers use a CPU-only dataset view (spawn context)', flush=True)

    sampler = DistributedSampler(dataloader_dataset, seed=int(cfg.seed))
    dataloader = DataLoader(dataloader_dataset, batch_size=cfg.batch_size, drop_last=True, num_workers=cfg.num_workers,
                            sampler=sampler, pin_memory=True, persistent_workers=cfg.num_workers > 0,
                            **dataloader_kwargs)

    trainer = hydra.utils.instantiate(list(cfg.sampler.values())[0])
    if is_consistency:
        trainer.set_dataset_and_model(infbagel_dataset, student_model, teacher_model, target_model)
    else:
        trainer.set_dataset_and_model(infbagel_dataset, model)

    if cfg.use_tensorboard and rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(cfg.exp_dir, 'tensorboard_logs'))

    if profile_timing:
        timing_init['loader_trainer_init_sec'] = (
            synchronized_time(True, device) - loader_init_started
        )
    optimizer_updates = 0
    stop_training = False
    timing_stage_names = (
        'data_wait_sec', 'h2d_sec', 'loss_compute_sec',
        'backward_ddp_sec', 'optimizer_sec', 'update_total_sec',
    )
    timing_stage_sums = {name: 0.0 for name in timing_stage_names}
    timing_first = {'first_data_wait_sec': 0.0, 'first_update_sec': 0.0}
    timing_measured_updates = 0
    if profile_timing and device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)

    def finish_timing(stage_name, started, measured):
        if not profile_timing:
            return 0.0, None
        finished = synchronized_time(True, device)
        elapsed = finished - started
        if measured and stage_name is not None:
            timing_stage_sums[stage_name] += elapsed
        return elapsed, finished

    for epoch in range(cfg.start_epoch, cfg.epochs):
        print(f'Start epoch {epoch}', flush=True)
        sampler.set_epoch(epoch)

        step = 0
        data_wait_started = synchronized_time(profile_timing, device)
        for batch in dataloader:
            step += 1
            measured_update = (
                profile_timing and
                optimizer_updates >= timing_warmup_updates
            )
            data_wait_finished = synchronized_time(profile_timing, device)
            if profile_timing:
                data_wait_elapsed = data_wait_finished - data_wait_started
                if optimizer_updates == 0:
                    timing_first['first_data_wait_sec'] = data_wait_elapsed
                if measured_update:
                    timing_stage_sums['data_wait_sec'] += data_wait_elapsed
                update_started = data_wait_finished

            zero_grad_started = synchronized_time(profile_timing, device)
            optimizer.zero_grad()
            zero_grad_elapsed, _ = finish_timing(
                None, zero_grad_started, measured_update
            )

            # async H2D copy for the training tensors (DataLoader sets pin_memory=True)
            h2d_started = synchronized_time(profile_timing, device)
            b = {k: batch[k].to(device, non_blocking=True) for k in TRAIN_BATCH_KEYS}
            finish_timing('h2d_sec', h2d_started, measured_update)

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

            loss_started = synchronized_time(profile_timing, device)
            t = torch.randint(0, trainer.timesteps, (joints.shape[0],), device=device).long()
            x_start = torch.cat([joints, global_rot_6d, object_trans, object_rot_mat.reshape(object_rot_mat.shape[0], object_rot_mat.shape[1], -1), contact_label], dim=-1) # 84 + 132 + 3 + 9 + 4
            with torch.no_grad():
                mask, _, _ = get_mask(x_start, -1, p=1., fixed_frame=cfg.auto_regre_num)

            if is_consistency:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss_dict = trainer.consistency_loss(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
                    need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

                loss_consistency, loss_object, loss_fk = \
                    loss_dict['loss_consistency'], loss_dict['loss_object'], loss_dict['loss_fk']

                if loss_object is not None:
                    loss = loss_consistency + cfg.loss_w_obj_pts * loss_object + cfg.loss_w_fk * loss_fk
                else:
                    loss = loss_consistency

                finish_timing(
                    'loss_compute_sec', loss_started, measured_update
                )

                if step % 10 == 0 or (cfg.max_optimizer_updates is not None and int(cfg.max_optimizer_updates) == 1):
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, LR: {current_lr:.6f}", flush=True)
                    if cfg.use_tensorboard and rank == 0:
                        writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)
                        writer.add_scalar('Loss_consistency', loss_consistency.item(), epoch * len(dataloader) + step)
                        if loss_object is not None:
                            writer.add_scalar('Loss_object', loss_object.item(), epoch * len(dataloader) + step)
                            writer.add_scalar('Loss_fk', loss_fk.item(), epoch * len(dataloader) + step)
            else:
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    loss_dict = trainer.p_losses(x_start, joints, mat, scene_flag, mask, t, text_clip_embedding, pelvis_goal, scene_goal, object_goal, \
                    need_scene, need_pelvis_dir, pi, end_pi, seg_len, need_pi, is_loco, is_object, obj_bps_data, obj_rot_mat_ref, rest_pose_obj_nn_pts, transformed_obj_verts, rest_human_offsets, object_points)

                loss, loss_object, loss_fk = \
                    loss_dict['loss'], loss_dict['loss_object'], loss_dict['loss_fk']
                    
                if loss_object is not None:
                    loss = loss + cfg.loss_w_obj_pts * loss_object + cfg.loss_w_fk * loss_fk

                finish_timing(
                    'loss_compute_sec', loss_started, measured_update
                )

                if step % 10 == 0 or (cfg.max_optimizer_updates is not None and int(cfg.max_optimizer_updates) == 1):
                    current_lr = optimizer.param_groups[0]['lr']
                    print(f"Epoch: {epoch}, Step: {step} / {len(dataloader)}   Loss: {loss.item()}, LR: {current_lr:.6f}", flush=True)
                    if cfg.use_tensorboard and rank == 0:
                        writer.add_scalar('Loss', loss.item(), epoch * len(dataloader) + step)
                        if loss_object is not None:
                            writer.add_scalar('Loss_object', loss_object.item(), epoch * len(dataloader) + step)
                            writer.add_scalar('Loss_fk', loss_fk.item(), epoch * len(dataloader) + step)

            backward_started = synchronized_time(profile_timing, device)
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            finish_timing(
                'backward_ddp_sec', backward_started, measured_update
            )

            optimizer_step_started = synchronized_time(profile_timing, device)
            if amp_enabled:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer_step_elapsed, update_finished = finish_timing(
                None, optimizer_step_started, measured_update
            )
            if profile_timing:
                if optimizer_updates == 0:
                    timing_first['first_update_sec'] = (
                        update_finished - update_started
                    )
                if measured_update:
                    timing_stage_sums['optimizer_sec'] += (
                        zero_grad_elapsed + optimizer_step_elapsed
                    )
                    timing_stage_sums['update_total_sec'] += (
                        update_finished - update_started
                    )
                    timing_measured_updates += 1
                data_wait_started = update_finished
            optimizer_updates += 1
            if cfg.max_optimizer_updates is not None and optimizer_updates >= int(cfg.max_optimizer_updates):
                stop_training = True
                break

        if rank == 0 and epoch % cfg.ckpt_interval == 0:
            print(f'Saving checkpoint', flush=True)
            checkpoint_started = synchronized_time(profile_timing, device)
            ckpt_folder = os.path.join(cfg.exp_dir, 'checkpoints')
            os.makedirs(ckpt_folder, exist_ok=True)
            torch.save(model.module.state_dict(), os.path.join(ckpt_folder, f"{cfg.exp_name}_epoch{epoch:03d}.pth"))
            if profile_timing:
                timing_init['checkpoint_sec'] = (
                    synchronized_time(True, device) - checkpoint_started
                )

        torch.distributed.barrier()

        print('Clearing cache', flush=True)
        torch.cuda.empty_cache()
        if stop_training:
            shutdown_dataloader(dataloader)
            break

    if profile_timing:
        init_names = (
            'model_init_sec', 'dataset_init_sec',
            'loader_trainer_init_sec', 'checkpoint_sec',
        )
        init_values = torch.tensor(
            [timing_init.get(name, 0.0) for name in init_names],
            dtype=torch.float64, device=device,
        )
        stage_values = torch.tensor(
            [timing_stage_sums[name] for name in timing_stage_names],
            dtype=torch.float64, device=device,
        )
        first_names = ('first_data_wait_sec', 'first_update_sec')
        first_values = torch.tensor(
            [timing_first[name] for name in first_names],
            dtype=torch.float64, device=device,
        )
        measured_updates = torch.tensor(
            timing_measured_updates, dtype=torch.long, device=device,
        )
        if device.type == 'cuda':
            memory_values = torch.tensor([
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            ], dtype=torch.float64, device=device)
        else:
            memory_values = torch.zeros(
                2, dtype=torch.float64, device=device
            )

        torch.distributed.all_reduce(
            init_values, op=torch.distributed.ReduceOp.MAX
        )
        torch.distributed.all_reduce(
            stage_values, op=torch.distributed.ReduceOp.MAX
        )
        torch.distributed.all_reduce(
            first_values, op=torch.distributed.ReduceOp.MAX
        )
        torch.distributed.all_reduce(
            measured_updates, op=torch.distributed.ReduceOp.MIN
        )
        torch.distributed.all_reduce(
            memory_values, op=torch.distributed.ReduceOp.MAX
        )

        count = int(measured_updates.item())
        summary = {
            'synchronized_profile': True,
            'warmup_updates': timing_warmup_updates,
            'measured_updates': count,
            'rank_max_init_sec': dict(zip(init_names, init_values.tolist())),
            'rank_max_first_sec': dict(zip(first_names, first_values.tolist())),
            'rank_max_cuda_memory_gib': {
                'allocated': memory_values[0].item() / (1024 ** 3),
                'reserved': memory_values[1].item() / (1024 ** 3),
            },
            'rank_max_warm_avg_sec': {
                name: value / count if count else None
                for name, value in zip(
                    timing_stage_names, stage_values.tolist()
                )
            },
        }
        if rank == 0:
            print(
                'PROFILE_TIMING ' + json.dumps(summary, sort_keys=True),
                flush=True,
            )


    # Persistent workers and NCCL must be closed explicitly on both bounded
    # smokes and naturally completed full runs. Otherwise rank processes can
    # tear down CUDA while worker queues still own shared tensor handles.
    shutdown_dataloader(dataloader)
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    torch.distributed.barrier()
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
