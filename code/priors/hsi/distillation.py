"""Body geometry retention against frozen low-noise teacher endpoints."""

import torch


def select_occupancy_rows(occupancy, rows, batch_size):
    """Keep temporal crop blocks paired with their own batch rows."""
    occ, crop, position = occupancy
    return (
        occ[rows],
        crop.reshape(-1, batch_size, *crop.shape[1:])[:, rows].flatten(0, 1),
        position[:, rows],
    )


class BodyEndpointObjective:
    def __init__(self, sampler, teacher, max_timestep=59):
        self.sampler = sampler
        self.teacher = teacher
        self.max_timestep = int(max_timestep)

    @torch.no_grad()
    def teacher_endpoint(self, noisy, clean, mask, start_index, occupancy, inputs, scale):
        """Actual DDIM endpoint, with the student's exact initial scene crops."""
        sampler = self.sampler
        sampler.batch_size = noisy.shape[0]
        state = noisy.float().clone()
        occ, crop, position = occupancy
        names = ("text_emb", "pelvis_goal", "scene_goal", "is_loco", "need_scene",
                 "need_pelvis_dir", "pi", "end_pi", "seq_length", "need_pi",
                 "object_goal", "is_object", "obj_bps_data")
        for index in range(start_index, -1, -1):
            indices = torch.full((state.shape[0],), index, device=state.device, dtype=torch.long)
            timestep = sampler.solver.ddim_timesteps[indices]
            args = (state, occ, timestep, *[inputs[key] for key in names], crop, position)
            conditional = self.teacher(*args, is_sample=True, is_uncondition=False)
            unconditional = self.teacher(*args, is_sample=True, is_uncondition=True)
            prediction = conditional + scale.unsqueeze(-1) * (conditional - unconditional)
            alpha = sampler.solver.ddim_alpha_cumprods[indices].reshape(-1, 1, 1)
            noise = (state - alpha.sqrt() * prediction) / (1 - alpha).sqrt()
            state = sampler.solver.ddim_step(prediction, noise, indices).float()
            state[mask] = clean[mask].float()
            if index > 0:
                occ, crop, position = sampler._compute_occ_sample(
                    state, prediction, inputs["mat"], inputs["scene_flag"], inputs["object_points"],
                    inputs["pelvis_goal"], inputs["scene_goal"], inputs["object_goal"],
                    inputs["is_loco"], inputs["is_object"], inputs["need_pelvis_dir"],
                    inputs["obj_rot_mat_ref"], False, {}, None, None,
                    int(sampler.solver.ddim_timesteps[index - 1]),
                )
        return state

    def __call__(self, prediction, noisy, clean, mask, indices, occupancy, inputs, scale):
        batch_size = prediction.shape[0]
        loss = prediction.sum() * 0.0
        counts = []
        max_index = self.max_timestep // self.sampler.solver.step_ratio
        # Preserve the original consistency path's CPU/current-device random stream.
        devices = [prediction.device] if prediction.is_cuda else []
        with torch.random.fork_rng(devices=devices), torch.autocast(
            device_type=prediction.device.type, enabled=False
        ):
            for index in range(max_index + 1):
                rows = torch.nonzero((indices == index) & ~inputs["is_object"], as_tuple=True)[0]
                counts.append(rows.numel())
                if rows.numel() == 0:
                    continue
                selected = {key: value[rows] for key, value in inputs.items()}
                endpoint = self.teacher_endpoint(
                    noisy[rows], clean[rows], mask[rows], index,
                    select_occupancy_rows(occupancy, rows, batch_size), selected, scale[rows],
                )
                with torch.no_grad():
                    _, teacher_joints = self.sampler._compute_human_joints(
                        endpoint, endpoint[..., :84], selected["mat"].float(),
                        selected["rest_offsets"].float(),
                    )
                _, student_joints = self.sampler._compute_human_joints(
                    prediction[rows].float(), prediction[rows, :, :84], selected["mat"].float(),
                    selected["rest_offsets"].float(),
                )
                # Sum rows, averaging the fixed14 x22 x3 elements per row, then
                # divide by the original batch (including all high-noise rows).
                loss = loss + (student_joints[:, 2:, :22] - teacher_joints[:, 2:, :22]).square().mean((1, 2, 3)).sum() / batch_size
        return loss, torch.tensor(counts, device=prediction.device)
