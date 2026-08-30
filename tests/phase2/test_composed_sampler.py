"""The composed reverse chain, and the property that makes it trustworthy.

The load-bearing test is C1: driving the composed loop with G == 0 reproduces
``HOIPriorSampler.p_sample_loop`` BITWISE, on the real diffusion object with a
real 500-step schedule.  That is what extends the operator's A1 anchor from one
arithmetic expression to the whole sampler -- the composed loop reimplements the
reverse chain, so "G == 0 is HOI alone" has to be measured against the expert's
own loop and not argued from the code's shape.

The model here is a stub, because the property under test is the CHAIN and not a
checkpoint: same schedule, same posterior, same history pinning, same generator
draw order.  A stub makes the comparison exact and runs on CPU in a second.
"""

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from mixer.composed_sampler import HOSIComposedSampler  # noqa: E402
from mixer.composition import OBJECT_CHANNEL_START  # noqa: E402
from mixer.gates import (  # noqa: E402
    ChannelBlockGate,
    ConstantGate,
    ObjectConditionedGate,
    ScheduleGate,
)
from priors.core.representation import REPRESENTATION  # noqa: E402


class DeterministicModel(torch.nn.Module):
    """A stand-in whose output depends on every conditioning input it is given.

    Deterministic and cheap, but not constant: if the composed loop passed a
    different state, timestep or conditioning tensor than the expert's own loop,
    the outputs would differ and C1 would fail.
    """

    architecture_variant = None

    def __init__(self, scale=0.01):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, current, timesteps, text_embedding, object_bps, goals,
                progress, **kwargs):
        self.calls += 1
        del kwargs
        signature = (
            current.sum(dim=(1, 2), keepdim=True)
            + timesteps.reshape(-1, 1, 1).float()
            + text_embedding.sum(dim=-1, keepdim=True)[..., None]
            + object_bps.sum(dim=(1, 2)).reshape(-1, 1, 1)
            + goals.sum(dim=-1).reshape(-1, 1, 1)
            + progress.sum(dim=-1).reshape(-1, 1, 1)
        )
        channels = torch.arange(
            current.shape[-1], device=current.device, dtype=current.dtype,
        ).reshape(1, 1, -1)
        frames = torch.arange(
            current.shape[1], device=current.device, dtype=current.dtype,
        ).reshape(1, -1, 1)
        return torch.tanh(self.scale * (signature + channels + frames))


class StubDataset:
    max_window_size = REPRESENTATION.window_frames
    load_scene = False
    vis = False

    def __init__(self, device='cpu'):
        self.min_torch = torch.zeros(3, device=device)
        self.max_torch = torch.ones(3, device=device)
        self.obj_min_torch = torch.zeros(3, device=device)
        self.obj_max_torch = torch.ones(3, device=device)

    def normalize_torch(self, data, is_object=False):
        del is_object
        return data


def _evaluator_arguments(batch=2, device='cpu'):
    torch.manual_seed(7)
    frames = REPRESENTATION.window_frames
    return {
        'fixed_points': torch.randn(
            batch, REPRESENTATION.history_frames, REPRESENTATION.dimension,
            device=device,
        ),
        'mat': torch.eye(4, device=device).expand(batch, 4, 4).clone(),
        'scene_flag': torch.zeros(batch, dtype=torch.long, device=device),
        'text_emb': torch.randn(batch, 512, device=device),
        'pelvis_goal': torch.randn(batch, 3, device=device),
        'scene_goal': torch.zeros(batch, 3, device=device),
        'object_goal': torch.randn(batch, 3, device=device),
        'need_scene': torch.zeros(batch, dtype=torch.bool, device=device),
        'need_pelvis_dir': torch.ones(batch, dtype=torch.bool, device=device),
        'pi': torch.full((batch,), 10, device=device),
        'end_pi': torch.full((batch,), 58, device=device),
        'seq_length': torch.full((batch,), 200, device=device),
        'need_pi': torch.ones(batch, dtype=torch.bool, device=device),
        'is_loco': torch.zeros(batch, dtype=torch.bool, device=device),
        'is_object': torch.ones(batch, dtype=torch.bool, device=device),
        'obj_bps_data': torch.randn(batch, 1, 1024, 3, device=device),
        'object_points': torch.randn(batch, 1024, 3, device=device),
        'obj_rot_mat_ref': torch.eye(3, device=device).expand(batch, 1, 3, 3).clone(),
        'obj_rest_verts': {},
        'seq_name_dict': {index: f'sub16_clothesstand_{index}' for index in range(batch)},
    }


def _make_pair(timesteps=500, device='cpu'):
    """A HOIPriorSampler and a composed sampler over the same model and dataset."""
    from priors.hoi.diffusion import HOIPriorSampler

    from mixer.hoi_adapter import HOIExpertSamplerAdapter

    model = DeterministicModel()
    dataset = StubDataset(device=device)

    reference = HOIPriorSampler(device=device, timesteps=timesteps)
    reference.set_dataset_and_model(dataset, model)

    adapter = HOIExpertSamplerAdapter(device=device, timesteps=timesteps)
    composed = HOSIComposedSampler(adapter, hsi_sampler=None, gate=0)
    composed.set_dataset_and_model(dataset, model)
    return reference, composed, model, dataset


class ComposedChainAnchorTests(unittest.TestCase):
    def test_c1_gate_zero_reproduces_hoi_alone_bitwise(self):
        """The whole point: G == 0 is not merely close to HOI alone, it IS it.

        500 steps is not a choice: ``core/diffusion_schedule.py`` refuses any
        other count, so every test here runs the production schedule.
        """
        reference, composed, _, _ = _make_pair(timesteps=500)
        arguments = _evaluator_arguments()

        torch.manual_seed(42)
        expected, _ = reference.p_sample_loop(**arguments)
        torch.manual_seed(42)
        actual, _ = composed.p_sample_loop(**arguments)

        self.assertEqual(len(expected), 1)
        self.assertTrue(
            torch.equal(expected[0], actual[-1]),
            'composed G==0 diverged from HOIPriorSampler.p_sample_loop; max abs '
            f'diff {float((expected[0] - actual[-1]).abs().max())}',
        )

    def test_c1_holds_for_a_batch_of_one(self):
        """Batch 1 is what the HOSI evaluator actually runs, episode by episode."""
        reference, composed, _, _ = _make_pair(timesteps=500)
        arguments = _evaluator_arguments(batch=1)

        torch.manual_seed(42)
        expected, _ = reference.p_sample_loop(**arguments)
        torch.manual_seed(42)
        actual, _ = composed.p_sample_loop(**arguments)
        self.assertTrue(torch.equal(expected[0], actual[-1]))

    def test_the_stub_model_actually_discriminates(self):
        """Guard against C1 passing because the model ignores its inputs.

        Without this, a constant model would make the anchor test vacuous.
        """
        model = DeterministicModel()
        batch, frames = 2, REPRESENTATION.window_frames
        base = dict(
            current=torch.zeros(batch, frames, REPRESENTATION.dimension),
            timesteps=torch.zeros(batch, dtype=torch.long),
            text_embedding=torch.zeros(batch, 512),
            object_bps=torch.zeros(batch, 1024, 3),
            goals=torch.zeros(batch, 9),
            progress=torch.zeros(batch, 3),
        )
        reference = model(**base)
        for key in base:
            perturbed = dict(base)
            if key == 'timesteps':
                perturbed[key] = torch.ones(batch, dtype=torch.long)
            else:
                perturbed[key] = base[key] + 1.0
            self.assertFalse(
                torch.equal(model(**perturbed), reference),
                f'model output is insensitive to {key}',
            )

    def test_a_nonzero_gate_without_an_hsi_sampler_raises(self):
        from mixer.hoi_adapter import HOIExpertSamplerAdapter

        adapter = HOIExpertSamplerAdapter(device='cpu', timesteps=500)
        with self.assertRaises(ValueError):
            HOSIComposedSampler(adapter, hsi_sampler=None, gate=0.5)

    def test_state_is_reserved_on_the_constructor_and_the_loop(self):
        from mixer.hoi_adapter import HOIExpertSamplerAdapter

        adapter = HOIExpertSamplerAdapter(device='cpu', timesteps=500)
        with self.assertRaises(NotImplementedError):
            HOSIComposedSampler(adapter, state={'task': 'walk'})
        _, composed, _, _ = _make_pair(timesteps=500)
        with self.assertRaises(NotImplementedError):
            composed.p_sample_loop(**_evaluator_arguments(), state={'task': 'walk'})

    def test_a_per_call_guidance_fn_is_refused(self):
        _, composed, _, _ = _make_pair(timesteps=500)
        with self.assertRaises(ValueError):
            composed.p_sample_loop(
                **_evaluator_arguments(), guidance_fn=lambda *a, **k: None,
            )

    def test_cm_sample_loop_raises(self):
        _, composed, _, _ = _make_pair(timesteps=500)
        with self.assertRaises(NotImplementedError):
            composed.cm_sample_loop()

    def test_compose_is_called_once_per_reverse_step(self):
        _, composed, _, _ = _make_pair(timesteps=500)
        composed.p_sample_loop(**_evaluator_arguments())
        self.assertEqual(composed.compose_calls, 500)

    def test_audit_records_the_composition(self):
        _, composed, _, _ = _make_pair(timesteps=500)
        composed.p_sample_loop(**_evaluator_arguments())
        audit = composed.audit_dict()
        self.assertIn('composition', audit)
        self.assertEqual(audit['composition']['compose_calls'], 500)
        self.assertFalse(audit['composition']['hsi_expert_loaded'])
        self.assertTrue(audit['composition']['per_step_composition'])
        self.assertEqual(audit['composition']['gate'], {'kind': 'scalar', 'value': 0.0})

    def test_the_model_is_called_once_per_step_with_no_hsi_expert(self):
        """HOI has no inference CFG; a second call would mean one was introduced."""
        _, composed, model, _ = _make_pair(timesteps=500)
        before = model.calls
        composed.p_sample_loop(**_evaluator_arguments())
        self.assertEqual(model.calls - before, 500)


class RecordingHSISampler:
    """The released Sampler's inference surface, recording how it is called.

    Only the three things the composed loop touches are real: ``w``,
    ``student_model`` and ``_compute_occ_sample``.  Everything else the released
    Sampler owns belongs to its own posterior step, which the composed chain does
    not use.

    ``object_voxels`` seeds the returned grids with the occupancy alphabet's
    object value so the remap can be observed.
    """

    def __init__(self, w=1.0, channels=REPRESENTATION.dimension,
                 object_voxels=0):
        self.w = w
        self.channels = channels
        self.object_voxels = object_voxels
        self.batch_size = None
        self.dataset = None
        self.student_model = None
        self.occ_calls = []
        self.grids_returned = []

    def set_dataset_and_model(self, dataset, model):
        self.dataset = dataset
        self.student_model = model

    def _compute_occ_sample(self, x, x0, *args):
        del args
        # Clones, so a later in-place write by the loop cannot rewrite history.
        self.occ_calls.append((x.clone(), x0.clone()))
        batch = x.shape[0]
        occ = torch.zeros(batch, 1, 1, 8)
        occ_list = torch.zeros(batch * 4, 1, 1, 8)
        if self.object_voxels:
            # Anchor grid clean, temporal grids carrying the object value: the
            # HOSI-test shape measured in occ_distribution.json, where
            # add_object_voxel=false leaves occ_list[0] free of 2s.
            occ_list[batch:, ..., :self.object_voxels] = 2
        self.grids_returned.append((occ.clone(), occ_list.clone()))
        return occ, occ_list, torch.zeros(4, batch, 2)


class RecordingHSIModel(torch.nn.Module):
    """Returns a fixed cond and uncond so the CFG combination is checkable.

    Records the occupancy grids it was handed, which is how the object-voxel
    remap is observed at the point the ViT would actually see it.
    """

    def __init__(self, cond_value=2.0, uncond_value=0.5,
                 channels=REPRESENTATION.dimension):
        super().__init__()
        self.cond_value = cond_value
        self.uncond_value = uncond_value
        self.channels = channels
        self.cond_calls = 0
        self.uncond_calls = 0
        self.grids_seen = []

    def forward(self, current, *args, is_sample=False, is_uncondition=False,
                **kwargs):
        del kwargs
        # The released call order is (x, occ, t, ...) with occ_list at position
        # 15 of *args; only the two grids matter here.
        if args:
            occ = args[0]
            occ_list = args[15] if len(args) > 15 else None
            self.grids_seen.append((
                occ.clone() if torch.is_tensor(occ) else None,
                occ_list.clone() if torch.is_tensor(occ_list) else None,
            ))
        if is_uncondition:
            self.uncond_calls += 1
            value = self.uncond_value
        else:
            self.cond_calls += 1
            value = self.cond_value
        return torch.full_like(current, value)


def _make_composed_with_hsi(gate, timesteps=500, w=1.0, batch=2,
                            channel_mask='human', object_voxels=0,
                            hsi_object_voxel_mode='occupied'):
    from priors.hoi.diffusion import HOIPriorSampler  # noqa: F401

    from mixer.hoi_adapter import HOIExpertSamplerAdapter

    model = DeterministicModel()
    dataset = StubDataset()
    adapter = HOIExpertSamplerAdapter(device='cpu', timesteps=timesteps)
    hsi_sampler = RecordingHSISampler(w=w, object_voxels=object_voxels)
    hsi_model = RecordingHSIModel()
    composed = HOSIComposedSampler(
        adapter, hsi_sampler=hsi_sampler, gate=gate, channel_mask=channel_mask,
        hsi_object_voxel_mode=hsi_object_voxel_mode,
    )
    composed.set_dataset_and_model(dataset, model, hsi_model=hsi_model)
    return composed, hsi_sampler, hsi_model, model


class HSIExpertCallTests(unittest.TestCase):
    """The HSI expert's inference convention, which is not HOI's.

    HOIPrior calls its network once per step.  HSIPrior calls twice and combines
    ``cond + w * (cond - uncond)``, and its "uncond" pass is not unconditional --
    ``models/infbagel.py:1554`` zeroes only the TEMPORAL scene embeddings.  If the
    composed loop dropped the second call, the scene expert would silently lose
    the one term HOIPrior has no analogue for.
    """

    def test_the_hsi_model_is_called_twice_per_step(self):
        composed, _, hsi_model, _ = _make_composed_with_hsi(gate=0.5)
        composed.p_sample_loop(**_evaluator_arguments())
        self.assertEqual(hsi_model.cond_calls, 500)
        self.assertEqual(hsi_model.uncond_calls, 500)

    def test_the_cfg_combination_is_applied(self):
        """cond + w*(cond - uncond), not cond."""
        composed, _, _, _ = _make_composed_with_hsi(gate=1.0, w=3.0)
        arguments = _evaluator_arguments()
        torch.manual_seed(42)
        imgs, _ = composed.p_sample_loop(**arguments)
        # G == 1 is the HSI anchor, so x_hat_0 is exactly the CFG output:
        # 2.0 + 3.0 * (2.0 - 0.5) = 6.5 on every channel.
        #
        # The final image is NOT that value: the t == 0 posterior returns
        # coef1[0] * x_hat_0, and coef1[0] is 0.9998340606689453 rather than 1
        # because `1 - alpha_bar[0]` is `1 - (1 - 1e-4)` in float32 and loses
        # three digits.  The released sampler has the same buffers, so this
        # 1.66e-4 shrink is shared with every InfBaGel row ever produced, not
        # introduced here -- but it means the last reverse step is not an
        # identity on x_hat_0, and a test that assumed it was would be testing
        # its own arithmetic.
        coef1_at_zero = float(composed.inner_hoi.diffusion.posterior_mean_coef1[0])
        expected = 6.5 * coef1_at_zero
        final = imgs[-1]
        body = final[:, REPRESENTATION.history_frames:, :OBJECT_CHANNEL_START]
        self.assertTrue(
            torch.allclose(body, torch.full_like(body, expected), atol=1e-5),
            f'CFG output not reproduced at the anchor: {body.flatten()[:4]}',
        )

    def test_occupancy_sees_the_composed_x0_not_the_noisy_state(self):
        """`x0` must be the previous step's x_hat_0, as in the released loop.

        `_compute_occ_sample` reads `x0[:, :, :84]` to place the three temporal
        occupancy queries.  Passing `current` there would query the scene along a
        NOISY trajectory -- pure noise on the first step -- so the expert's whole
        dynamic-perception mechanism would be reading the wrong path.
        """
        composed, hsi_sampler, _, _ = _make_composed_with_hsi(gate=1.0)
        arguments = _evaluator_arguments()
        torch.manual_seed(42)
        composed.p_sample_loop(**arguments)

        self.assertEqual(len(hsi_sampler.occ_calls), 500)
        first_x, first_x0 = hsi_sampler.occ_calls[0]
        # Step 499 is the first call: the released loop's x0 list starts at the
        # initial noise, so x and x0 coincide there and only there.
        self.assertTrue(torch.equal(first_x, first_x0))

        # From step 498 on they must differ, and x0 must equal the PREVIOUS
        # step's x_hat_0 -- which under G == 1 is the CFG output.  With the
        # default w == 1 that is 2.0 + 1.0 * (2.0 - 0.5) == 3.5, and it holds on
        # the history frames too: _hsi_x0 returns the raw model output, and the
        # G == 1 anchor short-circuits before any history restoration.
        second_x, second_x0 = hsi_sampler.occ_calls[1]
        self.assertFalse(torch.equal(second_x, second_x0))
        self.assertTrue(
            torch.allclose(second_x0, torch.full_like(second_x0, 3.5), atol=1e-6),
            f'x0 is not the previous step x_hat_0: '
            f'{torch.unique(second_x0).tolist()[:4]}',
        )

    def test_the_hsi_anchor_returns_the_hsi_prediction_unmasked(self):
        """G == 1 short-circuits before the mask, including on 216:232."""
        composed, _, _, _ = _make_composed_with_hsi(gate=1.0)
        arguments = _evaluator_arguments()
        torch.manual_seed(42)
        imgs, _ = composed.p_sample_loop(**arguments)
        final = imgs[-1]
        # Object translation comes from HSI at the anchor, so it is the CFG value
        # (3.5 at the default w == 1, shrunk by coef1[0] on the last step) and NOT
        # HOI's prediction.  219:228 is excluded: p_sample_loop closes it on SO(3)
        # at the end, which rewrites those nine channels by design.
        coef1_at_zero = float(composed.inner_hoi.diffusion.posterior_mean_coef1[0])
        expected = 3.5 * coef1_at_zero
        object_translation = final[
            :, REPRESENTATION.history_frames:,
            OBJECT_CHANNEL_START:OBJECT_CHANNEL_START + 3,
        ]
        self.assertTrue(
            torch.allclose(
                object_translation, torch.full_like(object_translation, expected),
                atol=1e-5,
            ),
            'the HSI anchor did not pass HSI through on the object channels: '
            f'{torch.unique(object_translation).tolist()[:4]}',
        )

    def test_the_mask_keeps_object_channels_on_hoi_under_a_partial_gate(self):
        """The measurement the mask exists for, at the level of the whole loop."""
        composed, _, _, _ = _make_composed_with_hsi(gate=0.999)
        arguments = _evaluator_arguments()
        torch.manual_seed(42)
        imgs, _ = composed.p_sample_loop(**arguments)
        final = imgs[-1]
        object_translation = final[
            :, REPRESENTATION.history_frames:,
            OBJECT_CHANNEL_START:OBJECT_CHANNEL_START + 3,
        ]
        # HSI's CFG output is a uniform 6.5; HOI's stub output is a tanh, so it is
        # bounded by 1. Anything near 6.5 would mean the gate reached the object
        # channels.
        self.assertLess(
            float(object_translation.abs().max()), 1.0 + 1e-6,
            'a gate of 0.999 leaked into the object translation channels',
        )

    def test_set_dataset_and_model_requires_an_hsi_model(self):
        from mixer.hoi_adapter import HOIExpertSamplerAdapter

        adapter = HOIExpertSamplerAdapter(device='cpu', timesteps=500)
        composed = HOSIComposedSampler(
            adapter, hsi_sampler=RecordingHSISampler(), gate=0.5,
        )
        with self.assertRaises(ValueError):
            composed.set_dataset_and_model(StubDataset(), DeterministicModel())

    def test_the_hsi_expert_gets_the_full_dataset_not_the_blind_view(self):
        """HOI gets the scene-blind view; HSI must get the real dataset."""
        composed, hsi_sampler, _, _ = _make_composed_with_hsi(gate=0.5)
        dataset = composed.dataset
        self.assertIs(hsi_sampler.dataset, dataset)
        self.assertFalse(composed.inner_hoi.dataset.load_scene)

    def test_audit_reports_the_hsi_expert_as_loaded(self):
        composed, _, _, _ = _make_composed_with_hsi(gate=0.5)
        composed.p_sample_loop(**_evaluator_arguments())
        audit = composed.audit_dict()
        self.assertTrue(audit['composition']['hsi_expert_loaded'])
        self.assertEqual(audit['composition']['channel_mask'], 'human')


class ObjectVoxelModeTests(unittest.TestCase):
    """What the manipulated object looks like to the LINGO-trained scene expert.

    The occupancy alphabet is 0 free / 1 occupied / 2 object.  HSIPrior never saw
    a real 2 in training (LINGO's object_points is the 999.0 sentinel, clamped to
    voxel 0), and HOSI-test's temporal grids carry 225-239 of them.  Measured with
    real weights, the two modes' x_hat_0 differ by up to 0.158 m of joint
    position, so this must be an explicit, audited choice rather than whatever the
    released arithmetic happens to do.
    """

    def test_occupied_mode_rewrites_the_object_value_before_the_model(self):
        composed, _, hsi_model, _ = _make_composed_with_hsi(
            gate=0.5, object_voxels=3, hsi_object_voxel_mode='occupied',
        )
        composed.p_sample_loop(**_evaluator_arguments())
        for occ, occ_list in hsi_model.grids_seen:
            self.assertEqual(int((occ == 2).sum()), 0)
            self.assertEqual(int((occ_list == 2).sum()), 0)
        # And the voxels are still OCCUPIED, not cleared to free: the object is an
        # obstacle either way, only its identity is dropped.
        _, occ_list = hsi_model.grids_seen[0]
        self.assertGreater(int((occ_list == 1).sum()), 0)

    def test_object_mode_passes_the_object_value_through(self):
        composed, _, hsi_model, _ = _make_composed_with_hsi(
            gate=0.5, object_voxels=3, hsi_object_voxel_mode='object',
        )
        composed.p_sample_loop(**_evaluator_arguments())
        occ, occ_list = hsi_model.grids_seen[0]
        self.assertEqual(int((occ == 2).sum()), 0)
        self.assertGreater(int((occ_list == 2).sum()), 0)

    def test_the_remap_count_is_audited(self):
        composed, _, _, _ = _make_composed_with_hsi(
            gate=0.5, object_voxels=3, hsi_object_voxel_mode='occupied',
        )
        composed.p_sample_loop(**_evaluator_arguments())
        audit = composed.audit_dict()
        self.assertEqual(audit['composition']['hsi_object_voxel_mode'], 'occupied')
        # 2 batch elements x 3 temporal grids x 3 voxels x 500 steps.
        self.assertEqual(
            audit['composition']['hsi_object_voxels_remapped'],
            2 * 3 * 3 * 500,
        )

    def test_object_mode_remaps_nothing(self):
        composed, _, _, _ = _make_composed_with_hsi(
            gate=0.5, object_voxels=3, hsi_object_voxel_mode='object',
        )
        composed.p_sample_loop(**_evaluator_arguments())
        audit = composed.audit_dict()
        self.assertEqual(audit['composition']['hsi_object_voxel_mode'], 'object')
        self.assertEqual(audit['composition']['hsi_object_voxels_remapped'], 0)

    def test_an_unknown_mode_is_refused(self):
        from mixer.hoi_adapter import HOIExpertSamplerAdapter

        adapter = HOIExpertSamplerAdapter(device='cpu', timesteps=500)
        with self.assertRaises(ValueError):
            HOSIComposedSampler(adapter, hsi_object_voxel_mode='ignore')

    def test_the_mode_does_not_touch_the_hoi_anchor(self):
        """G == 0 is HOI alone whatever the HSI expert is shown."""
        reference, _, _, _ = _make_pair(timesteps=500)
        arguments = _evaluator_arguments()
        torch.manual_seed(42)
        expected, _ = reference.p_sample_loop(**arguments)
        for mode in ('occupied', 'object'):
            composed, _, _, _ = _make_composed_with_hsi(
                gate=0, object_voxels=3, hsi_object_voxel_mode=mode,
            )
            torch.manual_seed(42)
            actual, _ = composed.p_sample_loop(**_evaluator_arguments())
            self.assertTrue(
                torch.equal(expected[0], actual[-1]),
                f'the anchor moved under hsi_object_voxel_mode={mode!r}',
            )


class GateTests(unittest.TestCase):
    def setUp(self):
        self.hoi = torch.zeros(3, REPRESENTATION.window_frames, REPRESENTATION.dimension)
        self.hsi = torch.ones_like(self.hoi)

    def test_constant_gate_anchors(self):
        for value in (0.0, 1.0, 0.25):
            gate = ConstantGate(value)
            self.assertEqual(
                gate(step=3, current=self.hoi, hoi=self.hoi, hsi=self.hsi), value
            )

    def test_constant_gate_rejects_out_of_range(self):
        with self.assertRaises(ValueError):
            ConstantGate(1.5)

    def test_schedule_gate_late_starts_at_zero_and_ends_at_peak(self):
        gate = ScheduleGate(timesteps=100, peak=0.4, mode='late')
        first = gate(step=99, current=self.hoi, hoi=self.hoi, hsi=self.hsi)
        last = gate(step=0, current=self.hoi, hoi=self.hoi, hsi=self.hsi)
        self.assertAlmostEqual(first, 0.0, places=6)
        self.assertAlmostEqual(last, 0.4, places=6)

    def test_schedule_gate_early_is_the_reverse(self):
        gate = ScheduleGate(timesteps=100, peak=0.4, mode='early')
        self.assertAlmostEqual(
            gate(step=99, current=self.hoi, hoi=self.hoi, hsi=self.hsi), 0.4, places=6
        )
        self.assertAlmostEqual(
            gate(step=0, current=self.hoi, hoi=self.hoi, hsi=self.hsi), 0.0, places=6
        )

    def test_schedule_gate_stays_in_range_at_every_step(self):
        gate = ScheduleGate(timesteps=500, peak=1.0)
        values = [
            gate(step=step, current=self.hoi, hoi=self.hoi, hsi=self.hsi)
            for step in range(500)
        ]
        self.assertGreaterEqual(min(values), 0.0)
        self.assertLessEqual(max(values), 1.0)

    def test_object_conditioned_gate_is_per_batch_element(self):
        gate = ObjectConditionedGate(
            {'clothesstand': 0.6, 'tripod': 0.5}, default=0.1,
        ).set_object_names(['clothesstand', 'tripod', 'suitcase'])
        value = gate(step=7, current=self.hoi, hoi=self.hoi, hsi=self.hsi)
        self.assertEqual(tuple(value.shape), (3, 1, 1))
        self.assertAlmostEqual(float(value[0]), 0.6, places=6)
        self.assertAlmostEqual(float(value[1]), 0.5, places=6)
        self.assertAlmostEqual(float(value[2]), 0.1, places=6)

    def test_object_conditioned_gate_needs_names_first(self):
        gate = ObjectConditionedGate({'tripod': 0.5})
        with self.assertRaises(ValueError):
            gate(step=0, current=self.hoi, hoi=self.hoi, hsi=self.hsi)

    def test_object_conditioned_gate_checks_the_batch_size(self):
        gate = ObjectConditionedGate({'tripod': 0.5}).set_object_names(['tripod'])
        with self.assertRaises(ValueError):
            gate(step=0, current=self.hoi, hoi=self.hoi, hsi=self.hsi)

    def test_channel_block_gate_is_zero_on_object_and_contact(self):
        gate = ChannelBlockGate(positions=0.8, rotations=0.2)
        value = gate(step=0, current=self.hoi, hoi=self.hoi, hsi=self.hsi)
        self.assertEqual(tuple(value.shape), (REPRESENTATION.dimension,))
        self.assertTrue(torch.all(value[:84] == 0.8).item())
        self.assertTrue(torch.all(value[84:216] == 0.2).item())
        self.assertTrue(torch.all(value[OBJECT_CHANNEL_START:] == 0).item())

    def test_a_callable_gate_reaches_the_composed_loop(self):
        seen = []

        def gate(*, step, current, hoi, hsi):
            seen.append(step)
            return 0.0

        _, composed, _, _ = _make_pair(timesteps=500)
        composed.gate = gate
        composed.p_sample_loop(**_evaluator_arguments())
        self.assertEqual(seen, list(reversed(range(500))))

    def test_a_callable_gate_sees_only_outputs(self):
        """MixerMDM modularity, as a signature check rather than a promise."""
        import inspect

        for producer in (ConstantGate(0.5), ScheduleGate(10),
                         ObjectConditionedGate({}), ChannelBlockGate()):
            parameters = inspect.signature(producer.__call__).parameters
            self.assertEqual(
                sorted(parameters), ['current', 'hoi', 'hsi', 'step'],
                f'{type(producer).__name__} takes something other than expert outputs',
            )
            for name in ('current', 'hoi', 'hsi', 'step'):
                self.assertEqual(
                    parameters[name].kind, inspect.Parameter.KEYWORD_ONLY,
                    f'{type(producer).__name__}.{name} must be keyword-only',
                )


if __name__ == '__main__':
    unittest.main()
