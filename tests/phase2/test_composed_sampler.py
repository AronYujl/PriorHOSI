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
