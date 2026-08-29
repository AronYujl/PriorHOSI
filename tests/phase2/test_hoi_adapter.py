"""The HOSI-test -> HOIPrior adapter: argument translation and scene blindness.

B1  The adapter's ``p_sample_loop`` accepts the HOSI evaluator's call verbatim --
    the same positional order ``test_infbagel_hosi.sample_step`` uses -- and the
    19 shared arguments arrive at HOIPriorSampler unchanged and in order.
B2  HOIPrior cannot reach the scene.  The evaluator's dataset has
    ``load_scene=True`` (it needs ``scene_dict`` for the scene flag, ``scene_occ``
    for A*, and the SDF for penetration), and the view handed to HOIPrior makes
    the scene payload raise rather than merely go unused.
B3  ``cm_sample_loop`` raises.  HOIPrior was never distilled; silently returning
    something plausible would produce a row nobody could interpret.
"""

import ast
import inspect
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from mixer.hoi_adapter import (  # noqa: E402
    BLOCKED_SCENE_ATTRIBUTES,
    HOIExpertSamplerAdapter,
    SceneBlindDatasetView,
)


class FakeSceneDataset:
    """The scene-loaded shape of InfBaGelDataset, reduced to what matters here."""

    def __init__(self):
        self.load_scene = True
        self.min_torch = torch.zeros(3)
        self.max_torch = torch.ones(3)
        self.obj_min_torch = torch.zeros(3)
        self.obj_max_torch = torch.ones(3)
        # Per-SEQUENCE names on OMOMO, which the P2 guidance needs; not scene state.
        self.scene_name = ['sub16_clothesstand_001']
        self.rest_human_offsets = torch.zeros(1, 24, 3)
        # Scene payload and query surface.
        self.scene_occ = torch.zeros(1, 4, 4, 4, dtype=torch.bool)
        self.scene_occ_ref = object()
        self.scene_dict = {'Scene_1': 0}
        self.scene_grid_np = object()
        self.scene_grid_torch = object()
        self.scene_name2file = {}

    def get_occ_for_points(self, *_, **__):
        raise AssertionError('scene query reached from a scene-blind path')

    def get_pene_occ_count(self, *_, **__):
        raise AssertionError('scene query reached from a scene-blind path')

    def compute_occ_ref(self, *_, **__):
        raise AssertionError('scene query reached from a scene-blind path')

    def set_test_scene(self, *_, **__):
        raise AssertionError('scene mutation reached from a scene-blind path')

    def normalize_torch(self, value, is_object=False):
        return value

    def denormalize_torch(self, value, is_object=False):
        return value


class RecordingInnerSampler:
    """Captures exactly what the adapter forwards."""

    def __init__(self):
        self.dataset = None
        self.student_model = None
        self.calls = []
        self.audit_resets = 0

    def set_dataset_and_model(self, dataset, model):
        if getattr(dataset, 'load_scene', None):
            raise ValueError('HOIPrior evaluation dataset must have load_scene=false')
        self.dataset = dataset
        self.student_model = model

    def p_sample_loop(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return [torch.zeros(1, 16, 232)], []

    def reset_sampling_audit(self):
        self.audit_resets += 1

    def audit_dict(self):
        return {'sample_calls': len(self.calls)}


def build_adapter():
    adapter = HOIExpertSamplerAdapter(device='cpu')
    adapter.inner = RecordingInnerSampler()
    return adapter


# The 19 shared leading parameters, in the order both samplers agree on.
SHARED_LEADING = (
    'fixed_points', 'mat', 'scene_flag', 'text_emb', 'pelvis_goal', 'scene_goal',
    'object_goal', 'need_scene', 'need_pelvis_dir', 'pi', 'end_pi', 'seq_length',
    'need_pi', 'is_loco', 'is_object', 'obj_bps_data', 'object_points',
    'obj_rot_mat_ref', 'obj_rest_verts',
)


class ArgumentTranslationTests(unittest.TestCase):
    def test_shared_prefix_matches_the_hoi_sampler_signature(self):
        """The 19-parameter agreement is asserted against both real signatures."""
        from priors.hoi.diffusion import HOIPriorSampler
        hoi = list(inspect.signature(HOIPriorSampler.p_sample_loop).parameters)[1:]
        adapter = list(inspect.signature(HOIExpertSamplerAdapter.p_sample_loop).parameters)[1:]
        self.assertEqual(tuple(hoi[:19]), SHARED_LEADING)
        self.assertEqual(tuple(adapter[:19]), SHARED_LEADING)
        # ...and they diverge at 20, which is the whole reason this adapter exists.
        self.assertNotEqual(hoi[19], adapter[19])
        self.assertEqual(hoi[19], 'seq_name_dict')
        self.assertEqual(adapter[19], 'obj_vert_normals')

    def test_adapter_accepts_the_evaluator_call_verbatim(self):
        """The positional call `test_infbagel_hosi.sample_step` actually makes."""
        adapter = build_adapter()
        sentinels = {name: object() for name in SHARED_LEADING}
        obj_vert_normals = object()
        seq_name_dict = {0: 'sub16_clothesstand_001'}
        human_dict = {'rest_human_offsets': torch.zeros(1, 16, 24, 3)}
        prefix = object()

        adapter.p_sample_loop(
            *[sentinels[name] for name in SHARED_LEADING],
            obj_vert_normals, seq_name_dict, human_dict, None, 1.0,
            object_only=False, obj_rot_mat_prefix=prefix,
        )

        (args, kwargs), = adapter.inner.calls
        # The 19 shared arguments arrive unchanged, in order, by identity.
        self.assertEqual(len(args), 20)
        for index, name in enumerate(SHARED_LEADING):
            self.assertIs(args[index], sentinels[name], f'argument {index} ({name}) moved')
        self.assertIs(args[19], seq_name_dict)
        self.assertIs(kwargs['obj_rot_mat_prefix'], prefix)
        self.assertEqual(kwargs['object_only'], False)
        self.assertIsNone(kwargs['ground_truth_contact'])
        # The released-path guidance plumbing is dropped, not forwarded.
        for dropped in ('obj_vert_normals', 'human_dict', 'guidance_fn', 'guidance_scale'):
            self.assertNotIn(dropped, kwargs)
        self.assertNotIn(obj_vert_normals, args)
        self.assertNotIn(human_dict, args)

    def test_evaluator_call_site_ordering_is_the_one_asserted(self):
        """Read the evaluator's own call, so a reordering there fails here.

        ``sample_step`` calls ``sampler.p_sample_loop`` positionally.  If that
        argument order is ever edited, the assertion above stops describing the
        real call -- so the order is parsed out of the source rather than trusted.
        """
        source = (REPO / 'code' / 'test_infbagel_hosi.py').read_text()
        tree = ast.parse(source)
        function = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == 'sample_step'
        )
        calls = [
            node for node in ast.walk(function)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == 'p_sample_loop'
        ]
        self.assertEqual(len(calls), 1, 'expected exactly one p_sample_loop call site')
        positional = calls[0].args
        self.assertGreaterEqual(len(positional), 19)
        # The first 19 positional arguments are the shared prefix.  Names in the
        # evaluator's local scope differ from parameter names in three places;
        # everything else must match by name.
        renamed = {
            'text_emb': 'text_emb',
            'obj_rest_verts': 'obj_rest_verts',
        }
        for index, name in enumerate(SHARED_LEADING):
            node = positional[index]
            if isinstance(node, ast.Name):
                expected = renamed.get(name, name)
                self.assertEqual(node.id, expected, f'position {index} is {node.id}, expected {expected}')

    def test_a_per_call_guidance_fn_is_refused(self):
        """HOIPrior's guidance is preregistered on the sampler, not passed per call."""
        adapter = build_adapter()
        args = [object() for _ in SHARED_LEADING]
        with self.assertRaises(ValueError) as caught:
            adapter.p_sample_loop(*args, None, {0: 'x'}, {}, lambda *_: None, 1.0)
        self.assertIn('guidance_fn', str(caught.exception))
        self.assertEqual(adapter.inner.calls, [])

    def test_state_is_reserved(self):
        adapter = build_adapter()
        self.assertIn('state', inspect.signature(adapter.p_sample_loop).parameters)
        args = [object() for _ in SHARED_LEADING]
        with self.assertRaises(NotImplementedError):
            adapter.p_sample_loop(*args, None, {0: 'x'}, {}, None, 1.0, state='walk')
        self.assertEqual(adapter.inner.calls, [])

    def test_cm_sample_loop_raises(self):
        adapter = build_adapter()
        with self.assertRaises(NotImplementedError) as caught:
            adapter.cm_sample_loop()
        self.assertIn('diffusion', str(caught.exception))


class SceneBlindnessTests(unittest.TestCase):
    def test_evaluator_keeps_the_scene_dataset_and_hoiprior_does_not(self):
        adapter = build_adapter()
        dataset = FakeSceneDataset()
        model = torch.nn.Linear(1, 1)
        adapter.set_dataset_and_model(dataset, model)
        # The evaluator reads sampler.dataset for denormalization, the scene flag
        # and A*, so it must be the real, scene-loaded object.
        self.assertIs(adapter.dataset, dataset)
        self.assertTrue(adapter.dataset.load_scene)
        # HOIPrior sees the view.
        self.assertIsInstance(adapter.inner.dataset, SceneBlindDatasetView)
        self.assertFalse(adapter.inner.dataset.load_scene)

    def test_the_real_hoi_sampler_guard_accepts_the_view(self):
        """The guard this view exists to satisfy is the real one, not a stand-in."""
        from priors.hoi.diffusion import HOIPriorSampler
        sampler = HOIPriorSampler(device='cpu')
        view = SceneBlindDatasetView(FakeSceneDataset())
        sampler.set_dataset_and_model(view, torch.nn.Linear(1, 1))
        self.assertIs(sampler.dataset, view)
        with self.assertRaises(ValueError):
            HOIPriorSampler(device='cpu').set_dataset_and_model(
                FakeSceneDataset(), torch.nn.Linear(1, 1))

    def test_scene_payload_is_unreachable_through_the_view(self):
        view = SceneBlindDatasetView(FakeSceneDataset())
        for name in sorted(BLOCKED_SCENE_ATTRIBUTES):
            with self.subTest(attribute=name):
                with self.assertRaises(AttributeError):
                    getattr(view, name)
                self.assertFalse(hasattr(view, name))

    def test_normalization_forwards_by_identity(self):
        """Bit-identical normalization is what makes this row comparable to HOI's own."""
        dataset = FakeSceneDataset()
        view = SceneBlindDatasetView(dataset)
        for name in ('min_torch', 'max_torch', 'obj_min_torch', 'obj_max_torch'):
            self.assertIs(getattr(view, name), getattr(dataset, name))
        # Bound methods are freshly constructed per access, so compare the function.
        self.assertIs(view.normalize_torch.__func__, dataset.normalize_torch.__func__)
        self.assertIs(view.normalize_torch.__self__, dataset)
        self.assertIs(view.scene_name, dataset.scene_name)
        self.assertIs(view.rest_human_offsets, dataset.rest_human_offsets)

    def test_blocked_set_covers_the_dataset_scene_surface(self):
        """Parsed from the dataset source, so a new scene attribute fails here."""
        source = (REPO / 'code' / 'datasets' / 'infbagel.py').read_text()
        tree = ast.parse(source)
        cls = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == 'InfBaGelDataset'
        )
        methods = {
            node.name for node in cls.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        attributes = {
            node.attr for node in ast.walk(cls)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name) and node.value.id == 'self'
        }
        surface = {
            name for name in methods | attributes
            if 'scene' in name or 'occ' in name
        }
        # Metadata and the per-sequence name list are deliberately NOT blocked.
        allowed_unblocked = {
            'load_scene', 'load_scene_goal', 'need_scene', 'force_need_scene',
            'scene_folder', 'test_scene_name', 'scene_name',
        }
        unaccounted = surface - BLOCKED_SCENE_ATTRIBUTES - allowed_unblocked
        self.assertEqual(
            unaccounted, set(),
            'dataset grew scene state that is neither blocked nor explicitly allowed: '
            f'{sorted(unaccounted)}',
        )

    def test_view_is_read_only(self):
        view = SceneBlindDatasetView(FakeSceneDataset())
        with self.assertRaises(AttributeError):
            view.load_scene = True


class ProvenanceForwardingTests(unittest.TestCase):
    def test_audit_helpers_reach_the_inner_sampler(self):
        adapter = build_adapter()
        adapter.reset_sampling_audit()
        self.assertEqual(adapter.inner.audit_resets, 1)
        self.assertEqual(adapter.audit_dict(), {'sample_calls': 0})

    def test_unknown_attributes_forward(self):
        adapter = build_adapter()
        adapter.inner.guidance_settings = 'sentinel'
        self.assertEqual(adapter.guidance_settings, 'sentinel')

    def test_construction_builds_a_real_hoi_sampler(self):
        from priors.hoi.diffusion import HOIPriorSampler
        adapter = HOIExpertSamplerAdapter(device='cpu')
        self.assertIsInstance(adapter.inner, HOIPriorSampler)
        self.assertIsNone(adapter.inner.guidance_settings)

    def test_guidance_config_reaches_the_inner_sampler(self):
        adapter = HOIExpertSamplerAdapter(device='cpu', guidance={
            'enabled': True, 'arm': 'b', 'guidance_scale': 1000.0,
            'last_steps': 10, 'clamp': 1.0, 'clamp_target': 'update',
        })
        self.assertIsNotNone(adapter.inner.guidance_settings)
        self.assertEqual(adapter.inner.guidance_settings.arm, 'b')


if __name__ == '__main__':
    unittest.main()
