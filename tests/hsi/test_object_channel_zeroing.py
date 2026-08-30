"""Pin the object-channel x0 repair: inert when off, per-sample when on, B-only.

The knob forces channels 216:232 of the predicted x0 to exactly 0 on scene-only rows,
in ``Sampler.p_sample`` after CFG and before the posterior mean.  It exists because
HSI training presents those channels as ``q_sample(0)`` and nothing else, while
``self.out`` emits all 232 channels and ``embedding_input`` consumes all 232 at the
next reverse step -- so the P17-OC arm's E0, which severed the gradient to those
output rows and left them at ``nn.Linear`` init, made the reverse chain drive them
toward a value the trunk never saw at low t.

Four properties have to hold or the A/B that uses it means nothing:

  * off is an exact identity, so a knob-carrying build reproduces the sealed cells;
  * the human channels 0:216 are never touched, in either state;
  * the mask is per sample, never once for the whole batch -- a batch-level branch
    keyed on sample 0 is exactly how layout neutrality was broken before;
  * only ``Sampler.p_sample`` applies it.  ``Sampler.cm_sample`` is the consistency
    path and the user's standing constraint is that C is neither modified nor
    retrained, so the AST guard is what makes "C is untouched" a checked claim.

The arithmetic is tested through the real helper, not a re-implementation.
"""

import ast
import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import OBJECT_CHANNEL_SLICE, zero_object_x0

SOURCE = (REPO / "code" / "models" / "infbagel.py").read_text()

HUMAN = slice(0, 216)


def _x0(batch=1, frames=16, channels=232, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, frames, channels, generator=generator)


def _flags(*values):
    return torch.tensor(list(values), dtype=torch.bool)


class SliceTests(unittest.TestCase):
    def test_slice_is_the_16_object_channels(self):
        self.assertEqual(OBJECT_CHANNEL_SLICE, slice(216, 232))
        self.assertEqual(
            len(range(*OBJECT_CHANNEL_SLICE.indices(232))),
            16,
            "com 216:219 + rot 219:228 + contact 228:232",
        )


class ArithmeticTests(unittest.TestCase):
    def test_off_returns_the_identity_object(self):
        model_output = _x0()
        self.assertIs(zero_object_x0(model_output, _flags(False), False), model_output)

    def test_off_is_the_identity_for_an_object_row_too(self):
        model_output = _x0()
        self.assertIs(zero_object_x0(model_output, _flags(True), False), model_output)

    def test_on_zeroes_the_object_block_of_a_scene_only_row(self):
        model_output = _x0()
        result = zero_object_x0(model_output, _flags(False), True)
        block = result[..., OBJECT_CHANNEL_SLICE]
        self.assertTrue(torch.equal(block, torch.zeros_like(block)))
        self.assertTrue(torch.all(block == 0.0))

    def test_on_leaves_the_216_human_channels_bitwise_unchanged(self):
        model_output = _x0()
        result = zero_object_x0(model_output, _flags(False), True)
        self.assertTrue(torch.equal(result[..., HUMAN], model_output[..., HUMAN]))

    def test_on_does_not_mutate_its_argument(self):
        model_output = _x0()
        reference = model_output.clone()
        zero_object_x0(model_output, _flags(False), True)
        self.assertTrue(torch.equal(model_output, reference))

    def test_an_object_row_comes_back_bitwise_unchanged_when_on(self):
        model_output = _x0()
        result = zero_object_x0(model_output, _flags(True), True)
        self.assertTrue(torch.equal(result, model_output))

    def test_mask_is_applied_per_sample_not_over_the_batch(self):
        scene_only = _x0(seed=1)
        with_object = _x0(seed=2)
        batched = torch.cat((scene_only, with_object), dim=0)
        result = zero_object_x0(batched, _flags(False, True), True)
        zeroed = result[0, :, OBJECT_CHANNEL_SLICE]
        self.assertTrue(torch.equal(zeroed, torch.zeros_like(zeroed)))
        self.assertTrue(torch.equal(result[1], batched[1]))

    def test_single_row_result_is_independent_of_the_batch_it_rode_in(self):
        # The layout-neutrality property itself.
        scene_only = _x0(seed=1)
        with_object = _x0(seed=2)
        alone = zero_object_x0(scene_only, _flags(False), True)
        together = zero_object_x0(
            torch.cat((scene_only, with_object), dim=0), _flags(False, True), True
        )
        self.assertTrue(torch.equal(alone[0], together[0]))

    def test_scalar_and_python_bool_flags_are_accepted(self):
        # test_infbagel_hosi passes cond['need_object'], which the LINGO driver builds
        # as a length-1 bool tensor, but the same call site is reached with a plain
        # bool elsewhere.  Both must resolve to the same mask.
        model_output = _x0()
        from_tensor = zero_object_x0(model_output, _flags(False), True)
        from_bool = zero_object_x0(model_output, False, True)
        self.assertTrue(torch.equal(from_tensor, from_bool))
        self.assertTrue(
            torch.equal(zero_object_x0(model_output, True, True), model_output)
        )

    def test_a_length_one_flag_broadcasts_over_a_wider_batch(self):
        batched = _x0(batch=4)
        result = zero_object_x0(batched, _flags(False), True)
        block = result[..., OBJECT_CHANNEL_SLICE]
        self.assertTrue(torch.equal(block, torch.zeros_like(block)))
        self.assertTrue(torch.equal(result[..., HUMAN], batched[..., HUMAN]))

    def test_dtype_and_device_survive(self):
        model_output = _x0().to(torch.float64)
        result = zero_object_x0(model_output, _flags(False), True)
        self.assertEqual(result.dtype, torch.float64)
        self.assertEqual(result.device, model_output.device)


class CallSiteTests(unittest.TestCase):
    """Static guard: the diffusion path applies it, the consistency path does not."""

    def setUp(self):
        tree = ast.parse(SOURCE)
        sampler = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ClassDef) and node.name == "Sampler"
        )
        self.methods = {
            node.name: node for node in sampler.body if isinstance(node, ast.FunctionDef)
        }

    def _calls_helper(self, method):
        return any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "zero_object_x0"
            for node in ast.walk(self.methods[method])
        )

    def test_p_sample_applies_it(self):
        self.assertIn("p_sample", self.methods)
        self.assertTrue(self._calls_helper("p_sample"))

    def test_cm_sample_does_not_apply_it(self):
        self.assertIn("cm_sample", self.methods)
        self.assertFalse(
            self._calls_helper("cm_sample"),
            "the consistency path must stay untouched: C is neither modified nor retrained",
        )

    def test_p_losses_does_not_apply_it(self):
        # It is an inference-side repair.  Touching the training loss would make the
        # A/B a different experiment.
        self.assertFalse(self._calls_helper("p_losses"))

    def test_flag_is_read_from_kwargs_in_init(self):
        self.assertIn("hsi_zero_object_x0", ast.dump(self.methods["__init__"]))

    def test_it_runs_after_cfg_and_before_the_posterior_mean(self):
        # Order is the whole mechanism: after CFG so the mix cannot reintroduce a
        # non-zero object block, before model_mean so the posterior walks those
        # channels along the q_sample(0) path.
        body = self.methods["p_sample"].body
        lines = {"cfg": None, "zero": None, "mean": None}
        for node in ast.walk(self.methods["p_sample"]):
            if isinstance(node, ast.Assign) and len(node.targets) == 1:
                target = node.targets[0]
                if not isinstance(target, ast.Name):
                    continue
                if target.id == "model_output" and isinstance(node.value, ast.BinOp):
                    lines["cfg"] = node.lineno
                if (
                    target.id == "model_output"
                    and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "zero_object_x0"
                ):
                    lines["zero"] = node.lineno
                if target.id == "model_mean":
                    lines["mean"] = node.lineno
        self.assertTrue(all(v is not None for v in lines.values()), lines)
        self.assertLess(lines["cfg"], lines["zero"], lines)
        self.assertLess(lines["zero"], lines["mean"], lines)
        self.assertTrue(body)


class ConfigTests(unittest.TestCase):
    def test_the_sampling_config_defaults_it_off(self):
        text = (
            REPO / "code" / "config" / "config_sample_infbagel_lingo_hsi.yaml"
        ).read_text()
        self.assertIn("hsi_zero_object_x0: false", text)

    def test_the_sampler_group_forwards_it_with_a_false_default(self):
        text = (REPO / "code" / "config" / "sampler" / "pelvis.yaml").read_text()
        self.assertIn(
            "hsi_zero_object_x0: ${oc.select:hsi_zero_object_x0,false}",
            text,
            "the group is shared with configs that never define the top-level key",
        )


if __name__ == "__main__":
    unittest.main()
