"""Preregistered P10 repair of the P8 hand-object geometry term.

Two flags are added to ``masked_hand_object_distance_loss``:

* ``hinge`` -- stop pulling once the palm is within ``hinge`` metres of the
  predicted surface, matching the author's inference guidance
  (``code/guidance_loss.py:36-39``, ``contact_threshold = 0.02``).
* ``detach_object`` -- route the term's gradient into the hand only, so the
  cheapest way to satisfy it is no longer to drag the predicted object.

The 2x2 that these flags define reuses the sealed weight-3 run
``p1-hoi-p8-hand-object-geom-w3-s42-20260807`` as its (hinge=0, detach=off)
cell.  That reuse is only valid while the default path is *bit-identical* to
the sealed source, so the first test class compares against the sealed blob
itself rather than against a description of it.
"""

import subprocess
import sys
import types
import unittest
from pathlib import Path

import numpy as np
import torch
from pytorch3d import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

from datasets.utils import get_smpl_parents
from priors.losses import (
    P8_CONTACT_HAND_CHANNELS,
    P8_CONTACT_THRESHOLD,
    P8_PALM_JOINTS,
    hoi_training_losses,
    masked_hand_object_distance_loss,
)
from priors.representation import REPRESENTATION

# The exact Git blob of ``code/priors/losses.py`` as sealed by commit a01f879
# ("Seal P8-P9 training-side hand-object geometry: W3 (weight=3) selected"),
# which is the source the reused weight-3 cell was trained with.  Pinning the
# blob rather than a branch tip keeps this comparison meaningful after the P10
# implementation is itself committed; a moving reference would silently decay
# into comparing the new code against itself.
SEALED_LOSSES_BLOB = "9769a66cab5fe53f224918065e46ad1bb1ea46a7"

REUSE_GATE = (
    "REUSE-VALIDITY GATE FAILED: the hinge=0 / detach=False path is no longer "
    "bit-identical to sealed blob {blob}. The P10 2x2 reuses the sealed run "
    "p1-hoi-p8-hand-object-geom-w3-s42-20260807 as its (hinge=0, detach=off) "
    "cell; that cell is INVALID unless this path reproduces the sealed "
    "objective exactly. Do not train the P10 arms until this passes."
).format(blob=SEALED_LOSSES_BLOB)


def _load_sealed_losses_module():
    """Execute the sealed ``priors/losses.py`` blob as a sibling module."""
    source = subprocess.run(
        ["git", "cat-file", "blob", SEALED_LOSSES_BLOB],
        cwd=str(ROOT),
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    module = types.ModuleType("priors._p10_sealed_losses")
    # Relative imports inside the blob (``from .representation import ...``)
    # resolve through ``__package__``.
    module.__package__ = "priors"
    module.__file__ = f"<git blob {SEALED_LOSSES_BLOB}>"
    exec(compile(source, module.__file__, "exec"), module.__dict__)
    return module


def _original_masked_hand_object_distance_loss(fk, object_surface, contact_ground_truth):
    """Hand copy of the pre-P10 body, kept as a git-independent second witness."""
    active = slice(REPRESENTATION.history_frames, None)
    batch = fk.shape[0]
    palms = fk[:, active][:, :, P8_PALM_JOINTS, :]
    surface = object_surface[:, active]
    frames = palms.shape[1]
    engaged = (
        contact_ground_truth[:, active, P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD
    ).any(dim=-1)
    distances = torch.cdist(
        palms.reshape(batch * frames, len(P8_PALM_JOINTS), 3),
        surface.reshape(batch * frames, -1, 3),
    )
    nearest = distances.min(dim=-1)[0].reshape(batch, frames, len(P8_PALM_JOINTS))
    per_frame = nearest.square().mean(dim=-1)
    weight = engaged.to(per_frame)
    return (per_frame * weight).sum() / weight.sum().clamp_min(1.0)


def _synthetic_batch(B=6, T=16, V=64, seed=42):
    """Fixed-seed geometry with a synthetic but non-degenerate contact mask."""
    g = torch.Generator().manual_seed(seed)
    contact = torch.zeros(B, T, 4)
    engaged = torch.rand(B, T, 2, generator=g) < 0.6
    contact[..., :2] = engaged.to(contact)
    return {
        "fk": torch.randn(B, T, 24, 3, generator=g) * 0.5,
        "surface": torch.randn(B, T, V, 3, generator=g) * 0.3,
        "contact_gt": contact,
    }


def _real_contact_batch(B=4, T=16, V=80, seed=42):
    """Same construction as the P8 suite: real GT contact annotation files."""
    import glob

    g = torch.Generator().manual_seed(seed)
    files = sorted(glob.glob(str(ROOT / "data/test/contact_label_npy_files/*.npy")))
    if len(files) < B:
        raise FileNotFoundError(f"need {B} annotation files, found {len(files)}")
    rows = []
    for path in files[:B]:
        array = np.load(path)
        if array.shape[0] < T:
            array = np.pad(array, ((0, T - array.shape[0]), (0, 0)), mode="edge")
        rows.append(torch.from_numpy(array[:T].astype("float32")))
    return {
        "fk": torch.randn(B, T, 24, 3, generator=g) * 0.5,
        "surface": torch.randn(B, T, V, 3, generator=g) * 0.3,
        "contact_gt": torch.stack(rows),
    }


def _engaged_frame_count(contact):
    active = slice(REPRESENTATION.history_frames, None)
    return int(
        (contact[:, active, P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD)
        .any(dim=-1)
        .sum()
    )


class BitIdentityTest(unittest.TestCase):
    """The default path must reproduce the sealed objective bit for bit."""

    def _assert_identical(self, reference, produced, label):
        self.assertTrue(
            torch.equal(reference, produced),
            f"{REUSE_GATE}\n[{label}] sealed={reference.item()!r} new={produced.item()!r}",
        )
        self.assertEqual(
            repr(reference.item()), repr(produced.item()),
            f"{REUSE_GATE}\n[{label}] repr mismatch",
        )

    def test_defaults_match_sealed_blob(self):
        sealed = _load_sealed_losses_module()
        for label, batch in (
            ("synthetic", _synthetic_batch()),
            ("real-contact", _real_contact_batch()),
        ):
            self.assertGreater(_engaged_frame_count(batch["contact_gt"]), 0, label)
            reference = sealed.masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"],
            )
            self.assertGreater(float(reference), 0.0, f"{label} reference is a trivial zero")
            for kwargs in ({}, {"hinge": 0.0}, {"detach_object": False},
                           {"hinge": 0.0, "detach_object": False}):
                produced = masked_hand_object_distance_loss(
                    batch["fk"], batch["surface"], batch["contact_gt"], **kwargs,
                )
                self._assert_identical(reference, produced, f"{label} {kwargs}")

    def test_defaults_match_hand_copy_of_original_expression(self):
        """Second witness that does not depend on Git being available."""
        for label, batch in (
            ("synthetic", _synthetic_batch(seed=1)),
            ("real-contact", _real_contact_batch()),
        ):
            reference = _original_masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"],
            )
            produced = masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"],
            )
            self._assert_identical(reference, produced, label)

    def test_hand_copy_agrees_with_sealed_blob(self):
        """Guard the second witness itself against a transcription error."""
        sealed = _load_sealed_losses_module()
        batch = _synthetic_batch(seed=3)
        self._assert_identical(
            sealed.masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"],
            ),
            _original_masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"],
            ),
            "hand-copy",
        )

    def test_default_gradient_matches_sealed_blob(self):
        """A bit-identical value with a different gradient would still be wrong."""
        sealed = _load_sealed_losses_module()
        batch = _synthetic_batch(seed=5)
        grads = []
        for term in (sealed.masked_hand_object_distance_loss,
                     masked_hand_object_distance_loss):
            fk = batch["fk"].clone().requires_grad_(True)
            surface = batch["surface"].clone().requires_grad_(True)
            loss = term(fk, surface, batch["contact_gt"])
            grads.append(torch.autograd.grad(loss, [fk, surface]))
        self.assertTrue(torch.equal(grads[0][0], grads[1][0]), REUSE_GATE + "\nFK gradient")
        self.assertTrue(
            torch.equal(grads[0][1], grads[1][1]), REUSE_GATE + "\nsurface gradient"
        )
        self.assertGreater(float(grads[0][1].norm()), 0.0, "sealed surface gradient is zero")

    def test_full_objective_defaults_are_unchanged(self):
        """Explicit defaults at the ``hoi_training_losses`` level change nothing."""
        batch = _full_objective_batch()
        implicit = hoi_training_losses(
            *batch, hand_object_contact_weight=3.0,
            fk_weight=0.3569973401779424, object_surface_weight=0.4772322188400037,
            fk_foot_temporal_routing=True,
        )
        explicit = hoi_training_losses(
            *batch, hand_object_contact_weight=3.0,
            hand_object_contact_hinge=0.0, hand_object_contact_detach_object=False,
            fk_weight=0.3569973401779424, object_surface_weight=0.4772322188400037,
            fk_foot_temporal_routing=True,
        )
        self.assertEqual(sorted(implicit), sorted(explicit))
        self.assertGreater(
            float(implicit["hand_object_contact_geometry"]), 0.0,
            "the geometry term is zero, so this comparison would prove nothing",
        )
        for name in implicit:
            self.assertTrue(
                torch.equal(implicit[name], explicit[name]),
                f"{REUSE_GATE}\nloss '{name}': {implicit[name].item()!r} vs {explicit[name].item()!r}",
            )

    def test_full_objective_matches_the_sealed_blob(self):
        """The actual reuse claim: the sealed W3 objective == the P10 baseline cell.

        Compares every loss entry AND the gradient of ``total`` against the
        source the weight-3 run was trained with.
        """
        sealed = _load_sealed_losses_module()
        weights = dict(
            fk_weight=0.3569973401779424,
            object_surface_weight=0.4772322188400037,
            hand_object_contact_weight=3.0,
            fk_foot_temporal_routing=True,
        )
        outputs = []
        for term in (sealed.hoi_training_losses, hoi_training_losses):
            batch = _full_objective_batch()
            losses = term(*batch, **weights)
            gradient, = torch.autograd.grad(losses["total"], batch[0])
            outputs.append(({k: v.detach() for k, v in losses.items()}, gradient))
        reference, produced = outputs
        self.assertEqual(sorted(reference[0]), sorted(produced[0]))
        self.assertGreater(float(reference[0]["hand_object_contact_geometry"]), 0.0)
        for name, value in reference[0].items():
            self.assertTrue(
                torch.equal(value, produced[0][name]),
                f"{REUSE_GATE}\nloss '{name}': {value.item()!r} vs {produced[0][name].item()!r}",
            )
        self.assertTrue(
            torch.equal(reference[1], produced[1]),
            REUSE_GATE + "\ntotal-loss gradient differs from the sealed source",
        )

    def test_new_arguments_are_keyword_only(self):
        """A positional fourth argument must fail, not be read as ``hinge``."""
        batch = _synthetic_batch(B=2, T=8, V=16)
        with self.assertRaises(TypeError):
            masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"], 0.02,
            )

    def test_default_branch_keeps_the_literal_sealed_expression(self):
        """Pin the structure the reuse argument rests on.

        ``(nearest - 0.0).clamp_min(0.0).square()`` happens to be numerically
        identical to ``nearest.square()`` for the non-negative distances cdist
        produces, so no value comparison can enforce this.  The preregistration
        nevertheless claims the sealed cell runs the *same expression*, and this
        keeps a later "simplification" from quietly folding the branch away.
        """
        source = (ROOT / "code" / "priors" / "losses.py").read_text(encoding="utf-8")
        body = source.split("def masked_hand_object_distance_loss", 1)[1]
        body = body.split("\ndef ", 1)[0]
        self.assertIn("if hinge > 0.0:", body)
        self.assertIn("per_frame = nearest.square().mean(dim=-1)", body)
        # In-place detach would reach through into the caller's graph.
        self.assertNotIn(".detach_(", body)


class HingeSemanticsTest(unittest.TestCase):
    """The hinge must release inside the threshold and shrink the term outside."""

    @staticmethod
    def _touching_batch(gap, B=2, T=8, V=12):
        """Palms held exactly ``gap`` metres from a single-point object surface."""
        fk = torch.zeros(B, T, 24, 3)
        surface = torch.zeros(B, T, V, 3)
        fk[:, :, P8_PALM_JOINTS, 0] = gap
        contact = torch.zeros(B, T, 4)
        contact[..., :2] = 1.0
        return fk, surface, contact

    def test_inside_hinge_is_exactly_zero_with_zero_gradient(self):
        hinge = 0.02
        fk, surface, contact = self._touching_batch(gap=0.005)
        self.assertGreater(_engaged_frame_count(contact), 0, "mask must engage")
        unhinged = masked_hand_object_distance_loss(fk, surface, contact)
        self.assertGreater(
            float(unhinged), 0.0,
            "the zero below must come from the hinge, not from an empty mask",
        )
        fk_leaf = fk.clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(fk_leaf, surface, contact, hinge=hinge)
        self.assertEqual(float(loss), 0.0)
        gradient, = torch.autograd.grad(loss, fk_leaf)
        self.assertEqual(float(gradient.abs().max()), 0.0)

    def test_at_exactly_the_hinge_the_term_is_zero(self):
        fk, surface, contact = self._touching_batch(gap=0.02)
        loss = masked_hand_object_distance_loss(fk, surface, contact, hinge=0.02)
        self.assertLess(float(loss), 1e-12)

    def test_outside_hinge_is_a_strict_reduction(self):
        hinge = 0.02
        for gap in (0.05, 0.1, 0.3):
            fk, surface, contact = self._touching_batch(gap=gap)
            flat = masked_hand_object_distance_loss(fk, surface, contact)
            hinged = masked_hand_object_distance_loss(fk, surface, contact, hinge=hinge)
            self.assertGreater(float(flat), 0.0, gap)
            self.assertGreater(float(hinged), 0.0, gap)
            self.assertLess(float(hinged), float(flat), gap)
            self.assertAlmostEqual(float(hinged), (gap - hinge) ** 2, places=6, msg=gap)

    def test_hinge_matches_an_independent_formula_on_a_random_batch(self):
        """Guards the slicing/order of the hinged branch, not just its extremes."""
        hinge = 0.02
        batch = _synthetic_batch(seed=11)
        active = slice(REPRESENTATION.history_frames, None)
        palms = batch["fk"][:, active][:, :, P8_PALM_JOINTS, :]
        surface = batch["surface"][:, active]
        B, frames = palms.shape[0], palms.shape[1]
        nearest = torch.cdist(
            palms.reshape(B * frames, len(P8_PALM_JOINTS), 3),
            surface.reshape(B * frames, -1, 3),
        ).min(dim=-1)[0].reshape(B, frames, len(P8_PALM_JOINTS))
        engaged = (
            batch["contact_gt"][:, active, P8_CONTACT_HAND_CHANNELS] > P8_CONTACT_THRESHOLD
        ).any(dim=-1).to(nearest)
        per_frame = torch.clamp(nearest - hinge, min=0.0).square().mean(dim=-1)
        expected = (per_frame * engaged).sum() / engaged.sum().clamp_min(1.0)
        produced = masked_hand_object_distance_loss(
            batch["fk"], batch["surface"], batch["contact_gt"], hinge=hinge,
        )
        torch.testing.assert_close(produced, expected, rtol=0, atol=0)
        self.assertLess(
            float(produced),
            float(masked_hand_object_distance_loss(
                batch["fk"], batch["surface"], batch["contact_gt"])),
        )

    def test_negative_or_nonfinite_hinge_is_rejected(self):
        """A silently ignored negative hinge would look like a valid arm."""
        batch = _synthetic_batch(B=2, T=8, V=16)
        for bad in (-0.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError, msg=bad):
                masked_hand_object_distance_loss(
                    batch["fk"], batch["surface"], batch["contact_gt"], hinge=bad,
                )


class DetachSemanticsTest(unittest.TestCase):
    """``detach_object`` must remove the object gradient and nothing else."""

    def test_detach_kills_object_gradient_but_keeps_hand_gradient(self):
        batch = _synthetic_batch(seed=13)
        fk = batch["fk"].clone().requires_grad_(True)
        surface = batch["surface"].clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(
            fk, surface, batch["contact_gt"], detach_object=True,
        )
        grad_fk, grad_surface = torch.autograd.grad(
            loss, [fk, surface], allow_unused=True,
        )
        self.assertGreater(float(grad_fk.norm()), 0.0, "hand gradient vanished")
        self.assertTrue(
            grad_surface is None or float(grad_surface.abs().max()) == 0.0,
            f"object surface still received gradient: {grad_surface}",
        )

    def test_without_detach_the_object_receives_gradient(self):
        batch = _synthetic_batch(seed=13)
        fk = batch["fk"].clone().requires_grad_(True)
        surface = batch["surface"].clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(
            fk, surface, batch["contact_gt"], detach_object=False,
        )
        grad_fk, grad_surface = torch.autograd.grad(loss, [fk, surface])
        self.assertGreater(float(grad_fk.norm()), 0.0)
        self.assertGreater(float(grad_surface.norm()), 0.0)

    def test_detach_does_not_change_the_value(self):
        batch = _synthetic_batch(seed=17)
        flat = masked_hand_object_distance_loss(
            batch["fk"], batch["surface"], batch["contact_gt"],
        )
        detached = masked_hand_object_distance_loss(
            batch["fk"], batch["surface"], batch["contact_gt"], detach_object=True,
        )
        self.assertTrue(torch.equal(flat, detached))

    def test_detach_does_not_mutate_the_caller_tensor(self):
        """An in-place ``detach_()`` here would silently kill object supervision."""
        batch = _synthetic_batch(seed=19)
        fk = batch["fk"].clone().requires_grad_(True)
        surface = batch["surface"].clone().requires_grad_(True)
        downstream = surface.square().mean()
        masked_hand_object_distance_loss(
            fk, surface, batch["contact_gt"], detach_object=True,
        )
        self.assertTrue(surface.requires_grad)
        gradient, = torch.autograd.grad(downstream, surface)
        self.assertGreater(float(gradient.norm()), 0.0, "caller graph was destroyed")
        # A non-leaf caller tensor is the real situation in hoi_training_losses.
        non_leaf = surface * 2.0
        masked_hand_object_distance_loss(
            fk, non_leaf, batch["contact_gt"], detach_object=True,
        )
        self.assertIsNotNone(non_leaf.grad_fn, "caller tensor lost its grad_fn")

    def test_hinge_and_detach_compose(self):
        batch = _synthetic_batch(seed=23)
        fk = batch["fk"].clone().requires_grad_(True)
        surface = batch["surface"].clone().requires_grad_(True)
        loss = masked_hand_object_distance_loss(
            fk, surface, batch["contact_gt"], hinge=0.02, detach_object=True,
        )
        grad_fk, grad_surface = torch.autograd.grad(
            loss, [fk, surface], allow_unused=True,
        )
        self.assertGreater(float(loss), 0.0)
        self.assertGreater(float(grad_fk.norm()), 0.0)
        self.assertTrue(grad_surface is None or float(grad_surface.abs().max()) == 0.0)


def _full_objective_batch(B=3, T=16, points=24, seed=42):
    """A well-formed argument tuple for ``hoi_training_losses``."""
    g = torch.Generator().manual_seed(seed)
    prediction = torch.randn(B, T, 232, generator=g) * 0.05
    target = torch.randn(B, T, 232, generator=g) * 0.05
    identity_6d = torch.zeros(22, 6)
    identity_6d[:, 0] = 1.0
    identity_6d[:, 4] = 1.0
    identity_9 = torch.eye(3).reshape(9)
    for tensor in (prediction, target):
        tensor[..., 84:216] += identity_6d.reshape(1, 1, 132)
        tensor[..., 219:228] += identity_9.reshape(1, 1, 9)
    # Ground-truth hand contact on the second half of every window.  The
    # geometry term reads channels 228:230 of the TARGET; setting anything else
    # leaves the mask empty and every geometry assertion below becomes vacuous.
    target[..., 228:232] = 0.0
    target[:, T // 2:, 228:230] = 1.0
    if _engaged_frame_count(target[..., 228:232]) == 0:
        raise AssertionError("fixture engages no contact frame; assertions would be vacuous")
    prediction = prediction.requires_grad_(True)
    parents = torch.as_tensor(get_smpl_parents(use_joints24=True), dtype=torch.long)
    return (
        prediction,
        target,
        torch.randn(B, 9, generator=g) * 0.1,
        torch.randn(B, 24, 3, generator=g) * 0.1,
        parents,
        torch.full((3,), -2.0),
        torch.full((3,), 2.0),
        torch.full((3,), -2.0),
        torch.full((3,), 2.0),
        torch.ones(B, dtype=torch.bool),
        torch.randn(B, points, 3, generator=g) * 0.2,
        torch.eye(3).repeat(B, 1, 1),
        torch.eye(3).repeat(B, 1, 1),
    )


OBJECT_CHANNELS = slice(216, 228)
HUMAN_ROTATION_CHANNELS = slice(84, 216)


class ObjectSurfaceRegressionTest(unittest.TestCase):
    """Detaching inside the geometry term must not weaken object supervision."""

    def _losses(self, batch, **kwargs):
        losses = hoi_training_losses(
            *batch,
            fk_weight=0.3569973401779424,
            object_surface_weight=0.4772322188400037,
            hand_object_contact_weight=3.0,
            fk_foot_temporal_routing=True,
            **kwargs,
        )
        if not kwargs.get("hand_object_contact_hinge"):
            self.assertGreater(
                float(losses["hand_object_contact_geometry"]), 0.0,
                "fixture leaves the geometry term at zero; assertions are vacuous",
            )
        return losses

    def test_object_surface_still_receives_object_gradient_under_detach(self):
        batch = _full_objective_batch()
        prediction = batch[0]
        losses = self._losses(batch, hand_object_contact_detach_object=True)
        gradient, = torch.autograd.grad(losses["object_surface"], prediction)
        self.assertGreater(
            float(gradient[..., OBJECT_CHANNELS].abs().sum()), 0.0,
            "object_surface lost its gradient to the object channels: the detach "
            "leaked out of masked_hand_object_distance_loss (in-place detach_?)",
        )
        self.assertTrue(torch.isfinite(gradient).all())

    def test_object_surface_gradient_is_bitwise_unaffected_by_the_flags(self):
        reference = None
        for kwargs in (
            {},
            {"hand_object_contact_detach_object": True},
            {"hand_object_contact_hinge": 0.02},
            {"hand_object_contact_hinge": 0.02, "hand_object_contact_detach_object": True},
        ):
            batch = _full_objective_batch()
            losses = self._losses(batch, **kwargs)
            gradient, = torch.autograd.grad(losses["object_surface"], batch[0])
            if reference is None:
                reference = gradient
                self.assertGreater(float(reference[..., OBJECT_CHANNELS].abs().sum()), 0.0)
            else:
                self.assertTrue(
                    torch.equal(reference, gradient),
                    f"object_surface gradient changed under {kwargs}",
                )

    def test_geometry_term_object_gradient_is_removed_only_under_detach(self):
        batch = _full_objective_batch()
        prediction = batch[0]
        attached = self._losses(batch, hand_object_contact_detach_object=False)
        attached_grad, = torch.autograd.grad(
            attached["hand_object_contact_geometry"], prediction, retain_graph=True,
        )
        self.assertGreater(
            float(attached_grad[..., OBJECT_CHANNELS].abs().sum()), 0.0,
            "baseline geometry term does not reach the object channels; the "
            "detach arm would then be a no-op",
        )
        batch = _full_objective_batch()
        prediction = batch[0]
        detached = self._losses(batch, hand_object_contact_detach_object=True)
        detached_grad, = torch.autograd.grad(
            detached["hand_object_contact_geometry"], prediction,
        )
        self.assertEqual(
            float(detached_grad[..., OBJECT_CHANNELS].abs().max()), 0.0,
            "detach did not remove the object-channel gradient",
        )
        self.assertGreater(
            float(detached_grad[..., HUMAN_ROTATION_CHANNELS].abs().sum()), 0.0,
            "detach also removed the hand gradient",
        )

    def test_total_still_trains_the_object_under_detach(self):
        batch = _full_objective_batch()
        losses = self._losses(batch, hand_object_contact_detach_object=True)
        gradient, = torch.autograd.grad(losses["total"], batch[0])
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient[..., OBJECT_CHANNELS].abs().sum()), 0.0)

    def test_hinge_reduces_the_geometry_term_in_the_full_objective(self):
        flat = self._losses(_full_objective_batch())
        hinged = self._losses(_full_objective_batch(), hand_object_contact_hinge=0.02)
        self.assertGreater(float(flat["hand_object_contact_geometry"]), 0.0)
        self.assertLess(
            float(hinged["hand_object_contact_geometry"]),
            float(flat["hand_object_contact_geometry"]),
        )
        for name in ("joint_position", "object_surface", "fk", "velocity", "object_goal"):
            self.assertTrue(torch.equal(flat[name], hinged[name]), name)

    def test_weight_zero_still_omits_the_term(self):
        losses = hoi_training_losses(
            *_full_objective_batch(),
            hand_object_contact_hinge=0.02,
            hand_object_contact_detach_object=True,
        )
        self.assertNotIn("hand_object_contact_geometry", losses)


class MetricsSchemaTest(unittest.TestCase):
    """The P10 arms must record the term they manipulate, and only they.

    ``hoi_training_losses`` emits ``hand_object_contact_geometry`` only when the
    weight is non-zero, so the recorder derives its key list per configuration
    instead of extending ``LOSS_KEYS`` -- which eleven tools import to index a
    loss dictionary, and which ``tests/test_hoi_d2ag.py`` mirrors verbatim.
    """

    ACTIVE = (
        "config_train_hoi_prior_p10_hinge",
        "config_train_hoi_prior_p10_detach",
        "config_train_hoi_prior_p9w3",
    )
    INACTIVE = (
        "config_train_hoi_prior",
        "config_train_hoi_prior_d2ai",
        "config_train_hoi_prior_d2aj",
    )

    def setUp(self):
        import os

        os.environ.setdefault("ROOT_DIR", str(ROOT))
        self.config_dir = str((ROOT / "code" / "config").resolve())
        import train_hoi_prior

        self.train = train_hoi_prior

    def _compose(self, name, **overrides):
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf

        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            cfg = compose(config_name=name)
        # ``hand_object_contact_weight`` is not declared in the base config, so a
        # composed config rejects it under struct mode; arm config files add it
        # through composition instead.  Relax struct mode to build the synthetic
        # configurations this test needs.
        OmegaConf.set_struct(cfg, False)
        for key, value in overrides.items():
            cfg[key] = value
        return cfg

    def test_active_configs_record_the_geometry_term(self):
        for name in self.ACTIVE:
            keys = self.train._loss_keys(self._compose(name))
            self.assertEqual(keys[-1], self.train.HAND_OBJECT_GEOMETRY_KEY, name)
            self.assertEqual(keys[:-1], self.train.LOSS_KEYS, name)

    def test_inactive_configs_keep_the_sealed_schema(self):
        """A zero-weight run's metrics schema and all-reduce width are unchanged."""
        for name in self.INACTIVE:
            keys = self.train._loss_keys(self._compose(name))
            self.assertIs(keys, self.train.LOSS_KEYS, name)
            self.assertNotIn(self.train.HAND_OBJECT_GEOMETRY_KEY, keys, name)
            self.assertEqual(len(keys), 12, name)

    def test_every_recorded_key_exists_in_the_loss_dictionary(self):
        """The invariant the metrics writer indexes on; a miss is a KeyError mid-run."""
        for weight in (3.0, 0.0):
            losses = hoi_training_losses(
                *_full_objective_batch(),
                fk_weight=0.3569973401779424,
                object_surface_weight=0.4772322188400037,
                hand_object_contact_weight=weight,
                fk_foot_temporal_routing=True,
            )
            cfg = self._compose(
                "config_train_hoi_prior_p10_hinge", hand_object_contact_weight=weight,
            )
            missing = [k for k in self.train._loss_keys(cfg) if k not in losses]
            self.assertEqual(missing, [], f"weight={weight}")
            if weight == 0.0:
                self.assertNotIn(self.train.HAND_OBJECT_GEOMETRY_KEY, losses)

    def test_variants_that_drop_the_term_fail_closed(self):
        """D2-Z/D2-AB route to loss variants that never accept the weight."""
        for flag in ("d2z_immutable_gt_near_ground_gating", "d2ab_predicted_support_no_slip"):
            cfg = self._compose(
                "config_train_hoi_prior", hand_object_contact_weight=3.0, **{flag: True},
            )
            with self.assertRaises(ValueError, msg=flag):
                self.train._loss_keys(cfg)
            # Harmless while the term is off, which is how every such run is configured.
            cfg[flag] = True
            cfg["hand_object_contact_weight"] = 0.0
            self.assertIs(self.train._loss_keys(cfg), self.train.LOSS_KEYS, flag)

    def test_recorder_uses_the_derived_keys_everywhere(self):
        """A missed call site would silently drop the term from one record."""
        source = (ROOT / "code" / "train_hoi_prior.py").read_text(encoding="utf-8")
        body = source.split("HAND_OBJECT_GEOMETRY_KEY = ", 1)[1]
        self.assertNotIn("for key in LOSS_KEYS", body)
        self.assertNotIn("enumerate(LOSS_KEYS)", body)
        self.assertNotIn("for key in LOSS_KEYS}", body)
        self.assertNotIn("len(LOSS_KEYS)", body)


class ConfigPlumbingTest(unittest.TestCase):
    """The two new arm configs must resolve to exactly the intended objective."""

    SEALED_CELL = "config_train_hoi_prior_p9w3"
    ARMS = {
        "config_train_hoi_prior_p10_hinge": {
            "run_id": "p1-hoi-p10-geom-hinge-s42-20260809",
            "hand_object_contact_hinge": 0.02,
            "hand_object_contact_detach_object": False,
        },
        "config_train_hoi_prior_p10_detach": {
            "run_id": "p1-hoi-p10-geom-detach-s42-20260809",
            "hand_object_contact_hinge": 0.0,
            "hand_object_contact_detach_object": True,
        },
    }

    def setUp(self):
        import os

        os.environ.setdefault("ROOT_DIR", str(ROOT))
        self.config_dir = str((ROOT / "code" / "config").resolve())
        import train_hoi_prior

        self.train = train_hoi_prior

    def _compose(self, name):
        from hydra import compose, initialize_config_dir

        with initialize_config_dir(config_dir=self.config_dir, version_base=None):
            return compose(config_name=name)

    def test_base_config_declares_the_flags_as_no_ops(self):
        cfg = self._compose("config_train_hoi_prior")
        self.assertEqual(float(cfg.hand_object_contact_hinge), 0.0)
        self.assertIs(bool(cfg.hand_object_contact_detach_object), False)

    def test_sealed_cell_config_stays_a_no_op(self):
        cfg = self._compose(self.SEALED_CELL)
        self.assertEqual(float(cfg.hand_object_contact_hinge), 0.0)
        self.assertIs(bool(cfg.hand_object_contact_detach_object), False)
        self.assertEqual(float(cfg.hand_object_contact_weight), 3.0)

    def test_arm_configs_carry_the_intended_values(self):
        for name, expected in self.ARMS.items():
            cfg = self._compose(name)
            self.assertEqual(str(cfg.run_id), expected["run_id"], name)
            self.assertEqual(str(cfg.exp_name), expected["run_id"], name)
            self.assertEqual(str(cfg.subphase), "1B-P10", name)
            self.assertEqual(
                float(cfg.hand_object_contact_hinge),
                expected["hand_object_contact_hinge"], name,
            )
            # A YAML string would make bool(...) True for "false"; require a bool.
            self.assertIsInstance(cfg.hand_object_contact_detach_object, bool, name)
            self.assertIs(
                cfg.hand_object_contact_detach_object,
                expected["hand_object_contact_detach_object"], name,
            )

    def test_arms_keep_the_sealed_budget_and_recipe(self):
        for name in self.ARMS:
            cfg = self._compose(name)
            self.assertEqual(int(cfg.max_processed_windows), 299_520_000, name)
            self.assertEqual(int(cfg.effective_batch_size), 2048, name)
            self.assertEqual(int(cfg.batch_size), 512, name)
            self.assertEqual(int(cfg.num_gpus), 4, name)
            self.assertEqual(int(cfg.seed), 42, name)
            self.assertEqual(float(cfg.hand_object_contact_weight), 3.0, name)
            self.assertEqual(float(cfg.fk_weight), 0.3569973401779424, name)
            self.assertEqual(float(cfg.object_surface_weight), 0.4772322188400037, name)
            self.assertIsNone(cfg.init_checkpoint, name)
            self.assertIsNone(cfg.weight_init_checkpoint, name)
            self.assertTrue(self.train._is_d2ai(cfg), name)
            self.assertFalse(self.train._is_d2x(cfg), name)
            self.assertTrue(bool(cfg.fk_foot_temporal_routing), name)
            self.train._validate_fk_foot_temporal_routing_mode(cfg)

    def test_arms_differ_from_the_sealed_cell_only_where_intended(self):
        from omegaconf import OmegaConf

        sealed = OmegaConf.to_container(self._compose(self.SEALED_CELL), resolve=False)
        allowed = {
            "exp_name", "run_id", "subphase",
            "hand_object_contact_hinge", "hand_object_contact_detach_object",
        }
        for name in self.ARMS:
            arm = OmegaConf.to_container(self._compose(name), resolve=False)
            self.assertEqual(set(sealed), set(arm), name)
            changed = {key for key in sealed if sealed[key] != arm[key]}
            self.assertTrue(
                changed <= allowed,
                f"{name} changed unexpected keys: {sorted(changed - allowed)}",
            )
            self.assertIn("subphase", changed, name)

    def test_resume_contract_records_the_objective(self):
        for name, expected in self.ARMS.items():
            contract = self.train._resume_contract(self._compose(name))
            self.assertEqual(contract["hand_object_contact_weight"], 3.0, name)
            self.assertEqual(
                contract["hand_object_contact_hinge"],
                expected["hand_object_contact_hinge"], name,
            )
            self.assertIs(
                contract["hand_object_contact_detach_object"],
                expected["hand_object_contact_detach_object"], name,
            )
        sealed = self.train._resume_contract(self._compose(self.SEALED_CELL))
        self.assertEqual(sealed["hand_object_contact_hinge"], 0.0)
        self.assertIs(sealed["hand_object_contact_detach_object"], False)
        self.assertNotEqual(
            sealed, self.train._resume_contract(self._compose("config_train_hoi_prior_p10_hinge")),
        )

    def test_trainer_forwards_both_flags_to_the_loss(self):
        """A declared-but-unread config field would be a silent no-op arm."""
        source = (ROOT / "code" / "train_hoi_prior.py").read_text(encoding="utf-8")
        call = source.split("return hoi_training_losses(", 1)[1].split("\n    )", 1)[0]
        self.assertIn("hand_object_contact_hinge=", call)
        self.assertIn("hand_object_contact_detach_object=", call)


if __name__ == "__main__":
    unittest.main()
