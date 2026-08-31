"""Pin the B-match seam term: OFF by default, bitwise inert when off, and exactly
the two frames and the one channel block it claims.

WHY THIS TEST EXISTS.  The seam term is a training-objective change on the expert
whose checkpoint every sealed B v2 cell was produced from.  If it were not exactly
inert when off, every future default-configuration run would silently stop being
comparable to those cells, and nothing else in the suite would notice -- the
difference would be a small change in a loss scalar, not a crash.  So inertness is
proven two ways here: STRUCTURALLY, that no seam statement can execute outside the
weight guard and that the five-term assembly is untouched, and NUMERICALLY, that the
assembled value is the released expression itself.

WHY THE TERM IS A PER-FRAME REWEIGHT AND NOT AN EXPLICIT ACCELERATION MATCH.  The
history frames are clean GT at every step (get_mask runs at p=1.0), so

    a_hat[1] - a_star[1] = (p_hat[2] - 2p[1] + p[0]) - (p[2] - 2p[1] + p[0])
                         = p_hat[2] - p[2]

identically.  Matching the seam acceleration IS weighting the first generated
frame's position residual.  The measured defect shape settles the second frame too:
the per-episode ratio of the d=0 excess to the d=-1 excess is 0.730 [0.695, 0.771],
which rejects an isolated single-frame error (that predicts 2.0) and matches a
constant offset of the new window's body, under which the reweight and the explicit
second-difference form are algebraically the same.

Model C is out of scope by instruction: consistency_loss must carry no seam term,
and test_consistency_loss_has_no_seam_term asserts it.

Frozen spec: .claude/scratch/bmatch_20260825/FROZEN_PILOT_SPEC.md sections 1 and 2.
Preregistration: docs/plan/PHASE_1C_HSI.md, 2026-08-25 third section, commit 519caba.
"""

import ast
import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "code"))

from models.infbagel import Sampler

SOURCE = (REPO / "code" / "models" / "infbagel.py").read_text()
TREE = ast.parse(SOURCE)

WINDOW_FRAMES = 16
CHANNELS = 232
POS_CHANNELS = 84
AUTO_REGRE_NUM = 2
SEAM_FRAMES = 2


def _sampler(**kwargs):
    return Sampler(
        device="cpu", mask_ind=0, emb_f=None, batch_size=1, channel=CHANNELS,
        auto_regre_num=AUTO_REGRE_NUM, timesteps=500, ddim_timesteps=25,
        cm_timesteps=16, **kwargs
    )


def _method(name):
    for node in ast.walk(TREE):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError("method %r not found in infbagel.py" % name)


def _uses_sampler_attribute(node, attribute):
    return any(
        isinstance(child, ast.Attribute)
        and isinstance(child.value, ast.Name)
        and child.value.id == "self"
        and child.attr == attribute
        for child in ast.walk(node)
    )


def _batch(seed=20260825):
    """A fixed synthetic batch.  No model, no dataset, no scene voxels, no GPU."""
    generator = torch.Generator().manual_seed(seed)
    shape = (3, WINDOW_FRAMES, CHANNELS)
    x_start = torch.randn(shape, generator=generator, dtype=torch.float32)
    predicted = torch.randn(shape, generator=generator, dtype=torch.float32)
    # get_mask(x_start, -1, p=1., fixed_frame=auto_regre_num): the prefix frames of
    # every sample, all channels.  p=1.0 makes this deterministic, which is what
    # makes the loss denominators below exact rather than expected values.
    mask = torch.zeros(shape, dtype=torch.bool)
    mask[:, :AUTO_REGRE_NUM, :] = True
    return x_start, predicted, mask


def _base_terms(x_start, predicted, mask):
    """The five released base terms, transcribed from p_losses."""
    inv = torch.logical_not(mask)
    return (
        F.mse_loss(x_start[:, :, :84][inv[:, :, :84]], predicted[:, :, :84][inv[:, :, :84]]),
        F.l1_loss(x_start[:, :, 84:216][inv[:, :, 84:216]], predicted[:, :, 84:216][inv[:, :, 84:216]]),
        F.mse_loss(x_start[:, :, 216:219][inv[:, :, 216:219]], predicted[:, :, 216:219][inv[:, :, 216:219]]),
        F.l1_loss(x_start[:, :, 219:228][inv[:, :, 219:228]], predicted[:, :, 219:228][inv[:, :, 219:228]]),
        F.l1_loss(x_start[:, :, 228:232][inv[:, :, 228:232]], predicted[:, :, 228:232][inv[:, :, 228:232]]),
    )


def _seam_term(x_start, predicted, n=AUTO_REGRE_NUM):
    return F.mse_loss(x_start[:, n:n + SEAM_FRAMES, :84], predicted[:, n:n + SEAM_FRAMES, :84])


# The base objective, as released: five terms, in this order, and the ONE weight
# that 2026-08-26 added to the position term.  loss_w_jpos defaults to exactly 1.0,
# so the default path is bitwise the released sum -- but the assembly itself is now
# the thing that has to be pinned, because a literal source string no longer can be.
BASE_TERMS = ("loss_jpos", "loss_jrot", "loss_otrans", "loss_orot", "loss_contact")
WEIGHTED_TERM = "loss_jpos"
WEIGHT_ATTR = "loss_w_jpos"

RELEASED_ASSEMBLY = (
    "loss = self.loss_w_jpos * loss_jpos + loss_jrot + loss_otrans + loss_orot + loss_contact"
)


def _parse_assign(line):
    """One statement, parsed standalone.  Lets the tests below run the real
    checker against deliberate mutations instead of only against the real file."""
    return ast.parse(line).body[0]


def _add_chain(node):
    """Flatten the left-nested `a + b + c` BinOp that Python builds into [a, b, c].

    Only `.left` recurses, so a parenthesised `(b + c)` stays one node and is
    therefore rejected downstream as a non-bare term rather than silently
    flattened into two.
    """
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        return _add_chain(node.left) + [node.right]
    return [node]


def _base_assembly(func):
    """Locate the base objective structurally.

    Returns (assembly_node_or_None, every_unguarded_rebinding_of_loss).  The base
    assembly is the `loss = ...` in the function's own body that does not read
    `loss` back -- which distinguishes it from `loss = loss + ...` accumulations
    without depending on line numbers or on how many of them there are.
    """
    rebinds = []
    for stmt in func.body:
        if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
                and getattr(stmt.targets[0], "id", None) == "loss"):
            rebinds.append(stmt)
        elif isinstance(stmt, ast.AugAssign) and getattr(stmt.target, "id", None) == "loss":
            rebinds.append(stmt)
    fresh = [
        stmt for stmt in rebinds
        if isinstance(stmt, ast.Assign)
        and "loss" not in {n.id for n in ast.walk(stmt.value) if isinstance(n, ast.Name)}
    ]
    return (fresh[0] if fresh else None), rebinds


def _assembly_violations(node):
    """Every way the base objective could stop being the released five terms.

    Returns a list of violations; empty means the invariant holds.  It takes a node
    rather than reading the file so the tests can drive it with mutated source --
    which is what makes "this pin cannot be weakened" a demonstration instead of a
    claim.
    """
    if node is None:
        return ["no unguarded `loss = ...` assembly found"]
    if not isinstance(node, ast.Assign):
        return ["the base objective must be a plain assignment, not %s" % type(node).__name__]
    terms = _add_chain(node.value)
    if len(terms) != len(BASE_TERMS):
        return ["expected exactly %d summed terms, found %d" % (len(BASE_TERMS), len(terms))]
    violations = []
    for term, expected in zip(terms, BASE_TERMS):
        if expected == WEIGHTED_TERM:
            # The only coefficient permitted anywhere in the assembly.  Either
            # operand order is accepted: the invariant is WHICH factor multiplies
            # the term, not how it was typed.
            if not (isinstance(term, ast.BinOp) and isinstance(term.op, ast.Mult)):
                violations.append("%s must be multiplied by self.%s" % (expected, WEIGHT_ATTR))
                continue
            operands = (term.left, term.right)
            names = [o.id for o in operands if isinstance(o, ast.Name)]
            weights = [
                o for o in operands
                if isinstance(o, ast.Attribute) and o.attr == WEIGHT_ATTR
                and isinstance(o.value, ast.Name) and o.value.id == "self"
            ]
            if names != [expected] or len(weights) != 1:
                violations.append(
                    "%s may be multiplied by self.%s and by nothing else"
                    % (expected, WEIGHT_ATTR)
                )
        elif not (isinstance(term, ast.Name) and term.id == expected):
            violations.append("term %r must stay bare, unweighted and in position" % expected)
    return violations


class DefaultIsOffTests(unittest.TestCase):
    def test_absent_kwarg_is_zero(self):
        self.assertEqual(_sampler().seam_loss_weight, 0.0)

    def test_falsy_values_all_resolve_to_zero(self):
        # oc.select yields 0.0, but a null in a stale config, or an empty override,
        # must not become a live weight through float(None) or float('').
        for value in (0.0, 0, None, ""):
            with self.subTest(value=value):
                self.assertEqual(_sampler(seam_loss_weight=value).seam_loss_weight, 0.0)

    def test_explicit_weight_is_kept_as_float(self):
        sampler = _sampler(seam_loss_weight=1.0)
        self.assertIsInstance(sampler.seam_loss_weight, float)
        self.assertEqual(sampler.seam_loss_weight, 1.0)


class InertWhenOffTests(unittest.TestCase):
    """The load-bearing pair: structural, then numeric."""

    def test_five_term_assembly_is_untouched(self):
        """The base objective is the released five terms, weighted only on jpos.

        If this assembly ever gains a term, loses one, reorders them, or grows a
        second coefficient, every sealed B v2 cell's objective changes and no other
        test in the suite would notice -- the difference is a small change in a loss
        scalar, not a crash.  This used to be a literal source-string pin; the
        2026-08-26 loss_w_jpos change rewrote that string, so the invariant is now
        enforced structurally, which is strictly stronger: it is insensitive to
        whitespace and line wrapping but sensitive to every semantic change above.
        """
        assembly, rebinds = _base_assembly(_method("p_losses"))
        self.assertIsNotNone(assembly, "the base `loss = ...` assembly is gone")
        self.assertEqual(_assembly_violations(assembly), [])
        # `loss += extra_term` in the function body would change every default run
        # while leaving the assembly above pristine, so the assembly must be the
        # ONLY rebinding of `loss` that is not inside a conditional guard.
        self.assertEqual(
            len(rebinds), 1,
            "the objective is modified outside a guard at lines %r"
            % [node.lineno for node in rebinds],
        )

    def test_the_released_assembly_is_what_the_mutation_controls_mutate(self):
        # Ties the synthetic controls in the next test to the real file: if the real
        # assembly is edited in any way that survives _assembly_violations (an
        # operand-order swap, say), this fails and the controls get revisited rather
        # than silently drifting out of correspondence with the source.
        assembly, _ = _base_assembly(_method("p_losses"))
        self.assertEqual(
            ast.dump(assembly.value),
            ast.dump(_parse_assign(RELEASED_ASSEMBLY).value),
        )

    def test_the_assembly_pin_rejects_every_way_of_weakening_it(self):
        # The pin is only worth its comment if it actually bites.  Each mutation is
        # a real way someone could change the objective; all of them must be caught.
        self.assertEqual(_assembly_violations(_parse_assign(RELEASED_ASSEMBLY)), [])
        mutations = {
            "sixth term appended": RELEASED_ASSEMBLY + " + loss_seam",
            "a term dropped":
                "loss = self.loss_w_jpos * loss_jpos + loss_jrot + loss_otrans + loss_contact",
            "loss_jpos left unweighted":
                "loss = loss_jpos + loss_jrot + loss_otrans + loss_orot + loss_contact",
            "a different coefficient on loss_jpos":
                "loss = 20.0 * loss_jpos + loss_jrot + loss_otrans + loss_orot + loss_contact",
            "the weight attribute renamed":
                "loss = self.loss_w * loss_jpos + loss_jrot + loss_otrans + loss_orot"
                " + loss_contact",
            "a coefficient added to another term":
                "loss = self.loss_w_jpos * loss_jpos + 2.0 * loss_jrot + loss_otrans"
                " + loss_orot + loss_contact",
            "the weight moved onto another term":
                "loss = loss_jpos + self.loss_w_jpos * loss_jrot + loss_otrans + loss_orot"
                " + loss_contact",
            "terms reordered":
                "loss = loss_jrot + self.loss_w_jpos * loss_jpos + loss_otrans + loss_orot"
                " + loss_contact",
            "a term renamed":
                "loss = self.loss_w_jpos * loss_jpos + loss_jrot + loss_otrans + loss_orot"
                " + loss_penetration",
            "a term replaced by a nested sum":
                "loss = self.loss_w_jpos * loss_jpos + (loss_jrot + loss_extra) + loss_otrans"
                " + loss_orot + loss_contact",
            "a term divided instead of summed":
                "loss = self.loss_w_jpos * loss_jpos + loss_jrot / 2 + loss_otrans + loss_orot"
                " + loss_contact",
        }
        for label, line in mutations.items():
            with self.subTest(label):
                self.assertNotEqual(
                    _assembly_violations(_parse_assign(line)), [],
                    "the pin failed to reject: %s" % label,
                )

    def test_every_seam_statement_is_inside_the_weight_guard(self):
        p_losses = _method("p_losses")
        guards = [
            node for node in ast.walk(p_losses)
            if isinstance(node, ast.If)
            and _uses_sampler_attribute(node.test, "seam_loss_weight")
        ]
        self.assertEqual(len(guards), 1, "expected exactly one seam weight guard")
        guard = guards[0]
        self.assertEqual(guard.orelse, [], "the guard must have no else branch")
        # A strict positivity test, so a 0.0 weight cannot enter the branch.
        self.assertIsInstance(guard.test, ast.Compare)
        self.assertEqual(len(guard.test.ops), 1)
        self.assertIsInstance(guard.test.ops[0], ast.Gt)
        self.assertEqual(guard.test.comparators[0].value, 0.0)

        guarded = {id(node) for node in ast.walk(guard)}
        offenders = []
        for stmt in p_losses.body:
            if stmt is guard or id(stmt) in guarded:
                continue
            if "loss_seam" not in ast.dump(stmt):
                continue
            is_none_assign = (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and getattr(stmt.targets[0], "id", None) == "loss_seam"
                and isinstance(stmt.value, ast.Constant)
                and stmt.value.value is None
            )
            if is_none_assign or isinstance(stmt, ast.Return):
                continue
            offenders.append(ast.dump(stmt)[:120])
        self.assertEqual(offenders, [], "seam statements outside the guard: %r" % offenders)

    def test_loss_seam_is_preassigned_none_before_the_guard(self):
        p_losses = _method("p_losses")
        none_line = guard_line = None
        for node in ast.walk(p_losses):
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and getattr(node.targets[0], "id", None) == "loss_seam"
                    and isinstance(node.value, ast.Constant) and node.value.value is None):
                none_line = node.lineno
            if (isinstance(node, ast.If)
                    and _uses_sampler_attribute(node.test, "seam_loss_weight")):
                guard_line = node.lineno
        self.assertIsNotNone(none_line, "loss_seam = None is missing")
        self.assertIsNotNone(guard_line)
        self.assertLess(none_line, guard_line)

    def test_return_always_carries_the_key(self):
        returns = [n for n in ast.walk(_method("p_losses")) if isinstance(n, ast.Return)]
        self.assertEqual(len(returns), 1, "p_losses must have exactly one return")
        self.assertIn("loss_seam", ast.dump(returns[0]))

    def test_assembled_loss_is_the_released_expression_when_off(self):
        x_start, predicted, mask = _batch()
        terms = _base_terms(x_start, predicted, mask)
        released = terms[0] + terms[1] + terms[2] + terms[3] + terms[4]

        weight = _sampler().seam_loss_weight
        self.assertEqual(weight, 0.0)
        with_switch = released
        if weight > 0.0:                                     # pragma: no cover
            with_switch = released + weight * _seam_term(x_start, predicted)

        self.assertTrue(torch.equal(with_switch, released))
        # Identity, not just equality: the guard means no new tensor is produced.
        # `released + 0.0 * seam` would be a different object and is not guaranteed
        # bitwise equal, which is exactly why this is a branch and not a multiply.
        self.assertIs(with_switch, released)

    def test_off_arm_differs_from_on_arm(self):
        # Guards against a vacuous inertness proof: if the term were a no-op even
        # when on, every assertion above would still pass.
        x_start, predicted, mask = _batch()
        terms = _base_terms(x_start, predicted, mask)
        released = terms[0] + terms[1] + terms[2] + terms[3] + terms[4]
        on = released + 1.0 * _seam_term(x_start, predicted)
        self.assertFalse(torch.equal(on, released))
        self.assertGreater(float(on - released), 0.0)


class TermIsExactlyItsSliceTests(unittest.TestCase):
    def test_value_matches_an_independent_reference(self):
        x_start, predicted, _ = _batch()
        n = AUTO_REGRE_NUM
        reference = ((x_start[:, n:n + 2, :84] - predicted[:, n:n + 2, :84]) ** 2).mean()
        self.assertTrue(torch.equal(_seam_term(x_start, predicted), reference))

    def test_only_the_first_two_generated_frames_matter(self):
        x_start, predicted, _ = _batch()
        n = AUTO_REGRE_NUM
        base = _seam_term(x_start, predicted)
        for frame, should_change in ((n, True), (n + 1, True), (n + 2, False),
                                     (WINDOW_FRAMES - 1, False)):
            with self.subTest(frame=frame):
                bumped = predicted.clone()
                bumped[:, frame, :84] += 1.0
                changed = not torch.equal(_seam_term(x_start, bumped), base)
                self.assertEqual(changed, should_change)

    def test_history_frames_are_not_in_the_term(self):
        # Frames 0..auto_regre_num-1 are clean GT that set_fixed_points overwrites at
        # every sampling step; supervising them would supervise an output nobody reads.
        x_start, predicted, _ = _batch()
        base = _seam_term(x_start, predicted)
        for frame in range(AUTO_REGRE_NUM):
            with self.subTest(frame=frame):
                bumped = predicted.clone()
                bumped[:, frame, :] += 1.0
                self.assertTrue(torch.equal(_seam_term(x_start, bumped), base))

    def test_only_the_position_channels_matter(self):
        x_start, predicted, _ = _batch()
        base = _seam_term(x_start, predicted)
        bumped = predicted.clone()
        bumped[:, :, POS_CHANNELS:] += 1.0
        self.assertTrue(torch.equal(_seam_term(x_start, bumped), base))

    def test_reduction_is_a_per_element_mean(self):
        # All five base terms are per-element means; a sum here would silently carry
        # a hidden factor of 168 into the weight's meaning.
        x_start, predicted, _ = _batch()
        n = AUTO_REGRE_NUM
        diff = x_start[:, n:n + 2, :84] - predicted[:, n:n + 2, :84]
        self.assertEqual(diff.numel(), 3 * SEAM_FRAMES * POS_CHANNELS)
        self.assertTrue(torch.equal(_seam_term(x_start, predicted), (diff ** 2).sum() / diff.numel()))


class WeightMeaningTests(unittest.TestCase):
    """The counting facts that give the frozen weight 1.0 its meaning.

    Pinned by a test so the number cannot be reinterpreted later from prose alone.
    """

    N_GEN = (WINDOW_FRAMES - AUTO_REGRE_NUM) * POS_CHANNELS      # 1176
    N_SEAM = SEAM_FRAMES * POS_CHANNELS                          # 168

    def test_denominators(self):
        self.assertEqual(self.N_GEN, 1176)
        self.assertEqual(self.N_SEAM, 168)
        self.assertEqual(self.N_GEN // self.N_SEAM, 7)

    def test_per_element_ratio_is_one_plus_seven_w(self):
        for w in (0.0, 0.5, 1.0, 2.0):
            with self.subTest(w=w):
                seam_coeff = 1.0 / self.N_GEN + w / self.N_SEAM
                interior_coeff = 1.0 / self.N_GEN
                self.assertAlmostEqual(seam_coeff / interior_coeff, 1.0 + 7.0 * w, places=12)

    def test_frozen_weight_gives_eight_x_and_leaves_the_interior_a_plurality(self):
        rho = 1.0 + 7.0 * 1.0
        self.assertEqual(rho, 8.0)
        interior_frames = WINDOW_FRAMES - AUTO_REGRE_NUM - SEAM_FRAMES     # 12
        seam_share = SEAM_FRAMES * rho / (SEAM_FRAMES * rho + interior_frames)
        self.assertAlmostEqual(seam_share, 16.0 / 28.0, places=12)
        self.assertAlmostEqual(1.0 - seam_share, 12.0 / 28.0, places=12)
        self.assertGreater(1.0 - seam_share, 0.40)          # the interior keeps a plurality

    def test_the_frozen_weight_is_the_one_in_the_configs(self):
        t1 = (REPO / "code" / "config" / "config_train_hsi_b_seam_t1.yaml").read_text()
        self.assertIn("seam_loss_weight: 1.0", t1)
        t0 = (REPO / "code" / "config" / "config_train_hsi_b_seam_t0.yaml").read_text()
        self.assertNotIn("seam_loss_weight:", t0)

    def test_the_two_fragments_differ_ONLY_in_name_and_weight(self):
        """The strongest available statement that the arms differ only in the loss.

        Comparing the two config fragments line by line beats a forbidden-key list:
        it catches a scientific quantity being changed in one arm and not the other
        even for a key nobody thought to forbid.
        """
        def body(arm):
            text = (REPO / "code" / "config" / ("config_train_hsi_b_seam_%s.yaml" % arm)).read_text()
            return [l.rstrip() for l in text.splitlines()
                    if l.strip() and not l.lstrip().startswith("#")]

        t0, t1 = body("t0"), body("t1")
        only_t0 = [l for l in t0 if l not in t1]
        only_t1 = [l for l in t1 if l not in t0]
        self.assertEqual(only_t0, ["exp_name: hsi_b_seam_t0"])
        self.assertEqual(sorted(only_t1),
                         sorted(["exp_name: hsi_b_seam_t1", "seam_loss_weight: 1.0"]))

    def test_neither_arm_restates_an_inherited_scientific_quantity(self):
        # These must come from config_train_hsi_b_lingo_full so the two arms cannot
        # drift apart, and so the pilot cannot silently change the trained geometry.
        forbidden = ("seed:", "batch_size:", "num_gpus:", "lr:", "warmup_updates:",
                     "loss_w_fk:", "auto_regre_num:", "precision:",
                     "effective_batch_size:", "gradient_accumulation_steps:")
        for arm in ("t0", "t1"):
            text = (REPO / "code" / "config" / ("config_train_hsi_b_seam_%s.yaml" % arm)).read_text()
            body = "\n".join(l for l in text.splitlines() if not l.lstrip().startswith("#"))
            for key in forbidden:
                with self.subTest(arm=arm, key=key):
                    self.assertNotIn(key, body)

    def test_the_frozen_budget_and_start_state(self):
        for arm, expected in (("t0", "hsi_b_seam_t0"), ("t1", "hsi_b_seam_t1")):
            text = (REPO / "code" / "config" / ("config_train_hsi_b_seam_%s.yaml" % arm)).read_text()
            self.assertIn("config_train_hsi_b_lingo_full", text)
            self.assertIn("exp_name: %s" % expected, text)
            # 223..242 inclusive; start_epoch=223 puts optimizer_updates at
            # 223*656 = 146,288, so the lr is already past its 2000-update warmup and
            # 146,288 + 20*656 = 159,408 makes the epoch and update bounds coincide.
            self.assertIn("start_epoch: 223", text)
            self.assertIn("epochs: 243", text)
            self.assertIn("max_optimizer_updates: 159408", text)
            self.assertEqual(223 * 656 + 20 * 656, 159408)
            # Weights-only continuation: the origin's rolling resume state is refused
            # by check_resume_compatibility because it was written at the update
            # budget with epoch_completed=False.  That guard is correct and untouched.
            self.assertIn("load_state_dict: true", text)
            self.assertIn("hsi_b_lingo_full_v2_epoch222.pth", text)
            self.assertIn('resume_from: ""', text)
            # Restart cadence, identical in both arms and trajectory-neutral: the save
            # block sits between epochs, consumes no RNG and touches no optimizer state.
            self.assertIn("ckpt_interval: 5", text)

    def test_the_origin_resume_guard_is_not_weakened(self):
        # The pilot must not have "fixed" the guard that refused the origin state.
        trainer = (REPO / "code" / "train_infbagel.py").read_text()
        self.assertIn("if not bool(state.get('epoch_completed', False)):", trainer)
        self.assertIn("the run is finished and there is nothing to resume", trainer)


class ModelCIsOutOfScopeTests(unittest.TestCase):
    def test_consistency_loss_has_no_seam_term(self):
        self.assertNotIn("seam", ast.dump(_method("consistency_loss")).lower())

    def test_sampler_config_default_is_off(self):
        text = (REPO / "code" / "config" / "sampler" / "pelvis.yaml").read_text()
        self.assertIn("seam_loss_weight: ${oc.select:seam_loss_weight,0.0}", text)


class FullBodySeamObjectiveTests(unittest.TestCase):
    def test_default_is_off_and_explicit_weight_is_kept(self):
        self.assertEqual(_sampler().fullbody_seam_loss_weight, 0.0)
        self.assertEqual(
            _sampler(fullbody_seam_loss_weight=1.25).fullbody_seam_loss_weight,
            1.25,
        )

    def test_two_second_differences_match_the_registered_formula(self):
        generator = torch.Generator().manual_seed(20260901)
        target = torch.randn((3, 6, 24, 3), generator=generator)
        predicted = torch.randn((3, 6, 24, 3), generator=generator)
        sampler = _sampler()
        n = AUTO_REGRE_NUM

        mixed = torch.cat([target[:, n - 2:n], predicted[:, n:n + 2]], dim=1)
        expected = F.mse_loss(
            mixed[:, 2:] - 2.0 * mixed[:, 1:-1] + mixed[:, :-2],
            target[:, n:n + 2] - 2.0 * target[:, n - 1:n + 1]
            + target[:, n - 2:n],
        )
        actual = sampler._compute_fullbody_seam_loss(predicted, target)
        self.assertTrue(torch.equal(actual, expected))

    def test_only_first_two_generated_predictions_have_gradients(self):
        generator = torch.Generator().manual_seed(42)
        target = torch.randn((2, 8, 24, 3), generator=generator)
        predicted = torch.randn((2, 8, 24, 3), generator=generator, requires_grad=True)
        loss = _sampler()._compute_fullbody_seam_loss(predicted, target)
        gradient = torch.autograd.grad(loss, predicted)[0]

        self.assertTrue(torch.equal(gradient[:, :AUTO_REGRE_NUM], torch.zeros_like(gradient[:, :AUTO_REGRE_NUM])))
        self.assertTrue(bool((gradient[:, AUTO_REGRE_NUM:AUTO_REGRE_NUM + 2] != 0).any()))
        self.assertTrue(torch.equal(
            gradient[:, AUTO_REGRE_NUM + 2:],
            torch.zeros_like(gradient[:, AUTO_REGRE_NUM + 2:]),
        ))

    def test_the_two_generated_frames_are_coupled(self):
        target = torch.zeros((1, 4, 24, 3))
        sampler = _sampler()
        n = AUTO_REGRE_NUM

        first = target.clone()
        first[:, n] = 1.0
        first_loss = sampler._compute_fullbody_seam_loss(first, target)
        first[:, n + 1] = 2.0
        coupled_loss = sampler._compute_fullbody_seam_loss(first, target)

        self.assertGreater(float(first_loss), 0.0)
        self.assertLess(float(coupled_loss), float(first_loss))

    def test_fk_path_sends_gradients_to_pelvis_translation_and_rotations(self):
        class RotationAwareDataset:
            def denormalize_torch(self, points, is_object=False):
                return points

            def quat_ik_torch(self, rotations):
                return rotations

            def quat_fk_torch(self, rotations, positions):
                joints = positions.clone()
                joints[:, :22] = joints[:, :22] + rotations[:, :, 0, :]
                return None, joints

        generator = torch.Generator().manual_seed(7)
        target = torch.randn((1, 4, CHANNELS), generator=generator)
        prediction = target.clone()
        prediction[:, AUTO_REGRE_NUM:AUTO_REGRE_NUM + 2, :216] += 0.2
        prediction.requires_grad_(True)
        sampler = _sampler()
        sampler.dataset = RotationAwareDataset()
        mat = torch.eye(4).unsqueeze(0)
        offsets = torch.zeros((1, 24, 3))

        _, predicted_joints = sampler._compute_human_joints(
            prediction, target[:, :, :84], mat, offsets
        )
        _, target_joints = sampler._compute_human_joints(
            target, target[:, :, :84], mat, offsets
        )
        loss = sampler._compute_fullbody_seam_loss(predicted_joints, target_joints)
        gradient = torch.autograd.grad(loss, prediction)[0]

        n = AUTO_REGRE_NUM
        self.assertTrue(bool((gradient[:, n:n + 2, :3] != 0).any()))
        self.assertTrue(bool((gradient[:, n:n + 2, 84:216] != 0).any()))
        self.assertTrue(torch.equal(gradient[:, :n], torch.zeros_like(gradient[:, :n])))
        self.assertTrue(torch.equal(gradient[:, n + 2:], torch.zeros_like(gradient[:, n + 2:])))

    def test_r2_config_inherits_r1_and_sets_one_new_objective_weight(self):
        text = (REPO / "code" / "config" / "config_train_hsi_b_r2_fullbody_seam.yaml").read_text()
        body = [
            line.strip() for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        self.assertEqual(
            body,
            [
                "defaults: [config_train_hsi_b_r1_reliable_teacher, _self_]",
                "exp_name: hsi_b_r2_fullbody_seam",
                "fullbody_seam_loss_weight: 0.5494500113254572",
            ],
        )
        sampler_config = (REPO / "code" / "config" / "sampler" / "pelvis.yaml").read_text()
        self.assertIn(
            "fullbody_seam_loss_weight: ${oc.select:fullbody_seam_loss_weight,0.0}",
            sampler_config,
        )


if __name__ == "__main__":                                       # pragma: no cover
    unittest.main()
