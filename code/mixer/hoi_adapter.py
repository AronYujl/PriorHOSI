"""Drive HOIPrior from the HOSI-test evaluator.

``code/test_infbagel_hosi.py`` was written against ``models.infbagel.Sampler``.
HOIPrior lives behind ``priors.hoi.diffusion.HOIPriorSampler``, and the two
sampler interfaces agree on the first 19 positional parameters of
``p_sample_loop`` and then diverge:

    released  ... obj_rot_mat_ref, obj_rest_verts, obj_vert_normals,
                  seq_name_dict, human_dict, guidance_fn, guidance_scale
    HOIPrior  ... obj_rot_mat_ref, obj_rest_verts, seq_name_dict,
                  obj_rot_mat_prefix=, object_only=, ground_truth_contact=

This adapter is the translation, and it is the whole of it: no evaluator logic
is duplicated here.  ``cm_sample_loop`` is absent on the HOI side -- HOIPrior
was never distilled -- so it raises rather than returning something plausible.

Two facts about the anchor this produces, both measured and neither obvious:

* HOIPrior is scene-blind, but the HOSI-test row is NOT a scene-blind row.
  ``code/astar.py:get_path`` reads the scene occupancy, dilates it and plans a
  collision-free path, and the evaluator feeds a point on that path in as
  ``pelvis_goal`` at every window.  So the G==0 anchor is a scene-blind MODEL
  under scene-aware WAYPOINT supervision.  Every row on this benchmark shares
  that, so it does not confound row-to-row comparison, but it does cap the
  headroom a scene-aware expert can demonstrate here.
* The evaluator's dataset must have ``load_scene=True`` -- it needs
  ``scene_dict`` for the scene flag, ``scene_occ`` for A*, and the scene SDF for
  the penetration metrics -- while ``HOIPriorSampler.set_dataset_and_model``
  refuses such a dataset outright.  ``SceneBlindDatasetView`` resolves that
  without weakening the guarantee: the inner sampler is handed a view whose
  scene payload is not merely unused but unreachable.
"""

from typing import Optional

from priors.hoi.diffusion import HOIPriorSampler


# The dataset's scene payload and its query surface.  `scene_name` is NOT here:
# on OMOMO that attribute is the per-sequence name list (`sub16_clothesstand_...`),
# which the preregistered P2 guidance needs to recover each sequence's SMPL rest
# offsets.  `load_scene`, `scene_folder` and `test_scene_name` are likewise
# metadata, not payload.
BLOCKED_SCENE_ATTRIBUTES = frozenset({
    'scene_occ',
    'scene_occ_ref',
    'scene_dict',
    'scene_grid_np',
    'scene_grid_torch',
    'scene_name2file',
    'get_occ_for_points',
    'get_pene_occ_count',
    'compute_occ_ref',
    'set_test_scene',
})


class SceneBlindDatasetView:
    """A read-through view of a scene-loaded dataset with the scene removed.

    Every attribute forwards to the wrapped dataset unchanged -- crucially the
    normalization tensors, so HOIPrior normalizes exactly as it did in its own
    evaluation -- except that ``load_scene`` reports False and the scene payload
    raises ``AttributeError``.  The point is to make the scene-blindness of the
    HOI expert a property the process enforces rather than one a comment
    asserts: if a future change routes scene occupancy into HOIPrior, this view
    raises at the first access instead of silently conditioning on it.
    """

    def __init__(self, dataset):
        # Bypass __setattr__/__getattr__ for the one real attribute we own.
        object.__setattr__(self, '_dataset', dataset)

    @property
    def dataset(self):
        """The wrapped, scene-loaded dataset (for callers that need it back)."""
        return object.__getattribute__(self, '_dataset')

    @property
    def load_scene(self):
        return False

    def __getattr__(self, name):
        if name in BLOCKED_SCENE_ATTRIBUTES:
            raise AttributeError(
                f'{name!r} is scene state and is not reachable from HOIPrior: '
                'the HOI expert is scene-blind by construction'
            )
        return getattr(object.__getattribute__(self, '_dataset'), name)

    def __setattr__(self, name, value):
        raise AttributeError('SceneBlindDatasetView is read-only')

    def __repr__(self):
        return f'SceneBlindDatasetView({object.__getattribute__(self, "_dataset")!r})'


class HOIExpertSamplerAdapter:
    """``models.infbagel.Sampler``'s sampling interface, backed by HOIPrior."""

    def __init__(self, device, auto_regre_num=2, timesteps=500,
                 object_so3_x0=False, guidance=None, **_):
        self.inner = HOIPriorSampler(
            device=device,
            auto_regre_num=auto_regre_num,
            timesteps=timesteps,
            object_so3_x0=object_so3_x0,
            guidance=guidance,
        )
        self.dataset = None
        self.student_model = None

    # -- evaluator setup -------------------------------------------------

    def set_dataset_and_model(self, dataset, model):
        """Keep the scene-loaded dataset; hand HOIPrior the scene-blind view.

        The evaluator reads ``sampler.dataset`` for denormalization, the scene
        flag and A*, so the adapter's ``dataset`` is the real one.  HOIPrior
        only ever sees the view.
        """
        self.dataset = dataset
        self.inner.set_dataset_and_model(SceneBlindDatasetView(dataset), model)
        self.student_model = model

    # -- sampling --------------------------------------------------------

    def p_sample_loop(self, fixed_points, mat, scene_flag, text_emb, pelvis_goal,
                      scene_goal, object_goal, need_scene, need_pelvis_dir, pi,
                      end_pi, seq_length, need_pi, is_loco, is_object,
                      obj_bps_data, object_points, obj_rot_mat_ref,
                      obj_rest_verts, obj_vert_normals=None, seq_name_dict=None,
                      human_dict=None, guidance_fn=None, guidance_scale=None,
                      object_only=False, obj_rot_mat_prefix=None,
                      ground_truth_contact=None, state=None):
        """The released signature; forwards the 19 shared arguments verbatim.

        ``obj_vert_normals``, ``human_dict``, ``guidance_fn`` and
        ``guidance_scale`` are the released path's guidance plumbing.  HOIPrior
        carries its own preregistered guidance, configured on the sampler rather
        than passed per call, so a caller that supplies a ``guidance_fn`` here is
        asking for something this adapter will not do, and is told so.

        ``state`` is RESERVED for the LLM state machine, exactly as in
        ``mixer.composition.compose_x0``; supplying it raises.
        """
        if state is not None:
            raise NotImplementedError(
                'p_sample_loop accepts `state` only as a reserved parameter; '
                'the LLM state machine is not implemented'
            )
        if guidance_fn is not None:
            raise ValueError(
                'HOIPrior does not take a per-call guidance_fn: its guidance is '
                'the preregistered Phase 1B P2 term, configured on the sampler '
                '(sampler.pelvis.guidance.*). Pass use_guidance=false so the '
                'evaluator selects no guidance_fn.'
            )
        return self.inner.p_sample_loop(
            fixed_points, mat, scene_flag, text_emb, pelvis_goal, scene_goal,
            object_goal, need_scene, need_pelvis_dir, pi, end_pi, seq_length,
            need_pi, is_loco, is_object, obj_bps_data, object_points,
            obj_rot_mat_ref, obj_rest_verts, seq_name_dict,
            obj_rot_mat_prefix=obj_rot_mat_prefix,
            object_only=object_only,
            ground_truth_contact=ground_truth_contact,
        )

    def cm_sample_loop(self, *_, **__):
        raise NotImplementedError(
            'HOIPrior has no consistency-model sampler: it was never distilled. '
            'Set sample_type=diffusion for the HOI expert. Note this makes the '
            'HOI-alone row 500 network calls per window against the released '
            "baseline's 16, so the two rows' latency is not comparable."
        )

    # -- provenance ------------------------------------------------------

    def reset_sampling_audit(self):
        self.inner.reset_sampling_audit()

    def audit_dict(self):
        return self.inner.audit_dict()

    def __getattr__(self, name):
        # Anything the evaluator reaches for that the adapter does not define
        # (audit counters, guidance settings) comes from the inner sampler.
        # Only reached for names absent from the instance dict, so `dataset`
        # and `student_model` never land here.
        if name in {'inner'}:
            raise AttributeError(name)
        return getattr(object.__getattribute__(self, 'inner'), name)
