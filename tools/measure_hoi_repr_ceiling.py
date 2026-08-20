#!/usr/bin/env python3
"""Measure the HOI representation-ceiling and ground-truth-floor reference rows.

The 438-sequence OMOMO protocol scores a 126-frame motion that the network never
predicts directly: it predicts 42 keyframes on a stride-3 / 10 Hz grid, and
``code/test_infbagel_hoi.py:161-179`` reconstructs the scored 126 frames from
them.  Every native HOI number therefore contains a representation tax that no
model can avoid.  This tool measures that tax by pushing *ground truth* through
the same reconstruction and scoring the result with the evaluator's own metric
functions, producing two permanent reference rows:

ROW-GT (floor)
    native 30 Hz ground truth, bit-for-bit the evaluator's ``points_fk_all_gt_48``
    plus the native ground-truth object pose.

ROW-CEILING
    the same ground truth decimated to the stride-3 keyframe grid the network
    predicts, then reconstructed exactly as the model path does: linear
    ``interpolate_joints(scale=3)`` on the root, ``quat_ik_torch`` ->
    ``matrix_to_quaternion`` -> ``interp_jrot`` slerp -> ``quaternion_to_matrix``
    on the 22 local rotations, ``quat_fk_torch`` against the rigid per-subject
    ``rest_human_offsets`` template, and ``interp_object`` for the object.

It is read-only and CPU-only: no checkpoint, no model, no GPU, no write into any
existing run directory.  The two differences from the exploration it promotes
(``.claude/scratch/repr-ceiling/``, 2026-08-20) are that it imports
``InfBaGelDataset``'s bound ``quat_ik_torch``/``quat_fk_torch`` instead of copying
their bodies, and that it additionally computes the four penetration terms for
both rows -- the stage the exploration never did.

Preregistered in ``docs/plan/PHASE_1B_HOI/07_REPRESENTATION_FRAME.md``, section
"2026-08-21".  The construction is fixed there and must not be changed after a
result is seen.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import chois_evaluator


class CeilingError(RuntimeError):
    """Raised for an unreproducible or contract-violating probe invocation."""


# This probe deliberately declares its own execution envelope so
# tests/test_research_governance.py can assert it mechanically instead of
# trusting the docstring.
EXECUTION_CONTRACT = {
    "device": "cpu",
    "requires_gpu": False,
    "requires_checkpoint": False,
    "requires_model_inference": False,
    "requires_training": False,
    "writes_inside_run_directory": False,
    "read_only_inputs": True,
}

# Fixed by the preregistration.  interp_s=3, max_window_size=16 and
# auto_regre_num=2 give 42 scored frames per window and 14 keyframes per window.
PROTOCOL = {
    "sequences": 438,
    "windows_per_sequence": 3,
    "max_window_size": 16,
    "auto_regre_num": 2,
    "interp_s": 3,
    "scored_frames_per_window": 42,
    "keyframes_per_window": 14,
}

# code/test_infbagel_hoi.py:276 hard-codes this exclusion.  Copied, never fixed:
# fixing it would make the penetration columns incomparable with every existing
# HOI row.
PENETRATION_EXCLUDED_OBJECTS = (
    "woodchair", "whitechair", "largebox", "largetable", "plasticbox", "trashcan",
)

# Correctness gates A1-A7 of the preregistration.  A1-A3 are sealed anchors that
# must be reproduced inside this run; A4-A5 are counts; A6-A7 are structural.
GATES = {
    "A1_gt_foot_sliding": {"expected": 0.26346464114890705, "tolerance": 1e-07},
    "A2_gt_feet_height": {"expected": 0.034326739609241486, "tolerance": 1e-08},
    "A3_gt_contact_percent": {"expected": 0.6618830180474017, "tolerance": 0.0},
    "A4_zero_gt_contact_sequences": {"expected": 41},
    "A5_penetration_covered_sequences": {"expected": 181},
}
# 41 of 438 sequences carry no ground-truth contact frame at all, and
# code/eval_metrics.py:309-323 returns precision=recall=f1=0 for them even under
# a ground-truth self-comparison.  397/438 is therefore a hard analytic cap on
# those three columns for ANY model.
ANALYTIC_CONTACT_CAP = 397 / 438

# The 18 native metrics, split by what this row can say about each.
INFORMATIVE_METRICS = (
    "foot_sliding", "feet_height",
    "mpjpe", "trans_dist", "obj_trans_dist", "obj_rot_dist",
    "contact_precision", "contact_recall", "contact_f1", "contact_acc",
    "contact_percent",
    "hand_pen_loss_omomo", "hand_pen_ratio",
    "human_pen_loss_infbagel", "human_pen_ratio",
)
ANALYTICALLY_FIXED_METRICS = {
    "end_obj_trans_err": "0: reads the pre-interpolation keyframe channel",
    "xy_points_err": "0: reads the pre-interpolation keyframe channel",
    "gt_contact_percent": "invariant: determined by the ground-truth channel only",
}

# The 2026-08-20 read-only exploration this row promotes, at the precision it
# recorded.  A disagreement beyond float32 noise means one of the two
# measurements is wrong, which is stop classification
# "repr-ceiling-contradicts-exploration": both are kept and neither is sealed.
# Only meaningful at the full 438 sequences.
EXPLORATION_438 = {
    "ground_truth_floor": {
        "foot_sliding": 0.2634646156648616,
        "feet_height": 0.03432674026414412,
    },
    "representation_ceiling": {
        "foot_sliding": 0.30888668343082337,
        "feet_height": 0.03426290431047139,
        "mpjpe": 0.15603576321154833,
        "trans_dist": 0.09827817557379603,
        "obj_trans_dist": 0.11386023834347725,
        "obj_rot_dist": 0.0055702434487986605,
        "contact_precision": 0.9035295547070061,
        "contact_recall": 0.9022804362750585,
        "contact_f1": 0.9027963392207574,
        "contact_acc": 0.9968833804450242,
        "contact_percent": 0.6607595854171195,
        "gt_contact_percent": 0.6618830180474017,
    },
}
EXPLORATION_RELATIVE_TOLERANCE = 1e-05
EXPLORATION_ABSOLUTE_FLOOR = 1e-06

# The sealed P12 model row, used only by gate A7 as an order-of-magnitude guard
# on the ground-truth penetration floor: an SDF query that leaves the object's
# box collapses the penetration terms toward zero without crashing.
P12_PENETRATION_REFERENCE = {
    "hand_pen_loss_omomo": 0.17201,
    "hand_pen_ratio": 0.13282,
    "human_pen_loss_infbagel": 2.72824,
    "human_pen_ratio": 0.13654,
}
A7_RATIO_BOUNDS = (0.01, 100.0)

# code/test_infbagel_hoi.py:287,364 scales the full-body term and leaves the hand
# term unscaled.  Gate A6 pins that the recorded values carry this convention.
HUMAN_PEN_SCALE = 10475 / 100

STOP_CLASSIFICATIONS = (
    "repr-ceiling-row-established",
    "repr-ceiling-penetration-partial",
    "repr-ceiling-anchor-fail-stop",
    "repr-ceiling-contradicts-exploration",
    "repr-ceiling-subset-smoke",
)


@contextlib.contextmanager
def project_code_context() -> Iterator[Any]:
    """Import the project's own evaluation code the way the evaluator sees it.

    ``code/constants.py`` sets ``ROOT_DIR = '..'``, so every asset path the
    evaluator builds is relative to the ``code`` directory; the probe therefore
    runs with that as its working directory, which is also what the 2026-08-20
    exploration did when it reproduced the sealed anchors.  ``code`` is not an
    importable package name (it shadows the standard library module), so its
    directory goes on ``sys.path`` instead.
    """
    code_dir = ROOT / "code"
    if not code_dir.is_dir():
        raise CeilingError(f"missing project code directory: {code_dir}")
    previous_cwd = Path.cwd()
    inserted = str(code_dir) not in sys.path
    if inserted:
        sys.path.insert(0, str(code_dir))
    os.environ.setdefault("ROOT_DIR", str(ROOT))
    os.chdir(code_dir)
    try:
        yield code_dir
    finally:
        os.chdir(previous_cwd)
        if inserted:
            with contextlib.suppress(ValueError):
                sys.path.remove(str(code_dir))


def build_dataset(config_name: str) -> Any:
    """Instantiate the OMOMO test dataset on CPU under its own Hydra config."""
    from hydra import compose, initialize_config_dir

    with initialize_config_dir(config_dir=str(ROOT / "code" / "config"), version_base=None):
        cfg = compose(
            config_name=config_name,
            overrides=["device=cpu", "exp_name=hoi_repr_ceiling_probe"],
        )
    for key, expected in (
        ("max_window_size", PROTOCOL["max_window_size"]),
        ("auto_regre_num", PROTOCOL["auto_regre_num"]),
        ("interp_s", PROTOCOL["interp_s"]),
    ):
        actual = int(cfg[key])
        if actual != expected:
            raise CeilingError(f"config {key}={actual} contradicts the preregistered {expected}")
    from datasets.infbagel import InfBaGelDataset

    return InfBaGelDataset(**cfg.dataset), cfg


def load_sequence_offsets(limit: Optional[int]) -> List[int]:
    """Return the dataset index of each protocol sequence's first window."""
    import pickle

    with open(ROOT / "data" / "test" / "seq_id.pkl", "rb") as handle:
        offsets = [0] + list(pickle.load(handle).values())
    available = len(offsets) - 1
    if available != PROTOCOL["sequences"]:
        raise CeilingError(f"expected {PROTOCOL['sequences']} protocol sequences, found {available}")
    if limit is None:
        return offsets[:-1]
    if limit <= 0 or limit > available:
        raise CeilingError(f"invalid --sequences: {limit}")
    return offsets[:limit]


def load_rest_object_vertices() -> Dict[str, Any]:
    """Load the rest object geometry the evaluator loads: the .ply mesh vertices.

    ``code/test_infbagel_hoi.py:505-517`` loads ``rest_object_geo/*.ply`` through
    ``trimesh`` and converts it with ``zup_to_yup``.  The 1024-point
    ``rest_object_geo/*.npy`` is the BPS conditioning input, not the contact
    geometry; substituting it moves ``gt_contact_percent`` off its sealed value,
    which is how the exploration caught its own first attempt.
    """
    import numpy as np
    import torch
    import trimesh
    from utils import zup_to_yup

    root = ROOT / "data" / "object" / "rest_object_geo"
    vertices: Dict[str, Any] = {}
    for path in sorted(root.glob("*.ply")):
        mesh = trimesh.load_mesh(str(path))
        vertices[path.stem] = torch.from_numpy(
            zup_to_yup(np.asarray(mesh.vertices))
        ).float()
    if not vertices:
        raise CeilingError(f"no rest object meshes under {root}")
    return vertices


class SignedDistanceFields:
    """Lazily hold the 256^3 rest-object SDF volumes plus their box metadata.

    The evaluator loads all thirteen eagerly; each is 67 MB, and the six
    penetration-excluded classes are never queried, so this loads on first use.
    The values queried are identical either way.
    """

    def __init__(self) -> None:
        self.root = ROOT / "data" / "object" / "rest_object_sdf_256_npy_files"
        self._volumes: Dict[str, Any] = {}
        self._boxes: Dict[str, Any] = {}

    def get(self, object_name: str) -> Tuple[Any, Any]:
        if object_name not in self._volumes:
            import numpy as np

            matches = sorted(self.root.glob(f"{object_name}.*.npy"))
            if len(matches) != 1:
                raise CeilingError(f"expected exactly one SDF volume for {object_name}, found {len(matches)}")
            volume = matches[0]
            metadata = volume.with_suffix(".json")
            if not metadata.is_file():
                raise CeilingError(f"missing SDF box metadata: {metadata}")
            self._volumes[object_name] = np.load(volume)
            self._boxes[object_name] = json.loads(metadata.read_text(encoding="utf-8"))
        return self._volumes[object_name], self._boxes[object_name]


def load_hand_vertex_indices() -> Any:
    """The 1556 MANO hand vertices the OMOMO hand-penetration term scores."""
    import numpy as np
    import pickle

    path = ROOT / "smpl_models" / "MANO_SMPLX_vertex_ids.pkl"
    if not path.is_file():
        raise CeilingError(f"missing required kinematic asset: {path}")
    with path.open("rb") as handle:
        indices = pickle.load(handle)
    return np.concatenate([indices["left_hand"], indices["right_hand"]])


def reconstruct_sequence(dataset: Any, windows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Build the ground-truth floor and representation-ceiling motions for one sequence.

    The ceiling path is function-for-function the model path of
    ``code/test_infbagel_hoi.py:161-179``, and the interpolation deliberately runs
    across the three windows concatenated, exactly as the evaluator does.
    """
    import torch
    import pytorch3d.transforms as transforms
    from utils import interp_jrot, interp_object, interpolate_joints

    frames = PROTOCOL["max_window_size"] * PROTOCOL["interp_s"]
    scored = PROTOCOL["scored_frames_per_window"]
    skip = frames - scored

    gt_joints, gt_pose, gt_root = [], [], []
    gt_obj_trans, gt_obj_rot = [], []
    keyframe_global, keyframe_root = [], []
    keyframe_obj_trans, keyframe_obj_rot = [], []

    for window in windows:
        mat = torch.as_tensor(window["mat"]).reshape(4, 4)
        rest = torch.as_tensor(window["rest_human_offsets"]).reshape(24, 3)
        joints_gt = torch.as_tensor(window["joints_gt"]).reshape(frames, 28, 3)
        global_rotation = mat[None, :3, :3] @ transforms.rotation_6d_to_matrix(
            torch.as_tensor(window["global_rot_6d_gt"]).reshape(frames, 22, 6)
        )
        local_rotation = dataset.quat_ik_torch(global_rotation)
        local_positions = rest[None].repeat(frames, 1, 1).clone()
        local_positions[:, 0, :] = joints_gt[:, 0, :]
        gt_joints.append(dataset.quat_fk_torch(local_rotation, local_positions)[1][skip:])
        gt_pose.append(transforms.matrix_to_axis_angle(local_rotation[skip:]))
        gt_root.append(joints_gt[skip:, 0, :])

        stride = PROTOCOL["interp_s"]
        offset = PROTOCOL["auto_regre_num"]
        keyframe_global.append(global_rotation[::stride][offset:])
        keyframe_root.append(joints_gt[::stride][offset:, 0, :])

        reference = torch.as_tensor(window["obj_rot_mat_ref"]).reshape(3, 3)
        translation = dataset.denormalize_torch(
            torch.as_tensor(window["object_trans"]).reshape(-1, 3), is_object=True
        )
        translation = (mat[:3, :3] @ translation.T).T + mat[:3, 3]
        rotation = torch.as_tensor(window["object_rot_mat"]).reshape(-1, 3, 3) @ reference
        keyframe_obj_trans.append(translation[offset:])
        keyframe_obj_rot.append(rotation[offset:].reshape(-1, 9))
        gt_obj_trans.append(torch.as_tensor(window["object_trans_gt"]).reshape(frames, 3)[skip:])
        gt_obj_rot.append(torch.as_tensor(window["object_rot_mat_gt"]).reshape(frames, 9)[skip:])

    total = scored * len(windows)
    keyframe_global = torch.cat(keyframe_global, 0)
    root = interpolate_joints(torch.cat(keyframe_root, 0), scale=PROTOCOL["interp_s"])
    local_rotation = transforms.quaternion_to_matrix(
        interp_jrot(
            transforms.matrix_to_quaternion(dataset.quat_ik_torch(keyframe_global)),
            PROTOCOL["interp_s"],
        )
    )
    local_positions = torch.as_tensor(windows[0]["rest_human_offsets"]).reshape(1, 24, 3)
    local_positions = local_positions.repeat(total, 1, 1).clone()
    local_positions[:, 0, :] = root
    ceiling_joints = dataset.quat_fk_torch(local_rotation, local_positions)[1]
    obj_trans, obj_rot = interp_object(
        torch.cat(keyframe_obj_trans, 0).numpy(),
        torch.cat(keyframe_obj_rot, 0).numpy(),
        PROTOCOL["interp_s"],
    )

    transl = torch.as_tensor(windows[0]["transl"]).reshape(1, 3)
    name = str(windows[0]["seq_name"])
    return {
        "sequence_name": name,
        "object_name": name.split("_")[1],
        "betas": torch.as_tensor(windows[0]["betas"]).reshape(16),
        "gender": windows[0]["gender"],
        "ground_truth_floor": {
            "joints": torch.cat(gt_joints, 0),
            "pose": torch.cat(gt_pose, 0),
            "root_trans": torch.cat(gt_root, 0) + transl,
            "obj_trans": torch.cat(gt_obj_trans, 0).float(),
            "obj_rot": torch.cat(gt_obj_rot, 0).float(),
        },
        "representation_ceiling": {
            "joints": ceiling_joints,
            "pose": transforms.matrix_to_axis_angle(local_rotation),
            "root_trans": root + transl,
            "obj_trans": torch.from_numpy(obj_trans).float(),
            "obj_rot": torch.from_numpy(obj_rot).float(),
        },
    }


def score_foot_metrics(joints: Any) -> Tuple[float, float]:
    """Floor height and foot sliding, with the evaluator's own functions.

    ``compute_foot_sliding_for_smpl`` subtracts the floor height from its input in
    place.  On the CUDA production path ``tensor.cpu().numpy()`` allocates a fresh
    host buffer, so that mutation never reaches the joints the contact and MPJPE
    terms consume afterwards.  On CPU ``.cpu()`` returns the same tensor and the
    mutation would leak, silently scoring contact against floor-shifted joints, so
    each call gets its own copy.
    """
    import numpy as np
    from eval_metrics import compute_foot_sliding_for_smpl, determine_floor_height_and_contacts

    array = joints.detach().cpu().numpy().astype(np.float32).reshape(-1, 24, 3)
    floor_height = determine_floor_height_and_contacts(array.copy())
    sliding = compute_foot_sliding_for_smpl(array.copy(), floor_height)
    return float(floor_height), float(sliding)


def score_contact(row: Dict[str, Any], reference: Dict[str, Any], rest_vertices: Any) -> Dict[str, float]:
    """The five contact terms plus the invariant ``gt_contact_percent``.

    The object vertices come from each side's own object pose: the ceiling row is
    scored against its round-tripped object, the ground-truth floor against the
    native one, because that is how ``compute_hand_object_interaction`` receives
    ``obj_rest_verts_pred_seg`` and ``obj_rest_verts_gt_seg``.
    """
    from eval_metrics import compute_hand_object_interaction
    from utils import load_object_geometry_w_rest_geo

    predicted = load_object_geometry_w_rest_geo(
        row["obj_rot"].reshape(-1, 3, 3), row["obj_trans"].reshape(-1, 3), rest_vertices
    )
    truth = load_object_geometry_w_rest_geo(
        reference["obj_rot"].reshape(-1, 3, 3), reference["obj_trans"].reshape(-1, 3), rest_vertices
    )
    values = compute_hand_object_interaction(
        row["joints"].reshape(-1, 24, 3), reference["joints"].reshape(-1, 24, 3), predicted, truth
    )
    keys = (
        "gt_contact_percent", "contact_percent", "contact_acc",
        "contact_precision", "contact_recall", "contact_f1",
    )
    return {key: float(value) for key, value in zip(keys, values)}


def score_penetration(
    row: Dict[str, Any],
    betas: Any,
    gender: str,
    object_name: str,
    fields: SignedDistanceFields,
    hand_indices: Any,
    smplx_cache: Dict[str, Any],
) -> Dict[str, float]:
    """The four penetration terms, on the SMPL-X surface this row's pose produces.

    Both the vertices and the object pose entering ``compute_collision`` come from
    the same side, because the function derives its SDF frame from the object pose
    it is handed (``code/test_infbagel_hoi.py:277,281``).  Mixing a round-tripped
    body with a native object would measure neither a floor nor a ceiling.  The
    ``yup_to_zup`` conversions are the genuine frame change the released code
    needs here and are kept; the compensating sandwich deleted in 2026-08-19 was a
    different call site.
    """
    import torch
    from eval_metrics import compute_collision
    from utils import create_smplx_model, run_smplx_model, yup_to_zup, yup_to_zup_rotation_matrix

    if gender not in smplx_cache:
        smplx_cache[gender] = create_smplx_model(gender, "cpu", batch_size=1)
    frames = row["root_trans"].shape[0]
    with torch.no_grad():
        vertices, _ = run_smplx_model(
            row["pose"].reshape(frames, 22, 3),
            row["root_trans"].reshape(frames, 3),
            betas[None].repeat(frames, 1),
            gender,
            joints_ind=None,
            smpl_model=smplx_cache[gender],
        )
    vertices = vertices.reshape(frames, -1, 3)
    volume, box = fields.get(object_name)
    rotation = yup_to_zup_rotation_matrix(row["obj_rot"].reshape(frames, 3, 3))
    translation = yup_to_zup(row["obj_trans"].reshape(frames, 3))
    hand_loss, hand_ratio = compute_collision(
        yup_to_zup(vertices[:, hand_indices, :]), volume, box, rotation, translation
    )
    body_loss, body_ratio = compute_collision(
        yup_to_zup(vertices), volume, box, rotation, translation
    )
    return {
        "hand_pen_loss_omomo": float(hand_loss),
        "hand_pen_ratio": float(hand_ratio),
        "human_pen_loss_raw": float(body_loss),
        "human_pen_loss_infbagel": float(body_loss) * HUMAN_PEN_SCALE,
        "human_pen_ratio": float(body_ratio),
    }


ROW_NAMES = ("ground_truth_floor", "representation_ceiling")


def measure(
    dataset: Any,
    offsets: Sequence[int],
    rest_vertices: Dict[str, Any],
    penetration: bool,
    progress_every: int,
) -> Dict[str, Any]:
    """Score both rows over the requested protocol sequences."""
    import numpy as np
    import torch
    from eval_metrics import compute_gt_difference

    fields = SignedDistanceFields()
    hand_indices = load_hand_vertex_indices() if penetration else None
    smplx_cache: Dict[str, Any] = {}
    per_sequence: Dict[str, Dict[str, Any]] = {}
    accumulated: Dict[str, Dict[str, List[Any]]] = {
        name: {"joints": [], "obj_trans": [], "obj_rot": []} for name in ROW_NAMES
    }
    scalars: Dict[str, Dict[str, List[float]]] = {name: {} for name in ROW_NAMES}
    covered: List[str] = []
    zero_contact: List[str] = []
    started = time.perf_counter()

    for ordinal, offset in enumerate(offsets):
        windows = [dataset[offset + step] for step in range(PROTOCOL["windows_per_sequence"])]
        built = reconstruct_sequence(dataset, windows)
        object_name = built["object_name"]
        excluded = object_name in PENETRATION_EXCLUDED_OBJECTS
        if not excluded:
            covered.append(built["sequence_name"])
        record: Dict[str, Any] = {"object_name": object_name, "penetration_covered": not excluded}
        for name in ROW_NAMES:
            row = built[name]
            reference = built["ground_truth_floor"]
            floor_height, sliding = score_foot_metrics(row["joints"])
            values: Dict[str, Any] = {"feet_height": floor_height, "foot_sliding": sliding}
            values.update(score_contact(row, reference, rest_vertices[object_name]))
            metrics = compute_gt_difference(
                row["joints"].reshape(1, -1, 72), reference["joints"].reshape(1, -1, 72),
                row["obj_trans"].reshape(1, -1, 3), reference["obj_trans"].reshape(1, -1, 3),
                row["obj_rot"].reshape(1, -1, 9), reference["obj_rot"].reshape(1, -1, 9),
            )
            for key, value in zip(("mpjpe", "trans_dist", "obj_trans_dist", "obj_rot_dist"), metrics):
                values[key] = float(value)
            if penetration and not excluded:
                values.update(score_penetration(
                    row, built["betas"], built["gender"], object_name,
                    fields, hand_indices, smplx_cache,
                ))
            record[name] = values
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    scalars[name].setdefault(key, []).append(float(value))
            span = row["joints"].shape[0]
            for key in ("joints", "obj_trans", "obj_rot"):
                accumulated[name][key].append(row[key].detach().reshape(span, -1))
        if record["ground_truth_floor"]["gt_contact_percent"] == 0.0:
            zero_contact.append(built["sequence_name"])
        per_sequence[built["sequence_name"]] = record
        if progress_every and (ordinal + 1) % progress_every == 0:
            print(f"{ordinal + 1}/{len(offsets)} {time.perf_counter() - started:.1f}s", flush=True)

    rows: Dict[str, Dict[str, Any]] = {}
    truth = accumulated["ground_truth_floor"]
    for name in ROW_NAMES:
        current = accumulated[name]
        frames = torch.cat(current["joints"], 0).shape[0]
        aggregate = compute_gt_difference(
            torch.cat(current["joints"], 0).reshape(1, frames, 72),
            torch.cat(truth["joints"], 0).reshape(1, frames, 72),
            torch.cat(current["obj_trans"], 0).reshape(1, frames, 3),
            torch.cat(truth["obj_trans"], 0).reshape(1, frames, 3),
            torch.cat(current["obj_rot"], 0).reshape(1, frames, 9),
            torch.cat(truth["obj_rot"], 0).reshape(1, frames, 9),
        )
        values = {key: float(np.mean(series)) for key, series in scalars[name].items()}
        for key, value in zip(("mpjpe", "trans_dist", "obj_trans_dist", "obj_rot_dist"), aggregate):
            values[key] = float(value)
        values["end_obj_trans_err"] = 0.0
        values["xy_points_err"] = 0.0
        rows[name] = values
    return {
        "rows": rows,
        "per_sequence": per_sequence,
        "zero_gt_contact_sequences": zero_contact,
        "penetration_covered_sequences": covered,
        "elapsed_s": round(time.perf_counter() - started, 1),
    }


def restricted_contact_means(per_sequence: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Contact means over only the sequences where the metric is defined.

    The three capped columns are the degenerate-sequence fallback averaged in with
    real scores; on the defined subset the round trip costs about 0.004, which is
    the number to quote when a true ceiling is wanted.
    """
    import numpy as np

    defined = [
        record for record in per_sequence.values()
        if record["ground_truth_floor"]["gt_contact_percent"] > 0.0
    ]
    keys = ("contact_precision", "contact_recall", "contact_f1")
    result: Dict[str, Any] = {"sequences": len(defined)}
    for name in ROW_NAMES:
        result[name] = {
            key: float(np.mean([record[name][key] for record in defined])) for key in keys
        } if defined else {key: None for key in keys}
    return result


def evaluate_gates(measured: Dict[str, Any], penetration: bool, full_protocol: bool) -> List[Dict[str, Any]]:
    """Run correctness gates A1-A7 inside this run, never by citing a past pass."""
    truth = measured["rows"]["ground_truth_floor"]
    ceiling = measured["rows"]["representation_ceiling"]
    gates: List[Dict[str, Any]] = []

    def anchor(gate_id: str, value: float) -> None:
        spec = GATES[gate_id]
        delta = abs(value - spec["expected"])
        gates.append({
            "gate": gate_id, "expected": spec["expected"], "measured": value,
            "delta": delta, "tolerance": spec["tolerance"],
            "status": "skipped-subset" if not full_protocol
            else ("pass" if delta <= spec["tolerance"] else "fail"),
        })

    anchor("A1_gt_foot_sliding", truth["foot_sliding"])
    anchor("A2_gt_feet_height", truth["feet_height"])
    anchor("A3_gt_contact_percent", truth["gt_contact_percent"])
    for gate_id, value in (
        ("A4_zero_gt_contact_sequences", len(measured["zero_gt_contact_sequences"])),
        ("A5_penetration_covered_sequences", len(measured["penetration_covered_sequences"])),
    ):
        expected = GATES[gate_id]["expected"]
        gates.append({
            "gate": gate_id, "expected": expected, "measured": value,
            "status": "skipped-subset" if not full_protocol
            else ("pass" if value == expected else "fail"),
        })
    gates.append({
        "gate": "A4b_analytic_contact_cap",
        "expected": ANALYTIC_CONTACT_CAP,
        "measured": truth["contact_f1"],
        "status": "skipped-subset" if not full_protocol
        else ("pass" if abs(truth["contact_f1"] - ANALYTIC_CONTACT_CAP) <= 1e-12 else "fail"),
        "note": "the ground-truth floor's own contact_f1 must equal 397/438, not 1.0",
    })
    gates.extend(_penetration_gates(truth, ceiling, penetration))
    gates.extend(_exploration_gates(measured["rows"], full_protocol))
    return gates


def _penetration_gates(truth: Dict[str, Any], ceiling: Dict[str, Any], penetration: bool) -> List[Dict[str, Any]]:
    """A6, the scaling convention, and A7, the out-of-box guard on the floor row."""
    if not penetration:
        return [
            {"gate": gate_id, "status": "skipped-no-penetration"}
            for gate_id in ("A6_human_pen_scaling", "A7_sdf_out_of_box_guard")
        ]
    gates: List[Dict[str, Any]] = []
    scaled = truth.get("human_pen_loss_infbagel")
    raw = truth.get("human_pen_loss_raw")
    hand = truth.get("hand_pen_loss_omomo")
    consistent = (
        raw not in (None, 0.0)
        and abs(scaled - raw * HUMAN_PEN_SCALE) <= 1e-09 * max(1.0, abs(scaled))
    )
    gates.append({
        "gate": "A6_human_pen_scaling", "expected": HUMAN_PEN_SCALE,
        "measured": None if raw in (None, 0.0) else scaled / raw,
        "status": "pass" if consistent else "fail",
        "note": "wiring check, not an independent anchor: it pins that the full-body "
                "term carries x10475/100 and the hand term does not",
        "hand_pen_loss_omomo_unscaled": hand,
    })
    # The out-of-box failure mode is a collapse of the penetration LOSS: if the
    # SDF query leaves the object's box, grid_sample's border padding returns
    # positive distances and min(.,0) is exactly zero.  The two ratio terms are a
    # frame count at a 4 cm depth and can legitimately be 0.0 for ground truth
    # that simply never penetrates that deep -- measured on the first 12
    # sequences, where both GT ratios are exactly 0.0 while both loss terms are
    # non-zero and within an order of magnitude of the P12 model row.  Keying the
    # gate on the ratios would therefore fail a healthy floor row, so the guard
    # keys on the losses and records a zero ratio as a warning instead.
    losses = ("hand_pen_loss_omomo", "human_pen_loss_infbagel")
    ratios: Dict[str, Optional[float]] = {}
    failures: List[str] = []
    warnings: List[str] = []
    for key, model_value in P12_PENETRATION_REFERENCE.items():
        value = truth.get(key)
        if value is None:
            ratios[key] = None
            failures.append(f"{key} missing")
            continue
        ratios[key] = value / model_value if model_value else None
        if key in losses:
            if value <= 0.0:
                failures.append(f"{key} collapsed to {value}")
            elif not A7_RATIO_BOUNDS[0] <= ratios[key] <= A7_RATIO_BOUNDS[1]:
                failures.append(f"{key} is {ratios[key]:.3g}x the P12 row")
        elif value == 0.0:
            warnings.append(f"{key} is exactly 0.0: no ground-truth frame reaches 4 cm depth")
    gates.append({
        "gate": "A7_sdf_out_of_box_guard",
        "expected": {"loss_terms_strictly_positive": True,
                     "loss_ratio_bounds_vs_p12": list(A7_RATIO_BOUNDS)},
        "measured": {"ratio_vs_p12": ratios,
                     "ceiling_human_pen_ratio": ceiling.get("human_pen_ratio")},
        "status": "fail" if failures else ("warn" if warnings else "pass"),
        "failures": failures,
        "warnings": warnings,
        "note": "a penetration LOSS of exactly zero is the 1.87 m vertex-displacement "
                "failure mode and forbids sealing; a zero penetration RATIO with a "
                "non-zero loss is a legitimate ground-truth reading",
    })
    return gates


def _exploration_gates(rows: Dict[str, Dict[str, Any]], full_protocol: bool) -> List[Dict[str, Any]]:
    """Compare against the 2026-08-20 exploration this row promotes."""
    disagreements: Dict[str, Dict[str, float]] = {}
    for name, expected_row in EXPLORATION_438.items():
        for key, expected in expected_row.items():
            measured = rows[name].get(key)
            if measured is None:
                continue
            delta = abs(measured - expected)
            tolerance = max(EXPLORATION_ABSOLUTE_FLOOR, EXPLORATION_RELATIVE_TOLERANCE * abs(expected))
            if delta > tolerance:
                disagreements.setdefault(name, {})[key] = delta
    return [{
        "gate": "E1_agrees_with_2026_08_20_exploration",
        "expected": "all carried-in values within float32 noise",
        "measured": disagreements or "all within tolerance",
        "status": "skipped-subset" if not full_protocol else ("pass" if not disagreements else "fail"),
        "note": "a disagreement means one of the two measurements is wrong; neither is "
                "sealed and neither is preferred",
    }]


def classify(gates: Sequence[Dict[str, Any]], penetration: bool, full_protocol: bool) -> str:
    """Map the gate outcomes onto the preregistered stop classifications."""
    if not full_protocol:
        return "repr-ceiling-subset-smoke"
    status = {gate["gate"]: gate["status"] for gate in gates}
    if status.get("E1_agrees_with_2026_08_20_exploration") == "fail":
        return "repr-ceiling-contradicts-exploration"
    anchors = ("A1_gt_foot_sliding", "A2_gt_feet_height", "A3_gt_contact_percent")
    if any(status.get(gate) == "fail" for gate in anchors):
        return "repr-ceiling-anchor-fail-stop"
    if status.get("A4_zero_gt_contact_sequences") == "fail":
        return "repr-ceiling-anchor-fail-stop"
    later = ("A5_penetration_covered_sequences", "A6_human_pen_scaling", "A7_sdf_out_of_box_guard")
    # "warn" is a recorded observation, not a failed gate: A7 warns when a
    # ground-truth penetration ratio is exactly 0.0 while its loss is non-zero.
    if not penetration or any(status.get(gate) not in ("pass", "warn") for gate in later):
        return "repr-ceiling-penetration-partial"
    if any(gate["status"] == "fail" for gate in gates):
        return "repr-ceiling-penetration-partial"
    return "repr-ceiling-row-established"


def provenance() -> Dict[str, Any]:
    import numpy
    import torch

    try:
        commit = chois_evaluator.git_output(ROOT, "rev-parse", "HEAD")
        dirty = bool(chois_evaluator.git_output(ROOT, "status", "--porcelain"))
    except chois_evaluator.EvaluatorError:
        commit, dirty = None, None
    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "tool": "tools/measure_hoi_repr_ceiling.py",
        "tool_sha256": chois_evaluator.sha256_file(Path(__file__).resolve()),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
    }


def build_parser() -> argparse.ArgumentParser:
    """CPU-only by construction: there is no device, checkpoint or model option."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--output", type=Path, required=True,
                        help="JSON result path; an existing path is never overwritten")
    parser.add_argument("--sequences", type=int, default=None,
                        help="score only the first N protocol sequences (subset smoke); "
                             "omitting it scores all 438 and is the only sealable form")
    parser.add_argument("--skip-penetration", action="store_true",
                        help="omit the four penetration terms; forces a non-sealing classification")
    parser.add_argument("--config-name", default="config_eval_hoi_prior")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=4,
                        help="torch CPU threads; the default matches the project's OMP cap")
    parser.add_argument("--progress-every", type=int, default=50)
    return parser


def run(args: argparse.Namespace) -> Dict[str, Any]:
    import numpy as np
    import torch

    output = args.output.resolve()
    if output.exists():
        raise CeilingError(f"refusing to overwrite reference-row output: {output}")
    torch.set_num_threads(max(1, int(args.threads)))
    penetration = not args.skip_penetration
    offsets = load_sequence_offsets(args.sequences)
    full_protocol = len(offsets) == PROTOCOL["sequences"]
    started = time.perf_counter()
    with project_code_context():
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        dataset, _ = build_dataset(args.config_name)
        build_seconds = round(time.perf_counter() - started, 1)
        rest_vertices = load_rest_object_vertices()
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        measured = measure(dataset, offsets, rest_vertices, penetration, args.progress_every)
    gates = evaluate_gates(measured, penetration, full_protocol)
    result = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiment_id": "p1-hoi-p12-repr-ceiling-row-s42-20260821",
        "plan": "docs/plan/PHASE_1B_HOI/07_REPRESENTATION_FRAME.md",
        "execution_contract": EXECUTION_CONTRACT,
        "provenance": provenance(),
        "protocol": dict(PROTOCOL, sequences=len(offsets), seed=args.seed,
                         config_name=args.config_name, full_protocol=full_protocol),
        "penetration": {
            "computed": penetration,
            "excluded_objects": list(PENETRATION_EXCLUDED_OBJECTS),
            "excluded_source": "code/test_infbagel_hoi.py:276, copied and not fixed",
            "covered_sequences": len(measured["penetration_covered_sequences"]),
            "excluded_sequences": len(offsets) - len(measured["penetration_covered_sequences"]),
            "human_pen_scale": HUMAN_PEN_SCALE,
        },
        "informative_metrics": list(INFORMATIVE_METRICS),
        "analytically_fixed_metrics": ANALYTICALLY_FIXED_METRICS,
        "analytic_contact_cap": ANALYTIC_CONTACT_CAP,
        "rows": measured["rows"],
        "restricted_to_defined_contact": restricted_contact_means(measured["per_sequence"]),
        "degenerate_sequences": {
            "zero_gt_contact": sorted(measured["zero_gt_contact_sequences"]),
            "count": len(measured["zero_gt_contact_sequences"]),
        },
        "gates": gates,
        "classification": classify(gates, penetration, full_protocol),
        "runtime_s": {"dataset_build": build_seconds, "scoring": measured["elapsed_s"],
                      "total": round(time.perf_counter() - started, 1)},
        "per_sequence_metrics": measured["per_sequence"],
    }
    chois_evaluator.atomic_output(output, result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run(args)
    except (CeilingError, chois_evaluator.EvaluatorError, OSError, ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = {
        "classification": result["classification"],
        "sequences": result["protocol"]["sequences"],
        "rows": result["rows"],
        "gates": [{"gate": gate["gate"], "status": gate["status"]} for gate in result["gates"]],
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
