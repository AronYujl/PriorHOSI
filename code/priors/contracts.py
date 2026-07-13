"""Immutable Phase 1A domain contracts."""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class DatasetContract:
    expert: str
    source: str
    scene_condition: str
    object_condition: str
    split: str
    filter_rule: str
    normalization: str
    window_frames: int = 16
    frame_stride: int = 3
    history_frames: int = 2
    diffusion_steps: int = 500

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


HOI_CONTRACT = DatasetContract(
    expert="hoi",
    source="scene-free fields from real OMOMO only",
    scene_condition="forbidden: dataset and model API expose no scene input",
    object_condition="required: dynamic object BPS/pose/contact and full OMOMO instruction",
    split="official processed OMOMO train and test/validation roots",
    filter_rule="no LINGO samples; retain OMOMO windows",
    normalization="data/train/norm.npy (existing OMOMO normalization; never recompute)",
)

HSI_CONTRACT = DatasetContract(
    expert="hsi",
    source="real LINGO scene-motion only",
    scene_condition="required: real LINGO occupancy plus text/pelvis/static-interaction goals",
    object_condition="forbidden supervision: object pose/contact fixed empty and loss-masked",
    split="experiments/splits/lingo_scene_disjoint_seed42.json",
    filter_rule=("left_hand_inter_frame == -1 and right_hand_inter_frame == -1; "
                 "then reject seq_length <= 48 as invalid for the 48-source-frame/16-output-frame contract"),
    normalization="data/dataset/norm.npy, byte-identical to OMOMO normalization; never recompute",
)


def validate_contract_paths(repo: Path, expert: Optional[str] = None) -> None:
    if expert not in {None, "hoi", "hsi"}:
        raise ValueError(f"unknown expert contract: {expert}")
    required = [repo / "data/train/norm.npy"]
    if expert in {None, "hsi"}:
        required.extend((repo / "data/dataset/norm.npy", repo / HSI_CONTRACT.split))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing contract assets: {missing}")
