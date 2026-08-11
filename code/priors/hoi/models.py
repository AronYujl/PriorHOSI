"""Independent randomly initialized HOI expert.

The HOI network is deliberately scene-free.  Its condition tokens mirror the
capacity of the released single-model Transformer while exposing only the
Phase 1A HOI contract: language/progress, dynamic-object BPS, and object goal.

``HSIPrior`` lives in ``priors.hsi.models``; ``build_expert``,
``_time_embedding`` and ``assert_parameter_independence`` live in
``priors.core.expert_api``.  ``build_expert`` is re-exported here so
``from priors.hoi.models import build_expert`` keeps resolving.
"""

from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
from torch import nn

from ..core.expert_api import _time_embedding, build_expert
from ..core.representation import REPRESENTATION
from .interaction_adapter import (
    LEGACY_BASIS_COORDINATE_SYSTEM,
    LOCAL_BASIS_COORDINATE_SYSTEM,
    LocalObjectInteractionAdapter,
)
from .sparse_relation import (
    ARCHITECTURE_VARIANT as HOI_ARCHITECTURE_D2AE,
    D2AF_ARCHITECTURE_VARIANT as HOI_ARCHITECTURE_D2AF,
    D2AG_ARCHITECTURE_VARIANT as HOI_ARCHITECTURE_D2AG,
    SparseCurrentStateRelationField,
    validate_diffusion_reliability_contract,
    validate_selfcond_relation_source_contract,
    validate_sparse_relation_contract,
)


HOI_ARCHITECTURE_BASE = "base"
HOI_ARCHITECTURE_D2AC = "d2ac_interaction_adapter"
HOI_ARCHITECTURE_D2AD = "d2ad_local_frame_interaction_adapter"
HOI_ARCHITECTURE_D2AJ = "d2aj_split_goal_tokens"
HOI_SPARSE_RELATION_ARCHITECTURES = frozenset({
    HOI_ARCHITECTURE_D2AE,
    HOI_ARCHITECTURE_D2AF,
    HOI_ARCHITECTURE_D2AG,
})
HOI_ARCHITECTURES = frozenset({
    HOI_ARCHITECTURE_BASE,
    HOI_ARCHITECTURE_D2AC,
    HOI_ARCHITECTURE_D2AD,
    HOI_ARCHITECTURE_D2AJ,
    *HOI_SPARSE_RELATION_ARCHITECTURES,
})

# D2-AJ splits the single fused goal/progress conditioning token into separate
# pelvis, object and progress tokens. The pelvis projection reads xz only,
# because the pelvis goal's y channel is zeroed on both the training
# (priors/data.py) and sampling (priors/diffusion.py) sides, and neither side
# ever writes goals[3:6].
HOI_D2AJ_CONDITION_TOKENS = 6
HOI_BASE_CONDITION_TOKENS = 4
HOI_D2AJ_PELVIS_GOAL_CHANNELS = (0, 2)
HOI_D2AJ_OBJECT_GOAL_SLICE = slice(6, 9)


class _HOICleanMotionNetwork(nn.Module):
    """Condition-token Transformer that predicts the clean 232-D window."""

    def __init__(
        self,
        dim_model: int,
        num_heads: int,
        num_layers: int,
        dropout: float = 0.1,
        *,
        architecture_variant: str = HOI_ARCHITECTURE_BASE,
        bps_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        if architecture_variant not in HOI_ARCHITECTURES:
            raise ValueError(f"unknown HOIPrior architecture variant: {architecture_variant}")
        if architecture_variant in {
            HOI_ARCHITECTURE_D2AC,
            HOI_ARCHITECTURE_D2AD,
            *HOI_SPARSE_RELATION_ARCHITECTURES,
        } and num_layers != 8:
            raise ValueError(
                "D2-AC/D2-AD/D2-AE/D2-AF/D2-AG require the locked eight-layer "
                "HOIPrior trunk"
            )
        if architecture_variant in HOI_SPARSE_RELATION_ARCHITECTURES and (
            dim_model != 512 or num_heads != 16
        ):
            raise ValueError(
                "D2-AE/D2-AF/D2-AG require the locked 512-wide, 16-head "
                "HOIPrior trunk"
            )
        self.motion_input = nn.Linear(REPRESENTATION.dimension, dim_model)
        self.text = nn.Sequential(
            nn.Linear(768, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
        )
        self.bps = nn.Sequential(
            nn.Linear(1024 * 3, 768), nn.SiLU(), nn.Linear(768, dim_model),
        )
        self.goal_progress: Optional[nn.Module] = None
        self.pelvis_goal: Optional[nn.Module] = None
        self.object_goal: Optional[nn.Module] = None
        self.progress: Optional[nn.Module] = None
        if architecture_variant == HOI_ARCHITECTURE_D2AJ:
            self.pelvis_goal = nn.Sequential(
                nn.Linear(2, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
            )
            self.object_goal = nn.Sequential(
                nn.Linear(3, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
            )
            self.progress = nn.Sequential(
                nn.Linear(3, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
            )
            condition_tokens = HOI_D2AJ_CONDITION_TOKENS
        else:
            self.goal_progress = nn.Sequential(
                nn.Linear(12, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
            )
            condition_tokens = HOI_BASE_CONDITION_TOKENS
        self.condition_tokens = condition_tokens
        self.time = nn.Sequential(
            nn.Linear(dim_model, dim_model), nn.SiLU(), nn.Linear(dim_model, dim_model),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dim_model,
            nhead=num_heads,
            dim_feedforward=dim_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.position = nn.Parameter(
            torch.zeros(1, REPRESENTATION.window_frames + condition_tokens, dim_model)
        )
        nn.init.normal_(self.position, std=0.02)
        self.output_norm = nn.LayerNorm(dim_model)
        self.output = nn.Linear(dim_model, REPRESENTATION.dimension)
        self.dim_model = dim_model
        self.architecture_variant = architecture_variant
        self.interaction_adapter = (
            LocalObjectInteractionAdapter(
                dim_model,
                bps_path=bps_path,
                basis_coordinate_system=(
                    LOCAL_BASIS_COORDINATE_SYSTEM
                    if architecture_variant == HOI_ARCHITECTURE_D2AD
                    else LEGACY_BASIS_COORDINATE_SYSTEM
                ),
            )
            if architecture_variant in {
                HOI_ARCHITECTURE_D2AC,
                HOI_ARCHITECTURE_D2AD,
            }
            else None
        )
        self.sparse_relation_field = (
            SparseCurrentStateRelationField(
                dim_model,
                diffusion_reliability=(
                    architecture_variant == HOI_ARCHITECTURE_D2AF
                ),
                selfcond_relation_source=(
                    architecture_variant == HOI_ARCHITECTURE_D2AG
                ),
            )
            if architecture_variant in HOI_SPARSE_RELATION_ARCHITECTURES
            else None
        )

    def _encode(
        self, tokens: torch.Tensor, object_bps: torch.Tensor,
    ) -> torch.Tensor:
        if self.interaction_adapter is None:
            # Preserve the exact legacy/base execution path and checkpoint
            # behavior for every non-D2-AC HOIPrior.
            return self.transformer(tokens)
        encoded = tokens
        for layer_index, layer in enumerate(self.transformer.layers):
            encoded = layer(encoded)
            if layer_index == 3:
                motion = self.interaction_adapter(encoded[:, 4:], object_bps)
                encoded = torch.cat((encoded[:, :4], motion), dim=1)
        if self.transformer.norm is not None:
            encoded = self.transformer.norm(encoded)
        return encoded

    def set_interaction_diagnostic_variant(self, variant: str) -> None:
        if self.interaction_adapter is None:
            raise ValueError("HOIPrior has no D2-AC interaction adapter")
        self.interaction_adapter.set_diagnostic_variant(variant)

    def set_interaction_attention_capture(self, enabled: bool) -> None:
        if self.interaction_adapter is None:
            raise ValueError("HOIPrior has no D2-AC interaction adapter")
        self.interaction_adapter.set_capture_attention(enabled)

    def interaction_attention_snapshot(self):
        if self.interaction_adapter is None:
            raise ValueError("HOIPrior has no D2-AC interaction adapter")
        return self.interaction_adapter.attention_snapshot()

    def set_sparse_relation_diagnostic_variant(self, variant: str) -> None:
        if self.sparse_relation_field is None:
            raise ValueError("HOIPrior has no sparse relation field")
        self.sparse_relation_field.set_diagnostic_variant(variant)

    def set_sparse_relation_gate_override(self, value: Optional[float]) -> None:
        if self.sparse_relation_field is None:
            raise ValueError("HOIPrior has no sparse relation field")
        self.sparse_relation_field.set_gate_override(value)

    def set_sparse_relation_rho_override(self, value: Optional[float]) -> None:
        if self.sparse_relation_field is None:
            raise ValueError("HOIPrior has no sparse relation field")
        self.sparse_relation_field.set_rho_override(value)

    def set_sparse_relation_capture(self, enabled: bool) -> None:
        if self.sparse_relation_field is None:
            raise ValueError("HOIPrior has no sparse relation field")
        self.sparse_relation_field.set_capture(enabled)

    def sparse_relation_snapshot(self):
        if self.sparse_relation_field is None:
            raise ValueError("HOIPrior has no sparse relation field")
        return self.sparse_relation_field.snapshot()

    def forward(
        self,
        noisy: torch.Tensor,
        timesteps: torch.Tensor,
        text_embedding: torch.Tensor,
        object_bps: torch.Tensor,
        goals: torch.Tensor,
        progress: torch.Tensor,
        *,
        local_object_bps: Optional[torch.Tensor] = None,
        rest_object_points: Optional[torch.Tensor] = None,
        world_to_local_rotation: Optional[torch.Tensor] = None,
        object_rotation_reference: Optional[torch.Tensor] = None,
        position_minimum: Optional[torch.Tensor] = None,
        position_maximum: Optional[torch.Tensor] = None,
        object_minimum: Optional[torch.Tensor] = None,
        object_maximum: Optional[torch.Tensor] = None,
        relation_source: Optional[torch.Tensor] = None,
        relation_source_estimate: bool = False,
    ) -> torch.Tensor:
        if noisy.ndim != 3 or noisy.shape[1:] != (
            REPRESENTATION.window_frames, REPRESENTATION.dimension,
        ):
            raise ValueError(f"expected noisy [B,16,232], got {tuple(noisy.shape)}")
        if text_embedding.ndim == 3 and text_embedding.shape[1] == 1:
            text_embedding = text_embedding[:, 0]
        if object_bps.ndim != 3 or object_bps.shape[1:] != (1024, 3):
            raise ValueError(f"expected global object BPS [B,1024,3], got {tuple(object_bps.shape)}")
        if self.architecture_variant != HOI_ARCHITECTURE_D2AG:
            if relation_source is not None:
                raise ValueError(
                    "an explicit relation source state is restricted to D2-AG"
                )
            if bool(relation_source_estimate):
                raise ValueError(
                    "the relation-source estimate pass is restricted to D2-AG"
                )
        elif relation_source is not None and bool(relation_source_estimate):
            raise ValueError(
                "the D2-AG estimate pass must use the current-state relation source"
            )
        if self.architecture_variant == HOI_ARCHITECTURE_D2AD:
            if (
                local_object_bps is None
                or local_object_bps.ndim != 3
                or local_object_bps.shape != object_bps.shape
            ):
                raise ValueError(
                    "D2-AD requires current human-local object BPS [B,1024,3]"
                )
            adapter_bps = local_object_bps
        else:
            if local_object_bps is not None:
                raise ValueError("human-local object BPS is restricted to D2-AD")
            adapter_bps = object_bps
        relation_values = (
            rest_object_points,
            world_to_local_rotation,
            object_rotation_reference,
            position_minimum,
            position_maximum,
            object_minimum,
            object_maximum,
        )
        if self.architecture_variant in HOI_SPARSE_RELATION_ARCHITECTURES:
            if any(value is None for value in relation_values):
                raise ValueError(
                    "D2-AE/D2-AF/D2-AG require current-state sparse relation metadata"
                )
        elif any(value is not None for value in relation_values):
            raise ValueError(
                "sparse relation metadata is restricted to D2-AE/D2-AF/D2-AG"
            )
        shared_conditions = (
            self.time(_time_embedding(timesteps, self.dim_model)),
            self.text(text_embedding),
            self.bps(object_bps.reshape(object_bps.shape[0], -1)),
        )
        if self.architecture_variant == HOI_ARCHITECTURE_D2AJ:
            # Separate pelvis / object / progress tokens so attention can weight
            # the two goals independently.  The pelvis projection reads xz only
            # and goals[3:6] is never read, because both are identically zero on
            # the training and sampling sides alike.
            pelvis_xz = goals[..., HOI_D2AJ_PELVIS_GOAL_CHANNELS]
            conditions = torch.stack((
                *shared_conditions,
                self.pelvis_goal(pelvis_xz),
                self.object_goal(goals[..., HOI_D2AJ_OBJECT_GOAL_SLICE]),
                self.progress(progress),
            ), dim=1)
        else:
            conditions = torch.stack((
                *shared_conditions,
                self.goal_progress(torch.cat((goals, progress), dim=-1)),
            ), dim=1)
        motion = self.motion_input(noisy)
        if self.sparse_relation_field is not None:
            # Mark the in-flight pass on the module itself before the call so
            # forward pre-hooks, which never see keyword arguments, can tell the
            # D2-AG no-grad estimate pass from the graph-building pass.
            self.sparse_relation_field.relation_source_estimate = bool(
                relation_source_estimate
            )
            try:
                motion = self.sparse_relation_field(
                    motion,
                    noisy,
                    rest_object_points,
                    world_to_local_rotation,
                    object_rotation_reference,
                    position_minimum,
                    position_maximum,
                    object_minimum,
                    object_maximum,
                    timesteps=(
                        timesteps
                        if self.architecture_variant in {
                            HOI_ARCHITECTURE_D2AF,
                            HOI_ARCHITECTURE_D2AG,
                        }
                        else None
                    ),
                    relation_source=relation_source,
                )
            finally:
                self.sparse_relation_field.relation_source_estimate = False
        tokens = torch.cat((conditions, motion), dim=1) + self.position
        encoded = self._encode(tokens, adapter_bps)
        return self.output(self.output_norm(encoded[:, -REPRESENTATION.window_frames:]))


class HOIPrior(nn.Module):
    """HOI API intentionally has no scene argument or scene encoder."""
    def __init__(
        self,
        dim_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        *,
        architecture_variant: str = HOI_ARCHITECTURE_BASE,
        bps_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.network = _HOICleanMotionNetwork(
            dim_model,
            num_heads,
            num_layers,
            architecture_variant=architecture_variant,
            bps_path=bps_path,
        )
        self.architecture_variant = architecture_variant

    def forward(
        self, noisy: torch.Tensor, timesteps: torch.Tensor, text_embedding: torch.Tensor,
        object_bps: torch.Tensor, goals: torch.Tensor, progress: torch.Tensor,
        *,
        local_object_bps: Optional[torch.Tensor] = None,
        rest_object_points: Optional[torch.Tensor] = None,
        world_to_local_rotation: Optional[torch.Tensor] = None,
        object_rotation_reference: Optional[torch.Tensor] = None,
        position_minimum: Optional[torch.Tensor] = None,
        position_maximum: Optional[torch.Tensor] = None,
        object_minimum: Optional[torch.Tensor] = None,
        object_maximum: Optional[torch.Tensor] = None,
        relation_source: Optional[torch.Tensor] = None,
        relation_source_estimate: bool = False,
    ) -> torch.Tensor:
        return self.network(
            noisy,
            timesteps,
            text_embedding,
            object_bps,
            goals,
            progress,
            local_object_bps=local_object_bps,
            rest_object_points=rest_object_points,
            world_to_local_rotation=world_to_local_rotation,
            object_rotation_reference=object_rotation_reference,
            position_minimum=position_minimum,
            position_maximum=position_maximum,
            object_minimum=object_minimum,
            object_maximum=object_maximum,
            relation_source=relation_source,
            relation_source_estimate=relation_source_estimate,
        )


def load_trained_hoi_prior(
    checkpoint_path: str, device: torch.device, *, use_ema: bool = True,
    weight_variant: Optional[str] = None,
    expected_architecture_variant: Optional[str] = None,
) -> Tuple[HOIPrior, Dict[str, object]]:
    """Strictly load a Phase 1B checkpoint for evaluation or same-run resume.

    Released InfBaGel state dictionaries do not carry this checkpoint schema and
    are rejected before any parameter is loaded.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = torch.load(str(path), map_location="cpu")
    if not isinstance(checkpoint, dict) or checkpoint.get("checkpoint_type") != "hoi_prior_phase1b":
        raise ValueError("checkpoint is not a Phase 1B HOIPrior checkpoint; released InfBaGel initialization is forbidden")
    if checkpoint.get("expert") != "hoi" or checkpoint.get("initialization") != "random":
        raise ValueError("invalid HOIPrior checkpoint provenance")
    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, dict):
        raise ValueError("HOIPrior checkpoint is missing model_config")
    architecture_variant = str(
        model_config.get("architecture_variant", HOI_ARCHITECTURE_BASE)
    )
    if architecture_variant not in HOI_ARCHITECTURES:
        raise ValueError(f"unknown HOIPrior checkpoint architecture variant: {architecture_variant}")
    checkpoint_variant = checkpoint.get("architecture_variant")
    if architecture_variant in {HOI_ARCHITECTURE_D2AC, HOI_ARCHITECTURE_D2AD}:
        if checkpoint_variant != architecture_variant:
            raise ValueError("interaction-adapter checkpoint is missing its architecture provenance")
        adapter_contract = checkpoint.get("interaction_adapter_contract")
        if not isinstance(adapter_contract, dict):
            raise ValueError("checkpoint is missing interaction-adapter provenance")
        from .interaction_adapter import ASSIGNMENT_SHA256, BPS_SHA256
        if (
            adapter_contract.get("bps_sha256") != BPS_SHA256
            or adapter_contract.get("assignment_sha256") != ASSIGNMENT_SHA256
            or adapter_contract.get("adapter_parameters") != 349697
        ):
            raise ValueError("checkpoint interaction-adapter provenance mismatch")
        if architecture_variant == HOI_ARCHITECTURE_D2AD:
            from .d2ad import (
                BPS_YUP_TENSOR_SHA256,
                DEFAULT_QUERY_WORKERS,
                OBJECT_MAPPING_SHA256,
                REST_MESH_MANIFEST_SHA256,
            )
            if (
                adapter_contract.get("basis_coordinate_system")
                != LOCAL_BASIS_COORDINATE_SYSTEM
                or adapter_contract.get("basis_yup_tensor_sha256")
                != BPS_YUP_TENSOR_SHA256
                or adapter_contract.get("rest_mesh_manifest_sha256")
                != REST_MESH_MANIFEST_SHA256
                or adapter_contract.get("object_mapping_sha256")
                != OBJECT_MAPPING_SHA256
                or adapter_contract.get("query_backend")
                != "scipy.spatial.cKDTree.query"
                or adapter_contract.get("query_parameters")
                != {"k": 1, "eps": 0.0, "p": 2}
                or adapter_contract.get("query_workers")
                != DEFAULT_QUERY_WORKERS
                or adapter_contract.get("full_rest_mesh") is not True
                or adapter_contract.get("mesh_subsample") is not False
                or adapter_contract.get("stored_per_window_local_bps") is not False
            ):
                raise ValueError("D2-AD checkpoint local-geometry provenance mismatch")
    elif architecture_variant == HOI_ARCHITECTURE_D2AE:
        if model_config != {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AE,
        }:
            raise ValueError("D2-AE checkpoint model_config violates the locked trunk")
        if checkpoint_variant != HOI_ARCHITECTURE_D2AE:
            raise ValueError("D2-AE checkpoint is missing its architecture provenance")
        if checkpoint.get("selfcond_relation_source_contract") is not None:
            raise ValueError(
                "D2-AE checkpoint carries a D2-AG selfcond-relation-source contract"
            )
        validate_sparse_relation_contract(checkpoint.get("sparse_relation_contract"))
    elif architecture_variant == HOI_ARCHITECTURE_D2AF:
        if model_config != {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AF,
        }:
            raise ValueError("D2-AF checkpoint model_config violates the locked trunk")
        if checkpoint_variant != HOI_ARCHITECTURE_D2AF:
            raise ValueError("D2-AF checkpoint is missing its architecture provenance")
        if checkpoint.get("selfcond_relation_source_contract") is not None:
            raise ValueError(
                "D2-AF checkpoint carries a D2-AG selfcond-relation-source contract"
            )
        validate_diffusion_reliability_contract(
            checkpoint.get("diffusion_reliability_contract")
        )
    elif architecture_variant == HOI_ARCHITECTURE_D2AG:
        if model_config != {
            "dim_model": 512,
            "num_heads": 16,
            "num_layers": 8,
            "architecture_variant": HOI_ARCHITECTURE_D2AG,
        }:
            raise ValueError("D2-AG checkpoint model_config violates the locked trunk")
        if checkpoint_variant != HOI_ARCHITECTURE_D2AG:
            raise ValueError("D2-AG checkpoint is missing its architecture provenance")
        if (
            checkpoint.get("sparse_relation_contract") is not None
            or checkpoint.get("diffusion_reliability_contract") is not None
        ):
            raise ValueError(
                "D2-AG checkpoint carries a predecessor sparse-relation contract"
            )
        validate_selfcond_relation_source_contract(
            checkpoint.get("selfcond_relation_source_contract")
        )
    elif architecture_variant == HOI_ARCHITECTURE_D2AJ:
        if checkpoint_variant != HOI_ARCHITECTURE_D2AJ:
            raise ValueError("D2-AJ checkpoint is missing its architecture provenance")
        contract = checkpoint.get("split_goal_token_contract")
        if not isinstance(contract, dict):
            raise ValueError("D2-AJ checkpoint is missing its split-goal-token contract")
        if (
            contract.get("architecture_variant") != HOI_ARCHITECTURE_D2AJ
            or int(contract.get("condition_tokens", 0)) != HOI_D2AJ_CONDITION_TOKENS
            or bool(contract.get("goals_3_to_6_read", True))
            or bool(contract.get("pelvis_goal_y_read", True))
            or bool(contract.get("fused_goal_progress_module_present", True))
        ):
            raise ValueError("D2-AJ checkpoint split-goal-token contract mismatch")
    elif checkpoint_variant not in (None, HOI_ARCHITECTURE_BASE):
        raise ValueError("base HOIPrior checkpoint carries an incompatible architecture variant")
    if (
        expected_architecture_variant is not None
        and architecture_variant != expected_architecture_variant
    ):
        raise ValueError(
            "HOIPrior checkpoint architecture variant mismatch: "
            f"{architecture_variant} != {expected_architecture_variant}"
        )
    model = build_expert(
        "hoi",
        init_checkpoint=None,
        dim_model=int(model_config["dim_model"]),
        num_heads=int(model_config["num_heads"]),
        num_layers=int(model_config["num_layers"]),
        architecture_variant=architecture_variant,
    )
    if weight_variant is None:
        weight_variant = "ema_0.9999" if use_ema else "online"
    if weight_variant == "online":
        state_key = "model"
        state = checkpoint.get(state_key)
    elif weight_variant in {"ema_0.999", "ema_0.9999"}:
        decay = weight_variant[len("ema_"):]
        ema_models = checkpoint.get("ema_models")
        if isinstance(ema_models, dict) and decay in ema_models:
            state_key = f"ema_models[{decay}]"
            state = ema_models[decay]
        elif decay == "0.9999":
            state_key = "ema_model"
            state = checkpoint.get(state_key)
        else:
            state_key = f"ema_models[{decay}]"
            state = None
    else:
        raise ValueError(f"unknown HOIPrior weight variant: {weight_variant}")
    if not isinstance(state, dict):
        raise ValueError(f"HOIPrior checkpoint is missing {state_key} weights")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    metadata = {
        key: checkpoint.get(key) for key in (
            "schema_version", "checkpoint_type", "expert", "initialization", "run_id",
            "seed", "git_commit", "processed_windows", "processed_frames", "optimizer_updates",
            "model_config", "architecture_variant", "interaction_adapter_contract",
            "sparse_relation_contract", "diffusion_reliability_contract",
            "selfcond_relation_source_contract",
            "data_contract_sha256", "split_sha256", "window_state_codec",
        )
    }
    metadata["architecture_variant"] = architecture_variant
    metadata["weights"] = state_key
    metadata["weight_variant"] = weight_variant
    metadata["path"] = str(path)
    return model, metadata
