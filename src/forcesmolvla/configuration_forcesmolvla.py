"""Strict two-camera Cartesian7D configuration bound to LeRobot v0.6.0."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.utils.constants import ACTION, OBS_STATE


CAMERA1 = "observation.images.camera1"
CAMERA2 = "observation.images.camera2"
WRENCH = "observation.wrench"
OFFLINE_FULL_FINETUNE = "offline_full_finetune"
ONLINE_HIL_VLM_FROZEN = "online_hil_vlm_frozen"
SMOLVLA_CARTESIAN7D = "smolvla_cartesian7d"
FORCE_TOKEN_DENSE_COMPUTE = "force_token_dense_compute"
FORCE_TOKEN_DENSE_PARAM = "force_token_dense_param"
FORCE_TOKEN_MOE = "force_token_moe"
FORCE_TOKEN_MOE_ADDITIVE = "force_token_moe_additive"


@PreTrainedConfig.register_subclass("force_smolvla")
@dataclass
class ForceSmolVLAConfig(SmolVLAConfig):
    training_stage: str = OFFLINE_FULL_FINETUNE
    force_variant: str = SMOLVLA_CARTESIAN7D
    acceptance_status: str = "development_only"
    force_init_seed: int = 42
    fusion_include_state_token: bool = False
    fusion_num_blocks: int = 2
    fusion_num_heads: int = 8
    force_cross_attention_type: str = "single_head_scaled_dot_product"
    force_cross_attention_num_heads: int = 1
    dense_param_hidden_dim: int = 15364
    moe_num_experts: int = 4
    router_temperature: float = 1.0
    router_top_k: int = 1
    router_capacity_free: bool = True
    router_token_drop: bool = False
    adapter_query_mode: str = "action_query"

    def __post_init__(self) -> None:
        super().__post_init__()
        expected_images = (CAMERA1, CAMERA2)
        if tuple(self.image_features) != expected_images:
            raise ValueError(f"expected exact ordered image features {expected_images}")
        if self.robot_state_feature is None or self.robot_state_feature.shape != (7,):
            raise ValueError("ForceSmolVLA requires observation.state shape (7,)")
        wrench_feature = self.input_features.get(WRENCH) if self.input_features else None
        if (
            wrench_feature is None
            or wrench_feature.type is not FeatureType.ENV
            or wrench_feature.shape != (6,)
        ):
            raise ValueError("ForceSmolVLA requires explicit observation.wrench ENV shape (6,)")
        if self.action_feature is None or self.action_feature.shape != (7,):
            raise ValueError("ForceSmolVLA requires action shape (7,)")
        if self.rtc_config is not None:
            raise ValueError("ForceSmolVLA requires rtc_config=None")
        if self.empty_cameras != 0:
            raise ValueError("ForceSmolVLA forbids empty/placeholder cameras")
        if self.adapt_to_pi_aloha or self.use_delta_joint_actions_aloha:
            raise ValueError("ForceSmolVLA forbids inherited Aloha transforms")
        if self.max_state_dim != 32 or self.max_action_dim != 32:
            raise ValueError("base max state/action dimensions must remain 32")
        if self.chunk_size != 50 or self.num_steps != 10:
            raise ValueError("base chunk_size=50 and num_steps=10 are frozen")
        if self.n_action_steps != 1:
            raise ValueError("base n_action_steps=1 is frozen")
        if self.tokenizer_max_length != 48 or self.pad_language_to != "max_length":
            raise ValueError("language physical layout requires max_length=48")
        frozen_upstream_topology = {
            "resize_imgs_with_padding": (self.resize_imgs_with_padding, (512, 512)),
            "use_cache": (self.use_cache, True),
            "add_image_special_tokens": (self.add_image_special_tokens, False),
            "attention_mode": (self.attention_mode, "cross_attn"),
            "num_expert_layers": (self.num_expert_layers, 0),
            "num_vlm_layers": (self.num_vlm_layers, 16),
            "self_attn_every_n_layers": (self.self_attn_every_n_layers, 2),
            "expert_width_multiplier": (self.expert_width_multiplier, 0.75),
        }
        drift = {
            name: {"actual": actual, "expected": expected}
            for name, (actual, expected) in frozen_upstream_topology.items()
            if actual != expected
        }
        if drift:
            raise ValueError(f"frozen upstream SmolVLA topology drift: {drift}")
        if self.prefix_length != 177:
            raise ValueError("two-camera physical prefix length must be 177")
        if self.load_vlm_weights:
            raise ValueError("constructor must not fetch VLM weights; load the full local state dict instead")
        if self.acceptance_status != "development_only":
            raise ValueError("P5 implementation is development-only")
        if self.force_variant not in {
            SMOLVLA_CARTESIAN7D,
            FORCE_TOKEN_DENSE_COMPUTE,
            FORCE_TOKEN_DENSE_PARAM,
            FORCE_TOKEN_MOE,
            FORCE_TOKEN_MOE_ADDITIVE,
        }:
            raise ValueError(f"unsupported force variant: {self.force_variant!r}")
        if self.force_init_seed != 42:
            raise ValueError("P5 force initialization seed must be 42")
        if self.fusion_include_state_token:
            raise ValueError("P5 fusion must exclude the state token")
        if self.fusion_num_blocks != 2 or self.fusion_num_heads != 8:
            raise ValueError("P5 fusion requires two blocks and eight heads")
        if (
            self.force_cross_attention_type != "single_head_scaled_dot_product"
            or self.force_cross_attention_num_heads != 1
        ):
            raise ValueError("P5 adapter requires frozen single-head cross-attention")
        if self.dense_param_hidden_dim != 15364:
            raise ValueError("P6 Dense-Param hidden dimension must be 15364")
        if (
            self.moe_num_experts != 4
            or self.router_temperature != 1.0
            or self.router_top_k != 1
            or not self.router_capacity_free
            or self.router_token_drop
        ):
            raise ValueError("P6 requires four-expert capacity-free deterministic top-1 routing")
        expected_query_mode = (
            "additive" if self.force_variant == FORCE_TOKEN_MOE_ADDITIVE else "action_query"
        )
        if self.adapter_query_mode != expected_query_mode:
            raise ValueError(
                f"adapter query mode must be {expected_query_mode!r} for {self.force_variant!r}"
            )
        if (
            self.optimizer_lr != 1e-4
            or self.optimizer_betas != (0.9, 0.95)
            or self.optimizer_eps != 1e-8
            or self.optimizer_weight_decay != 1e-10
            or self.optimizer_grad_clip_norm != 10
            or self.scheduler_warmup_steps != 1000
            or self.scheduler_decay_steps != 20000
            or self.scheduler_decay_lr != 2.5e-6
        ):
            raise ValueError("P7 optimizer/scheduler recipe drift")
        if self.training_stage == OFFLINE_FULL_FINETUNE:
            if self.freeze_vision_encoder or self.train_expert_only:
                raise ValueError("offline full finetuning requires the VLM and vision encoder unfrozen")
        elif self.training_stage == ONLINE_HIL_VLM_FROZEN:
            if not self.freeze_vision_encoder or not self.train_expert_only:
                raise ValueError("online HIL requires the VLM and vision encoder frozen")
        else:
            raise ValueError(f"unsupported training stage: {self.training_stage!r}")


def load_force_config(
    base_checkpoint: Path,
    constructor_assets: Path,
    *,
    device: str,
    training_stage: str = OFFLINE_FULL_FINETUNE,
    force_variant: str = SMOLVLA_CARTESIAN7D,
    acceptance_status: str = "development_only",
    force_init_seed: int = 42,
) -> ForceSmolVLAConfig:
    base = PreTrainedConfig.from_pretrained(base_checkpoint, local_files_only=True)
    if not isinstance(base, SmolVLAConfig):
        raise TypeError(f"expected SmolVLAConfig, got {type(base).__name__}")
    values = {field.name: getattr(base, field.name) for field in fields(SmolVLAConfig)}
    if training_stage == OFFLINE_FULL_FINETUNE:
        freeze_vision_encoder = False
        train_expert_only = False
    elif training_stage == ONLINE_HIL_VLM_FROZEN:
        freeze_vision_encoder = True
        train_expert_only = True
    else:
        raise ValueError(f"unsupported training stage: {training_stage!r}")
    values.update(
        input_features={
            OBS_STATE: PolicyFeature(type=FeatureType.STATE, shape=(7,)),
            CAMERA1: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            CAMERA2: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 480, 640)),
            WRENCH: PolicyFeature(type=FeatureType.ENV, shape=(6,)),
        },
        output_features={ACTION: PolicyFeature(type=FeatureType.ACTION, shape=(7,))},
        normalization_mapping={
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.IDENTITY,
            "ACTION": NormalizationMode.IDENTITY,
            "ENV": NormalizationMode.IDENTITY,
        },
        device=device,
        vlm_model_name=str(constructor_assets.resolve()),
        load_vlm_weights=False,
        rtc_config=None,
        empty_cameras=0,
        adapt_to_pi_aloha=False,
        use_delta_joint_actions_aloha=False,
        compile_model=False,
        prefix_length=177,
        training_stage=training_stage,
        force_variant=force_variant,
        acceptance_status=acceptance_status,
        force_init_seed=force_init_seed,
        adapter_query_mode=(
            "additive" if force_variant == FORCE_TOKEN_MOE_ADDITIVE else "action_query"
        ),
        scheduler_warmup_steps=1000,
        scheduler_decay_steps=20000,
        scheduler_decay_lr=2.5e-6,
        freeze_vision_encoder=freeze_vision_encoder,
        train_expert_only=train_expert_only,
    )
    return ForceSmolVLAConfig(**values)
