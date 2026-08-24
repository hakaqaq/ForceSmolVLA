from pathlib import Path

import pytest

import torch

from forcesmolvla.checkpoint import (
    ALLOWED_DROPPED_NORMALIZER_KEYS,
    normalize_frozen_base_state_dict,
)
from forcesmolvla.configuration_forcesmolvla import (
    CAMERA1,
    CAMERA2,
    WRENCH,
    OFFLINE_FULL_FINETUNE,
    ONLINE_HIL_VLM_FROZEN,
    FORCE_TOKEN_DENSE_COMPUTE,
    FORCE_TOKEN_DENSE_PARAM,
    FORCE_TOKEN_MOE,
    FORCE_TOKEN_MOE_ADDITIVE,
    load_force_config,
)
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy


ROOT = Path(__file__).parents[1]


def test_two_camera_cartesian7d_config_offline():
    config = load_force_config(
        ROOT / "assets" / "base_checkpoint",
        ROOT / "assets" / "smolvlm_constructor",
        device="cpu",
    )
    assert tuple(config.image_features) == (CAMERA1, CAMERA2)
    assert config.robot_state_feature.shape == (7,)
    assert config.input_features[WRENCH].shape == (6,)
    assert config.input_features[WRENCH].type.value == "ENV"
    assert config.normalization_mapping["ENV"].value == "IDENTITY"
    assert config.action_feature.shape == (7,)
    assert config.chunk_size == 50
    assert config.n_action_steps == 1
    assert config.num_steps == 10
    assert config.prefix_length == 177
    assert config.resize_imgs_with_padding == (512, 512)
    assert config.use_cache is True
    assert config.add_image_special_tokens is False
    assert config.attention_mode == "cross_attn"
    assert config.num_expert_layers == 0
    assert config.num_vlm_layers == 16
    assert config.self_attn_every_n_layers == 2
    assert config.expert_width_multiplier == 0.75
    assert config.rtc_config is None
    assert config.load_vlm_weights is False
    assert config.training_stage == OFFLINE_FULL_FINETUNE
    assert config.freeze_vision_encoder is False
    assert config.train_expert_only is False


def test_p5_dense_variant_freezes_single_head_adapter_and_eight_head_fusion():
    config = load_force_config(
        ROOT / "assets" / "base_checkpoint",
        ROOT / "assets" / "smolvlm_constructor",
        device="cpu",
        force_variant=FORCE_TOKEN_DENSE_COMPUTE,
    )
    assert config.acceptance_status == "development_only"
    assert config.force_init_seed == 42
    assert config.fusion_num_blocks == 2
    assert config.fusion_num_heads == 8
    assert config.fusion_include_state_token is False
    assert config.force_cross_attention_type == "single_head_scaled_dot_product"
    assert config.force_cross_attention_num_heads == 1


@pytest.mark.parametrize(
    "variant", [FORCE_TOKEN_DENSE_PARAM, FORCE_TOKEN_MOE, FORCE_TOKEN_MOE_ADDITIVE]
)
def test_p6_variants_freeze_dense_param_and_moe_contract(variant):
    config = load_force_config(
        ROOT / "assets" / "base_checkpoint",
        ROOT / "assets" / "smolvlm_constructor",
        device="cpu",
        force_variant=variant,
    )
    assert config.acceptance_status == "development_only"
    assert config.dense_param_hidden_dim == 15364
    assert config.moe_num_experts == 4
    assert config.router_temperature == 1.0
    assert config.router_top_k == 1
    assert config.router_capacity_free is True
    assert config.router_token_drop is False


def test_online_hil_stage_freezes_vlm_flags():
    config = load_force_config(
        ROOT / "assets" / "base_checkpoint",
        ROOT / "assets" / "smolvlm_constructor",
        device="cpu",
        training_stage=ONLINE_HIL_VLM_FROZEN,
    )
    assert config.training_stage == ONLINE_HIL_VLM_FROZEN
    assert config.freeze_vision_encoder is True
    assert config.train_expert_only is True


def test_training_stage_flag_mismatch_fails_fast():
    config = load_force_config(
        ROOT / "assets" / "base_checkpoint",
        ROOT / "assets" / "smolvlm_constructor",
        device="cpu",
    )
    config.freeze_vision_encoder = True
    with pytest.raises(ValueError, match="offline full finetuning"):
        config.__post_init__()


def test_extra_camera_fails_fast():
    config = load_force_config(
        ROOT / "assets" / "base_checkpoint",
        ROOT / "assets" / "smolvlm_constructor",
        device="cpu",
    )
    config.input_features["observation.images.extra"] = config.input_features[CAMERA1]
    with pytest.raises(ValueError, match="exact ordered image"):
        config.__post_init__()


@pytest.mark.parametrize(
    "keys",
    [
        (CAMERA1, CAMERA2),
        (CAMERA2, CAMERA1),
        (CAMERA1,),
        (CAMERA1, CAMERA2, "observation.images.extra"),
    ],
)
def test_visual_batch_exact_order(keys):
    batch = {key: torch.zeros(1, 3, 8, 8) for key in keys}
    if keys == (CAMERA1, CAMERA2):
        ForceSmolVLAPolicy._validate_visual_batch(batch)
    else:
        with pytest.raises(ValueError, match="exact ordered visual"):
            ForceSmolVLAPolicy._validate_visual_batch(batch)


def test_base_state_dict_transform_is_closed_allowlist():
    source = {
        "model._orig_mod.weight": torch.ones(1),
        **{key: torch.zeros(1) for key in ALLOWED_DROPPED_NORMALIZER_KEYS},
    }
    normalized, dropped = normalize_frozen_base_state_dict(source)
    assert set(normalized) == {"model.weight"}
    assert set(dropped) == ALLOWED_DROPPED_NORMALIZER_KEYS

    source["unknown.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="UNKNOWN_KEYS"):
        normalize_frozen_base_state_dict(source)
