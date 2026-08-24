"""Minimal P0 ForceSmolVLA skeleton with fail-closed RTC isolation."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    SmolVLAPolicy,
    VLAFlowMatching,
    make_att_2d_masks,
    pad_vector,
)
from lerobot.utils.import_utils import require_package
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS

from .configuration_forcesmolvla import (
    FORCE_TOKEN_DENSE_COMPUTE,
    FORCE_TOKEN_DENSE_PARAM,
    FORCE_TOKEN_MOE,
    FORCE_TOKEN_MOE_ADDITIVE,
    SMOLVLA_CARTESIAN7D,
)
from .configuration_forcesmolvla import ForceSmolVLAConfig
from .configuration_forcesmolvla import OFFLINE_FULL_FINETUNE, ONLINE_HIL_VLM_FROZEN
from .action_delta import (
    ActionDeltaProcessor,
    ActionSafetyProfile,
    decode_binary_gripper_width,
)
from .context import ChunkContext
from .force_token import (
    ForceActionAdapter,
    ForceContext,
    PreparedForceContext,
    PreparedForceContextBinding,
    ForceTokenDenseCompute,
    ForceTokenDenseParam,
    ForceTokenMoE,
    fp32_action_projection,
    module_state_sha256,
)
from .prefix import PrefixContext, PrefixLayout, assert_cache_unchanged, clone_cache
from .training_data import RuntimeArtifactBundle


class ActionInferenceError(RuntimeError):
    def __init__(self, code: str, detail: str):
        self.code = code
        super().__init__(f"{code}: {detail}")


def _action_failure_code(error: Exception) -> str:
    candidate = str(error).partition(":")[0]
    if candidate and all(character.isupper() or character.isdigit() or character == "_" for character in candidate):
        return candidate
    if isinstance(error, KeyError):
        return "ACTION_INFERENCE_INPUT_MISSING"
    if isinstance(error, ValueError):
        return "ACTION_INFERENCE_VALUE_INVALID"
    return "ACTION_INFERENCE_FAILED"


class ForceVLAFlowMatching(VLAFlowMatching):
    def __init__(self, config: ForceSmolVLAConfig, rtc_processor=None):
        super().__init__(config, rtc_processor=rtc_processor)
        tokenizer = self.vlm_with_expert.processor.tokenizer
        tokenizer.padding_side = "right"
        tokenizer.truncation_side = "right"
        self.force_branch = None
        self.force_adapter = None
        self.force_initialization_sha256 = None
        if self.config.force_variant in {
            FORCE_TOKEN_DENSE_COMPUTE,
            FORCE_TOKEN_DENSE_PARAM,
            FORCE_TOKEN_MOE,
            FORCE_TOKEN_MOE_ADDITIVE,
        }:
            self._seed_force_initialization(self.config.force_init_seed)
            d_vlm = self.vlm_with_expert.config.text_config.hidden_size
            d_expert = self.vlm_with_expert.expert_hidden_size
            if (d_vlm, d_expert) != (960, 720):
                raise RuntimeError(
                    f"FORCE_WIDTH_CONTRACT_MISMATCH: d_vlm={d_vlm}, d_expert={d_expert}"
                )
            branch_type = {
                FORCE_TOKEN_DENSE_COMPUTE: ForceTokenDenseCompute,
                FORCE_TOKEN_DENSE_PARAM: ForceTokenDenseParam,
                FORCE_TOKEN_MOE: ForceTokenMoE,
                FORCE_TOKEN_MOE_ADDITIVE: ForceTokenMoE,
            }[self.config.force_variant]
            self.force_branch = branch_type(
                d_vlm=d_vlm,
                d_expert=d_expert,
                initialization_seed=self.config.force_init_seed,
            )
            self.force_adapter = ForceActionAdapter(
                d_expert=d_expert,
                horizon=self.config.chunk_size,
                query_mode=self.config.adapter_query_mode,
                initialization_seed=self.config.force_init_seed,
            )
            self.force_initialization_sha256 = module_state_sha256(
                torch.nn.ModuleDict(
                    {"force_branch": self.force_branch, "force_adapter": self.force_adapter}
                )
            )

    @staticmethod
    def _seed_force_initialization(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

    def _force_enabled(self) -> bool:
        if self.config.force_variant == SMOLVLA_CARTESIAN7D:
            return False
        if self.config.force_variant not in {
            FORCE_TOKEN_DENSE_COMPUTE,
            FORCE_TOKEN_DENSE_PARAM,
            FORCE_TOKEN_MOE,
            FORCE_TOKEN_MOE_ADDITIVE,
        }:
            raise RuntimeError(f"UNSUPPORTED_FORCE_VARIANT: {self.config.force_variant!r}")
        if self.force_branch is None or self.force_adapter is None:
            raise RuntimeError("FORCE_MODULES_NOT_INITIALIZED")
        return True

    def parameter_generation(self) -> int:
        """Cheaply fingerprint in-place parameter updates without hashing tensor bytes."""
        digest = hashlib.blake2b(digest_size=8)
        for name, parameter in self.named_parameters():
            digest.update(f"{name}\0{parameter._version}\n".encode())
        return int.from_bytes(digest.digest(), byteorder="big", signed=False)

    def build_force_context(
        self,
        prefix_out: torch.Tensor,
        prefix_valid_mask: torch.Tensor,
        wrench: torch.Tensor | None,
    ) -> ForceContext | None:
        if not self._force_enabled():
            return None
        if wrench is None:
            raise KeyError("observation.wrench is required for ForceToken variants")
        return self.force_branch(prefix_out, prefix_valid_mask, wrench)

    def project_velocity(
        self,
        suffix_out: torch.Tensor,
        x_t: torch.Tensor,
        timestep: torch.Tensor,
        force_context: ForceContext | None,
        *,
        action_feature_mask: torch.Tensor,
        suffix_valid_mask: torch.Tensor,
        prepared_force_binding: PreparedForceContextBinding | None = None,
    ) -> torch.Tensor:
        suffix_out = suffix_out[:, -self.config.chunk_size :].to(dtype=torch.float32)
        if not self._force_enabled():
            return fp32_action_projection(self.action_out_proj, suffix_out, action_feature_mask)
        if force_context is None:
            raise RuntimeError("FORCE_CONTEXT_REQUIRED")
        return self.force_adapter.velocity(
            suffix_out,
            x_t,
            timestep,
            force_context,
            suffix_valid_mask=suffix_valid_mask,
            action_feature_mask=action_feature_mask,
            action_out_proj=self.action_out_proj,
            prepared_binding=prepared_force_binding,
        )

    def _rtc_enabled(self) -> bool:
        if self.config.rtc_config is not None:
            raise RuntimeError("RTC_CONFIG_FORBIDDEN")
        return False

    def embed_suffix(self, noisy_actions, timestep, suffix_valid_mask=None):
        embs, _parent_pad_masks, att_masks = super().embed_suffix(noisy_actions, timestep)
        if suffix_valid_mask is None:
            suffix_valid_mask = torch.ones(
                embs.shape[:2], dtype=torch.bool, device=embs.device
            )
        if suffix_valid_mask.shape != embs.shape[:2]:
            raise ValueError("suffix_valid_mask must have shape [B,H]")
        suffix_valid_mask = suffix_valid_mask.to(device=embs.device, dtype=torch.bool)
        embs = embs * suffix_valid_mask.unsqueeze(-1).to(dtype=embs.dtype)
        return embs, suffix_valid_mask, att_masks

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        *,
        action_feature_mask,
        suffix_valid_mask,
        wrench=None,
        return_force_context: bool = False,
        force_context_override: ForceContext | None = None,
    ):
        if action_feature_mask.shape != actions.shape:
            raise ValueError("action_feature_mask must match padded action shape")
        mask = action_feature_mask.to(device=actions.device, dtype=actions.dtype)
        suffix_valid_mask = suffix_valid_mask.to(device=actions.device, dtype=torch.bool)
        actions = actions * mask
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        if noise.shape != actions.shape:
            raise ValueError("padded training noise must match action shape")
        noise = noise * mask
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = (time_expanded * noise + (1 - time_expanded) * actions) * mask
        u_t = (noise - actions) * mask
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(
            x_t, time, suffix_valid_mask=suffix_valid_mask
        )
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        if force_context_override is None:
            force_context = self.build_force_context(prefix_out, prefix_pad_masks, wrench)
        else:
            if self.config.force_variant not in {FORCE_TOKEN_MOE, FORCE_TOKEN_MOE_ADDITIVE}:
                raise RuntimeError("FORCE_CONTEXT_OVERRIDE_REQUIRES_MOE_VARIANT")
            force_context_override.validate()
            if force_context_override.z_action_fp32.shape[0] != actions.shape[0]:
                raise ValueError("force context override batch mismatch")
            force_context = force_context_override
        velocity = self.project_velocity(
            suffix_out,
            x_t,
            time,
            force_context,
            action_feature_mask=action_feature_mask,
            suffix_valid_mask=suffix_valid_mask,
        )
        losses = F.mse_loss(u_t, velocity, reduction="none") * mask
        return (losses, force_context) if return_force_context else losses

    def prefix_only_force_context(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        wrench,
    ):
        if self.config.force_variant not in {FORCE_TOKEN_MOE, FORCE_TOKEN_MOE_ADDITIVE}:
            raise RuntimeError("PREFIX_ONLY_FORCE_CONTEXT_REQUIRES_MOE_VARIANT")
        prefix_embs, prefix_valid, prefix_attention = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        attention_2d = make_att_2d_masks(prefix_valid, prefix_attention)
        position_ids = torch.cumsum(prefix_valid, dim=1) - 1
        (prefix_out, _), _ = self.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
            fill_kv_cache=True,
        )
        force_context = self.build_force_context(prefix_out, prefix_valid, wrench)
        if force_context is None or force_context.router_state is None:
            raise RuntimeError("MOE_ROUTER_STATE_MISSING")
        return force_context

    def router_pass_a(self, images, img_masks, lang_tokens, lang_masks, state, wrench):
        return self.prefix_only_force_context(
            images, img_masks, lang_tokens, lang_masks, state, wrench
        ).router_state

    def encode_prefix(
        self, images, img_masks, lang_tokens, lang_masks, state, *, audit_cache: bool = True
    ) -> PrefixContext:
        prefix_embs, prefix_valid, prefix_attention = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        layout = PrefixLayout()
        layout.validate()
        if prefix_embs.shape[1] != layout.physical_length:
            raise RuntimeError(
                f"PREFIX_PHYSICAL_LENGTH_MISMATCH: {prefix_embs.shape[1]} != {layout.physical_length}"
            )
        attention_2d = make_att_2d_masks(prefix_valid, prefix_attention)
        position_ids = torch.cumsum(prefix_valid, dim=1) - 1
        (prefix_out, _), cache = self.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
            fill_kv_cache=True,
        )
        segments = layout.segment_ids(device=prefix_embs.device).view(1, -1)
        segments = segments.expand(prefix_embs.shape[0], -1)
        context = PrefixContext(
            prefix_out=prefix_out,
            prefix_valid_mask=prefix_valid,
            prefix_segment_ids=segments,
            prefix_position_ids=position_ids,
            layout=layout,
            past_key_values=cache,
            cache_snapshot=clone_cache(cache) if audit_cache else None,
        )
        context.validate(check_cache=audit_cache)
        return context

    def velocity_full(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        x_t,
        timestep,
        *,
        action_feature_mask,
        suffix_valid_mask,
        wrench=None,
    ) -> torch.Tensor:
        mask = action_feature_mask.to(device=x_t.device, dtype=x_t.dtype)
        x_t = x_t * mask
        prefix_embs, prefix_valid, prefix_attention = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_valid, suffix_attention = self.embed_suffix(
            x_t, timestep, suffix_valid_mask=suffix_valid_mask
        )
        valid = torch.cat([prefix_valid, suffix_valid], dim=1)
        attention = torch.cat([prefix_attention, suffix_attention], dim=1)
        attention_2d = make_att_2d_masks(valid, attention)
        position_ids = torch.cumsum(valid, dim=1) - 1
        (prefix_out, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        force_context = self.build_force_context(prefix_out, prefix_valid, wrench)
        return self.project_velocity(
            suffix_out,
            x_t,
            timestep,
            force_context,
            action_feature_mask=action_feature_mask,
            suffix_valid_mask=suffix_valid_mask,
        )

    def velocity_cached(
        self,
        context: PrefixContext,
        x_t,
        timestep,
        *,
        action_feature_mask,
        suffix_valid_mask,
        force_context: ForceContext | PreparedForceContext | None = None,
        force_context_binding: PreparedForceContextBinding | None = None,
        audit_cache: bool = False,
    ) -> torch.Tensor:
        context.validate(check_cache=False)
        if x_t.shape[0] != context.prefix_valid_mask.shape[0]:
            raise ValueError("PrefixContext batch size mismatch")
        mask = action_feature_mask.to(device=x_t.device, dtype=x_t.dtype)
        x_t = x_t * mask
        suffix_embs, suffix_valid, suffix_attention = self.embed_suffix(
            x_t, timestep, suffix_valid_mask=suffix_valid_mask
        )
        batch_size, suffix_length = suffix_valid.shape
        prefix_to_suffix = suffix_valid[:, :, None] & context.prefix_valid_mask[:, None, :]
        suffix_2d = make_att_2d_masks(suffix_valid, suffix_attention)
        attention_2d = torch.cat([prefix_to_suffix, suffix_2d], dim=2)
        prefix_valid_count = context.prefix_valid_mask.sum(dim=-1, keepdim=True)
        position_ids = prefix_valid_count + torch.cumsum(suffix_valid, dim=1) - 1
        outputs, _ = self.vlm_with_expert.forward(
            attention_mask=attention_2d,
            position_ids=position_ids,
            past_key_values=context.past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        if audit_cache:
            if context.cache_snapshot is None:
                raise RuntimeError("PREFIX_CACHE_AUDIT_NOT_ENABLED")
            assert_cache_unchanged(
                context.past_key_values,
                context.cache_snapshot,
                physical_length=context.layout.physical_length,
            )
        suffix_out = outputs[1][:, -suffix_length:]
        if isinstance(force_context, PreparedForceContext):
            if force_context_binding is None:
                raise RuntimeError("PREPARED_FORCE_CONTEXT_BINDING_REQUIRED")
            if force_context_binding.model_generation != self.parameter_generation():
                raise RuntimeError("PREPARED_FORCE_CONTEXT_MODEL_GENERATION_STALE")
        return self.project_velocity(
            suffix_out,
            x_t,
            timestep,
            force_context,
            action_feature_mask=action_feature_mask,
            suffix_valid_mask=suffix_valid_mask,
            prepared_force_binding=force_context_binding,
        )

    def sample_actions_masked(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise32,
        *,
        action_feature_mask,
        suffix_valid_mask,
        wrench=None,
        force_context_binding: PreparedForceContextBinding | None = None,
        audit_cache: bool = False,
    ) -> torch.Tensor:
        context = self.encode_prefix(
            images, img_masks, lang_tokens, lang_masks, state, audit_cache=audit_cache
        )
        force_context = self.build_force_context(
            context.prefix_out, context.prefix_valid_mask, wrench
        )
        if force_context is not None and force_context_binding is None:
            raise RuntimeError("PREPARED_FORCE_CONTEXT_BINDING_REQUIRED")
        prepared_force_context = (
            None
            if force_context is None
            else self.force_adapter.cross_attention.prepare(
                force_context, binding=force_context_binding
            )
        )
        mask = action_feature_mask.to(device=noise32.device, dtype=noise32.dtype)
        x_t = noise32 * mask
        dt = -1.0 / self.config.num_steps
        for step in range(self.config.num_steps):
            timestep = torch.full(
                (noise32.shape[0],),
                1.0 + step * dt,
                dtype=torch.float32,
                device=noise32.device,
            )
            velocity = self.velocity_cached(
                context,
                x_t,
                timestep,
                action_feature_mask=action_feature_mask,
                suffix_valid_mask=suffix_valid_mask,
                force_context=prepared_force_context,
                force_context_binding=force_context_binding,
            )
            x_t = (x_t + dt * velocity) * mask
        if audit_cache:
            assert_cache_unchanged(
                context.past_key_values,
                context.cache_snapshot,
                physical_length=context.layout.physical_length,
            )
        return x_t


class ForceSmolVLAPolicy(SmolVLAPolicy):
    config_class = ForceSmolVLAConfig
    name = "force_smolvla"

    def __init__(self, config: ForceSmolVLAConfig, **kwargs: Any):
        require_package("transformers", extra="smolvla")
        PreTrainedPolicy.__init__(self, config)
        config.validate_features()
        self.config = config
        self._context_generation = -1
        self._consumed_chunk_ids: set[str] = set()
        self._runtime_artifacts: RuntimeArtifactBundle | None = None
        self._action_safety_profile: ActionSafetyProfile | None = None
        self._expected_action_delta_spec_sha256: str | None = None
        self._expected_normalizer_manifest_sha256: str | None = None
        self.init_rtc_processor()
        self.model = ForceVLAFlowMatching(config, rtc_processor=None)
        self.apply_training_stage()
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_name_or_path: str | Path,
        *,
        config=None,
        force_download: bool = False,
        local_files_only: bool = True,
        revision: str | None = None,
        strict: bool = True,
        artifact_use: str = "development",
        resume_download=None,
        proxies=None,
        token=None,
        cache_dir=None,
        **kwargs,
    ):
        if any(
            value is not None
            for value in (resume_download, proxies, token, cache_dir)
        ) or kwargs:
            raise RuntimeError("FORCE_CHECKPOINT_REMOTE_OR_UNKNOWN_ARGUMENT_FORBIDDEN")
        from .checkpoint import (
            prepare_strict_force_config,
            resolve_local_force_checkpoint_dir,
            validate_trainability_manifest,
        )

        checkpoint_dir = resolve_local_force_checkpoint_dir(
            pretrained_name_or_path,
            force_download=force_download,
            local_files_only=local_files_only,
            strict=strict,
            revision=revision,
            config=config,
        )
        resolved_config = prepare_strict_force_config(
            checkpoint_dir, artifact_use=artifact_use
        )
        policy = super().from_pretrained(
            checkpoint_dir,
            config=resolved_config,
            force_download=False,
            local_files_only=True,
            revision=None,
            strict=True,
        )
        validate_trainability_manifest(policy, checkpoint_dir)
        manifest = json.loads(
            (checkpoint_dir / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        payloads = manifest["payloads"]
        policy._expected_action_delta_spec_sha256 = payloads[
            "manifests/action_delta_spec.json"
        ]["sha256"]
        policy._expected_normalizer_manifest_sha256 = payloads[
            "manifests/normalizer_manifest.json"
        ]["sha256"]
        return policy

    def apply_training_stage(self) -> None:
        """Set the only two v4.1 trainable-parameter modes explicitly."""
        for parameter in self.parameters():
            parameter.requires_grad_(True)
        if self.config.training_stage == ONLINE_HIL_VLM_FROZEN:
            for parameter in self.model.vlm_with_expert.vlm.parameters():
                parameter.requires_grad_(False)
        elif self.config.training_stage != OFFLINE_FULL_FINETUNE:
            raise RuntimeError(f"UNSUPPORTED_TRAINING_STAGE: {self.config.training_stage}")

    def force_initialization_state_keys(self) -> tuple[str, ...]:
        if self.config.force_variant not in {
            FORCE_TOKEN_DENSE_COMPUTE,
            FORCE_TOKEN_DENSE_PARAM,
            FORCE_TOKEN_MOE,
            FORCE_TOKEN_MOE_ADDITIVE,
        }:
            return ()
        prefixes = ("model.force_branch.", "model.force_adapter.")
        return tuple(sorted(key for key in self.state_dict() if key.startswith(prefixes)))

    def force_initialization_tensor_hash(self) -> str | None:
        return self.model.force_initialization_sha256

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.config.training_stage == ONLINE_HIL_VLM_FROZEN:
            self.model.vlm_with_expert.vlm.eval()
        return self

    def reset(self):
        super().reset()
        self._context_generation = getattr(self, "_context_generation", -1) + 1
        self._consumed_chunk_ids = set()

    def bind_runtime_artifacts(self, artifacts: RuntimeArtifactBundle) -> None:
        if not isinstance(artifacts, RuntimeArtifactBundle):
            raise TypeError("runtime artifacts must be a RuntimeArtifactBundle")
        artifacts.validate_action_contract()
        if (
            self._expected_action_delta_spec_sha256 is not None
            and artifacts.action_delta_spec_sha256
            != self._expected_action_delta_spec_sha256
        ):
            raise RuntimeError("CHECKPOINT_RUNTIME_ACTION_DELTA_SPEC_HASH_MISMATCH")
        if (
            self._expected_normalizer_manifest_sha256 is not None
            and artifacts.normalizer_manifest_sha256
            != self._expected_normalizer_manifest_sha256
        ):
            raise RuntimeError("CHECKPOINT_RUNTIME_NORMALIZER_HASH_MISMATCH")
        self._runtime_artifacts = artifacts

    def bind_action_safety_rules(self, rulespec: dict, *, rules_sha256: str) -> None:
        self._action_safety_profile = ActionSafetyProfile.from_rulespec(
            rulespec, rules_sha256=rules_sha256
        )

    @classmethod
    def supports_rtc(cls) -> bool:
        return False

    def init_rtc_processor(self) -> None:
        if self.config.rtc_config is not None:
            raise RuntimeError("RTC_CONFIG_FORBIDDEN")
        self.rtc_processor = None

    def _rtc_enabled(self) -> bool:
        if self.config.rtc_config is not None:
            raise RuntimeError("RTC_CONFIG_FORBIDDEN")
        return False

    @staticmethod
    def _validate_visual_batch(batch) -> None:
        expected = ("observation.images.camera1", "observation.images.camera2")
        actual = tuple(
            key
            for key in batch
            if key.startswith("observation.images.") and not key.endswith("_padding_mask")
        )
        if actual != expected:
            raise ValueError(f"expected exact ordered visual batch keys {expected}, got {actual}")

    def prepare_images(self, batch):
        self._validate_visual_batch(batch)
        return super().prepare_images(batch)

    def _prepare_wrench(self, batch, *, device) -> torch.Tensor | None:
        if getattr(self.config, "force_variant", SMOLVLA_CARTESIAN7D) == SMOLVLA_CARTESIAN7D:
            return None
        if "observation.wrench" not in batch:
            raise KeyError("observation.wrench is required for ForceToken variants")
        wrench = batch["observation.wrench"].to(device=device, dtype=torch.float32)
        if wrench.ndim != 2 or wrench.shape[-1] != 6:
            raise ValueError("observation.wrench must have shape [B,6]")
        if not torch.all(torch.isfinite(wrench)):
            raise ValueError("observation.wrench contains nonfinite values")
        return wrench

    def select_action(self, *args: Any, **kwargs: Any):
        raise RuntimeError("SELECT_ACTION_FORBIDDEN_READ_ONLY_OFFLINE_PROJECT")

    @torch.inference_mode()
    def _predict_normalized_delta_chunk(
        self,
        batch,
        chunk_context: ChunkContext | None = None,
        noise=None,
        **kwargs,
    ):
        forbidden = {
            "inference_delay",
            "prev_chunk_left_over",
            "execution_horizon",
        } & kwargs.keys()
        if forbidden:
            raise TypeError(f"RTC kwargs are forbidden: {sorted(forbidden)}")
        if kwargs:
            raise TypeError(f"unexpected inference kwargs: {sorted(kwargs)}")
        if chunk_context is None:
            raise TypeError("ChunkContext is required")
        self._validate_visual_batch(batch)
        batch_size = batch["observation.state"].shape[0]
        chunk_context.validate(
            batch_size=batch_size,
            horizon=self.config.chunk_size,
            policy_generation=self._context_generation,
        )
        if "raw_state_snapshot" not in batch:
            raise KeyError("raw_state_snapshot is required for ChunkContext binding")
        raw_snapshot = batch["raw_state_snapshot"].detach().cpu()
        if not torch.equal(raw_snapshot, chunk_context.raw_state_snapshot.detach().cpu()):
            raise RuntimeError("CHUNK_CONTEXT_RAW_STATE_MISMATCH")
        if not torch.all(chunk_context.wrench_geometry_valid):
            raise RuntimeError("WRENCH_GEOMETRY_INVALID")
        if not torch.all(chunk_context.runtime_artifact_compatible):
            raise RuntimeError("CALIBRATION_NORMALIZER_INCOMPATIBLE")
        reused = self._consumed_chunk_ids & set(chunk_context.registry_key)
        if reused:
            raise RuntimeError(f"CHUNK_CONTEXT_ALREADY_CONSUMED: {sorted(reused)}")

        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        wrench = self._prepare_wrench(batch, device=state.device)
        state_mask = (torch.arange(32, device=state.device) < 7).view(1, 32)
        state = state * state_mask.to(dtype=state.dtype)
        suffix_valid = chunk_context.suffix_valid_mask.to(device=state.device)
        feature_mask = suffix_valid.unsqueeze(-1) & (
            torch.arange(32, device=state.device).view(1, 1, 32) < 7
        )
        if noise is None:
            noise7 = torch.randn(
                batch_size, self.config.chunk_size, 7, device=state.device, dtype=torch.float32
            )
            noise32 = pad_vector(noise7, 32)
        elif isinstance(noise, int):
            generator = torch.Generator(device=state.device).manual_seed(noise)
            noise7 = torch.randn(
                batch_size,
                self.config.chunk_size,
                7,
                generator=generator,
                device=state.device,
                dtype=torch.float32,
            )
            noise32 = pad_vector(noise7, 32)
        elif isinstance(noise, torch.Tensor) and noise.shape == (
            batch_size,
            self.config.chunk_size,
            7,
        ):
            noise32 = pad_vector(noise.to(device=state.device), 32)
        elif isinstance(noise, torch.Tensor) and noise.shape == (
            batch_size,
            self.config.chunk_size,
            32,
        ) and not torch.any(noise[..., 7:] != 0):
            noise32 = noise.to(device=state.device)
        else:
            raise ValueError("noise must be a seed, 7D tensor, or exactly zero-padded 32D tensor")
        noise32 = noise32 * feature_mask.to(dtype=noise32.dtype)
        actions32 = self.model.sample_actions_masked(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            noise32,
            action_feature_mask=feature_mask,
            suffix_valid_mask=suffix_valid,
            wrench=wrench,
            force_context_binding=PreparedForceContextBinding(
                chunk_id=chunk_context.chunk_id,
                sample_id=chunk_context.sample_id,
                context_generation=self._context_generation,
                model_generation=self.model.parameter_generation(),
                device=state.device,
                dtype=torch.float32,
            ),
        )
        normalized_delta7 = actions32[..., :7].to(dtype=torch.float32)
        if normalized_delta7.shape != (batch_size, self.config.chunk_size, 7):
            raise RuntimeError("NORMALIZED_DELTA_OUTPUT_SHAPE_MISMATCH")
        return normalized_delta7

    @torch.inference_mode()
    def _predict_action_chunks(
        self,
        batch,
        chunk_context: ChunkContext | None = None,
        noise=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.training:
            raise RuntimeError("PREDICT_ACTION_CHUNK_REQUIRES_EVAL_MODE")
        if self._runtime_artifacts is None:
            raise RuntimeError("RUNTIME_ARTIFACTS_NOT_BOUND")
        self._runtime_artifacts.validate_action_contract()
        if self._action_safety_profile is None:
            raise RuntimeError("ACTION_SAFETY_RULES_NOT_BOUND")
        normalized_delta7 = self._predict_normalized_delta_chunk(
            batch, chunk_context=chunk_context, noise=noise, **kwargs
        )
        assert chunk_context is not None  # validated by the private path
        self._runtime_artifacts.validate_context_hashes(chunk_context)
        normalized_numpy = (
            normalized_delta7.detach().cpu().to(torch.float32).numpy().astype(np.float64)
        )
        unnormalized_delta7 = self._runtime_artifacts.normalizer.delta_action7.inverse(
            normalized_numpy
        )
        unnormalized_delta7 = decode_binary_gripper_width(unnormalized_delta7)
        raw_state7 = chunk_context.raw_state_snapshot.detach().cpu().numpy().astype(np.float64)
        absolute7 = ActionDeltaProcessor.from_delta(unnormalized_delta7, raw_state7)
        self._action_safety_profile.validate_chunk(
            absolute7,
            chunk_context.action_valid_mask.detach().cpu().numpy(),
            raw_state7,
        )
        valid_mask = chunk_context.action_valid_mask.detach().cpu().numpy()[..., None]
        absolute7 = np.where(valid_mask, absolute7, 0.0)
        absolute_tensor = torch.from_numpy(np.ascontiguousarray(absolute7)).to(
            device=normalized_delta7.device, dtype=torch.float32
        )
        if not torch.all(torch.isfinite(absolute_tensor)):
            raise RuntimeError("ABSOLUTE_ACTION_NONFINITE_AFTER_INVERSE")
        self._consumed_chunk_ids.update(chunk_context.registry_key)
        return normalized_delta7, absolute_tensor

    @torch.inference_mode()
    def predict_action_chunk(
        self,
        batch,
        chunk_context: ChunkContext | None = None,
        noise=None,
        **kwargs,
    ) -> torch.Tensor:
        """Return executable v4.2 absolute TCP target + gripper-width chunks."""
        self.eval()
        try:
            _normalized, absolute = self._predict_action_chunks(
                batch, chunk_context=chunk_context, noise=noise, **kwargs
            )
        except ActionInferenceError:
            raise
        except Exception as error:
            raise ActionInferenceError(_action_failure_code(error), str(error)) from error
        return absolute

    @staticmethod
    def _action_masks(batch, *, horizon: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        if "action_valid_mask" not in batch:
            raise KeyError("action_valid_mask is required")
        suffix_valid = batch["action_valid_mask"].to(device=device, dtype=torch.bool)
        if suffix_valid.ndim != 2 or suffix_valid.shape[1] != horizon:
            raise ValueError(f"action_valid_mask must have shape [B,{horizon}]")
        active = torch.arange(32, device=device) < 7
        feature = suffix_valid.unsqueeze(-1) & active.view(1, 1, 32)
        return feature, suffix_valid

    def _flow_training_outputs(
        self,
        batch,
        noise,
        time,
        *,
        return_force_context: bool,
        exact_router_replay: bool = False,
    ):
        if exact_router_replay and not return_force_context:
            raise ValueError("exact_router_replay requires return_force_context=True")
        self._validate_visual_batch(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        wrench = self._prepare_wrench(batch, device=state.device)
        if state.shape[-1] != 32:
            raise AssertionError("state must be padded to 32D")
        state_feature_mask = (torch.arange(32, device=state.device) < 7).view(1, 32)
        state = state * state_feature_mask.to(dtype=state.dtype)
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        feature_mask, suffix_valid = self._action_masks(
            batch, horizon=actions.shape[1], device=actions.device
        )
        if noise is not None:
            if noise.shape[-1] == 7:
                noise = pad_vector(noise, self.config.max_action_dim)
            elif noise.shape[-1] != 32 or torch.any(noise[..., 7:] != 0):
                raise ValueError("noise must be 7D or an exactly zero-padded 32D equivalent")
        model_kwargs = {
            "action_feature_mask": feature_mask,
            "suffix_valid_mask": suffix_valid,
            "wrench": wrench,
        }
        if return_force_context:
            model_kwargs["return_force_context"] = True
            if exact_router_replay and self.config.force_variant in {
                FORCE_TOKEN_MOE,
                FORCE_TOKEN_MOE_ADDITIVE,
            }:
                model_kwargs["force_context_override"] = self.model.prefix_only_force_context(
                    images,
                    img_masks,
                    batch[OBS_LANGUAGE_TOKENS],
                    batch[OBS_LANGUAGE_ATTENTION_MASK],
                    state,
                    wrench,
                )
        result = self.model.forward(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            actions,
            noise,
            time,
            **model_kwargs,
        )
        if return_force_context:
            losses, force_context = result
        else:
            losses, force_context = result, None
        return losses, feature_mask, force_context

    def router_pass_a(self, batch):
        self._validate_visual_batch(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        wrench = self._prepare_wrench(batch, device=state.device)
        if state.shape[-1] != 32:
            raise AssertionError("state must be padded to 32D")
        state_mask = (torch.arange(32, device=state.device) < 7).view(1, 32)
        state = state * state_mask.to(dtype=state.dtype)
        return self.model.router_pass_a(
            images,
            img_masks,
            batch[OBS_LANGUAGE_TOKENS],
            batch[OBS_LANGUAGE_ATTENTION_MASK],
            state,
            wrench,
        )

    def forward_training_terms(self, batch, *, noise, time):
        if self.config.force_variant not in {FORCE_TOKEN_MOE, FORCE_TOKEN_MOE_ADDITIVE}:
            raise RuntimeError("TWO_PASS_TERMS_REQUIRE_MOE_VARIANT")
        losses, feature_mask, force_context = self._flow_training_outputs(
            batch,
            noise,
            time,
            return_force_context=True,
            exact_router_replay=True,
        )
        if force_context is None or force_context.router_state is None:
            raise RuntimeError("MOE_ROUTER_STATE_MISSING")
        return losses, feature_mask, force_context.router_state

    def forward_single_pass_training_terms(self, batch, *, noise, time):
        """Return flow and router terms from one shared full-prefix forward."""
        if self.config.force_variant not in {FORCE_TOKEN_MOE, FORCE_TOKEN_MOE_ADDITIVE}:
            raise RuntimeError("SINGLE_PASS_TERMS_REQUIRE_MOE_VARIANT")
        losses, feature_mask, force_context = self._flow_training_outputs(
            batch,
            noise,
            time,
            return_force_context=True,
            exact_router_replay=False,
        )
        if force_context is None or force_context.router_state is None:
            raise RuntimeError("MOE_ROUTER_STATE_MISSING")
        return losses, feature_mask, force_context.router_state

    def forward(self, batch, noise=None, time=None, reduction: str = "mean"):
        if reduction not in {"mean", "none"}:
            raise ValueError("reduction must be 'mean' or 'none'")
        losses, feature_mask, _ = self._flow_training_outputs(
            batch, noise, time, return_force_context=False
        )
        denominator_per_sample = feature_mask.sum(dim=(1, 2)).clamp_min(1)
        per_sample = losses.sum(dim=(1, 2)) / denominator_per_sample
        global_denominator = feature_mask.sum().clamp_min(1)
        loss = losses.sum() / global_denominator if reduction == "mean" else per_sample
        return loss, {
            "loss": float((loss.mean() if loss.ndim else loss).detach()),
            "valid_feature_tokens": int(feature_mask.sum().detach()),
            "reduction": reduction,
        }
