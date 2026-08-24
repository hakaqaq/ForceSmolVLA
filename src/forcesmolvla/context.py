"""Batch-bound, reset-invalidated context for the only offline chunk API."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ChunkContext:
    policy_generation: int
    raw_state_snapshot: torch.Tensor
    t_ref_ns: torch.Tensor
    tau0_ns: torch.Tensor
    clock_domain_id: tuple[str, ...]
    episode_id: tuple[str, ...]
    session_id: tuple[str, ...]
    sample_id: tuple[str, ...]
    chunk_id: tuple[str, ...]
    action_valid_mask: torch.Tensor
    suffix_valid_mask: torch.Tensor
    calibration_bundle_hash: tuple[str, ...]
    wrench_geometry_spec_hash: tuple[str, ...]
    normalizer_hash: tuple[str, ...]
    calibration_mapping_hash_or_none: tuple[str | None, ...]
    wrench_geometry_valid: torch.Tensor
    runtime_artifact_compatible: torch.Tensor
    selected_provenance: tuple[dict, ...]

    def validate(self, *, batch_size: int, horizon: int, policy_generation: int) -> None:
        if self.policy_generation != policy_generation:
            raise RuntimeError("CHUNK_CONTEXT_INVALIDATED_BY_RESET")
        if self.raw_state_snapshot.shape != (batch_size, 7):
            raise ValueError("raw_state_snapshot must have shape [B,7]")
        for name, value in (("t_ref_ns", self.t_ref_ns), ("tau0_ns", self.tau0_ns)):
            if value.shape != (batch_size,) or value.dtype not in (torch.int64, torch.long):
                raise ValueError(f"{name} must have int64 shape [B]")
        for name, value in (
            ("action_valid_mask", self.action_valid_mask),
            ("suffix_valid_mask", self.suffix_valid_mask),
        ):
            if value.shape != (batch_size, horizon) or value.dtype != torch.bool:
                raise ValueError(f"{name} must have bool shape [B,H]")
        if not torch.equal(self.action_valid_mask, self.suffix_valid_mask):
            raise ValueError("action_valid_mask and suffix_valid_mask must be identical")
        if torch.any(self.action_valid_mask.sum(dim=1) < 3):
            raise ValueError("each chunk must contain at least three valid labels")
        counts = self.action_valid_mask.sum(dim=1)
        expected = (
            torch.arange(horizon, device=self.action_valid_mask.device).unsqueeze(0)
            < counts.unsqueeze(1)
        )
        if not torch.equal(self.action_valid_mask, expected):
            raise ValueError("action_valid_mask must be physically right-padded")
        for name, value in (
            ("wrench_geometry_valid", self.wrench_geometry_valid),
            ("runtime_artifact_compatible", self.runtime_artifact_compatible),
        ):
            if value.shape != (batch_size,) or value.dtype != torch.bool:
                raise ValueError(f"{name} must have bool shape [B]")
        string_fields = (
            self.clock_domain_id,
            self.episode_id,
            self.session_id,
            self.sample_id,
            self.chunk_id,
            self.calibration_bundle_hash,
            self.wrench_geometry_spec_hash,
            self.normalizer_hash,
            self.calibration_mapping_hash_or_none,
            self.selected_provenance,
        )
        if any(len(value) != batch_size for value in string_fields):
            raise ValueError("ChunkContext sequence fields must be batch-bound")
        if len(set(self.chunk_id)) != batch_size:
            raise ValueError("chunk_id values must be unique within a batch")
        for hashes in (
            self.calibration_bundle_hash,
            self.wrench_geometry_spec_hash,
            self.normalizer_hash,
        ):
            for digest in hashes:
                if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                    raise ValueError("ChunkContext compatibility hashes must be lowercase SHA256")

    @property
    def registry_key(self) -> tuple[str, ...]:
        return self.chunk_id
