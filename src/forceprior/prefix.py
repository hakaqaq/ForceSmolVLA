"""Fixed physical prefix layout and immutable cache contract for Cartesian7D."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PrefixSpan:
    name: str
    start: int
    stop: int
    segment_id: int

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass(frozen=True)
class PrefixLayout:
    camera1: PrefixSpan = PrefixSpan("camera1", 0, 64, 0)
    camera2: PrefixSpan = PrefixSpan("camera2", 64, 128, 1)
    language: PrefixSpan = PrefixSpan("language", 128, 176, 2)
    state: PrefixSpan = PrefixSpan("state", 176, 177, 3)
    physical_length: int = 177

    def segment_ids(self, *, device) -> torch.Tensor:
        result = torch.empty(self.physical_length, dtype=torch.int64, device=device)
        for span in (self.camera1, self.camera2, self.language, self.state):
            result[span.start : span.stop] = span.segment_id
        return result

    def validate(self) -> None:
        spans = (self.camera1, self.camera2, self.language, self.state)
        cursor = 0
        for span in spans:
            if span.start != cursor or span.stop <= span.start:
                raise ValueError("PrefixLayout spans must be contiguous and nonempty")
            cursor = span.stop
        if cursor != self.physical_length:
            raise ValueError("PrefixLayout physical length mismatch")


def clone_cache(cache: dict) -> dict:
    return {
        int(layer): {
            name: tensor.detach().clone()
            for name, tensor in tensors.items()
        }
        for layer, tensors in cache.items()
    }


def assert_cache_unchanged(cache: dict, snapshot: dict, *, physical_length: int) -> None:
    if set(cache) != set(snapshot):
        raise RuntimeError("PREFIX_CACHE_LAYER_SET_MUTATED")
    for layer in cache:
        if set(cache[layer]) != {"key_states", "value_states"}:
            raise RuntimeError(f"PREFIX_CACHE_FIELDS_INVALID: layer={layer}")
        for name in ("key_states", "value_states"):
            current = cache[layer][name]
            frozen = snapshot[layer][name]
            if current.shape[1] != physical_length or current.shape != frozen.shape:
                raise RuntimeError(f"PREFIX_CACHE_PHYSICAL_LENGTH_MUTATED: layer={layer}/{name}")
            if not torch.equal(current, frozen):
                raise RuntimeError(f"PREFIX_CACHE_CONTENT_MUTATED: layer={layer}/{name}")


@dataclass(frozen=True)
class PrefixContext:
    prefix_out: torch.Tensor
    prefix_valid_mask: torch.Tensor
    prefix_segment_ids: torch.Tensor
    prefix_position_ids: torch.Tensor
    layout: PrefixLayout
    past_key_values: dict
    cache_snapshot: dict | None

    def validate(self, *, check_cache: bool = True) -> None:
        self.layout.validate()
        batch, physical, _hidden = self.prefix_out.shape
        if physical != self.layout.physical_length:
            raise ValueError("prefix_out physical length mismatch")
        if self.prefix_valid_mask.shape != (batch, physical):
            raise ValueError("prefix_valid_mask shape mismatch")
        if self.prefix_position_ids.shape != (batch, physical):
            raise ValueError("prefix_position_ids shape mismatch")
        if self.prefix_segment_ids.shape != (batch, physical):
            raise ValueError("prefix_segment_ids shape mismatch")
        if check_cache:
            if self.cache_snapshot is None:
                raise RuntimeError("PREFIX_CACHE_AUDIT_NOT_ENABLED")
            assert_cache_unchanged(
                self.past_key_values,
                self.cache_snapshot,
                physical_length=self.layout.physical_length,
            )
