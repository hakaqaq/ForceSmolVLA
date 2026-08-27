"""Synthetic-only causal Reward Detector primitives for R0 preparation."""

from __future__ import annotations

from dataclasses import dataclass
import math


def align_confirmation_to_policy_boundary(
    confirmation_frame: int,
    *,
    data_hz: int = 30,
    policy_hz: int = 10,
    anchor_frame: int = 0,
) -> tuple[int, int]:
    if confirmation_frame < anchor_frame:
        raise ValueError("confirmation frame precedes policy anchor")
    if data_hz <= 0 or policy_hz <= 0 or data_hz % policy_hz != 0:
        raise ValueError("data_hz must be a positive integer multiple of policy_hz")
    stride = data_hz // policy_hz
    offset = confirmation_frame - anchor_frame
    aligned = anchor_frame + math.ceil(offset / stride) * stride
    delay = aligned - confirmation_frame
    if delay < 0 or delay >= stride:
        raise AssertionError("causal policy-boundary alignment violation")
    return aligned, delay


@dataclass(frozen=True)
class DetectionEvent:
    episode_id: str
    confirmation_frame: int
    aligned_policy_frame: int
    alignment_delay_frames: int


class SyntheticCausalRewardDetector:
    """Consecutive-positive detector that cannot be used with real parameters."""

    def __init__(self, *, probability_threshold: float, consecutive_frames: int) -> None:
        if not 0.0 < probability_threshold < 1.0:
            raise ValueError("synthetic probability threshold must be in (0, 1)")
        if consecutive_frames < 1:
            raise ValueError("synthetic consecutive frame count must be positive")
        self._threshold = probability_threshold
        self._required = consecutive_frames
        self._episode_id: str | None = None
        self._last_frame: int | None = None
        self._streak = 0
        self._event: DetectionEvent | None = None

    def reset(self, episode_id: str) -> None:
        if not episode_id:
            raise ValueError("episode_id must be non-empty")
        self._episode_id = episode_id
        self._last_frame = None
        self._streak = 0
        self._event = None

    def update(self, *, episode_id: str, frame_index: int, probability: float) -> DetectionEvent | None:
        if episode_id != self._episode_id:
            self.reset(episode_id)
        if frame_index < 0 or not 0.0 <= probability <= 1.0:
            raise ValueError("invalid frame index or probability")
        if self._last_frame is not None and frame_index != self._last_frame + 1:
            raise ValueError("detector requires strictly consecutive causal frames")
        self._last_frame = frame_index
        if self._event is not None:
            return self._event
        self._streak = self._streak + 1 if probability >= self._threshold else 0
        if self._streak == self._required:
            aligned, delay = align_confirmation_to_policy_boundary(frame_index)
            self._event = DetectionEvent(episode_id, frame_index, aligned, delay)
        return self._event
