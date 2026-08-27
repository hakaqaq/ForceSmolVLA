"""Strict current-frame LeRobot-v3 to ConRFT observation adapter.

This module owns only shape/order/provenance adaptation. ConRFT's frozen
ResNet-10 source owns resize and ImageNet normalization. No image is copied
back into the immutable LeRobot dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


SOURCE_CAMERA_KEYS = (
    "observation.images.camera1",
    "observation.images.camera2",
)
CLASSIFIER_CAMERA_KEYS = ("d435_third_person", "d405_wrist")
SOURCE_SHAPE = (3, 480, 640)
CLASSIFIER_SHAPE = (1, 1, 480, 640, 3)
FRAME_STACK = 1


@dataclass(frozen=True)
class RowReference:
    dataset_root_id: str
    parquet_relative_path: str
    row_index: int
    episode_id: str
    frame_index: int
    timestamp: float

    def __post_init__(self) -> None:
        if not self.dataset_root_id or not self.parquet_relative_path or not self.episode_id:
            raise ValueError("row reference identifiers must be non-empty")
        if self.row_index < 0 or self.frame_index < 0 or self.timestamp < 0:
            raise ValueError("row reference indices and timestamp must be non-negative")


@dataclass(frozen=True)
class CameraRowIdentity:
    camera1_receive_monotonic_ns: int
    camera2_receive_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.camera1_receive_monotonic_ns < 0 or self.camera2_receive_monotonic_ns < 0:
            raise ValueError("camera row timestamps must be non-negative")


@dataclass(frozen=True)
class AdaptedObservation:
    observation: dict[str, np.ndarray]
    row_reference: RowReference
    camera_row_identity: CameraRowIdentity
    reset_at_episode_boundary: bool


def _validate_rgb_chw(name: str, image: object, *, source_color_order: str) -> np.ndarray:
    if source_color_order != "RGB":
        raise ValueError(f"{name}: expected explicit RGB source, got {source_color_order!r}")
    if not isinstance(image, np.ndarray):
        raise TypeError(f"{name}: expected numpy.ndarray, got {type(image).__name__}")
    if image.dtype != np.uint8:
        raise TypeError(f"{name}: expected uint8, got {image.dtype}")
    if image.shape != SOURCE_SHAPE:
        raise ValueError(f"{name}: expected CHW shape {SOURCE_SHAPE}, got {image.shape}")
    return image


def chw_rgb_to_conrft_bthwc(image: np.ndarray) -> np.ndarray:
    """Convert one validated RGB CHW frame to ConRFT B,T,H,W,C."""

    hwc = np.transpose(image, (1, 2, 0))
    output = np.ascontiguousarray(hwc)[None, None, ...]
    if output.shape != CLASSIFIER_SHAPE or output.dtype != np.uint8:
        raise AssertionError("internal classifier tensor contract violation")
    return output


class ConRFTLeRobotV3Adapter:
    """Stateless-image adapter with explicit episode reset accounting.

    Only the active episode id and reset count are retained. No image, frame,
    stack, probability, or classifier state survives an invocation.
    """

    def __init__(self) -> None:
        self._active_episode_id: str | None = None
        self._episode_reset_count = 0

    @property
    def episode_reset_count(self) -> int:
        return self._episode_reset_count

    @property
    def retained_frame_count(self) -> int:
        return 0

    def adapt(
        self,
        sample: Mapping[str, object],
        *,
        row_reference: RowReference,
        camera_row_identity: CameraRowIdentity,
        source_color_order: str = "RGB",
    ) -> AdaptedObservation:
        if tuple(sample.keys()) != SOURCE_CAMERA_KEYS:
            raise ValueError(
                f"camera keys/order must be exactly {SOURCE_CAMERA_KEYS}, got {tuple(sample.keys())}"
            )
        reset = row_reference.episode_id != self._active_episode_id
        if reset:
            self._active_episode_id = row_reference.episode_id
            self._episode_reset_count += 1

        camera1 = _validate_rgb_chw(
            SOURCE_CAMERA_KEYS[0], sample[SOURCE_CAMERA_KEYS[0]], source_color_order=source_color_order
        )
        camera2 = _validate_rgb_chw(
            SOURCE_CAMERA_KEYS[1], sample[SOURCE_CAMERA_KEYS[1]], source_color_order=source_color_order
        )
        observation = {
            CLASSIFIER_CAMERA_KEYS[0]: chw_rgb_to_conrft_bthwc(camera1),
            CLASSIFIER_CAMERA_KEYS[1]: chw_rgb_to_conrft_bthwc(camera2),
        }
        return AdaptedObservation(
            observation=observation,
            row_reference=row_reference,
            camera_row_identity=camera_row_identity,
            reset_at_episode_boundary=reset,
        )


def to_jax_observation(observation: Mapping[str, np.ndarray]) -> dict[str, object]:
    """Lazily transfer an already validated observation into isolated JAX."""

    if tuple(observation.keys()) != CLASSIFIER_CAMERA_KEYS:
        raise ValueError("classifier camera keys/order mismatch")
    import jax.numpy as jnp

    converted: dict[str, object] = {}
    for key in CLASSIFIER_CAMERA_KEYS:
        value = observation[key]
        if not isinstance(value, np.ndarray) or value.shape != CLASSIFIER_SHAPE:
            raise ValueError(f"{key}: invalid classifier tensor")
        if value.dtype != np.uint8:
            raise TypeError(f"{key}: expected uint8")
        converted[key] = jnp.asarray(value)
    return converted


def apply_conrft_classifier(classifier: object, observation: Mapping[str, np.ndarray]):
    """Call an unmodified ConRFT TrainState on an adapted observation."""

    jax_observation = to_jax_observation(observation)
    return classifier.apply_fn({"params": classifier.params}, jax_observation, train=False)
