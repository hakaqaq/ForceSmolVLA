"""State-bound 7D absolute-action delta transform."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


GIMBAL_MARGIN_RAD = np.deg2rad(2.0)
GRIPPER_WIDTH_RANGE_M = (0.0, 0.1)
BINARY_GRIPPER_CLOSED_WIDTH_M = 0.0
BINARY_GRIPPER_OPEN_WIDTH_M = 0.085
BINARY_GRIPPER_SWITCH_WIDTH_M = 0.0425
MODEL_GRIPPER_CANDIDATE_RANGE_M = (-0.01, 0.095)
INTRINSIC_ACTION_RULE_IDS = frozenset(
    {
        "SS_WORKSPACE",
        "SS_ORIENTATION",
        "SS_DELTA_XYZ",
        "SS_DELTA_ROT_GEODESIC",
        "SS_GRIPPER_RANGE_RATE",
        "SS_CONTINUITY",
    }
)


def wrap_to_pi(angle: np.ndarray) -> np.ndarray:
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def canonicalize_zyx(rpy: np.ndarray) -> np.ndarray:
    """Return the principal ZYX chart for Rz(yaw) @ Ry(pitch) @ Rx(roll)."""

    value = np.asarray(rpy, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("RPY must be finite and end in shape [3]")
    roll, pitch, yaw = np.moveaxis(value, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    r00, r10, r20 = cy * cp, sy * cp, -sp
    r21, r22 = cp * sr, cp * cr
    principal = np.stack(
        (
            np.arctan2(r21, r22),
            np.arctan2(-r20, np.hypot(r00, r10)),
            np.arctan2(r10, r00),
        ),
        axis=-1,
    )
    principal[..., 0] = wrap_to_pi(principal[..., 0])
    principal[..., 2] = wrap_to_pi(principal[..., 2])
    return principal


def rpy_matrix_zyx(rpy: np.ndarray) -> np.ndarray:
    value = np.asarray(rpy, dtype=np.float64)
    if value.shape[-1:] != (3,) or not np.all(np.isfinite(value)):
        raise ValueError("RPY must be finite and end in shape [3]")
    roll, pitch, yaw = np.moveaxis(value, -1, 0)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    matrix = np.empty(value.shape[:-1] + (3, 3), dtype=np.float64)
    matrix[..., 0, 0] = cy * cp
    matrix[..., 0, 1] = cy * sp * sr - sy * cr
    matrix[..., 0, 2] = cy * sp * cr + sy * sr
    matrix[..., 1, 0] = sy * cp
    matrix[..., 1, 1] = sy * sp * sr + cy * cr
    matrix[..., 1, 2] = sy * sp * cr - cy * sr
    matrix[..., 2, 0] = -sp
    matrix[..., 2, 1] = cp * sr
    matrix[..., 2, 2] = cp * cr
    return matrix


def rotation_geodesic_zyx(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(rpy_matrix_zyx(left), -1, -2) @ rpy_matrix_zyx(right)
    cosine = (np.trace(relative, axis1=-2, axis2=-1) - 1.0) / 2.0
    return np.arccos(np.clip(cosine, -1.0, 1.0))


def validate_absolute_action7(
    action7: np.ndarray,
    *,
    workspace_min_xyz_m: np.ndarray | None = None,
    workspace_max_xyz_m: np.ndarray | None = None,
) -> np.ndarray:
    """Canonicalize and validate the invariant v4.1 absolute-action fields."""

    value = np.asarray(action7, dtype=np.float64)
    if value.shape[-1:] != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("absolute action must be finite and end in shape [7]")
    result = value.copy()
    result[..., 3:6] = canonicalize_zyx(result[..., 3:6])
    if np.any(np.abs(np.abs(result[..., 4]) - np.pi / 2) < GIMBAL_MARGIN_RAD):
        raise ValueError("absolute action is inside the forbidden ZYX singular region")
    low, high = GRIPPER_WIDTH_RANGE_M
    if np.any((result[..., 6] < low) | (result[..., 6] > high)):
        raise ValueError("absolute action gripper width is outside [0,0.1] m")
    if (workspace_min_xyz_m is None) != (workspace_max_xyz_m is None):
        raise ValueError("workspace min/max must be provided together")
    if workspace_min_xyz_m is not None:
        minimum = np.asarray(workspace_min_xyz_m, dtype=np.float64)
        maximum = np.asarray(workspace_max_xyz_m, dtype=np.float64)
        if minimum.shape != (3,) or maximum.shape != (3,) or np.any(minimum >= maximum):
            raise ValueError("workspace bounds must be finite ordered xyz vectors")
        if not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
            raise ValueError("workspace bounds must be finite")
        if np.any((result[..., :3] < minimum) | (result[..., :3] > maximum)):
            raise ValueError("absolute action lies outside the bound workspace")
    return result


def decode_binary_gripper_width(action_target7: np.ndarray) -> np.ndarray:
    """Decode a task2 binary gripper candidate into an exact physical width.

    This is not clipping: candidates outside the frozen ForceVLA-compatible
    tolerance fail closed. Accepted candidates map to one of the two exact
    widths present in the training target population.
    """

    value = np.asarray(action_target7, dtype=np.float64)
    if value.shape[-1:] != (7,) or not np.all(np.isfinite(value)):
        raise ValueError("action target must be finite and end in shape [7]")
    candidate = value[..., 6]
    low, high = MODEL_GRIPPER_CANDIDATE_RANGE_M
    if np.any((candidate < low) | (candidate > high)):
        raise ValueError(
            "model gripper candidate is outside the frozen [-0.01,0.095] m tolerance"
        )
    result = value.copy()
    result[..., 6] = np.where(
        candidate < BINARY_GRIPPER_SWITCH_WIDTH_M,
        BINARY_GRIPPER_CLOSED_WIDTH_M,
        BINARY_GRIPPER_OPEN_WIDTH_M,
    )
    return result


@dataclass(frozen=True)
class ActionSafetyProfile:
    mode: str
    rules_sha256: str
    workspace_min_xyz_m: np.ndarray
    workspace_max_xyz_m: np.ndarray
    orientation_min_rpy_rad: np.ndarray
    orientation_max_rpy_rad: np.ndarray
    gimbal_margin_rad: float
    max_delta_xyz_m: float
    max_delta_rotation_rad: float
    gripper_min_width_m: float
    gripper_max_width_m: float
    max_gripper_rate_m_per_s: float
    continuity_max_xyz_m: float
    continuity_max_rotation_rad: float
    continuity_max_gripper_delta_m: float

    @classmethod
    def from_rulespec(cls, rulespec: dict, *, rules_sha256: str) -> "ActionSafetyProfile":
        if len(rules_sha256) != 64 or any(c not in "0123456789abcdef" for c in rules_sha256):
            raise ValueError("action safety rules must have a lowercase SHA256")
        mode = rulespec.get("mode")
        if mode == "test_only":
            if (
                rulespec.get("artifact_status") != "development_only"
                or rulespec.get("acceptance_status") != "development_only"
            ):
                raise RuntimeError("TEST_ONLY_ACTION_SAFETY_STATUS_MISMATCH")
        elif mode == "production":
            if (
                rulespec.get("artifact_status") != "approved"
                or rulespec.get("approval", {}).get("status") != "approved"
                or rulespec.get("signature", {}).get("status") != "verified"
            ):
                raise RuntimeError("PRODUCTION_ACTION_SAFETY_NOT_TRUSTED")
            raise RuntimeError("PRODUCTION_ACTION_SAFETY_VERIFIER_NOT_CONFIGURED")
        else:
            raise RuntimeError("ACTION_SAFETY_MODE_INVALID")
        rules = {rule["rule_id"]: rule for rule in rulespec.get("rules", ())}
        if not INTRINSIC_ACTION_RULE_IDS.issubset(rules):
            raise RuntimeError("INTRINSIC_ACTION_SAFETY_RULES_MISSING")

        def threshold(rule_id: str) -> float:
            value = rules[rule_id]["threshold"]["value"]
            if value is None or not np.isfinite(float(value)) or float(value) < 0:
                raise RuntimeError(f"ACTION_SAFETY_THRESHOLD_UNRESOLVED:{rule_id}")
            return float(value)

        workspace = rules["SS_WORKSPACE"]["parameters"]
        workspace_min = np.asarray(workspace["min_xyz_m"], dtype=np.float64)
        workspace_max = np.asarray(workspace["max_xyz_m"], dtype=np.float64)
        orientation = rules["SS_ORIENTATION"]["parameters"]
        orientation_min = np.asarray(orientation["min_rpy_rad"], dtype=np.float64)
        orientation_max = np.asarray(orientation["max_rpy_rad"], dtype=np.float64)
        gimbal_margin = float(orientation["gimbal_margin_rad"])
        gripper = rules["SS_GRIPPER_RANGE_RATE"]["parameters"]
        gripper_min = float(gripper["min_value"])
        gripper_max = float(gripper["max_value"])
        continuity = rules["SS_CONTINUITY"]["parameters"]
        vectors = (workspace_min, workspace_max, orientation_min, orientation_max)
        if any(value.shape != (3,) or not np.all(np.isfinite(value)) for value in vectors):
            raise RuntimeError("ACTION_SAFETY_VECTOR_INVALID")
        if np.any(workspace_min >= workspace_max) or np.any(orientation_min >= orientation_max):
            raise RuntimeError("ACTION_SAFETY_BOUNDS_INVALID")
        if not np.isclose(gimbal_margin, GIMBAL_MARGIN_RAD, rtol=0, atol=1e-9):
            raise RuntimeError("ACTION_SAFETY_GIMBAL_MARGIN_DRIFT")
        if not (
            np.isclose(gripper_min, GRIPPER_WIDTH_RANGE_M[0], rtol=0, atol=1e-12)
            and np.isclose(gripper_max, GRIPPER_WIDTH_RANGE_M[1], rtol=0, atol=1e-12)
        ):
            raise RuntimeError("ACTION_SAFETY_GRIPPER_WIDTH_SEMANTICS_DRIFT")
        scalar_values = (
            gimbal_margin,
            float(gripper["max_rate_per_s"]),
            float(continuity["max_xyz_m"]),
            float(continuity["max_rotation_rad"]),
            float(continuity["max_gripper_delta"]),
        )
        if any(not np.isfinite(value) or value < 0 for value in scalar_values):
            raise RuntimeError("ACTION_SAFETY_SCALAR_INVALID")
        for value in (workspace_min, workspace_max, orientation_min, orientation_max):
            value.setflags(write=False)
        return cls(
            mode=mode,
            rules_sha256=rules_sha256,
            workspace_min_xyz_m=workspace_min,
            workspace_max_xyz_m=workspace_max,
            orientation_min_rpy_rad=orientation_min,
            orientation_max_rpy_rad=orientation_max,
            gimbal_margin_rad=gimbal_margin,
            max_delta_xyz_m=threshold("SS_DELTA_XYZ"),
            max_delta_rotation_rad=threshold("SS_DELTA_ROT_GEODESIC"),
            gripper_min_width_m=gripper_min,
            gripper_max_width_m=gripper_max,
            max_gripper_rate_m_per_s=float(gripper["max_rate_per_s"]),
            continuity_max_xyz_m=float(continuity["max_xyz_m"]),
            continuity_max_rotation_rad=float(continuity["max_rotation_rad"]),
            continuity_max_gripper_delta_m=float(continuity["max_gripper_delta"]),
        )

    def validate_chunk(
        self,
        absolute_action7: np.ndarray,
        action_valid_mask: np.ndarray,
        raw_state7: np.ndarray,
    ) -> None:
        actions = np.asarray(absolute_action7, dtype=np.float64)
        mask = np.asarray(action_valid_mask, dtype=np.bool_)
        state = np.asarray(raw_state7, dtype=np.float64)
        if actions.ndim != 3 or actions.shape[-1] != 7:
            raise ValueError("absolute action chunk must have shape [B,H,7]")
        if mask.shape != actions.shape[:2] or state.shape != (actions.shape[0], 7):
            raise ValueError("action safety mask/state batch shape mismatch")
        counts = mask.sum(axis=1)
        expected = np.arange(mask.shape[1])[None, :] < counts[:, None]
        if not np.array_equal(mask, expected):
            raise RuntimeError("ACTION_VALID_MASK_MUST_BE_RIGHT_PADDED")
        for batch_index, count in enumerate(counts.tolist()):
            if count <= 0:
                raise RuntimeError("ACTION_SAFETY_EMPTY_CHUNK")
            active = validate_absolute_action7(
                actions[batch_index, :count],
                workspace_min_xyz_m=self.workspace_min_xyz_m,
                workspace_max_xyz_m=self.workspace_max_xyz_m,
            )
            previous = validate_absolute_action7(state[batch_index])
            if np.any(
                (active[:, 3:6] < self.orientation_min_rpy_rad)
                | (active[:, 3:6] > self.orientation_max_rpy_rad)
            ):
                raise RuntimeError("SHADOW_ORIENTATION_INVALID")
            if np.linalg.norm(active[0, :3] - previous[:3]) > self.continuity_max_xyz_m:
                raise RuntimeError("SHADOW_CONTINUITY_INVALID")
            if (
                rotation_geodesic_zyx(previous[3:6], active[0, 3:6])
                > self.continuity_max_rotation_rad
                or abs(float(active[0, 6] - previous[6]))
                > self.continuity_max_gripper_delta_m
            ):
                raise RuntimeError("SHADOW_CONTINUITY_INVALID")
            if count > 1:
                if np.any(
                    np.linalg.norm(np.diff(active[:, :3], axis=0), axis=1)
                    > self.max_delta_xyz_m
                ):
                    raise RuntimeError("SHADOW_DELTA_XYZ_EXCEEDED")
                if np.any(
                    rotation_geodesic_zyx(active[:-1, 3:6], active[1:, 3:6])
                    > self.max_delta_rotation_rad
                ):
                    raise RuntimeError("SHADOW_DELTA_ROT_EXCEEDED")
                gripper_rate = np.abs(np.diff(active[:, 6])) * 30.0
                if np.any(gripper_rate > self.max_gripper_rate_m_per_s):
                    raise RuntimeError("SHADOW_GRIPPER_INVALID")


class ActionDeltaProcessor:
    @staticmethod
    def _broadcast_state(actions: np.ndarray, raw_state7: np.ndarray) -> np.ndarray:
        state = np.asarray(raw_state7, dtype=np.float64)
        if state.ndim < 1 or state.shape[-1] != 7:
            raise ValueError("raw_state7 must end in shape [7]")
        expanded = state
        while expanded.ndim < actions.ndim:
            expanded = np.expand_dims(expanded, axis=-2)
        try:
            if np.broadcast_shapes(actions.shape, expanded.shape) != actions.shape:
                raise ValueError
        except ValueError as error:
            raise ValueError("raw_state7 batch dimensions do not match action chunk") from error
        return expanded

    @staticmethod
    def to_delta(absolute_action7: np.ndarray, raw_state7: np.ndarray) -> np.ndarray:
        actions = validate_absolute_action7(absolute_action7)
        if actions.ndim < 2 or actions.shape[-1] != 7:
            raise ValueError("expected absolute_action7 [...,H,7]")
        raw_state = validate_absolute_action7(raw_state7)
        state = ActionDeltaProcessor._broadcast_state(actions, raw_state)
        result = actions.copy()
        result[..., :3] -= state[..., :3]
        result[..., 3:6] = wrap_to_pi(result[..., 3:6] - state[..., 3:6])
        return result

    @staticmethod
    def from_delta(delta_action7: np.ndarray, raw_state7: np.ndarray) -> np.ndarray:
        actions = np.asarray(delta_action7, dtype=np.float64)
        if actions.ndim < 2 or actions.shape[-1] != 7 or not np.all(np.isfinite(actions)):
            raise ValueError("expected delta_action7 [...,H,7]")
        raw_state = validate_absolute_action7(raw_state7)
        state = ActionDeltaProcessor._broadcast_state(actions, raw_state)
        result = actions.copy()
        result[..., :3] += state[..., :3]
        result[..., 3:6] = canonicalize_zyx(result[..., 3:6] + state[..., 3:6])
        return validate_absolute_action7(result)
