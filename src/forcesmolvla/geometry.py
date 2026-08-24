"""Measured-TCP-conditioned wrench geometry for ForceSmolVLA v4.1."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _vector(value: np.ndarray, size: int, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be finite shape ({size},)")
    return vector


def quaternion_xyzw_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    q = _vector(quaternion_xyzw, 4, "quaternion_xyzw")
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        raise ValueError("quaternion_xyzw cannot be zero")
    x, y, z, w = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class StaticWrenchCalibration:
    calibration_id: str
    translation_tcp_sensor_m: np.ndarray
    quaternion_tcp_sensor_xyzw: np.ndarray
    sensor_bias6: np.ndarray
    wrench_sign6: np.ndarray
    downstream_mass_kg: float
    downstream_com_sensor_m: np.ndarray
    gravity_base_m_s2: np.ndarray

    def __post_init__(self) -> None:
        if not self.calibration_id:
            raise ValueError("calibration_id is required")
        object.__setattr__(
            self,
            "translation_tcp_sensor_m",
            _vector(self.translation_tcp_sensor_m, 3, "translation_tcp_sensor_m"),
        )
        object.__setattr__(
            self,
            "quaternion_tcp_sensor_xyzw",
            _vector(self.quaternion_tcp_sensor_xyzw, 4, "quaternion_tcp_sensor_xyzw"),
        )
        object.__setattr__(self, "sensor_bias6", _vector(self.sensor_bias6, 6, "sensor_bias6"))
        sign = np.asarray(self.wrench_sign6, dtype=np.float64)
        if sign.shape == ():
            sign = np.full(6, float(sign))
        if sign.shape != (6,) or not np.all(np.isin(sign, (-1.0, 1.0))):
            raise ValueError("wrench_sign6 must contain six +1/-1 values")
        object.__setattr__(self, "wrench_sign6", sign)
        if not np.isfinite(self.downstream_mass_kg) or self.downstream_mass_kg < 0:
            raise ValueError("downstream_mass_kg must be finite and non-negative")
        object.__setattr__(
            self,
            "downstream_com_sensor_m",
            _vector(self.downstream_com_sensor_m, 3, "downstream_com_sensor_m"),
        )
        object.__setattr__(
            self,
            "gravity_base_m_s2",
            _vector(self.gravity_base_m_s2, 3, "gravity_base_m_s2"),
        )


@dataclass(frozen=True)
class CalibratedWrench:
    wrench_base_at_tcp6: np.ndarray
    position_base_sensor_m: np.ndarray
    rotation_base_sensor: np.ndarray
    calibration_id: str


def calibrated_tcp_wrench_conditioned_on_measured_tcp_pose(
    raw_wrench_sensor6: np.ndarray,
    measured_position_base_tcp_m: np.ndarray,
    measured_quaternion_base_tcp_xyzw: np.ndarray,
    calibration: StaticWrenchCalibration,
) -> CalibratedWrench:
    """Apply sign, bias, payload gravity, rotation, and TCP moment shift."""

    raw = _vector(raw_wrench_sensor6, 6, "raw_wrench_sensor6")
    position_base_tcp = _vector(
        measured_position_base_tcp_m, 3, "measured_position_base_tcp_m"
    )
    rotation_base_tcp = quaternion_xyzw_to_matrix(measured_quaternion_base_tcp_xyzw)
    rotation_tcp_sensor = quaternion_xyzw_to_matrix(calibration.quaternion_tcp_sensor_xyzw)
    rotation_base_sensor = rotation_base_tcp @ rotation_tcp_sensor
    position_base_sensor = (
        position_base_tcp + rotation_base_tcp @ calibration.translation_tcp_sensor_m
    )

    signed_unbiased = calibration.wrench_sign6 * raw - calibration.sensor_bias6
    gravity_force_sensor = rotation_base_sensor.T @ (
        calibration.downstream_mass_kg * calibration.gravity_base_m_s2
    )
    gravity_moment_sensor = np.cross(
        calibration.downstream_com_sensor_m, gravity_force_sensor
    )
    force_sensor = signed_unbiased[:3] - gravity_force_sensor
    moment_sensor = signed_unbiased[3:] - gravity_moment_sensor

    force_base = rotation_base_sensor @ force_sensor
    moment_base_at_sensor = rotation_base_sensor @ moment_sensor
    moment_base_at_tcp = moment_base_at_sensor + np.cross(
        position_base_sensor - position_base_tcp, force_base
    )
    return CalibratedWrench(
        np.concatenate((force_base, moment_base_at_tcp)),
        position_base_sensor,
        rotation_base_sensor,
        calibration.calibration_id,
    )
