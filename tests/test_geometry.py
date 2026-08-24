import json
from pathlib import Path

import numpy as np

from forcesmolvla.geometry import (
    StaticWrenchCalibration,
    calibrated_tcp_wrench_conditioned_on_measured_tcp_pose,
)


FIXTURE = Path(__file__).parents[1] / "golden_fixtures" / "wrench_geometry.json"


def test_wrench_moment_shift_golden_fixture():
    fixture = json.loads(FIXTURE.read_text())
    transform = fixture["T_TCP_sensor"]
    calibration = StaticWrenchCalibration(
        calibration_id=fixture["fixture_id"],
        translation_tcp_sensor_m=transform["translation_m"],
        quaternion_tcp_sensor_xyzw=transform["quaternion_xyzw"],
        sensor_bias6=fixture["bias_sensor"],
        wrench_sign6=fixture["wrench_sign"],
        downstream_mass_kg=fixture["payload_mass_kg"],
        downstream_com_sensor_m=[0, 0, 0],
        gravity_base_m_s2=[0, 0, -9.80665],
    )
    result = calibrated_tcp_wrench_conditioned_on_measured_tcp_pose(
        fixture["raw_wrench_sensor"],
        [0, 0, 0],
        [0, 0, 0, 1],
        calibration,
    )
    np.testing.assert_allclose(
        result.wrench_base_at_tcp6, fixture["expected_wrench_base_at_tcp"], atol=1e-12
    )


def test_nonfinite_raw_wrench_fails_before_geometry():
    calibration = StaticWrenchCalibration(
        calibration_id="fixture",
        translation_tcp_sensor_m=[0, 0, 0],
        quaternion_tcp_sensor_xyzw=[0, 0, 0, 1],
        sensor_bias6=np.zeros(6),
        wrench_sign6=np.ones(6),
        downstream_mass_kg=0,
        downstream_com_sensor_m=np.zeros(3),
        gravity_base_m_s2=[0, 0, -9.80665],
    )
    with np.testing.assert_raises(ValueError):
        calibrated_tcp_wrench_conditioned_on_measured_tcp_pose(
            [np.nan, 0, 0, 0, 0, 0], [0, 0, 0], [0, 0, 0, 1], calibration
        )
