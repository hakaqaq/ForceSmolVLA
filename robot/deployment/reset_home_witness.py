#!/usr/bin/env python3
"""Move to the recorder's real Home and emit one quiescent reset witness."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Callable, Mapping


DEFAULT_CONTROL_SCRIPT = Path(
    "/home/rlc123/fr3_client_ws/scripts/record_franka_spacemouse_publisher.py"
)


class HomeWitnessError(RuntimeError):
    pass


def _read_sealed_generation(path: Path) -> tuple[int, dict[str, Any]]:
    try:
        seal = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise HomeWitnessError("HOME_WITNESS_EPISODE_SEAL_INVALID") from error
    request_count = int(seal.get("policy_request_count", -1))
    completed_count = int(seal.get("policy_result_count", -1)) + int(
        seal.get("policy_request_canceled_count", 0)
    )
    if (
        seal.get("technical_seal") != "complete"
        or int(seal.get("sealed_monotonic_ns", 0)) <= 0
        or int(seal.get("reset_generation", -1)) < 0
        or request_count < 0
        or request_count != completed_count
        or int(seal.get("controller_process_count", -1)) != 1
        or seal.get("deploy_controller_started") is not False
    ):
        raise HomeWitnessError("HOME_WITNESS_EPISODE_NOT_QUIESCENT")
    return int(seal["reset_generation"]), seal


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise HomeWitnessError("HOME_WITNESS_OUTPUT_ALREADY_EXISTS")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def write_reset_home_witness(
    *,
    output: Path,
    previous_episode_seal: Path,
    home_backend: Callable[[], Mapping[str, Any]],
) -> dict[str, Any]:
    previous_generation, seal = _read_sealed_generation(previous_episode_seal)
    result = dict(home_backend())
    quiescent = result.pop("quiescent", None)
    if (
        result.get("home_completed") is not True
        or result.get("controller_idle") is not True
        or result.get("gateway_status") != "completed: joint position target"
        or result.get("home_implementation")
        != "record_franka_spacemouse_publisher.FrankaRecordSpaceMousePublisher.move_to_recorded_home"
        or int(result.get("completed_monotonic_ns", 0)) <= 0
        or int(result.get("controller_owner_count", -1)) != 1
        or not math.isfinite(float(result.get("max_joint_error_rad", math.nan)))
        or not math.isfinite(
            float(result.get("max_joint_velocity_rad_s", math.nan))
        )
        or float(result["max_joint_error_rad"])
        > float(result.get("home_joint_tolerance_rad", -1.0))
        or float(result["max_joint_velocity_rad_s"])
        > float(result.get("home_velocity_tolerance_rad_s", -1.0))
        or not isinstance(quiescent, Mapping)
        or quiescent.get("active_episode") is not False
        or int(quiescent.get("inflight_inference", -1)) != 0
        or int(quiescent.get("queued_actions", -1)) != 0
        or int(quiescent.get("unconsumed_acks", -1)) != 0
        or quiescent.get("wal_sealed") is not True
    ):
        raise HomeWitnessError("HOME_WITNESS_REAL_HOME_INCOMPLETE")
    witness = {
        "kind": "reset_home_quiescent",
        "source": "recorded_home_backend",
        "robot_home": True,
        "reset_generation": previous_generation + 1,
        "completed_monotonic_ns": int(result["completed_monotonic_ns"]),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "previous_episode": {
            "episode_id": seal.get("episode_id"),
            "sealed_monotonic_ns": int(seal["sealed_monotonic_ns"]),
            "reset_generation": previous_generation,
        },
        "home_result": result,
        "quiescent": dict(quiescent),
    }
    _atomic_json(output, witness)
    return witness


def real_home_backend(
    *, control_script: Path, interface_timeout: float, home_timeout: float
) -> dict[str, Any]:
    if os.environ.get("ROS_DOMAIN_ID") != "30":
        raise HomeWitnessError("HOME_WITNESS_ROS_DOMAIN_ID_MUST_BE_30")
    if os.environ.get("ROS_LOCALHOST_ONLY", "0").lower() not in {"0", "false"}:
        raise HomeWitnessError("HOME_WITNESS_ROS_LOCALHOST_ONLY_MUST_BE_0")
    spec = importlib.util.spec_from_file_location(
        "stage3_recorded_home_control", control_script.resolve()
    )
    if spec is None or spec.loader is None:
        raise HomeWitnessError("HOME_WITNESS_CONTROL_SCRIPT_MISSING")
    control = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = control
    spec.loader.exec_module(control)

    class HomeOnlyController(control.FrankaRecordSpaceMousePublisher):
        def __init__(self) -> None:
            control.Node.__init__(
                self,
                "stage3_reset_home_witness",
                start_parameter_services=False,
                enable_rosout=False,
            )
            self.joint_position_publisher = self.create_publisher(
                control.JointState, control.DEFAULT_JOINT_POSITION_TOPIC, 1
            )
            self.stop_publisher = self.create_publisher(
                control.Empty, control.DEFAULT_STOP_TOPIC, 1
            )
            self.joint_state_subscription = self.create_subscription(
                control.JointState,
                control.DEFAULT_JOINT_STATE_TOPIC,
                self._joint_state_callback,
                10,
            )
            self.status_subscription = self.create_subscription(
                control.String,
                control.DEFAULT_STATUS_TOPIC,
                self._status_callback,
                10,
            )
            self.latest_joint_state = None
            self.latest_status = None
            self.home_motion_requested = False

        def wait_for_home_interfaces(self, timeout_s: float) -> None:
            deadline = time.monotonic() + timeout_s
            while control.rclpy.ok() and time.monotonic() < deadline:
                control.rclpy.spin_once(self, timeout_sec=0.1)
                if (
                    self.count_subscribers(control.DEFAULT_JOINT_POSITION_TOPIC) > 0
                    and self.count_publishers(control.DEFAULT_JOINT_POSITION_TOPIC)
                    == 1
                    and self.count_publishers(control.DEFAULT_VELOCITY_TOPIC) == 0
                    and self.count_publishers(control.DEFAULT_STATUS_TOPIC) > 0
                    and self.latest_joint_state is not None
                ):
                    self._ordered_joint_values(self.latest_joint_state, "position")
                    return
            raise HomeWitnessError("HOME_WITNESS_HOME_INTERFACES_NOT_QUIESCENT")

    control.rclpy.init()
    node: HomeOnlyController | None = None
    try:
        node = HomeOnlyController()
        node.wait_for_home_interfaces(interface_timeout)
        node.move_to_recorded_home(home_timeout)
        if node.latest_joint_state is None:
            raise HomeWitnessError("HOME_WITNESS_JOINT_STATE_MISSING")
        positions = node._ordered_joint_values(node.latest_joint_state, "position")
        velocities = node._ordered_joint_values(node.latest_joint_state, "velocity")
        max_error = max(
            abs(actual - expected)
            for actual, expected in zip(
                positions, control.HOME_JOINT_POSITIONS_RAD, strict=True
            )
        )
        max_velocity = max(abs(value) for value in velocities)
        return {
            "home_completed": True,
            "controller_idle": node.latest_status
            == "completed: joint position target",
            "gateway_status": node.latest_status,
            "completed_monotonic_ns": time.monotonic_ns(),
            "controller_owner_count": 1,
            "max_joint_error_rad": max_error,
            "max_joint_velocity_rad_s": max_velocity,
            "home_joint_tolerance_rad": control.HOME_JOINT_TOLERANCE_RAD,
            "home_velocity_tolerance_rad_s": control.HOME_VELOCITY_TOLERANCE_RAD_S,
            "settle_time_s": control.HOME_SETTLE_TIME_S,
            "home_implementation": (
                "record_franka_spacemouse_publisher."
                "FrankaRecordSpaceMousePublisher.move_to_recorded_home"
            ),
            "quiescent": {
                "active_episode": False,
                "inflight_inference": 0,
                "queued_actions": 0,
                "unconsumed_acks": 0,
                "wal_sealed": True,
            },
        }
    finally:
        if node is not None:
            if control.rclpy.ok() and node.home_motion_requested:
                node.stop_publisher.publish(control.Empty())
                node.get_logger().warn(
                    "published Franky stop for incomplete Home motion"
                )
                time.sleep(0.05)
            node.destroy_node()
        if control.rclpy.ok():
            control.rclpy.shutdown()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--previous-episode-seal", type=Path, required=True)
    parser.add_argument("--interface-timeout", type=float, default=10.0)
    parser.add_argument("--home-timeout", type=float, default=30.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    witness = write_reset_home_witness(
        output=args.output,
        previous_episode_seal=args.previous_episode_seal,
        home_backend=lambda: real_home_backend(
            control_script=DEFAULT_CONTROL_SCRIPT,
            interface_timeout=args.interface_timeout,
            home_timeout=args.home_timeout,
        ),
    )
    print(json.dumps(witness, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
