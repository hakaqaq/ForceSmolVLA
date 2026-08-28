from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import sys


ROOT = Path(__file__).parents[1]
STAGE3 = ROOT / "src/forcesmolvla/rft/stage3"
BANNED_IMPORT_ROOTS = {
    "rclpy", "rospy", "roslib", "franka", "franka_msgs", "moveit",
    "requests", "httpx", "socket", "subprocess",
}


def test_stage3_cpu_modules_have_no_ros_robot_or_network_imports() -> None:
    violations = []
    for path in sorted(STAGE3.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            for name in names:
                if name.split(".", 1)[0] in BANNED_IMPORT_ROOTS:
                    violations.append((path.name, name))
    assert violations == []


def test_importing_stage3_stays_cpu_only_and_does_not_connect_or_command() -> None:
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    before = set(sys.modules)
    for name in (
        "contracts", "transition", "replay", "batch", "losses", "update_credit",
        "protocol", "publication", "checkpoint", "temporal_parity",
    ):
        importlib.import_module(f"forcesmolvla.rft.stage3.{name}")
    added = set(sys.modules) - before
    ros = [name for name in added if name.split(".", 1)[0] in {"rclpy", "rospy", "roslib"}]
    assert ros == []
    assert not any("deploy_forcesmolvla" in name or "serve_policy" in name for name in added)
