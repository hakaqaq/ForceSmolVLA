from __future__ import annotations

import ast
import importlib
import os
from pathlib import Path
import subprocess
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
        if path.name == "integrated_shadow_backend.py":
            continue
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
        "protocol", "publication", "checkpoint", "temporal_parity", "learner", "loopback",
        "parent",
    ):
        importlib.import_module(f"forcesmolvla.rft.stage3.{name}")
    added = set(sys.modules) - before
    ros = [name for name in added if name.split(".", 1)[0] in {"rclpy", "rospy", "roslib"}]
    assert ros == []
    assert not any("deploy_forcesmolvla" in name or "serve_policy" in name for name in added)


def test_integrated_capture_import_does_not_require_torch() -> None:
    script = """
import importlib.abc
import sys

class BlockTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "torch" or fullname.startswith("torch."):
            raise ModuleNotFoundError("torch blocked for capture import test")
        return None

sys.meta_path.insert(0, BlockTorch())
import forcesmolvla.rft.stage3.integrated_capture
assert "torch" not in sys.modules
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(ROOT / "src")
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
