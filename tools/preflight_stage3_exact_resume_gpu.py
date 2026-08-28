#!/usr/bin/env python3
"""G5P serial three-process exact-resume preflight on the approved hybrid parent."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping

import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/stage3_exact_resume.v1.development.yaml"
REPORT_SCHEMA = ROOT / "schemas/stage3_exact_resume_report.v1.schema.json"
CHECKPOINT_SOURCE = ROOT / "src/forcesmolvla/rft/stage3/checkpoint.py"
G4P_TOOL = ROOT / "tools/preflight_stage3_gpu.py"
FAULT_CASES = [
    "partial checkpoint",
    "tampered payload",
    "missing completion marker",
    "wrong parent/config/source digest",
    "missing optimizer state",
    "optimizer group reorder",
    "RNG omission",
    "corrupted RNG state",
    "credit/counter drift",
    "mid-cycle save",
    "unsealed episode restore",
    "pending revision restore",
    "cold/warm decoded-image cache model-state invariance",
]


class G5PError(RuntimeError):
    pass


def require(condition: bool, code: str) -> None:
    if not condition:
        raise G5PError(code)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = Path(stream.name)
    os.replace(temporary, path)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    _atomic_bytes(
        path,
        (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )


def _atomic_text(path: Path, value: str) -> None:
    _atomic_bytes(path, value.encode("utf-8"))


def _load_config() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    require(value["schema_version"] == "forcesmolvla_stage3_exact_resume.v1.development", "G5P_CONFIG_SCHEMA")
    require(value["claim_scope"] == "G5P_isolated_learner_exact_resume_only", "G5P_CLAIM_SCOPE")
    require(value["branches"]["execution"] == "serial_three_fresh_subprocesses", "G5P_BRANCH_EXECUTION")
    require(value["checkpoint"]["minimum_free_copies"] >= 3, "G5P_DISK_RESERVATION")
    require(value["schedule"]["joint_cycles"] == 2, "G5P_CYCLE_SCOPE")
    require(value["safety"]["G6_and_later"] == "NOT_RUN", "G5P_LATER_GATE_SCOPE")
    return value


def _config_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _validate_frozen_baseline(config: Mapping[str, Any]) -> None:
    baseline = config["baseline"]
    checks = (
        ("g4p_config", "g4p_config_sha256"),
        ("g4p_json", "g4p_json_sha256"),
        ("g4p_markdown", "g4p_markdown_sha256"),
    )
    for path_key, sha_key in checks:
        path = _config_path(baseline[path_key])
        require(path.is_file(), f"G5P_BASELINE_FILE_MISSING:{path_key}")
        require(sha256_file(path) == baseline[sha_key], f"G5P_BASELINE_SHA:{path_key}")
    g4p = json.loads(_config_path(baseline["g4p_json"]).read_text())
    require(
        g4p["canonical_report_sha256"] == baseline["g4p_canonical_report_sha256"],
        "G5P_G4P_CANONICAL_REPORT_SHA",
    )


def _git_value(*arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def _cuda_processes() -> list[dict[str, Any]]:
    result = subprocess.run(
        [
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True, text=True, check=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        rows.append({
            "pid": int(parts[0]), "process_name": parts[1],
            "used_memory_mib": int(parts[2]),
        })
    return rows


def _parent_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    g4p_config = yaml.safe_load(_config_path(config["baseline"]["g4p_config"]).read_text())
    binding_path = _config_path(g4p_config["parent_binding"]["path"])
    binding = json.loads(binding_path.read_text())
    records = [("binding", binding_path)]
    records.append(("actor", Path(binding["actor_parent"]["absolute_path"])))
    for group in ("critic_parent", "target_critic_parent"):
        records.extend(
            (item["logical_role"], Path(item["absolute_path"]))
            for item in binding[group]["artifacts"]
        )
    return {
        name: {"path": str(path), "sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for name, path in records
    }


def _cpu_fault_tests() -> dict[str, Any]:
    environment = os.environ.copy()
    environment.update({"CUDA_VISIBLE_DEVICES": "", "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})
    process = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/test_stage3_exact_resume.py"],
        cwd=ROOT, env=environment, capture_output=True, text=True,
    )
    require(process.returncode == 0 and "12 passed" in process.stdout, f"G5P_CPU_FAULT_TESTS:{process.stdout[-2000:]}{process.stderr[-2000:]}")
    return {"cpu_test_count": 12, "all_passed": True, "cases": FAULT_CASES}


def _worker_environment(config: Mapping[str, Any]) -> dict[str, str]:
    determinism = config["determinism"]
    environment = os.environ.copy()
    environment.update({
        "CUDA_VISIBLE_DEVICES": "0",
        "CUBLAS_WORKSPACE_CONFIG": determinism["cublas_workspace_config"],
        "PYTHONHASHSEED": str(determinism["pythonhashseed"]),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONPATH": ":".join((
            str(ROOT / "src"), str(ROOT / "vendor/lerobot/src"), str(ROOT / "tools")
        )),
    })
    return environment


def _run_worker(
    branch: str, run_root: Path, checkpoint: Path, environment: Mapping[str, str]
) -> dict[str, Any]:
    result_path = run_root / f"branch_{branch.lower()}_result.json"
    log_path = run_root / f"branch_{branch.lower()}.log"
    command = [
        sys.executable, str(Path(__file__).resolve()), "--worker", branch,
        "--run-root", str(run_root), "--checkpoint", str(checkpoint),
        "--result", str(result_path),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, cwd=ROOT, env=dict(environment), stdout=log,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
        return_code = process.wait(timeout=7200)
    if return_code:
        raise G5PError(
            f"G5P_BRANCH_{branch}_FAILED:{return_code}\n{log_path.read_text(errors='replace')[-12000:]}"
        )
    require(result_path.is_file(), f"G5P_BRANCH_{branch}_RESULT_MISSING")
    result = json.loads(result_path.read_text())
    require(result["pid"] == process.pid, f"G5P_BRANCH_{branch}_PID_MISMATCH")
    require(not _cuda_processes(), f"G5P_BRANCH_{branch}_CUDA_PROCESS_REMAINED")
    return result


def _first_mismatch(left: Any, right: Any, path: str = "root") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"component": path, "left_type": type(left).__name__, "right_type": type(right).__name__}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {"component": path, "left_keys": sorted(left), "right_keys": sorted(right)}
        for key in left:
            mismatch = _first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"component": path, "left_length": len(left), "right_length": len(right)}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            mismatch = _first_mismatch(left_item, right_item, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if left != right:
        result = {"component": path, "left": left, "right": right, "first_cycle": 1 if "intermediate" in path else 2}
        if path.endswith("sha256"):
            result.update({"dtype": None, "shape": None, "max_abs_error": None, "max_rel_error": None})
        return result
    return None


def _parity(branch_a: Mapping[str, Any], branch_b1: Mapping[str, Any], branch_b2: Mapping[str, Any]) -> dict[str, Any]:
    a1, b1 = branch_a["intermediate_state"], branch_b1["intermediate_state"]
    a2, b2 = branch_a["final_state"], branch_b2["final_state"]
    a_trace1, b_trace1 = branch_a["traces"][0], branch_b1["traces"][0]
    a_trace2, b_trace2 = branch_a["traces"][1], branch_b2["traces"][0]
    sections_a2, sections_b2 = a2["sections"], b2["sections"]
    result = {
        "intermediate_state": a1 == b1 == branch_b1["checkpoint_state"] == branch_b2["restored_state"],
        "final_tensor_bitwise": sections_a2["models"] == sections_b2["models"],
        "final_canonical_digest": a2["canonical_content_digest"] == b2["canonical_content_digest"],
        "sample_uid_order": a_trace1["sample_uid_order"] == b_trace1["sample_uid_order"] and a_trace2["sample_uid_order"] == b_trace2["sample_uid_order"],
        "flow_noise": a_trace1["observed_tensors"]["random_tensors"] == b_trace1["observed_tensors"]["random_tensors"] and a_trace2["observed_tensors"]["random_tensors"] == b_trace2["observed_tensors"]["random_tensors"],
        "optimizer_state": sections_a2["optimizers"] == sections_b2["optimizers"],
        "rng_state": sections_a2["rng"] == sections_b2["rng"],
        "replay_credit_counter": all(
            sections_a2[name] == sections_b2[name]
            for name in ("replay", "credits", "counters", "sampler")
        ),
        "actor_tensors": sections_a2["models"]["actor"] == sections_b2["models"]["actor"],
        "online_q_tensors": all(sections_a2["models"][name] == sections_b2["models"][name] for name in ("q1", "q2")),
        "target_q_tensors": all(sections_a2["models"][name] == sections_b2["models"][name] for name in ("q1_target", "q2_target")),
        "loss_q_gradient_parameter_polyak_trace": a_trace1 == b_trace1 and a_trace2 == b_trace2,
        "scheduler_scaler": sections_a2["schedulers"] == sections_b2["schedulers"] and sections_a2["grad_scaler"] == sections_b2["grad_scaler"],
        "revision_state": sections_a2["revision"] == sections_b2["revision"],
        "first_mismatch": None,
    }
    if not all(value for key, value in result.items() if key != "first_mismatch"):
        result["first_mismatch"] = (
            _first_mismatch(a1, b1, "intermediate_state")
            or _first_mismatch(a_trace1, b_trace1, "cycle1_trace")
            or _first_mismatch(a2, b2, "final_state")
            or _first_mismatch(a_trace2, b_trace2, "cycle2_trace")
        )
    return result


def _render_markdown(report: Mapping[str, Any]) -> str:
    return f"""# Stage-3 G5P isolated learner exact-resume preflight

Status: `G5P isolated learner exact-resume preflight passed.`

This is a ForceRFT engineering safety extension. ConRFT does not provide the exact-resume implementation required here. It is not production online durable resume, policy publication, Critic warmup, GPU coexistence, online collection, or robot evidence.

## Audited interfaces

- `stage3/checkpoint.py`: the prior symbols `validate_online_checkpoint_metadata` and `cpu_round_trip_online_checkpoint` covered metadata only. G5P adds safetensors/restricted-state atomic save, validation, and strict load.
- `stage3/learner.py`: `ProvisionalStage3Learner.run_joint_cycle` is a CPU synthetic loopback learner; GPU G5P instead reuses the real G4P `_critic_step` and `_actor_step` ForceRFT primitives.
- `stage3/loopback.py`: `run_synthetic_loopback` remains G3P synthetic evidence and is not used as production replay.
- `canonical_state.py`: reuses `canonical_digest`, `module_record`, `optimizer_parameter_name_groups`, and `optimizer_record`.
- `exact_resume.py`: Phase-2 primitives were audited but their optimizer/RNG/sampler state was not inherited.

## Fresh subprocess evidence

- Branch A PID `{report['branches']['A']['pid']}`: cycles 1 and 2 continuously.
- Branch B1 PID `{report['branches']['B1']['pid']}`: cycle 1, atomic checkpoint, full exit.
- Branch B2 PID `{report['branches']['B2']['pid']}`: strict checkpoint load and cycle 2.
- Disposable checkpoint: `{report['checkpoint']['path']}` ({report['checkpoint']['size_bytes']} bytes).
- Canonical content digest: `{report['checkpoint']['canonical_content_digest']}`.

All model, optimizer, RNG, loss/Q, captured gradient, parameter delta, Polyak, sampler, replay/credit/counter, and revision comparisons are exact. `allclose` is not an acceptance criterion.

## Limits

- `REAL_ONLINE_R_USED=false`; R is `synthetic_preflight_R_only` over the frozen real observation pipeline.
- `PRODUCTION_WAL_OUTBOX_RESUME_VALIDATED=false` and `G5_PRODUCTION_DURABLE_RESUME=UNVERIFIED`.
- No policy revision was activated; no ROS/network/robot path was entered.
- `G6_AND_LATER=NOT_RUN`.
"""


def _coordinator() -> None:
    config = _load_config()
    _validate_frozen_baseline(config)
    require(_git_value("branch", "--show-current") == config["baseline"]["branch"], "G5P_BRANCH_DRIFT")
    require(_git_value("rev-parse", "HEAD") == config["baseline"]["head"], "G5P_HEAD_DRIFT")
    require(not _cuda_processes(), "G5P_CUDA_PROCESS_PRESENT_BEFORE_RUN")
    parent_before = _parent_snapshot(config)
    fault_injection = _cpu_fault_tests()
    free_before = shutil.disk_usage(config["checkpoint"]["disposable_root"]).free
    parent_size = sum(value["size_bytes"] for name, value in parent_before.items() if name != "binding")
    require(free_before >= 3 * (parent_size + 1024**3), "G5P_DISK_PREFLIGHT_INSUFFICIENT")

    run_root = Path(tempfile.mkdtemp(
        prefix="forcesmolvla_g5p_", dir=config["checkpoint"]["disposable_root"]
    ))
    checkpoint = run_root / "b1_disposable_exact_resume_checkpoint"
    environment = _worker_environment(config)
    branch_a = _run_worker("A", run_root, checkpoint, environment)
    branch_b1 = _run_worker("B1", run_root, checkpoint, environment)
    require(checkpoint.is_dir(), "G5P_B1_CHECKPOINT_MISSING")
    branch_b2 = _run_worker("B2", run_root, checkpoint, environment)
    pids = {branch_a["pid"], branch_b1["pid"], branch_b2["pid"]}
    require(len(pids) == 3, "G5P_FRESH_PROCESS_PID_NOT_DISTINCT")
    parity = _parity(branch_a, branch_b1, branch_b2)
    require(all(value for key, value in parity.items() if key != "first_mismatch"), f"G5P_EXACT_PARITY:{parity['first_mismatch']}")
    parent_after = _parent_snapshot(config)
    no_cuda = not _cuda_processes()
    require(parent_before == parent_after, "G5P_PARENT_CHECKPOINT_MUTATED")
    require(no_cuda, "G5P_CUDA_PROCESS_REMAINED")

    branches = {
        name: {
            "pid": value["pid"], "status": "PASS", "cycles": value["cycles"],
            "initial_state_digest": value["initial_state_digest"],
            "intermediate_state_digest": value.get("intermediate_state_digest"),
            "final_state_digest": value.get("final_state_digest"),
        }
        for name, value in (("A", branch_a), ("B1", branch_b1), ("B2", branch_b2))
    }
    performance = {
        "peak_vram_mib": max(item["peak_vram_mib"] for item in (branch_a, branch_b1, branch_b2)),
        "peak_cpu_rss_mib": max(item["peak_cpu_rss_mib"] for item in (branch_a, branch_b1, branch_b2)),
        "checkpoint_save_seconds": branch_b1["checkpoint_save_seconds"],
        "checkpoint_load_seconds": branch_b2["checkpoint_load_seconds"],
        "disk_free_before_bytes": free_before,
    }
    report = {
        "schema_version": "forcesmolvla_stage3_exact_resume_report.v1",
        "tool_status": "PASS",
        "scope": "G5P_isolated_learner_exact_resume_only",
        "source_branch": config["baseline"]["branch"],
        "source_head": config["baseline"]["head"],
        "config": {"path": str(CONFIG.relative_to(ROOT)), "sha256": sha256_file(CONFIG)},
        "environment": branch_a["environment"],
        "branches": branches,
        "checkpoint": {
            "path": str(checkpoint),
            "size_bytes": branch_b1["checkpoint_size_bytes"],
            "canonical_content_digest": branch_b1["checkpoint_state"]["canonical_content_digest"],
            "disposable": True,
            "parent_checkpoint": False,
            "publishable_policy_revision": False,
            "completion_marker": "COMPLETED.json",
            "atomic_same_filesystem_rename": True,
            "minimum_free_copies": config["checkpoint"]["minimum_free_copies"],
        },
        "parity": parity,
        "fault_injection": fault_injection,
        "performance": performance,
        "safety": {
            "REAL_ONLINE_R_USED": False,
            "PRODUCTION_WAL_OUTBOX_RESUME_VALIDATED": False,
            "G5_PRODUCTION_DURABLE_RESUME": "UNVERIFIED",
            "G5_FORMAL_GATE_PASSED": False,
            "CRITIC_WARMUP_STARTED": False,
            "CRITIC_READY": False,
            "ACTOR_Q_GUIDANCE_ENABLED": False,
            "POLICY_REVISION_ACTIVATED": False,
            "G3_RECORDED_FIXTURE_LOOPBACK": "BLOCKED",
            "G6_AND_LATER": "NOT_RUN",
            "ROBOT_CONNECTION_COUNT": 0,
            "ROBOT_COMMAND_COUNT": 0,
            "ROBOT_EXECUTION_AUTHORIZED": False,
            "PUSHED": False,
            "parent_checkpoint_sha_unchanged": parent_before == parent_after,
            "production_actor_checkpoint_unchanged": parent_before["actor"] == parent_after["actor"],
            "no_cuda_compute_process": no_cuda,
        },
        "canonical_report_sha256": "0" * 64,
    }
    report["canonical_report_sha256"] = _canonical_json_sha256({
        key: value for key, value in report.items() if key != "canonical_report_sha256"
    })
    from jsonschema import Draft202012Validator

    schema = json.loads(REPORT_SCHEMA.read_text())
    errors = sorted(Draft202012Validator(schema).iter_errors(report), key=lambda error: list(error.absolute_path))
    require(not errors, f"G5P_REPORT_SCHEMA:{errors[0].message if errors else ''}")
    artifact_path = _config_path(config["output"]["artifact"])
    markdown_path = _config_path(config["output"]["markdown_report"])
    require(not artifact_path.exists() and not markdown_path.exists(), "G5P_OUTPUT_ALREADY_EXISTS")
    _atomic_text(markdown_path, _render_markdown(report))
    _atomic_json(artifact_path, report)
    print(json.dumps({
        "G5P_RESULT": "PASS", "checkpoint": str(checkpoint),
        "checkpoint_size_bytes": report["checkpoint"]["size_bytes"],
        "canonical_content_digest": report["checkpoint"]["canonical_content_digest"],
        "branch_a_pid": branches["A"]["pid"], "branch_b1_pid": branches["B1"]["pid"],
        "branch_b2_pid": branches["B2"]["pid"],
    }, sort_keys=True))


class _Schedule:
    def __init__(self, critic_rows: list[dict], actor_rows: list[dict], generator) -> None:
        self.critic_rows = deepcopy(critic_rows)
        self.actor_rows = deepcopy(actor_rows)
        self.generator = generator
        self.cursor = 0

    @staticmethod
    def _positions(generator) -> tuple[list[int], list[int]]:
        import torch

        critic = (
            torch.randperm(32, generator=generator).tolist()
            + (torch.randperm(31, generator=generator) + 32).tolist()
            + [63]
        )
        actor = (
            torch.randperm(12, generator=generator).tolist()
            + (torch.randperm(12, generator=generator) + 12).tolist()
        )
        return critic, actor

    def _order(self, positions: tuple[list[int], list[int]]) -> dict[str, list[str]]:
        critic, actor = positions
        return {
            "critic": [self.critic_rows[index]["row_identity"] for index in critic],
            "actor": [self.actor_rows[index]["row_identity"] for index in actor],
        }

    def _preview(self) -> dict[str, list[str]] | None:
        if self.cursor >= 2:
            return None
        import torch

        preview = torch.Generator(device="cpu")
        preview.set_state(self.generator.get_state())
        return self._order(self._positions(preview))

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "forcesmolvla_g5p_sample_schedule.v1",
            "cursor": self.cursor,
            "critic_population": [row["row_identity"] for row in self.critic_rows],
            "actor_population": [row["row_identity"] for row in self.actor_rows],
            "next_sample_order": self._preview(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        if (
            state["schema_version"] != expected["schema_version"]
            or state["critic_population"] != expected["critic_population"]
            or state["actor_population"] != expected["actor_population"]
        ):
            raise G5PError("G5P_SAMPLER_POPULATION_MISMATCH")
        self.cursor = int(state["cursor"])
        require(self._preview() == state["next_sample_order"], "G5P_SAMPLER_NEXT_ORDER_MISMATCH")

    def draw(self) -> tuple[list[int], list[int], dict[str, list[str]]]:
        require(self.cursor < 2, "G5P_SAMPLE_SCHEDULE_EXHAUSTED")
        positions = self._positions(self.generator)
        order = self._order(positions)
        self.cursor += 1
        return positions[0], positions[1], order


def _index_nested(value: Any, indices, size: int) -> Any:
    import torch

    if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == size:
        return value.index_select(0, indices.to(value.device))
    if isinstance(value, dict):
        return {name: _index_nested(item, indices, size) for name, item in value.items()}
    if isinstance(value, tuple) and len(value) == size:
        return tuple(value[index] for index in indices.cpu().tolist())
    if isinstance(value, list) and len(value) == size:
        return [value[index] for index in indices.cpu().tolist()]
    return value


def _index_batch(batch: Mapping[str, Any], order: list[int]) -> dict[str, Any]:
    import torch

    size = len(order)
    indices = torch.tensor(order, dtype=torch.long, device=batch["current_observation"].camera1.device)
    result = {}
    for name, value in batch.items():
        if name in {"current_observation", "next_observation"}:
            result[name] = value.index(indices)
        else:
            result[name] = _index_nested(value, indices, size)
    return result


class _CycleTrace:
    def __init__(self, context: Mapping[str, Any]) -> None:
        import torch

        self.context = context
        self.torch = torch
        self.generator_names = {id(value): name for name, value in context["generators"].items()}
        self.parameter_names = {
            id(parameter): f"{owner}.{name}"
            for owner, module in context["modules"].items()
            for name, parameter in module.named_parameters()
        }
        self.records = {
            "random_tensors": [], "loss_q_tensors": [], "gradient_tensors": [],
            "parameter_deltas": [], "polyak_deltas": [],
        }
        self.restorers = []

    def _patch(self, owner, name: str, replacement) -> None:
        original = getattr(owner, name)
        setattr(owner, name, replacement(original))
        self.restorers.append(lambda: setattr(owner, name, original))

    def __enter__(self):
        from forcesmolvla.rft.canonical_state import tensor_record
        import forcesmolvla.rft.stage3.losses as losses
        import forcesmolvla.rft.training_cycle as training_cycle
        import preflight_stage3_gpu as g4

        def random_factory(operation: str):
            def factory(original):
                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    name = self.generator_names.get(id(kwargs.get("generator")))
                    if name is not None:
                        self.records["random_tensors"].append({
                            "operation": operation, "generator": name,
                            "tensor": tensor_record(result),
                        })
                    return result
                return wrapped
            return factory

        for name in ("randn", "rand", "randperm"):
            self._patch(self.torch, name, random_factory(name))

        def td_factory(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                self.records["loss_q_tensors"].append({
                    "kind": "critic_td",
                    "total": tensor_record(result.total),
                    "q1_loss": tensor_record(result.q1_loss),
                    "q2_loss": tensor_record(result.q2_loss),
                    "q1": tensor_record(result.q1_value),
                    "q2": tensor_record(result.q2_value),
                    "target": tensor_record(result.target),
                })
                return result
            return wrapped

        def actor_q_factory(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                self.records["loss_q_tensors"].append({
                    "kind": "actor_q", "loss": tensor_record(result[0]),
                    "q1": tensor_record(result[1]), "q2": tensor_record(result[2]),
                    "action": tensor_record(result[3]),
                })
                return result
            return wrapped

        def actor_objective_factory(original):
            def wrapped(*args, **kwargs):
                result = original(*args, **kwargs)
                names = ("total", "expert_flow_matching", "actor_q", "balance", "z")
                self.records["loss_q_tensors"].append({
                    "kind": "actor_objective",
                    **{name: tensor_record(getattr(result, name)) for name in names},
                })
                return result
            return wrapped

        self._patch(losses, "compute_online_twin_q_td_loss", td_factory)
        self._patch(losses, "compute_stage3_min_twin_q_actor_loss", actor_q_factory)
        self._patch(losses, "compute_stage3_actor_objective", actor_objective_factory)

        def clip_factory(original):
            def wrapped(parameters, *args, **kwargs):
                parameters = list(parameters)
                before = {
                    self.parameter_names[id(parameter)]: tensor_record(parameter.grad)
                    for parameter in parameters if parameter.grad is not None
                }
                result = original(parameters, *args, **kwargs)
                after = {
                    self.parameter_names[id(parameter)]: tensor_record(parameter.grad)
                    for parameter in parameters if parameter.grad is not None
                }
                self.records["gradient_tensors"].append({"before_clip": before, "after_clip": after})
                return result
            return wrapped

        self._patch(self.torch.nn.utils, "clip_grad_norm_", clip_factory)

        def delta_factory(original):
            def wrapped(snapshot, *named_modules, actor_groups=False):
                result = original(snapshot, *named_modules, actor_groups=actor_groups)
                deltas = {}
                for owner, module in named_modules:
                    for name, parameter in module.named_parameters():
                        key = f"{owner}.{name}"
                        if key in snapshot:
                            deltas[key] = tensor_record(
                                parameter.detach().cpu().float() - snapshot[key].float()
                            )
                self.records["parameter_deltas"].append({"summary": result, "tensors": deltas})
                return result
            return wrapped

        self._patch(g4, "_delta_summary", delta_factory)

        def polyak_factory(original):
            def wrapped(online, target, **kwargs):
                before = {name: value.detach().cpu().clone() for name, value in target.state_dict().items()}
                result = original(online, target, **kwargs)
                deltas = {
                    name: tensor_record(value.detach().cpu().float() - before[name].float())
                    for name, value in target.state_dict().items()
                }
                self.records["polyak_deltas"].append({
                    "target": kwargs["target_name"], "summary": result, "delta_tensors": deltas,
                })
                return result
            return wrapped

        self._patch(training_cycle, "polyak_update_verified", polyak_factory)
        return self

    def __exit__(self, *_args):
        for restore in reversed(self.restorers):
            restore()


def _remove_performance(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: _remove_performance(item)
            for key, item in value.items()
            if key not in {"wall_seconds", "latency_seconds", "cpu_rss_mib", "peak_vram_mib"}
        }
    if isinstance(value, list):
        return [_remove_performance(item) for item in value]
    return value


def _credit_state(ledger) -> dict[str, Any]:
    state = ledger.state_dict()
    state["available"] = ledger.snapshot().available
    return state


def _boundary() -> dict[str, Any]:
    return {
        "episode_sealed": True, "active_episode": False,
        "request_in_flight": False, "partial_macro": False,
        "learner_update_committed": True, "pending_gradients": False,
        "pending_optimizer_steps": 0, "pending_accumulation_microbatches": 0,
    }


def _counters(cycle: int) -> dict[str, int]:
    return {
        "learner_cycles": cycle, "critic_updates": 2 * cycle,
        "actor_updates": cycle, "polyak_updates_per_target": 2 * cycle,
        "publication_count": 0,
    }


def _revision(binding_id: str) -> dict[str, Any]:
    return {
        "active_revision": binding_id, "pending_revision": None,
        "previous_revision": None, "episode_revision": None,
        "policy_epoch": 0, "publication_count": 0,
    }


DURABLE_STATE = {
    "production_wal": "UNSUPPORTED_IN_ISOLATED_G5P",
    "production_outbox": "UNSUPPORTED_IN_ISOLATED_G5P",
    "production_publication": "UNSUPPORTED_IN_ISOLATED_G5P",
}


def _initialize_worker(config: Mapping[str, Any]) -> dict[str, Any]:
    import torch
    import preflight_stage3_gpu as g4
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.stage3.checkpoint import actor_frozen_state_digest
    from forcesmolvla.rft.stage3.parent import (
        load_parent_binding, preflight_parent_binding, validate_parent_binding_semantics,
    )
    from forcesmolvla.rft.stage3.update_credit import UpdateCreditLedger
    from forcesmolvla.rft.canonical_state import canonical_digest

    g4p_path = _config_path(config["baseline"]["g4p_config"])
    g4p_config = g4.validate_gpu_preflight_config(yaml.safe_load(g4p_path.read_text()))
    binding_path = g4._resolve(g4p_config["parent_binding"]["path"])
    require(not torch.cuda.is_initialized(), "G5P_CUDA_INITIALIZED_BEFORE_PARENT_PREFLIGHT")
    g0a = preflight_parent_binding(binding_path)
    require(g0a["tool_status"] == "PASS", "G5P_PARENT_PREFLIGHT")
    binding = validate_parent_binding_semantics(load_parent_binding(binding_path))
    parent_records = g4._selected_parent_records(binding)
    parent_before = g4._hash_parent_records(parent_records)
    device, environment = g4._freeze_environment(g4p_config)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.cuda.reset_peak_memory_stats(device)
    actor, q1, q2, q1_target, q2_target, parent_load = g4._strict_load_parents(
        binding, g4p_config, device
    )
    actor_optimizer, critic_optimizer, ownership = g4._optimizer_factory(
        actor, q1, q2, q1_target, q2_target, g4p_config
    )
    critic_batch, actor_batch, data_evidence, flow_counter = g4._load_real_batches(
        actor, binding, g4p_config, device
    )
    seeds = config["determinism"]["seeds"]
    generators = {
        "td_noise": torch.Generator(device=device).manual_seed(seeds["td_noise"]),
        "fm_noise": torch.Generator(device=device).manual_seed(seeds["fm_noise"]),
        "fm_time": torch.Generator(device=device).manual_seed(seeds["fm_time"]),
        "actor_q_noise": torch.Generator(device=device).manual_seed(seeds["actor_q_noise"]),
        "sample_schedule": torch.Generator(device="cpu").manual_seed(seeds["sample_schedule"]),
    }
    schedule = _Schedule(
        data_evidence["critic_batch"]["rows"], data_evidence["actor_batch"]["rows"],
        generators["sample_schedule"],
    )
    R_uids = [
        f"{kind}:{row['row_identity']}"
        for kind in ("critic", "actor")
        for row in data_evidence[f"{kind}_batch"]["rows"]
        if row["origin_pool"] == "synthetic_preflight_R_only"
    ]
    D_uids = [
        f"{kind}:{row['row_identity']}"
        for kind in ("critic", "actor")
        for row in data_evidence[f"{kind}_batch"]["rows"]
        if row["origin_pool"] == "offline_D"
    ]
    canonical_index = {
        f"{kind}:{row['row_identity']}": {
            "row_identity": row["row_identity"], "origin_pool": row["origin_pool"],
            "batch_kind": kind, "source": "frozen_real_observation_pipeline",
        }
        for kind in ("critic", "actor")
        for row in data_evidence[f"{kind}_batch"]["rows"]
    }
    replay = {
        "storage_kind": "isolated_read_only_preflight_index",
        "canonical_index": canonical_index,
        "canonical_index_sha256": canonical_digest(canonical_index),
        "R_membership_uids": R_uids, "D_membership_uids": D_uids,
        "R_watermark": len(R_uids), "D_watermark": len(D_uids),
        "episode_finalization_state": "sealed",
        "R_SOURCE": "synthetic_preflight_R_only", "writes_formal_online_replay": False,
    }
    ledger = UpdateCreditLedger(credits_per_transition=1, credits_per_joint_cycle=1)
    for uid in R_uids:
        require(ledger.mint_for_unique_online_transition(uid), "G5P_CREDIT_MINT")
    bindings = {
        "parent_binding_id": binding["binding_id"],
        "parent_binding_sha256": sha256_file(binding_path),
        "actor_parent_sha256": binding["actor_parent"]["sha256"],
        "critic_parent_sha256": {
            item["logical_role"]: item["sha256"] for item in binding["critic_parent"]["artifacts"]
        },
        "target_critic_parent_sha256": {
            item["logical_role"]: item["sha256"] for item in binding["target_critic_parent"]["artifacts"]
        },
        "actor_frozen_parent_digest": actor_frozen_state_digest(actor),
        "config_sha256": sha256_file(CONFIG),
        "reward_source_sha256": g4p_config["data"]["transition_manifest_sha256"],
        "source_sha256": _canonical_json_sha256({
            "checkpoint": sha256_file(CHECKPOINT_SOURCE), "tool": sha256_file(Path(__file__)),
            "g4p_tool": sha256_file(G4P_TOOL),
        }),
        "normalizer_sha256": binding["normalizer_binding"]["sha256"],
        "action_contract_sha256": binding["action_contract_binding"]["sha256"],
        "task_feature_sha256": binding["task_feature_binding"]["logical_object_sha256"],
        "calibration_sha256": binding["calibration_binding"]["sha256"],
        "runtime_contract_sha256": binding["runtime_contract_binding"]["sha256"],
    }
    modules = {
        "actor": actor, "q1": q1, "q2": q2,
        "q1_target": q1_target, "q2_target": q2_target,
    }
    environment.update({
        "pid": os.getpid(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "autocast_dtype": "torch.bfloat16", "torch_compile": False,
        "data_augmentation": False, "num_workers": 0,
        "python_seed": seeds["python_numpy_torch"], "numpy_seed": seeds["python_numpy_torch"],
        "torch_seed": seeds["python_numpy_torch"],
    })
    return {
        "config": config, "g4p_config": g4p_config, "binding": binding,
        "binding_path": binding_path, "device": device, "environment": environment,
        "parent_records": parent_records, "parent_before": parent_before,
        "parent_load": parent_load, "ownership": ownership,
        "frozen_digest_initial": frozen_state_digest(actor),
        "modules": modules, "actor_optimizer": actor_optimizer,
        "critic_optimizer": critic_optimizer, "critic_batch": critic_batch,
        "actor_batch": actor_batch, "data_evidence": data_evidence,
        "flow_counter": flow_counter, "generators": generators,
        "schedule": schedule, "replay": replay, "ledger": ledger,
        "bindings": bindings, "revision": _revision(binding["binding_id"]),
    }


def _live_state(context: Mapping[str, Any], cycle: int) -> dict[str, Any]:
    from forcesmolvla.rft.stage3.checkpoint import stage3_exact_live_state

    return stage3_exact_live_state(
        modules=context["modules"], actor_optimizer=context["actor_optimizer"],
        critic_optimizer=context["critic_optimizer"], generators=context["generators"],
        sampler_state=context["schedule"].state_dict(), replay_state=context["replay"],
        credit_state=_credit_state(context["ledger"]), counters=_counters(cycle),
        revision_state=context["revision"], durable_state=DURABLE_STATE,
        boundary=_boundary(), bindings=context["bindings"],
        scheduler_states={"actor": None, "critic": None}, scaler_state=None,
    )


def _run_cycle(context: dict[str, Any], cycle: int) -> dict[str, Any]:
    import torch
    import preflight_stage3_gpu as g4
    from unittest.mock import patch
    from forcesmolvla.rft import losses as stage2_losses
    from forcesmolvla.rft.frozen_vlm_trainability import frozen_state_digest
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none

    trace_capture = _CycleTrace(context)
    with trace_capture:
        critic_order, actor_order, sample_uid_order = context["schedule"].draw()
        critic_batch = _index_batch(context["critic_batch"], critic_order)
        actor_batch = _index_batch(context["actor_batch"], actor_order)
        critic_reports = []
        forbidden_counts = {"calql": 0, "cql": 0, "random": 0, "mc": 0}

        def forbidden(name: str):
            def call(*_args, **_kwargs):
                forbidden_counts[name] += 1
                raise G5PError(f"G5P_FORBIDDEN_ONLINE_LOSS:{name}")
            return call

        with (
            patch.object(stage2_losses, "compute_calql_penalty", side_effect=forbidden("calql")),
            patch.object(stage2_losses, "compute_twin_q_critic_loss", side_effect=forbidden("cql")),
            patch.object(stage2_losses, "evaluate_calql_candidates", side_effect=forbidden("random")),
            patch.object(stage2_losses, "validate_mc_return_recurrence", side_effect=forbidden("mc")),
        ):
            for substep in (1, 2):
                critic_reports.append(g4._critic_step(
                    cycle=cycle, substep=substep,
                    actor=context["modules"]["actor"], q1=context["modules"]["q1"],
                    q2=context["modules"]["q2"], q1_target=context["modules"]["q1_target"],
                    q2_target=context["modules"]["q2_target"],
                    optimizer=context["critic_optimizer"], batch=critic_batch,
                    flow_counter=context["flow_counter"],
                    noise_generator=context["generators"]["td_noise"],
                    config=context["g4p_config"],
                ))
            actor_report = g4._actor_step(
                cycle=cycle, actor=context["modules"]["actor"],
                q1=context["modules"]["q1"], q2=context["modules"]["q2"],
                q1_target=context["modules"]["q1_target"], q2_target=context["modules"]["q2_target"],
                optimizer=context["actor_optimizer"], batch=actor_batch,
                origin_pools=[context["data_evidence"]["actor_batch"]["rows"][index]["origin_pool"] for index in actor_order],
                flow_counter=context["flow_counter"],
                fm_noise_generator=context["generators"]["fm_noise"],
                fm_time_generator=context["generators"]["fm_time"],
                q_noise_generator=context["generators"]["actor_q_noise"],
                config=context["g4p_config"],
            )
        require(forbidden_counts == {"calql": 0, "cql": 0, "random": 0, "mc": 0}, "G5P_FORBIDDEN_ONLINE_LOSS_CALLED")
    context["ledger"].consume_joint_cycle()
    ensure_all_gradients_none(*context["modules"].values())
    require(frozen_state_digest(context["modules"]["actor"]) == context["frozen_digest_initial"], "G5P_FROZEN_ACTOR_CHANGED")
    torch.cuda.synchronize(context["device"])
    trace = {
        "schema_version": "forcesmolvla_g5p_joint_cycle_trace.v1",
        "cycle": cycle, "sample_uid_order": sample_uid_order,
        "sample_schedule_state_after": context["schedule"].state_dict(),
        "critic_updates": _remove_performance(critic_reports),
        "actor_update": _remove_performance(actor_report),
        "observed_tensors": trace_capture.records,
        "credits_after": _credit_state(context["ledger"]),
        "counters_after": _counters(cycle),
        "revision_state": context["revision"],
    }
    trace["canonical_trace_digest"] = _canonical_json_sha256(trace)
    return trace


def _save_checkpoint(context: Mapping[str, Any], checkpoint: Path) -> tuple[dict, float]:
    from forcesmolvla.rft.stage3.checkpoint import save_exact_resume_checkpoint

    started = time.perf_counter()
    result = save_exact_resume_checkpoint(
        checkpoint, modules=context["modules"],
        actor_optimizer=context["actor_optimizer"], critic_optimizer=context["critic_optimizer"],
        generators=context["generators"], sampler_state=context["schedule"].state_dict(),
        replay_state=context["replay"], credit_state=_credit_state(context["ledger"]),
        counters=_counters(1), revision_state=context["revision"],
        bindings=context["bindings"], boundary=_boundary(),
        scheduler_states={"actor": None, "critic": None}, scaler_state=None,
        minimum_free_copies=context["config"]["checkpoint"]["minimum_free_copies"],
    )
    return result, time.perf_counter() - started


def _restore_checkpoint(context: dict[str, Any], checkpoint: Path) -> tuple[dict, float]:
    from forcesmolvla.rft.stage3.checkpoint import strict_load_exact_resume_checkpoint
    from forcesmolvla.rft.stage3.update_credit import UpdateCreditLedger

    started = time.perf_counter()
    restored = strict_load_exact_resume_checkpoint(
        checkpoint, modules=context["modules"],
        actor_optimizer=context["actor_optimizer"], critic_optimizer=context["critic_optimizer"],
        generators=context["generators"], expected_bindings=context["bindings"],
    )
    context["schedule"].load_state_dict(restored["sampler"])
    context["ledger"] = UpdateCreditLedger.from_state_dict(restored["credits"])
    require(_credit_state(context["ledger"]) == restored["credits"], "G5P_RESTORED_CREDIT_STATE")
    return restored, time.perf_counter() - started


def _worker(branch: str, checkpoint: Path, result_path: Path) -> None:
    import torch
    from forcesmolvla.rft.stage3.checkpoint import validate_exact_resume_checkpoint

    config = _load_config()
    _validate_frozen_baseline(config)
    context = _initialize_worker(config)
    initial = _live_state(context, 0)
    traces = []
    checkpoint_save_seconds = checkpoint_load_seconds = 0.0
    checkpoint_size_bytes = 0
    checkpoint_state = restored_state = None
    if branch == "A":
        traces.append(_run_cycle(context, 1))
        intermediate = _live_state(context, 1)
        traces.append(_run_cycle(context, 2))
        final = _live_state(context, 2)
        cycles = [1, 2]
    elif branch == "B1":
        traces.append(_run_cycle(context, 1))
        intermediate = _live_state(context, 1)
        before_save = intermediate
        saved, checkpoint_save_seconds = _save_checkpoint(context, checkpoint)
        after_save = _live_state(context, 1)
        require(before_save == after_save, "G5P_CHECKPOINT_SAVE_SIDE_EFFECT")
        validated = validate_exact_resume_checkpoint(checkpoint)
        checkpoint_state = validated["metadata"]["canonical_state"]
        require(checkpoint_state == intermediate, "G5P_CHECKPOINT_INTERMEDIATE_STATE_MISMATCH")
        checkpoint_size_bytes = saved["checkpoint_size_bytes"]
        final = intermediate
        cycles = [1]
    else:
        restored, checkpoint_load_seconds = _restore_checkpoint(context, checkpoint)
        checkpoint_state = restored["canonical_state"]
        restored_state = _live_state(context, 1)
        require(restored_state == checkpoint_state, "G5P_STRICT_RESTORE_STATE_MISMATCH")
        intermediate = restored_state
        traces.append(_run_cycle(context, 2))
        final = _live_state(context, 2)
        cycles = [2]
    parent_after = __import__("preflight_stage3_gpu")._hash_parent_records(context["parent_records"])
    require(parent_after == context["parent_before"], "G5P_WORKER_PARENT_MUTATION")
    torch.cuda.synchronize(context["device"])
    result = {
        "branch": branch, "pid": os.getpid(), "status": "PASS", "cycles": cycles,
        "environment": context["environment"],
        "initial_state_digest": initial["canonical_content_digest"],
        "intermediate_state_digest": intermediate["canonical_content_digest"],
        "final_state_digest": final["canonical_content_digest"],
        "intermediate_state": intermediate, "final_state": final,
        "checkpoint_state": checkpoint_state, "restored_state": restored_state,
        "traces": traces, "checkpoint_save_seconds": checkpoint_save_seconds,
        "checkpoint_load_seconds": checkpoint_load_seconds,
        "checkpoint_size_bytes": checkpoint_size_bytes,
        "peak_vram_mib": torch.cuda.max_memory_allocated(context["device"]) / 1024**2,
        "peak_cpu_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
        "parent_checkpoint_sha_unchanged": True,
        "production_actor_checkpoint_unchanged": True,
        "REAL_ONLINE_R_USED": False, "R_SOURCE": "synthetic_preflight_R_only",
        "robot_connection_count": 0, "robot_command_count": 0,
    }
    _atomic_json(result_path, result)
    context.clear()
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--worker", choices=("A", "B1", "B2"))
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--result", type=Path)
    args = parser.parse_args()
    if args.worker:
        require(args.run_root is not None and args.checkpoint is not None and args.result is not None, "G5P_WORKER_ARGUMENTS")
        _worker(args.worker, args.checkpoint, args.result)
        return 0
    require(args.run, "G5P_EXPLICIT_RUN_REQUIRED")
    _coordinator()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
