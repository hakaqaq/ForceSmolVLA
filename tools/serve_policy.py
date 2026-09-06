#!/usr/bin/env python3
"""ForceSmolVLA model inference service.

This process owns PyTorch/LeRobot/checkpoint loading only.  It never imports
ROS, opens cameras, connects to Franky, or sends robot commands.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import socket
import threading
import time
from typing import Any

import numpy as np
import torch

from forcesmolvla.checkpoint import sha256_file
from forcesmolvla.action_delta import (
    ActionDeltaProcessor,
    ActionSafetyProfile,
    MODEL_GRIPPER_CANDIDATE_RANGE_M,
    decode_binary_gripper_width,
)
from forcesmolvla.inference import (
    CLOCK_DOMAIN,
    HORIZON,
    PROTOCOL_VERSION,
    load_checkpoint_inference_contract,
    prepare_policy_inputs,
)
from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
from forcesmolvla.rules import load_and_validate_rulespec
from forcesmolvla.training_data import load_checkpoint_runtime_artifacts
from forcesmolvla.rft.online.transition_authority import (
    ONLINE_SEMANTICS_VERSION,
    normalized_behavior_residual,
)


MAX_REQUEST_BYTES = 6 * 1024 * 1024
DEPLOYMENT_BINDING_SCHEMA = "forcesmolvla-live-deployment-binding-v1"
DEPLOYMENT_PROFILE_SCHEMA = "forcesmolvla-deployment-profile-v1"


def source_tree_sha256(root: Path) -> str:
    """Hash the exact local Python implementation used by model inference."""
    root = root.resolve()
    files = [root / "tools/serve_policy.py"]
    files.extend(sorted((root / "src/forcesmolvla").glob("*.py")))
    files.extend(sorted((root / "vendor/lerobot/src/lerobot/policies/smolvla").glob("*.py")))
    mapping = {
        str(path.relative_to(root)): sha256_file(path)
        for path in files
        if path.is_file()
    }
    encoded = json.dumps(mapping, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _approved_rulespec_for_execution(rulespec: dict[str, Any]) -> None:
    if rulespec.get("mode") == "test_only":
        raise PermissionError("TEST_ONLY_RULESPEC_CANNOT_AUTHORIZE_ROBOT_EXECUTION")
    if rulespec.get("mode") != "development_only":
        raise PermissionError("ROBOT_EXECUTION_RULESPEC_MODE_INVALID")
    if (
        rulespec.get("artifact_status") != "development_only"
        or rulespec.get("acceptance_status") != "development_only"
    ):
        raise PermissionError("ROBOT_EXECUTION_RULESPEC_DEVELOPMENT_STATUS_INVALID")
    if rulespec.get("approval", {}).get("status") != "approved":
        raise PermissionError("ROBOT_EXECUTION_RULESPEC_APPROVAL_MISSING")
    approval = rulespec["approval"]
    if any(
        not isinstance(approval.get(field), str) or not approval[field].strip()
        for field in (
            "approval_id",
            "approver_identity",
            "approver_role",
            "approved_at",
        )
    ):
        raise PermissionError("ROBOT_EXECUTION_RULESPEC_APPROVAL_PROVENANCE_MISSING")
    if rulespec.get("signature", {}).get("status") != "configuration_pending":
        raise PermissionError("ROBOT_EXECUTION_RULESPEC_SIGNATURE_STATUS_INVALID")
    unresolved = [
        rule.get("rule_id", "UNKNOWN")
        for rule in rulespec.get("rules", ())
        if rule.get("threshold", {}).get("value") is None
        or rule.get("threshold", {}).get("approval_status") != "approved"
    ]
    if unresolved:
        raise PermissionError(f"ROBOT_EXECUTION_RULESPEC_UNRESOLVED:{unresolved}")


def load_deployment_binding(
    path: Path,
    trusted_sha256: str,
    *,
    model_sha256: str,
    rulespec_sha256: str,
    server_source_sha256: str,
) -> tuple[dict[str, Any], str]:
    path = path.resolve()
    if len(trusted_sha256) != 64 or any(c not in "0123456789abcdef" for c in trusted_sha256):
        raise PermissionError("TRUSTED_DEPLOYMENT_BINDING_SHA256_INVALID")
    actual_sha256 = sha256_file(path)
    if actual_sha256 != trusted_sha256:
        raise PermissionError("DEPLOYMENT_BINDING_TRUST_ANCHOR_MISMATCH")
    binding = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact_status",
        "model_sha256",
        "rulespec_sha256",
        "server_source_sha256",
        "client_source_sha256",
        "state_pose_max_age_ms",
        "camera_max_age_ms",
        "max_intercamera_skew_ms",
        "gripper_max_age_ms",
        "controller_ack_timeout_ms",
        "approval",
    }
    if not isinstance(binding, dict) or set(binding) != required:
        raise PermissionError("DEPLOYMENT_BINDING_FIELDS_MISMATCH")
    if binding["schema_version"] != DEPLOYMENT_BINDING_SCHEMA:
        raise PermissionError("DEPLOYMENT_BINDING_SCHEMA_MISMATCH")
    if binding["artifact_status"] != "approved":
        raise PermissionError("DEPLOYMENT_BINDING_NOT_APPROVED")
    if binding.get("approval", {}).get("status") != "approved":
        raise PermissionError("DEPLOYMENT_BINDING_APPROVAL_MISSING")
    approval = binding["approval"]
    if set(approval) != {
        "status",
        "approval_id",
        "approver_identity",
        "approver_role",
        "approved_at",
    } or any(
        not isinstance(approval.get(field), str) or not approval[field].strip()
        for field in (
            "approval_id",
            "approver_identity",
            "approver_role",
            "approved_at",
        )
    ):
        raise PermissionError("DEPLOYMENT_BINDING_APPROVAL_PROVENANCE_MISSING")
    expected = {
        "model_sha256": model_sha256,
        "rulespec_sha256": rulespec_sha256,
        "server_source_sha256": server_source_sha256,
    }
    for field, value in expected.items():
        if binding[field] != value:
            raise PermissionError(f"DEPLOYMENT_BINDING_{field.upper()}_MISMATCH")
    client_hash = binding["client_source_sha256"]
    if not isinstance(client_hash, str) or len(client_hash) != 64 or any(
        c not in "0123456789abcdef" for c in client_hash
    ):
        raise PermissionError("DEPLOYMENT_BINDING_CLIENT_SOURCE_SHA256_INVALID")
    for field in (
        "state_pose_max_age_ms",
        "camera_max_age_ms",
        "max_intercamera_skew_ms",
        "gripper_max_age_ms",
        "controller_ack_timeout_ms",
    ):
        value = binding[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise PermissionError(f"DEPLOYMENT_BINDING_{field.upper()}_INVALID")
    if float(binding["controller_ack_timeout_ms"]) >= 1000.0 / 30.0:
        raise PermissionError(
            "DEPLOYMENT_BINDING_CONTROLLER_ACK_TIMEOUT_MUST_FIT_ONE_30HZ_SLOT"
        )
    return binding, actual_sha256


def development_execution_metadata(
    binding: dict[str, Any] | None, binding_sha256: str | None
) -> dict[str, Any]:
    enabled = binding is not None
    return {
        "robot_execution_allowed": enabled,
        "robot_execution_mode": (
            "approved_binding_supervised_development" if enabled else "disabled"
        ),
        "development_execution_override": enabled,
        "deployment_binding_sha256": binding_sha256,
        "required_client_source_sha256": (
            None if binding is None else binding["client_source_sha256"]
        ),
        "controller_ack_timeout_ms": (
            None if binding is None else float(binding["controller_ack_timeout_ms"])
        ),
    }


def development_live_contract(contract: Any, binding: dict[str, Any] | None) -> Any:
    if binding is None:
        return contract
    return replace(
        contract,
        state_pose_max_age_ms=float(binding["state_pose_max_age_ms"]),
        camera_max_age_ms=float(binding["camera_max_age_ms"]),
        max_intercamera_skew_ms=float(binding["max_intercamera_skew_ms"]),
    )


def bind_policy_action_safety(
    policy: ForceSmolVLAPolicy,
    rulespec: dict[str, Any],
    *,
    rules_sha256: str,
    approved_development_execution: bool,
) -> None:
    if not approved_development_execution:
        policy.bind_action_safety_rules(rulespec, rules_sha256=rules_sha256)
        return
    # Reuse the frozen intrinsic-rule parser without changing model/training
    # source or its P5-P8 bindings.  All numerical values remain the exact
    # approved development RuleSpec values; only the parser's legacy mode gate
    # is adapted here, after independent RuleSpec and deployment-binding trust
    # checks have already succeeded.
    parser_view = copy.deepcopy(rulespec)
    parser_view["mode"] = "test_only"
    parser_view["artifact_status"] = "development_only"
    parser_view["acceptance_status"] = "development_only"
    profile = ActionSafetyProfile.from_rulespec(
        parser_view, rules_sha256=rules_sha256
    )
    policy._action_safety_profile = replace(profile, mode="development_only")


class InferenceEngine:
    def __init__(
        self,
        checkpoint: Path,
        rulespec_path: Path,
        schema_path: Path,
        device: torch.device,
        *,
        allow_development_robot_execution: bool = False,
        allow_development_policy_execution_smoke: bool = False,
        deployment_binding_path: Path | None = None,
        trusted_deployment_binding_sha256: str | None = None,
    ) -> None:
        self.checkpoint = checkpoint.resolve()
        self.device = device
        self.checkpoint_contract = load_checkpoint_inference_contract(self.checkpoint)
        self.runtime_artifacts = load_checkpoint_runtime_artifacts(self.checkpoint)
        self.rulespec = load_and_validate_rulespec(
            rulespec_path.resolve(), schema_path.resolve(), formal=False
        )
        self.rulespec_sha256 = sha256_file(rulespec_path.resolve())
        self.server_source_sha256 = source_tree_sha256(
            Path(__file__).resolve().parents[1]
        )
        self.model_sha256 = sha256_file(self.checkpoint / "model.safetensors")
        self.deployment_binding: dict[str, Any] | None = None
        self.deployment_binding_sha256: str | None = None
        if allow_development_robot_execution and allow_development_policy_execution_smoke:
            raise PermissionError("ROBOT_EXECUTION_MODE_CONFLICT")
        if allow_development_robot_execution:
            if (
                deployment_binding_path is None
                or trusted_deployment_binding_sha256 is None
            ):
                raise PermissionError(
                    "ROBOT_EXECUTION_REQUIRES_DEPLOYMENT_BINDING_AND_TRUSTED_SHA256"
                )
            _approved_rulespec_for_execution(self.rulespec)
            (
                self.deployment_binding,
                self.deployment_binding_sha256,
            ) = load_deployment_binding(
                deployment_binding_path,
                trusted_deployment_binding_sha256,
                model_sha256=self.model_sha256,
                rulespec_sha256=self.rulespec_sha256,
                server_source_sha256=self.server_source_sha256,
            )
        elif allow_development_policy_execution_smoke:
            _approved_rulespec_for_execution(self.rulespec)
        elif (
            deployment_binding_path is not None
            or trusted_deployment_binding_sha256 is not None
        ):
            raise PermissionError(
                "DEPLOYMENT_BINDING_IS_ONLY_VALID_WITH_EXPLICIT_ROBOT_EXECUTION_OPT_IN"
            )
        self.contract = development_live_contract(
            self.checkpoint_contract, self.deployment_binding
        )
        self.policy = ForceSmolVLAPolicy.from_pretrained(
            self.checkpoint,
            local_files_only=True,
            strict=True,
            artifact_use="development",
        )
        self.policy.bind_runtime_artifacts(self.runtime_artifacts)
        bind_policy_action_safety(
            self.policy,
            self.rulespec,
            rules_sha256=self.rulespec_sha256,
            approved_development_execution=(
                self.deployment_binding is not None
                or allow_development_policy_execution_smoke
            ),
        )
        self.policy.to(self.device)
        self.policy.eval()
        self.residual_actor: torch.nn.Module | None = None
        self._base_lock = threading.Lock()
        self._residual_lock = threading.Lock()
        self._lock = self._base_lock
        artifact_manifest = json.loads(
            (self.checkpoint / "artifact_manifest.json").read_text(encoding="utf-8")
        )
        self.metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "service_role": "model_inference_only",
            "robot_io_present": False,
            "hostname": socket.gethostname(),
            "clock_domain_id": CLOCK_DOMAIN,
            "checkpoint": str(self.checkpoint),
            "checkpoint_acceptance_status": artifact_manifest["acceptance_status"],
            "formal_eligible": artifact_manifest["formal_eligible"],
            "model_sha256": self.model_sha256,
            "server_source_sha256": self.server_source_sha256,
            "rulespec_mode": self.rulespec["mode"],
            "rulespec_artifact_status": self.rulespec["artifact_status"],
            "rulespec_approval_status": self.rulespec["approval"]["status"],
            "rulespec_signature_status": self.rulespec["signature"]["status"],
            "rulespec_sha256": self.rulespec_sha256,
            **development_execution_metadata(
                self.deployment_binding, self.deployment_binding_sha256
            ),
            "dataset_repo_id": self.contract.repo_id,
            "conversion_manifest_sha256": self.contract.conversion_manifest_sha256,
            "tool_profile_sha256": self.contract.tool_profile_sha256,
            "calibration_id": self.contract.calibration_id,
            "fps": 30,
            "action_horizon": HORIZON,
            "action_semantics": "absolute TCP xyz+rpy plus absolute gripper width in metres",
            "gripper_decoder": {
                "type": "binary_width",
                "model_candidate_tolerance_m": [-0.01, 0.095],
                "switch_width_m": 0.0425,
                "decoded_widths_m": [0.0, 0.085],
                "finite_candidate_saturation_m": [-0.01, 0.095],
                "clipping": True,
            },
            "camera_order": ["camera1:D435-third-person", "camera2:D405-wrist"],
            "camera_runtime_profile": (
                "deployment_binding_v1"
                if self.deployment_binding is not None
                else "checkpoint_v4_1"
            ),
            "checkpoint_camera_max_age_ms": self.checkpoint_contract.camera_max_age_ms,
            "checkpoint_max_intercamera_skew_ms": (
                self.checkpoint_contract.max_intercamera_skew_ms
            ),
            "checkpoint_max_pose_age_ms": self.checkpoint_contract.max_pose_age_ms,
            "max_pose_age_ms": self.contract.max_pose_age_ms,
            "state_pose_max_age_ms": self.contract.state_pose_max_age_ms,
            "gripper_max_age_ms": (
                None
                if self.deployment_binding is None
                else float(self.deployment_binding["gripper_max_age_ms"])
            ),
            "camera_max_age_ms": self.contract.camera_max_age_ms,
            "max_intercamera_skew_ms": self.contract.max_intercamera_skew_ms,
            "max_wrench_source_gap_ms": self.contract.max_wrench_source_gap_ms,
            "filter_warmup_samples": self.contract.filter_warmup_samples,
            "calibration_bundle": self.contract.calibration_bundle,
            "wrench_geometry_spec": self.contract.wrench_geometry_spec,
            "converter_runtime_spec": self.contract.converter_runtime_spec,
        }

    def infer(self, request: dict[str, Any]) -> dict[str, Any]:
        started_ns = time.monotonic_ns()
        if self.deployment_binding is not None:
            provenance = request.get("provenance")
            if not isinstance(provenance, dict):
                raise ValueError("INFERENCE_PROVENANCE_MISSING")
            t_ref_ns = int(provenance.get("t_ref_ns", 0))
            gripper_receive_ns = int(
                provenance.get("gripper_receive_monotonic_ns", 0)
            )
            gripper_age_ms = (t_ref_ns - gripper_receive_ns) / 1.0e6
            if (
                t_ref_ns <= 0
                or gripper_receive_ns <= 0
                or not 0.0 <= gripper_age_ms
                <= float(self.deployment_binding["gripper_max_age_ms"])
            ):
                raise RuntimeError("INFERENCE_GRIPPER_AGE_EXCEEDED")
        batch, context = prepare_policy_inputs(
            self.policy,
            request,
            self.runtime_artifacts,
            self.contract,
            self.device,
        )
        with self._lock:
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.bfloat16,
                enabled=self.device.type == "cuda",
            ):
                if self.residual_actor is None:
                    base_normalized = None
                    actions = self.policy.predict_action_chunk(
                        batch, chunk_context=context
                    )
                else:
                    base_normalized, actions = self.policy._predict_action_chunks(
                        batch, chunk_context=context
                    )
        completed_ns = time.monotonic_ns()
        actions_cpu = actions[0].detach().to(torch.float32).cpu()
        if tuple(actions_cpu.shape) != (HORIZON, 7):
            raise RuntimeError(f"MODEL_ACTION_SHAPE_MISMATCH:{tuple(actions_cpu.shape)}")
        response = {
            "protocol_version": PROTOCOL_VERSION,
            "request_id": request["request_id"],
            "chunk_id": request["chunk_id"],
            "t_ref_ns": int(request["provenance"]["t_ref_ns"]),
            "server_started_monotonic_ns": started_ns,
            "server_completed_monotonic_ns": completed_ns,
            "inference_latency_ms": (completed_ns - started_ns) / 1.0e6,
            "actions": actions_cpu.tolist(),
            "action_semantics": "absolute7",
            "valid_horizon": HORIZON,
            "acceptance_status": "development_only",
            "formal_eligible": False,
        }
        if base_normalized is not None:
            response.update(
                base_normalized_actions=base_normalized[0].detach().float().cpu().tolist(),
                base_actions_absolute7=actions[0].detach().float().cpu().tolist(),
            )
        return response

    def residual_decision(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply the episode-pinned residual Actor to one selected dispatch slot."""

        if self.residual_actor is None:
            raise RuntimeError("RESIDUAL_DECISION_ACTOR_UNAVAILABLE")
        state = np.asarray(request.get("state7"), dtype=np.float64)
        wrench = np.asarray(request.get("wrench6"), dtype=np.float64)
        wrench_delta = np.asarray(request.get("wrench_delta6"), dtype=np.float64)
        base_absolute = np.asarray(
            request.get("base_absolute_action7"), dtype=np.float64
        )
        decision_ns = request.get("decision_monotonic_ns")
        if (
            state.shape != (7,)
            or wrench.shape != (6,)
            or wrench_delta.shape != (6,)
            or base_absolute.shape != (7,)
            or not all(
                np.isfinite(value).all()
                for value in (state, wrench, wrench_delta, base_absolute)
            )
            or isinstance(decision_ns, bool)
            or not isinstance(decision_ns, int)
            or decision_ns <= 0
        ):
            raise ValueError("RESIDUAL_DECISION_CONTEXT_INVALID")
        normalizer = self.runtime_artifacts.normalizer
        base_normalized, _accepted, _zero = normalized_behavior_residual(
            base_absolute_k7=base_absolute[None, :],
            accepted_absolute_k7=base_absolute[None, :],
            decision_state7=state,
            normalize_delta7=normalizer.delta_action7.apply,
            valid_mask=np.ones(1, dtype=np.bool_),
        )
        normalized_state = normalizer.state7.apply(state).astype(np.float32)
        normalized_wrench = normalizer.wrench6.apply(wrench).astype(np.float32)
        normalized_wrench_delta = (
            wrench_delta / normalizer.wrench6.std
        ).astype(np.float32)
        actor_device = next(self.residual_actor.parameters()).device
        with self._residual_lock, torch.no_grad():
            residual = self.residual_actor(
                normalized_state7=torch.from_numpy(normalized_state[None]).to(actor_device),
                normalized_wrench6=torch.from_numpy(normalized_wrench[None]).to(actor_device),
                normalized_wrench_delta6=torch.from_numpy(
                    normalized_wrench_delta[None]
                ).to(actor_device),
                base_action6=torch.from_numpy(base_normalized[:, :6]).to(actor_device),
            )[0].detach().float().cpu().numpy()
        composed_normalized = base_normalized.copy()
        composed_normalized[0, :6] += residual
        composed_delta = normalizer.delta_action7.inverse(composed_normalized)
        composed_delta[..., 6] = np.clip(
            composed_delta[..., 6], *MODEL_GRIPPER_CANDIDATE_RANGE_M
        )
        composed_delta = decode_binary_gripper_width(composed_delta)
        composed_absolute = ActionDeltaProcessor.from_delta(
            composed_delta, state
        )[0]
        composed_absolute[6] = base_absolute[6]
        self.policy._action_safety_profile.validate_chunk(
            composed_absolute[None, None, :],
            np.ones((1, 1), dtype=np.bool_),
            state[None, :],
        )
        safety = self.policy._action_safety_profile
        return {
            "online_semantics_version": ONLINE_SEMANTICS_VERSION,
            "decision_monotonic_ns": decision_ns,
            "normalizer_manifest_sha256": (
                self.runtime_artifacts.normalizer_manifest_sha256
            ),
            "normalized_state7": normalized_state.tolist(),
            "normalized_wrench6": normalized_wrench.tolist(),
            "normalized_wrench_delta6": normalized_wrench_delta.tolist(),
            "base_normalized_action7": base_normalized[0].tolist(),
            "base_absolute_action7": base_absolute.tolist(),
            "applied_residual_tcp6": residual.tolist(),
            "composed_normalized_action7": composed_normalized[0].tolist(),
            "composed_absolute_action7": composed_absolute.tolist(),
            "policy_single_action_guard": {
                "workspace_min_xyz_m": safety.workspace_min_xyz_m.tolist(),
                "workspace_max_xyz_m": safety.workspace_max_xyz_m.tolist(),
                "orientation_min_rpy_rad": safety.orientation_min_rpy_rad.tolist(),
                "orientation_max_rpy_rad": safety.orientation_max_rpy_rad.tolist(),
                "gimbal_margin_rad": float(safety.gimbal_margin_rad),
                "gripper_width_m": float(base_absolute[6]),
                "gripper_min_width_m": float(safety.gripper_min_width_m),
                "gripper_max_width_m": float(safety.gripper_max_width_m),
                "continuity_max_xyz_m": float(safety.continuity_max_xyz_m),
                "continuity_max_rotation_rad": float(
                    safety.continuity_max_rotation_rad
                ),
                "continuity_max_gripper_delta_m": float(
                    safety.continuity_max_gripper_delta_m
                ),
            },
        }

    def reset_residual_episode_context(self) -> None:
        # Dispatch owns wrench history; the inference copy is intentionally stateless.
        return None


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "ForceSmolVLA/1"

    @property
    def engine(self) -> InferenceEngine:
        return self.server.engine  # type: ignore[attr-defined]

    def _write_json(self, status: int, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(encoded)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/healthz":
            self._write_json(200, {"ok": True, **self.engine.metadata})
        elif self.path == "/metadata":
            self._write_json(200, self.engine.metadata)
        else:
            self._write_json(404, {"error": "NOT_FOUND"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/infer", "/residual-decision"}:
            self._write_json(404, {"error": "NOT_FOUND"})
            return
        try:
            length = int(self.headers.get("Content-Length", "-1"))
            if length <= 0 or length > MAX_REQUEST_BYTES:
                raise ValueError("INFERENCE_REQUEST_SIZE_INVALID")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("INFERENCE_REQUEST_MUST_BE_OBJECT")
            method = (
                self.engine.infer
                if self.path == "/infer"
                else self.engine.residual_decision
            )
            self._write_json(200, method(payload))
        except Exception as error:
            self._write_json(
                422,
                {"error": type(error).__name__, "detail": str(error)},
            )

    def log_message(self, format: str, *args: Any) -> None:
        print(f"[http] {self.address_string()} {format % args}", flush=True)


def load_deployment_profile(path: Path, root: Path) -> dict[str, Any]:
    profile = json.loads(path.resolve().read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "artifact_status",
        "deployment_id",
        "checkpoint",
        "rulespec",
        "deployment_binding",
        "deployment_binding_sha256",
        "dataset_manifest",
        "raw_session",
        "tool_profile",
    }
    if not isinstance(profile, dict) or set(profile) != required:
        raise ValueError("DEPLOYMENT_PROFILE_FIELDS_MISMATCH")
    if (
        profile["schema_version"] != DEPLOYMENT_PROFILE_SCHEMA
        or profile["artifact_status"] != "development_only"
    ):
        raise ValueError("DEPLOYMENT_PROFILE_STATUS_INVALID")
    for field in (
        "deployment_id",
        "checkpoint",
        "rulespec",
        "deployment_binding",
        "dataset_manifest",
        "raw_session",
        "tool_profile",
    ):
        if not isinstance(profile[field], str) or not profile[field].strip():
            raise ValueError(f"DEPLOYMENT_PROFILE_{field.upper()}_INVALID")
    digest = profile["deployment_binding_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError("DEPLOYMENT_PROFILE_BINDING_SHA256_INVALID")
    for field in ("checkpoint", "rulespec", "deployment_binding", "dataset_manifest"):
        value = Path(profile[field])
        profile[field] = value if value.is_absolute() else root / value
    profile["raw_session"] = Path(profile["raw_session"])
    return profile


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--deployment-profile",
        type=Path,
        default=root / "configs/deployment.active.development.json",
        help="shared task/checkpoint/data deployment profile",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="override the checkpoint selected by the deployment profile",
    )
    parser.add_argument(
        "--rulespec",
        type=Path,
        help="override the RuleSpec selected from the execution mode",
    )
    parser.add_argument(
        "--rulespec-schema",
        type=Path,
        default=root / "schemas/rulespec.schema.json",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--allow-development-robot-execution",
        action="store_true",
        help=(
            "explicitly enable the supervised deployment selected by the profile"
        ),
    )
    parser.add_argument(
        "--deployment-binding",
        type=Path,
        help="override the live deployment binding selected by the profile",
    )
    parser.add_argument(
        "--trusted-deployment-binding-sha256",
        help="optional explicit SHA256 override; defaults to the selected binding file",
    )
    args = parser.parse_args()
    try:
        profile = load_deployment_profile(args.deployment_profile, root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        parser.error(str(error))
    if args.checkpoint is None:
        args.checkpoint = profile["checkpoint"]
    if args.allow_development_robot_execution:
        if args.rulespec is None:
            args.rulespec = profile["rulespec"]
        if args.deployment_binding is None:
            args.deployment_binding = profile["deployment_binding"]
        if args.trusted_deployment_binding_sha256 is None:
            args.trusted_deployment_binding_sha256 = profile[
                "deployment_binding_sha256"
            ]
    else:
        if args.rulespec is None:
            args.rulespec = (
                root / "tests/fixtures/shadow_safety_thresholds.test_only.yaml"
            )
        if (
            args.deployment_binding is not None
            or args.trusted_deployment_binding_sha256 is not None
        ):
            parser.error("live binding requires --allow-development-robot-execution")
    if args.host not in {"127.0.0.1", "localhost"}:
        parser.error("only loopback binding is allowed until a clock-map protocol is approved")
    if args.device != "cuda":
        parser.error("ForceSmolVLA inference is GPU-only")
    if not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    return args


def main() -> None:
    args = parse_args()
    engine = InferenceEngine(
        args.checkpoint,
        args.rulespec,
        args.rulespec_schema,
        torch.device("cuda"),
        allow_development_robot_execution=(
            args.allow_development_robot_execution
        ),
        deployment_binding_path=args.deployment_binding,
        trusted_deployment_binding_sha256=(
            args.trusted_deployment_binding_sha256
        ),
    )
    server = ThreadingHTTPServer((args.host, args.port), RequestHandler)
    server.engine = engine  # type: ignore[attr-defined]
    print(
        f"[model] ForceSmolVLA ready checkpoint={engine.checkpoint} "
        f"acceptance={engine.metadata['checkpoint_acceptance_status']}",
        flush=True,
    )
    print(
        f"[model] dataset={engine.metadata['dataset_repo_id']} "
        f"horizon={engine.metadata['action_horizon']} "
        f"device={engine.device}",
        flush=True,
    )
    print(
        f"[server] listening on http://{args.host}:{args.port} "
        f"robot_io=false execution={engine.metadata['robot_execution_mode']}",
        flush=True,
    )
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
