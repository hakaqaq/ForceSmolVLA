#!/usr/bin/env python3
"""Opt-in Stage-3 audit wrapper for the unchanged production deploy client."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEPLOY_DIR = Path("/home/rlc123/fr3_client_ws/scripts")


def _wrapper_args(arguments: list[str]) -> tuple[str, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--stage3-lineage-episode-id", required=True)
    parsed, remaining = parser.parse_known_args(arguments)
    if "--execute" in remaining:
        raise PermissionError(
            "STAGE3_LINEAGE_WRAPPER_NOT_IN_APPROVED_DEPLOYMENT_BINDING"
        )
    return parsed.stage3_lineage_episode_id, remaining


def _install_audit(deploy: Any, *, episode_id: str) -> None:
    from forcesmolvla.rft.stage3.gripper_provenance import GripperGeneration
    from forcesmolvla.rft.stage3.policy_lineage import (
        InitialGripperAuthority,
        PolicyLineageAudit,
    )

    state: dict[str, Any] = {
        "audit": None,
        "metadata": None,
        "bridge": None,
        "controller": None,
        "observation": None,
        "results_by_chunk": {},
        "lineage_by_sequence": {},
    }

    original_validate_metadata = deploy.validate_metadata
    original_validate_response = deploy.validate_response
    original_observation = deploy.LiveForceSmolObservation
    original_output = deploy.ForceSmolControlOutput
    original_bridge = deploy.ForceSmolActionBridge
    original_controller = deploy.ForceSmolHeadlessRobotController

    def validate_metadata(metadata: dict[str, Any]) -> None:
        original_validate_metadata(metadata)
        state["metadata"] = metadata
        state["audit"] = PolicyLineageAudit(
            episode_id=episode_id,
            policy_revision=str(metadata["model_sha256"]),
            reset_generation=1,
        )

    class AuditedObservation(original_observation):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            state["observation"] = self

        def request(self, metadata: dict[str, Any]) -> dict[str, Any]:
            request = super().request(metadata)
            audit = state["audit"]
            bridge = state["bridge"]
            epoch = 0 if bridge is None else bridge.arbiter.policy_epoch
            audit.record_request(
                request,
                policy_epoch=epoch,
                takeover_generation=epoch,
                recorded_monotonic_ns=time.monotonic_ns(),
            )
            return request

    def validate_response(
        result: dict[str, Any], request: dict[str, Any], workspace: Any
    ) -> Any:
        actions = original_validate_response(result, request, workspace)
        lineage = state["audit"].record_result(
            request,
            result,
            recorded_monotonic_ns=time.monotonic_ns(),
        )
        state["results_by_chunk"][lineage.chunk_id] = lineage
        return actions

    def initial_gripper_authority() -> dict[str, Any] | None:
        controller = state["controller"]
        observation = state["observation"]
        metadata = state["metadata"]
        audit = state["audit"]
        bridge = state["bridge"]
        if None in (controller, observation, metadata, audit, bridge):
            return None
        origin = controller.stage3_gripper_origin()
        if origin is None:
            return None
        accepted, terminal = origin
        captured_ns = time.monotonic_ns()
        with observation._lock:
            feedback_width_m = observation.gripper_width_m
            feedback_ns = observation.gripper_receive_ns
        if feedback_width_m is None or feedback_ns <= 0:
            return None
        requested_state = str(accepted["requested_state"])
        requested_width_m = (
            controller.args.gripper_open_width_m
            if requested_state == "OPEN"
            else controller.args.gripper_closed_width_m
        )
        epoch = bridge.arbiter.policy_epoch
        generation = GripperGeneration(
            episode_id=audit.episode_id,
            reset_generation=audit.reset_generation,
            takeover_generation=epoch,
            policy_revision=audit.policy_revision,
            policy_epoch=epoch,
        )
        try:
            return InitialGripperAuthority(
                episode_id=audit.episode_id,
                origin_local_goal_sequence=int(accepted["local_goal_sequence"]),
                origin_action_goal_id=str(accepted["action_goal_id"]),
                origin_accepted_monotonic_ns=int(
                    accepted["accepted_monotonic_ns"]
                ),
                requested_state=requested_state,
                requested_width_m=float(requested_width_m),
                terminal_outcome=str(terminal["outcome"]),
                terminal_finished_monotonic_ns=int(
                    terminal["finished_monotonic_ns"]
                ),
                feedback_width_m=float(feedback_width_m),
                feedback_state=(
                    "OPEN" if float(feedback_width_m) >= 0.055 else "CLOSED"
                ),
                feedback_monotonic_ns=int(feedback_ns),
                captured_monotonic_ns=captured_ns,
                feedback_age_ns=captured_ns - int(feedback_ns),
                clock_domain_id=audit.clock_domain_id,
                generation=generation,
            ).validate(
                max_feedback_age_ns=int(
                    min(float(metadata["gripper_max_age_ms"]), 100.0) * 1.0e6
                )
            ).to_dict()
        except Exception:
            return None

    class AuditedOutput(original_output):
        def record_policy_selection(
            self, sequence: int, payload: dict[str, Any]
        ) -> None:
            lineage = state["lineage_by_sequence"].pop(int(sequence), None)
            if lineage is None:
                raise RuntimeError("STAGE3_POLICY_SELECTION_LINEAGE_MISSING")
            payload.update(lineage)
            payload["dispatch_sequence"] = int(sequence)
            payload["selected_index"] = int(payload["action_index"])
            super().record_policy_selection(sequence, payload)

        def enqueue_safe_action(self, payload: dict[str, Any]) -> None:
            raw = payload.get("arbitration", {}).get("raw_action", {})
            if raw.get("phase") == "episode_start":
                initial = initial_gripper_authority()
                if initial is not None:
                    payload["stage3_initial_gripper_authority"] = initial
            super().enqueue_safe_action(payload)

    class AuditedBridge(original_bridge):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            state["bridge"] = self

        def submit_absolute_chunk(
            self,
            actions: Any,
            *,
            t_ref_ns: int,
            fps: int,
            policy_epoch: int,
            chunk_id: str,
        ) -> int:
            lineage = state["results_by_chunk"].get(str(chunk_id))
            if lineage is None:
                raise RuntimeError("STAGE3_POLICY_RESULT_LINEAGE_MISSING")
            sequence = int(self._forcesmol_sequence)
            state["lineage_by_sequence"][sequence] = state["audit"].bind_dispatch(
                lineage,
                policy_epoch=policy_epoch,
                takeover_generation=policy_epoch,
            )
            try:
                return super().submit_absolute_chunk(
                    actions,
                    t_ref_ns=t_ref_ns,
                    fps=fps,
                    policy_epoch=policy_epoch,
                    chunk_id=chunk_id,
                )
            except Exception:
                state["lineage_by_sequence"].pop(sequence, None)
                raise

    class AuditedController(original_controller):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self._stage3_last_gripper_accepted: dict[str, Any] | None = None
            self._stage3_last_gripper_terminal: dict[str, Any] | None = None
            self._stage3_gripper_target_publisher = self.create_publisher(
                deploy.String, deploy.forcevla.collector.GRIPPER_TARGET_TOPIC, 10
            )
            self._stage3_gripper_status_publisher = self.create_publisher(
                deploy.String, deploy.forcevla.collector.GRIPPER_STATUS_TOPIC, 20
            )
            state["controller"] = self

        def _publish_gripper_event(self, payload: dict[str, Any], publisher: Any) -> None:
            message = deploy.String()
            message.data = json.dumps(payload, separators=(",", ":"))
            publisher.publish(message)

        def _on_gripper_goal_accepted(self, metadata: dict[str, Any]) -> None:
            self._stage3_last_gripper_accepted = dict(metadata)
            self._stage3_last_gripper_terminal = None
            payload = dict(metadata, event_type="accepted_goal", token=episode_id)
            self._publish_gripper_event(
                payload, self._stage3_gripper_target_publisher
            )

        def _on_gripper_goal_terminal(
            self, metadata: dict[str, Any], outcome: str
        ) -> None:
            payload = dict(
                metadata,
                event_type="terminal_goal",
                token=episode_id,
                outcome=outcome,
                finished_monotonic_ns=time.monotonic_ns(),
            )
            self._stage3_last_gripper_terminal = payload
            self._publish_gripper_event(
                payload, self._stage3_gripper_status_publisher
            )

        def stage3_gripper_origin(
            self,
        ) -> tuple[dict[str, Any], dict[str, Any]] | None:
            accepted = self._stage3_last_gripper_accepted
            terminal = self._stage3_last_gripper_terminal
            if accepted is None or terminal is None:
                return None
            if (
                accepted.get("action_goal_id") != terminal.get("action_goal_id")
                or accepted.get("local_goal_sequence")
                != terminal.get("local_goal_sequence")
            ):
                return None
            return dict(accepted), dict(terminal)

    deploy.validate_metadata = validate_metadata
    deploy.validate_response = validate_response
    deploy.LiveForceSmolObservation = AuditedObservation
    deploy.ForceSmolControlOutput = AuditedOutput
    deploy.ForceSmolActionBridge = AuditedBridge
    deploy.ForceSmolHeadlessRobotController = AuditedController


def main() -> int:
    episode_id, deploy_arguments = _wrapper_args(sys.argv[1:])
    sys.path.insert(0, str(ROOT / "src"))
    sys.path.insert(0, str(DEPLOY_DIR))
    import deploy_forcesmolvla as deploy

    _install_audit(deploy, episode_id=episode_id)
    sys.argv = [str(DEPLOY_DIR / "deploy_forcesmolvla.py"), *deploy_arguments]
    deploy.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
