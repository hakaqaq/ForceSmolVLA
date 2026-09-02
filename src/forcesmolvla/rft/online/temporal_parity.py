"""Recorded ACK parity between the offline converter and online projector."""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
import tempfile
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
import numpy as np

from forcesmolvla.action_delta import ActionDeltaProcessor
from forcesmolvla.normalizer import FrozenFeatureNormalizer, NormalizationLedger
from forcesmolvla.raw_to_lerobot_v3 import (
    RuntimeContract,
    _associate_acknowledged_actions,
    _normalize_quaternions,
    _rpy_unwrapped,
    prepare_episode,
)
from forcesmolvla.temporal import controller_reference_grid

from forcesmolvla.rft.online.transition_authority import (
    AcceptedAck,
    causal_zoh_ack_macro,
    normalized_ack_behavior_action,
)
from forcesmolvla.rft.critic_action_adapter_v2 import CRITIC_ACTION_CONTRACT


ROOT = Path(__file__).parents[4]
FIXTURE_SCHEMA = ROOT / "schemas/stage3_recorded_ack_fixture.v1.schema.json"
REPORT_SCHEMA = ROOT / "schemas/stage3_temporal_parity_report.v1.schema.json"
DEFAULT_RECORDED_FIXTURE = ROOT / "golden_fixtures/stage3_recorded_ack_fixture.v1.json"
DEFAULT_P0A_RECORDED_LIVE_EVIDENCE = (
    ROOT / "golden_fixtures/stage3_p0a_recorded_live_evidence.v1.json"
)
MISSING_RECORDED_FIELDS = (
    "fixture file",
    "fixture_kind=recorded_live",
    "provenance.raw_session_path/raw_episode_path and tree SHA256",
    "provenance.capture_manifest_path and SHA256",
    "Phase-2 source/runtime/calibration/normalizer/ActionContract bindings",
    "session start, episode end, and terminal boundary monotonic timestamps",
    "recorded accepted_reference stream with gripper command IDs",
    "recorded positive reference_ack stream with ACK and gripper ACK IDs",
    "recorded human/policy action-source lineage for each accepted Pose ACK",
    "recorded gripper target and terminal goal-status origin",
    "frozen full/terminal transition rows and current/next/terminal observations",
    "30 Hz anchor states for the frozen 10 Hz phase",
)


class TemporalParityError(ValueError):
    pass


P0A_REQUIRED_PATHS = (
    "policy_ack_behavior",
    "human_ack_behavior",
    "offline_demo_behavior",
    "actor_candidate",
    "bootstrap_candidate",
)


def p0a_formal_gate(value: Mapping[str, Any]) -> str:
    """Synthetic or incomplete evidence can never unlock formal P0-A."""

    eligible = bool(
        value.get("fixture_kind") == "recorded_live"
        and value.get("synthetic") is False
        and value.get("recorded_live_evidence") is True
    )
    paths = value.get("paths", {})
    bindings = value.get("bindings", {})
    comparisons = value.get("comparisons", {})
    passed = bool(
        eligible
        and all(paths.get(name) is True for name in P0A_REQUIRED_PATHS)
        and (not bindings or all(bindings.values()))
        and (not comparisons or all(comparisons.values()))
    )
    return "PASS" if passed else "BLOCKED"


def _p0a_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _p0a_file_matches(path: Path, expected: str) -> bool:
    return path.is_file() and sha256_file(path) == expected


def _p0a_action_sha256(values: Sequence[float]) -> str:
    encoded = json.dumps(
        list(values), separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@lru_cache(maxsize=2)
def run_p0a_recorded_live_parity(
    evidence_path: Path = DEFAULT_P0A_RECORDED_LIVE_EVIDENCE,
) -> dict[str, Any]:
    """Rebuild real replay on CPU and compare all five Critic action paths."""

    evidence_path = Path(evidence_path)
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    if (
        evidence.get("schema_version")
        != "forcesmolvla_stage3_p0a_recorded_live_evidence.v1"
        or evidence.get("contract_version") != CRITIC_ACTION_CONTRACT.version
    ):
        raise TemporalParityError("P0A_RECORDED_LIVE_EVIDENCE_CONTRACT_INVALID")

    bindings = {
        name: _p0a_file_matches(
            _p0a_path(binding["path"]), str(binding["sha256"])
        )
        for name, binding in evidence["bindings"].items()
    }
    provenance = evidence["provenance"]
    session = _p0a_path(provenance["raw_session_path"])
    episode = _p0a_path(provenance["raw_episode_path"])
    integrated = session / "integrated_capture" / episode.name / "streams"
    source_bindings = {
        "session_manifest": _p0a_file_matches(
            session / "session.json", provenance["session_manifest_sha256"]
        ),
        "episode_result": _p0a_file_matches(
            episode / "episode_result.json", provenance["episode_result_sha256"]
        ),
        "integrated_seal": _p0a_file_matches(
            integrated / "policy_execute_episode_seal.json",
            provenance["integrated_seal_sha256"],
        ),
        "integrated_transition_stream": _p0a_file_matches(
            integrated / "policy_execute_transition.jsonl",
            provenance["integrated_transition_stream_sha256"],
        ),
        "integrated_intervention_stream": _p0a_file_matches(
            integrated / "policy_execute_intervention.jsonl",
            provenance["integrated_intervention_stream_sha256"],
        ),
    }
    bindings.update(source_bindings)
    if not all(bindings.values()):
        return {
            "fixture_kind": evidence.get("fixture_kind"),
            "synthetic": evidence.get("synthetic"),
            "recorded_live_evidence": evidence.get("recorded_live_evidence"),
            "bindings": bindings,
            "comparisons": {},
            "paths": {name: False for name in P0A_REQUIRED_PATHS},
            "formal_gate": "BLOCKED",
            "robot_command_count": 0,
        }

    from forcesmolvla.rft.online.production_bridge import (
        ProductionBridge,
        load_bridge_config,
    )
    from forcesmolvla.rft.online.replay_training import (
        build_ack_macros,
        load_formal_online_r,
    )

    config, _ = load_bridge_config(
        ROOT / "configs/online_replay_production_bridge.v1.development.yaml"
    )
    with tempfile.TemporaryDirectory(prefix="forcesmolvla-p0a-parity-") as root:
        state_root = Path(root)
        bridge = ProductionBridge(config=config, state_root=state_root)
        report = bridge.admit_policy_execution_smoke(
            episode, operator_task_outcome="success"
        )
        policy_rows, policy_macros, _, human_rows = load_formal_online_r(
            state_root
        )
        human_macros = build_ack_macros(human_rows)

        def macro_by_uid(macros, uid: str):
            return next(
                macro
                for macro in macros
                if macro.transition["identity"]["transition_uid"] == uid
            )

        selected = evidence["selection"]
        policy = macro_by_uid(
            policy_macros, selected["policy"]["transition_uid"]
        )
        human = macro_by_uid(
            human_macros, selected["human"]["transition_uid"]
        )
        terminal = macro_by_uid(
            policy_macros, selected["terminal"]["transition_uid"]
        )

        def behavior_matches(macro, expected: Mapping[str, Any]) -> bool:
            valid = tuple(bool(value) for value in expected["behavior_mask"])
            ack_ids = tuple(
                ack_id
                for ack_id, keep in zip(
                    macro.behavior.ack_ids, valid, strict=True
                )
                if keep
            )
            command_ids = tuple(
                command_id
                for command_id, keep in zip(
                    macro.behavior.source_command_ids, valid, strict=True
                )
                if keep
            )
            return bool(
                macro.transition["identity"]["decision_id"]
                == expected["decision_id"]
                and macro.behavior.contract_version
                == CRITIC_ACTION_CONTRACT.version
                and list(macro.behavior.grid_monotonic_ns)
                == expected["grid_timestamp_ns"]
                and macro.behavior.next_timestamp_ns
                == expected["next_timestamp_ns"]
                and tuple(macro.behavior.behavior_mask) == valid
                and set(ack_ids) == {expected["source_ack_id"]}
                and set(command_ids) == {expected["source_command_id"]}
                and all(
                    value == expected["gripper_command_id"]
                    for value, keep in zip(
                        macro.behavior.gripper_command_ids, valid, strict=True
                    )
                    if keep
                )
                and macro.behavior.gripper_command_ids
                == macro.behavior.gripper_ack_command_ids
            )

        policy_ok = behavior_matches(policy, selected["policy"])
        human_ok = behavior_matches(human, selected["human"])
        terminal_ok = bool(
            list(terminal.behavior.grid_monotonic_ns)
            == selected["terminal"]["grid_timestamp_ns"]
            and terminal.behavior.next_timestamp_ns
            == selected["terminal"]["next_timestamp_ns"]
            and list(terminal.behavior.behavior_mask)
            == selected["terminal"]["behavior_mask"]
            and terminal.behavior.macro_duration_ns
            == selected["terminal"]["macro_duration_ns"]
            and terminal.transition["outcome"]["terminated"] is True
            and terminal.transition["outcome"]["truncated"] is False
            and terminal.transition["outcome"]["bootstrap_mask"] == 0.0
            and terminal.transition["outcome"]["discount"]
            == selected["terminal"]["discount"]
        )

        native_result = json.loads(
            (episode / "episode_result.json").read_text(encoding="utf-8")
        )
        streams = bridge._load_streams(episode)
        bridge._validate_seal(native_result, streams)
        integrated_capture = bridge._load_integrated_capture(
            episode_dir=episode,
            native_result=native_result,
            streams=streams,
            operator_task_outcome="success",
        )
        candidate_expected = selected["candidate"]
        transition = next(
            row
            for row in integrated_capture["transitions"]
            if row["request_id"] == candidate_expected["request_id"]
            and row["selection"]["sequence"]
            == selected["policy"]["source_dispatch_sequence"]
        )
        proposal = integrated_capture["proposals"][transition["request_id"]]
        observation_source = integrated_capture["observations"][
            candidate_expected["observation_id"]
        ]
        candidate_observation = bridge._formal_online_r_observation(
            episode_dir=episode,
            episode_id=str(policy.transition["identity"]["episode_id"]),
            source=observation_source,
            prepared=integrated_capture["prepared"],
        )

    from forcesmolvla.rft.losses import load_authorized_reward_train_transitions
    import torch

    demo_expected = evidence["selection"]["offline_demo"]
    demo = next(
        row
        for row in load_authorized_reward_train_transitions().to_pylist()
        if row["episode_id"] == demo_expected["episode_id"]
        and int(row["transition_index"]) == demo_expected["transition_index"]
    )
    demo_ok = bool(
        int(demo["anchor_frame"]) == demo_expected["anchor_frame"]
        and int(demo["next_frame"]) == demo_expected["next_frame"]
        and list(demo["executed_action_mask"]) == demo_expected["behavior_mask"]
        and bool(demo["terminated"]) is demo_expected["terminated"]
        and int(demo["bootstrap_mask"]) == demo_expected["bootstrap_mask"]
        and float(demo["discount"]) == demo_expected["discount"]
        and _p0a_action_sha256(demo["normalized_delta_action_exec_flat"])
        == demo_expected["normalized_action_sha256"]
    )

    normalizer = json.loads(
        _p0a_path(evidence["bindings"]["normalizer_manifest"]["path"])
        .read_text(encoding="utf-8")
    )["features"]["delta_action7"]
    mean = np.asarray(normalizer["mean"], dtype=np.float64)
    std = np.asarray(normalizer["std"], dtype=np.float64)
    state = np.asarray(
        candidate_observation["state7_absolute"], dtype=np.float64
    )
    absolute = np.asarray(proposal["actions_absolute7"], dtype=np.float64)
    normalized = (ActionDeltaProcessor.to_delta(absolute, state) - mean) / std
    chunk = torch.tensor(normalized, dtype=torch.float32).unsqueeze(0)
    mean_tensor = torch.tensor(mean, dtype=torch.float32)
    std_tensor = torch.tensor(std, dtype=torch.float32)
    from forcesmolvla.rft.critic_action_adapter_v2 import (
        bootstrap_command_effective_candidate_action,
        command_effective_candidate_action,
        command_effective_execution_index_map,
    )

    actor_candidate = command_effective_candidate_action(
        chunk,
        anchor_timestamp_ns=int(candidate_observation["materialized_timestamp_monotonic_ns"]),
        delta_action_mean7=mean_tensor,
        delta_action_std7=std_tensor,
    )
    bootstrap_candidate = bootstrap_command_effective_candidate_action(
        chunk,
        anchor_timestamp_ns=int(candidate_observation["materialized_timestamp_monotonic_ns"]),
        delta_action_mean7=mean_tensor,
        delta_action_std7=std_tensor,
    )

    def tensor_sha256(value) -> str:
        array = np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32)
        return hashlib.sha256(array.tobytes()).hexdigest()

    actor_ok = bool(
        proposal["proposal_id"] == candidate_expected["proposal_id"]
        and proposal["chunk_id"] == candidate_expected["chunk_id"]
        and int(candidate_observation["materialized_timestamp_monotonic_ns"])
        == candidate_expected["anchor_timestamp_ns"]
        and list(
            command_effective_execution_index_map(
                anchor_timestamp_ns=candidate_expected["anchor_timestamp_ns"]
            )
        )
        == candidate_expected["execution_index_map"]
        and tensor_sha256(actor_candidate)
        == candidate_expected["actor_candidate_sha256"]
    )
    bootstrap_ok = bool(
        bootstrap_command_effective_candidate_action
        is command_effective_candidate_action
        and torch.equal(actor_candidate, bootstrap_candidate)
        and tensor_sha256(bootstrap_candidate)
        == candidate_expected["bootstrap_candidate_sha256"]
    )
    all_macros = (*policy_macros, *human_macros)
    causal = all(
        provenance["receive_monotonic_ns"] <= tick
        for macro in all_macros
        for provenance, tick, valid in zip(
            macro.ack_provenance,
            macro.behavior.grid_monotonic_ns,
            macro.behavior.behavior_mask,
            strict=True,
        )
        if valid
    )
    ack_age = all(
        (tick - provenance["receive_monotonic_ns"])
        <= int(CRITIC_ACTION_CONTRACT.max_ack_age_ms * 1e6)
        for macro in all_macros
        for provenance, tick, valid in zip(
            macro.ack_provenance,
            macro.behavior.grid_monotonic_ns,
            macro.behavior.behavior_mask,
            strict=True,
        )
        if valid
    )
    paths = {
        "policy_ack_behavior": policy_ok,
        "human_ack_behavior": human_ok,
        "offline_demo_behavior": demo_ok,
        "actor_candidate": actor_ok,
        "bootstrap_candidate": bootstrap_ok,
    }
    comparisons = {
        "reconstructed_transition_counts": bool(
            report.accepted_unique_r_transition_count
            == provenance["reconstructed_transition_count"]
            and len(policy_rows) == provenance["reconstructed_policy_count"]
            and len(human_rows) == provenance["reconstructed_human_count"]
        ),
        "critic_grid_and_horizon": all(
            macro.behavior.next_timestamp_ns
            - macro.behavior.grid_monotonic_ns[0]
            == (
                macro.behavior.macro_duration_ns
            )
            for macro in all_macros
        ),
        "ack_causal_validity": causal,
        "ack_age": ack_age,
        "pose_and_gripper_ack_identity": policy_ok and human_ok,
        "terminal_partial_mask_and_discount": terminal_ok,
        "normalizer_manifest": bool(
            policy.transition["critic_action_contract"]["normalizer_manifest_hash"]
            == evidence["bindings"]["normalizer_manifest"]["sha256"]
            == human.transition["critic_action_contract"]["normalizer_manifest_hash"]
        ),
        "invalidated_policy_suffix_count_zero": bool(
            provenance["invalidated_policy_suffix_training_count"] == 0
            and all(
                macro.transition.get("policy_lineage", {})
                .get("proposal", {})
                .get("invalidated_by_takeover")
                is not True
                for macro in policy_macros
            )
        ),
    }
    result = {
        "fixture_kind": evidence["fixture_kind"],
        "synthetic": evidence["synthetic"],
        "recorded_live_evidence": evidence["recorded_live_evidence"],
        "fixture_id": evidence["fixture_id"],
        "contract_version": evidence["contract_version"],
        "bindings": bindings,
        "comparisons": comparisons,
        "paths": paths,
        "provenance": provenance,
        "robot_command_count": 0,
        "robot_execution_authorized": False,
    }
    result["formal_gate"] = p0a_formal_gate(result)
    return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_tree_sha256(path: Path) -> str:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(root)
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_RAW_SESSION_EMPTY")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(sha256_file(item)))
    return digest.hexdigest()


def _load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate(schema_path: Path, value: Mapping[str, Any], label: str) -> None:
    errors = sorted(
        Draft202012Validator(_load_schema(schema_path)).iter_errors(dict(value)),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        path = ".".join(str(part) for part in errors[0].absolute_path)
        raise TemporalParityError(f"{label}:{path}:{errors[0].message}")


def validate_recorded_ack_fixture(value: Mapping[str, Any]) -> dict[str, Any]:
    fixture = json.loads(json.dumps(dict(value), allow_nan=False))
    _validate(FIXTURE_SCHEMA, fixture, "ONLINE_REPLAY_RECORDED_ACK_FIXTURE_SCHEMA")
    return fixture


def validate_temporal_parity_report(value: Mapping[str, Any]) -> dict[str, Any]:
    report = json.loads(json.dumps(dict(value), allow_nan=False))
    _validate(REPORT_SCHEMA, report, "ONLINE_REPLAY_TEMPORAL_PARITY_REPORT_SCHEMA")
    return report


def blocked_temporal_parity_report(
    fixture_path: Path | None = None,
    *,
    missing_fields: Sequence[str] = MISSING_RECORDED_FIELDS,
) -> dict[str, Any]:
    return validate_temporal_parity_report({
        "schema_version": "forcesmolvla_stage3_temporal_parity_report.v1",
        "fixture_path": None if fixture_path is None else str(fixture_path),
        "fixture_kind": "not_available",
        "fixture_id": None,
        "tool_status": "blocked",
        "bindings": {},
        "comparisons": {},
        "missing_required_fields": sorted(set(str(value) for value in missing_fields)),
        "stage2": None,
        "stage3": None,
        "G1_TEMPORAL_PARITY_GATE": "BLOCKED",
        "RECORDED_FIXTURE_CAPTURE_REQUIRED": True,
        "G1_GATE_PASSED": False,
        "G2_FORMAL_GATE": "BLOCKED_ON_G1",
        "ROBOT_COMMAND_COUNT": 0,
        "ROBOT_EXECUTION_AUTHORIZED": False,
    })


def _resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _capture_manifest_matches(fixture: Mapping[str, Any], path: Path) -> bool:
    if fixture["fixture_kind"] != "recorded_live":
        return True
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError):
        return False
    selection_keys = (
        "prepared_grid_start_index", "prepared_grid_stop_index_exclusive",
        "current_observation_grid_index", "next_observation_grid_index",
        "terminal_observation_grid_index", "last_executable_grid_index",
        "full_macro_transition_index", "terminal_transition_index",
    )
    expected_ack_identities = [
        {
            "ack_id": row["ack_id"],
            "request_sequence": row["request_sequence"],
            "request_stamp_ns": row["request_stamp_ns"],
            "receive_monotonic_ns": row["receive_monotonic_ns"],
        }
        for row in fixture["reference_acks"]
    ]
    return bool(
        manifest.get("schema_version")
        == "forcesmolvla_stage3_recorded_ack_capture_manifest.v1"
        and manifest.get("fixture_id") == fixture["fixture_id"]
        and manifest.get("fixture_kind") == fixture["fixture_kind"]
        and manifest.get("synthetic") == fixture["synthetic"]
        and manifest.get("action_source") == fixture["action_source"]
        and manifest.get("capture_origin") == fixture["capture_origin"]
        and manifest.get("raw_session_path") == fixture["provenance"]["raw_session_path"]
        and manifest.get("raw_episode_path") == fixture["provenance"]["raw_episode_path"]
        and manifest.get("selection")
        == {key: fixture["selection"][key] for key in selection_keys}
        and manifest.get("gripper_authority")
        == fixture["selection"]["gripper_authority"]
        and manifest.get("ack_natural_identities") == expected_ack_identities
        and manifest.get("observation_boundaries")
        == fixture["selection"]["observation_provenance"]
    )


def _binding_checks(fixture: Mapping[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, binding in fixture["bindings"].items():
        path = _resolve(binding["path"])
        checks[name] = path.is_file() and sha256_file(path) == binding["sha256"]
    provenance = fixture["provenance"]
    raw_root = _resolve(provenance["raw_session_path"])
    raw_episode = _resolve(provenance["raw_episode_path"])
    capture_manifest = _resolve(provenance["capture_manifest_path"])
    checks["raw_session_tree"] = (
        raw_root.is_dir()
        and directory_tree_sha256(raw_root) == provenance["raw_session_tree_sha256"]
    )
    try:
        checks["raw_episode"] = raw_episode.is_dir() and raw_episode.resolve().is_relative_to(
            raw_root.resolve()
        )
    except (FileNotFoundError, RuntimeError):
        checks["raw_episode"] = False
    checks["capture_manifest"] = (
        capture_manifest.is_file()
        and sha256_file(capture_manifest) == provenance["capture_manifest_sha256"]
        and _capture_manifest_matches(fixture, capture_manifest)
    )
    return checks


def _delta_normalizer(fixture: Mapping[str, Any]) -> FrozenFeatureNormalizer:
    binding = fixture["bindings"]["normalizer_manifest"]
    manifest = json.loads(_resolve(binding["path"]).read_text(encoding="utf-8"))
    feature = manifest["features"]["delta_action7"]
    return FrozenFeatureNormalizer(
        "delta_action7",
        np.asarray(feature["mean"], dtype=np.float64),
        np.asarray(feature["std"], dtype=np.float64),
        tuple(str(value) for value in feature["fit_episode_ids"]),
    )


def _runtime_contract(fixture: Mapping[str, Any]) -> RuntimeContract:
    path = _resolve(fixture["bindings"]["stage2_runtime_contract"]["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_status") == "approved":
        return RuntimeContract.from_approved_json(path)
    if payload.get("artifact_status") == "development_only":
        return RuntimeContract.from_development_json(path)
    raise TemporalParityError("ONLINE_REPLAY_PHASE2_RUNTIME_CONTRACT_STATUS")


def _phase2_episode_inputs(
    fixture: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, Any], RuntimeContract]:
    provenance = fixture["provenance"]
    raw_root = _resolve(provenance["raw_session_path"]).resolve()
    episode = _resolve(provenance["raw_episode_path"]).resolve()
    if not raw_root.is_dir():
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_RAW_SESSION_MISSING")
    if not episode.is_dir() or not episode.is_relative_to(raw_root):
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_RAW_EPISODE_MISSING_OR_OUTSIDE_SESSION")
    session_path = raw_root / "session.json"
    if not session_path.is_file():
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_SESSION_MANIFEST_MISSING")
    calibration_path = _resolve(fixture["bindings"]["calibration_bundle"]["path"])
    if not calibration_path.is_file():
        raise TemporalParityError("ONLINE_REPLAY_PHASE2_CALIBRATION_BUNDLE_MISSING")
    return (
        episode,
        json.loads(session_path.read_text(encoding="utf-8")),
        json.loads(calibration_path.read_text(encoding="utf-8")),
        _runtime_contract(fixture),
    )


def _stream(episode: Path, name: str) -> list[dict[str, Any]]:
    path = episode / "streams" / f"{name}.jsonl"
    if not path.is_file():
        raise TemporalParityError(f"ONLINE_REPLAY_RECORDED_STREAM_MISSING:{name}")
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    except (OSError, TypeError, ValueError) as error:
        raise TemporalParityError(f"ONLINE_REPLAY_RECORDED_STREAM_INVALID:{name}") from error


def _pose_without_frame_id(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "position_m": value.get("position_m"),
        "quaternion_xyzw": value.get("quaternion_xyzw"),
    }


def _recorded_ack_id(*, request_sequence: int, request_stamp_ns: int) -> str:
    return f"reference-ack:{request_sequence}:{request_stamp_ns}"


def _validate_recorded_live_sources(
    fixture: Mapping[str, Any], episode: Path,
) -> None:
    """Re-read native streams so a recorded-live gate cannot trust enriched IDs alone."""

    if fixture["fixture_kind"] != "recorded_live":
        return
    if (
        fixture["synthetic"] is not False
        or fixture["capture_origin"] != "historical_native_real"
    ):
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_CAPTURE_CLASSIFICATION_INVALID")

    references = {
        int(row["accepted_receive_monotonic_ns"]): row
        for row in _stream(episode, "accepted_reference")
    }
    acknowledgements = {
        int(row["receive_monotonic_ns"]): row
        for row in _stream(episode, "reference_ack")
    }
    safe_actions = {
        int(row["receive_monotonic_ns"]): row
        for row in _stream(episode, "safe_action")
    }
    for reference in fixture["accepted_references"]:
        raw = references.get(int(reference["accepted_receive_monotonic_ns"]))
        if raw is None:
            raise TemporalParityError("ONLINE_REPLAY_RECORDED_REFERENCE_IDENTITY_MISSING")
        if (
            int(raw.get("source_stamp_ns", 0)) != int(reference["source_stamp_ns"])
            or raw.get("frame_id") != reference["frame_id"]
            or _pose_without_frame_id(raw.get("pose", {})) != reference["pose"]
            or float(raw.get("target_gripper_width_m", -1.0))
            != float(reference["target_gripper_width_m"])
            or reference["gripper_command_id_origin"]
            != "recorded_gripper_target_and_goal_status"
        ):
            raise TemporalParityError("ONLINE_REPLAY_RECORDED_REFERENCE_SOURCE_MISMATCH")

    authority = fixture["selection"]["gripper_authority"]
    target_rows = [
        row for row in _stream(episode, "gripper_target")
        if row.get("action_goal_id") == authority["action_goal_id"]
    ]
    status_rows = [
        row for row in _stream(episode, "gripper_goal_status")
        if row.get("action_goal_id") == authority["action_goal_id"]
    ]
    if len(target_rows) != 1 or len(status_rows) != 1:
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_GRIPPER_ORIGIN_MISSING")
    target, status = target_rows[0], status_rows[0]
    expected_authority = {
        "action_goal_id": target.get("action_goal_id"),
        "local_goal_sequence": target.get("local_goal_sequence"),
        "accepted_monotonic_ns": target.get("accepted_monotonic_ns"),
        "target_receive_monotonic_ns": target.get("receive_monotonic_ns"),
        "finished_monotonic_ns": status.get("finished_monotonic_ns"),
        "status_receive_monotonic_ns": status.get("receive_monotonic_ns"),
        "requested_state": target.get("requested_state"),
        "target_width_m": target.get("target_width_m"),
        "outcome": status.get("outcome"),
    }
    if expected_authority != authority:
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_GRIPPER_ORIGIN_MISMATCH")
    if int(authority["finished_monotonic_ns"]) > int(fixture["temporal"]["session_start_ack_ns"]):
        raise TemporalParityError("ONLINE_REPLAY_INITIAL_GRIPPER_AUTHORITY_NOT_ESTABLISHED")
    if any(
        reference["gripper_command_id"] != authority["action_goal_id"]
        for reference in fixture["accepted_references"]
    ):
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_REFERENCE_GRIPPER_ORIGIN_MISMATCH")

    for acknowledgement in fixture["reference_acks"]:
        raw = acknowledgements.get(int(acknowledgement["receive_monotonic_ns"]))
        if raw is None:
            raise TemporalParityError("ONLINE_REPLAY_RECORDED_ACK_IDENTITY_MISSING")
        payload = raw.get("payload", {})
        sequence = int(payload.get("request_sequence", -1))
        stamp = int(payload.get("request_stamp_ns", 0))
        if (
            acknowledgement["ack_id_origin"] != "recorded_request_sequence_and_stamp"
            or acknowledgement["ack_id"]
            != _recorded_ack_id(request_sequence=sequence, request_stamp_ns=stamp)
            or sequence != int(acknowledgement["request_sequence"])
            or stamp != int(acknowledgement["request_stamp_ns"])
            or int(payload.get("ack_monotonic_ns", 0))
            != int(acknowledgement["controller_ack_monotonic_ns"])
            or bool(payload.get("accepted")) is not True
            or _pose_without_frame_id(payload.get("accepted_pose", {}))
            != acknowledgement["payload"]["accepted_pose"]
        ):
            raise TemporalParityError("ONLINE_REPLAY_RECORDED_ACK_SOURCE_MISMATCH")
        safe = safe_actions.get(int(acknowledgement["action_source_receive_monotonic_ns"]))
        safe_payload = {} if safe is None else safe.get("payload", {})
        arbitration = safe_payload.get("arbitration", {})
        raw_action = arbitration.get("raw_action", {})
        workspace = safe_payload.get("workspace_clipped", [])
        if (
            safe is None
            or int(safe_payload.get("decision_id", -1))
            != int(acknowledgement["action_decision_id"])
            or int(safe_payload.get("equilibrium_source_stamp_ns", 0)) != stamp
            or safe_payload.get("equilibrium_published") is not True
            or arbitration.get("accepted") is not True
            or raw_action.get("source") != acknowledgement["accepted_action_source"]
            or bool(raw_action.get("intervention")) != acknowledgement["intervention"]
            or bool(any(workspace)) != acknowledgement["workspace_clipped"]
            or acknowledgement["accepted_action_source"] != fixture["action_source"]
            or acknowledgement["slot_owner"] != "human_intervention"
            or acknowledgement["gripper_command_id"] != authority["action_goal_id"]
            or acknowledgement["gripper_ack_command_id"] != authority["action_goal_id"]
        ):
            raise TemporalParityError("ONLINE_REPLAY_RECORDED_ACTION_SOURCE_LINEAGE_MISMATCH")


def _validate_transition_selection(fixture: Mapping[str, Any]) -> None:
    if fixture["fixture_kind"] != "recorded_live":
        return
    try:
        import pandas as pd
    except ImportError as error:
        raise TemporalParityError("ONLINE_REPLAY_TERMINAL_INDEX_READER_MISSING") from error
    selection = fixture["selection"]
    path = _resolve(fixture["bindings"]["terminal_transition_index"]["path"])
    try:
        frame = pd.read_parquet(path)
        full_rows = frame[frame["transition_index"] == selection["full_macro_transition_index"]]
        terminal_rows = frame[frame["transition_index"] == selection["terminal_transition_index"]]
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise TemporalParityError("ONLINE_REPLAY_TERMINAL_INDEX_INVALID") from error
    if len(full_rows) != 1 or len(terminal_rows) != 1:
        raise TemporalParityError("ONLINE_REPLAY_TERMINAL_TRANSITION_IDENTITY_MISSING")
    full = full_rows.iloc[0]
    terminal = terminal_rows.iloc[0]
    start = int(selection["prepared_grid_start_index"])
    terminal_global = start + int(selection["terminal_observation_grid_index"])
    terminal_anchor = start + int(selection["next_observation_grid_index"])
    mask = [bool(value) for value in terminal["executed_action_mask"]]
    if not (
        full["episode_id"] == terminal["episode_id"] == Path(
            fixture["provenance"]["raw_episode_path"]
        ).name
        and int(full["anchor_frame"]) == start
        and int(full["next_frame"]) == terminal_anchor
        and int(full["executed_steps"]) == 3
        and [bool(value) for value in full["executed_action_mask"]] == [True, True, True]
        and bool(full["terminated"]) is False
        and int(terminal["anchor_frame"]) == terminal_anchor
        and int(terminal["next_frame"]) == terminal_global
        and int(terminal["detector_terminal_frame"]) == terminal_global
        and int(terminal["executed_steps"]) == 2
        and mask == [True, True, False]
        and bool(terminal["terminated"]) is True
        and int(terminal["bootstrap_mask"]) == 0
    ):
        raise TemporalParityError("ONLINE_REPLAY_TERMINAL_TRANSITION_SOURCE_MISMATCH")


def _validate_selection_against_prepared(
    fixture: Mapping[str, Any], prepared: Any, episode: Path,
) -> slice:
    selection = fixture["selection"]
    start = int(selection["prepared_grid_start_index"])
    stop = int(selection["prepared_grid_stop_index_exclusive"])
    terminal_local = int(selection["terminal_observation_grid_index"])
    last_executable = int(selection["last_executable_grid_index"])
    if not (
        0 <= start < stop <= len(prepared.tuple_host_ns)
        and stop - start == terminal_local + 1
        and int(selection["next_observation_grid_index"]) == 3
        and last_executable == terminal_local - 1
    ):
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_SELECTION_BOUNDARY_INVALID")
    grid = np.asarray(prepared.tuple_host_ns[start:stop], dtype=np.int64)
    temporal = fixture["temporal"]
    if not (
        int(temporal["session_start_ack_ns"]) == int(grid[0])
        and int(temporal["episode_end_ns"]) == int(grid[-1])
        and int(temporal["terminal_boundary_ns"]) == int(grid[last_executable])
        and int(temporal["policy_anchor_phase_on_30hz_grid"]) == 0
    ):
        raise TemporalParityError("ONLINE_REPLAY_RECORDED_SELECTION_TEMPORAL_MISMATCH")

    expected_roles = {"current": 0, "next": 3, "terminal": terminal_local}
    observations = selection["observation_provenance"]
    if {row["role"]: int(row["local_grid_index"]) for row in observations} != expected_roles:
        raise TemporalParityError("ONLINE_REPLAY_OBSERVATION_BOUNDARY_ROLE_MISMATCH")
    provenance_fields = (
        "state_pose_source_stamp_ns", "camera1_receive_monotonic_ns",
        "camera2_receive_monotonic_ns", "gripper_source_stamp_ns",
        "wrench_filter_output_stamp_ns", "action_ack_receive_monotonic_ns",
        "validity_bits",
    )
    for row in observations:
        local = int(row["local_grid_index"])
        global_index = start + local
        try:
            external_path = prepared.camera1_paths[global_index].relative_to(episode).as_posix()
            wrist_path = prepared.camera2_paths[global_index].relative_to(episode).as_posix()
        except ValueError as error:
            raise TemporalParityError("ONLINE_REPLAY_OBSERVATION_CAMERA_OUTSIDE_EPISODE") from error
        if not (
            int(row["global_grid_index"]) == global_index
            and int(row["grid_monotonic_ns"]) == int(prepared.tuple_host_ns[global_index])
            and np.array_equal(np.asarray(row["state7"]), prepared.state7[global_index])
            and np.array_equal(np.asarray(row["wrench6"]), prepared.wrench6[global_index])
            and row["external_camera_relative_path"] == external_path
            and row["wrist_camera_relative_path"] == wrist_path
            and all(
                int(row[field]) == int(prepared.provenance[field][global_index])
                for field in provenance_fields
            )
        ):
            raise TemporalParityError("ONLINE_REPLAY_OBSERVATION_BOUNDARY_SOURCE_MISMATCH")
    _validate_transition_selection(fixture)
    _validate_recorded_live_sources(fixture, episode)
    return slice(start, stop)


def _ack_action7(
    references: Sequence[dict[str, Any]],
    acknowledgements: Sequence[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    contract = SimpleNamespace(
        action_pose_tolerance_m=1e-9,
        action_quaternion_tolerance_rad=1e-9,
    )
    times, pose_quaternion_width = _associate_acknowledged_actions(
        references, acknowledgements, contract,
    )
    quaternions = _normalize_quaternions(
        pose_quaternion_width[:, 3:7], "stage3 parity acknowledged action",
    )
    rpy = _rpy_unwrapped(quaternions)
    action7 = np.column_stack(
        (pose_quaternion_width[:, :3], rpy, pose_quaternion_width[:, 7])
    ).astype(np.float32)
    return times, action7


def _validate_gripper_identity(
    references: Sequence[dict[str, Any]],
    acknowledgements: Sequence[dict[str, Any]],
) -> None:
    reference_times = np.asarray(
        [row["accepted_receive_monotonic_ns"] for row in references], dtype=np.int64,
    )
    for acknowledgement in acknowledgements:
        index = int(np.searchsorted(reference_times, acknowledgement["receive_monotonic_ns"], side="right") - 1)
        if index < 0:
            raise TemporalParityError("ONLINE_REPLAY_ACK_HAS_NO_CAUSAL_REFERENCE")
        expected = references[index]["gripper_command_id"]
        if not (
            expected == acknowledgement["gripper_command_id"]
            == acknowledgement["gripper_ack_command_id"]
        ):
            raise TemporalParityError("ONLINE_REPLAY_RECORDED_GRIPPER_COMMAND_ID_MISMATCH")


def _macro_plan(grid: np.ndarray, temporal: Mapping[str, Any]) -> tuple[list[int], list[dict[str, Any]]]:
    phase = int(temporal["policy_anchor_phase_on_30hz_grid"])
    terminal = int(temporal["terminal_boundary_ns"])
    full: list[int] = []
    partial: list[dict[str, Any]] = []
    for anchor in range(phase, len(grid), 3):
        if int(grid[anchor]) > terminal:
            break
        available = min(3, int(np.searchsorted(grid, terminal, side="right")) - anchor)
        if anchor + 2 >= len(grid) or int(grid[anchor + 2]) > terminal:
            partial.append({
                "anchor_grid_index": anchor,
                "available_slots": max(0, available),
                "reason": "partial_macro_crosses_terminal_or_episode_end",
            })
        else:
            full.append(anchor)
    return full, partial


def _anchor_states(fixture: Mapping[str, Any]) -> dict[int, np.ndarray]:
    result = {
        int(row["grid_index"]): np.asarray(row["state7"], dtype=np.float64)
        for row in fixture["anchor_states"]
    }
    if len(result) != len(fixture["anchor_states"]):
        raise TemporalParityError("ONLINE_REPLAY_DUPLICATE_ANCHOR_STATE")
    return result


def _normalized_once(
    normalizer: FrozenFeatureNormalizer,
    delta: np.ndarray,
    *,
    owner: str,
    anchor: int,
) -> tuple[np.ndarray, int]:
    ledger = NormalizationLedger()
    ledger.claim(f"{owner}/anchor={anchor}", normalizer.name)
    normalized = normalizer.apply(delta)
    return normalized, ledger.counts.get(normalizer.name, 0)


def _identity_for_ack_times(
    acknowledgements: Sequence[dict[str, Any]], ack_times: Sequence[int],
) -> dict[str, list[str]]:
    by_time: dict[int, dict[str, Any]] = {}
    for row in acknowledgements:
        timestamp = int(row["receive_monotonic_ns"])
        if timestamp in by_time:
            raise TemporalParityError("ONLINE_REPLAY_DUPLICATE_RECORDED_ACK_TIMESTAMP")
        by_time[timestamp] = row
    try:
        selected = [by_time[int(timestamp)] for timestamp in ack_times]
    except KeyError as error:
        raise TemporalParityError("ONLINE_REPLAY_CONVERTER_ACK_PROVENANCE_MISSING") from error
    return {
        "ack_ids": [str(row["ack_id"]) for row in selected],
        "gripper_command_ids": [str(row["gripper_command_id"]) for row in selected],
        "gripper_ack_command_ids": [str(row["gripper_ack_command_id"]) for row in selected],
    }


def _stage2_project(fixture: Mapping[str, Any]) -> dict[str, Any]:
    temporal = fixture["temporal"]
    references = fixture["accepted_references"]
    acknowledgements = fixture["reference_acks"]
    _validate_gripper_identity(references, acknowledgements)
    episode_dir, session, calibration, contract = _phase2_episode_inputs(fixture)
    prepared = prepare_episode(
        episode_dir,
        session=session,
        calibration_payload=calibration,
        contract=contract,
    )
    selected = _validate_selection_against_prepared(fixture, prepared, episode_dir)
    grid = np.asarray(prepared.tuple_host_ns[selected], dtype=np.int64)
    ack_times = np.asarray(
        prepared.provenance["action_ack_receive_monotonic_ns"][selected], dtype=np.int64,
    )
    ages_ms = np.asarray(
        prepared.provenance["action_ack_age_ms"][selected], dtype=np.float64,
    )
    if np.any(ages_ms > float(temporal["max_ack_age_ms"])):
        raise TemporalParityError("STAGE2_ACK_MISSING_OR_STALE_FOR_PARITY")
    full, partial = _macro_plan(grid, temporal)
    normalizer = _delta_normalizer(fixture)
    macros: list[dict[str, Any]] = []
    identity_macros: list[dict[str, Any]] = []
    for anchor in full:
        global_anchor = selected.start + anchor
        absolute = prepared.action7[global_anchor:global_anchor + 3]
        anchor_state = prepared.state7[global_anchor]
        delta = ActionDeltaProcessor.to_delta(absolute, anchor_state)
        normalized, count = _normalized_once(
            normalizer, delta, owner="stage2", anchor=anchor,
        )
        macros.append({
            "anchor_grid_index": anchor,
            "grid_monotonic_ns": grid[anchor:anchor + 3].tolist(),
            "anchor_state7": anchor_state.tolist(),
            "accepted_absolute_action_k7": absolute.tolist(),
            "anchor_relative_delta_k7": delta.tolist(),
            "normalized_delta_action_k7": normalized.tolist(),
            "normalizer_application_count": count,
        })
        identity_macros.append({
            "anchor_grid_index": anchor,
            **_identity_for_ack_times(
                acknowledgements, ack_times[anchor:anchor + 3],
            ),
        })
    result = {
        "converter": {
            "module": "forcesmolvla.raw_to_lerobot_v3",
            "symbol": "prepare_episode",
            "call_count": 1,
            "numeric_output_source": "PreparedEpisode",
        },
        "grid_monotonic_ns": grid.tolist(),
        "policy_anchor_phase_on_30hz_grid": int(temporal["policy_anchor_phase_on_30hz_grid"]),
        "anchor_grid_indices": full,
        "macros": macros,
        "terminal_boundary_ns": int(temporal["terminal_boundary_ns"]),
        "partial_macro_quarantine": partial,
        "observation_boundary": fixture["selection"]["observation_provenance"],
        "raw_identity_provenance": {
            "source": "recorded accepted_reference/reference_ack streams; not PreparedEpisode",
            "macros": identity_macros,
        },
    }
    return result


def _online_project(fixture: Mapping[str, Any]) -> dict[str, Any]:
    temporal = fixture["temporal"]
    references = fixture["accepted_references"]
    acknowledgements = fixture["reference_acks"]
    _validate_gripper_identity(references, acknowledgements)
    event_ns, action7 = _ack_action7(references, acknowledgements)
    records = [
        AcceptedAck(
            ack_id=row["ack_id"],
            receive_monotonic_ns=int(row["receive_monotonic_ns"]),
            accepted_absolute_action7=tuple(float(value) for value in action),
            gripper_command_id=row["gripper_command_id"],
            gripper_ack_command_id=row["gripper_ack_command_id"],
            slot_owner=row["slot_owner"],
            accepted_action_source=row["accepted_action_source"],
            intervention=bool(row["intervention"]),
            accepted=bool(row["payload"]["accepted"]),
            workspace_clipped=bool(row["workspace_clipped"]),
            source_command_id=(
                f"reference-command:{int(row['request_sequence'])}:"
                f"{int(row['request_stamp_ns'])}"
            ),
            source_dispatch_sequence=int(row["request_sequence"]),
            source_model_index=(
                int(row["action_decision_id"])
                if row["accepted_action_source"] == "policy"
                else -1
            ),
            episode_id=Path(
                fixture["provenance"]["raw_episode_path"]
            ).name,
            policy_revision=str(fixture["fixture_id"]),
            takeover_generation=0,
            reset_generation=0,
            chunk_id=f"parity-{fixture['fixture_id']}",
            chunk_compatibility_key=(
                f"parity:{fixture['fixture_id']}:"
                f"{row['accepted_action_source']}"
            ),
            clock_domain="upper-host-monotonic",
            controller_authority="recorded-reference-controller",
        )
        for row, action in zip(acknowledgements, action7, strict=True)
    ]
    if not np.array_equal(event_ns, [record.receive_monotonic_ns for record in records]):
        raise TemporalParityError("ONLINE_REPLAY_ACK_EVENT_ORDER_DRIFT")
    grid = controller_reference_grid(
        session_start_ack_ns=int(temporal["session_start_ack_ns"]),
        episode_end_ns=int(temporal["episode_end_ns"]),
        fps=30,
    )
    full, partial = _macro_plan(grid, temporal)
    states = _anchor_states(fixture)
    normalizer = _delta_normalizer(fixture)
    macros: list[dict[str, Any]] = []
    identity_macros: list[dict[str, Any]] = []
    for anchor in full:
        if anchor not in states:
            raise TemporalParityError(f"ONLINE_REPLAY_ANCHOR_STATE_MISSING:{anchor}")
        macro = causal_zoh_ack_macro(
            records,
            grid[anchor:anchor + 3],
            max_ack_age_ms=float(temporal["max_ack_age_ms"]),
        )
        delta = ActionDeltaProcessor.to_delta(
            macro.accepted_absolute_action_k7, states[anchor],
        )
        ledger = NormalizationLedger()

        def normalize_once(value: np.ndarray) -> np.ndarray:
            ledger.claim(f"stage3/anchor={anchor}", normalizer.name)
            return normalizer.apply(value)

        normalized = normalized_ack_behavior_action(
            macro, anchor_state7=states[anchor], normalize_delta7=normalize_once,
        )
        macros.append({
            "anchor_grid_index": anchor,
            "grid_monotonic_ns": list(macro.grid_monotonic_ns),
            "anchor_state7": states[anchor].tolist(),
            "accepted_absolute_action_k7": macro.accepted_absolute_action_k7.tolist(),
            "anchor_relative_delta_k7": delta.tolist(),
            "normalized_delta_action_k7": normalized.tolist(),
            "normalizer_application_count": ledger.counts.get(normalizer.name, 0),
        })
        identity_macros.append({
            "anchor_grid_index": anchor,
            "ack_ids": list(macro.ack_ids),
            "gripper_command_ids": list(macro.gripper_command_ids),
            "gripper_ack_command_ids": list(macro.gripper_ack_command_ids),
        })
    result = {
        "projector": {
            "module": "forcesmolvla.rft.online.temporal_parity",
            "symbol": "_online_project",
        },
        "grid_monotonic_ns": grid.tolist(),
        "policy_anchor_phase_on_30hz_grid": int(temporal["policy_anchor_phase_on_30hz_grid"]),
        "anchor_grid_indices": full,
        "macros": macros,
        "terminal_boundary_ns": int(temporal["terminal_boundary_ns"]),
        "partial_macro_quarantine": partial,
        "observation_boundary": fixture["selection"]["observation_provenance"],
        "raw_identity_provenance": {
            "source": "recorded accepted_reference/reference_ack streams",
            "macros": identity_macros,
        },
    }
    return result


def _comparison(stage2: Mapping[str, Any], online: Mapping[str, Any]) -> dict[str, bool]:
    grid = np.asarray(online["grid_monotonic_ns"], dtype=np.int64)
    indices = (grid * 30 + 500_000_000) // 1_000_000_000
    rational = np.array_equal(grid, (indices * 1_000_000_000 + 15) // 30)
    anchors = online["anchor_grid_indices"]
    phase = online["policy_anchor_phase_on_30hz_grid"]
    paired = zip(stage2["macros"], online["macros"], strict=True)
    pairs = list(paired)
    has_macros = bool(pairs)
    identity_pairs = list(zip(
        stage2["raw_identity_provenance"]["macros"],
        online["raw_identity_provenance"]["macros"],
        strict=True,
    ))

    def field_equal(name: str) -> bool:
        return has_macros and all(left[name] == right[name] for left, right in pairs)

    def array_equal(name: str) -> bool:
        return has_macros and all(
            np.array_equal(np.asarray(left[name]), np.asarray(right[name]))
            for left, right in pairs
        )

    def identity_equal(name: str) -> bool:
        return bool(identity_pairs) and all(
            left[name] == right[name] for left, right in identity_pairs
        )

    result = {
        "rational_30hz_grid": rational and stage2["grid_monotonic_ns"] == online["grid_monotonic_ns"],
        "policy_10hz_anchor_phase": (
            stage2["anchor_grid_indices"] == anchors
            and stage2["policy_anchor_phase_on_30hz_grid"] == phase
            and (not anchors or all(anchor % 3 == phase for anchor in anchors))
            and (len(anchors) < 2 or np.all(np.diff(anchors) == 3))
        ),
        "phase2_prepare_episode_called_once": (
            stage2["converter"]["module"] == "forcesmolvla.raw_to_lerobot_v3"
            and stage2["converter"]["symbol"] == "prepare_episode"
            and stage2["converter"]["call_count"] == 1
            and stage2["converter"]["numeric_output_source"] == "PreparedEpisode"
        ),
        "matching_positive_ack_id": identity_equal("ack_ids"),
        "gripper_command_identity": (
            identity_equal("gripper_command_ids")
            and identity_equal("gripper_ack_command_ids")
            and all(
                left["gripper_command_ids"] == left["gripper_ack_command_ids"]
                for left, _right in identity_pairs
            )
        ),
        "anchor_state7": array_equal("anchor_state7"),
        "accepted_absolute7": array_equal("accepted_absolute_action_k7"),
        "anchor_relative_delta7": array_equal("anchor_relative_delta_k7"),
        "normalizer_exactly_once": has_macros and all(
            left["normalizer_application_count"] == right["normalizer_application_count"] == 1
            for left, right in pairs
        ),
        "normalized_kx7": array_equal("normalized_delta_action_k7"),
        "macro_100ms": (
            has_macros
            and all(len(row["grid_monotonic_ns"]) == 3 for row in online["macros"])
            and (len(anchors) < 2 or all(
                grid[right] - grid[left] == 100_000_000
                for left, right in zip(anchors, anchors[1:])
            ))
        ),
        "terminal_boundary": stage2["terminal_boundary_ns"] == online["terminal_boundary_ns"],
        "partial_macro_quarantine": stage2["partial_macro_quarantine"] == online["partial_macro_quarantine"],
        "current_next_terminal_observations": (
            stage2["observation_boundary"] == online["observation_boundary"]
            and [row["role"] for row in online["observation_boundary"]]
            == ["current", "next", "terminal"]
        ),
    }
    return {name: bool(value) for name, value in result.items()}


def run_recorded_ack_parity(
    fixture: Mapping[str, Any],
    *,
    fixture_path: Path | None = None,
) -> dict[str, Any]:
    value = validate_recorded_ack_fixture(fixture)
    bindings = _binding_checks(value)
    stage2 = _stage2_project(value)
    online = _online_project(value)
    comparisons = _comparison(stage2, online)
    eligible = (
        value["fixture_kind"] == "recorded_live"
        and value["provenance"]["recorded_live_evidence"] is True
        and value["synthetic"] is False
        and value["capture_origin"] == "historical_native_real"
    )
    all_pass = bool(bindings) and all(bindings.values()) and bool(comparisons) and all(comparisons.values())
    gate_pass = eligible and all_pass
    gate = "PASS" if gate_pass else "FAIL" if eligible else "BLOCKED"
    tool_status = (
        "pass" if gate_pass else
        "synthetic_tool_test_pass" if not eligible and all_pass else
        "fail"
    )
    return validate_temporal_parity_report({
        "schema_version": "forcesmolvla_stage3_temporal_parity_report.v1",
        "fixture_path": None if fixture_path is None else str(fixture_path),
        "fixture_kind": value["fixture_kind"],
        "fixture_id": value["fixture_id"],
        "tool_status": tool_status,
        "bindings": bindings,
        "comparisons": comparisons,
        "missing_required_fields": [] if eligible else list(MISSING_RECORDED_FIELDS),
        "stage2": stage2,
        "stage3": online,
        "G1_TEMPORAL_PARITY_GATE": gate,
        "RECORDED_FIXTURE_CAPTURE_REQUIRED": not eligible,
        "G1_GATE_PASSED": gate_pass,
        "G2_FORMAL_GATE": "PASS" if gate_pass else "BLOCKED_ON_G1",
        "ROBOT_COMMAND_COUNT": 0,
        "ROBOT_EXECUTION_AUTHORIZED": False,
    })
