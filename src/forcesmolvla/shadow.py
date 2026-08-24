"""Deterministic, record-only P9 Shadow scheduler and replay verifier."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .rules import load_and_validate_rulespec


REQUIRED_RULE_IDS = frozenset(
    {
        "SS_WORKSPACE",
        "SS_ORIENTATION",
        "SS_DELTA_XYZ",
        "SS_DELTA_ROT_GEODESIC",
        "SS_GRIPPER_RANGE_RATE",
        "SS_CONTINUITY",
        "SS_OBSERVATION_AGE",
        "SS_TRANSPORT",
        "SS_END_TO_APPLY",
        "SS_SLOT_LATENESS",
        "SS_EXPIRED_RATE",
        "SS_MISSED_TICK_RATE",
        "SS_HOLD_OVERRUN",
    }
)


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ceil_div(numerator: int, denominator: int) -> int:
    return -(-numerator // denominator)


@dataclass(frozen=True)
class ShadowProtocol:
    action_period_numerator_ns: int
    action_period_denominator: int
    policy_period_ns: int
    controller_tick_ns: int
    horizon: int
    execution_horizon: int
    max_hold_extension_ns: int

    @classmethod
    def from_dict(cls, value: dict) -> "ShadowProtocol":
        protocol = cls(**{name: int(value[name]) for name in cls.__annotations__})
        if protocol != cls(100_000_000, 3, 100_000_000, 1_000_000, 50, 3, 100_000_000):
            raise RuntimeError("P9_PROTOCOL_DRIFT")
        return protocol

    def action_slot_ns(self, tau0_ns: int, index: int) -> int:
        return tau0_ns + _ceil_div(
            index * self.action_period_numerator_ns, self.action_period_denominator
        )

    def chunk_index(self, t_candidate_ns: int, tau0_ns: int) -> int:
        delta = t_candidate_ns - tau0_ns
        return max(
            0,
            _ceil_div(
                delta * self.action_period_denominator,
                self.action_period_numerator_ns,
            ),
        )

    def controller_tick_at_or_after(self, timestamp_ns: int) -> int:
        return _ceil_div(timestamp_ns, self.controller_tick_ns) * self.controller_tick_ns


@dataclass(frozen=True)
class ShadowResolution:
    mode: str
    valid: bool
    reasons: tuple[str, ...]
    rules: dict | None
    rules_sha256: str | None
    clock_map: dict | None
    clock_map_sha256: str | None


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _load_clock_map(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "mode",
        "acceptance_status",
        "formal_eligible",
        "synthetic",
        "controller_clock_domain",
        "created_controller_ns",
        "valid_until_controller_ns",
        "max_age_ns",
        "mappings",
        "provenance",
        "detached_signature",
        "approval",
    }
    if set(payload) != required:
        raise ValueError("SHADOW_CLOCK_MAP_SCHEMA_MISMATCH")
    mappings = payload["mappings"]
    if set(mappings) != {"sensor_to_controller", "gpu_to_controller"}:
        raise ValueError("SHADOW_CLOCK_MAP_REQUIRED_MAPPINGS_MISSING")
    for mapping in mappings.values():
        if set(mapping) != {
            "source_clock_domain",
            "slope_numerator",
            "slope_denominator",
            "offset_ns",
            "residual_bound_ns",
            "drift_bound_ppm",
        }:
            raise ValueError("SHADOW_CLOCK_MAPPING_SCHEMA_MISMATCH")
        if int(mapping["slope_denominator"]) <= 0:
            raise ValueError("SHADOW_CLOCK_MAPPING_DENOMINATOR_INVALID")
    return payload


def resolve_shadow_artifacts(
    *,
    mode: str,
    rules_path: Path,
    schema_path: Path,
    clock_map_path: Path | None,
    test_fixture_root: Path,
) -> ShadowResolution:
    """Resolve test-only artifacts or return a fail-closed production resolution."""

    if mode not in {"test_only", "production"}:
        raise ValueError("shadow mode must be test_only or production")
    reasons: list[str] = []
    rules = None
    clock_map = None
    rules_hash = None
    clock_hash = None
    try:
        rules = load_and_validate_rulespec(
            rules_path, schema_path, formal=mode == "production"
        )
        rules_hash = sha256_file(rules_path)
    except (FileNotFoundError, PermissionError, ValueError) as error:
        reasons.append(f"SHADOW_RULESPEC_FAIL_CLOSED:{type(error).__name__}:{error}")
    if clock_map_path is None:
        reasons.append("SHADOW_CLOCK_MAP_MISSING")
    else:
        try:
            clock_map = _load_clock_map(clock_map_path)
            clock_hash = sha256_file(clock_map_path)
        except (FileNotFoundError, json.JSONDecodeError, ValueError) as error:
            reasons.append(f"SHADOW_CLOCK_MAP_INVALID:{type(error).__name__}:{error}")

    if mode == "test_only":
        if not _is_within(rules_path, test_fixture_root):
            reasons.append("TEST_ONLY_RULESPEC_OUTSIDE_TEST_FIXTURES")
        if rules is not None:
            if rules.get("mode") != "test_only" or rules.get("artifact_status") != "development_only":
                reasons.append("TEST_ONLY_RULESPEC_STATUS_MISMATCH")
            ids = {rule["rule_id"] for rule in rules["rules"]}
            if ids != REQUIRED_RULE_IDS:
                reasons.append("TEST_ONLY_RULESPEC_RULE_SET_MISMATCH")
            if any(rule["threshold"]["value"] is None for rule in rules["rules"]):
                reasons.append("TEST_ONLY_RULESPEC_UNRESOLVED_THRESHOLD")
        if clock_map_path is not None and not _is_within(clock_map_path, test_fixture_root):
            reasons.append("TEST_ONLY_CLOCK_MAP_OUTSIDE_TEST_FIXTURES")
        if clock_map is not None and (
            clock_map.get("mode") != "test_only"
            or clock_map.get("acceptance_status") != "development_only"
            or clock_map.get("formal_eligible") is not False
            or clock_map.get("synthetic") is not True
            or clock_map.get("detached_signature") is not None
            or clock_map.get("approval") is not None
        ):
            reasons.append("TEST_ONLY_CLOCK_MAP_STATUS_MISMATCH")
    else:
        if rules is not None and rules.get("mode") == "test_only":
            reasons.append("PRODUCTION_REJECTS_TEST_ONLY_RULESPEC")
        if clock_map is not None and (
            clock_map.get("mode") != "production"
            or clock_map.get("acceptance_status") != "approved"
            or clock_map.get("formal_eligible") is not True
            or clock_map.get("synthetic") is not False
            or not clock_map.get("detached_signature")
            or not clock_map.get("approval")
        ):
            reasons.append("PRODUCTION_CLOCK_MAP_SIGNATURE_OR_APPROVAL_MISSING")

    return ShadowResolution(
        mode=mode,
        valid=not reasons,
        reasons=tuple(reasons),
        rules=rules,
        rules_sha256=rules_hash,
        clock_map=clock_map,
        clock_map_sha256=clock_hash,
    )


def _mapping(clock_map: dict, name: str, timestamp_ns: int) -> int:
    mapping = clock_map["mappings"][name]
    return (
        int(timestamp_ns) * int(mapping["slope_numerator"])
        // int(mapping["slope_denominator"])
        + int(mapping["offset_ns"])
    )


def _clock_valid_at(resolution: ShadowResolution, controller_ns: int, source: dict) -> list[str]:
    if not resolution.valid or resolution.clock_map is None:
        return list(resolution.reasons or ("SHADOW_CLOCK_MAP_MISSING",))
    clock_map = resolution.clock_map
    reasons = []
    if not (
        int(clock_map["created_controller_ns"])
        <= controller_ns
        <= int(clock_map["valid_until_controller_ns"])
    ):
        reasons.append("SHADOW_CLOCK_MAP_STALE")
    if controller_ns - int(clock_map["created_controller_ns"]) > int(clock_map["max_age_ns"]):
        reasons.append("SHADOW_CLOCK_MAP_AGE_EXCEEDED")
    expected = {
        "sensor_to_controller": source["sensor_clock_domain"],
        "gpu_to_controller": source["gpu_clock_domain"],
    }
    for name, domain in expected.items():
        if clock_map["mappings"][name]["source_clock_domain"] != domain:
            reasons.append(f"SHADOW_CLOCK_DOMAIN_MISMATCH:{name}")
    return reasons


def _rules_by_id(resolution: ShadowResolution) -> dict[str, dict]:
    if resolution.rules is None:
        return {}
    return {rule["rule_id"]: rule for rule in resolution.rules["rules"]}


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=np.float64,
    )


def _rotation_geodesic(left: np.ndarray, right: np.ndarray) -> float:
    cosine = (np.trace(left.T @ right) - 1.0) / 2.0
    return float(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _check_continuity(previous: np.ndarray, current: np.ndarray, rule: dict) -> bool:
    params = rule["parameters"]
    return (
        np.linalg.norm(current[:3] - previous[:3]) <= float(params["max_xyz_m"])
        and _rotation_geodesic(_rpy_matrix(previous[3:6]), _rpy_matrix(current[3:6]))
        <= float(params["max_rotation_rad"])
        and abs(float(current[6] - previous[6])) <= float(params["max_gripper_delta"])
    )


def evaluate_shadow_candidate(
    source: dict,
    *,
    resolution: ShadowResolution,
    protocol: ShadowProtocol,
) -> dict:
    """Evaluate one candidate without ever calling a controller or queue."""

    reasons: list[str] = list(resolution.reasons)
    rules = _rules_by_id(resolution)
    absolute = np.asarray(source["absolute_action7_chunk"], dtype=np.float64)
    normalized = np.asarray(source["normalized_delta7_chunk"], dtype=np.float64)
    valid_mask = np.asarray(source["action_valid_mask"], dtype=np.bool_)
    if absolute.shape != (protocol.horizon, 7) or normalized.shape != (protocol.horizon, 7):
        reasons.append("SHADOW_CHUNK_SHAPE_INVALID")
    if valid_mask.shape != (protocol.horizon,):
        reasons.append("SHADOW_ACTION_VALID_MASK_SHAPE_INVALID")
    if not np.all(np.isfinite(absolute)) or not np.all(np.isfinite(normalized)):
        reasons.append("SHADOW_TARGET_NONFINITE")
    if not source["runtime_artifact_compatible"]:
        reasons.append("CALIBRATION_NORMALIZER_INCOMPATIBLE")
    if not source["wrench_geometry_valid"]:
        reasons.append("WRENCH_GEOMETRY_INVALID")

    timing = {
        "t_ref_controller_ns": None,
        "t_ready_controller_ns": None,
        "t_candidate_controller_ns": None,
        "t_controller_apply_ns": None,
        "tau_j_ns": None,
        "j": None,
        "end_to_apply_age_ns": None,
        "slot_lateness_ns": None,
        "planned_arrival_ns": [],
    }
    if resolution.clock_map is not None:
        t_ref = _mapping(resolution.clock_map, "sensor_to_controller", int(source["t_ref_sensor_ns"]))
        t_ready = _mapping(resolution.clock_map, "gpu_to_controller", int(source["t_ready_gpu_ns"]))
        reasons.extend(_clock_valid_at(resolution, t_ref, source))
        t_candidate = t_ready + int(source["transport_ns"])
        tau0 = int(source["tau0_controller_ns"])
        j = protocol.chunk_index(t_candidate, tau0)
        tau_j = protocol.action_slot_ns(tau0, j)
        apply_ns = protocol.controller_tick_at_or_after(max(t_candidate, tau_j))
        arrivals = [
            protocol.controller_tick_at_or_after(
                max(t_candidate, protocol.action_slot_ns(tau0, j + offset))
            )
            for offset in range(protocol.execution_horizon)
        ]
        timing.update(
            {
                "t_ref_controller_ns": t_ref,
                "t_ready_controller_ns": t_ready,
                "t_candidate_controller_ns": t_candidate,
                "t_controller_apply_ns": apply_ns,
                "tau_j_ns": tau_j,
                "j": j,
                "end_to_apply_age_ns": apply_ns - t_ref,
                "slot_lateness_ns": apply_ns - tau_j,
                "planned_arrival_ns": arrivals,
            }
        )
        if j >= protocol.horizon or j + protocol.execution_horizon > protocol.horizon:
            reasons.append("SHADOW_CHUNK_EXPIRED_OR_TAIL_SHORT")
        elif valid_mask.shape == (protocol.horizon,) and not np.all(
            valid_mask[j : j + protocol.execution_horizon]
        ):
            reasons.append("SHADOW_ACTION_LABEL_INVALID")
        if t_ready < t_ref:
            reasons.append("SHADOW_GPU_READY_PRECEDES_REFERENCE")
        observation_ages = []
        for timestamp in source["observation_timestamps_sensor_ns"].values():
            mapped = _mapping(resolution.clock_map, "sensor_to_controller", int(timestamp))
            observation_ages.append(t_ref - mapped)
        if any(age < 0 for age in observation_ages):
            reasons.append("SHADOW_OBSERVATION_FROM_FUTURE")

        if rules:
            def limit(rule_id: str) -> float:
                return float(rules[rule_id]["threshold"]["value"])

            if observation_ages and max(observation_ages) > limit("SS_OBSERVATION_AGE") * 1e6:
                reasons.append(rules["SS_OBSERVATION_AGE"]["failure_code"])
            if int(source["transport_ns"]) > limit("SS_TRANSPORT") * 1e6:
                reasons.append(rules["SS_TRANSPORT"]["failure_code"])
            if apply_ns - t_ref > limit("SS_END_TO_APPLY") * 1e6:
                reasons.append(rules["SS_END_TO_APPLY"]["failure_code"])
            if apply_ns - tau_j > limit("SS_SLOT_LATENESS") * 1e6:
                reasons.append(rules["SS_SLOT_LATENESS"]["failure_code"])

            if j + protocol.execution_horizon <= protocol.horizon:
                targets = absolute[j : j + protocol.execution_horizon]
                workspace = rules["SS_WORKSPACE"]["parameters"]
                if not np.all(
                    (targets[:, :3] >= np.asarray(workspace["min_xyz_m"]))
                    & (targets[:, :3] <= np.asarray(workspace["max_xyz_m"]))
                ):
                    reasons.append(rules["SS_WORKSPACE"]["failure_code"])
                orientation = rules["SS_ORIENTATION"]["parameters"]
                if not np.all(
                    (targets[:, 3:6] >= np.asarray(orientation["min_rpy_rad"]))
                    & (targets[:, 3:6] <= np.asarray(orientation["max_rpy_rad"]))
                ) or np.any(
                    np.abs(np.abs(targets[:, 4]) - math.pi / 2)
                    < float(orientation["gimbal_margin_rad"])
                ):
                    reasons.append(rules["SS_ORIENTATION"]["failure_code"])
                if len(targets) > 1:
                    xyz_delta = np.linalg.norm(np.diff(targets[:, :3], axis=0), axis=1)
                    if np.max(xyz_delta) > limit("SS_DELTA_XYZ"):
                        reasons.append(rules["SS_DELTA_XYZ"]["failure_code"])
                    rotation_delta = [
                        _rotation_geodesic(_rpy_matrix(targets[k, 3:6]), _rpy_matrix(targets[k + 1, 3:6]))
                        for k in range(len(targets) - 1)
                    ]
                    if max(rotation_delta) > limit("SS_DELTA_ROT_GEODESIC"):
                        reasons.append(rules["SS_DELTA_ROT_GEODESIC"]["failure_code"])
                gripper = rules["SS_GRIPPER_RANGE_RATE"]["parameters"]
                rates = np.abs(np.diff(targets[:, 6])) / (
                    protocol.action_period_numerator_ns
                    / protocol.action_period_denominator
                    / 1e9
                )
                if (
                    np.any(targets[:, 6] < float(gripper["min_value"]))
                    or np.any(targets[:, 6] > float(gripper["max_value"]))
                    or (len(rates) and np.max(rates) > float(gripper["max_rate_per_s"]))
                ):
                    reasons.append(rules["SS_GRIPPER_RANGE_RATE"]["failure_code"])

    reasons = sorted(set(reasons))
    j = timing["j"]
    indices = (
        list(range(j, j + protocol.execution_horizon))
        if isinstance(j, int) and j + protocol.execution_horizon <= protocol.horizon
        else []
    )
    targets = absolute[indices].tolist() if indices and absolute.shape == (protocol.horizon, 7) else []
    return {
        "generation": int(source["generation"]),
        "policy_tick_index": int(source["policy_tick_index"]),
        "candidate_valid": not reasons,
        "candidate_reasons": reasons,
        "candidate_indices": indices,
        "candidate_targets_absolute7": targets,
        "timing": timing,
    }


def _dispatch_candidate_time(outcome: dict) -> int:
    value = outcome["timing"]["t_controller_apply_ns"]
    return int(value) if value is not None else 2**63 - 1


def arbitrate_shadow_candidates(
    sources: list[dict],
    outcomes: list[dict],
    *,
    rules: dict | None,
    protocol: ShadowProtocol,
    run_end_controller_ns: int,
) -> tuple[list[dict], dict]:
    """Pure event simulation of latest-generation-wins; dispatch means record-only."""

    by_generation = {outcome["generation"]: outcome for outcome in outcomes}
    source_by_generation = {int(source["generation"]): source for source in sources}
    if len(by_generation) != len(outcomes) or len(source_by_generation) != len(sources):
        raise ValueError("SHADOW_GENERATIONS_MUST_BE_UNIQUE")
    results = {
        generation: {
            "dispatch_valid": False,
            "dispatch_reasons": [],
            "actual_dispatched_indices": [],
            "cancelled_indices": [],
            "actual_intervals": [],
        }
        for generation in by_generation
    }
    pending: list[dict] = []
    dispatched: list[dict] = []
    latest_seen = -1

    def dispatch_before(timestamp_ns: int, *, inclusive: bool) -> None:
        nonlocal pending
        ready = [
            item
            for item in pending
            if item["arrival_ns"] < timestamp_ns
            or (inclusive and item["arrival_ns"] == timestamp_ns)
        ]
        pending = [item for item in pending if item not in ready]
        dispatched.extend(sorted(ready, key=lambda item: (item["arrival_ns"], item["generation"])))

    groups: dict[int, list[int]] = {}
    for outcome in outcomes:
        groups.setdefault(_dispatch_candidate_time(outcome), []).append(outcome["generation"])
    continuity_rule = _rules_by_id(
        ShadowResolution("test_only", True, (), rules, None, None, None)
    ).get("SS_CONTINUITY")
    for arrival_ns in sorted(groups):
        dispatch_before(arrival_ns, inclusive=False)
        for generation in sorted(groups[arrival_ns], reverse=True):
            outcome = by_generation[generation]
            result = results[generation]
            if generation < latest_seen:
                result["dispatch_reasons"].append("SHADOW_STALE_GENERATION")
                continue
            latest_seen = generation
            if not outcome["candidate_valid"]:
                result["dispatch_reasons"].append("SHADOW_CANDIDATE_INVALID")
                continue
            previous = (
                np.asarray(dispatched[-1]["target"], dtype=np.float64)
                if dispatched
                else np.asarray(source_by_generation[generation]["raw_state7"], dtype=np.float64)
            )
            if dispatched and arrival_ns - dispatched[-1]["arrival_ns"] > protocol.max_hold_extension_ns:
                result["dispatch_reasons"].append("SHADOW_HOLD_OVERRUN")
                continue
            current = np.asarray(outcome["candidate_targets_absolute7"][0], dtype=np.float64)
            if continuity_rule is None or not _check_continuity(previous, current, continuity_rule):
                result["dispatch_reasons"].append("SHADOW_CONTINUITY_INVALID")
                continue
            result["dispatch_valid"] = True
            for item in pending:
                results[item["generation"]]["cancelled_indices"].append(item["index"])
            pending = []
            for index, target, target_arrival in zip(
                outcome["candidate_indices"],
                outcome["candidate_targets_absolute7"],
                outcome["timing"]["planned_arrival_ns"],
                strict=True,
            ):
                pending.append(
                    {
                        "generation": generation,
                        "index": index,
                        "target": target,
                        "arrival_ns": int(target_arrival),
                    }
                )
        dispatch_before(arrival_ns, inclusive=True)
    dispatch_before(2**63 - 1, inclusive=True)

    dispatched.sort(key=lambda item: (item["arrival_ns"], item["generation"]))
    monotonic = all(
        dispatched[index]["arrival_ns"] < dispatched[index + 1]["arrival_ns"]
        for index in range(len(dispatched) - 1)
    )
    hold_overrun = False
    for index, item in enumerate(dispatched):
        natural_end = item["arrival_ns"] + protocol.max_hold_extension_ns
        next_arrival = (
            dispatched[index + 1]["arrival_ns"] if index + 1 < len(dispatched) else run_end_controller_ns
        )
        if next_arrival > natural_end:
            hold_overrun = True
        hold_end = min(next_arrival, natural_end)
        interval = {
            "index": item["index"],
            "arrival_ns": item["arrival_ns"],
            "hold_start_ns": item["arrival_ns"],
            "hold_end_ns": hold_end,
        }
        results[item["generation"]]["actual_dispatched_indices"].append(item["index"])
        results[item["generation"]]["actual_intervals"].append(interval)

    expired = sum(
        "SHADOW_CHUNK_EXPIRED_OR_TAIL_SHORT" in outcome["candidate_reasons"]
        for outcome in outcomes
    )
    ticks = sorted({outcome["policy_tick_index"] for outcome in outcomes})
    missed = sum(max(0, right - left - 1) for left, right in zip(ticks, ticks[1:]))
    aggregate_reasons = []
    rules_by_id = {rule["rule_id"]: rule for rule in rules["rules"]} if rules else {}
    if rules_by_id:
        expired_rate = expired / len(outcomes) if outcomes else 0.0
        missed_rate = missed / (len(ticks) + missed) if ticks or missed else 0.0
        if expired_rate > float(rules_by_id["SS_EXPIRED_RATE"]["threshold"]["value"]):
            aggregate_reasons.append(rules_by_id["SS_EXPIRED_RATE"]["failure_code"])
        if missed_rate > float(rules_by_id["SS_MISSED_TICK_RATE"]["threshold"]["value"]):
            aggregate_reasons.append(rules_by_id["SS_MISSED_TICK_RATE"]["failure_code"])
        if hold_overrun:
            aggregate_reasons.append(rules_by_id["SS_HOLD_OVERRUN"]["failure_code"])
    run = {
        "actual_dispatch_count": len(dispatched),
        "actual_dispatched_generations": [item["generation"] for item in dispatched],
        "actual_dispatched_indices": [item["index"] for item in dispatched],
        "arrival_monotonic_strict": monotonic,
        "intervals_nonnegative_nonoverlap": monotonic
        and all(
            interval["hold_end_ns"] >= interval["hold_start_ns"]
            for result in results.values()
            for interval in result["actual_intervals"]
        ),
        "expired_chunk_rate": expired / len(outcomes) if outcomes else 0.0,
        "missed_policy_tick_rate": missed / (len(ticks) + missed) if ticks or missed else 0.0,
        "hold_overrun": hold_overrun,
        "run_valid": monotonic and not aggregate_reasons,
        "run_reasons": sorted(set(aggregate_reasons)),
    }
    return [results[outcome["generation"]] for outcome in outcomes], run


def build_shadow_record_artifact(
    sources: list[dict],
    *,
    resolution: ShadowResolution,
    protocol: ShadowProtocol,
    run_end_controller_ns: int,
    artifact_hashes: dict,
) -> dict:
    if resolution.mode != "test_only" or not resolution.valid:
        raise RuntimeError("P9_RECORD_BUILD_REQUIRES_VALID_TEST_ONLY_ARTIFACTS")
    outcomes = [
        evaluate_shadow_candidate(source, resolution=resolution, protocol=protocol)
        for source in sources
    ]
    dispatch, run = arbitrate_shadow_candidates(
        sources,
        outcomes,
        rules=resolution.rules,
        protocol=protocol,
        run_end_controller_ns=run_end_controller_ns,
    )
    records = []
    for source, outcome, dispatch_outcome in zip(sources, outcomes, dispatch, strict=True):
        payload = {
            "schema_version": "1.0",
            "acceptance_status": "development_only",
            "formal_eligible": False,
            "shadow_status": "algorithmic_development_replay",
            "production_shadow": False,
            "test_only_inputs": True,
            "clock_map": resolution.clock_map,
            "clock_map_sha256": resolution.clock_map_sha256,
            "rulespec": resolution.rules,
            "rulespec_sha256": resolution.rules_sha256,
            "source": source,
            "candidate_outcome": outcome,
            "dispatch_outcome": dispatch_outcome,
            "artifact_hashes": artifact_hashes,
            "rtc": {"configured": False, "enabled": False},
            "native_queue": {"configured": False, "used": False},
            "ros_connected": False,
            "robot_actions_sent": 0,
        }
        payload["record_sha256"] = canonical_sha256(payload)
        records.append(payload)
    artifact = {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P9",
        "shadow_status": "algorithmic_development_replay",
        "production_shadow": False,
        "test_only_inputs": True,
        "protocol": protocol.__dict__,
        "run_end_controller_ns": int(run_end_controller_ns),
        "records": records,
        "run_outcome": run,
        "robot_actions_sent": 0,
        "detached_signature": None,
        "approval": None,
    }
    artifact["artifact_sha256"] = canonical_sha256(artifact)
    return artifact


def replay_shadow_record_artifact(artifact: dict) -> dict:
    expected_artifact_hash = artifact.get("artifact_sha256")
    without_hash = {key: value for key, value in artifact.items() if key != "artifact_sha256"}
    if canonical_sha256(without_hash) != expected_artifact_hash:
        raise RuntimeError("P9_RECORD_ARTIFACT_HASH_MISMATCH")
    records = artifact["records"]
    for record in records:
        expected = record["record_sha256"]
        payload = {key: value for key, value in record.items() if key != "record_sha256"}
        if canonical_sha256(payload) != expected:
            raise RuntimeError("P9_RECORD_HASH_MISMATCH")
    first = records[0]
    resolution = ShadowResolution(
        mode="test_only",
        valid=True,
        reasons=(),
        rules=first["rulespec"],
        rules_sha256=first["rulespec_sha256"],
        clock_map=first["clock_map"],
        clock_map_sha256=first["clock_map_sha256"],
    )
    if any(
        record["rulespec_sha256"] != resolution.rules_sha256
        or record["clock_map_sha256"] != resolution.clock_map_sha256
        for record in records
    ):
        raise RuntimeError("P9_RECORD_ARTIFACT_BINDING_MISMATCH")
    protocol = ShadowProtocol(**artifact["protocol"])
    sources = [record["source"] for record in records]
    outcomes = [
        evaluate_shadow_candidate(source, resolution=resolution, protocol=protocol)
        for source in sources
    ]
    dispatch, run = arbitrate_shadow_candidates(
        sources,
        outcomes,
        rules=resolution.rules,
        protocol=protocol,
        run_end_controller_ns=int(artifact["run_end_controller_ns"]),
    )
    if outcomes != [record["candidate_outcome"] for record in records]:
        raise RuntimeError("P9_CANDIDATE_REPLAY_MISMATCH")
    if dispatch != [record["dispatch_outcome"] for record in records] or run != artifact["run_outcome"]:
        raise RuntimeError("P9_DISPATCH_REPLAY_MISMATCH")
    return {
        "schema_version": "1.0",
        "acceptance_status": "development_only",
        "formal_eligible": False,
        "gate": "P9_replay",
        "gate_status": "pass",
        "record_count": len(records),
        "source_artifact_sha256": expected_artifact_hash,
        "actual_dispatch_sha256": canonical_sha256(
            [record["dispatch_outcome"] for record in records]
        ),
        "replay_exact": True,
        "production_shadow": False,
        "robot_actions_sent": 0,
        "detached_signature": None,
        "approval": None,
    }
