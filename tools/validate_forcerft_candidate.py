#!/usr/bin/env python3
"""Offline, zero-update validation for a Stage-3 joint candidate."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

import train_forcerft_critic_warmup as warmup  # noqa: E402


JOINT_CHECKPOINT = (
    ROOT
    / "artifacts/development/stage3/formal_online_r"
    / "task2_policy_execute_stage3_cycle210_smoke_20260829_001"
    / "checkpoints/stage3_joint_cycle_000010"
)
PACKAGED_CHECKPOINT = (
    ROOT
    / "artifacts/development/stage3/published"
    / "stage3_joint_cycle_000010_candidate.v1"
)
PARENT_BINDING = ROOT / "configs/stage3_parent_binding.v1.development.json"
TRAINING_CONFIG = ROOT / "configs/forcerft_actor_critic_training.development.yaml"
RULESPEC = ROOT / "configs/live_action_safety.task2.development.yaml"
EXECUTION_BINDING = (
    ROOT / "artifacts/development/live/task2_cycle210_policy_execution_smoke_binding.v1.json"
)
EXPECTED_REVISION = "stage3-online-r-joint-cycle-000010-candidate"
FIXED_EPISODE_ID = (
    "task2_policy_execute_stage3_cycle210_smoke_20260829_001/episode_000000"
)
FIXED_OBSERVATION_COUNT = 8
FLOW_NOISE_SEED = 20260830


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def extract_saved_td_losses(runtime: Mapping[str, Any]) -> list[float] | None:
    """Read a full saved trace; first/last summaries are deliberately rejected."""

    candidates = (
        runtime.get("step_metrics", {}).get("critic_td_loss")
        if isinstance(runtime.get("step_metrics"), Mapping)
        else None,
        runtime.get("metrics", {}).get("td_losses")
        if isinstance(runtime.get("metrics"), Mapping)
        else None,
    )
    for candidate in candidates:
        if not isinstance(candidate, Sequence) or isinstance(candidate, (str, bytes)):
            continue
        values = [float(value) for value in candidate]
        if len(values) == 20 and all(math.isfinite(value) for value in values):
            return values
    return None


def summarize_td_losses(values: list[float] | None) -> dict[str, Any]:
    if values is None:
        return {
            "TD_LOSSES": [],
            "TD_FIRST_5_MEDIAN": None,
            "TD_LAST_5_MEDIAN": None,
            "TD_TREND": "UNAVAILABLE:SAVED_20_STEP_TD_TRACE_MISSING",
        }
    first = float(np.median(np.asarray(values[:5], dtype=np.float64)))
    last = float(np.median(np.asarray(values[-5:], dtype=np.float64)))
    signs = "".join("+" if right > left else "-" if right < left else "0" for left, right in zip(values, values[1:]))
    longest = current = 0
    for left, right in zip(values, values[1:]):
        current = current + 1 if right > left else 0
        longest = max(longest, current)
    prior = np.asarray(values[:-1], dtype=np.float64)
    prior_median = float(np.median(prior))
    mad = float(np.median(np.abs(prior - prior_median)))
    tail_anomaly = abs(values[-1] - prior_median) > 6.0 * max(mad, np.finfo(np.float64).eps)
    return {
        "TD_LOSSES": values,
        "TD_FIRST_5_MEDIAN": first,
        "TD_LAST_5_MEDIAN": last,
        "TD_TREND": {
            "adjacent_direction_sequence": signs,
            "continuous_growth": longest >= 3,
            "longest_consecutive_growth": longest,
            "single_tail_anomalous_batch": bool(tail_anomaly),
        },
    }


def _resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def _fixed_samples(rows, source_episode: Path, normalizer):
    from forcesmolvla.normalizer import NormalizationLedger

    ordered = sorted(rows, key=lambda row: int(row["identity"]["decision_id"]))
    positions = np.linspace(0, len(ordered) - 1, FIXED_OBSERVATION_COUNT, dtype=np.int64)
    selected = [ordered[int(index)] for index in positions]
    ledger = NormalizationLedger()
    samples: list[dict[str, Any]] = []
    raw_state: list[np.ndarray] = []
    decisions: list[int] = []
    for row in selected:
        observation = row["observation"]
        decision = int(row["identity"]["decision_id"])
        identity = f"offline-fixed-real:{decision}"
        state = np.asarray(observation["state7_absolute"], dtype=np.float64)
        wrench = np.asarray(observation["wrench6_calibrated_tcp"], dtype=np.float64)
        ledger.claim(identity, "state7")
        normalized_state = normalizer.state7.apply(state).astype(np.float32)
        ledger.claim(identity, "wrench6")
        normalized_wrench = normalizer.wrench6.apply(wrench).astype(np.float32)
        samples.append(
            {
                "camera1": warmup._decode_path(
                    str(source_episode / observation["camera_external"]["blob_reference"])
                ),
                "camera2": warmup._decode_path(
                    str(source_episode / observation["camera_wrist"]["blob_reference"])
                ),
                "state7": normalized_state,
                "wrench6": normalized_wrench,
                "task": warmup.TASK,
                "sample_identity": identity,
            }
        )
        raw_state.append(state)
        decisions.append(decision)
    require(
        ledger.counts == {"state7": FIXED_OBSERVATION_COUNT, "wrench6": FIXED_OBSERVATION_COUNT},
        "STAGE3_OFFLINE_NORMALIZER_APPLICATION_COUNT",
    )
    return samples, np.stack(raw_state), decisions, ledger


def _load_actor(path: Path, device: torch.device):
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy

    actor = ForceSmolVLAPolicy.from_pretrained(
        path,
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    ).to(device)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _load_direct_candidate(
    joint_checkpoint: Path,
    parent_path: Path,
    device: torch.device,
) -> Any:
    """Load the saved joint Actor state directly into its bound parent architecture."""

    from safetensors.torch import load_file

    actor = _load_actor(parent_path, device)
    state = load_file(
        str(joint_checkpoint / "candidate_policy/model.safetensors"), device="cpu"
    )
    actor.load_state_dict(state, strict=True)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)
    return actor


def _sample_actor(actor, batch, noise: torch.Tensor, *, label: str) -> torch.Tensor:
    from forcesmolvla.rft.throughput_v2 import FrozenPrefixFlowCounter

    flow = FrozenPrefixFlowCounter(inference_batch_size=4)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        action = flow.sample(
            actor,
            batch,
            noise.clone(),
            call_id=f"stage3-offline-{label}",
            purpose="actor_guidance",
        )
    return action.detach().float()


def _frozen_state_equal(parent_file: Path, candidate_file: Path) -> bool:
    from safetensors import safe_open
    from forcesmolvla.rft.frozen_vlm_trainability import FROZEN_PREFIXES

    with safe_open(parent_file, framework="pt", device="cpu") as parent, safe_open(
        candidate_file, framework="pt", device="cpu"
    ) as candidate:
        names = tuple(name for name in parent.keys() if name.startswith(FROZEN_PREFIXES))
        require(names, "STAGE3_OFFLINE_FROZEN_STATE_EMPTY")
        require(all(name in candidate.keys() for name in names), "STAGE3_OFFLINE_FROZEN_STATE_KEY_MISSING")
        return all(torch.equal(parent.get_tensor(name), candidate.get_tensor(name)) for name in names)


def _load_post_joint_critics(checkpoint: Path, device: torch.device, config):
    from forcesmolvla.rft.critic import build_twin_q

    data = config["data"]
    q1, q2, _q1_target, _q2_target, _conversion = build_twin_q(
        _resolve(data["critic_backbone_npz"]),
        _resolve(data["critic_backbone_manifest"]),
        seed=0,
    )
    for name, module in (("q1", q1), ("q2", q2)):
        state = torch.load(
            checkpoint / "models" / f"{name}_state.pt",
            map_location="cpu",
            weights_only=True,
        )
        module.load_state_dict(state, strict=True)
        module.eval().to(device)
        for parameter in module.parameters():
            parameter.requires_grad_(False)
    return q1, q2


def _physical_actions(label, normalized, raw_state, normalizer, ledger):
    from forcesmolvla.action_delta import ActionDeltaProcessor, decode_binary_gripper_width

    ledger.claim(label, "delta_action7")
    delta = normalizer.delta_action7.inverse(normalized.detach().cpu().numpy())
    delta = decode_binary_gripper_width(delta)
    absolute = ActionDeltaProcessor.from_delta(delta, raw_state)
    roundtrip = ActionDeltaProcessor.to_delta(absolute, raw_state)
    require(np.allclose(roundtrip, delta, rtol=0.0, atol=1e-6), "STAGE3_OFFLINE_ACTION_DELTA_ROUNDTRIP")
    return delta, absolute


def _validate_existing_action_contract(absolute, raw_state) -> None:
    from forcesmolvla.action_delta import ActionSafetyProfile

    rulespec = yaml.safe_load(RULESPEC.read_text(encoding="utf-8"))
    parser_view = copy.deepcopy(rulespec)
    parser_view["mode"] = "test_only"
    binding = json.loads(EXECUTION_BINDING.read_text(encoding="utf-8"))
    profile = ActionSafetyProfile.from_rulespec(
        parser_view,
        rules_sha256=str(binding["rulespec_sha256"]),
    )
    profile.validate_chunk(
        absolute,
        np.ones(absolute.shape[:2], dtype=np.bool_),
        raw_state,
    )


def run(
    checkpoint: Path,
    packaged_checkpoint: Path,
    *,
    expected_revision: str = EXPECTED_REVISION,
    fixed_episode_id: str = FIXED_EPISODE_ID,
) -> dict[str, Any]:
    from forcesmolvla.rft.batch import build_actor_batch
    from forcesmolvla.rft.critic import frozen_task_feature
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
    from forcesmolvla.training_data import load_normalizer_manifest

    checkpoint = checkpoint.resolve()
    runtime = torch.load(
        checkpoint / "state/runtime_state.pt", map_location="cpu", weights_only=False
    )
    result = summarize_td_losses(extract_saved_td_losses(runtime))
    errors: list[str] = []

    require(torch.cuda.is_available(), "STAGE3_OFFLINE_CUDA_UNAVAILABLE")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.manual_seed(FLOW_NOISE_SEED)
    torch.cuda.manual_seed_all(FLOW_NOISE_SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")

    parent_binding = json.loads(PARENT_BINDING.read_text(encoding="utf-8"))
    training_config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    packaged_checkpoint = packaged_checkpoint.resolve()
    candidate_meta = json.loads(
        (packaged_checkpoint / "candidate.json").read_text(encoding="utf-8")
    )
    require(candidate_meta["revision_id"] == expected_revision, "STAGE3_OFFLINE_CANDIDATE_REVISION")
    require(candidate_meta["activated"] is False, "STAGE3_OFFLINE_CANDIDATE_ALREADY_ACTIVATED")

    rows, _macros, source_episodes = warmup.load_formal_online_r(warmup.FORMAL_R_ROOT)
    require(fixed_episode_id in source_episodes, "STAGE3_OFFLINE_FIXED_EPISODE_MISMATCH")
    rows = [
        row for row in rows
        if str(row["identity"]["episode_id"]) == fixed_episode_id
    ]
    require(rows, "STAGE3_OFFLINE_FIXED_EPISODE_EMPTY")
    source_episode = source_episodes[fixed_episode_id]
    normalizer = load_normalizer_manifest(Path(parent_binding["normalizer_binding"]["absolute_path"]))
    samples, raw_state, decisions, ledger = _fixed_samples(rows, source_episode, normalizer)

    generator = torch.Generator(device=device)
    generator.manual_seed(FLOW_NOISE_SEED)
    noise = torch.randn(
        FIXED_OBSERVATION_COUNT, 50, 7,
        dtype=torch.float32, device=device, generator=generator,
    )
    parent_path = Path(parent_binding["actor_parent"]["architecture_binding"]["container_path"])
    parent = _load_actor(parent_path, device)
    batch = build_actor_batch(parent, samples, device, include_action=False)
    parent_action = _sample_actor(parent, batch, noise, label="parent")
    del parent
    torch.cuda.empty_cache()

    direct_candidate = _load_direct_candidate(checkpoint, parent_path, device)
    direct_action = _sample_actor(
        direct_candidate, batch, noise, label="candidate"
    )
    del direct_candidate
    torch.cuda.empty_cache()

    packaged_candidate = _load_actor(packaged_checkpoint, device)
    candidate_action = _sample_actor(
        packaged_candidate, batch, noise, label="candidate"
    )
    del packaged_candidate
    torch.cuda.empty_cache()
    packaged_direct_parity = bool(torch.equal(candidate_action, direct_action))
    if not packaged_direct_parity:
        errors.append("PACKAGED_DIRECT_OUTPUT_MISMATCH")

    expected_shape = (FIXED_OBSERVATION_COUNT, 50, 7)
    shape_valid = (
        tuple(parent_action.shape)
        == tuple(candidate_action.shape)
        == tuple(direct_action.shape)
        == expected_shape
    )
    finite = bool(
        torch.isfinite(parent_action).all()
        and torch.isfinite(candidate_action).all()
        and torch.isfinite(direct_action).all()
    )
    dtype_valid = (
        parent_action.dtype == candidate_action.dtype == direct_action.dtype == torch.float32
    )
    frozen_equal = _frozen_state_equal(
        parent_path / "model.safetensors",
        packaged_checkpoint / "model.safetensors",
    )
    if not shape_valid:
        errors.append("ACTION_SHAPE_INVALID")
    if not finite:
        errors.append("ACTION_NONFINITE")
    if not dtype_valid:
        errors.append("ACTION_DTYPE_INVALID")
    if not frozen_equal:
        errors.append("FROZEN_VLM_CHANGED")

    parent_delta = parent_absolute = candidate_delta = candidate_absolute = None
    contract_valid = False
    try:
        parent_delta, parent_absolute = _physical_actions(
            "parent-output", parent_action, raw_state, normalizer, ledger
        )
        candidate_delta, candidate_absolute = _physical_actions(
            "candidate-output", candidate_action, raw_state, normalizer, ledger
        )
        _validate_existing_action_contract(parent_absolute, raw_state)
        _validate_existing_action_contract(candidate_absolute, raw_state)
        contract_valid = True
    except (RuntimeError, ValueError) as error:
        errors.append(f"ACTION_CONTRACT:{type(error).__name__}:{error}")

    tcp_drift = gripper_drift = None
    if parent_absolute is not None and candidate_absolute is not None:
        tcp = np.abs(candidate_absolute[..., :6] - parent_absolute[..., :6])
        gripper = np.abs(candidate_absolute[..., 6] - parent_absolute[..., 6])
        tcp_drift = [float(tcp.mean()), float(tcp.max())]
        gripper_drift = [float(gripper.mean()), float(gripper.max())]

    delta_mean = torch.tensor(normalizer.delta_action7.mean, dtype=torch.float32, device=device)
    delta_std = torch.tensor(normalizer.delta_action7.std, dtype=torch.float32, device=device)
    parent_q_action = critic_action_for_q_guidance_v2(
        parent_action,
        delta_action_mean7=delta_mean,
        delta_action_std7=delta_std,
    )
    candidate_q_action = critic_action_for_q_guidance_v2(
        candidate_action,
        delta_action_mean7=delta_mean,
        delta_action_std7=delta_std,
    )
    feature = torch.from_numpy(frozen_task_feature()).to(device=device, dtype=torch.float32)
    observation = warmup._critic_observation(samples, feature, device)
    mask = torch.ones(FIXED_OBSERVATION_COUNT, 3, dtype=torch.bool, device=device)
    q1, q2 = _load_post_joint_critics(checkpoint, device, training_config)
    with torch.no_grad():
        parent_q = torch.minimum(
            q1(*observation.as_tuple(), parent_q_action, mask),
            q2(*observation.as_tuple(), parent_q_action, mask),
        )
        candidate_q = torch.minimum(
            q1(*observation.as_tuple(), candidate_q_action, mask),
            q2(*observation.as_tuple(), candidate_q_action, mask),
        )
    require(bool(torch.isfinite(parent_q).all() and torch.isfinite(candidate_q).all()), "STAGE3_OFFLINE_Q_NONFINITE")

    result.update(
        {
            "CANDIDATE_OFFLINE_VALIDATION": "PASS" if not errors else "FAIL",
            "CANDIDATE_STRICT_LOAD": True,
            "PACKAGED_DIRECT_OUTPUT_PARITY": packaged_direct_parity,
            "ACTION_SHAPE_VALID": shape_valid,
            "ACTION_DTYPE": str(candidate_action.dtype),
            "ACTION_FINITE": finite,
            "ACTION_CONTRACT_VALID": contract_valid,
            "TCP6_DRIFT_MEAN_MAX": tcp_drift,
            "GRIPPER_DRIFT_MEAN_MAX": gripper_drift,
            "PARENT_MIN_Q_MEAN": float(parent_q.mean().cpu()),
            "CANDIDATE_MIN_Q_MEAN": float(candidate_q.mean().cpu()),
            "FROZEN_VLM_UNCHANGED": frozen_equal,
            "NORMALIZER_APPLICATION_COUNTS": dict(ledger.counts),
            "FIXED_REAL_OBSERVATION_DECISION_IDS": decisions,
            "CANDIDATE_PUBLISHED": bool(candidate_meta["published"]),
            "CANDIDATE_ACTIVATED": False,
            "DEPLOYMENT_PROFILE_PATH": None,
            "DEPLOYMENT_BINDING_PATH": None,
            "MODEL_UPDATE_COUNT": 0,
            "HARD_ERRORS": errors,
        }
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=JOINT_CHECKPOINT)
    parser.add_argument(
        "--packaged-checkpoint", type=Path, default=PACKAGED_CHECKPOINT
    )
    parser.add_argument("--expected-revision", default=EXPECTED_REVISION)
    parser.add_argument("--fixed-episode-id", default=FIXED_EPISODE_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run(
        args.checkpoint,
        args.packaged_checkpoint,
        expected_revision=args.expected_revision,
        fixed_episode_id=args.fixed_episode_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["CANDIDATE_OFFLINE_VALIDATION"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
