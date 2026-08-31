#!/usr/bin/env python3
"""Run the formal online-replay Critic-only warmup."""

from __future__ import annotations

import argparse
from functools import lru_cache
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import tempfile
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT / "tools") not in sys.path:
    sys.path.insert(0, str(ROOT / "tools"))

FORMAL_R_ROOT = (
    ROOT
    / "artifacts/development/stage3/formal_online_r"
    / "task2_policy_execute_stage3_cycle210_smoke_20260829_001"
)
CHECKPOINT = FORMAL_R_ROOT / "checkpoints/online_replay_critic_warmup_step_000100"
PARENT_BINDING = ROOT / "configs/online_replay_bootstrap_parent_binding.v1.development.json"
TRAINING_CONFIG = ROOT / "configs/forcerft_actor_critic_training.development.yaml"
DATASET = ROOT / "datasets/task2_lerobotv3"
REWARD_TRANSITION_ROOT = ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
SEED = 4404
TASK = "Pick up the purple ring and place it onto the red peg."


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _generation(row: Mapping[str, Any]) -> tuple[int, int, int]:
    value = row["generation"]
    return (
        int(value["policy_epoch"]),
        int(value["takeover_generation"]),
        int(value["reset_generation"]),
    )


def build_ack_macros(rows: Iterable[Mapping[str, Any]]) -> tuple[tuple[Mapping[str, Any], ...], ...]:
    """Build full K=3 macros without crossing an override/takeover boundary."""

    macros: list[tuple[Mapping[str, Any], ...]] = []
    episodes: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        episodes.setdefault(str(row["identity"].get("episode_id", "single")), []).append(row)
    for episode_rows in episodes.values():
        ordered = sorted(
            episode_rows, key=lambda row: int(row["identity"]["decision_id"])
        )
        for stop in range(2, len(ordered)):
            window = tuple(ordered[stop - 2 : stop + 1])
            decisions = [int(row["identity"]["decision_id"]) for row in window]
            sequences = [int(row["policy_lineage"]["selection"]["sequence"]) for row in window]
            if (
                len({_generation(row) for row in window}) == 1
                and decisions == list(range(decisions[0], decisions[0] + 3))
                and sequences == list(range(sequences[0], sequences[0] + 3))
            ):
                macros.append(window)
    return tuple(macros)


def load_formal_online_r(root: Path) -> tuple[
    list[dict[str, Any]],
    tuple[tuple[Mapping[str, Any], ...], ...],
    dict[str, Path],
]:
    admission_files = tuple(sorted((root / "admissions").glob("*.json")))
    require(admission_files, "ONLINE_REPLAY_WARMUP_ADMISSION_RECORD_COUNT")
    expected = 0
    source_episodes: dict[str, Path] = {}
    for path in admission_files:
        admission = json.loads(path.read_text(encoding="utf-8"))
        require(admission.get("policy_execution_smoke_bridge") == "PASS", "ONLINE_REPLAY_WARMUP_BRIDGE_NOT_PASS")
        require(admission.get("source_episode_semantics") == {"formal_replay": False, "real_online_r": False}, "ONLINE_REPLAY_WARMUP_SOURCE_SEMANTICS")
        episode_id = str(admission["episode_id"])
        require(episode_id not in source_episodes, "ONLINE_REPLAY_WARMUP_ADMISSION_EPISODE_DUPLICATE")
        source_episodes[episode_id] = Path(admission["source_episode"])
        expected += int(admission["accepted_unique_r_transition_count"])

    rows = []
    for path in (root / "replay").glob("*.json"):
        envelope = json.loads(path.read_text(encoding="utf-8"))
        row = envelope["payload"]
        require(
            row["classification"] == "recorded_live_policy_execution_smoke"
            and row["action_authority"]["executed_action_source"] == "policy"
            and row["eligibility"] == {
                "formal_replay": True,
                "formal_training_replay_eligible": True,
                "real_online_r": True,
                "replay_membership": "R_online",
            },
            "ONLINE_REPLAY_WARMUP_R_MEMBERSHIP",
        )
        require(
            str(row["identity"]["episode_id"]) in source_episodes,
            "ONLINE_REPLAY_WARMUP_R_SOURCE_EPISODE_MISSING",
        )
        rows.append(row)
    require(len(rows) == expected >= 100, "ONLINE_REPLAY_WARMUP_TRAINING_STARTS")
    require(len({row["identity"]["transition_uid"] for row in rows}) == len(rows), "ONLINE_REPLAY_WARMUP_R_UID_DUPLICATE")
    macros = build_ack_macros(rows)
    require(macros and any(macro[-1]["outcome"]["terminated"] for macro in macros), "ONLINE_REPLAY_WARMUP_R_MACRO_TERMINAL_MISSING")
    return rows, macros, source_episodes


@lru_cache(maxsize=512)
def _decode_path(path: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "ONLINE_REPLAY_WARMUP_IMAGE_SHAPE")
    return np.ascontiguousarray(value.transpose(2, 0, 1))


def _decode_bytes(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        value = np.asarray(image.convert("RGB"), dtype=np.uint8)
    require(value.shape == (480, 640, 3), "ONLINE_REPLAY_WARMUP_DEMO_IMAGE_SHAPE")
    return np.ascontiguousarray(value.transpose(2, 0, 1))


class FormalReplay:
    def __init__(self, macros, source_episodes: Mapping[str, Path], normalizer) -> None:
        self.macros = tuple(macros)
        self.source_episodes = dict(source_episodes)
        self.normalizer = normalizer

    def _sample(
        self, observation: Mapping[str, Any], identity: str, episode_id: str
    ) -> dict[str, Any]:
        source_episode = self.source_episodes[episode_id]
        return {
            "camera1": _decode_path(str(source_episode / observation["camera_external"]["blob_reference"])),
            "camera2": _decode_path(str(source_episode / observation["camera_wrist"]["blob_reference"])),
            "state7": self.normalizer.state7.apply(np.asarray(observation["state7_absolute"], dtype=np.float64)).astype(np.float32),
            "wrench6": self.normalizer.wrench6.apply(np.asarray(observation["wrench6_calibrated_tcp"], dtype=np.float64)).astype(np.float32),
            "task": TASK,
            "sample_identity": identity,
        }

    def materialize(self, index: int) -> dict[str, Any]:
        from forcesmolvla.action_delta import ActionDeltaProcessor

        macro = self.macros[index]
        first, final = macro[0], macro[-1]
        state = np.asarray(first["observation"]["state7_absolute"], dtype=np.float64)
        absolute = np.asarray(
            [row["action_authority"]["accepted_absolute_action7"] for row in macro],
            dtype=np.float64,
        )
        for slot in range(3):
            width = absolute[slot, 6]
            require(np.isclose(width, 0.0, atol=1e-6) or np.isclose(width, 0.085, atol=1e-6), "ONLINE_REPLAY_WARMUP_R_GRIPPER_ENDPOINT")
            absolute[slot, 6] = 0.0 if width < 0.0425 else 0.085
        action = self.normalizer.delta_action7.apply(
            ActionDeltaProcessor.to_delta(absolute, state)
        ).astype(np.float32)
        uid = str(final["identity"]["transition_uid"])
        episode_id = str(final["identity"]["episode_id"])
        return {
            "current": self._sample(first["observation"], f"R:{uid}:current", episode_id),
            "next": self._sample(final["next_observation"], f"R:{uid}:next", episode_id),
            "behavior_action": action,
            "reward": float(final["outcome"]["reward"]),
            "terminated": bool(final["outcome"]["terminated"]),
            "bootstrap": bool(final["outcome"]["bootstrap_mask"]),
            "discount": float(final["outcome"]["discount"]),
            "identity": f"R:{uid}",
        }


class DemoReplay:
    """Read the already converted online-training demonstration replay."""

    COLUMNS = (
        "observation.images.camera1",
        "observation.images.camera2",
        "observation.state",
        "observation.wrench",
    )

    def __init__(self, normalizer) -> None:
        from forcesmolvla.rft.losses import load_authorized_reward_train_transitions

        self.rows = load_authorized_reward_train_transitions(
            REWARD_TRANSITION_ROOT
        ).to_pylist()
        self.population = tuple(
            index for index, row in enumerate(self.rows)
            if all(row["executed_action_mask"])
        )
        require(self.population, "ONLINE_REPLAY_WARMUP_DEMO_POPULATION_EMPTY")
        conversion = json.loads((DATASET / "conversion_manifest.json").read_text(encoding="utf-8"))
        self.tasks = {item["raw_episode_id"]: item["task"] for item in conversion["episodes"]}
        self.normalizer = normalizer
        self.raw: dict[tuple[str, int], dict[str, Any]] = {}

    def prefetch(self, schedule: Iterable[Iterable[int]]) -> None:
        import pyarrow.parquet as pq

        requested: dict[str, set[int]] = {}
        for batch in schedule:
            for index in batch:
                row = self.rows[index]
                for key in ("observation_row_reference", "next_observation_row_reference"):
                    reference = row[key]
                    requested.setdefault(reference["data_relative_path"], set()).add(int(reference["row_index"]))
        for position, (relative, indices) in enumerate(sorted(requested.items()), start=1):
            table = pq.read_table(DATASET / relative, columns=list(self.COLUMNS))
            for index in indices:
                self.raw[(relative, index)] = table.slice(index, 1).to_pylist()[0]
            del table
            if position % 10 == 0 or position == len(requested):
                print(f"[warmup] prefetched demonstration files {position}/{len(requested)}", file=sys.stderr, flush=True)

    def _sample(self, reference: Mapping[str, Any], identity: str, task: str) -> dict[str, Any]:
        source = self.raw[(reference["data_relative_path"], int(reference["row_index"]))]
        return {
            "camera1": _decode_bytes(source["observation.images.camera1"]["bytes"]),
            "camera2": _decode_bytes(source["observation.images.camera2"]["bytes"]),
            "state7": self.normalizer.state7.apply(np.asarray(source["observation.state"], dtype=np.float64)).astype(np.float32),
            "wrench6": self.normalizer.wrench6.apply(np.asarray(source["observation.wrench"], dtype=np.float64)).astype(np.float32),
            "task": task,
            "sample_identity": identity,
        }

    def materialize(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        identity = f"D:{row['episode_id']}:{row['transition_index']}"
        action = np.asarray(row["normalized_delta_action_exec_flat"], dtype=np.float32).reshape(3, 7)
        require(action.shape == (3, 7), "ONLINE_REPLAY_WARMUP_D_ACTION_SHAPE")
        return {
            "current": self._sample(row["observation_row_reference"], identity + ":current", self.tasks[row["episode_id"]]),
            "next": self._sample(row["next_observation_row_reference"], identity + ":next", self.tasks[row["episode_id"]]),
            "behavior_action": action,
            "reward": float(row["reward"]),
            "terminated": bool(row["terminated"]),
            "bootstrap": bool(row["bootstrap_mask"]),
            "discount": float(row["discount"]),
            "identity": identity,
        }


def _resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def load_parents(device: torch.device):
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.critic import build_twin_q

    binding = json.loads(PARENT_BINDING.read_text(encoding="utf-8"))
    config = yaml.safe_load(TRAINING_CONFIG.read_text(encoding="utf-8"))
    require(binding["binding_id"] == "approved_hybrid_cycle210_actor_g7a_r2_twin_q.v1", "ONLINE_REPLAY_WARMUP_PARENT_BINDING")
    actor = ForceSmolVLAPolicy.from_pretrained(
        Path(binding["actor_parent"]["architecture_binding"]["container_path"]),
        local_files_only=True,
        force_download=False,
        strict=True,
        artifact_use="development",
    ).to(device)
    actor.eval()
    for parameter in actor.parameters():
        parameter.requires_grad_(False)

    data = config["data"]
    q1, q2, q1_target, q2_target, _conversion = build_twin_q(
        _resolve(data["critic_backbone_npz"]),
        _resolve(data["critic_backbone_manifest"]),
        seed=0,
    )
    modules = {"online_q1": q1, "online_q2": q2, "target_q1": q1_target, "target_q2": q2_target}
    for group in ("critic_parent", "target_critic_parent"):
        for record in binding[group]["artifacts"]:
            state = torch.load(record["absolute_path"], map_location="cpu", weights_only=True)
            modules[record["logical_role"]].load_state_dict(state, strict=True)
    q1.train(True)
    q2.train(True)
    q1_target.make_permanent_eval_target()
    q2_target.make_permanent_eval_target()
    q1, q2, q1_target, q2_target = (
        module.to(device) for module in (q1, q2, q1_target, q2_target)
    )
    require(not actor.training and all(not p.requires_grad for p in actor.parameters()), "ONLINE_REPLAY_WARMUP_ACTOR_NOT_FROZEN")
    return actor, q1, q2, q1_target, q2_target, binding, config


def _critic_observation(samples: list[dict[str, Any]], feature: torch.Tensor, device: torch.device):
    from forcesmolvla.rft.losses import CriticObservation

    return CriticObservation(
        torch.from_numpy(np.stack([item["camera1"] for item in samples])).to(device),
        torch.from_numpy(np.stack([item["camera2"] for item in samples])).to(device),
        feature[None, :].expand(len(samples), -1).clone(),
        torch.from_numpy(np.stack([item["state7"] for item in samples])).to(device),
        torch.from_numpy(np.stack([item["wrench6"] for item in samples])).to(device),
    ).validate()


def build_batch(rows: list[dict[str, Any]], actor, feature: torch.Tensor, device: torch.device) -> dict[str, Any]:
    from forcesmolvla.rft.batch import build_actor_batch

    rows = sorted(rows, key=lambda row: row["terminated"])
    current = [row["current"] for row in rows]
    following = [row["next"] for row in rows]
    return {
        "current_observation": _critic_observation(current, feature, device),
        "next_observation": _critic_observation(following, feature, device),
        "next_actor_batch": build_actor_batch(actor, following, device, include_action=False),
        "behavior_action": torch.from_numpy(np.stack([row["behavior_action"] for row in rows])).to(device),
        "behavior_mask": torch.ones(len(rows), 3, dtype=torch.bool, device=device),
        "reward": torch.tensor([row["reward"] for row in rows], dtype=torch.float32, device=device),
        "terminated": torch.tensor([row["terminated"] for row in rows], dtype=torch.bool, device=device),
        "bootstrap": torch.tensor([row["bootstrap"] for row in rows], dtype=torch.bool, device=device),
        "discount": torch.tensor([row["discount"] for row in rows], dtype=torch.float32, device=device),
        "identities": tuple(row["identity"] for row in rows),
    }


def _gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    total = sum(
        float(parameter.grad.detach().float().square().sum().cpu())
        for parameter in parameters if parameter.grad is not None
    )
    return math.sqrt(total)


def _range_update(current: list[float], value: torch.Tensor) -> None:
    value = value.detach().float()
    require(bool(torch.isfinite(value).all()), "ONLINE_REPLAY_WARMUP_NONFINITE_METRIC")
    current[0] = min(current[0], float(value.min().cpu()))
    current[1] = max(current[1], float(value.max().cpu()))


def save_checkpoint(
    path: Path,
    *,
    modules: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    runtime_state: Mapping[str, Any],
    actor_parent: Mapping[str, Any],
) -> None:
    require(not path.exists(), "ONLINE_REPLAY_WARMUP_CHECKPOINT_EXISTS")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + ".tmp-", dir=path.parent))
    try:
        (temporary / "models").mkdir()
        (temporary / "optimizers").mkdir()
        (temporary / "state").mkdir()
        for name, module in modules.items():
            torch.save(module.state_dict(), temporary / "models" / f"{name}_state.pt")
        torch.save(optimizer.state_dict(), temporary / "optimizers/critic_optimizer_state.pt")
        torch.save(dict(runtime_state), temporary / "state/runtime_state.pt")
        metadata = {
            "kind": "online_replay_critic_warmup",
            "complete": True,
            "critic_warmup_steps": int(runtime_state["counters"]["critic_warmup_steps"]),
            "actor_parent_binding": {
                "binding_id": actor_parent["binding_id"],
                "binding_path": str(PARENT_BINDING),
                "actor_path": actor_parent["actor_parent"]["absolute_path"],
                "frozen": True,
                "eval": True,
                "no_grad": True,
                "optimizer_created": False,
                "update_count": 0,
            },
        }
        (temporary / "metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, path)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_checkpoint_once(
    path: Path,
    *,
    modules: Mapping[str, torch.nn.Module],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, Any]:
    from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger

    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    require(metadata.get("complete") is True, "ONLINE_REPLAY_WARMUP_CHECKPOINT_INCOMPLETE")
    for name, module in modules.items():
        state = torch.load(path / "models" / f"{name}_state.pt", map_location=device, weights_only=True)
        module.load_state_dict(state, strict=True)
    optimizer.load_state_dict(torch.load(
        path / "optimizers/critic_optimizer_state.pt", map_location=device, weights_only=True
    ))
    runtime = torch.load(path / "state/runtime_state.pt", map_location="cpu", weights_only=False)
    restored = UpdateCreditLedger.from_state_dict(runtime["sample_credit"])
    require(restored.snapshot().available == runtime["sample_credit"]["minted"] - runtime["sample_credit"]["consumed"], "ONLINE_REPLAY_WARMUP_CREDIT_RESTORE")
    probe = random.Random()
    probe.setstate(runtime["sampler_state"]["r_rng"])
    probe.setstate(runtime["sampler_state"]["d_rng"])
    noise = torch.Generator(device="cpu")
    noise.set_state(runtime["rng_state"]["noise_generator_cpu_probe"])
    return runtime


def run(*, steps: int, checkpoint: Path) -> dict[str, Any]:
    from forcesmolvla.rft.critic import frozen_task_feature
    from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2
    from forcesmolvla.rft.online.training_losses import compute_online_twin_q_td_loss
    from forcesmolvla.rft.online.sample_credit import UpdateCreditLedger
    from forcesmolvla.rft.throughput_v2 import FrozenPrefixFlowCounter, fast_polyak_update, index_actor_batch
    from forcesmolvla.training_data import load_normalizer_manifest

    require(steps == 100, "ONLINE_REPLAY_WARMUP_REQUIRES_100_STEPS")
    require(torch.cuda.is_available(), "ONLINE_REPLAY_WARMUP_CUDA_UNAVAILABLE")
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    all_r, r_macros, source_episodes = load_formal_online_r(FORMAL_R_ROOT)
    actor, q1, q2, q1_target, q2_target, binding, config = load_parents(device)
    normalizer = load_normalizer_manifest(Path(binding["normalizer_binding"]["absolute_path"]))
    r_replay = FormalReplay(r_macros, source_episodes, normalizer)
    d_replay = DemoReplay(normalizer)

    r_rng = random.Random(SEED + 1)
    d_rng = random.Random(SEED + 2)
    r_schedule = [r_rng.sample(range(len(r_replay.macros)), 32) for _ in range(steps)]
    d_schedule = [d_rng.sample(d_replay.population, 32) for _ in range(steps)]
    d_replay.prefetch(d_schedule)

    credits = UpdateCreditLedger(credits_per_transition=1, credits_per_joint_cycle=1)
    for row in all_r:
        require(credits.mint_for_unique_online_transition(row["identity"]["transition_uid"]), "ONLINE_REPLAY_WARMUP_CREDIT_DUPLICATE")
    require(credits.snapshot().credited_transition_count == len(all_r), "ONLINE_REPLAY_WARMUP_CREDIT_COUNT")

    trainable = [
        parameter for module in (q1, q2) for parameter in module.parameters()
        if parameter.requires_grad
    ]
    require(trainable, "ONLINE_REPLAY_WARMUP_NO_CRITIC_PARAMETERS")
    optimizer = torch.optim.Adam(trainable, lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    require(not optimizer.state, "ONLINE_REPLAY_WARMUP_CRITIC_OPTIMIZER_NOT_FRESH")
    feature = torch.from_numpy(frozen_task_feature()).to(device=device, dtype=torch.float32)
    delta_mean = torch.tensor(normalizer.delta_action7.mean, dtype=torch.float32, device=device)
    delta_std = torch.tensor(normalizer.delta_action7.std, dtype=torch.float32, device=device)
    noise_generator = torch.Generator(device=device).manual_seed(SEED + 3)
    flow = FrozenPrefixFlowCounter(inference_batch_size=int(config["batching"]["flow_inference_subbatch"]))

    losses: list[float] = []
    q1_range = [math.inf, -math.inf]
    q2_range = [math.inf, -math.inf]
    target_range = [math.inf, -math.inf]
    gradient_range = [math.inf, -math.inf]
    target_updates = 0
    nonfinite_count = 0
    oom_count = 0
    r_consumed = d_consumed = 0
    terminal_samples = 0

    for step in range(steps):
        credits.consume_joint_cycle()
        rows = [r_replay.materialize(index) for index in r_schedule[step]]
        rows.extend(d_replay.materialize(index) for index in d_schedule[step])
        r_consumed += 32
        d_consumed += 32
        batch = build_batch(rows, actor, feature, device)
        terminal_samples += int(batch["terminated"].sum())
        nonterminal_count = int((~batch["terminated"]).sum())
        next_actor_batch = index_actor_batch(batch["next_actor_batch"], list(range(nonterminal_count)))
        optimizer.zero_grad(set_to_none=True)
        target_outputs: list[torch.Tensor] = []
        hooks = [
            q1_target.register_forward_hook(lambda _m, _i, output: target_outputs.append(output.detach())),
            q2_target.register_forward_hook(lambda _m, _i, output: target_outputs.append(output.detach())),
        ]

        def next_action(_observation) -> torch.Tensor:
            require(not actor.training, "ONLINE_REPLAY_WARMUP_ACTOR_TRAIN_MODE")
            noise = torch.randn(
                nonterminal_count, 50, 7,
                dtype=torch.float32, device=device, generator=noise_generator,
            )
            with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                chunk = flow.sample(
                    actor, next_actor_batch, noise,
                    call_id=f"online-critic-warmup-{step:03d}", purpose="td_next",
                )
            return critic_action_for_q_guidance_v2(
                chunk, delta_action_mean7=delta_mean, delta_action_std7=delta_std,
            ).detach().float()

        try:
            result = compute_online_twin_q_td_loss(
                q1=q1,
                q2=q2,
                q1_target=q1_target,
                q2_target=q2_target,
                observation=batch["current_observation"],
                next_observation=batch["next_observation"],
                ack_behavior_action_k7=batch["behavior_action"],
                behavior_mask=batch["behavior_mask"],
                reward=batch["reward"],
                discount=batch["discount"],
                terminated=batch["terminated"],
                bootstrap_mask=batch["bootstrap"],
                next_policy_action_fn=next_action,
            )
            require(
                result.calql_candidate_calls == result.random_candidate_calls == result.mc_return_reads == 0,
                "ONLINE_REPLAY_WARMUP_NOT_PURE_TD",
            )
            result.total.backward()
            require(
                all(parameter.grad is None for parameter in actor.parameters())
                and all(parameter.grad is None for target in (q1_target, q2_target) for parameter in target.parameters()),
                "ONLINE_REPLAY_WARMUP_GRADIENT_OWNERSHIP",
            )
            gradient = _gradient_norm(trainable)
            require(math.isfinite(gradient), "ONLINE_REPLAY_WARMUP_GRADIENT_NONFINITE")
            gradient_range[0] = min(gradient_range[0], gradient)
            gradient_range[1] = max(gradient_range[1], gradient)
            torch.nn.utils.clip_grad_norm_(trainable, 10.0)
            optimizer.step()
            fast_polyak_update(q1, q1_target, tau=0.005, target_name="q1_target")
            fast_polyak_update(q2, q2_target, tau=0.005, target_name="q2_target")
            target_updates += 1
            losses.append(float(result.total.detach().cpu()))
            _range_update(q1_range, result.q1_value)
            _range_update(q2_range, result.q2_value)
            require(len(target_outputs) == (2 if nonterminal_count else 0), "ONLINE_REPLAY_WARMUP_TARGET_CALL_COUNT")
            for value in target_outputs:
                _range_update(target_range, value)
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            raise
        except FloatingPointError:
            nonfinite_count += 1
            raise
        finally:
            for hook in hooks:
                hook.remove()
        if (step + 1) % 10 == 0:
            print(f"[warmup] critic steps {step + 1}/{steps}", file=sys.stderr, flush=True)

    require(len(losses) == steps and target_updates == steps, "ONLINE_REPLAY_WARMUP_STEP_COUNT")
    require(terminal_samples > 0, "ONLINE_REPLAY_WARMUP_TERMINAL_NOT_SAMPLED")
    require(nonfinite_count == oom_count == 0, "ONLINE_REPLAY_WARMUP_RUNTIME_FAILURE")
    for module in (q1, q2, q1_target, q2_target):
        require(all(bool(torch.isfinite(value).all()) for value in module.state_dict().values() if value.is_floating_point()), "ONLINE_REPLAY_WARMUP_MODEL_NONFINITE")

    cpu_probe = torch.Generator(device="cpu")
    cpu_probe.set_state(torch.get_rng_state())
    runtime_state = {
        "counters": {
            "critic_warmup_steps": steps,
            "r_samples_consumed": r_consumed,
            "d_samples_consumed": d_consumed,
            "target_polyak_update_count": target_updates,
            "actor_update_count": 0,
        },
        "replay": {
            "formal_r_root": str(FORMAL_R_ROOT),
            "unique_r_transition_count": len(all_r),
            "eligible_ack_macro_count": len(r_macros),
            "demo_transition_root": str(REWARD_TRANSITION_ROOT),
            "mix": {"R": 32, "D": 32},
        },
        "sample_credit": credits.state_dict(),
        "sampler_state": {
            "step": steps,
            "r_rng": r_rng.getstate(),
            "d_rng": d_rng.getstate(),
        },
        "rng_state": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all(),
            "noise_generator": noise_generator.get_state().cpu(),
            "noise_generator_cpu_probe": cpu_probe.get_state(),
        },
        "actor": {
            "frozen": True,
            "eval": True,
            "no_grad": True,
            "optimizer_created": False,
            "update_count": 0,
            "q_guidance_enabled": False,
        },
    }
    modules = {"q1": q1, "q2": q2, "q1_target": q1_target, "q2_target": q2_target}
    save_checkpoint(
        checkpoint,
        modules=modules,
        optimizer=optimizer,
        runtime_state=runtime_state,
        actor_parent=binding,
    )
    restored = load_checkpoint_once(
        checkpoint, modules=modules, optimizer=optimizer, device=device,
    )
    require(restored["counters"]["critic_warmup_steps"] == steps, "ONLINE_REPLAY_WARMUP_CHECKPOINT_LOAD")

    median = lambda values: float(np.median(np.asarray(values, dtype=np.float64)))
    return {
        "CRITIC_WARMUP_STEPS": steps,
        "R_SAMPLES_CONSUMED": r_consumed,
        "D_SAMPLES_CONSUMED": d_consumed,
        "TD_LOSS_FIRST_20_MEDIAN": median(losses[:20]),
        "TD_LOSS_LAST_20_MEDIAN": median(losses[-20:]),
        "Q1_MIN_MAX": q1_range,
        "Q2_MIN_MAX": q2_range,
        "TARGET_Q_MIN_MAX": target_range,
        "GRADIENT_NORM_MIN_MAX": gradient_range,
        "NONFINITE_COUNT": nonfinite_count,
        "OOM_COUNT": oom_count,
        "ACTOR_OPTIMIZER_CREATED": False,
        "ACTOR_UPDATE_COUNT": 0,
        "ACTOR_Q_GUIDANCE_ENABLED": False,
        "TARGET_POLYAK_UPDATE_COUNT": target_updates,
        "REMAINING_SAMPLE_CREDITS": credits.snapshot().available,
        "ONLINE_REPLAY_CHECKPOINT_PATH": str(checkpoint),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    print(json.dumps(run(steps=args.steps, checkpoint=args.checkpoint), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
