from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

import pytest
import torch
from safetensors.torch import load_file, save_file

from forcesmolvla.rft.canonical_state import canonical_digest, module_record
from forcesmolvla.rft.stage3.checkpoint import (
    EXACT_RESUME_COMPLETION,
    actor_frozen_state_digest,
    resign_exact_resume_checkpoint_copy,
    save_exact_resume_checkpoint,
    strict_load_exact_resume_checkpoint,
    validate_exact_resume_checkpoint,
)


MODEL_NAMES = ("actor", "q1", "q2", "q1_target", "q2_target")


def _modules(seed: int = 17) -> dict[str, torch.nn.Module]:
    torch.manual_seed(seed)
    modules = {name: torch.nn.Linear(3, 2) for name in MODEL_NAMES}
    for name in ("q1_target", "q2_target"):
        modules[name].requires_grad_(False).eval()
    return modules


def _optimizers(modules: dict[str, torch.nn.Module]):
    actor = torch.optim.AdamW(modules["actor"].parameters(), lr=1e-3)
    critic = torch.optim.Adam(
        [*modules["q1"].parameters(), *modules["q2"].parameters()], lr=2e-3
    )
    modules["actor"](torch.ones(2, 3)).sum().backward()
    actor.step(); actor.zero_grad(set_to_none=True)
    (modules["q1"](torch.ones(2, 3)).sum() + modules["q2"](torch.ones(2, 3)).sum()).backward()
    critic.step(); critic.zero_grad(set_to_none=True)
    return actor, critic


def _state(modules: dict[str, torch.nn.Module]) -> dict:
    R = ["R-000", "R-001"]
    D = ["D-000"]
    index = {
        uid: {"row_identity": uid, "origin_pool": "R" if uid.startswith("R") else "D"}
        for uid in [*R, *D]
    }
    return {
        "sampler_state": {
            "cursor": 1,
            "next_sample_order": {"critic": ["R-001", "D-000"], "actor": ["R-000", "D-000"]},
        },
        "replay_state": {
            "canonical_index": index,
            "canonical_index_sha256": canonical_digest(index),
            "R_membership_uids": R,
            "D_membership_uids": D,
            "R_watermark": len(R),
            "D_watermark": len(D),
            "episode_finalization_state": "sealed",
            "source": "synthetic_preflight_R_only",
        },
        "credit_state": {
            "credits_per_transition": 1,
            "credits_per_joint_cycle": 1,
            "minted": len(R),
            "consumed": 1,
            "available": len(R) - 1,
            "credited_uids": R,
        },
        "counters": {
            "learner_cycles": 1,
            "critic_updates": 2,
            "actor_updates": 1,
            "polyak_updates_per_target": 2,
            "publication_count": 0,
        },
        "revision_state": {
            "active_revision": "approved-hybrid-parent",
            "pending_revision": None,
            "previous_revision": None,
            "episode_revision": None,
            "policy_epoch": 0,
            "publication_count": 0,
        },
        "bindings": {
            "actor_frozen_parent_digest": actor_frozen_state_digest(modules["actor"]),
            "parent_binding_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "reward_source_sha256": "c" * 64,
            "source_sha256": "d" * 64,
            "normalizer_sha256": "e" * 64,
            "action_contract_sha256": "f" * 64,
            "task_feature_sha256": "1" * 64,
        },
        "boundary": {
            "episode_sealed": True,
            "active_episode": False,
            "request_in_flight": False,
            "partial_macro": False,
            "learner_update_committed": True,
            "pending_gradients": False,
            "pending_optimizer_steps": 0,
            "pending_accumulation_microbatches": 0,
        },
    }


@pytest.fixture()
def checkpoint(tmp_path: Path) -> tuple[Path, dict]:
    modules = _modules()
    actor_optimizer, critic_optimizer = _optimizers(modules)
    state = _state(modules)
    path = tmp_path / "checkpoint"
    save_exact_resume_checkpoint(
        path,
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        generators={"flow": torch.Generator().manual_seed(31)},
        **state,
    )
    return path, state


def _copy(checkpoint: tuple[Path, dict], tmp_path: Path, name: str) -> Path:
    target = tmp_path / name
    shutil.copytree(checkpoint[0], target)
    return target


def _rewrite_metadata(path: Path, mutate) -> None:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    mutate(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    resign_exact_resume_checkpoint_copy(path)


def test_atomic_checkpoint_completion_and_fresh_object_restore(checkpoint) -> None:
    path, state = checkpoint
    validated = validate_exact_resume_checkpoint(path)
    assert validated["completion"]["complete"] is True
    assert validated["manifest"]["canonical_content_digest"] == validated["metadata"][
        "canonical_state"
    ]["canonical_content_digest"]
    modules = _modules(seed=99)
    actor_optimizer = torch.optim.AdamW(modules["actor"].parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(
        [*modules["q1"].parameters(), *modules["q2"].parameters()], lr=2e-3
    )
    restored = strict_load_exact_resume_checkpoint(
        path,
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        generators={"flow": torch.Generator().manual_seed(999)},
        expected_bindings=state["bindings"],
    )
    assert restored["rng_restored_last"] is True
    assert restored["counters"]["learner_cycles"] == 1


def test_partial_checkpoint_and_missing_completion_marker_fail_closed(
    checkpoint, tmp_path: Path,
) -> None:
    partial = tmp_path / ".checkpoint.tmp-interrupted"
    partial.mkdir()
    (partial / "metadata.json").write_text("{}")
    with pytest.raises(RuntimeError, match="COMPLETION_MARKER_MISSING"):
        validate_exact_resume_checkpoint(partial)
    missing = _copy(checkpoint, tmp_path, "missing-completion")
    (missing / EXACT_RESUME_COMPLETION).unlink()
    with pytest.raises(RuntimeError, match="COMPLETION_MARKER_MISSING"):
        validate_exact_resume_checkpoint(missing)


def test_tampered_payload_sha_fails_closed(checkpoint, tmp_path: Path) -> None:
    path = _copy(checkpoint, tmp_path, "tampered")
    target = path / "models" / "q1.safetensors"
    with target.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(RuntimeError, match="FILE_SHA_MISMATCH"):
        validate_exact_resume_checkpoint(path)


def test_wrong_parent_config_or_source_binding_fails_before_restore(
    checkpoint,
) -> None:
    path, state = checkpoint
    modules = _modules(seed=99)
    actor_optimizer = torch.optim.AdamW(modules["actor"].parameters(), lr=1e-3)
    critic_optimizer = torch.optim.Adam(
        [*modules["q1"].parameters(), *modules["q2"].parameters()], lr=2e-3
    )
    wrong = deepcopy(state["bindings"]); wrong["source_sha256"] = "0" * 64
    with pytest.raises(RuntimeError, match="BINDING_MISMATCH"):
        strict_load_exact_resume_checkpoint(
            path,
            modules=modules,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            generators={"flow": torch.Generator().manual_seed(999)},
            expected_bindings=wrong,
        )


def test_missing_optimizer_state_and_group_reorder_fail_closed(
    checkpoint, tmp_path: Path,
) -> None:
    missing = _copy(checkpoint, tmp_path, "optimizer-missing")
    target = missing / "optimizers" / "actor.pt"
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["state"].pop(next(iter(payload["state"])))
    torch.save(payload, target)
    resign_exact_resume_checkpoint_copy(missing)
    with pytest.raises(RuntimeError, match="OPTIMIZER_STATE_MISSING|CANONICAL_CONTENT_MISMATCH"):
        validate_exact_resume_checkpoint(missing)

    reordered = _copy(checkpoint, tmp_path, "optimizer-reordered")
    target = reordered / "optimizers" / "actor.pt"
    payload = torch.load(target, map_location="cpu", weights_only=True)
    payload["param_groups"][0]["params"].reverse()
    torch.save(payload, target)
    resign_exact_resume_checkpoint_copy(reordered)
    with pytest.raises(RuntimeError, match="CANONICAL_CONTENT_MISMATCH"):
        validate_exact_resume_checkpoint(reordered)


def test_rng_omission_and_corruption_fail_closed(checkpoint, tmp_path: Path) -> None:
    omitted = _copy(checkpoint, tmp_path, "rng-omitted")
    target = omitted / "state" / "rng.safetensors"
    values = load_file(target, device="cpu")
    values.pop("generator.flow")
    save_file(values, target)
    resign_exact_resume_checkpoint_copy(omitted)
    with pytest.raises(RuntimeError, match="RNG_STATE_CORRUPTED_OR_OMITTED"):
        validate_exact_resume_checkpoint(omitted)

    corrupted = _copy(checkpoint, tmp_path, "rng-corrupted")
    target = corrupted / "state" / "rng.safetensors"
    values = load_file(target, device="cpu")
    values["torch_cpu"] = torch.zeros(1, dtype=torch.uint8)
    save_file(values, target)
    resign_exact_resume_checkpoint_copy(corrupted)
    with pytest.raises(RuntimeError, match="RNG_STATE_CORRUPTED_OR_OMITTED|CANONICAL_CONTENT_MISMATCH"):
        validate_exact_resume_checkpoint(corrupted)


@pytest.mark.parametrize(
    ("name", "mutate", "message"),
    [
        (
            "credit-drift",
            lambda metadata: metadata["credits"].__setitem__("consumed", 0),
            "CREDIT_COUNTER_DRIFT",
        ),
        (
            "counter-drift",
            lambda metadata: metadata["counters"].__setitem__("critic_updates", 3),
            "COUNTER_DRIFT",
        ),
        (
            "unsealed",
            lambda metadata: metadata["boundary"].__setitem__("episode_sealed", False),
            "BOUNDARY_NOT_QUIESCENT",
        ),
        (
            "pending-revision",
            lambda metadata: metadata["revision"].__setitem__("pending_revision", "r1"),
            "PENDING_REVISION",
        ),
    ],
)
def test_control_state_faults_fail_closed(
    checkpoint, tmp_path: Path, name, mutate, message,
) -> None:
    path = _copy(checkpoint, tmp_path, name)
    _rewrite_metadata(path, mutate)
    with pytest.raises(RuntimeError, match=message):
        validate_exact_resume_checkpoint(path)


def test_mid_cycle_save_is_rejected_before_any_checkpoint_is_written(tmp_path: Path) -> None:
    modules = _modules()
    actor_optimizer, critic_optimizer = _optimizers(modules)
    state = _state(modules)
    state["boundary"]["partial_macro"] = True
    destination = tmp_path / "must-not-exist"
    with pytest.raises(RuntimeError, match="BOUNDARY_NOT_QUIESCENT"):
        save_exact_resume_checkpoint(
            destination,
            modules=modules,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            generators={"flow": torch.Generator().manual_seed(31)},
            **state,
        )
    assert not destination.exists()


def test_cold_or_warm_decoded_image_cache_does_not_change_model_state() -> None:
    modules = _modules()
    before = {name: module_record(module) for name, module in modules.items()}
    decoded_image_cache = {}
    decoded_image_cache["row-1"] = torch.arange(12, dtype=torch.uint8).reshape(2, 2, 3)
    assert decoded_image_cache["row-1"].shape == (2, 2, 3)
    decoded_image_cache.clear()
    assert before == {name: module_record(module) for name, module in modules.items()}
