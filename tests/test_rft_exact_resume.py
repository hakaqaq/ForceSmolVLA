from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from forcesmolvla.rft.canonical_state import (
    assert_payload_exact,
    canonical_digest,
    module_record,
    optimizer_parameter_name_groups,
    optimizer_record,
    tensor_record,
)
from forcesmolvla.rft.exact_resume import (
    G6_CHECKPOINT_MARKERS,
    boundary_state_manifest,
    save_g6_checkpoint,
    validate_checkpoint_files,
)


def test_tensor_record_is_original_dtype_shape_and_bytes() -> None:
    value = torch.tensor([[1.0, -0.0]], dtype=torch.float32)
    same_numbers_different_dtype = value.double()
    changed_bit = value.clone()
    changed_bit.view(torch.int32)[0, 0] += 1

    record = tensor_record(value)
    assert record["dtype"] == "torch.float32"
    assert record["shape"] == [1, 2]
    assert record["byte_count"] == 8
    assert record["sha256"] != tensor_record(same_numbers_different_dtype)["sha256"]
    assert record["sha256"] != tensor_record(changed_bit)["sha256"]


def test_optimizer_record_uses_stable_parameter_names_not_object_ids() -> None:
    torch.manual_seed(7)
    left = torch.nn.Linear(3, 2)
    right = copy.deepcopy(left)
    left_optimizer = torch.optim.Adam(left.parameters(), lr=3e-4)
    right_optimizer = torch.optim.Adam(right.parameters(), lr=3e-4)
    sample = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    for module, optimizer in ((left, left_optimizer), (right, right_optimizer)):
        module(sample).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    left_names = optimizer_parameter_name_groups(
        left_optimizer, dict(left.named_parameters())
    )
    right_names = optimizer_parameter_name_groups(
        right_optimizer, dict(right.named_parameters())
    )
    assert left_names == right_names
    assert optimizer_record(left_optimizer, left_names) == optimizer_record(
        right_optimizer, right_names
    )


def test_exact_comparator_rejects_one_tensor_bit() -> None:
    first = {"training_state_digest": "a", "section_digests": {"modules": "x"}, "sections": {}}
    second = {"training_state_digest": "b", "section_digests": {"modules": "y"}, "sections": {}}
    with pytest.raises(RuntimeError, match="G6_EXACT_PARITY_FAILED"):
        assert_payload_exact(first, second, "fixture")


def test_boundary_manifest_requires_no_pending_gradient() -> None:
    module = torch.nn.Linear(2, 1)
    module(torch.ones(1, 2)).sum().backward()
    with pytest.raises(RuntimeError, match="G6_BOUNDARY_GRADIENT_NOT_NONE"):
        boundary_state_manifest({"actor": module})
    module.zero_grad(set_to_none=True)
    boundary = boundary_state_manifest({"actor": module})
    assert boundary["all_gradients_none"] is True
    assert boundary["pending_accumulation"] == 0


def test_g6_atomic_checkpoint_is_side_effect_free_and_integrity_bound(tmp_path: Path) -> None:
    torch.manual_seed(11)
    modules = {
        name: torch.nn.Linear(2, 2)
        for name in ("actor", "q1", "q2", "q1_target", "q2_target")
    }
    actor_optimizer = torch.optim.AdamW(modules["actor"].parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam(
        [*modules["q1"].parameters(), *modules["q2"].parameters()], lr=3e-4
    )
    actor_scheduler = torch.optim.lr_scheduler.LambdaLR(actor_optimizer, lambda _: 1.0)
    critic_scheduler = torch.optim.lr_scheduler.LambdaLR(critic_optimizer, lambda _: 1.0)
    counters = {
        "training_cycles": 1,
        "critic_optimizer_updates": 2,
        "actor_optimizer_updates": 1,
        "q1_target_polyak_updates": 2,
        "q2_target_polyak_updates": 2,
        "actor_target_updates": 0,
        "critic_scheduler_steps": 2,
        "actor_scheduler_steps": 1,
    }
    before_modules = {name: module_record(module) for name, module in modules.items()}
    before_rng = torch.get_rng_state().clone()
    destination = tmp_path / "checkpoint"
    save_g6_checkpoint(
        destination,
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        actor_scheduler=actor_scheduler,
        critic_scheduler=critic_scheduler,
        counters=counters,
        sampler_states={"fixture": {"draws": 0}},
        rng_states={"torch": before_rng},
        startup_snapshot_bytes={"fixture/config.txt": b"frozen\n"},
        parameter_ownership_manifest={"disjoint": True},
        trainability_manifest={"fixture": True},
        proposal_population_manifest={"count": 1},
        parameter_map={"actor": ["weight", "bias"]},
        boundary_state=boundary_state_manifest(modules),
        trace={"digest": canonical_digest("trace")},
    )
    validated = validate_checkpoint_files(
        destination, expected_markers=G6_CHECKPOINT_MARKERS
    )
    assert validated["tree"]["file_count"] > 1
    assert torch.equal(before_rng, torch.get_rng_state())
    assert before_modules == {
        name: module_record(module) for name, module in modules.items()
    }

    target = destination / "manifests/trainability.json"
    with target.open("r+b") as stream:
        byte = stream.read(1)
        stream.seek(0)
        stream.write(bytes([byte[0] ^ 1]))
    with pytest.raises(RuntimeError, match="G6_CHECKPOINT_INTERNAL_FILE_SHA_MISMATCH"):
        validate_checkpoint_files(destination, expected_markers=G6_CHECKPOINT_MARKERS)
