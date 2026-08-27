"""Strict G6 checkpoint validation, canonical loading, and atomic test saves."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from forcesmolvla.rft.canonical_state import training_state_payload
from forcesmolvla.rft.training_cycle import ensure_all_gradients_none


G6_CHECKPOINT_MARKERS = {
    "artifact_status": "DEVELOPMENT_EXACT_RESUME_TEST_ONLY",
    "deployment_status": "NOT_FOR_DEPLOYMENT",
    "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
    "long_train_parent_status": "NOT_AN_APPROVED_LONG_TRAIN_PARENT",
    "robot_execution_authorized": False,
}
G5_MARKERS = {
    "artifact_status": "DEVELOPMENT_SINGLE_CYCLE_ONLY",
    "deployment_status": "NOT_FOR_DEPLOYMENT",
    "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
    "long_train_parent_status": "NOT_AN_APPROVED_LONG_TRAIN_PARENT",
    "robot_execution_authorized": False,
    "resume_exactness_tested": False,
}
G5_CONFIG_SHA256 = "877499afc58a2af546be8dee7ce1144b8ea38b5fe486a450e07da989ea4e5ed7"
G5_SOURCE_MANIFEST_SHA256 = "9da745a4295a431c5fecb6e54f8d0d56aab18daaff6480804810d742aea4bf10"
G5_CHECKPOINT_MANIFEST_SHA256 = "90644bf82dbb100bd7880944f142d7face75024b9080c357d2923bce2712cf02"
G5_CHECKPOINT_TREE_SHA256 = "a65de98f270898ef40f32e0398492bbead918df67255575b4e558347a52bf715"
REQUIRED_RNG_KEYS = {
    "python_random_state",
    "numpy_random_state",
    "torch_cpu_rng_state",
    "torch_cuda_rng_states",
    "named_generator_states",
}
REQUIRED_SAMPLERS = {"td", "calql", "actor", "empirical_random_proposal"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def directory_entries(root: Path) -> list[dict]:
    root = Path(root)
    return [
        {
            "relative_path": path.relative_to(root).as_posix(),
            "sha256": sha256_file(path),
            "file_size": path.stat().st_size,
        }
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "checkpoint_manifest.json"
    ]


def checkpoint_tree(root: Path) -> dict:
    root = Path(root)
    files = sorted(path for path in root.rglob("*") if path.is_file())
    digest = hashlib.sha256()
    size = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        value = sha256_file(path)
        digest.update(f"{relative}\0{value}\n".encode())
        size += path.stat().st_size
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_file_size": size,
    }


def _manifest_payload_sha256(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("manifest_payload_sha256", None)
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def validate_checkpoint_files(
    checkpoint: Path,
    *,
    expected_markers: Mapping[str, Any],
    expected_manifest_sha256: str | None = None,
    expected_tree_sha256: str | None = None,
) -> dict:
    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    if expected_manifest_sha256 and sha256_file(manifest_path) != expected_manifest_sha256:
        raise RuntimeError("G6_CHECKPOINT_MANIFEST_SHA_MISMATCH")
    manifest = json.loads(manifest_path.read_text())
    if any(manifest.get(key) != value for key, value in expected_markers.items()):
        raise RuntimeError("G6_CHECKPOINT_MARKER_MISMATCH")
    if manifest.get("manifest_payload_sha256") != _manifest_payload_sha256(manifest):
        raise RuntimeError("G6_CHECKPOINT_MANIFEST_PAYLOAD_SHA_MISMATCH")
    entries = directory_entries(checkpoint)
    if entries != manifest.get("files"):
        raise RuntimeError("G6_CHECKPOINT_INTERNAL_FILE_SHA_MISMATCH")
    files_digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if files_digest != manifest.get("files_sha256"):
        raise RuntimeError("G6_CHECKPOINT_FILES_DIGEST_MISMATCH")
    tree = checkpoint_tree(checkpoint)
    if expected_tree_sha256 and tree["tree_sha256"] != expected_tree_sha256:
        raise RuntimeError("G6_CHECKPOINT_TREE_SHA_MISMATCH")
    return {"manifest": manifest, "tree": tree}


def resign_checkpoint_copy(checkpoint: Path) -> dict:
    """Rebuild integrity metadata for an isolated negative-test copy only."""

    checkpoint = Path(checkpoint)
    manifest_path = checkpoint / "checkpoint_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = directory_entries(checkpoint)
    manifest["files"] = entries
    manifest["files_sha256"] = hashlib.sha256(
        json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    manifest["manifest_payload_sha256"] = _manifest_payload_sha256(manifest)
    temporary = manifest_path.with_name(".checkpoint_manifest.negative-test.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    os.replace(temporary, manifest_path)
    return manifest


def validate_boundary_payload(checkpoint: Path, *, expected_cycles: int) -> dict:
    checkpoint = Path(checkpoint)
    counters = json.loads((checkpoint / "state/counters.json").read_text())
    expected = {
        "training_cycles": expected_cycles,
        "critic_optimizer_updates": 2 * expected_cycles,
        "actor_optimizer_updates": expected_cycles,
        "q1_target_polyak_updates": 2 * expected_cycles,
        "q2_target_polyak_updates": 2 * expected_cycles,
        "actor_target_updates": 0,
        "critic_scheduler_steps": 2 * expected_cycles,
        "actor_scheduler_steps": expected_cycles,
    }
    if counters != expected:
        raise RuntimeError("G6_CHECKPOINT_COUNTER_BOUNDARY_MISMATCH")
    manifest = json.loads((checkpoint / "checkpoint_manifest.json").read_text())
    if not manifest.get("cycle_boundary_complete"):
        raise RuntimeError("G6_CHECKPOINT_NOT_COMPLETE_CYCLE_BOUNDARY")
    if (
        manifest.get("pending_graphs") != 0
        or manifest.get("pending_accumulation_microbatches") != 0
        or manifest.get("pending_optimizer_steps") != 0
        or not manifest.get("all_gradients_none")
    ):
        raise RuntimeError("G6_CHECKPOINT_PENDING_WORK")
    actor_scheduler = torch.load(
        checkpoint / "schedulers/actor_scheduler_state.pt",
        map_location="cpu", weights_only=False,
    )
    critic_scheduler = torch.load(
        checkpoint / "schedulers/critic_scheduler_state.pt",
        map_location="cpu", weights_only=False,
    )
    if (
        actor_scheduler.get("last_epoch") != expected_cycles
        or actor_scheduler.get("_step_count") != expected_cycles + 1
        or critic_scheduler.get("last_epoch") != 2 * expected_cycles
        or critic_scheduler.get("_step_count") != 2 * expected_cycles + 1
    ):
        raise RuntimeError("G6_SCHEDULER_COUNTER_MISMATCH")
    sampler_states = torch.load(
        checkpoint / "state/sampler_states.pt", map_location="cpu", weights_only=False
    )
    rng_states = torch.load(
        checkpoint / "state/rng_states.pt", map_location="cpu", weights_only=False
    )
    if set(sampler_states) != REQUIRED_SAMPLERS:
        raise RuntimeError("G6_SAMPLER_STATE_MISSING_OR_EXTRA")
    if set(rng_states) != REQUIRED_RNG_KEYS:
        raise RuntimeError("G6_RNG_STATE_MISSING_OR_EXTRA")
    named = rng_states["named_generator_states"]
    required_named = {
        "td_sampler", "calql_sampler", "actor_sampler", "empirical_random_proposal",
        "td_next_action_flow_noise", "calql_current_policy_flow_noise",
        "calql_next_policy_flow_noise", "actor_q_flow_noise",
        "flow_matching_noise", "flow_matching_timestep", "moe_router_stochastic_state",
    }
    if set(named) != required_named:
        raise RuntimeError("G6_NAMED_GENERATOR_STATE_MISSING_OR_EXTRA")
    sampler_to_generator = {
        "td": "td_sampler", "calql": "calql_sampler", "actor": "actor_sampler",
        "empirical_random_proposal": "empirical_random_proposal",
    }
    for sampler_name, generator_name in sampler_to_generator.items():
        if not torch.equal(
            sampler_states[sampler_name]["generator_state"], named[generator_name]
        ):
            raise RuntimeError("G6_SAMPLER_NAMED_GENERATOR_STATE_MISMATCH")
    return {
        "counters": counters,
        "sampler_states": sampler_states,
        "rng_states": rng_states,
        "actor_scheduler_state": actor_scheduler,
        "critic_scheduler_state": critic_scheduler,
    }


def validate_optimizer_steps(checkpoint: Path, *, expected_cycles: int) -> None:
    checkpoint = Path(checkpoint)
    expected = {"actor": expected_cycles, "critic": 2 * expected_cycles}
    for name in ("actor", "critic"):
        state = torch.load(
            checkpoint / f"optimizers/{name}_optimizer_state.pt",
            map_location="cpu", weights_only=False,
        )["state"]
        steps = {
            int(value["step"].item())
            for value in state.values()
            if "step" in value
        }
        if steps != {expected[name]}:
            raise RuntimeError(f"G6_{name.upper()}_OPTIMIZER_STEP_MISMATCH:{steps}")
        del state


def validate_g5_bindings(root: Path, checkpoint: Path) -> dict:
    root, checkpoint = Path(root), Path(checkpoint)
    config = root / "configs/stage2_g5_single_cycle.development.yaml"
    source = root / "artifacts/development/stage2/stage2_source_manifest.v7_g5.json"
    if sha256_file(config) != G5_CONFIG_SHA256:
        raise RuntimeError("G6_G5_CONFIG_SHA_MISMATCH")
    if sha256_file(source) != G5_SOURCE_MANIFEST_SHA256:
        raise RuntimeError("G6_G5_SOURCE_MANIFEST_SHA_MISMATCH")
    if (checkpoint / "startup_snapshot/resolved_config/stage2_g5_single_cycle.development.yaml").read_bytes() != config.read_bytes():
        raise RuntimeError("G6_G5_CHECKPOINT_CONFIG_SNAPSHOT_MISMATCH")
    if (checkpoint / "startup_snapshot/source/stage2_source_manifest.v7_g5.json").read_bytes() != source.read_bytes():
        raise RuntimeError("G6_G5_CHECKPOINT_SOURCE_SNAPSHOT_MISMATCH")
    frozen = json.loads(
        (checkpoint / "startup_snapshot/bindings/frozen_inputs_startup.json").read_text()
    )
    for item in frozen["files"].values():
        path = root / item["path"]
        if sha256_file(path) != item["sha256"] or path.stat().st_size != item["file_size"]:
            raise RuntimeError(f"G6_FROZEN_FILE_BINDING_MISMATCH:{item['path']}")
    return {
        "g5_config_sha256": G5_CONFIG_SHA256,
        "g5_source_manifest_sha256": G5_SOURCE_MANIFEST_SHA256,
        "frozen_file_binding_count": len(frozen["files"]),
        "frozen_bindings_sha256": sha256_file(
            checkpoint / "startup_snapshot/bindings/frozen_inputs_startup.json"
        ),
    }


def preflight_g5_checkpoint(root: Path, checkpoint: Path) -> dict:
    files = validate_checkpoint_files(
        checkpoint,
        expected_markers=G5_MARKERS,
        expected_manifest_sha256=G5_CHECKPOINT_MANIFEST_SHA256,
        expected_tree_sha256=G5_CHECKPOINT_TREE_SHA256,
    )
    boundary = validate_boundary_payload(checkpoint, expected_cycles=1)
    validate_optimizer_steps(checkpoint, expected_cycles=1)
    bindings = validate_g5_bindings(root, checkpoint)
    return {"files": files, "boundary": boundary["counters"], "bindings": bindings}


def checkpoint_training_payload(
    checkpoint: Path,
    parameter_map: Mapping[str, Any],
    *,
    g5: bool,
) -> dict:
    checkpoint = Path(checkpoint)
    modules = {}
    for name in ("actor", "q1", "q2", "q1_target", "q2_target"):
        modules[name] = torch.load(
            checkpoint / f"models/{name}_state.pt", map_location="cpu", weights_only=False
        )
    actor_optimizer = torch.load(
        checkpoint / "optimizers/actor_optimizer_state.pt",
        map_location="cpu", weights_only=False,
    )
    critic_optimizer = torch.load(
        checkpoint / "optimizers/critic_optimizer_state.pt",
        map_location="cpu", weights_only=False,
    )
    actor_scheduler = torch.load(
        checkpoint / "schedulers/actor_scheduler_state.pt",
        map_location="cpu", weights_only=False,
    )
    critic_scheduler = torch.load(
        checkpoint / "schedulers/critic_scheduler_state.pt",
        map_location="cpu", weights_only=False,
    )
    sampler_states = torch.load(
        checkpoint / "state/sampler_states.pt", map_location="cpu", weights_only=False
    )
    rng_states = torch.load(
        checkpoint / "state/rng_states.pt", map_location="cpu", weights_only=False
    )
    counters = json.loads((checkpoint / "state/counters.json").read_text())
    ownership = json.loads((checkpoint / "manifests/parameter_ownership.json").read_text())
    boundary = (
        parameter_map["expected_s1_boundary"]
        if g5
        else json.loads((checkpoint / "manifests/boundary_state.json").read_text())
    )
    result = training_state_payload(
        modules=modules,
        actor_optimizer=actor_optimizer,
        critic_optimizer=critic_optimizer,
        actor_parameter_name_groups=parameter_map["actor_optimizer_parameter_name_groups"],
        critic_parameter_name_groups=parameter_map["critic_optimizer_parameter_name_groups"],
        actor_scheduler_state=actor_scheduler,
        critic_scheduler_state=critic_scheduler,
        sampler_states=sampler_states,
        rng_states=rng_states,
        counters=counters,
        boundary_manifest=boundary,
        ownership_manifest=ownership,
    )
    return result


def restore_rng_states_last(
    rng_states: Mapping[str, Any], generators: Mapping[str, torch.Generator]
) -> None:
    if set(rng_states) != REQUIRED_RNG_KEYS:
        raise RuntimeError("G6_RNG_RESTORE_SCHEMA_INVALID")
    if set(rng_states["named_generator_states"]) != set(generators):
        raise RuntimeError("G6_NAMED_GENERATOR_RESTORE_SCHEMA_INVALID")
    for name, generator in generators.items():
        generator.set_state(rng_states["named_generator_states"][name])
    random.setstate(rng_states["python_random_state"])
    np.random.set_state(rng_states["numpy_random_state"])
    torch.set_rng_state(rng_states["torch_cpu_rng_state"])
    torch.cuda.set_rng_state_all(rng_states["torch_cuda_rng_states"])


def strict_restore_into(
    checkpoint: Path,
    *,
    modules: Mapping[str, torch.nn.Module],
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    actor_scheduler,
    critic_scheduler,
    samplers: Mapping[str, Any],
    generators: Mapping[str, torch.Generator],
) -> dict:
    checkpoint = Path(checkpoint)
    boundary = validate_boundary_payload(checkpoint, expected_cycles=1)
    validate_optimizer_steps(checkpoint, expected_cycles=1)
    for name, module in modules.items():
        state = torch.load(
            checkpoint / f"models/{name}_state.pt", map_location="cpu", weights_only=False
        )
        incompatible = module.load_state_dict(state, strict=True)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(f"G6_STRICT_MODEL_LOAD_FAILED:{name}")
        del state
    actor_state = torch.load(
        checkpoint / "optimizers/actor_optimizer_state.pt",
        map_location="cpu", weights_only=False,
    )
    critic_state = torch.load(
        checkpoint / "optimizers/critic_optimizer_state.pt",
        map_location="cpu", weights_only=False,
    )
    actor_optimizer.load_state_dict(actor_state)
    critic_optimizer.load_state_dict(critic_state)
    del actor_state, critic_state
    actor_scheduler.load_state_dict(boundary["actor_scheduler_state"])
    critic_scheduler.load_state_dict(boundary["critic_scheduler_state"])

    modules["actor"].eval()
    modules["q1"].train(True)
    modules["q2"].train(True)
    modules["q1_target"].make_permanent_eval_target()
    modules["q2_target"].make_permanent_eval_target()
    ensure_all_gradients_none(*modules.values())

    states = boundary["sampler_states"]
    for name, sampler in samplers.items():
        state = states[name]
        if name == "empirical_random_proposal":
            if sampler.name != state["name"] or sampler.population_size != state["population_size"]:
                raise RuntimeError("G6_PROPOSAL_SAMPLER_TOPOLOGY_MISMATCH")
        else:
            if sampler.name != state["name"] or sampler.population != tuple(state["population"]):
                raise RuntimeError(f"G6_{name.upper()}_SAMPLER_TOPOLOGY_MISMATCH")
        sampler.draws = int(state["draws"])

    # This is deliberately the final operation: no random sanity forward follows.
    restore_rng_states_last(boundary["rng_states"], generators)
    return boundary["counters"]


def _write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(value)
        stream.flush()
        os.fsync(stream.fileno())


def _write_json(path: Path, value: Any) -> None:
    _write_bytes(
        path, (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    )


def _torch_save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)
    with path.open("rb") as stream:
        os.fsync(stream.fileno())


def save_g6_checkpoint(
    destination: Path,
    *,
    modules: Mapping[str, torch.nn.Module],
    actor_optimizer,
    critic_optimizer,
    actor_scheduler,
    critic_scheduler,
    counters: Mapping[str, int],
    sampler_states: Mapping[str, Any],
    rng_states: Mapping[str, Any],
    startup_snapshot_bytes: Mapping[str, bytes],
    parameter_ownership_manifest: Mapping[str, Any],
    trainability_manifest: Mapping[str, Any],
    proposal_population_manifest: Mapping[str, Any],
    parameter_map: Mapping[str, Any],
    boundary_state: Mapping[str, Any],
    trace: Mapping[str, Any],
) -> dict:
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError(f"G6_CHECKPOINT_EXISTS:{destination}")
    cycles = int(counters["training_cycles"])
    expected = {
        "training_cycles": cycles,
        "critic_optimizer_updates": 2 * cycles,
        "actor_optimizer_updates": cycles,
        "q1_target_polyak_updates": 2 * cycles,
        "q2_target_polyak_updates": 2 * cycles,
        "actor_target_updates": 0,
        "critic_scheduler_steps": 2 * cycles,
        "actor_scheduler_steps": cycles,
    }
    if dict(counters) != expected or cycles not in (1, 2):
        raise RuntimeError("G6_CHECKPOINT_COUNTER_BOUNDARY_INVALID")
    ensure_all_gradients_none(*modules.values())
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=destination.parent))
    try:
        for relative, value in (
            ("models/actor_state.pt", modules["actor"].state_dict()),
            ("models/q1_state.pt", modules["q1"].state_dict()),
            ("models/q2_state.pt", modules["q2"].state_dict()),
            ("models/q1_target_state.pt", modules["q1_target"].state_dict()),
            ("models/q2_target_state.pt", modules["q2_target"].state_dict()),
            ("optimizers/actor_optimizer_state.pt", actor_optimizer.state_dict()),
            ("optimizers/critic_optimizer_state.pt", critic_optimizer.state_dict()),
            ("schedulers/actor_scheduler_state.pt", actor_scheduler.state_dict()),
            ("schedulers/critic_scheduler_state.pt", critic_scheduler.state_dict()),
            ("state/sampler_states.pt", dict(sampler_states)),
            ("state/rng_states.pt", dict(rng_states)),
        ):
            _torch_save(temporary / relative, value)
        for relative, value in (
            ("state/counters.json", dict(counters)),
            ("manifests/parameter_ownership.json", dict(parameter_ownership_manifest)),
            ("manifests/trainability.json", dict(trainability_manifest)),
            ("manifests/proposal_population.json", dict(proposal_population_manifest)),
            ("manifests/parameter_map.json", dict(parameter_map)),
            ("manifests/boundary_state.json", dict(boundary_state)),
            ("trace/cycle_trace.json", dict(trace)),
        ):
            _write_json(temporary / relative, value)
        for relative, value in sorted(startup_snapshot_bytes.items()):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError("G6_STARTUP_SNAPSHOT_PATH_INVALID")
            _write_bytes(temporary / "startup_snapshot" / path, value)
        entries = directory_entries(temporary)
        manifest = {
            "schema_version": "forcesmolvla_g6_exact_resume_test_checkpoint.v1",
            **G6_CHECKPOINT_MARKERS,
            "cycle_boundary_complete": True,
            "pending_graphs": 0,
            "pending_accumulation_microbatches": 0,
            "pending_optimizer_steps": 0,
            "all_gradients_none": True,
            "approved_as_g7_parent": False,
            "counters": dict(counters),
            "files": entries,
            "files_sha256": hashlib.sha256(
                json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        manifest["manifest_payload_sha256"] = _manifest_payload_sha256(manifest)
        _write_json(temporary / "checkpoint_manifest.json", manifest)
        directory_fd = os.open(temporary, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.replace(temporary, destination)
        parent_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def boundary_state_manifest(modules: Mapping[str, torch.nn.Module]) -> dict:
    modes = {name: bool(module.training) for name, module in sorted(modules.items())}
    requires_grad = {
        f"{module_name}.{parameter_name}": bool(parameter.requires_grad)
        for module_name, module in sorted(modules.items())
        for parameter_name, parameter in module.named_parameters()
    }
    if any(
        parameter.grad is not None
        for module in modules.values()
        for parameter in module.parameters()
    ):
        raise RuntimeError("G6_BOUNDARY_GRADIENT_NOT_NONE")
    return {
        "module_training_modes": modes,
        "requires_grad": requires_grad,
        "all_gradients_none": True,
        "pending_accumulation": 0,
        "pending_optimizer_step": False,
        "pending_scheduler_step": False,
        "pending_polyak_update": False,
    }
