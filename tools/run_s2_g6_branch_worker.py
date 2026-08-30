#!/usr/bin/env python3
"""Fresh-process G6 branch worker. Never run outside the G6 coordinator."""

from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
G5_CHECKPOINT = ROOT / "artifacts/development/stage2/g5_single_cycle_checkpoint.development"
G6_CONFIG = ROOT / "configs/stage2_g6_exact_resume.development.yaml"
G6_SOURCE_MANIFEST = ROOT / "artifacts/development/stage2/stage2_source_manifest.v8_g6.json"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def environment_audit() -> dict:
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=uuid,name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()[0]
    uuid, name, driver = [item.strip() for item in query.split(",", 2)]
    return {
        "pid": os.getpid(),
        "python": sys.version,
        "pytorch": torch.__version__,
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu_uuid": uuid,
        "gpu_name": name,
        "driver": driver,
        "runtime_import_roots": [str(ROOT / "src"), str(ROOT / "vendor/lerobot/src"), str(ROOT / "tools")],
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "actor_autocast_dtype": "torch.bfloat16",
        "critic_dtype": "torch.float32",
        "pythonhashseed": os.environ.get("PYTHONHASHSEED"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_compile": False,
        "data_augmentation": False,
        "num_workers": 0,
    }


def _sampler_states(context: dict) -> dict:
    return {name: sampler.state_dict() for name, sampler in context["samplers"].items()}


def _rng_states(context: dict) -> dict:
    from forcesmolvla.rft.training_cycle import capture_rng_states

    return capture_rng_states(context["generators"])


def _live_payload(context: dict, counters: dict) -> dict:
    from forcesmolvla.rft.canonical_state import training_state_payload
    from forcesmolvla.rft.exact_resume import boundary_state_manifest

    return training_state_payload(
        modules=context["modules"],
        actor_optimizer=context["actor_optimizer"],
        critic_optimizer=context["critic_optimizer"],
        actor_parameter_name_groups=context["parameter_map"]["actor_optimizer_parameter_name_groups"],
        critic_parameter_name_groups=context["parameter_map"]["critic_optimizer_parameter_name_groups"],
        actor_scheduler_state=context["actor_scheduler"].state_dict(),
        critic_scheduler_state=context["critic_scheduler"].state_dict(),
        sampler_states=_sampler_states(context),
        rng_states=_rng_states(context),
        counters=counters,
        boundary_manifest=boundary_state_manifest(context["modules"]),
        ownership_manifest=context["ownership"],
    )


class CycleTrace:
    """Observe the frozen G5 cycle without changing any tensor operation."""

    def __init__(self, context: dict) -> None:
        from forcesmolvla.rft.canonical_state import canonical_digest

        self.context = context
        self.records: dict[str, list] = {
            "random_tensors": [], "router": [], "gradients": [],
            "policy_actions": [], "td_targets": [], "candidate_q": [],
        }
        self.generator_names = {
            id(generator): name for name, generator in context["generators"].items()
        }
        self.parameter_names = {
            id(parameter): f"{module_name}.{name}"
            for module_name, module in context["modules"].items()
            for name, parameter in module.named_parameters()
        }
        self._canonical_digest = canonical_digest
        self._restorers = []
        self._hooks = []

    def _tensor(self, value: torch.Tensor) -> dict:
        from forcesmolvla.rft.canonical_state import tensor_record

        return tensor_record(value)

    def _patch(self, owner, name: str, replacement) -> None:
        original = getattr(owner, name)
        setattr(owner, name, replacement(original))
        self._restorers.append(lambda owner=owner, name=name, original=original: setattr(owner, name, original))

    def __enter__(self):
        import forcesmolvla.rft.losses as losses
        from forcesmolvla.rft import training_cycle as g5

        def random_wrapper(operation: str):
            def factory(original):
                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    generator = kwargs.get("generator")
                    name = self.generator_names.get(id(generator))
                    if name is not None:
                        self.records["random_tensors"].append({
                            "operation": operation,
                            "generator": name,
                            "tensor": self._tensor(result),
                        })
                    return result
                return wrapped
            return factory

        for name in ("randn", "rand", "randperm", "randint"):
            self._patch(torch, name, random_wrapper(name))

        def clip_factory(original):
            def wrapped(parameters, *args, **kwargs):
                parameters = list(parameters)
                before = {
                    self.parameter_names[id(parameter)]: self._tensor(parameter.grad)
                    for parameter in parameters if parameter.grad is not None
                }
                result = original(parameters, *args, **kwargs)
                after = {
                    self.parameter_names[id(parameter)]: self._tensor(parameter.grad)
                    for parameter in parameters if parameter.grad is not None
                }
                self.records["gradients"].append({
                    "before": self._canonical_digest(before),
                    "after": self._canonical_digest(after),
                    "parameter_count": len(parameters),
                })
                return result
            return wrapped

        self._patch(torch.nn.utils, "clip_grad_norm_", clip_factory)

        def tensor_output_factory(category: str):
            def factory(original):
                def wrapped(*args, **kwargs):
                    result = original(*args, **kwargs)
                    self.records[category].append(self._tensor(result))
                    return result
                return wrapped
            return factory

        self._patch(losses, "compute_td_target_from_current_actor", tensor_output_factory("td_targets"))
        self._patch(losses, "evaluate_calql_candidates", tensor_output_factory("candidate_q"))
        self._patch(losses, "critic_action_for_q_guidance", tensor_output_factory("policy_actions"))
        self._patch(g5, "sample_policy_candidates", tensor_output_factory("policy_actions"))

        def router_hook(_module, _inputs, output):
            if not isinstance(output, tuple) or len(output) != 2:
                return
            state = output[1]
            route_ids = state.route_ids.detach()
            valid_routes = route_ids[state.valid_mask]
            counts = torch.bincount(valid_routes, minlength=4)
            self.records["router"].append({
                "logits": self._tensor(state.logits_fp32),
                "dispatch_indices": self._tensor(route_ids),
                "route_counts": self._tensor(counts),
            })

        self._hooks.append(
            self.context["modules"]["actor"].model.force_branch.refiner.register_forward_hook(router_hook)
        )
        return self

    def __exit__(self, exc_type, exc, tb):
        for hook in self._hooks:
            hook.remove()
        for restore in reversed(self._restorers):
            restore()

    def result(self) -> dict:
        result = dict(self.records)
        result["digest"] = self._canonical_digest(result)
        return result


def initialize_context(*, restore_checkpoint: bool) -> dict:
    from forcesmolvla.modeling_forcesmolvla import ForceSmolVLAPolicy
    from forcesmolvla.rft.canonical_state import optimizer_parameter_name_groups
    from forcesmolvla.rft.critic import build_twin_q, modules_storage_independent, state_exact
    from forcesmolvla.rft.exact_resume import boundary_state_manifest, strict_restore_into
    from forcesmolvla.rft.training_cycle import (
        SerializableReplacementSampler,
        SerializableUniqueSampler,
        build_stage2_optimizers,
    )
    from forcesmolvla.rft.training_cycle import (
        PARENT_ACTOR_CHECKPOINT,
        REWARD_BACKBONE_MANIFEST,
        REWARD_BACKBONE_PARAMETERS,
        TrainData,
        named_generator,
        verify_config,
    )

    require(torch.cuda.is_available(), "G6_CUDA_REQUIRED_NO_CPU_FALLBACK")
    device = torch.device("cuda:0")
    require("4090" in torch.cuda.get_device_name(device), "G6_RTX4090D_REQUIRED")
    require(os.environ.get("PYTHONHASHSEED") == "42", "G6_PYTHONHASHSEED_REQUIRED")
    require(os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8", "G6_CUBLAS_REQUIRED")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    config = verify_config()
    g6 = yaml.safe_load(G6_CONFIG.read_text())
    require(g6["inherited_training_contract"]["hyperparameter_changes_allowed"] is False, "G6_CONFIG_MUTATION_ALLOWED")

    data = TrainData()
    data.canonicalize_proposal_gripper_for_runtime(device)
    seeds = config["rng"]["named_stream_seeds"]
    generators = {
        "td_sampler": named_generator("cpu", seeds["td_sampler"]),
        "calql_sampler": named_generator("cpu", seeds["calql_sampler"]),
        "actor_sampler": named_generator("cpu", seeds["actor_sampler"]),
        "empirical_random_proposal": named_generator("cpu", seeds["empirical_random_proposal"]),
        "td_next_action_flow_noise": named_generator("cuda", seeds["td_next_action_flow_noise"]),
        "calql_current_policy_flow_noise": named_generator("cuda", seeds["calql_current_policy_flow_noise"]),
        "calql_next_policy_flow_noise": named_generator("cuda", seeds["calql_next_policy_flow_noise"]),
        "actor_q_flow_noise": named_generator("cuda", seeds["actor_q_flow_noise"]),
        "flow_matching_noise": named_generator("cuda", seeds["flow_matching_noise"]),
        "flow_matching_timestep": named_generator("cuda", seeds["flow_matching_timestep"]),
        "moe_router_stochastic_state": named_generator("cuda", seeds["moe_router_stochastic_state"]),
    }
    samplers = {
        "td": SerializableUniqueSampler("TD_sampler", data.td_population, generators["td_sampler"]),
        "calql": SerializableUniqueSampler("CalQL_sampler", data.calql_population, generators["calql_sampler"]),
        "actor": SerializableUniqueSampler("Actor_sampler", data.actor_population, generators["actor_sampler"]),
        "empirical_random_proposal": SerializableReplacementSampler(
            "empirical_random_proposal", len(data.proposal_population),
            generators["empirical_random_proposal"],
        ),
    }
    with redirect_stdout(sys.stderr):
        actor = ForceSmolVLAPolicy.from_pretrained(
            PARENT_ACTOR_CHECKPOINT,
            local_files_only=True,
            force_download=False,
            strict=True,
            artifact_use="development",
        ).to(device)
    q1, q2, q1_target, q2_target, _conversion = build_twin_q(
        REWARD_BACKBONE_PARAMETERS, REWARD_BACKBONE_MANIFEST, seed=0
    )
    q1, q2, q1_target, q2_target = (
        module.to(device) for module in (q1, q2, q1_target, q2_target)
    )
    require(
        modules_storage_independent(q1, q2)
        and modules_storage_independent(q1, q1_target)
        and modules_storage_independent(q2, q2_target)
        and state_exact(q1, q1_target)
        and state_exact(q2, q2_target),
        "G6_TWIN_Q_INITIALIZATION_INVALID",
    )
    actor.eval()
    q1.train(True)
    q2.train(True)
    q1_target.eval()
    q2_target.eval()
    actor_optimizer, critic_optimizer, actor_scheduler, critic_scheduler, ownership = (
        build_stage2_optimizers(actor, q1, q2)
    )
    owned_ids = {
        id(parameter)
        for optimizer in (actor_optimizer, critic_optimizer)
        for group in optimizer.param_groups for parameter in group["params"]
    }
    ownership["target_parameter_ids_in_optimizer"] = sum(
        id(parameter) in owned_ids
        for target in (q1_target, q2_target) for parameter in target.parameters()
    )
    require(ownership["target_parameter_ids_in_optimizer"] == 0, "G6_TARGET_IN_OPTIMIZER")
    modules = {
        "actor": actor, "q1": q1, "q2": q2,
        "q1_target": q1_target, "q2_target": q2_target,
    }
    actor_names = dict(actor.named_parameters())
    critic_names = {
        **{f"q1.{name}": parameter for name, parameter in q1.named_parameters()},
        **{f"q2.{name}": parameter for name, parameter in q2.named_parameters()},
    }
    parameter_map = {
        "actor_optimizer_parameter_name_groups": optimizer_parameter_name_groups(
            actor_optimizer, actor_names
        ),
        "critic_optimizer_parameter_name_groups": optimizer_parameter_name_groups(
            critic_optimizer, critic_names
        ),
        "expected_s1_boundary": boundary_state_manifest(modules),
    }
    trainability = {
        "actor_all_checkpoint_trainable_parameters_owned": True,
        "actor_trainable_tensor_count": sum(parameter.requires_grad for parameter in actor.parameters()),
        "actor_trainable_parameter_count": sum(parameter.numel() for parameter in actor.parameters() if parameter.requires_grad),
        "lm_head_gradient_required": False,
        "q1_trainable_parameter_count": sum(parameter.numel() for parameter in q1.parameters() if parameter.requires_grad),
        "q2_trainable_parameter_count": sum(parameter.numel() for parameter in q2.parameters() if parameter.requires_grad),
        "target_trainable_parameter_count": 0,
        "frozen_resnet_in_critic_optimizer": False,
        "lora_used": False,
        "vlm_frozen": False,
        "camera_count": 2,
        "flow_horizon": 50,
        "flow_euler_steps": 10,
        "torch_compile": False,
    }
    context = {
        "device": device, "config": config, "data": data,
        "generators": generators, "samplers": samplers, "modules": modules,
        "actor_optimizer": actor_optimizer, "critic_optimizer": critic_optimizer,
        "actor_scheduler": actor_scheduler, "critic_scheduler": critic_scheduler,
        "ownership": ownership, "trainability": trainability,
        "parameter_map": parameter_map,
    }
    if restore_checkpoint:
        counters = strict_restore_into(
            G5_CHECKPOINT, modules=modules,
            actor_optimizer=actor_optimizer, critic_optimizer=critic_optimizer,
            actor_scheduler=actor_scheduler, critic_scheduler=critic_scheduler,
            samplers=samplers, generators=generators,
        )
        require(boundary_state_manifest(modules) == parameter_map["expected_s1_boundary"], "G6_RESTORED_BOUNDARY_MANIFEST_MISMATCH")
        context["restored_counters"] = counters
    return context


def draw_cycle_batches(context: dict) -> tuple[list[list[int]], list[list[int]], list[int], list[dict]]:
    from forcesmolvla.rft.canonical_state import canonical_digest

    audit = []
    td = []
    calql = []
    for _ in range(2):
        td.append(context["samplers"]["td"].draw(16))
        audit.append({"sampler": "td", "state_after_draw": canonical_digest(context["samplers"]["td"].state_dict())})
    for _ in range(2):
        calql.append(context["samplers"]["calql"].draw(16))
        audit.append({"sampler": "calql", "state_after_draw": canonical_digest(context["samplers"]["calql"].state_dict())})
    actor = context["samplers"]["actor"].draw(4)
    audit.append({"sampler": "actor", "state_after_draw": canonical_digest(context["samplers"]["actor"].state_dict())})
    require(len({tuple(batch) for batch in [*td, *calql, actor]}) == 5, "G6_BATCH_REUSE")
    return td, calql, actor, audit


def run_cycle(context: dict, *, cycle: int, include_g5_scale_probe: bool, trace_enabled: bool) -> tuple[dict, dict]:
    from forcesmolvla.rft.canonical_state import canonical_digest, tensor_record
    from forcesmolvla.rft.training_cycle import ensure_all_gradients_none
    from forcesmolvla.rft.training_cycle import (
        FlowCounter, actor_gradient_scale_probe, actor_update, critic_update,
    )

    modules = context["modules"]
    data = context["data"]
    device = context["device"]
    td_draws, calql_draws, actor_draw, sampler_draw_audit = draw_cycle_batches(context)
    flow_counter = FlowCounter(inference_batch_size=4)
    trace_context = CycleTrace(context) if trace_enabled else None
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    if trace_context:
        trace_context.__enter__()
    critic_reports = []
    rng_after_substep = []
    try:
        for local_step in (0, 1):
            global_step = 2 * (cycle - 1) + local_step + 1
            td_batch = data.build_batch(
                td_draws[local_step], modules["actor"], device,
                canonical_task_feature=modules["q1"].canonical_task_feature,
            )
            calql_batch = data.build_batch(
                calql_draws[local_step], modules["actor"], device,
                canonical_task_feature=modules["q1"].canonical_task_feature,
            )
            mc_return = tensor_record(calql_batch["mc_return"])
            report = critic_update(
                step=global_step, policy=modules["actor"], q1=modules["q1"], q2=modules["q2"],
                q1_target=modules["q1_target"], q2_target=modules["q2_target"],
                optimizer=context["critic_optimizer"], scheduler=context["critic_scheduler"],
                td_batch=td_batch, calql_batch=calql_batch, train_data=data,
                proposal_sampler=context["samplers"]["empirical_random_proposal"],
                generators=context["generators"], flow_counter=flow_counter,
                config=context["config"],
            )
            report["mc_return_tensor"] = mc_return
            proposal_indices = report["proposal_population_indices"]
            proposal_actions = data.proposal_actions[proposal_indices].reshape(
                16, context["config"]["loss"]["cql_candidates_per_source_M"], 3, 7
            )
            report["empirical_proposal_actions"] = tensor_record(proposal_actions)
            report["loss_fp32_bit_pattern"] = tensor_record(torch.tensor(
                [
                    report["loss"]["L_TD_Q1"],
                    report["loss"]["L_TD_Q2"],
                    report["loss"]["L_CalQL_Q1"],
                    report["loss"]["L_CalQL_Q2"],
                    report["loss"]["L_critic"],
                ],
                dtype=torch.float32,
            ))
            report.pop("latency_seconds", None)
            critic_reports.append(report)
            rng_after_substep.append({
                "substep": f"critic_{global_step}",
                "rng_digest": canonical_digest(_rng_states(context)),
                "proposal_sampler_digest": canonical_digest(
                    context["samplers"]["empirical_random_proposal"].state_dict()
                ),
            })
            del td_batch, calql_batch
            gc.collect()
            torch.cuda.empty_cache()

        actor_batch = data.build_batch(
            actor_draw, modules["actor"], device,
            canonical_task_feature=modules["q1"].canonical_task_feature,
            include_flow_actions=True,
        )
        scale_probe = None
        if include_g5_scale_probe:
            first = torch.tensor([True, False, False, False], dtype=torch.bool, device=device)
            probe_batch = {
                "reward": actor_batch["reward"][first],
                "current_observation": actor_batch["current_observation"].index(first),
                "current_actor_batch": {
                    name: (
                        value[first]
                        if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 4
                        else type(value)(item for item, keep in zip(value, first.cpu().tolist(), strict=True) if keep)
                        if isinstance(value, (tuple, list)) and len(value) == 4
                        else value
                    ) for name, value in actor_batch["current_actor_batch"].items()
                },
                "delta_mean": actor_batch["delta_mean"],
                "delta_std": actor_batch["delta_std"],
            }
            scale_probe = actor_gradient_scale_probe(
                policy=modules["actor"], q1=modules["q1"], q2=modules["q2"],
                microbatch=probe_batch, generators=context["generators"],
                flow_counter=flow_counter, eta=context["config"]["loss"]["eta_actor_q"],
            )
            del probe_batch
        actor_report = actor_update(
            policy=modules["actor"], q1=modules["q1"], q2=modules["q2"],
            q1_target=modules["q1_target"], q2_target=modules["q2_target"],
            optimizer=context["actor_optimizer"], scheduler=context["actor_scheduler"],
            actor_batch=actor_batch, generators=context["generators"],
            flow_counter=flow_counter, config=context["config"],
        )
        actor_report["loss_bit_pattern"] = tensor_record(torch.tensor(
            list(actor_report["loss"].values()), dtype=torch.float64
        ))
        actor_report.pop("latency_seconds", None)
        rng_after_substep.append({
            "substep": f"actor_{cycle}",
            "rng_digest": canonical_digest(_rng_states(context)),
        })
        del actor_batch
    finally:
        if trace_context:
            trace_context.__exit__(None, None, None)
    torch.cuda.synchronize()
    ensure_all_gradients_none(*modules.values())
    counters = {
        "training_cycles": cycle,
        "critic_optimizer_updates": 2 * cycle,
        "actor_optimizer_updates": cycle,
        "q1_target_polyak_updates": 2 * cycle,
        "q2_target_polyak_updates": 2 * cycle,
        "actor_target_updates": 0,
        "critic_scheduler_steps": 2 * cycle,
        "actor_scheduler_steps": cycle,
    }
    trace = {
        "schema_version": "forcesmolvla_g6_cycle_trace.v1",
        "cycle": cycle,
        "td_batch_rows": [data.identity_records(indices) for indices in td_draws],
        "calql_batch_rows": [data.identity_records(indices) for indices in calql_draws],
        "actor_accumulation_rows": data.identity_records(actor_draw),
        "sampler_draw_state": sampler_draw_audit,
        "critic_updates": critic_reports,
        "actor_update": actor_report,
        "g5_scale_probe_replayed": include_g5_scale_probe,
        "scale_probe": scale_probe,
        "rng_after_substep": rng_after_substep,
        "observed_tensors": trace_context.result() if trace_context else None,
        "flow_counts": flow_counter.report(),
    }
    trace["canonical_trace_digest"] = canonical_digest(trace)
    runtime = {
        "latency_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(device),
    }
    return {"counters": counters, "trace": trace}, runtime


def startup_snapshot_bytes() -> dict[str, bytes]:
    paths = {
        "g6/stage2_g6_exact_resume.development.yaml": G6_CONFIG,
        "g6/stage2_source_manifest.v8_g6.json": G6_SOURCE_MANIFEST,
        "g5/stage2_g5_single_cycle.development.yaml": ROOT / "configs/stage2_g5_single_cycle.development.yaml",
        "g5/stage2_source_manifest.v7_g5.json": ROOT / "artifacts/development/stage2/stage2_source_manifest.v7_g5.json",
        "g5/checkpoint_manifest.json": G5_CHECKPOINT / "checkpoint_manifest.json",
        "automatic_g1/g1_manifest.json": ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1/g1_manifest.json",
    }
    return {relative: path.read_bytes() for relative, path in paths.items()}


def save_branch_checkpoint(context: dict, destination: Path, state: dict, trace: dict) -> tuple[dict, dict]:
    from forcesmolvla.rft.canonical_state import assert_payload_exact
    from forcesmolvla.rft.exact_resume import (
        boundary_state_manifest, save_g6_checkpoint, validate_checkpoint_files,
    )

    before = _live_payload(context, state["counters"])
    manifest = save_g6_checkpoint(
        destination,
        modules=context["modules"], actor_optimizer=context["actor_optimizer"],
        critic_optimizer=context["critic_optimizer"], actor_scheduler=context["actor_scheduler"],
        critic_scheduler=context["critic_scheduler"], counters=state["counters"],
        sampler_states=_sampler_states(context), rng_states=_rng_states(context),
        startup_snapshot_bytes=startup_snapshot_bytes(),
        parameter_ownership_manifest=context["ownership"],
        trainability_manifest=context["trainability"],
        proposal_population_manifest=context["data"].population_manifest,
        parameter_map=context["parameter_map"],
        boundary_state=boundary_state_manifest(context["modules"]), trace=trace,
    )
    after = _live_payload(context, state["counters"])
    assert_payload_exact(before, after, "checkpoint_save_side_effect")
    files = validate_checkpoint_files(destination, expected_markers={
        "artifact_status": "DEVELOPMENT_EXACT_RESUME_TEST_ONLY",
        "deployment_status": "NOT_FOR_DEPLOYMENT",
        "policy_evaluation_status": "NOT_FOR_POLICY_EVALUATION",
        "long_train_parent_status": "NOT_AN_APPROVED_LONG_TRAIN_PARENT",
        "robot_execution_authorized": False,
    })
    return manifest, {
        "training_state_before_sha256": before["training_state_digest"],
        "training_state_after_sha256": after["training_state_digest"],
        "exact_unchanged": True,
        "tree": files["tree"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=("A", "B"), required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--parameter-map", type=Path, required=True)
    parser.add_argument("--expected-s1", type=Path, required=True)
    parser.add_argument("--cycle1-checkpoint", type=Path)
    parser.add_argument("--cycle2-checkpoint", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    args = parser.parse_args()
    require(G6_SOURCE_MANIFEST.is_file(), "G6_SOURCE_MANIFEST_MISSING")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    from forcesmolvla.rft.canonical_state import assert_payload_exact
    from forcesmolvla.rft.exact_resume import preflight_g5_checkpoint
    from forcesmolvla.rft.training_cycle import FORBIDDEN_OPENS, install_open_audit

    install_open_audit()
    preflight = preflight_g5_checkpoint(ROOT, G5_CHECKPOINT)
    context = initialize_context(restore_checkpoint=args.branch == "B")
    environment = environment_audit()
    if args.branch == "A":
        require(args.cycle1_checkpoint is not None, "G6_BRANCH_A_CYCLE1_CHECKPOINT_REQUIRED")
        atomic_json(args.parameter_map, context["parameter_map"])
        deadline = time.monotonic() + 900
        while not args.expected_s1.exists():
            require(time.monotonic() < deadline, "G6_EXPECTED_S1_HANDSHAKE_TIMEOUT")
            time.sleep(0.05)
        expected_s1 = json.loads(args.expected_s1.read_text())
        cycle1, runtime1 = run_cycle(
            context, cycle=1, include_g5_scale_probe=True, trace_enabled=False
        )
        s1 = _live_payload(context, cycle1["counters"])
        assert_payload_exact(s1, expected_s1, "branch_A_cycle1_vs_G5")
        manifest1, save_audit1 = save_branch_checkpoint(
            context, args.cycle1_checkpoint, cycle1, cycle1["trace"]
        )
        cycle2, runtime2 = run_cycle(
            context, cycle=2, include_g5_scale_probe=False, trace_enabled=True
        )
        manifest2, save_audit2 = save_branch_checkpoint(
            context, args.cycle2_checkpoint, cycle2, cycle2["trace"]
        )
        result = {
            "branch": "A", "environment": environment, "preflight": preflight,
            "loaded_g5_training_state": False,
            "cycle1_g5_exact": True,
            "cycle1_state_digest": s1["training_state_digest"],
            "cycle1_runtime": runtime1,
            "cycle1_checkpoint_manifest_payload_sha256": manifest1["manifest_payload_sha256"],
            "cycle1_checkpoint_save_side_effect": save_audit1,
            "cycle2_runtime": runtime2,
            "cycle2_state_digest": _live_payload(context, cycle2["counters"])["training_state_digest"],
            "cycle2_trace": cycle2["trace"],
            "cycle2_checkpoint_manifest_payload_sha256": manifest2["manifest_payload_sha256"],
            "cycle2_checkpoint_save_side_effect": save_audit2,
            "final_counters": cycle2["counters"],
        }
    else:
        require(args.parameter_map.is_file() and args.expected_s1.is_file(), "G6_BRANCH_B_HANDSHAKE_INPUT_MISSING")
        expected_map = json.loads(args.parameter_map.read_text())
        require(context["parameter_map"] == expected_map, "G6_BRANCH_B_PARAMETER_MAP_MISMATCH")
        expected_s1 = json.loads(args.expected_s1.read_text())
        restored = _live_payload(context, context["restored_counters"])
        assert_payload_exact(restored, expected_s1, "branch_B_restored_S1")
        cycle2, runtime2 = run_cycle(
            context, cycle=2, include_g5_scale_probe=False, trace_enabled=True
        )
        manifest2, save_audit2 = save_branch_checkpoint(
            context, args.cycle2_checkpoint, cycle2, cycle2["trace"]
        )
        result = {
            "branch": "B", "environment": environment, "preflight": preflight,
            "loaded_g5_training_state_strict": True,
            "rng_restored_last": True,
            "random_sanity_forward_after_rng_restore": 0,
            "sampler_draws_before_cycle2": 0,
            "restored_s1_exact": True,
            "restored_s1_state_digest": restored["training_state_digest"],
            "cycle2_runtime": runtime2,
            "cycle2_state_digest": _live_payload(context, cycle2["counters"])["training_state_digest"],
            "cycle2_trace": cycle2["trace"],
            "cycle2_checkpoint_manifest_payload_sha256": manifest2["manifest_payload_sha256"],
            "cycle2_checkpoint_save_side_effect": save_audit2,
            "final_counters": cycle2["counters"],
        }
    require(not FORBIDDEN_OPENS["manual_g1"] and not FORBIDDEN_OPENS["manual_labels"], "G6_FORBIDDEN_MANUAL_READ")
    result["data_access_audit"] = {
        **context["data"].population_audit(),
        "manual_g1_files_opened": 0, "manual_label_files_opened": 0,
        "reward_classifier_inference_calls": 0,
        "reward_classifier_optimizer_updates": 0,
    }
    result["worker_status"] = "pass"
    atomic_json(args.result, result)


if __name__ == "__main__":
    main()
