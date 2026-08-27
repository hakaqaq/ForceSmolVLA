#!/usr/bin/env python3
"""Build append-only Stage-2 v4 R0-preparation source closure."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PARENT_RELATIVE = "artifacts/development/stage2/stage2_source_manifest.v4.json"
PARENT_SHA = "defa5b1d1a975c465154ac62e009863163947065127c557b5600025ce77b29eb"
SPEC_RELATIVE = "docs/ForceRFT_Stage2_Offline_TwinQ_Implementation_Spec_v4.md"
SPEC_SHA = "0d0ad0312e9758ede7b6910b232096dcaeed338d3a7d4b5aa96347d988ecdce4"
CONRFT = Path("/home/rlc123/conrft")
CONRFT_HEAD = "a779fde7fa5db5a469960a8490c100f35b41b49e"

DELTA_FILES = (
    ("configs/stage2_conrft_reward_environment.r0prep.txt", "environment_lock_input", True),
    ("configs/stage2_reward_classifier_input.development.yaml", "reward_classifier_input_contract", True),
    ("configs/stage2_reward_detector.development.yaml", "reward_detector_contract", True),
    ("schemas/stage2_reward_classifier_example.schema.json", "label_schema", False),
    ("schemas/stage2_reward_classifier_review_template.json", "human_review_template", False),
    ("src/forcesmolvla/rft/reward_detector.py", "synthetic_detector_source", True),
    ("tests/test_s2_r0_reward_adapter.py", "r0prep_test_source", False),
    ("tools/build_stage2_r0prep_source_manifest.py", "manifest_builder", True),
    ("tools/preflight_s2_r0_preparation.py", "r0prep_preflight", True),
    ("tools/reward_classifier/conrft_lerobot_v3_adapter.py", "lerobot_v3_adapter", True),
    ("tools/reward_classifier/prepare_conrft_resnet10.py", "pretrained_asset_preparer", False),
)

CONRFT_FILES = (
    ("LICENSE", "conrft_license", False),
    ("README.md", "environment_instruction_source", False),
    ("examples/experiments/resnet10_params.pkl", "pretrained_resnet10", True),
    ("examples/train_reward_classifier.py", "reward_classifier_training_reference", False),
    ("serl_launcher/requirements.txt", "environment_requirement_source", False),
    ("serl_launcher/serl_launcher/__init__.py", "conrft_runtime_package", True),
    ("serl_launcher/serl_launcher/common/encoding.py", "encoding_wrapper_source", True),
    ("serl_launcher/serl_launcher/networks/classifier.py", "legacy_classifier_reference", False),
    ("serl_launcher/serl_launcher/networks/reward_classifier.py", "reward_classifier_source", True),
    ("serl_launcher/serl_launcher/vision/__init__.py", "conrft_runtime_package", True),
    ("serl_launcher/serl_launcher/vision/data_augmentations.py", "augmentation_and_resize_source", True),
    ("serl_launcher/serl_launcher/vision/film_conditioning_layer.py", "resnet_import_dependency", True),
    ("serl_launcher/serl_launcher/vision/resnet_v1.py", "resnet_encoder_source", True),
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _record(root: Path, relative: str, role: str, runtime_imported: bool) -> dict[str, Any]:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "relative_path": relative,
        "sha256": _sha256(path),
        "file_size": path.stat().st_size,
        "artifact_role": role,
        "runtime_imported": runtime_imported,
    }


def _run(command: list[str]) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("LD_LIBRARY_PATH", None)
    return subprocess.run(
        command, check=True, text=True, capture_output=True, env=environment
    ).stdout.strip()


def _environment(environment: str) -> dict[str, Any]:
    snippet = (
        "import json,sys,jax,jaxlib,flax,optax,numpy;"
        "print(json.dumps({'python':sys.version.split()[0],'jax':jax.__version__,"
        "'jaxlib':jaxlib.__version__,'flax':flax.__version__,'optax':optax.__version__,"
        "'numpy':numpy.__version__,'roots':{'jax':jax.__file__,'jaxlib':jaxlib.__file__,"
        "'flax':flax.__file__,'optax':optax.__file__,'numpy':numpy.__file__}}))"
    )
    versions = json.loads(_run(["conda", "run", "--name", environment, "python", "-c", snippet]))
    pip_freeze = _run(["conda", "run", "--name", environment, "python", "-m", "pip", "freeze"]).splitlines()
    conda_explicit = _run(["conda", "list", "--name", environment, "--explicit"]).splitlines()
    pip_check = _run(["conda", "run", "--name", environment, "python", "-m", "pip", "check"])
    prefix = Path(versions["roots"]["jax"]).parents[4]
    nvidia_root = prefix / "lib/python3.10/site-packages/nvidia"
    runtime_library_path = sorted(
        str(path) for path in nvidia_root.glob("*/lib") if path.is_dir()
    )
    lock = {
        "environment_name": environment,
        "versions": versions,
        "pip_freeze": pip_freeze,
        "conda_explicit": conda_explicit,
        "pip_check": pip_check,
        "cudnn": next(
            line.split("==", 1)[1]
            for line in pip_freeze
            if line.lower().startswith("nvidia-cudnn-cu11==")
        ),
        "runtime_library_path": runtime_library_path,
        "inherited_pythonpath_and_ld_library_path": "forbidden",
        "input_lock": _record(
            ROOT,
            "configs/stage2_conrft_reward_environment.r0prep.txt",
            "environment_lock_input",
            True,
        ),
    }
    lock["environment_lock_sha256"] = _canonical_sha(lock)
    return lock


def _git(*arguments: str) -> str:
    return _run(["git", "-C", str(CONRFT), *arguments])


def _qualification(parent: dict[str, Any]) -> list[dict[str, Any]]:
    records = list(parent["qualification_files"])
    for name in (
        "p9_v4_2_r8_gpu_preflight.json",
        "p9_v4_2_r8_records.json",
        "p9_v4_2_r8_replay.json",
        "p9_v4_2_r8_resolved_config.json",
        "p9_v4_2_r8_source_binding.json",
    ):
        records.append(
            _record(ROOT, f"artifacts/development/{name}", "stage1_p9_frozen_artifact", False)
        )
    return records


def build(environment: str) -> dict[str, Any]:
    parent_path = ROOT / PARENT_RELATIVE
    if _sha256(parent_path) != PARENT_SHA:
        raise RuntimeError("parent v4 source manifest drift")
    parent = json.loads(parent_path.read_text())
    spec = ROOT / SPEC_RELATIVE
    if _sha256(spec) != SPEC_SHA:
        raise RuntimeError("active v4 specification drift")

    head = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain=v1", "--untracked-files=all")
    if head != CONRFT_HEAD or status:
        raise RuntimeError(f"ConRFT authority mismatch: head={head}, status={status!r}")

    delta = [_record(ROOT, *entry) for entry in DELTA_FILES]
    conrft_files = [_record(CONRFT, *entry) for entry in CONRFT_FILES]
    effective_files = sorted(
        [*parent["files"], *delta], key=lambda entry: entry["relative_path"]
    )
    if len({entry["relative_path"] for entry in effective_files}) != len(effective_files):
        raise RuntimeError("duplicate path in effective Stage-2 source closure")
    qualification = _qualification(parent)
    environment_lock = _environment(environment)
    prepared_pretrained = _record(
        ROOT,
        "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.pkl",
        "runtime_pretrained_resnet10",
        True,
    )
    prepared_pretrained_provenance = _record(
        ROOT,
        "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.provenance.json",
        "runtime_pretrained_resnet10_provenance",
        True,
    )
    payload = {
        "schema_version": "1.0.0",
        "artifact_status": "append_only_source_closure",
        "manifest_generation": "v4_r0prep_1",
        "active_gate": "R0_PREPARATION_ONLY",
        "parent_manifest_path": PARENT_RELATIVE,
        "parent_manifest_sha256": PARENT_SHA,
        "supersedes_for_future_gates": True,
        "parent_remains_authoritative_for_g0_g3": True,
        "self_included": False,
        "self_exclusion_reason": "manifest cannot recursively hash itself",
        "active_specification": {
            "relative_path": SPEC_RELATIVE,
            "sha256": SPEC_SHA,
            "file_size": spec.stat().st_size,
            "sole_active_specification": True,
        },
        "git_head": _run(["git", "-C", str(ROOT), "rev-parse", "HEAD"]),
        "parent_checkpoint": parent["parent_checkpoint"],
        "qualification_files": qualification,
        "qualification_files_sha256": _canonical_sha(qualification),
        "historical_source_binding_disposition": {
            "p5": "HISTORICAL_EXPECTED_SOURCE_MISMATCH",
            "p7": "HISTORICAL_EXPECTED_SOURCE_MISMATCH",
            "historical_snapshots_modified": False,
        },
        "parent_files": parent["files"],
        "parent_files_sha256": parent["files_sha256"],
        "delta_files": delta,
        "delta_files_sha256": _canonical_sha(delta),
        "effective_files": effective_files,
        "effective_files_sha256": _canonical_sha(effective_files),
        "conrft_repository": {
            "repository_path": str(CONRFT),
            "git_head_sha": head,
            "git_status": "clean",
            "git_describe": _git("describe", "--always", "--dirty"),
            "git_remote_url": _git("remote", "get-url", "origin"),
            "environment_binding_status": "R0_PREPARATION_READY",
            "runtime_imported": True,
            "octo_dependency": "not installed; type-only import shim for unused EncodingWrapper annotations",
            "files": conrft_files,
            "files_sha256": _canonical_sha(conrft_files),
            "license_sha256": _sha256(CONRFT / "LICENSE"),
        },
        "runtime_pretrained_resnet10": {
            "asset": prepared_pretrained,
            "provenance": prepared_pretrained_provenance,
            "conrft_repository_copy_status": "invalid_truncated_pickle_preserved_read_only",
        },
        "isolated_environment": environment_lock,
    }
    payload["closure_sha256"] = _canonical_sha(
        {
            "parent_manifest_sha256": PARENT_SHA,
            "active_specification": payload["active_specification"],
            "effective_files_sha256": payload["effective_files_sha256"],
            "qualification_files_sha256": payload["qualification_files_sha256"],
            "conrft_files_sha256": payload["conrft_repository"]["files_sha256"],
            "runtime_pretrained_resnet10_sha256": prepared_pretrained["sha256"],
            "environment_lock_sha256": environment_lock["environment_lock_sha256"],
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--environment", default="conrft_reward")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/development/stage2/stage2_source_manifest.v4_r0prep.json",
    )
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite append-only manifest: {output}")
    payload = build(args.environment)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=output.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(output)
    print(json.dumps({"output": str(output), "sha256": _sha256(output), "closure_sha256": payload["closure_sha256"]}))


if __name__ == "__main__":
    main()
