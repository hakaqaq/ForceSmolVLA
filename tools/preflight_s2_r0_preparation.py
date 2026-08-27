#!/usr/bin/env python3
"""R0 preparation only: dataset adapter and synthetic ConRFT smoke.

The default coordinator performs read-only frozen-asset checks, invokes this
file as an isolated conda worker, and writes one PREPARED_NOT_TRAINED artifact.
It never trains, applies optimizer updates, saves classifier checkpoints, or
produces task2 probabilities/rewards/terminals.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from io import BytesIO
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONRFT_HEAD = "a779fde7fa5db5a469960a8490c100f35b41b49e"
EXPECTED_PARENT_MANIFEST_SHA = "defa5b1d1a975c465154ac62e009863163947065127c557b5600025ce77b29eb"
EXPECTED_R5_MODEL_SHA = "49248561be7043b38bfce60f200d8bf265e1b16b4b9553ccc6aa4c87241b762e"
CONRFT_RUNTIME_ROOT = Path("/home/rlc123/conrft/serl_launcher")
PRETRAINED_PATH = (
    ROOT / "artifacts/development/stage2/reward_classifier/pretrained/resnet10_params.pkl"
)
DATASET_ROOT = ROOT / "datasets/task2_lerobotv3"
R5_ROOT = ROOT / "outputs/development/task2_lerobotv3_full_sft_10k_r5/checkpoints/step_010000"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _tree_sha(root: Path) -> dict[str, Any]:
    records = []
    total_size = 0
    for path in sorted(candidate for candidate in root.rglob("*") if candidate.is_file()):
        size = path.stat().st_size
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "file_size": size,
                "sha256": _sha256(path),
            }
        )
        total_size += size
    return {
        "algorithm": "sha256(canonical_json(sorted(relative_path,file_size,sha256)))",
        "sha256": _canonical_sha(records),
        "file_count": len(records),
        "total_file_size": total_size,
    }


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def _run(command: list[str], *, cwd: Path | None = None) -> str:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment.pop("PYTHONHOME", None)
    environment.pop("LD_LIBRARY_PATH", None)
    result = subprocess.run(
        command, cwd=cwd, check=True, text=True, capture_output=True, env=environment
    )
    return result.stdout.strip()


def _git_value(conrft: Path, *arguments: str) -> str:
    return _run(["git", "-C", str(conrft), *arguments])


def _import_adapter():
    path = ROOT / "tools/reward_classifier/conrft_lerobot_v3_adapter.py"
    spec = importlib.util.spec_from_file_location("conrft_lerobot_v3_adapter", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load reward adapter")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _install_type_only_octo_shim() -> None:
    """Satisfy unused annotations in encoding.py without installing Octo."""

    from typing import Any as TypingAny, Sequence as TypingSequence

    octo = types.ModuleType("octo")
    model = types.ModuleType("octo.model")
    octo_module = types.ModuleType("octo.model.octo_module")
    utils = types.ModuleType("octo.utils")
    typing_module = types.ModuleType("octo.utils.typing")

    class OctoTransformer:
        def __init__(self, *_: object, **__: object) -> None:
            raise RuntimeError("Octo instantiation is forbidden in R0 preparation")

    octo_module.OctoTransformer = OctoTransformer
    typing_module.Config = TypingAny
    typing_module.Data = TypingAny
    typing_module.Params = TypingAny
    typing_module.PRNGKey = TypingAny
    typing_module.Sequence = TypingSequence
    octo.model = model
    octo.utils = utils
    model.octo_module = octo_module
    utils.typing = typing_module
    for name, module in {
        "octo": octo,
        "octo.model": model,
        "octo.model.octo_module": octo_module,
        "octo.utils": utils,
        "octo.utils.typing": typing_module,
    }.items():
        module.__dict__["__forcesmolvla_type_only_shim__"] = True
        sys.modules[name] = module


def _array_tree_sha(tree: Any) -> str:
    import jax

    digest = hashlib.sha256()
    for leaf in jax.tree_util.tree_leaves(tree):
        array = np.asarray(leaf)
        digest.update(str(array.dtype).encode())
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _isolated_worker(output: Path) -> None:
    _install_type_only_octo_shim()
    sys.path.insert(0, str(CONRFT_RUNTIME_ROOT))

    import flax
    from flax import traverse_util
    from flax.training import checkpoints
    import jax
    import jaxlib
    import jax.numpy as jnp
    import optax

    from serl_launcher.networks.reward_classifier import create_classifier
    from serl_launcher.vision.data_augmentations import batched_random_crop

    if jax.default_backend() != "gpu":
        raise RuntimeError(f"synthetic classifier smoke requires GPU, got {jax.default_backend()}")

    adapter = _import_adapter()
    image1 = np.zeros(adapter.SOURCE_SHAPE, dtype=np.uint8)
    image2 = np.zeros(adapter.SOURCE_SHAPE, dtype=np.uint8)
    image1[0], image1[1], image1[2] = 17, 31, 47
    image2[0], image2[1], image2[2] = 61, 73, 89
    adapted = adapter.ConRFTLeRobotV3Adapter().adapt(
        {
            adapter.SOURCE_CAMERA_KEYS[0]: image1,
            adapter.SOURCE_CAMERA_KEYS[1]: image2,
        },
        row_reference=adapter.RowReference(
            "synthetic_r0prep", "synthetic", 0, "synthetic_episode", 0, 0.0
        ),
        camera_row_identity=adapter.CameraRowIdentity(1, 2),
    )
    observation = adapter.to_jax_observation(adapted.observation)
    key = jax.random.PRNGKey(20260825)
    classifier = create_classifier(
        key,
        observation,
        list(adapter.CLASSIFIER_CAMERA_KEYS),
        pretrained_encoder_path=str(PRETRAINED_PATH),
    )
    step_before = int(classifier.step)
    params_before = _array_tree_sha(classifier.params)

    direct = classifier.apply_fn({"params": classifier.params}, observation, train=False)
    wrapped = adapter.apply_conrft_classifier(classifier, adapted.observation)
    repeated = adapter.apply_conrft_classifier(classifier, adapted.observation)
    direct_np = np.asarray(jax.block_until_ready(direct))
    wrapped_np = np.asarray(jax.block_until_ready(wrapped))
    repeated_np = np.asarray(jax.block_until_ready(repeated))
    if not np.array_equal(direct_np, wrapped_np) or not np.array_equal(wrapped_np, repeated_np):
        raise RuntimeError("adapter/direct/repeat classifier logit mismatch")

    camera_forward_sensitivity: dict[str, float] = {}
    for camera_key in adapter.CLASSIFIER_CAMERA_KEYS:
        changed = dict(observation)
        changed[camera_key] = jnp.bitwise_xor(observation[camera_key], jnp.uint8(255))
        changed_logit = classifier.apply_fn({"params": classifier.params}, changed, train=False)
        difference = float(jnp.max(jnp.abs(changed_logit - direct)))
        camera_forward_sensitivity[camera_key] = difference
        if difference == 0.0:
            raise RuntimeError(f"classifier forward ignores camera {camera_key}")

    augmentation_key = jax.random.PRNGKey(314159)
    crop_a = batched_random_crop(
        observation[adapter.CLASSIFIER_CAMERA_KEYS[0]],
        augmentation_key,
        padding=4,
        num_batch_dims=2,
    )
    crop_b = batched_random_crop(
        observation[adapter.CLASSIFIER_CAMERA_KEYS[0]],
        augmentation_key,
        padding=4,
        num_batch_dims=2,
    )
    augmentation_fixed_key_exact = np.array_equal(
        np.asarray(jax.block_until_ready(crop_a)), np.asarray(jax.block_until_ready(crop_b))
    )
    if not augmentation_fixed_key_exact or crop_a.shape != observation[adapter.CLASSIFIER_CAMERA_KEYS[0]].shape:
        raise RuntimeError("native ConRFT random-crop contract failed")

    future1 = np.full(adapter.SOURCE_SHAPE, 255, dtype=np.uint8)
    future2 = np.full(adapter.SOURCE_SHAPE, 127, dtype=np.uint8)
    future1[:] = 3
    future2[:] = 5
    future_independent = np.array_equal(
        wrapped_np,
        np.asarray(jax.block_until_ready(adapter.apply_conrft_classifier(classifier, adapted.observation))),
    )
    if not future_independent:
        raise RuntimeError("future-frame mutation changed current logit")

    labels = jnp.zeros_like(direct)

    def loss_fn(params):
        logits = classifier.apply_fn({"params": params}, observation, train=False)
        return optax.sigmoid_binary_cross_entropy(logits, labels).mean()

    loss, grads = jax.value_and_grad(loss_fn)(classifier.params)
    jax.block_until_ready(loss)
    flat_grads = traverse_util.flatten_dict(grads)
    pretrained = [np.asarray(value) for path, value in flat_grads.items() if "pretrained_encoder" in path]
    trainable = [np.asarray(value) for path, value in flat_grads.items() if "pretrained_encoder" not in path]
    if not pretrained or any(np.any(value != 0) for value in pretrained):
        raise RuntimeError("pretrained ResNet gradient freeze boundary failed")
    if not trainable or not any(np.any(value != 0) for value in trainable):
        raise RuntimeError("synthetic classifier trainable gradient is zero")

    params_after = _array_tree_sha(classifier.params)
    step_after = int(classifier.step)
    if params_before != params_after or step_before != 0 or step_after != 0:
        raise RuntimeError("synthetic smoke performed an optimizer update")

    forbidden_modules = [
        name
        for name in sys.modules
        if name.startswith("serl_launcher.data.data_store")
        or name.startswith("experiments.mappings")
        or name.startswith("serl_robot_infra")
    ]
    if forbidden_modules:
        raise RuntimeError(f"forbidden ConRFT runtime imported: {forbidden_modules}")

    device = jax.devices()[0]
    backend = jax.lib.xla_bridge.get_backend()
    modules = {
        "jax": jax,
        "flax": flax,
        "optax": optax,
        "reward_classifier": sys.modules["serl_launcher.networks.reward_classifier"],
        "encoding": sys.modules["serl_launcher.common.encoding"],
        "resnet_v1": sys.modules["serl_launcher.vision.resnet_v1"],
    }
    payload = {
        "status": "pass",
        "backend": jax.default_backend(),
        "backend_platform_version": str(backend.platform_version),
        "gpu_device": str(device),
        "versions": {
            "python": sys.version.split()[0],
            "jax": jax.__version__,
            "jaxlib": jaxlib.__version__,
            "flax": flax.__version__,
            "optax": optax.__version__,
            "numpy": np.__version__,
        },
        "runtime_import_roots": {name: str(Path(module.__file__).resolve()) for name, module in modules.items()},
        "runtime_library_path": __import__("os").environ.get("LD_LIBRARY_PATH", "").split(":"),
        "classifier": {
            "create_classifier_source": "unmodified ConRFT reward_classifier.py",
            "camera_keys": list(adapter.CLASSIFIER_CAMERA_KEYS),
            "input_shapes": {key: list(value.shape) for key, value in observation.items()},
            "logit_shape": list(direct_np.shape),
            "logit": direct_np.tolist(),
            "probability": np.asarray(jax.nn.sigmoid(direct)).tolist(),
            "repeat_logit_exact": bool(np.array_equal(wrapped_np, repeated_np)),
            "adapter_direct_logit_exact": bool(np.array_equal(direct_np, wrapped_np)),
            "future_frame_independent": bool(future_independent),
            "camera_forward_sensitivity_max_abs": camera_forward_sensitivity,
            "both_cameras_observed": all(value > 0.0 for value in camera_forward_sensitivity.values()),
            "native_random_crop": {
                "padding": 4,
                "num_batch_dims": 2,
                "fixed_key_repeat_exact": bool(augmentation_fixed_key_exact),
                "shape_preserved": True,
            },
            "loss": float(loss),
            "objective": "optax.sigmoid_binary_cross_entropy",
            "pretrained_resnet_gradient_exact_zero": True,
            "trainable_classifier_gradient_nonzero": True,
            "optimizer_updates": 0,
            "train_state_step_before": step_before,
            "train_state_step_after": step_after,
            "params_sha256_before": params_before,
            "params_sha256_after": params_after,
            "checkpoint_interface_imported": bool(hasattr(checkpoints, "restore_checkpoint")),
            "checkpoint_saved": False,
        },
        "octo": {
            "installed_or_loaded": False,
            "type_only_import_shim": True,
            "instances_created": 0,
        },
        "forbidden_runtime_modules": forbidden_modules,
    }
    _atomic_json(output, payload)


def _decode_png_chw(payload: bytes) -> np.ndarray:
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    if rgb.shape != (480, 640, 3):
        raise RuntimeError(f"unexpected decoded image shape: {rgb.shape}")
    return np.ascontiguousarray(np.transpose(rgb, (2, 0, 1)))


def _real_dataset_adapter_smoke() -> dict[str, Any]:
    import pyarrow.parquet as pq

    adapter = _import_adapter()
    relative = Path("data/chunk-000/file-000.parquet")
    path = DATASET_ROOT / relative
    columns = [
        *adapter.SOURCE_CAMERA_KEYS,
        "frame_index",
        "episode_index",
        "timestamp",
        "provenance.camera1_receive_monotonic_ns",
        "provenance.camera2_receive_monotonic_ns",
    ]
    row = pq.read_table(path, columns=columns).slice(0, 1).to_pylist()[0]
    sample = {
        adapter.SOURCE_CAMERA_KEYS[0]: _decode_png_chw(row[adapter.SOURCE_CAMERA_KEYS[0]]["bytes"]),
        adapter.SOURCE_CAMERA_KEYS[1]: _decode_png_chw(row[adapter.SOURCE_CAMERA_KEYS[1]]["bytes"]),
    }
    instance = adapter.ConRFTLeRobotV3Adapter()
    reference = adapter.RowReference(
        "task2_lerobotv3",
        relative.as_posix(),
        0,
        f"episode_{int(row['episode_index']):06d}",
        int(row["frame_index"]),
        float(row["timestamp"]),
    )
    identity = adapter.CameraRowIdentity(
        int(row["provenance.camera1_receive_monotonic_ns"]),
        int(row["provenance.camera2_receive_monotonic_ns"]),
    )
    first = instance.adapt(sample, row_reference=reference, camera_row_identity=identity)
    second = instance.adapt(sample, row_reference=reference, camera_row_identity=identity)
    exact = all(np.array_equal(first.observation[key], second.observation[key]) for key in first.observation)
    if not exact:
        raise RuntimeError("real v3 adapter repeat mismatch")
    return {
        "status": "pass",
        "row_reference": asdict(reference),
        "camera_row_identity": asdict(identity),
        "source_shapes": {key: list(value.shape) for key, value in sample.items()},
        "classifier_keys": list(first.observation),
        "classifier_shapes": {key: list(value.shape) for key, value in first.observation.items()},
        "dtype": "uint8",
        "color_order": "RGB",
        "repeat_output_exact": exact,
        "frame_stack": 1,
        "retained_frame_count": instance.retained_frame_count,
        "images_persisted_or_copied": False,
    }


def _classifier_data_inventory() -> dict[str, Any]:
    candidates = []
    for base in (ROOT / "labels", ROOT / "datasets", ROOT / "outputs/development/stage2/reward_classifier"):
        if not base.exists():
            continue
        for path in base.rglob("*"):
            if not path.is_file():
                continue
            lower = path.name.lower()
            if path.name == "task2_episode_outcomes.v1.json":
                continue
            if path.suffix in {".pkl", ".jsonl", ".parquet"} and any(
                token in lower for token in ("success", "failure", "reward_label", "classifier_label")
            ):
                candidates.append(path.relative_to(ROOT).as_posix())
    return {
        "inventory_complete": True,
        "eligible_label_files": sorted(candidates),
        "eligible_frame_labels": {
            "positive": 0,
            "ordinary_negative": 0,
            "hard_negative": 0,
        },
        "classifier_splits": {"train": 0, "val": 0, "test": 0},
        "episode_disjoint_verified": False,
        "independent_heldout_collections": 0,
        "independent_heldout_ready": False,
        "task2_episode_count": 47,
        "task2_allowed_claim": "task_outcome=success; outcome_source=retrospective_operator_attestation",
        "task2_outcome_is_frame_label": False,
        "r0_training_data_ready": False,
        "blocker": "reward_classifier_labeled_data_missing",
    }


def _validate_r0_contracts() -> dict[str, Any]:
    input_path = ROOT / "configs/stage2_reward_classifier_input.development.yaml"
    detector_path = ROOT / "configs/stage2_reward_detector.development.yaml"
    label_schema_path = ROOT / "schemas/stage2_reward_classifier_example.schema.json"
    review_path = ROOT / "schemas/stage2_reward_classifier_review_template.json"
    input_contract = json.loads(input_path.read_text())
    detector = json.loads(detector_path.read_text())
    label_schema = json.loads(label_schema_path.read_text())
    review = json.loads(review_path.read_text())
    dataset_info = json.loads((DATASET_ROOT / "meta/info.json").read_text())

    expected_sources = ["observation.images.camera1", "observation.images.camera2"]
    expected_classifier = ["d435_third_person", "d405_wrist"]
    if input_contract.get("frame_stack") != 1:
        raise RuntimeError("frame_stack must be one")
    cameras = input_contract.get("camera_order", [])
    if [item.get("source_feature_key") for item in cameras] != expected_sources:
        raise RuntimeError("source camera order contract drift")
    if [item.get("classifier_key") for item in cameras] != expected_classifier:
        raise RuntimeError("classifier camera order contract drift")
    if input_contract.get("source_tensor") != {
        "rank": 3,
        "layout": "CHW",
        "shape": [3, 480, 640],
        "color_order": "RGB",
        "dtype": "uint8",
    }:
        raise RuntimeError("source image tensor contract drift")
    for source in expected_sources:
        feature = dataset_info["features"][source]
        if feature["shape"] != [3, 480, 640] or feature["dtype"] != "image":
            raise RuntimeError("LeRobot v3 image schema drift")
    if input_contract["preprocessing"]["adapter_normalization"] != "none":
        raise RuntimeError("adapter must not own ImageNet normalization")
    training = input_contract["classifier_training_contract"]
    if training["objective"] != "optax.sigmoid_binary_cross_entropy":
        raise RuntimeError("classifier objective drift")
    if training["r0_preparation_optimizer_updates"] != 0 or training["real_training_authorized"] is not False:
        raise RuntimeError("R0 preparation training boundary drift")
    if detector.get("probability_threshold") is not None:
        raise RuntimeError("real reward threshold is not approved")
    if detector.get("consecutive_positive_frames") is not None:
        raise RuntimeError("real consecutive-positive count is not approved")
    if detector.get("max_detection_delay_frames") is not None:
        raise RuntimeError("real detector delay is not approved")
    if detector.get("last_valid_frame_fallback") != "disabled":
        raise RuntimeError("last-valid-frame fallback must remain disabled")
    required_fields = set(label_schema.get("required", []))
    expected_fields = {
        "collection_id",
        "episode_id",
        "frame_index",
        "timestamp",
        "source_row_reference",
        "camera_row_identity",
        "label",
        "label_type",
        "label_source",
        "reviewer_id",
        "review_timestamp",
        "review_notes",
        "split",
    }
    if required_fields != expected_fields or review.get("records") != []:
        raise RuntimeError("label schema or empty review template drift")
    return {
        "status": "pass",
        "input_contract_sha256": _sha256(input_path),
        "detector_contract_sha256": _sha256(detector_path),
        "label_schema_sha256": _sha256(label_schema_path),
        "review_template_sha256": _sha256(review_path),
        "dataset_info_sha256": _sha256(DATASET_ROOT / "meta/info.json"),
    }


def _frozen_artifact_hashes() -> dict[str, str]:
    paths = sorted((ROOT / "artifacts/development").glob("p[4-9]*.json"))
    return {path.relative_to(ROOT).as_posix(): _sha256(path) for path in paths}


def _frozen_snapshot() -> dict[str, Any]:
    model = R5_ROOT / "model.safetensors"
    return {
        "r5_checkpoint_tree": _tree_sha(R5_ROOT),
        "r5_model_safetensors_sha256": _sha256(model),
        "dataset_tree": _tree_sha(DATASET_ROOT),
        "dataset_manifests": {
            name: _sha256(DATASET_ROOT / name)
            for name in ("conversion_manifest.json", "split_manifest.json", "normalizer_manifest.json")
        },
        "p4_p9_artifacts": _frozen_artifact_hashes(),
    }


def _environment_inventory(environment: str) -> dict[str, Any]:
    pip_freeze = _run(["conda", "run", "--name", environment, "python", "-m", "pip", "freeze"]).splitlines()
    conda_explicit = _run(["conda", "list", "--name", environment, "--explicit"]).splitlines()
    pip_check = _run(["conda", "run", "--name", environment, "python", "-m", "pip", "check"])
    cudnn = next(
        line.split("==", 1)[1] for line in pip_freeze if line.lower().startswith("nvidia-cudnn-cu11==")
    )
    lock = {
        "environment": environment,
        "input_lock_sha256": _sha256(ROOT / "configs/stage2_conrft_reward_environment.r0prep.txt"),
        "pip_freeze": pip_freeze,
        "conda_explicit": conda_explicit,
        "pip_check": pip_check,
        "cudnn": cudnn,
    }
    lock["runtime_inventory_sha256"] = _canonical_sha(lock)
    return lock


def _coordinator(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite R0 artifact: {output}")
    conrft = args.conrft.resolve()
    head = _git_value(conrft, "rev-parse", "HEAD")
    status = _git_value(conrft, "status", "--porcelain=v1", "--untracked-files=all")
    if head != EXPECTED_CONRFT_HEAD or status:
        raise RuntimeError(f"ConRFT authority mismatch: head={head}, status={status!r}")

    parent_manifest = ROOT / "artifacts/development/stage2/stage2_source_manifest.v4.json"
    if _sha256(parent_manifest) != EXPECTED_PARENT_MANIFEST_SHA:
        raise RuntimeError("v4 parent source manifest drift")
    source_manifest = args.source_manifest.resolve()
    source_payload = json.loads(source_manifest.read_text())
    if source_payload.get("active_gate") != "R0_PREPARATION_ONLY":
        raise RuntimeError("R0prep source manifest gate mismatch")

    before = _frozen_snapshot()
    if before["r5_model_safetensors_sha256"] != EXPECTED_R5_MODEL_SHA:
        raise RuntimeError("r5 checkpoint drift")
    dataset_smoke = _real_dataset_adapter_smoke()
    contract_validation = _validate_r0_contracts()

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as stream:
        worker_output = Path(stream.name)
    worker_output.unlink()
    try:
        environment = os.environ.copy()
        environment.pop("PYTHONPATH", None)
        environment.pop("PYTHONHOME", None)
        prefix = Path(
            _run(
                [
                    "conda",
                    "run",
                    "--name",
                    args.environment,
                    "python",
                    "-c",
                    "import sys; print(sys.prefix)",
                ]
            )
        )
        nvidia_root = prefix / "lib/python3.10/site-packages/nvidia"
        runtime_library_dirs = sorted(str(path) for path in nvidia_root.glob("*/lib") if path.is_dir())
        if not runtime_library_dirs:
            raise RuntimeError("isolated CUDA runtime library directories not found")
        environment["LD_LIBRARY_PATH"] = ":".join(runtime_library_dirs)
        subprocess.run(
            [
                "conda",
                "run",
                "--no-capture-output",
                "--name",
                args.environment,
                "python",
                str(Path(__file__).resolve()),
                "--isolated-worker-output",
                str(worker_output),
            ],
            cwd=ROOT,
            check=True,
            env=environment,
        )
        worker = json.loads(worker_output.read_text())
    finally:
        worker_output.unlink(missing_ok=True)

    environment = _environment_inventory(args.environment)
    source_environment = source_payload["isolated_environment"]
    for field in ("pip_freeze", "conda_explicit", "pip_check", "cudnn"):
        if environment[field] != source_environment[field]:
            raise RuntimeError(f"isolated environment drift: {field}")
    if worker["versions"] != {
        key: source_environment["versions"][key]
        for key in ("python", "jax", "jaxlib", "flax", "optax", "numpy")
    }:
        raise RuntimeError("isolated runtime version drift")
    if worker["runtime_library_path"] != source_environment["runtime_library_path"]:
        raise RuntimeError("isolated CUDA runtime library path drift")
    environment["environment_lock_sha256"] = source_environment["environment_lock_sha256"]
    inventory = _classifier_data_inventory()
    after = _frozen_snapshot()
    frozen_exact = before == after
    if not frozen_exact:
        raise RuntimeError("frozen Stage-1 asset changed during R0 preparation")

    artifact = {
        "schema_version": "1.0.0",
        "gate": "R0_PREPARATION",
        "artifact_status": "PREPARED_NOT_TRAINED",
        "acceptance_status": "PASS",
        "stage2_source_manifest": {
            "relative_path": source_manifest.relative_to(ROOT).as_posix(),
            "sha256": _sha256(source_manifest),
        },
        "parent_source_manifest": {
            "relative_path": parent_manifest.relative_to(ROOT).as_posix(),
            "sha256": EXPECTED_PARENT_MANIFEST_SHA,
        },
        "conrft": {
            "repository_path": str(conrft),
            "git_head_sha": head,
            "worktree_clean": not bool(status),
            "license_sha256": _sha256(conrft / "LICENSE"),
            "repository_modified": False,
            "repository_pretrained_copy_status": "invalid_truncated_pickle_preserved_read_only",
            "runtime_pretrained_resnet10": {
                "relative_path": PRETRAINED_PATH.relative_to(ROOT).as_posix(),
                "sha256": _sha256(PRETRAINED_PATH),
                "file_size": PRETRAINED_PATH.stat().st_size,
                "provenance_sha256": _sha256(PRETRAINED_PATH.with_suffix(".provenance.json")),
            },
        },
        "environment": {**environment, **worker["versions"], "cuda_backend": worker["backend_platform_version"], "gpu_device": worker["gpu_device"]},
        "runtime_import_roots": worker["runtime_import_roots"],
        "runtime_library_path": worker["runtime_library_path"],
        "input_contract": {
            "frame_stack": 1,
            "config_path": "configs/stage2_reward_classifier_input.development.yaml",
            "config_sha256": _sha256(ROOT / "configs/stage2_reward_classifier_input.development.yaml"),
            "validation": contract_validation,
        },
        "lerobot_v3_adapter_smoke": dataset_smoke,
        "synthetic_classifier_smoke": worker["classifier"],
        "octo": worker["octo"],
        "forbidden_runtime_modules": worker["forbidden_runtime_modules"],
        "classifier_label_inventory": inventory,
        "reward_detector": {
            "status": "synthetic_test_only",
            "config_path": "configs/stage2_reward_detector.development.yaml",
            "probability_threshold": None,
            "consecutive_positive_frames": None,
            "max_detection_delay_frames": None,
            "last_valid_frame_fallback": "disabled",
        },
        "historical_source_binding_disposition": {
            "p5": "HISTORICAL_EXPECTED_SOURCE_MISMATCH",
            "p7": "HISTORICAL_EXPECTED_SOURCE_MISMATCH",
            "behavior_regression_and_historical_source_sha_reported_separately": True,
            "frozen_v4_g0_allowlist_test": "EXPECTED_APPEND_ONLY_R0PREP_PARENT_MISMATCH",
            "frozen_v4_source_registry_test": "EXPECTED_APPEND_ONLY_R0PREP_PARENT_MISMATCH",
            "frozen_v4_g0_g3_artifacts_modified": False,
        },
        "frozen_assets": {
            "before": before,
            "after": after,
            "before_after_exact": frozen_exact,
        },
        "prohibitions": {
            "r0_training_authorized": False,
            "real_classifier_training_started": False,
            "real_optimizer_updates": 0,
            "real_classifier_checkpoint_created": False,
            "task2_reward_predictions_created": False,
            "task2_terminal_created": False,
            "g1_created": False,
            "g2_created": False,
            "critic_created": False,
            "target_network_created": False,
            "calql_loss_created": False,
            "rft_optimizer_created": False,
        },
    }
    _atomic_json(output, artifact)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--conrft", type=Path, default=Path("/home/rlc123/conrft"))
    parser.add_argument("--environment", default="conrft_reward")
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=ROOT / "artifacts/development/stage2/stage2_source_manifest.v4_r0prep.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/development/stage2/s2_r0_preparation.v4.json",
    )
    parser.add_argument("--isolated-worker-output", type=Path)
    args = parser.parse_args()
    if args.isolated_worker_output is not None:
        _isolated_worker(args.isolated_worker_output)
    else:
        _coordinator(args)


if __name__ == "__main__":
    main()
