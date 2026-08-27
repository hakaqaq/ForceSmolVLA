"""Development-only mask-aware three-step Force-aware Twin-Q topology."""

from __future__ import annotations

from collections import OrderedDict
import copy
import hashlib
from itertools import pairwise
import json
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AUTHORIZED_G1_ROOT = (
    PROJECT_ROOT / "artifacts/development/stage2/g1_frozen_detector_transition_view.v1"
)
AUTHORIZED_G1_MANIFEST_SHA256 = (
    "96dcc37abc365c945a075086efd60198c3391ad2d5fb3f0b53ff869e565e7bd5"
)
SAFE_RESNET10_SHA256 = "16052142a3ef841a12fb1d2a03965951e8fbf0dda3d89b995244419be7e1f9a5"
CANONICAL_TASK = "Pick up the purple ring and place it onto the red peg."
CANONICAL_TASK_SHA256 = "f85a8ba9fe524b42220c85e71737bdcf3ffd441b772981b92a87d6a89e324219"
TASK_FEATURE_DIM = 256
ACTION_SLOTS = 3
ACTION_DIM = 7
AUTHORIZED_G2_COLUMNS = (
    "transition_index",
    "episode_id",
    "split",
    "anchor_frame",
    "executed_steps",
    "executed_action_mask",
    "normalized_delta_action_exec_flat",
    "observation_row_reference",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(value).tobytes(order="C")).hexdigest()


def frozen_task_feature(task: str = CANONICAL_TASK) -> np.ndarray:
    """Return the deterministic, hash-bound canonical task feature."""

    if task != CANONICAL_TASK or hashlib.sha256(task.encode()).hexdigest() != CANONICAL_TASK_SHA256:
        raise ValueError("G2_NONCANONICAL_TASK_REJECTED")
    raw = np.frombuffer(
        hashlib.shake_256(b"ForceSmolVLA/G2/task-feature/v1\0" + task.encode()).digest(
            TASK_FEATURE_DIM
        ),
        dtype=np.uint8,
    ).astype(np.float32)
    feature = (raw - np.float32(127.5)) / np.float32(127.5)
    feature /= np.linalg.norm(feature)
    return np.ascontiguousarray(feature, dtype=np.float32)


def frozen_task_feature_sha256(task: str = CANONICAL_TASK) -> str:
    return _array_sha256(frozen_task_feature(task))


class _ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, 3, stride=stride, padding=0 if stride == 2 else 1,
            bias=False,
        )
        self.norm1 = nn.GroupNorm(4, out_channels, eps=1e-5)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(4, out_channels, eps=1e-5)
        self.proj_conv = (
            nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)
            if in_channels != out_channels else None
        )
        self.proj_norm = (
            nn.GroupNorm(4, out_channels, eps=1e-5) if self.proj_conv is not None else None
        )

    def forward(self, value: Tensor) -> Tensor:
        residual = value
        if self.conv1.stride == (2, 2):
            value = F.pad(value, (0, 1, 0, 1))
        value = F.relu(self.norm1(self.conv1(value)))
        value = self.norm2(self.conv2(value))
        if self.proj_conv is not None:
            residual = self.proj_norm(self.proj_conv(residual))
        return F.relu(residual + value)


class FrozenConRFTResNet10(nn.Module):
    """Exact safe-NPZ ResNetV1-10 backbone with native ConRFT GroupNorm."""

    def __init__(self) -> None:
        super().__init__()
        self.conv_init = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.norm_init = nn.GroupNorm(4, 64, eps=1e-5)
        self.blocks = nn.ModuleList(
            [
                _ResidualBlock(64, 64, 1),
                _ResidualBlock(64, 128, 2),
                _ResidualBlock(128, 256, 2),
                _ResidualBlock(256, 512, 2),
            ]
        )
        self.register_buffer("imagenet_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("imagenet_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def train(self, mode: bool = True):
        del mode
        return super().train(False)

    def forward(self, image: Tensor) -> Tensor:
        if image.ndim != 4 or image.shape[1] != 3 or image.shape[0] < 1:
            raise ValueError("G2_CAMERA_SHAPE_REJECTED")
        if image.dtype != torch.uint8 and not image.is_floating_point():
            raise TypeError("G2_CAMERA_DTYPE_REJECTED")
        value = image.to(dtype=torch.float32)
        if not torch.isfinite(value).all():
            raise ValueError("G2_NONFINITE_CAMERA_REJECTED")
        if value.shape[-2:] != (128, 128):
            value = F.interpolate(value, size=(128, 128), mode="bilinear", align_corners=False)
        value = (value / 255.0 - self.imagenet_mean) / self.imagenet_std
        value = F.relu(self.norm_init(self.conv_init(value)))
        value = F.max_pool2d(F.pad(value, (0, 1, 0, 1)), kernel_size=3, stride=2)
        for block in self.blocks:
            value = block(value)
        if value.shape[1:] != (512, 4, 4):
            raise RuntimeError(f"G2_RESNET_OUTPUT_SHAPE_DRIFT:{tuple(value.shape)}")
        return value


def _torch_parameter_name(path: tuple[str, ...]) -> tuple[str, bool]:
    if path == ("conv_init", "kernel"):
        return "conv_init.weight", True
    if path == ("norm_init", "scale"):
        return "norm_init.weight", False
    if path == ("norm_init", "bias"):
        return "norm_init.bias", False
    if len(path) == 3 and path[0].startswith("ResNetBlock_"):
        block = int(path[0].rsplit("_", 1)[1])
        part, leaf = path[1:]
        prefix = f"blocks.{block}."
        mapping = {
            ("Conv_0", "kernel"): ("conv1.weight", True),
            ("Conv_1", "kernel"): ("conv2.weight", True),
            ("MyGroupNorm_0", "scale"): ("norm1.weight", False),
            ("MyGroupNorm_0", "bias"): ("norm1.bias", False),
            ("MyGroupNorm_1", "scale"): ("norm2.weight", False),
            ("MyGroupNorm_1", "bias"): ("norm2.bias", False),
            ("conv_proj", "kernel"): ("proj_conv.weight", True),
            ("norm_proj", "scale"): ("proj_norm.weight", False),
            ("norm_proj", "bias"): ("proj_norm.bias", False),
        }
        if (part, leaf) in mapping:
            name, convolution = mapping[(part, leaf)]
            return prefix + name, convolution
    raise KeyError(path)


def load_frozen_conrft_resnet10(
    safe_npz_path: Path, asset_manifest_path: Path
) -> tuple[FrozenConRFTResNet10, dict]:
    safe_npz_path = Path(safe_npz_path).resolve()
    asset_manifest_path = Path(asset_manifest_path).resolve()
    if _sha256_file(safe_npz_path) != SAFE_RESNET10_SHA256:
        raise RuntimeError("G2_SAFE_RESNET10_SHA_DRIFT")
    manifest = json.loads(asset_manifest_path.read_text(encoding="utf-8"))
    records = manifest["parameter_inventory"]
    backbone_records = [record for record in records if record["parameter_path"][0] != "output_head"]
    head_records = [record for record in records if record["parameter_path"][0] == "output_head"]
    if len(backbone_records) != 36 or len(head_records) != 2:
        raise RuntimeError("G2_SAFE_RESNET10_COVERAGE_DRIFT")

    module = FrozenConRFTResNet10()
    parameters = dict(module.named_parameters())
    mapped: list[dict] = []
    with np.load(safe_npz_path, allow_pickle=False) as arrays:
        for record in backbone_records:
            source = np.asarray(arrays[record["safe_npz_key"]], dtype=np.float32)
            if tuple(source.shape) != tuple(record["shape"]) or _array_sha256(source) != record["array_sha256"]:
                raise RuntimeError(f"G2_SAFE_RESNET10_ARRAY_DRIFT:{record['safe_npz_key']}")
            name, convolution = _torch_parameter_name(tuple(record["parameter_path"]))
            converted = source.transpose(3, 2, 0, 1) if convolution else source
            parameter = parameters[name]
            if tuple(parameter.shape) != tuple(converted.shape):
                raise RuntimeError(f"G2_RESNET10_TORCH_SHAPE_MISMATCH:{name}")
            parameter.data.copy_(torch.from_numpy(np.ascontiguousarray(converted)))
            roundtrip = parameter.detach().cpu().numpy()
            if convolution:
                roundtrip = roundtrip.transpose(2, 3, 1, 0)
            parity = _array_sha256(roundtrip) == record["array_sha256"]
            if not parity:
                raise RuntimeError(f"G2_RESNET10_TENSOR_PARITY_FAILED:{name}")
            mapped.append(
                {
                    "safe_npz_key": record["safe_npz_key"],
                    "source_parameter_path": record["parameter_path"],
                    "source_shape": record["shape"],
                    "torch_parameter_name": name,
                    "torch_shape": list(parameter.shape),
                    "transform": "HWIO_to_OIHW" if convolution else "identity",
                    "array_sha256": record["array_sha256"],
                    "tensor_roundtrip_parity": parity,
                }
            )
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    module.eval()
    loaded_names = {item["torch_parameter_name"] for item in mapped}
    if loaded_names != set(parameters):
        raise RuntimeError("G2_RESNET10_PARAMETER_MAPPING_INCOMPLETE")
    evidence = {
        "model": "ConRFT resnetv1-10-frozen pre-pooling",
        "safe_npz_sha256": SAFE_RESNET10_SHA256,
        "source_key_count": len(records),
        "mapped_backbone_key_count": len(mapped),
        "mapped_shape_coverage": len(mapped) / len(parameters),
        "all_tensor_roundtrip_parity": all(item["tensor_roundtrip_parity"] for item in mapped),
        "unused_imagenet_output_head_keys": [record["safe_npz_key"] for record in head_records],
        "random_backbone_parameter_count": 0,
        "native_backbone_activation": "ReLU",
        "native_backbone_normalization": "GroupNorm(num_groups=4)",
        "new_trainable_layers_activation": "SiLU",
        "new_trainable_layers_normalization": "LayerNorm",
        "mapping": mapped,
    }
    return module, evidence


class _LayerNormMLP(nn.Module):
    def __init__(self, widths: tuple[int, ...]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        for source, target in pairwise(widths):
            layers.extend((nn.Linear(source, target), nn.LayerNorm(target), nn.SiLU()))
        self.layers = nn.Sequential(*layers)

    def forward(self, value: Tensor) -> Tensor:
        return self.layers(value)


class _SpatialProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.spatial_kernel = nn.Parameter(torch.empty(4, 4, 512, 8))
        nn.init.normal_(self.spatial_kernel, std=0.02)
        self.projection = nn.Linear(512 * 8, 256)
        self.norm = nn.LayerNorm(256)
        self.activation = nn.SiLU()

    def forward(self, feature_map: Tensor) -> Tensor:
        value = feature_map.permute(0, 2, 3, 1)
        value = torch.einsum("bhwc,hwcf->bcf", value, self.spatial_kernel).flatten(1)
        return self.activation(self.norm(self.projection(value)))


class ForceAwareMacroCritic(nn.Module):
    """Q(o^F, A^[3x7], mask) with hard post-MLP slot masking."""

    def __init__(
        self,
        camera1_backbone: FrozenConRFTResNet10,
        camera2_backbone: FrozenConRFTResNet10,
        *,
        task_feature: np.ndarray,
    ) -> None:
        super().__init__()
        self.camera1_backbone = camera1_backbone
        self.camera2_backbone = camera2_backbone
        self.camera1_spatial = _SpatialProjection()
        self.camera2_spatial = _SpatialProjection()
        self.task_projection = _LayerNormMLP((TASK_FEATURE_DIM, 128))
        self.state_mlp = _LayerNormMLP((7, 128, 128))
        self.wrench_mlp = _LayerNormMLP((6, 128, 128))
        self.action_slot_mlp = _LayerNormMLP((7, 128, 128))
        self.slot_position_embedding = nn.Parameter(torch.empty(ACTION_SLOTS, 128))
        nn.init.normal_(self.slot_position_embedding, std=0.02)
        self.fusion = _LayerNormMLP((1283, 1024, 512, 256))
        self.q_output = nn.Linear(256, 1)
        nn.init.uniform_(self.q_output.weight, -1e-3, 1e-3)
        nn.init.uniform_(self.q_output.bias, -1e-3, 1e-3)
        feature = np.asarray(task_feature, dtype=np.float32)
        if feature.shape != (TASK_FEATURE_DIM,) or not np.all(np.isfinite(feature)):
            raise ValueError("G2_TASK_FEATURE_INVALID")
        self.register_buffer("canonical_task_feature", torch.from_numpy(feature.copy()))
        self._permanent_eval = False

    def train(self, mode: bool = True):
        if self._permanent_eval:
            mode = False
        result = super().train(mode)
        self.camera1_backbone.eval()
        self.camera2_backbone.eval()
        return result

    def make_permanent_eval_target(self) -> None:
        self._permanent_eval = True
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.train(False)

    @staticmethod
    def _validate_vector(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
        if tuple(value.shape[1:]) != shape or value.ndim != len(shape) + 1 or value.shape[0] < 1:
            raise ValueError(f"G2_{name}_SHAPE_REJECTED")
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"G2_{name}_DTYPE_OR_FINITE_REJECTED")
        return value.to(dtype=torch.float32)

    @staticmethod
    def _validate_mask(mask: Tensor, batch: int) -> Tensor:
        if mask.dtype != torch.bool or tuple(mask.shape) != (batch, ACTION_SLOTS):
            raise ValueError("G2_ACTION_MASK_SHAPE_OR_DTYPE_REJECTED")
        counts = mask.sum(dim=1)
        if torch.any(counts == 0):
            raise ValueError("G2_ALL_FALSE_ACTION_MASK_REJECTED")
        expected = torch.arange(ACTION_SLOTS, device=mask.device)[None, :] < counts[:, None]
        if not torch.equal(mask, expected):
            raise ValueError("G2_NONPREFIX_ACTION_MASK_REJECTED")
        return mask

    def forward(
        self,
        camera1: Tensor,
        camera2: Tensor,
        task_feature: Tensor,
        normalized_state7: Tensor,
        normalized_wrench6: Tensor,
        normalized_action_k7: Tensor,
        executed_action_mask: Tensor,
    ) -> Tensor:
        batch = camera1.shape[0]
        if camera2.shape[0] != batch:
            raise ValueError("G2_CAMERA_BATCH_MISMATCH")
        state = self._validate_vector(normalized_state7, (7,), "STATE7")
        wrench = self._validate_vector(normalized_wrench6, (6,), "WRENCH6")
        action = self._validate_vector(normalized_action_k7, (ACTION_SLOTS, ACTION_DIM), "ACTION_K7")
        if state.shape[0] != batch or wrench.shape[0] != batch or action.shape[0] != batch:
            raise ValueError("G2_INPUT_BATCH_MISMATCH")
        mask = self._validate_mask(executed_action_mask, batch)
        task = self._validate_vector(task_feature, (TASK_FEATURE_DIM,), "TASK_FEATURE")
        expected_task = self.canonical_task_feature[None, :].expand(batch, -1)
        if task.shape[0] != batch or not torch.equal(task, expected_task):
            raise ValueError("G2_TASK_FEATURE_BINDING_REJECTED")

        with torch.no_grad():
            camera1_map = self.camera1_backbone(camera1)
            camera2_map = self.camera2_backbone(camera2)
        camera1_feature = self.camera1_spatial(camera1_map)
        camera2_feature = self.camera2_spatial(camera2_map)
        task_encoded = self.task_projection(task)
        state_encoded = self.state_mlp(state)
        wrench_encoded = self.wrench_mlp(wrench)
        slot_encoded = self.action_slot_mlp(action)
        slot_encoded = (slot_encoded + self.slot_position_embedding[None, :, :]) * mask[..., None]
        action_flat = slot_encoded.flatten(1)
        fused = torch.cat(
            (
                camera1_feature,
                camera2_feature,
                task_encoded,
                state_encoded,
                wrench_encoded,
                action_flat,
                mask.to(dtype=torch.float32),
            ),
            dim=1,
        )
        if fused.shape[1] != 1283:
            raise RuntimeError("G2_FUSION_DIMENSION_DRIFT")
        output = self.q_output(self.fusion(fused)).squeeze(-1).float()
        if output.shape != (batch,) or output.dtype != torch.float32 or not torch.isfinite(output).all():
            raise RuntimeError("G2_Q_OUTPUT_CONTRACT_FAILED")
        return output


def build_twin_q(
    safe_npz_path: Path,
    asset_manifest_path: Path,
    *,
    seed: int = 0,
) -> tuple[ForceAwareMacroCritic, ForceAwareMacroCritic, ForceAwareMacroCritic, ForceAwareMacroCritic, dict]:
    backbone, conversion = load_frozen_conrft_resnet10(safe_npz_path, asset_manifest_path)
    feature = frozen_task_feature()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        q1 = ForceAwareMacroCritic(copy.deepcopy(backbone), copy.deepcopy(backbone), task_feature=feature)
        torch.manual_seed(seed + 1)
        q2 = ForceAwareMacroCritic(copy.deepcopy(backbone), copy.deepcopy(backbone), task_feature=feature)
    q1_target = copy.deepcopy(q1)
    q2_target = copy.deepcopy(q2)
    q1_target.make_permanent_eval_target()
    q2_target.make_permanent_eval_target()
    return q1, q2, q1_target, q2_target, conversion


def polyak_blend_state(
    online: Mapping[str, Tensor], target: Mapping[str, Tensor], tau: float
) -> OrderedDict[str, Tensor]:
    """Pure Polyak formula: return a new state mapping and mutate neither input."""

    if not 0.0 <= tau <= 1.0 or online.keys() != target.keys():
        raise ValueError("G2_POLYAK_INPUT_REJECTED")
    blended: OrderedDict[str, Tensor] = OrderedDict()
    for name in online:
        source, prior = online[name], target[name]
        if source.shape != prior.shape or source.dtype != prior.dtype:
            raise ValueError(f"G2_POLYAK_STATE_MISMATCH:{name}")
        if source.is_floating_point():
            if tau == 0.0:
                value = prior.detach().clone()
            elif tau == 1.0:
                value = source.detach().clone()
            else:
                value = ((1.0 - tau) * prior.detach() + tau * source.detach()).clone()
        else:
            value = source.detach().clone() if tau == 1.0 else prior.detach().clone()
        blended[name] = value
    return blended


def load_authorized_g2_train_transitions(root: Path = AUTHORIZED_G1_ROOT):
    """Fail before opening anything unless the sole detector-G1 root is requested."""

    root = Path(root).resolve()
    if root != AUTHORIZED_G1_ROOT.resolve():
        raise RuntimeError("G2_REJECTED_NONAUTHORIZED_G1_ROOT_BEFORE_OPEN")
    if _sha256_file(root / "g1_manifest.json") != AUTHORIZED_G1_MANIFEST_SHA256:
        raise RuntimeError("G2_AUTHORIZED_G1_MANIFEST_SHA_DRIFT")
    from forcesmolvla.rft.detector_reward_transitions import load_training_transitions

    table = load_training_transitions(root)
    return table.select(AUTHORIZED_G2_COLUMNS)


def parameter_inventory(module: nn.Module) -> dict:
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    frozen = sum(parameter.numel() for parameter in module.parameters() if not parameter.requires_grad)
    return {"trainable": trainable, "frozen": frozen, "total": trainable + frozen}


def state_exact(left: nn.Module, right: nn.Module) -> bool:
    a, b = left.state_dict(), right.state_dict()
    return a.keys() == b.keys() and all(torch.equal(a[name], b[name]) for name in a)


def modules_storage_independent(left: nn.Module, right: nn.Module) -> bool:
    left_values = [*left.parameters(), *left.buffers()]
    right_values = [*right.parameters(), *right.buffers()]
    return not ({value.untyped_storage().data_ptr() for value in left_values} & {
        value.untyped_storage().data_ptr() for value in right_values
    })
