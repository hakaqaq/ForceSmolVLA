# ForceSmolVLA

**Release status: `v2.1.1` — development Stage-2 frozen-backbone,
value-guided force-action refinement plus local artifact-retention cleanup.** Stage-1 remains the full-model
force-conditioned behavior adaptation parent. See
[`PHASE2_RELEASE.md`](PHASE2_RELEASE.md) for the exact development scope,
limitations, evidence, and GitHub exclusions.

独立工程根目录：`/home/rlc123/ForceSmolVLA`  
独立 Conda 环境：`/home/rlc123/anaconda3/envs/forcesmolvla`

本工程基于固定的 LeRobot v0.6.0 commit 和 SmolVLA base revision，实现约
505.6M 参数的力觉条件化 Flow Actor；不修改 ForceVLA/OpenPI。当前架构、训练、
推理与 gate 的 source-of-truth 是 `ForceSmolVLA_Implementation_Spec_v4_2.md`，
v4.1 仅保留未被 v4.2 覆盖的 available-sensor 数据/几何契约。

## 方法定位

SmolVLA 的 post-VLM prefix 没有与未来 H 个动作位置天然对齐的表示，但 Action
Expert 内部仍有 H 个 action suffix hidden。ForceSmolVLA 保留 ForceVLA-inspired
post-VLM force fusion 与 MoE 思想，并新增 Action-Query Force Residual Adapter：
以 Action Expert hidden 为主要 query，显式结合 noisy action、action position 和
flow timestep，在原生 prefix K/V cache 之外查询固定的 Force Context。Force 分支
不接收、不拼接或修改 `past_key_values`。

v4.2 的 Stage-1 证明范围是离线全参数 Force-conditioned Actor。Stage-2 已实现并
验收 development-only 的 frozen-VLM Twin-Q、Flow-Matching + Q-guidance、
ActionContract-v2、exact resume 和 throughput-v2 路径；当前仍不声称 formal
Detector validation、无偏策略评估、部署发布或在线/真机训练已经完成。

## 环境

```bash
conda activate forcesmolvla
unset PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

## raw task1 → ForceSmolVLA LeRobot v3

转换入口与全部实现均在本工程：

- `tools/convert_franka_raw_to_lerobot_v3.py`
- `src/forcesmolvla/raw_to_lerobot_v3.py`

转换器只读 raw 输入并新建 LeRobot v3 输出；不 import、不执行 legacy v2.1
converter，也不允许覆盖已有输出。数据和本机路径不随 GitHub 仓库发布，调用时应
显式提供 `--raw-root`、`--output-root` 与 `--repo-id`。

```bash
python tools/convert_franka_raw_to_lerobot_v3.py \
  --raw-root /path/to/raw/task \
  --output-root datasets/task_lerobotv3 \
  --repo-id local/task_lerobotv3 \
  --preflight-only
```

当前命令会因未批准 RuleSpec/runtime/signature字段而 fail-closed，不会创建输出目录。
批准工件完整后，去掉 `--preflight-only` 才会执行正式转换。

全量开发审计转换必须显式使用 `--development-only`；输出 manifest 会保持
`artifact_status=development_only`，不能冒充正式训练数据。

训练/验证加载不得使用 `meta/info.json` 的存储级默认 `train=0:50`，必须通过项目
加载器执行 episode-disjoint `split_manifest.json`：

```python
from pathlib import Path
from forcesmolvla.dataset_v3 import load_dataset_split

dataset = load_dataset_split(
    Path("datasets/task_lerobotv3"),
    repo_id="local/task_lerobotv3",
    split_name="train",
    artifact_use="development",
)
```

`artifact_use="formal"` 会拒绝当前 development-only manifest。正式批准后必须转换
到一个新的、空的输出目录；转换器不会覆盖当前审计产物。

## 离线训练状态

当前training stage已按用户指令修订为`offline_full_finetune`。ForceToken-MoE共有
505,620,341个参数，全部`requires_grad=True`，VLM和vision encoder参与离线SFT。未来HIL online
fine-tuning使用独立`online_hil_vlm_frozen`阶段；切换阶段时禁止恢复旧optimizer state。

task2 raw的独立开发转换命令为：

```bash
python tools/convert_franka_raw_to_lerobot_v3.py \
  --raw-root /home/rlc123/fr3_client_ws/datasets/task2 \
  --output-root /home/rlc123/ForceSmolVLA/datasets/task2_lerobotv3 \
  --repo-id local/task2_lerobotv3 \
  --runtime-spec configs/converter_runtime_spec.task2.development.json \
  --development-only
```

完整GPU-only离线全参数训练入口为：

```bash
env PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
  python tools/train.py \
    --dataset datasets/task2_lerobotv3 \
    --config configs/train/task2.json
```

`tools/train.py`是通用训练入口，不包含task1/task2/task3分支。`--dataset`指定任意满足
ForceSmolVLA数据契约的LeRobot v3目录；任务名、输出目录、数据准入工件、P8证据和日志间隔
由`--config`指定。新增数据集时创建新的`configs/train/<experiment>.json`，无需修改训练代码。

该长程入口只有在当前源码绑定的 P8 B4×1 single-pass exact-resume dry-run 通过后
才允许启动；历史 P8、缺失 checkpoint 或旧 B2×8 evidence 均不能解锁训练。

该入口严格使用episode-disjoint train split、train-only normalizer、双相机、H=50、
ForceToken-MoE单遍联合flow/router loss、B4×1和40,000 samples（派生10,000 optimizer updates）；不含CPU fallback、
VLM冻结或LoRA。它沿用SmolVLA的Flow-Matching/AdamW/bf16 recipe和ForceVLA的batch=4、
每batch一次联合forward/backward语义；P7 B2×8 exact two-pass仅保留为独立gate/parity test。
10,000-update短程预算按LeRobot官方规则将原1000/20000 scheduler preset自动缩放为500-step warmup、
update 10,000衰减至2.5e-6。数据读取使用8线程有序单窗口预取，下一窗口indices随sampler state
写入resume contract，不改变uniform sampling语义。仅在最终update 10,000保存唯一strict checkpoint；周期validation只记录指标，不保存中间或best checkpoint。所有训练产物
仍为`acceptance_status=development_only`，不会被formal resolver接受。
