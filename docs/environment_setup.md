# ForceSmolVLA 独立 Conda 环境

冻结环境名：`forcesmolvla`  
冻结路径：`/home/rlc123/anaconda3/envs/forcesmolvla`  
Python：`3.12.13`

LeRobot 可以安装在 `venv` 或 Conda 中，并不要求 `venv`。本工程选择 Conda，是为了把 Python、PyTorch CUDA wheel 与 ForceVLA/OpenPI 环境完整隔离。禁止向 Conda `base` 安装本工程包。

## 日常进入环境

```bash
conda activate forcesmolvla
unset PYTHONPATH
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

必须清除外部 `PYTHONPATH`；当前主机的 ROS Python 3.10 路径会污染 Python 3.12 的 pytest 插件发现。测试命令同时关闭全局插件自动加载：

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

## 可复现重建顺序

从工程根目录执行，Conda 基础层只使用冻结的 defaults 包：

```bash
conda create -n forcesmolvla --override-channels -c defaults \
  python=3.12.13 pip=26.1.2
conda activate forcesmolvla
unset PYTHONPATH
python -m pip install --index-url https://download.pytorch.org/whl/cu128 \
  torch==2.11.0+cu128 torchvision==0.26.0+cu128
python -m pip install -e './vendor/lerobot[dataset,training,smolvla]'
python -m pip install -e '.[test]'
python -m pip check
```

`environment-manifest/conda-explicit.txt` 冻结 Conda 基础层，`pip-freeze.txt` 冻结最终 Python 层；两者必须按该顺序应用。LeRobot editable source固定在本工程 `vendor/lerobot`，其 HEAD 必须是 `30da8e687a6dfc617fcd94afc367ac7071c376ce`。

## 已验证运行时

- PyTorch `2.11.0+cu128`
- torchvision `0.26.0+cu128`
- CUDA runtime `12.8`
- Transformers `5.5.4`
- LeRobot `0.6.0`
- SciPy `1.16.3`（冻结的 causal SOS wrench filter实现）
- GPU：NVIDIA GeForce RTX 4090 D，driver `570.211.01`
- bf16 CUDA matrix smoke：通过

所有下载资产仅位于 `/home/rlc123/ForceSmolVLA/assets/`；strict offline preflight不会访问 Hub或其他网络服务。
