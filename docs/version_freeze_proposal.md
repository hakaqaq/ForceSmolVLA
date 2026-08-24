# ForceSmolVLA 版本冻结提案

提案日期：2026-08-19  
状态：`SELECTED_EXACT_REVISIONS` / `LOCAL_VERIFICATION_PASSED` / `UNSIGNED`  
下载闸门：用户已允许约 1 GB base assets；仅可写新工程本地资产目录  
训练闸门：P0–P4与已批准 RuleSpec gates通过后开放 within-session offline SFT

## 1. 建议冻结集合

### 1.1 LeRobot 源码

| 字段 | 值 |
|---|---|
| repository | `https://github.com/huggingface/lerobot.git` |
| release tag | `v0.6.0` |
| commit | `30da8e687a6dfc617fcd94afc367ac7071c376ce` |
| release date | 2026-07-06 |
| selection reason | 官方签名 release；包含 SmolVLA action-padding 修复、pretrained revision 支持和 v3.0 dataset API；避免 `main` 漂移 |
| `pyproject.toml` SHA256 | `bf2140842c4568ac77cb99f97bbe092def5fa8cb30d011a19a0d0077764ce80a` |
| `configuration_smolvla.py` SHA256 | `2fb637cb428fa2fdf1d114646dcffaf4728216bfe5b7039d5d0cac4857ffc4e0` |
| `modeling_smolvla.py` SHA256 | `5daaa297954acf9ae89397b571322f07bf30798e72f566e61a09393342cb1f99` |
| `smolvlm_with_expert.py` SHA256 | `f7542fa2bf904f9ef64d26843809f913a5e93c4dfa538dda06db0b288391ab4d` |

以上源码 SHA256 是从精确 commit 的 raw 内容计算的预期值；clone 后必须重新本地计算。未来 wheel/git archive SHA256 只能在独立环境构建后填写，当前必须为 `null` 并使正式入口失败。

### 1.2 SmolVLA 基础 checkpoint

| 字段 | 值 |
|---|---|
| repo | `lerobot/smolvla_base` |
| revision | `d5ef92b547b2bf36bdd50f18ea6ed6463cb5c5af` |
| `model.safetensors` size | `906720008` bytes |
| publisher LFS SHA256 | `8f8dc071d5b933e79edd2b73b8d6b5cca482ef0437c099ea3ec13ab978a38fc8` |
| `config.json` SHA256 | `971b1154cf562822319fc50b482d9b3234b5badb1dd3c553d6d0681ad8fbe47b` |
| `train_config.json` SHA256 | `b66ca306a88aa9c784df34a47a15fdcf4fea4431f90b4fa7c7f41a1f0d3b49a1` |

本地 size和 SHA256已全部验证通过。资产只位于 `/home/rlc123/ForceSmolVLA/assets/`，未写入现有 OpenPI cache或 checkpoint目录。

### 1.3 传递性 SmolVLM 构造器资产

SmolVLA 在 `load_vlm_weights=false` 时仍调用 `AutoConfig.from_pretrained` 和 `AutoProcessor.from_pretrained`。因此 strict offline constructor 还必须冻结下列仓库，但不需要下载其约 2.03 GB 的 `model.safetensors`：

| 字段 | 值 |
|---|---|
| repo | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` |
| revision | `7b375e1b73b11138ff12fe22c8f2822d8fe03467` |

必须复制到 checkpoint `base_assets/smolvlm_constructor/` 的文件及 SHA256：

| 文件 | SHA256 |
|---|---|
| `added_tokens.json` | `74135b8664b56088c0006f1c8e848d79a8eba003411f72ebf1dc2ee96227be3a` |
| `chat_template.json` | `b585e3598909a5687f9f9d738d35223724dedef256b9b274e1cbfb32b13c74bf` |
| `config.json` | `ea6bc1237e96247f6258de3e202e2e62b93d6f386dc47e7b36b5588bf3a15e17` |
| `merges.txt` | `0b54e8aa4e53d5383e2e4bc635a56b43f9647f7b13832d5d9ecd8f82dac4f510` |
| `preprocessor_config.json` | `149e315d9410368e5491455bb06e0f763426e9e56cca731c13b24404a29b6374` |
| `processor_config.json` | `f3ad45028447b3562b4752be0d5916d6806c1ef589091a469608dcf0faa1737c` |
| `special_tokens_map.json` | `2dfea2a426162316ff1567c82bc6d36d9690cd9f90455f075c77daca78b45c60` |
| `tokenizer.json` | `5ece781dc8d2b2f3e2f289ca0ae50b17cfc27dd27bfe7971bb8241e0b964331a` |
| `tokenizer_config.json` | `dd9ce2ab89a3dd881bd9378f1a79b943a064b9275a7e1706d5b7b47b68977913` |
| `vocab.json` | `82b84012e3add4d01d12ba14442026e49b8cbbaead1f79ecf3d919784f82dc79` |

`generation_config.json` 不在当前 ForceSmolVLA constructor 调用图中，默认不打包；若运行时调用图证明读取它，则先修订 allowlist 和 source binding，不能静默加入。

## 2. 独立环境提案

| 字段 | 候选值/状态 |
|---|---|
| project root | `/home/rlc123/ForceSmolVLA` |
| environment | Conda env `forcesmolvla` at `/home/rlc123/anaconda3/envs/forcesmolvla` |
| assets | `/home/rlc123/ForceSmolVLA/assets` |
| Python | environment精确版本 `3.12.13`；LeRobot v0.6.0声明 `>=3.12` |
| environment manager | `conda 25.11.0`；LeRobot/project Python包只安装到独立 env，不安装到 base |
| LeRobot install | exact git commit，不从 PyPI 浮动安装 |
| LeRobot extras | 最小集合 `dataset,training,smolvla`；不安装 hardware/async/RTC 服务依赖 |
| Transformers | `5.5.4` |
| Torch | `2.11.0+cu128`；torchvision `0.26.0+cu128` |
| CUDA/GPU | runtime 12.8；RTX 4090 D；driver 570.211.01；bf16 smoke通过 |
| dataset format | LeRobot `v3.0` |

Conda explicit manifest、from-history、pip freeze与 requirements lock已导出并写入 SHA256。普通受限进程看不到GPU设备；在授权GPU上下文中 `nvidia-smi`、CUDA 12.8和bf16 matrix smoke均通过。

## 3. Base config 解析结果与自定义 resolved 候选

从固定 checkpoint config 直接得到：

| 字段 | base 值 | ForceSmolVLA 候选 |
|---|---:|---:|
| `chunk_size` | 50 | 50 |
| `n_action_steps` | 1 | 1，原样保留 |
| `max_state_dim` | 32 | 32 |
| `max_action_dim` | 32 | 32 |
| `n_obs_steps` | 1 | 1 |
| `num_steps` | 10 | 10，唯一采样步数字段 |
| `use_cache` | true | true |
| `attention_mode` | `cross_attn` | `cross_attn` |
| `pad_language_to` | `max_length` | `max_length` |
| `tokenizer_max_length` | 48 | 48 |
| `adapt_to_pi_aloha` | false | false |
| `use_delta_joint_actions_aloha` | false | false |
| `empty_cameras` | 0 | 0 |
| input cameras | 3 | 严格重定为 2 |
| state/action active dims | 6/6 | 7/7 |
| `rtc_config` | absent/default None | 必须显式为 None |
| `compile_model` | default false | false（P0–P4） |

从固定 SmolVLM architecture config 与 expert multiplier 得到静态候选：

- `D_vlm=960`
- `D_expert=720`
- `D_action=32`
- VLM attention heads `15`
- VLM head dim `64`
- 每个 512×512 image 的预期 physical token 数为 `64`
- 二相机、48 个语言 physical slots 和 1 个 state slot的预期 `N_prefix_physical=177`
- 预期 physical spans：camera1 `[0,64)`, camera2 `[64,128)`, language `[128,176)`, state `[176,177)`

`177` 已由本地 processor + model以 B=2、language valid lengths 48/17实测确认，并写入 development-only `resolved_force_config.json`。运行时不根据 valid language length改变 physical span。

## 4. 上游兼容性结果

| v4.1 需求 | 静态结果 | P0 处置 |
|---|---|---|
| PyTorch SmolVLA + native Action Expert | 可用 | 继承固定源码，禁止复制后漂移 |
| 7D→32D padding | 上游支持维度 padding | 自定义逐维 mask、noise、Euler、loss 约束 |
| 50-step chunk | 支持 | 固定 H=50 |
| two-camera exact keys/order | 上游允许可变 image keys，但会容忍部分缺失 | 覆盖为 exact-set + exact-order fail-fast |
| language max length | 支持 `max_length` | 固定 right padding/right truncation 并做 manifest |
| `num_steps` single binding | 上游 sample 使用 `config.num_steps` | 自定义类保留唯一绑定并测试 |
| RTC fully disabled | 不满足 | policy/model `_rtc_enabled=false`，拒绝 kwargs，`init_rtc_processor` 断言 None |
| PrefixContext/Layout | 不存在 | 新增 typed immutable return value |
| physical cache crop | 未显式保证 | 以 `N_prefix_physical` 管理并逐 Euler step 校验 |
| heterogeneous valid-length parity | 上游 suffix position 使用 `sum(prefix_pad_masks)` | 不直接复用该路径；P4 必须用 frozen layout/position rules |
| active-feature-only flow noise/loss | 不满足 v4.1 | P3 新增 feature/time masks并做 padding invariance |
| strict local constructor | 上游会按 `vlm_model_name` 调 Hub | 改为签名本地目录；设置 offline env 并 monkeypatch/guard Hub calls |
| base state-dict allowlist | 通过 | 500模型 tensors严格匹配；仅丢弃18个明列旧 SO100 normalizer tensors；missing/unexpected=0 |

结论：该版本可作为扩展基座，但上游原生 policy 本身不满足完整 v4.1 ForceSmolVLA契约。任何找不到的字段或语义差异都必须让 P0 失败，不能写多版本兼容分支。

## 5. 下载前/冻结前必须通过

1. 三个精确 revisions已按用户的“按推荐执行”授权选定；不得自动升级。
2. GPU 驱动、CUDA 可见性和 bf16 支持预检可运行。
3. 生成独立 `pyproject.toml`、Conda explicit manifest和 pip freeze，报告完整解析版本。
4. 只把 checkpoint 和 constructor assets 下载到 `/home/rlc123/ForceSmolVLA/assets`。
5. 本地逐文件 SHA256 与本提案一致。
6. 用 `HF_HUB_OFFLINE=1`、`TRANSFORMERS_OFFLINE=1`、禁网 guard 完成 config/tokenizer/processor constructor smoke。
7. 完成 base state-dict missing/unexpected allowlist、D_vlm/D_expert/D_action 和 physical prefix 测量。
8. 写出 `source_binding.json` 与 `resolved_force_config.json`；在正式签名机制确认前二者保持 `development_only`，正式入口拒绝。

## 6. 尚需用户确认的签名字段

实现不会自行选择算法、密钥或批准人。正式 detached signature 前需要明确：

- `signature_format` 与文件扩展名
- `signature_algorithm`
- `digest_algorithm`
- canonical serialization 规则及版本
- `key_id` 命名规则
- 公钥/证书来源与 trust-store 路径
- signature 编码（raw/base64/armored 等）
- signature scope（单文件、manifest 或整个 artifact tree）
- signer 身份、实验批准人身份及二者能否相同
- approval quorum
- `approval_id` 的签发系统与格式
- signing/approval timestamp 来源、是否需要 TSA
- key expiry/revocation 查询与离线缓存规则
- artifact expiry/renewal 规则
- verification CLI/registry 入口与失败码
- emergency revocation/rollback 流程

这些字段任一未确认，所有工件都只能是 `development_only`。
