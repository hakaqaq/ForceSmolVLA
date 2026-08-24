# P8 implementation/preflight report

> Historical notice（2026-08-20）：v4.2 已修改 checkpoint/resume、payload contract、action API 与 source binding。本报告和旧 checkpoint 仅保留历史证据，当前 P8 必须在 P6/P7 重跑后重新 strict reload。

日期：2026-08-19（Asia/Shanghai）  
状态：`DEVELOPMENT_GATE_PASS / P9_NOT_STARTED`  
acceptance：`development_only`，`formal_eligible=false`

## 结论

P8 的本地完整 Force checkpoint、embedded constructor assets、artifact payload hash、精确 trainability manifest、同阶段训练状态恢复，以及全新离线进程 strict reload 已实现并通过真实 RTX 4090 D gate。全套 96 项测试通过。

真实 gate 从 task1 v4.1 train split 取固定 16 个 tuple，以双相机、`H=50`、`B=2`、8 microbatch、ForceToken-MoE 和全参数 offline finetuning 执行一次真实 two-pass optimizer update，然后保存完整 model/optimizer/scheduler/scaler/RNG/sampler/accumulation 状态。新的空 HF cache 进程在网络和 Hub API deny 下，以 `strict=true`、`local_files_only=true`、`force_download=false` 重载；固定 B=2 的 PrefixContext、KV cache、velocity、normalized delta7、unnormalized delta7、absolute7 和 normalizer call count 全部逐字节一致。

checkpoint 和全部报告仍为 development-only。缺少 trusted detached signature、批准人、正式 verifier 和 wheelhouse binding 时，formal load 明确 fail-closed。P9 未开始。

## 实现

- `src/forcesmolvla/checkpoint.py`
  - development artifact manifest 对 checkpoint 全部文件执行 exact fileset、size 和 SHA256 绑定；symlink 被拒绝。
  - Force loader 只接受本地目录、`strict=true`、`local_files_only=true`、`force_download=false`、无 revision/config override；remote repo id 和裸 SmolVLA config 被拒绝。
  - checkpoint 内嵌 `base_assets/smolvlm_constructor/`；加载时强制 `load_vlm_weights=false`。
  - trainability manifest 保存每个参数的 name/shape/dtype/numel/requires_grad 与 trainable/frozen name hashes。
  - 复用 LeRobot v0.6.0 原生 optimizer/scheduler/RNG state；补充 disabled GradScaler、uniform sampler RNG/cursor、8-microbatch accumulation boundary 与 exact resume contract。
  - optimizer state hash 直接覆盖 tensor 原始字节，支持 bf16 和 0-D scalar；跨 training stage optimizer restore 在读取状态前拒绝。
- `src/forcesmolvla/modeling_forcesmolvla.py`
  - `ForceSmolVLAPolicy.from_pretrained` 注册并强制 `type=force_smolvla`，仅走本地完整 Force checkpoint strict path。
- `tools/p8_checkpoint_common.py`
  - 固定 B=2 parity 计算，覆盖 prefix tensors/layout、逐层 key/value cache、cached velocity、三种 action representation 和 exactly-once normalizer count。
- `tools/preflight_p8_checkpoint_gpu.py`
  - source-bound CUDA-only一更新、完整 save、artifact hash、fresh-process cold worker orchestration；拒绝覆盖既有工件。
- `tools/p8_cold_start_worker.py`
  - 从空 cache 启动，禁用 socket 和 Hub APIs，恢复 deterministic CUDA flags及完整训练状态，再执行 exact parity。
  - 允许本地 LeRobot v3 parquet materialize 到隔离的 `HF_HOME/datasets/`；`HF_HUB_CACHE`、Transformers/model cache、网络和 Hub 调用仍必须为零。
- `configs/p8_checkpoint_contract.development.json`
  - 冻结 required payload、resume exact 字段、cold-start B=2 比对项以及 `P9 not implemented` 边界。

## 测试

- 全套：`96 passed`。
- P8专项：`10 passed`。
- 覆盖：artifact tamper/extra file/formal rejection、local strict arguments、remote identifier、exact trainability names/hash、bf16 optimizer bytes、optimizer/scheduler/scaler/Python/NumPy/Torch RNG/sampler/accumulation恢复和跨阶段拒绝。

## 真实 RTX 4090 D update

- 参数：total/trainable=`505,620,341/505,620,341`，frozen=`0`。
- 输入：task1 v4.1 train split，16 samples，8×B2，双相机，H=50。
- router：2368 valid tokens；route counts=`[263,95,2010,0]`；Pass A/B probability max error=`0.0`。
- loss：flow=`28.4218909740`，balance=`2.44499492645`，z=`4.53946560621`，weighted total=`28.4508800507`。
- gradient norm before clip=`179.9482574463`。
- optimizer/scheduler steps=`1/1`。
- CUDA/wall latency=`2203.474 ms/2.203458 s`。
- peak allocated/reserved=`7,186,935,296/7,660,896,256 bytes`=`6.6934/7.1348 GiB`。
- CPU fallback=false；architecture downgrade=false。

## 完整 checkpoint

- 路径：`outputs/development/p8_checkpoint_seed42_step000001`。
- payload count=`78`，总 payload bytes=`3,828,404,463`=`3.5655 GiB`。
- model SHA256=`e83f0d6d1ce6753ea0801c197b6d79cd23f77dca2d92e387931f617cbe91ec1d`。
- optimizer state SHA256（文件）=`e9e9332c505767fc3e40cb2b316647c9849d294c594fa74079237546f2eae587`。
- artifact manifest SHA256=`7b765940ae13407c7920cb79876254524e71f3956fafd8265dae960435100207`。
- artifact manifest状态：`development_only`、`formal_eligible=false`、`detached_signature=null`、`approval=null`。
- embedded local config/tokenizer/processor/chat template存在且参与payload hash。
- environment manifest、Conda explicit/from-history、pip freeze、requirements lock均参与payload hash。

## exact resume 与 cold start

- fresh process PID=`3053393`，Python=`/home/rlc123/anaconda3/envs/forcesmolvla/bin/python`。
- training stage=`offline_full_finetune`，step=`1`。
- optimizer、scheduler、disabled scaler、Python/NumPy/Torch CPU/CUDA RNG、sampler RNG/cursor和accumulation phase全部精确恢复。
- sampler恢复后继续8次draw，cursor由16精确到24。
- source process和fresh process optimizer canonical state SHA256=`502247a76fe673c318c2267a0a249e388e27febb1f185932a5e51ebf8f7f47c9`。
- 空 HF cache启动=true；Hub/model cache files after=0；network attempts=0；Hub API attempts=0。
- 本地数据加载在隔离cache生成15个`HF_HOME/datasets/`文件；没有远程cache或网络访问，临时目录由主preflight自动删除。
- cold peak allocated/reserved=`4,303,471,616/4,391,436,288 bytes`=`4.0079/4.0898 GiB`。
- formal resolver拒绝原因=`FORMAL_FORCE_CHECKPOINT_SIGNATURE_OR_APPROVAL_MISSING`。

## 固定 parity

- fixed validation `L_flow` source run1/run2/cold=`25.60464859008789`，精确相等。
- PrefixContext、prefix cache、velocity、normalized delta7 chunk、unnormalized delta7 chunk、absolute7 chunk和normalizer call count全部 exact。
- parity SHA256=`23db34b31f050663a96c167ce8e44b8a519db2b1e8b8a02da21268f038512252`。
- parity reference SHA256=`5c5838fc4a3b51d377a57abb2990cee6a1e11cced81c806344d1a06e81a623ee`。

## 工件 hash

- `p8_gpu_preflight.json`: `9a1b2942017bcf511b1dba30871a483da25edcaf086f64b68118d3edd1323c5f`
- `p8_resolved_config.json`: `a2999a6e34d59db9c24558c9eb791b51775de2568d9f250f0c992c5be3513b95`
- `p8_cold_start_result.json`: `599d33f29ac9d3b3df32bbe16c592ee6b3d5a46f04eae5036ed6bd27a4fd7b28`
- `p8_source_binding.json`: `a207a4fa4e0758c5b8ddda41c8d9bec4b107fceb95fff4a749e6b4b9cad3a211`

## Fail-fast记录与GPU占用

1. 第一次save因optimizer含bf16 tensor而在NumPy hash处失败；修复为raw-byte hash并加入bf16/0-D回归测试。无artifact manifest的失败副本未被接受。
2. 第二次cold parity暴露fresh process缺少deterministic CUDA flags；补齐后exact parity通过。随后将本地dataset Arrow materialization与Hub/model cache严格区分。
3. ForceVLA `task2_seed42` 的同名tmux session在P8期间被外部重新创建两次并各占约22.1 GiB。依据用户“暂时不跑ForceVLA”的指示，只停止了确切的`forcevla_task2_train` session；没有删除ForceVLA日志或checkpoint。

两个失败checkpoint和手动diagnostic的隔离`/tmp` cache合计约19.6 GiB。由于删除未获显式批准且不可恢复，清理请求被安全策略拒绝；它们被保留，最终通过checkpoint不受影响。

## 剩余 blocker 与 gate

1. trusted detached signature算法、key id、签名值、批准人/角色/时间和verifier尚未冻结；formal checkpoint acceptance继续fail-closed。
2. development checkpoint没有wheelhouse binding；正式可复现供应链仍需冻结wheelhouse manifest/hash。
3. threshold/provenance正式批准仍未完成；本checkpoint不得用于正式评测或Shadow acceptance。
4. P9仅允许纯离线record/replay Shadow，尚未实现；进入P9前仍需用户明确批准。
