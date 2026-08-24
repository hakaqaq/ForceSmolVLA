# P7 implementation/preflight report

> Historical notice（2026-08-20）：本报告记录 v4.1 修订前的 development gate。v4.2 已将 exact two-pass 限制为短程 acceptance oracle，并要求 active SFT/best validation 使用 single-pass；本报告及其 source hashes 不得作为当前 acceptance 证据。

日期：2026-08-19（Asia/Shanghai）  
状态：`DEVELOPMENT_GATE_PASS / P8_NOT_STARTED`  
acceptance：`development_only`，`formal_eligible=false`

## 结论

P7 的训练 recipe、精确两遍全局 router loss、全参数 optimizer 分组、可序列化 uniform sampler、固定 validation fixture，以及 parameter-matched Additive 均已实现并通过真实 RTX 4090 D gate。86 项测试全部通过。

真实 gate 使用 task1 v4.1 train split 的16个真实 tuple，固定为双相机、`H=50`、`batch_per_gpu=2`、8个microbatch、ForceToken-MoE、bf16 autocast和全参数offline finetuning。Pass A/B router probability最大重放误差为0；8次backward后只执行1次optimizer step和1次scheduler step。没有CPU训练fallback、冻结VLM、LoRA或架构降级。

全部P7工件均为development-only，不得用于正式checkpoint acceptance、正式评测或Shadow acceptance。P8尚未开始。

## 实现

- `src/forcesmolvla/router_training.py`
  - Pass A在整个8-microbatch window、所有ranks上以`no_grad`累计`sum(p)`、detached top-1 route counts、`N_I`和`N_flow_global`，支持distributed all-reduce。
  - Pass B逐microbatch使用冻结的world-size-scaled `L_flow`、`L_balance`、`L_z`公式。
  - `N_I=0`返回graph-connected zero；Pass A/B mask、route或probability不一致会fail-fast。
  - 8次backward后只允许一次optimizer/scheduler step；Pass A/B之间参数变化会fail-fast。
  - AdamW no-decay精确覆盖bias、normalization scale、embedding和alpha；其余`weight_decay=1e-10`；所有参数必须`requires_grad=true`。
  - uniform eligible-chunk sampler可恢复eligible set、seed、Python RNG state和cursor。
- `src/forcesmolvla/modeling_forcesmolvla.py`
  - Pass A使用真正的VLM-only prefix路径，不构造action expert suffix。
  - Pass B以完全相同的prefix-only VLM/fusion/router路径带图重算ForceContext，再将该context注入完整suffix flow forward。因此router重放bitwise相同且VLM/vision保持全参数梯度。
- `src/forcesmolvla/force_token.py`
  - 主模型`Q=S+C`；Additive唯一改为`Q=C`。两者参数名、shape、count和初始化完全一致。
- `configs/p7_training_recipe.development.yaml`
  - 固定20,000 updates、1,000 warmup、pinned LeRobot cosine scheduler、B2×8、seeds 42/43/44、checkpoint interval 500、fixed validation `L_flow` selection和P8/P9边界。
- `tools/preflight_p7_two_pass_gpu.py`
  - CUDA-only RTX4090D；真实train split B2×8；输出source-bound resolved config和包含tuple/mask/epsilon/time/ChunkContext的固定validation fixture。

## CPU/接口测试

- 全套：`86 passed`。
- 覆盖：全局loss数值与梯度等价、N=0、DDP world-size公式、8 microbatch/单update、Pass A no-grad、mask/route/probability replay、sampler RNG/cursor、optimizer分组及scheduler step语义。
- Additive：参数名/shape/count/init tensor完全相同；`W_out=0`时native step-0 output精确相同；非零测试只允许Q是否含suffix的差异；invalid query严格为零。

## 真实 RTX 4090 D 结果

- GPU：NVIDIA GeForce RTX 4090 D，24,564 MiB nominal / 23.523 GiB binary。
- 参数：total/trainable=`505,620,341/505,620,341`，frozen=`0`。
- Force初始化SHA256：`7cc70cb564f039faffcbd5a8bea8b6b2e99896b9f45eec959a5772b31b8a65d1`。
- 真实输入：16 samples、8 microbatch、B2、双相机、H=50、5600 valid flow features。
- router：2368 valid tokens；route counts=`[263,95,2010,0]`；zero drop；Pass A/B probability max error=`0.0`。
- loss：backward flow sum=`28.4218909740`，balance=`2.44499492645`，z=`4.53946560621`，weighted total=`28.4508800507`。
- gradient norm before clip=`179.9482574463`；clip=`10`。
- base gradient：499/500 tensors有非零梯度；唯一无梯度是upstream allowlisted/unused `lm_head.weight`。
- Force gradient：70/74 tensors有gradient，44/74非零。expert3本window route count为0，因此其4个tensor无gradient；W_out零初始化使部分下游force tensors首步为零梯度。这些参数仍全部trainable，没有冻结或drop token。
- two-pass optimizer update：CUDA=`1952.898 ms`，wall=`1.953652 s`。
- 峰值显存：allocated=`7,207,315,456 bytes`=`6.7123 GiB`；reserved=`8,057,257,984 bytes`=`7.5039 GiB`。
- scheduler：update前LR=`9.99000999e-8`，update后LR=`1.99800200e-7`，`last_epoch=1`。

## Additive硬匹配

- 完整state tensor count=`574`。
- 完整tensor逐项精确相等：true。
- state schema SHA256=`41e8631783397160fabb3fa98981050babfe900ecb44a6c6f728827d94872f14`。
- total parameters均为`505,620,341`；optimizer groups完全相同。
- Force initialization SHA256相同。
- 相同val inputs/noise/time下，step-0 native `L_flow`：main/additive=`25.62472152709961/25.62472152709961`。

## 固定validation

- tuple list：val split固定2个tuple。
- 工件包含完整action masks、epsilon7 tensor/hash、time tensor/hash和ChunkContext/hash。
- development update后`L_flow` run1/run2=`25.60464859008789/25.60464859008789`，Python float精确相等。
- selection只允许global-valid-feature-token-weighted fixed validation `L_flow`，不得使用auxiliary loss选择checkpoint。

## 工件hash

- `p7_gpu_preflight.json`: `a92ab972b92c5aa5241c1f3d94da15679a620d5cf3f827171997618fc7ea406f`
- `p7_resolved_config.json`: `68ae8abd52621f0f0fbd7541811882389b42a7d8cd9b992e5d4d9bd97114eafe`
- `p7_validation_fixture.json`: `4237daa97c91e1f708a1d0377e4523c25c008bd0eeb544327c9feb720b9241cd`
- `p7_source_binding.json`: `ce7647dadeff2a26b2ed90b171bb20185ea0fa86bbd3ac9a8a258af4e4f53a71`
- `p7_training_recipe.development.yaml`: `c3c5b030dbcee7fe1838525611147142b8e8259115fe3ce937ff86adb09b78b6`

## GPU占用事件

用户明确授权终止同一ForceVLA `task2_seed42`训练。旧PID 2932038在操作前已经退出；核对新PID 2940773为同一命令后，以SIGTERM正常终止。未删除其日志或checkpoint。随后4090D计算进程清空，P7重新运行并通过。

## 剩余blocker与gate

1. P8 strict checkpoint save/reload、optimizer/scheduler/scaler/RNG/sampler/accumulation state恢复和cold-start parity尚未执行。
2. P9 offline record/replay Shadow尚未实现。
3. trusted detached signature和formal approvals仍为空。
4. 进入P8前仍需用户明确批准；本报告不自动授权P8。
