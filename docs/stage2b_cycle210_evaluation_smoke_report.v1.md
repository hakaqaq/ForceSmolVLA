# Stage-2B cycle210 evaluation-smoke export

Status: **PASS**. The full cycle210 Actor state was strictly overlaid on the unchanged r5 runtime/config and exported without Critic, optimizer, scheduler, RNG, or sampler payloads.

- Export: `artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1`
- Exported model SHA-256: `e24c1d6bb0a778921659514ac47c692b952178aa39af2601ccf0fc32bf94774d`
- Cycle210 source Actor SHA-256: `73b35435e943823bb88c54decf68ce4bf08f39100999c5770b071aa76c3cf4c3`
- Strict load: missing keys 0, unexpected keys 0
- Frozen VLM/state-prefix tensors equal r5: exact (347 tensors)
- Direct / exported-public / serve HTTP raw action parity: exact
- Direct / exported-public / serve HTTP public action parity: exact
- H=50, 7D, finite, binary gripper endpoints, invalid-tail masking: pass
- Existing deploy client accepted the complete HTTP action chunk: pass
- Evaluation-smoke binding: `artifacts/development/live/task2_cycle210_evaluation_smoke_binding.v1.json`
- Binding SHA-256: `6f15f33aedbf4327388012dc7a0418de09f05ba070833ac95c092b95104471d5`

No robot connection, training, online update, persistent service, deployment release, or policy-performance claim occurred. The evaluation binding remains scoped to one supervised smoke rollout pending physical confirmation.

## Commands for the later approved physical smoke

Server:

```bash
cd /home/rlc123/ForceSmolVLA
python tools/serve_policy.py \
  --host 127.0.0.1 \
  --port 8000 \
  --checkpoint /home/rlc123/ForceSmolVLA/artifacts/development/stage2/stage2b_cycle210_evaluation_smoke_checkpoint.v1 \
  --rulespec /home/rlc123/ForceSmolVLA/configs/live_action_safety.task2.development.yaml \
  --allow-development-robot-execution \
  --deployment-binding /home/rlc123/ForceSmolVLA/artifacts/development/live/task2_cycle210_evaluation_smoke_binding.v1.json \
  --trusted-deployment-binding-sha256 6f15f33aedbf4327388012dc7a0418de09f05ba070833ac95c092b95104471d5
```

Client (do not run until workspace/estop/operator confirmation):

```bash
source /opt/ros/humble/setup.bash
source /home/rlc123/fr3_client_ws/install/setup.bash
source /home/rlc123/fr3_client_ws/.venv/bin/activate
cd /home/rlc123/fr3_client_ws
python scripts/deploy_forcesmolvla.py \
  --host 127.0.0.1 \
  --port 8000 \
  --allow-development-robot-execution \
  --trusted-deployment-binding-sha256 6f15f33aedbf4327388012dc7a0418de09f05ba070833ac95c092b95104471d5 \
  --execute \
  --duration 120
```

Parity evidence digest: `8897416884801e3caa4fb86853feaf94579f4f850ceacb3c91439c618f677246`.
