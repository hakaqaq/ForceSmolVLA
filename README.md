# ForceRFT End-to-End User Guide

This guide describes only the final, fixed pipeline:

```text
ForcePrior SFT
-> manual reward annotation
-> reward classifier training
-> build a frozen base policy + zero wrist-wrench residual Actor + random Twin-Q bootstrap
-> collect genuine sealed ACK replay
-> after collecting 1,000 valid autonomous policy TD transitions from at least three episodes, run a 256-step Critic warm-up
-> run two Twin-Q updates + one wrist-wrench residual Actor attempt per online cycle
```

## 1. Directory Layout and Environment

```bash
export FORCERFT_ROOT="$(pwd)"
export FR3_WS=/home/rlc123/fr3_client_ws
export TASK_ID=task2
export TASK_OUTPUT_ROOT="$FORCERFT_ROOT/outputs/$TASK_ID"
export ONLINE_CAPTURE_ROOT="$FORCERFT_ROOT/datasets/${TASK_ID}_forcerft_online"
export RAW_ROOT="$FR3_WS/datasets/$TASK_ID"
export LEROBOT_DATASET="$FORCERFT_ROOT/datasets/${TASK_ID}_lerobotv3"
export MODEL_PYTHON="$(command -v python)"
export ROBOT_PYTHON="$FR3_WS/.venv/bin/python"
```

The training machine uses Conda, PyTorch, and CUDA. The robot side uses ROS 2 Humble. The hardware stack consists of an FR3, HEX-E, Robotiq gripper, D435, D405, and SpaceMouse. Before running the robot, load:

```bash
source /opt/ros/humble/setup.bash
source "$FR3_WS/install/setup.bash"
source "$FR3_WS/.venv/bin/activate"
export ROS_DOMAIN_ID=30 ROS_LOCALHOST_ONLY=0
```

The directory convention is fixed: converted raw data goes to `datasets/{task_id}_lerobotv3`, training outputs go to `outputs/{task_id}`, and task configuration lives at `configs/forcerft/tasks/{task_id}.yaml`. All CLIs accept `--task-id` and explicit path overrides.

## 2. Native Data Collection

The native recorder entry point comes from the robot workspace and is not part of this repository:

```bash
"$ROBOT_PYTHON" "$FR3_WS/scripts/record_franka_hilserl_impedance.py" \
  --root "$RAW_ROOT" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episodes 10 \
  --episode-time 120 \
  --tool-profile onrobot_robotiq
```

In the recorder interface, press `Enter` to save, type `d` and press Enter to discard, or type `q` and press Enter to stop. Only episodes that pass integrity checks enter `episodes/`. Failed raw recordings enter `rejected_episodes/`; do not fabricate ACKs, gripper goals, or images to bypass validation.

Each accepted episode retains dual-camera JPEGs, state7, action7, calibrated TCP wrench6, accepted references, pose ACKs, gripper target/status/state, controller state, and `episode_result.json`. Formal online replay reads these files by absolute path, so raw episodes referenced by replay must not be moved or deleted.

## 3. Convert to LeRobot v3

```bash
"$MODEL_PYTHON" tools/convert_franka_raw_to_lerobot_v3.py \
  --raw-root "$RAW_ROOT" \
  --output-root "$LEROBOT_DATASET" \
  --repo-id "${TASK_ID}_lerobotv3" \
  --development-only \
  --runtime-spec "configs/converter_runtime_spec.${TASK_ID}.json"
```

The converted dataset must contain both camera streams, state7, wrench6, action7, episode/frame indices, a split, a conversion manifest, and a normalizer manifest. Normalization is applied exactly once at the training input; already normalized data must not be normalized again.

## 4. Full ForcePrior SFT

```bash
export PYTHONHASHSEED=42 CUBLAS_WORKSPACE_CONFIG=:4096:8
"$MODEL_PYTHON" tools/train_forceprior_sft.py \
  --dataset "$LEROBOT_DATASET" \
  --config "configs/train/${TASK_ID}.json" \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT"
```

Final SFT checkpoint:

```text
outputs/{task_id}/sft/checkpoints/forceprior_sft_step_010000
```

It contains the Actor, optimizer/scheduler, RNG state, sampler state, and runtime manifests. To resume training, use this CLI's `--resume` option; do not load only `model.safetensors` and reconstruct the optimizer.

## 5. Manual Reward Annotation and Classifier Training

First select a subset of episodes for manual annotation, then generate the review bundle and frame labels. You do not need to label every collected episode. An initial version may start with approximately 20 episodes, but the training and validation episodes must be disjoint and must cover positive, ordinary-negative, and hard-negative examples. Episodes absent from the reviewed labels are excluded from reward-classifier training and validation. Once frozen, the reward detector still scores the complete LeRobot dataset automatically.

```bash
"$MODEL_PYTHON" tools/reward_classifier/label_reward_frames.py \
  --task-id "$TASK_ID" \
  --dataset-root "$LEROBOT_DATASET" \
  --train-episodes 16 \
  --val-episodes 4
```

Build the classifier cache and train:

```bash
"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --reviewed-labels "$FORCERFT_ROOT/labels/${TASK_ID}_reward_frame_labels.json" \
  prepare-cache --cache-dir /absolute/path/to/reward_cache

"$MODEL_PYTHON" tools/reward_classifier/train_reward_classifier.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --reviewed-labels "$FORCERFT_ROOT/labels/${TASK_ID}_reward_frame_labels.json" \
  train --cache-dir /absolute/path/to/reward_cache

"$MODEL_PYTHON" tools/reward_classifier/calibrate_reward_detector.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --cache-dir /absolute/path/to/reward_cache
```

The production detector checkpoint is fixed at:

```text
outputs/{task_id}/reward_classifier/checkpoints/best/best_checkpoint.msgpack
```

## 6. Build the Online ACK-Residual Bootstrap Checkpoint

```bash
"$MODEL_PYTHON" tools/build_forcerft_online_residual_bootstrap.py \
  --task-id "$TASK_ID" --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --frozen-base-policy-checkpoint \
    "outputs/$TASK_ID/sft/checkpoints/forceprior_sft_step_010000"
```

Output:

```text
outputs/{task_id}/online_ack_residual_filter_leash/bootstrap_checkpoints/base_policy_zero_residual_filter_leash_random_twin_q
```

The bootstrap checkpoint stores the frozen base-policy path, a strictly zero-output wrist-wrench residual Actor, random proposal-space residual Twin-Q networks, target networks, both optimizers, and runtime counters. It does not read demonstrations, an old offline Critic, or an old accepted-Q checkpoint.

## 7. Autonomous Proposal Critic Warm-Up and Residual Actor-Critic Training

The learner progresses through `ack_replay_collection -> ack_critic_warmup -> residual_actor_critic_training`. Neither the Actor nor Twin-Q is updated until the replay contains at least 1,000 valid autonomous policy TD transitions from at least three formally admitted episodes. Once both thresholds are met, the same process runs a one-time, 256-step Twin-Q warm-up and then executes joint cycles under the cumulative `floor(unique_policy_td_rows/8)` credit limit. Each cycle contains exactly `2 Twin-Q + 1 wrist-wrench residual Actor attempt`. Human records may provide BC supervision but do not enter warm-up, joint TD updates, or credit accounting. Training reads only low-dimensional state, wrench, base, proposal, and ACK data; it does not run a second base policy, Flow sampler, or image Critic. Twin-Q receives a 26-dimensional context (state7, wrench6, wrench increment6, base TCP6, and base gripper1) plus a six-dimensional policy proposal, for 32 dimensions total. The Residual Actor retains its 25-dimensional input and outputs only TCP6.

The Critic behavior action is the recorded pre-dispatch `applied_residual_tcp6` policy proposal. It must align with the same decision anchor, Actor revision, base/composed target, dispatch lineage, and genuine ACK. It must never be reconstructed from ACK-minus-base, replaced with zeros, or recomputed with the current Actor. Actor-Q evaluates the current Actor proposal directly, and target-Q evaluates the target Actor proposal at the genuine successor context. None of the three Q paths transforms the proposal value into an accepted residual. The genuine ACK-minus-base residual remains unchanged for execution diagnostics.

Removing the lower-level filter/leash mirror from the Q action value does not bypass the controller during real execution. The actual adapter, filter, leash, workspace limits, force/torque limits, and ACK chain remain active. Actor-Q checks a candidate in the saved upper-level dispatch/profile guard context for the current row, while target-Q checks a candidate in the genuine next-decision guard context. Invalid candidates skip the corresponding value or TD term, and an unknown guard context is reported separately. Genuine `beta=0` terminal or truncated rows use only `y=r` and do not query the target Actor, target Q, or next guard. Missing lower-level filter mirror state alone no longer excludes proposal-Q, but a missing genuine proposal, ACK, anchor-aligned base/composed target, valid successor, or required upper-level guard still excludes the corresponding training use. This candidate-eligibility restriction is an implementation detail and does not imply that the 32-dimensional context is a complete Markov state. Identical context/proposal pairs may still produce different accepted actions under different filter histories.

## 8. HIL and Online Replay

`tools/serve_forcerft_residual_actor_critic.py` is the sole GPU owner, and `tools/run_forcerft_integrated_capture.py` is the sole robot-control chain. The current chunk/slot mapping works as follows: real dispatch `n` uses the currently adopted chunk `i(n)`, then selects `j(n)=ceil(30(selection_ns-t_ref_ns))` on the 30 Hz rational timebase using that chunk's `t_ref_ns` and the dispatch selection time. A slot beyond the H50 cache is discarded and replanned. A model delta generated at the request pose is first restored to an absolute chunk; the selected slot is then re-expressed at the genuine decision pose for the Residual Actor. A newly completed asynchronous chunk takes effect only after lineage validation. At most eight dispatches are issued from one chunk, and the current CLI replan/low-watermark values are 8/7. Takeover invalidates the old chunk and pending request; release requires a fresh observation and fresh inference. Only a decision that is actually sent and receives a controller ACK may form a successor. HOLD or rejected commands never fabricate transitions.

If the wrench causal filter resets because of a source gap and changes generation during inference, the old request/result and any unexecuted chunk are invalidated. After the existing 250-sample warm-up completes, the same episode resumes with a fresh observation and fresh inference. No transition is generated while waiting for recovery.

After an episode is sealed, the operator enters success or failure. Technically complete success and failure episodes are both materialized into TD transitions by one production-bridge admission call and appended to:

```text
outputs/{task_id}/online/replay
```

Admission is no longer preceded by a duplicate dry run. A detector terminal for success remains `reward=1.0, terminated=true, bootstrap_mask=false, discount=0.0`. For failure, the final valid Critic transition in the sealed episode becomes a zero-reward terminal and likewise does not bootstrap. If an operator failure conflicts with a frozen-detector success trigger, the entire episode is excluded from training.

For task2 seal materialization, the 30 Hz causal grid allows a dual-camera sample age of at most `100 ms` to cover normal scheduling jitter and occasional dropped frames. Dual-camera skew must still be at most `33 ms`; a sample older than `100 ms` is still rejected from replay. Provenance stored in historical checkpoints is not rewritten.

Formal sealed online ACK records first pass authenticity, source, identity, and seal validation, after which Critic TD and Actor BC eligibility are determined independently. Only autonomous policy rows with a trustworthy genuine proposal, anchor-aligned base/composed targets, a genuine ACK, and either a valid terminal boundary or a direct same-segment policy successor may set `critic_td_valid=true`. Human rows explicitly use `critic_td_valid=false`; when they contain a reliable pre-takeover base and human target, they may still set `human_residual_valid=true` and participate in BC. Autonomous rows without a successor and human rows without a reliable base or target retain their raw contents and exclusion reasons. The system does not fabricate a terminal, proposal, or supervision label.

`K=3` is retained only as a storage and ACK-validation adapter for one genuine dispatch. It does not count as three decisions or three credits. The three slots must agree on dispatch, ACK, chunk, model index, command, and accepted action; otherwise silent compression is rejected. The Critic no longer uses mutable human `control_source`; both current and next contexts use the frozen base gripper.

Only a human row with a reliable pre-takeover base action provides a Residual Actor supervision target. Without that baseline, only `human_residual_valid=false` is set; the episode is not rejected. Human orientation differences continue to use RPY deltas. The BC target is projected separately into the Residual Actor output range, while the original ACK and behavior residual remain unchanged. At takeover, the old chunk, pending request, and old observation are cleared; policy dispatch pauses during takeover; and release requires a fresh observation and fresh inference.

Takeover, release, and reset remain credit-truncation boundaries. Q is therefore a proxy for discounted terminal-success feedback for an autonomous policy proposal within the current control segment, not an unbiased estimate of full-episode autonomous success. In `autonomous A -> human B -> autonomous C -> success`, success propagates only through C's policy TD chain, while B provides bounded BC only. If a human always completes the last step, autonomous Q does not receive a fabricated positive terminal example; early changes may be driven mainly by BC.

## 9. Continuous Online Actor/Learner

```bash
"$MODEL_PYTHON" tools/run_forcerft_online_loop.py \
  --task-id "$TASK_ID" \
  --output-root "$TASK_OUTPUT_ROOT" \
  --dataset-root "$LEROBOT_DATASET" \
  --max-episodes 100 \
  --capture-output-root "$ONLINE_CAPTURE_ROOT" \
  --ack-replay-root "$TASK_OUTPUT_ROOT/online_ack_residual_filter_leash/formal_replay" \
  --online-residual-bootstrap-checkpoint \
    "$TASK_OUTPUT_ROOT/online_ack_residual_filter_leash/bootstrap_checkpoints/base_policy_zero_residual_filter_leash_random_twin_q" \
  --task "Pick up the purple ring and place it onto the red peg." \
  --episode-time 120 \
  --tool-profile onrobot_robotiq \
  --policy-replan-steps 8 \
  --policy-queue-low-watermark 7 \
  --max-force-n 25 \
  --max-torque-nm 2 \
  --allow-development-policy-execution-smoke
```

Online native episodes are always stored under the repository data directory. For example, the first session is `$FORCERFT_ROOT/datasets/{task_id}_forcerft_online_001`; it is not written to the robot workspace data directory. Omitting `--capture-output-root` uses the same repository-local default.

At every start, the unified server restores one Residual Actor/Twin-Q checkpoint and launches an independent learner worker immediately after validating checkpoint and replay integrity. Only after the replay contains at least 1,000 unique, formally admitted autonomous policy transitions materialized with `critic_td_valid`, drawn from at least three formal episodes, does it run the one-time 256-step Twin-Q warm-up and enter joint training. Human BC rows do not enter the Critic and do not increase the TD row count, contributing-episode count, or cycle credit. Every completed joint learner cycle contains two Twin-Q optimizer updates, two corresponding target Polyak updates, and one Residual Actor update attempt. An Actor skip due to missing valid support still completes and consumes that cycle. Warm-up is not counted as a joint cycle. Training neither reads demonstration images nor runs a second base policy or Flow sampler.

The learner and recorder run concurrently. While an episode is executing, Critic updates and Actor attempts may continue whenever the inference-priority coordinator leaves a suitable gap. The training Actor and the Actor pinned for execution are separate instances. Joint training uses the cumulative credit limit `allowed_cycles=floor(unique_policy_td_rows/8)`; completed cycles and one in-flight cycle both consume credit. New human BC does not add credit. Without new policy TD data, wall-clock time, operator-confirmation waits, and repeated sampling do not add credit either. The formal admission HTTP confirmation only validates the committed manifest and wakes the learner. Credit is registered only after replay refresh actually materializes and filters policy TD rows. When credit is exhausted or no currently valid update sample exists, the learner waits on a cancellable event while the collector proceeds directly to the next episode without a drain or foreground join.

An asynchronous capture manifest may cover startup-data waiting, Critic warm-up, credit waiting, joint training, or partial Q-cycle progress. `current_episode_sampled_by_learner=false` must be established jointly by replay membership and actual batch provenance. The active, uncommitted, unconfirmed, rejected, and discarded episodes must not enter replay.

After each episode, the canonical online loop prints only two capture/learner summary lines and one admission summary line. The full contract, stream-quality report, and episode seal remain in session files and are not expanded repeatedly in the terminal.

At startup, the loop first selects the highest-cycle structurally complete exact-resume checkpoint under `outputs/{task_id}/online_ack_residual_filter_leash/training_checkpoints/`. A checkpoint may instead be selected explicitly with `--learner-resume-checkpoint`. If no resumable checkpoint exists, an explicit `--online-residual-bootstrap-checkpoint` is required. Model structure, optimizers, losses, residual bound, batches, schedule, proposal-Q mode, policy-only TD mode, and candidate-guard mode all take `state/config.yaml` in the checkpoint as their sole authority. Old accepted-Q, human-TD, 60-dimensional Critic, or old-bounds checkpoints are not restored, and their Q networks, optimizers, warm-up progress, cycles, and credit ledgers are not migrated.

A normal resume under the new schema still requires an exact configuration match. Proposal-Q input, the policy-only TD ledger, scalar bound, losses, and Actor learning rate are algorithm state. An old checkpoint or bootstrap cannot be migrated or silently combined with a new YAML configuration; build a new zero-residual bootstrap from the frozen SFT prior.

`--allow-development-policy-execution-smoke` is the existing explicit robot-execution switch. It does not select a model or trigger publication, activation, candidate, profile, or binding flows. Force limits, takeover generation, stale-result rejection, ACK handling, and the recorder's single control chain remain unchanged.

Online inference applies finite saturation only to the denormalized gripper candidate: values below `-0.01 m` are treated as the closed end and values above `0.095 m` as the open end. The binary threshold remains `0.0425 m`, after which the output is exactly either `0.0 m` or `0.085 m`. `NaN/Inf` values are still rejected. TCP6, force limits, and the action normalizer are neither clipped nor rewritten.

The Residual Actor uses the scalar form from the paper: first compute `axis_caps=min(0.1, physical_limit6/delta_action7.std[:6])`, then set `c_res=min(axis_caps)`, and finally use the same value on all six axes as `proposal6=tanh(logits6)*c_res`. All six normalized caps are therefore identical. Each physical proposal bound is this shared scalar multiplied by the corresponding frozen normalizer standard deviation, and it does not exceed the configured limit of 1 mm for translation or 0.5 degrees for RPY. This is not a bound on measured TCP motion, total rotation angle, or an entire trajectory. The online Actor, target Actor, CPU inference path, and human-BC projection share this scalar bound. The genuine ACK-minus-base behavior residual retains its original value even when it exceeds the bound.

Completed joint learner cycles are the sole publication and periodic-checkpoint clock. Residual Actor candidates are published after cycles 100, 200, 300, and so on; full exact-resume checkpoints are saved after cycles 1,000, 2,000, 3,000, and so on. Publication does not depend on whether the Actor optimizer was applied in that cycle. A publication event is still retained when parameters did not change, although the same weight blob may be reused. A candidate records the policy TD/admission coverage available at export, actual policy/human sample draws, scalar-bound/loss/Q contracts, and a lightweight proposal probe over deterministic low-dimensional states from every loaded episode. This probe is not an autonomous-success evaluation. A candidate whose own training data does not cover at least 1,000 policy TD rows from three contributing episodes cannot be activated, and later replay growth cannot retroactively qualify it.

Publication and execution activation are separate events. The execution Actor remains pinned throughout an episode. If several candidates are produced during one episode, only the newest pending candidate is retained, and it is activated atomically at the next existing episode boundary. Activation no longer triggers a full training checkpoint. Status reports distinguish the current learner cycle, the last published cycle, and the active Actor's publication cycle, revision, and policy epoch.

Examples of periodic full checkpoints:

```text
outputs/{task_id}/online_ack_residual_filter_leash/training_checkpoints/residual_actor_critic_cycle_001000
outputs/{task_id}/online_ack_residual_filter_leash/training_checkpoints/residual_actor_critic_cycle_002000
```

Only the newest ten retention-managed full checkpoints are kept. Critic warm-up completion and graceful/operator exit are explicit exceptions that may save at a non-thousand cycle. Candidate publication and activation are not exceptions. At cycle 1,000, the system first completes publication state and then saves one unified training snapshot. The checkpoint's models, targets, optimizers, counters, watermarks, and active/pending Actor bindings therefore come from one safe boundary.

Use `q` in the recorder for a normal stop. The system first ends the capture/Actor window, cancels new learner scheduling, and then stops and idempotently saves the current consistent state at a safe boundary. It does not wait for continuous training to "finish" or reach the next 100/1,000-cycle boundary. Replay/inference coverage waits in the coordinator can be awakened by stop. If shutdown occurs between the two Q updates, `partial_cycle_q_updates` records the partial progress rather than reporting a complete cycle. A learner failure never modifies the raw episode or adds an unsealed episode to replay.

If capture exits because of a controller, communication, or process error, the canonical online loop automatically deletes the current unsealed session root and its `.inprogress` contents, leaving no partial episode. A session with an existing technical seal is not deleted automatically, even if later admission fails, so it can be retried after repair.

`--max-episodes 1` saves the progress actually obtained from that single episode and exits without draining old credit merely to manufacture updates. To observe genuine overlap between collection and learning, collect several episodes continuously, for example with `--max-episodes 3`. CPU tests validate scheduling and concurrency semantics only; they do not represent GPU inference latency or real-robot throughput.

Example status and log fields (illustrative only, not results from an actual run):

```text
completed_learner_cycles=1200 partial_cycle_q_updates=0
total_twin_q_optimizer_steps=2656 warmup_twin_q_optimizer_steps=256
residual_actor_update_attempts=1200 residual_actor_optimizer_steps=1194
residual_actor_updates_skipped_no_gradient=6
last_published_cycle=1200 last_periodic_checkpoint_cycle=1000
last_checkpoint_cycle=1000
active_actor_online_cycle=1100 pending_actor_revision=task3-residual-policy-cycle-001200-g...
capture_window.delta.completed_learner_cycles=37
capture_window.current_episode_sampled=false
learner_wait_reason=insufficient_action_coverage learner_wait_ms=18.4
```

## 10. Retention and Failure Handling

This repository does not track runtime data, models, checkpoints, logs, or development audit outputs. In a real experiment, back up the raw demonstration data and LeRobot v3 dataset, SFT checkpoint, reward classifier, current replay, newest ten online exact-resume checkpoints, and every raw episode referenced by formal replay, WAL, outbox, or admission records.

Common fail-closed causes include an incomplete exact-resume checkpoint, different origins for the inference Actor and learner checkpoint, missing raw JPEGs, a stale result after takeover, incomplete gripper origin, a missing ACK, or inconsistent checkpoint/replay UIDs or credit. Never bypass these checks with images from another episode, fabricated command IDs or ACKs, rebinding an old generation, or modified raw episodes.
