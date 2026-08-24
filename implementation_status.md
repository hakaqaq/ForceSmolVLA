# ForceSmolVLA implementation status

Release: `phase1-v0.1.0`  
Status: offline development implementation complete; formal acceptance not claimed.

## Phase 1 complete

- SmolVLA/LeRobot is pinned and integrated as the model backbone.
- Post-VLM force fusion, Dense/MoE refiners, and the Action-Query Force Residual
  Adapter are implemented.
- Offline SFT uses one joint forward/backward update with the full action-generation
  path trainable.
- LeRobot v3 conversion, 7D Cartesian action targets, train-only normalization,
  checkpoint/reload, cached inference, and offline replay are implemented.
- A local task2 development run completed 10,000 B4x1 optimizer updates. Its dataset,
  base assets, and r5 checkpoint are intentionally excluded from GitHub.

## Evidence boundary

P4-P9 development gates were completed during implementation. Later inference and
task-scope fixes changed files covered by the historical source bindings. The functional
release suite passes 165 tests; five historical gate-revalidation checks now fail closed
until the gated evidence chain is rerun. The old hashes are retained only as historical
development evidence and are not presented as acceptance of the current release tree.

All local experiment artifacts remain `development_only` and `formal_eligible=false`.

## Not part of Phase 1

- online Actor-Critic or HIL fine-tuning;
- production/formal Shadow acceptance;
- robot transport/controller code, which remains in `fr3_client_ws`;
- public distribution of robot data, model weights, or machine-local trust material.
