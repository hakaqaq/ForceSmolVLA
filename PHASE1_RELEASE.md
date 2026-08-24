# ForceSmolVLA Phase 1 — Offline Force-Conditioned Actor

Release label: `phase1-v0.1.0`

This release is the first-stage implementation of ForceSmolVLA. It combines a pinned
SmolVLA/LeRobot backbone with post-VLM force fusion, a four-expert capacity-free Top-1
MoE, and the Action-Query Force Residual Adapter. The offline actor is trained end to end
with a single joint forward/backward update path.

## Included

- LeRobot v0.6.0 pinned at commit `30da8e687a6dfc617fcd94afc367ac7071c376ce`.
- SmolVLA base revision `d5ef92b547b2bf36bdd50f18ea6ed6463cb5c5af`.
- Available-sensor v4.1 data conversion to LeRobot v3.
- Cartesian 7D action target construction and train-only normalization.
- Dense, MoE, additive, cache/parity, checkpoint/reload, and offline replay paths.
- GPU-only full-parameter offline SFT entry point and model-only inference server.
- Development tests and P4–P9 development evidence.

## Scope boundary

Phase 1 covers the offline full-parameter Force-conditioned Actor. It does not claim a
completed online Actor–Critic/HIL training stage or formal production acceptance. The
current real-data results and local checkpoints remain `development_only`.

Robot transport and HIL-SERL controller integration live in the separate
`fr3_client_ws` project. They are not part of this model repository.

## Deliberately excluded from GitHub

- datasets and raw robot recordings;
- base model weights and trained checkpoints;
- Hugging Face/pytest/Python caches;
- machine-local review bundles, logs, and trust keys.

Clone with the pinned LeRobot dependency:

```bash
git clone --recurse-submodules <repository-url>
cd ForceSmolVLA
conda env create -f environment.yml
conda activate forcesmolvla
python -m pip install -e vendor/lerobot
python -m pip install -e '.[test]'
```

The architecture and acceptance source of truth is
`ForceSmolVLA_Implementation_Spec_v4_2.md`.
