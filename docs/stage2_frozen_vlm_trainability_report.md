# Stage-2 Frozen-VLM Trainability Preflight

Status: **PASS**. This append-only contract freezes the visual-language prefix owner and keeps the Force/Action path trainable.

- Frozen parameters: 350,196,864
- Trainable Actor parameters: 155,423,477
- Trainable Twin-Q parameters: 8,584,194
- Frozen-VLM forward parity: exact (max abs error 0)
- Frozen parameter/buffer hash after temporary Critic+Actor updates: unchanged
- Prefix representation/cache: detached; Force K/V projection: once per chunk
- Vision/SmolVLM/state-prefix gradients: exact zero
- Force/Action Expert/Action I/O/router gradients: nonzero
- TCP6 Q gradient: 0.014094896
- Gripper Q gradient: 0.0
- Gripper FM gradient: 0.0038615442

The temporary updates were discarded. No checkpoint, long-run, evaluation, public-path modification, or robot execution was created. Existing full-Actor G7-B remains historical mechanics evidence only.
