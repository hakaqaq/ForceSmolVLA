#!/usr/bin/env python3
"""Run the frozen G3 gate with the append-only ActionContract-v2 adapter."""

from forcesmolvla.rft import flow_sampling
from forcesmolvla.rft.critic_action_adapter_v2 import critic_action_for_q_guidance_v2


flow_sampling.critic_action_for_q_guidance = critic_action_for_q_guidance_v2

from preflight_s2_differentiable_flow_gpu import main  # noqa: E402


if __name__ == "__main__":
    main()
