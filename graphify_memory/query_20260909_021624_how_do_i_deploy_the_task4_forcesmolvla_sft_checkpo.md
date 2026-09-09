---
type: "procedural"
date: "2026-09-09T02:16:24.026201+00:00"
question: "How do I deploy the task4 ForceSmolVLA SFT checkpoint?"
contributor: "graphify"
outcome: "useful"
source_nodes: ["tools/serve_policy.py", "load_deployment_profile", "deployment_binding.py", "ForceSmolVLAPolicy"]
---

# Q: How do I deploy the task4 ForceSmolVLA SFT checkpoint?

## Answer

Use configs/deployment.task4_sft_step010000.development.json for both tools/serve_policy.py and fr3_client_ws/scripts/deploy_forcesmolvla.py. The profile selects outputs/task4/sft/checkpoints/forcesmolvla_sft_step_010000, task4 LeRobot manifest, raw session, tool profile, RuleSpec, and binding. Explicit trusted SHA256 CLI input is unnecessary because both processes default to the profile value.

## Outcome

- Signal: useful

## Source Nodes

- tools/serve_policy.py
- load_deployment_profile
- deployment_binding.py
- ForceSmolVLAPolicy