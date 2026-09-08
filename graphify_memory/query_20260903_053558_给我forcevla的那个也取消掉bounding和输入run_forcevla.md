---
type: "query"
date: "2026-09-03T05:35:58.345150+00:00"
question: "给我forcevla的那个也取消掉bounding和输入RUN FORCEVLA"
contributor: "graphify"
outcome: "useful"
source_nodes: ["deployment_binding.py", "client_source_sha256", "ForceVLAFlowMatching"]
---

# Q: 给我forcevla的那个也取消掉bounding和输入RUN FORCEVLA

## Answer

Expanded from original query via graph vocab: [authorization, binding, client, confirmation, deploy, execute, execution, forcevla, live, robot]. Removed RUN FORCEVLA and the --yes branch from deploy_forcevla.py. Excluded both deploy_forcevla.py and deploy_forcesmolvla.py from CLIENT_SOURCE_FILES in the canonical server module and client mirror. Refreshed the binding once; ForceVLA self-test, binding validation, and 14 inference tests pass. Runtime control bounds were unchanged.

## Outcome

- Signal: useful

## Source Nodes

- deployment_binding.py
- client_source_sha256
- ForceVLAFlowMatching