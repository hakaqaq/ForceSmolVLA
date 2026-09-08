---
type: "query"
date: "2026-09-03T03:50:50.830992+00:00"
question: "PermissionError: DEPLOYMENT_BINDING_SERVER_SOURCE_SHA256_MISMATCH"
contributor: "graphify"
outcome: "useful"
source_nodes: ["load_deployment_profile()", "load_deployment_binding()", "serve_policy.py"]
---

# Q: PermissionError: DEPLOYMENT_BINDING_SERVER_SOURCE_SHA256_MISMATCH

## Answer

Expanded from the request via graph vocab: [deployment, binding, server, source, sha, mismatch, policy, profile, load, checkpoint, trusted, inference]. DFS located load_deployment_profile() and load_deployment_binding() in serve_policy.py. The profile trust anchor ab02d710... matches the binding file; model 74f58a0c... and rulespec d3a83b6f... also match. The binding expects server source closure 19945b69..., while the current working tree computes 950a8dc9.... A pure in-memory recomputation using the HEAD version of src/forcesmolvla/training_runtime.py and all other current closure files exactly reproduces 19945b69..., proving that post-binding edits to training_runtime.py caused the fail-closed mismatch.

## Outcome

- Signal: useful

## Source Nodes

- load_deployment_profile()
- load_deployment_binding()
- serve_policy.py