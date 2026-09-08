---
type: "query"
date: "2026-09-08T04:47:28.705086+00:00"
question: "侧键本身不会打断policy对吧，控制夹爪是否必须保持SpaceMouse机械臂接管"
contributor: "graphify"
outcome: "useful"
source_nodes: ["GripperToggleAuthority", "TakeoverWindow", "_toggle_authority()"]
---

# Q: 侧键本身不会打断policy对吧，控制夹爪是否必须保持SpaceMouse机械臂接管

## Answer

Expanded from original query via vocab: [space, gripper, takeover, authority, toggle, accepted, measured, feedback, pending, cancel, human, policy]. 侧键本身会发起人工接管：第一次按下设置 pending takeover toggle 和 human gripper lease，并发布 TCP6 全零且 intervention=true 的人工意图，因此 policy 会被打断。无需移动 SpaceMouse，也无需一直按住侧键。若 accepted authority 缺失且测量反馈陈旧或处于开闭阈值之间，夹爪目标保持 pending；第二次按键会取消 pending。

## Outcome

- Signal: useful

## Source Nodes

- GripperToggleAuthority
- TakeoverWindow
- _toggle_authority()