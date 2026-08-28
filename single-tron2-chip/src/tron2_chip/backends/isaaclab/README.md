# Isaac Lab backend status

This directory defines the simulator-independent contract needed by the future
Ubuntu Isaac Lab environment. It intentionally does not import Isaac Lab during
normal MuJoCo use.

Implemented now:

- fixed-base task timing and perturbation/compliance ranges;
- batched hindsight and deployment actor-goal tensor operations;
- a training hand-off script that emits `policy_spec.json` and a task contract;
- an explicit dependency check preventing accidental installation into
  `croco_env`.

Still required in the pinned Ubuntu environment:

1. validate/import the `WFYG_TRON2A` USD articulation;
2. register Isaac Lab observation, action, reward and event managers;
3. bind the ground-truth force only to the critic and hindsight training term;
4. connect RSL-RL PPO and export the policy plus normalizer to ONNX.

The script fails deliberately after validation until those simulator-specific
items exist; it must not silently claim to train a policy.

