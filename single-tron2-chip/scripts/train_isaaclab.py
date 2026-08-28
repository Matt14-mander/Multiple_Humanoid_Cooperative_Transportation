"""Validated hand-off point for the future Ubuntu Isaac Lab task registration."""

import argparse
import json
from pathlib import Path

from tron2_chip.backends.isaaclab.task_cfg import FixedBaseChipTaskCfg, require_isaaclab
from tron2_chip.core.policy_spec import PolicySpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/fixed_base_chip"))
    args = parser.parse_args()
    task = FixedBaseChipTaskCfg(num_envs=args.num_envs)
    task.validate()
    spec = PolicySpec.fixed_base_arm(
        history_steps=task.history_steps, control_dt_s=task.control_dt_s
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    spec.save(args.artifact_dir / "policy_spec.json")
    (args.artifact_dir / "task_contract.json").write_text(
        json.dumps(task.__dict__, indent=2), encoding="utf-8"
    )
    require_isaaclab()
    raise RuntimeError(
        "Isaac Lab is installed, but the WFYG_TRON2A articulation and manager-based "
        "environment are not registered yet. Import the validated USD on Ubuntu, then "
        "bind this task contract to the observation, action, reward and event managers."
    )


if __name__ == "__main__":
    main()

