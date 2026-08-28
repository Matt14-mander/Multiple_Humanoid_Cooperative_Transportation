"""Run the fixed-base CHIP hindsight-goal wiring sanity check."""

import argparse
import csv
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .build_scene import build_scene
from .configuration import load_config
from .control import AnalyticGoalController
from .hindsight import compliance_matrix, hindsight_goal
from .model_index import ModelIndex
from .model_loader import load_model
from .paths import DEFAULT_CONFIG, GENERATED_MODEL, PROJECT_ROOT
from .perturbation import ForcePulse, apply_body_force


def _initialize(model, data, config):
    index = ModelIndex(model)
    for name, value in config["model"].get("initial_joint_positions", {}).items():
        data.qpos[index.joint(name).qpos_adr] = float(value)
    mujoco.mj_forward(model, data)


def run(config_path=DEFAULT_CONFIG, model_path=GENERATED_MODEL, headless=False,
        duration_s=None, rebuild=False, compliance=None, force=None,
        record_path=None, goal_mode="hindsight", quiet=False):
    config = load_config(config_path)
    if rebuild or not Path(model_path).exists():
        build_scene(config_path, model_path)
    model = load_model(model_path)
    data = mujoco.MjData(model)
    _initialize(model, data, config)
    if (model.nq, model.nv, model.nu) != (25, 24, 18):
        raise RuntimeError("Unexpected single TRON2 dimensions: {}".format((model.nq, model.nv, model.nu)))

    chip = config["chip"]
    compliance = chip["compliance_m_per_n"] if compliance is None else compliance
    force = chip["force_world_n"] if force is None else force
    pulse = ForcePulse(np.asarray(force), float(chip["pulse_start_s"]), float(chip["pulse_duration_s"]))
    duration = float(duration_s if duration_s is not None else config["run"]["duration_s"])
    controller = AnalyticGoalController(model, data, config["control"])
    ee_id = controller.ee_body_id
    reference = data.xpos[ee_id].copy()
    rows = []
    if goal_mode not in {"hindsight", "deployment"}:
        raise ValueError("goal_mode must be hindsight or deployment")

    def step():
        applied_force = pulse.at(float(data.time))
        actor_goal = (
            hindsight_goal(reference, applied_force, compliance)
            if goal_mode == "hindsight" else reference
        )
        apply_body_force(data, ee_id, applied_force)
        controller.update_cartesian_reference(data, actor_goal, compliance)
        max_control_fraction = controller.apply(data)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError("MuJoCo state became non-finite")
        rows.append([
            float(data.time), *data.xpos[ee_id], *reference, *actor_goal,
            *applied_force, float(np.max(np.abs(data.ctrl))), max_control_fraction,
        ])

    if headless:
        while data.time < duration:
            step()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < duration:
                started = time.perf_counter()
                step()
                viewer.sync()
                delay = model.opt.timestep - (time.perf_counter() - started)
                if delay > 0.0:
                    time.sleep(delay)

    output = Path(record_path) if record_path else PROJECT_ROOT / config["run"]["record_csv"]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([
            "time_s", "ee_x", "ee_y", "ee_z", "ref_x", "ref_y", "ref_z",
            "actor_goal_x", "actor_goal_y", "actor_goal_z", "force_x_n",
            "force_y_n", "force_z_n", "max_abs_ctrl", "max_control_fraction",
        ])
        writer.writerows(rows)

    expected_goal_shift = (
        hindsight_goal(reference, np.asarray(force), compliance) - reference
        if goal_mode == "hindsight" else np.zeros(3)
    )
    expected_response = compliance_matrix(compliance) @ np.asarray(force, dtype=float)
    displacement = data.xpos[ee_id].copy() - reference
    position_samples = np.asarray([row[1:4] for row in rows])
    peak_ee_offset = float(np.max(np.linalg.norm(position_samples - reference, axis=1)))
    if not quiet:
        print("completed: mode={} time={:.3f}s nq={} nv={} nu={}".format(goal_mode, data.time, model.nq, model.nv, model.nu))
        print("command: force_N={} C_m_per_N={} actor_goal_shift_m={} expected_response_m={}".format(
            np.array2string(np.asarray(force), precision=4),
            np.array2string(np.asarray(compliance), precision=4),
            np.array2string(expected_goal_shift, precision=5),
            np.array2string(expected_response, precision=5),
        ))
        print("motion: final_ee_displacement_m={} max_abs_ctrl={:.3f} csv={}".format(
            np.array2string(displacement, precision=5),
            max(row[-2] for row in rows), output.resolve(),
        ))
        print("control: peak_limit_usage={:.2f}%".format(100.0 * max(row[-1] for row in rows)))
        print("response: peak_ee_offset={:.6f}m".format(peak_ee_offset))
        print("note: analytic Cartesian impedance is a wiring surrogate; PPO CHIP policy is not trained yet")
    return {
        "model": model,
        "data": data,
        "reference": reference,
        "goal_mode": goal_mode,
        "expected_goal_shift": expected_goal_shift,
        "expected_response": expected_response,
        "final_displacement": displacement,
        "peak_control_fraction": max(row[-1] for row in rows),
        "peak_ee_offset": peak_ee_offset,
        "csv": output,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=GENERATED_MODEL)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--compliance", type=float, nargs=3)
    parser.add_argument("--force", type=float, nargs=3)
    parser.add_argument("--record", type=Path)
    args = parser.parse_args()
    run(args.config, args.model, args.headless, args.duration, args.rebuild,
        args.compliance, args.force, args.record)


if __name__ == "__main__":
    main()
