"""Run the fixed-base dual-WFYG_TRON2A MuJoCo baseline on Windows."""

import argparse
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from .build_scene import build_scene
from .configuration import load_config
from .control import JointHoldController
from .model_index import ModelIndex
from .model_loader import load_model
from .paths import DEFAULT_CONFIG, GENERATED_MODEL, PROJECT_ROOT
from .recorder import CsvRecorder


def _apply_initial_joint_positions(model, data, config):
    index = ModelIndex(model)
    for prefix in ("r1_", "r2_"):
        for name, value in config["model"].get(
            "initial_joint_positions", {}
        ).items():
            data.qpos[index.joint(prefix + name).qpos_adr] = float(value)


def run(config_path=DEFAULT_CONFIG, model_path=GENERATED_MODEL, headless=False,
        duration_s=None, rebuild=False, payload_mass_kg=None,
        payload_com_offset_m=None):
    config = load_config(config_path)
    if rebuild or payload_mass_kg is not None or payload_com_offset_m is not None or not Path(model_path).exists():
        build_scene(config_path, model_path, payload_mass_kg, payload_com_offset_m)
    model = load_model(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    _apply_initial_joint_positions(model, data, config)
    mujoco.mj_forward(model, data)

    expected = (57, 54, 36)
    actual = (int(model.nq), int(model.nv), int(model.nu))
    if actual != expected:
        raise RuntimeError("Unexpected model dimensions: {} != {}".format(actual, expected))
    controllers = [
        JointHoldController(model, data, prefix, config["control"])
        for prefix in ("r1_", "r2_")
    ]
    duration = float(duration_s if duration_s is not None else config["run"]["duration_s"])
    recorder = CsvRecorder(PROJECT_ROOT / config["run"]["record_csv"], model)
    payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload_body")
    base_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_Link") for prefix in ("r1_", "r2_")]
    initial_payload = data.xpos[payload_id].copy()

    def step():
        for controller in controllers:
            controller.update(data)
        mujoco.mj_step(model, data)
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise FloatingPointError("MuJoCo state became non-finite")
        recorder.sample(data)

    if headless:
        while data.time < duration:
            step()
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.fixedcamid = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"
            )
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            while viewer.is_running() and data.time < duration:
                start = time.perf_counter()
                step()
                viewer.sync()
                delay = model.opt.timestep - (time.perf_counter() - start)
                if delay > 0:
                    time.sleep(delay)
    recorder.write()
    final_payload = data.xpos[payload_id].copy()
    gaps = recorder.rows[-1][10:12] if recorder.rows else [0.0, 0.0]
    print(
        "completed: time={:.3f}s nq={} nv={} nu={} csv={}".format(
            data.time, model.nq, model.nv, model.nu, recorder.path.resolve()
        )
    )
    print(
        "motion: payload_dxyz={} base_z=({:.4f},{:.4f}) grasp_gaps=({:.4f},{:.4f})m max_ctrl={:.3f}".format(
            np.array2string(final_payload - initial_payload, precision=4),
            data.xpos[base_ids[0], 2], data.xpos[base_ids[1], 2],
            gaps[0], gaps[1], recorder.rows[-1][-1] if recorder.rows else 0.0,
        )
    )
    return {
        "model": model,
        "data": data,
        "payload_displacement": final_payload - initial_payload,
        "grasp_gaps": np.asarray(gaps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model", type=Path, default=GENERATED_MODEL)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--payload-mass", type=float)
    parser.add_argument("--payload-com", type=float, nargs=3)
    args = parser.parse_args()
    run(args.config, args.model, args.headless, args.duration, args.rebuild,
        args.payload_mass, args.payload_com)


if __name__ == "__main__":
    main()
