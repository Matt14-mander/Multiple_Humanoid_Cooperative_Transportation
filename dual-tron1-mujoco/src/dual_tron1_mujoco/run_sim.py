"""Run the dual-TRON1 cooperative-carry model on Windows without ROS."""

import argparse
import time
from pathlib import Path

import mujoco
import numpy as np

from .build_scene import build_scene
from .carry_controller import CooperativeCarryHoldController
from .configuration import load_config
from .control import JointHoldController
from .commands import (
    DualCommandCoordinator,
    FormationHoldCoordinator,
    VelocitySchedule,
)
from .model_loader import load_model
from .paths import (
    CARRY_BALANCE_CONFIG,
    CARRY_BALANCE_MODEL,
    CARRY_CONFIG,
    CARRY_HOLD_CONFIG,
    CARRY_HOLD_MODEL,
    CARRY_MODEL,
    DEFAULT_CONFIG,
    FORWARD_CONFIG,
    FORWARD_MODEL,
    GENERATED_MODEL,
    POLICY_DIR,
    PROJECT_ROOT,
)
from .policy_controller import OnnxPolicyBackend, RobotPolicyController
from .recorder import CsvRecorder
from .safety import SafetyMonitor


def _set_equalities(model, data, names, active):
    for name in names:
        equality_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_EQUALITY, name
        )
        if equality_id < 0:
            raise KeyError("MuJoCo equality not found: " + name)
        data.eq_active[equality_id] = int(bool(active))


def _disable_payload_collisions(model):
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_GEOM, geom_id
        )
        if name and name.startswith("payload_"):
            model.geom_contype[geom_id] = 0
            model.geom_conaffinity[geom_id] = 0
            model.geom_rgba[geom_id, 3] = 0.0


def run(
    config_path: Path = DEFAULT_CONFIG,
    model_path: Path = GENERATED_MODEL,
    headless: bool = False,
    duration_s: float = None,
    rebuild: bool = False,
    controller_mode: str = "hold",
    world_vx: float = 0.0,
    world_vy: float = 0.0,
    yaw_rate: float = 0.0,
    command_start_s: float = 2.0,
    command_stop_s: float = 7.0,
    unlock_bases: bool = False,
    release_payload: bool = False,
    disable_payload_collision: bool = False,
    payload_mass_kg: float = None,
    enable_ifsm: bool = False,
) -> None:
    config = load_config(config_path)
    if rebuild or payload_mass_kg is not None or not Path(model_path).exists():
        build_scene(config_path, model_path, payload_mass_kg=payload_mass_kg)

    model = load_model(model_path)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)

    if unlock_bases:
        _set_equalities(
            model, data, ("r1_base_fix", "r2_base_fix"), active=False
        )
    if release_payload:
        _set_equalities(
            model,
            data,
            ("r1_grasp_weld", "r2_grasp_weld"),
            active=False,
        )
    if disable_payload_collision:
        _disable_payload_collisions(model)
    mujoco.mj_forward(model, data)

    expected = (49, 46, 28)
    actual = (int(model.nq), int(model.nv), int(model.nu))
    if actual != expected:
        raise RuntimeError(
            "Unexpected model dimensions: {} != {}".format(actual, expected)
        )

    coordinator = None
    safety = None
    carry_controller = None
    ifsm_controller = None
    if controller_mode == "hold":
        controllers = [
            JointHoldController(model, data, "r1_", config["control"]),
            JointHoldController(model, data, "r2_", config["control"]),
        ]
    elif controller_mode == "policy":
        backend = OnnxPolicyBackend(POLICY_DIR)
        controllers = [
            RobotPolicyController(
                model, prefix, config["control"], backend
            )
            for prefix in ("r1_", "r2_")
        ]
        coordinator = DualCommandCoordinator(
            model,
            VelocitySchedule(
                world_vx=world_vx,
                world_vy=world_vy,
                yaw_rate=yaw_rate,
                start_s=command_start_s,
                stop_s=command_stop_s,
            ),
        )
        safety = SafetyMonitor(
            model,
            config["control"].get("minimum_base_height_m", 0.30),
            config["control"].get("maximum_tilt_rad", 1.0),
        )
    elif controller_mode == "carry_hold":
        controllers = [
            JointHoldController(model, data, "r1_", config["control"]),
            JointHoldController(model, data, "r2_", config["control"]),
        ]
        impedance_config = dict(config["control"])
        impedance_config.update(config["carry_impedance"])
        carry_controller = CooperativeCarryHoldController(
            model, data, impedance_config
        )
        safety = SafetyMonitor(
            model,
            config["control"].get("minimum_base_height_m", 0.30),
            config["control"].get("maximum_tilt_rad", 1.0),
        )
    elif controller_mode == "carry_balance":
        backend = OnnxPolicyBackend(POLICY_DIR)
        controllers = [
            RobotPolicyController(
                model, prefix, config["control"], backend
            )
            for prefix in ("r1_", "r2_")
        ]
        coordinator = FormationHoldCoordinator(
            model, data, config["formation_control"]
        )
        impedance_config = dict(config["control"])
        impedance_config.update(config["carry_impedance"])
        carry_controller = CooperativeCarryHoldController(
            model, data, impedance_config
        )
        safety = SafetyMonitor(
            model,
            config["control"].get("minimum_base_height_m", 0.30),
            config["control"].get("maximum_tilt_rad", 1.0),
        )
    else:
        raise ValueError("Unknown controller mode: " + controller_mode)
    if enable_ifsm:
        if carry_controller is None:
            raise ValueError(
                "Internal-force suppression requires carry_hold or carry_balance mode"
            )
        from internal_force_suppression.config.ifsm_config import IFSMConfig

        from .internal_force_controller import MujocoInternalForceController

        ifsm_controller = MujocoInternalForceController(
            model, IFSMConfig().to_dict()
        )
    duration = (
        float(duration_s)
        if duration_s is not None
        else float(config["run"]["duration_s"])
    )
    record_path = PROJECT_ROOT / config["run"]["record_csv"]
    recorder = CsvRecorder(record_path, model)
    realtime = bool(config["run"].get("realtime", True)) and not headless

    def step() -> None:
        arm_wrench_correction = None
        if ifsm_controller is not None:
            arm_wrench_correction, _ = ifsm_controller.update(
                data, float(model.opt.timestep)
            )
        if controller_mode in {"policy", "carry_balance"}:
            for controller in controllers:
                controller.update(
                    data,
                    coordinator.command(data, controller.prefix),
                    actuate_arms=True,
                )
            if carry_controller is not None:
                carry_controller.update(data, arm_wrench_correction)
        else:
            for controller in controllers:
                controller.update(data)
            if carry_controller is not None:
                carry_controller.update(data, arm_wrench_correction)
        mujoco.mj_step(model, data)
        if safety is not None:
            safety.check(data)
        elif not np.all(np.isfinite(data.qpos)) or not np.all(
            np.isfinite(data.qvel)
        ):
            raise FloatingPointError("MuJoCo state became non-finite")
        recorder.sample(data)

    try:
        if headless:
            while data.time < duration:
                step()
        else:
            from mujoco import viewer as mj_viewer

            with mj_viewer.launch_passive(model, data) as viewer:
                viewer.cam.distance = 4.0
                viewer.cam.azimuth = 90.0
                viewer.cam.elevation = -20.0
                while viewer.is_running() and data.time < duration:
                    wall_start = time.perf_counter()
                    step()
                    viewer.sync()
                    if realtime:
                        remaining = model.opt.timestep - (
                            time.perf_counter() - wall_start
                        )
                        if remaining > 0:
                            time.sleep(remaining)
    finally:
        recorder.close()

    print(
        "completed: mode={} time={:.3f}s nq={} nv={} nu={} csv={}".format(
            controller_mode, data.time, model.nq, model.nv, model.nu, record_path
        )
    )
    acceptance = config.get("acceptance", {})
    if controller_mode in {"policy", "carry_hold", "carry_balance"} and recorder.rows:
        first, last = recorder.rows[0], recorder.rows[-1]
        print(
            "motion: r1_dx={:.4f}m r2_dx={:.4f}m "
            "payload_dx={:.4f}m r1_final_z={:.4f}m "
            "r2_final_z={:.4f}m payload_final_z={:.4f}m "
            "grasp_gaps=({:.4f},{:.4f})m".format(
                last[4] - first[4],
                last[7] - first[7],
                last[1] - first[1],
                last[6],
                last[9],
                last[3],
                last[13],
                last[14],
            )
        )
        if controller_mode == "carry_balance":
            r1_origin = np.asarray(first[4:6])
            r2_origin = np.asarray(first[7:9])
            max_base_drift = max(
                max(
                    np.linalg.norm(np.asarray(row[4:6]) - r1_origin),
                    np.linalg.norm(np.asarray(row[7:9]) - r2_origin),
                )
                for row in recorder.rows
            )
            max_base_tilt = max(
                max(row[21], row[22]) for row in recorder.rows
            )
            max_base_speed = max(
                max(row[23], row[24]) for row in recorder.rows
            )
            min_base_height = min(
                min(row[6], row[9]) for row in recorder.rows
            )
            balance_checks = {
                "base_drift": max_base_drift
                <= float(acceptance.get("maximum_base_drift_m", 0.15)),
                "base_tilt": max_base_tilt
                <= float(acceptance.get("maximum_base_tilt_rad", 0.35)),
                "base_speed": max_base_speed
                <= float(acceptance.get("maximum_base_speed_mps", 0.50)),
                "base_height": min_base_height
                >= float(acceptance.get("minimum_base_height_m", 0.55)),
            }
            print(
                "balance: {} drift={:.4f}m tilt={:.4f}rad speed={:.4f}m/s "
                "min_height={:.4f}m checks={}".format(
                    "PASS" if all(balance_checks.values()) else "FAIL",
                    max_base_drift,
                    max_base_tilt,
                    max_base_speed,
                    min_base_height,
                    ",".join(
                        "{}={}".format(name, "ok" if passed else "fail")
                        for name, passed in balance_checks.items()
                    ),
                )
            )
            print(
                "formation_commands: max_error=({:.4f},{:.4f})m "
                "max_vx=({:.4f},{:.4f}) max_yaw=({:.4f},{:.4f})".format(
                    coordinator.max_position_errors["r1_"],
                    coordinator.max_position_errors["r2_"],
                    coordinator.max_abs_commands["r1_"][0],
                    coordinator.max_abs_commands["r2_"][0],
                    coordinator.max_abs_commands["r1_"][2],
                    coordinator.max_abs_commands["r2_"][2],
                )
            )
    if carry_controller is not None:
        first, last = recorder.rows[0], recorder.rows[-1]
        payload_origin = np.asarray(first[1:4])
        if controller_mode == "carry_balance":
            initial_base_centroid = 0.5 * (
                np.asarray(first[4:7]) + np.asarray(first[7:10])
            )
            initial_payload_offset = payload_origin - initial_base_centroid
            max_payload_error = max(
                np.linalg.norm(
                    np.asarray(row[1:4])
                    - 0.5
                    * (np.asarray(row[4:7]) + np.asarray(row[7:10]))
                    - initial_payload_offset
                )
                for row in recorder.rows
            )
        else:
            max_payload_error = max(
                np.linalg.norm(np.asarray(row[1:4]) - payload_origin)
                for row in recorder.rows
            )
        max_tilt = max(row[18] for row in recorder.rows)
        max_grasp_gap = max(
            max(row[13], row[14]) for row in recorder.rows
        )
        max_saturation = float(np.max(carry_controller.saturation_fractions))
        print(
            "carry_control: payload_mass={:.3f}kg desired_fz={:.3f}N "
            "arm_fz=({:.3f},{:.3f})N payload_dz={:.4f}m "
            "max_tilt={:.4f}rad arm_peak_torque=({:.3f},{:.3f})Nm".format(
                carry_controller.payload_mass,
                carry_controller.last_payload_wrench[2],
                carry_controller.last_arm_wrenches[0, 2],
                carry_controller.last_arm_wrenches[1, 2],
                last[3] - first[3],
                max_tilt,
                max(row[19] for row in recorder.rows),
                max(row[20] for row in recorder.rows),
            )
        )
        print(
            "arm_joint_peaks_nm: r1={} r2={} max_saturation={:.2%}".format(
                np.array2string(
                    carry_controller.peak_abs_arm_torques[0], precision=3
                ),
                np.array2string(
                    carry_controller.peak_abs_arm_torques[1], precision=3
                ),
                max_saturation,
            )
        )
        checks = {
            "payload_position": max_payload_error
            <= float(acceptance.get("maximum_payload_position_error_m", 0.02)),
            "payload_tilt": max_tilt
            <= float(acceptance.get("maximum_payload_tilt_rad", 0.10)),
            "grasp_gap": max_grasp_gap
            <= float(acceptance.get("maximum_grasp_gap_m", 0.015)),
            "arm_saturation": max_saturation
            <= float(acceptance.get("maximum_arm_saturation_fraction", 0.05)),
        }
        print(
            "acceptance: {} payload_error={:.4f}m max_gap={:.4f}m checks={}".format(
                "PASS" if all(checks.values()) else "FAIL",
                max_payload_error,
                max_grasp_gap,
                ",".join(
                    "{}={}".format(name, "ok" if passed else "fail")
                    for name, passed in checks.items()
                ),
            )
        )
    if ifsm_controller is not None and ifsm_controller.last_force_info:
        print(
            "ifsm: updates={} internal_force={:.4f} internal_ratio={:.2%} "
            "max_correction={:.4f}".format(
                ifsm_controller.update_count,
                ifsm_controller.last_force_info["internal_magnitude"],
                ifsm_controller.last_force_info["internal_ratio"],
                float(np.max(np.abs(ifsm_controller.last_correction_wrenches))),
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument(
        "--controller",
        choices=("hold", "policy", "carry_hold", "carry_balance"),
        default="hold",
    )
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--forward-test",
        action="store_true",
        help="Policy mode, free bases, released non-colliding payload",
    )
    mode_group.add_argument(
        "--carry-test",
        action="store_true",
        help="Policy mode, free bases, visible payload with both grasp welds",
    )
    mode_group.add_argument(
        "--carry-hold-test",
        action="store_true",
        help="Fixed-base static payload hold with object impedance",
    )
    mode_group.add_argument(
        "--carry-balance-test",
        action="store_true",
        help="Free-base zero-command balance with policy legs and arm impedance",
    )
    parser.add_argument("--world-vx", type=float, default=0.0)
    parser.add_argument("--world-vy", type=float, default=0.0)
    parser.add_argument("--yaw-rate", type=float, default=0.0)
    parser.add_argument("--command-start", type=float, default=2.0)
    parser.add_argument("--command-stop", type=float, default=7.0)
    parser.add_argument("--unlock-bases", action="store_true")
    parser.add_argument("--release-payload", action="store_true")
    parser.add_argument("--disable-payload-collision", action="store_true")
    parser.add_argument("--payload-mass", type=float)
    parser.add_argument(
        "--enable-ifsm",
        action="store_true",
        help="Enable MuJoCo internal-force estimation and admittance suppression",
    )
    args = parser.parse_args()
    if args.forward_test:
        args.controller = "policy"
        args.unlock_bases = True
        args.release_payload = True
        args.disable_payload_collision = True
        if args.duration is None:
            args.duration = 9.0
    elif args.carry_test:
        args.controller = "policy"
        args.unlock_bases = True
        if args.duration is None:
            args.duration = 9.0
    elif args.carry_hold_test:
        args.controller = "carry_hold"
        if args.duration is None:
            args.duration = 10.0
    elif args.carry_balance_test:
        args.controller = "carry_balance"
        args.unlock_bases = True
        if args.duration is None:
            args.duration = 10.0
    config_path = args.config
    model_path = args.model
    if config_path is None:
        if args.forward_test:
            config_path = FORWARD_CONFIG
        elif args.carry_test:
            config_path = CARRY_CONFIG
        elif args.carry_hold_test:
            config_path = CARRY_HOLD_CONFIG
        elif args.carry_balance_test:
            config_path = CARRY_BALANCE_CONFIG
        else:
            config_path = DEFAULT_CONFIG
    if model_path is None:
        if args.forward_test:
            model_path = FORWARD_MODEL
        elif args.carry_test:
            model_path = CARRY_MODEL
        elif args.carry_hold_test:
            model_path = CARRY_HOLD_MODEL
        elif args.carry_balance_test:
            model_path = CARRY_BALANCE_MODEL
        else:
            model_path = GENERATED_MODEL
    if (
        args.forward_test
        or args.carry_test
        or args.carry_hold_test
        or args.carry_balance_test
    ):
        args.rebuild = True
    run(
        config_path=config_path,
        model_path=model_path,
        headless=args.headless,
        duration_s=args.duration,
        rebuild=args.rebuild,
        controller_mode=args.controller,
        world_vx=args.world_vx,
        world_vy=args.world_vy,
        yaw_rate=args.yaw_rate,
        command_start_s=args.command_start,
        command_stop_s=args.command_stop,
        unlock_bases=args.unlock_bases,
        release_payload=args.release_payload,
        disable_payload_collision=args.disable_payload_collision,
        payload_mass_kg=args.payload_mass,
        enable_ifsm=args.enable_ifsm,
    )


if __name__ == "__main__":
    main()
