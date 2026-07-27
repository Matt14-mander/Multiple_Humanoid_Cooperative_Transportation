import csv
import math
from pathlib import Path
from typing import List

import mujoco
import numpy as np


class CsvRecorder:
    HEADER = [
        "time_s",
        "payload_x",
        "payload_y",
        "payload_z",
        "r1_base_x",
        "r1_base_y",
        "r1_base_z",
        "r2_base_x",
        "r2_base_y",
        "r2_base_z",
        "max_constraint_error",
        "r1_grasp_error",
        "r2_grasp_error",
        "r1_grasp_gap_m",
        "r2_grasp_gap_m",
        "r1_grasp_peak_row_force",
        "r2_grasp_peak_row_force",
        "max_abs_ctrl",
        "payload_tilt_rad",
        "r1_arm_max_abs_ctrl",
        "r2_arm_max_abs_ctrl",
        "r1_base_tilt_rad",
        "r2_base_tilt_rad",
        "r1_base_planar_speed_mps",
        "r2_base_planar_speed_mps",
    ]

    def __init__(self, path: Path, model):
        self.path = Path(path)
        self.model = model
        self.rows: List[List[float]] = []
        self.payload_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "payload_body"
        )
        self.r1_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "r1_base_Link"
        )
        self.r2_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "r2_base_Link"
        )
        self.grasp_eq_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_EQUALITY, name
            )
            for name in ("r1_grasp_weld", "r2_grasp_weld")
        ]
        self.grasp_body_pairs = [
            (
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, robot),
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, payload),
            )
            for robot, payload in (
                ("r1_link6", "payload_grasp_left"),
                ("r2_link6", "payload_grasp_right"),
            )
        ]
        self.arm_actuator_ids = [
            [
                mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + joint
                )
                for joint in ("J1", "J2", "J3", "J4", "J5", "J6")
            ]
            for prefix in ("r1_", "r2_")
        ]

    def _equality_metrics(self, data, equality_id: int):
        equality_type = int(mujoco.mjtConstraint.mjCNSTR_EQUALITY)
        rows = np.flatnonzero(
            (data.efc_type == equality_type) & (data.efc_id == equality_id)
        )
        if not rows.size:
            return 0.0, 0.0
        return (
            float(np.max(np.abs(data.efc_pos[rows]))),
            float(np.max(np.abs(data.efc_force[rows]))),
        )

    def sample(self, data) -> None:
        constraint_error = (
            float(np.max(np.abs(data.efc_pos))) if data.nefc else 0.0
        )
        max_ctrl = float(np.max(np.abs(data.ctrl))) if data.ctrl.size else 0.0
        grasp_metrics = [
            self._equality_metrics(data, equality_id)
            for equality_id in self.grasp_eq_ids
        ]
        grasp_gaps = [
            float(np.linalg.norm(data.xpos[robot] - data.xpos[payload]))
            for robot, payload in self.grasp_body_pairs
        ]
        payload_z_axis = data.xmat[self.payload_id].reshape(3, 3)[:, 2]
        payload_tilt = math.acos(
            float(np.clip(payload_z_axis[2], -1.0, 1.0))
        )
        arm_max_ctrl = [
            float(np.max(np.abs(data.ctrl[actuator_ids])))
            for actuator_ids in self.arm_actuator_ids
        ]
        base_tilts = []
        base_speeds = []
        for body_id in (self.r1_id, self.r2_id):
            body_z_axis = data.xmat[body_id].reshape(3, 3)[:, 2]
            base_tilts.append(
                math.acos(float(np.clip(body_z_axis[2], -1.0, 1.0)))
            )
            velocity = np.zeros(6)
            mujoco.mj_objectVelocity(
                self.model,
                data,
                mujoco.mjtObj.mjOBJ_BODY,
                body_id,
                velocity,
                0,
            )
            base_speeds.append(float(np.linalg.norm(velocity[3:5])))
        self.rows.append(
            [
                float(data.time),
                *data.xpos[self.payload_id].tolist(),
                *data.xpos[self.r1_id].tolist(),
                *data.xpos[self.r2_id].tolist(),
                constraint_error,
                grasp_metrics[0][0],
                grasp_metrics[1][0],
                grasp_gaps[0],
                grasp_gaps[1],
                grasp_metrics[0][1],
                grasp_metrics[1][1],
                max_ctrl,
                payload_tilt,
                arm_max_ctrl[0],
                arm_max_ctrl[1],
                base_tilts[0],
                base_tilts[1],
                base_speeds[0],
                base_speeds[1],
            ]
        )

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.HEADER)
            writer.writerows(self.rows)
