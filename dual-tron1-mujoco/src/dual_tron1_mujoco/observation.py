"""Exact 74-D WF_TRON1A policy observation port from WheelfootController.cpp."""

import math

import mujoco
import numpy as np

from .control import ACTION_ORDER
from .model_index import ModelIndex


OBSERVATION_SIZE = 74
HISTORY_LENGTH = 5
EE_POSITION_COMMAND = np.array([0.446, 0.0, 0.241], dtype=np.float32)
EE_RPY_COMMAND = np.array([-math.pi / 2.0, 0.0, -math.pi / 2.0])


def _rpy_matrix(roll: float, pitch: float, yaw: float):
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


class ObservationBuilder:
    """Maintain the independent 5x74 history for one robot."""

    velocity_observation_order = list(range(10)) + [12, 13, 10, 11]

    def __init__(self, model, prefix: str):
        self.model = model
        self.prefix = prefix
        self.index = ModelIndex(model)
        self.joint_names = [prefix + name for name in ACTION_ORDER]
        self.joints = [self.index.joint(name) for name in self.joint_names]
        self.actuators = [
            self.index.actuator(name) for name in self.joint_names
        ]
        self.quat_sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "quat"
        )
        self.gyro_sensor = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "gyro"
        )
        self.history = None
        rotation = _rpy_matrix(*EE_RPY_COMMAND)
        self.ee_matrix_command = rotation[:2, :].reshape(-1).astype(
            np.float32
        )

    def _sensor(self, data, sensor_id: int):
        start = int(self.model.sensor_adr[sensor_id])
        size = int(self.model.sensor_dim[sensor_id])
        return np.asarray(data.sensordata[start : start + size], dtype=float)

    def build(self, data, command, last_actions, base_height_command=0.75):
        quat = self._sensor(data, self.quat_sensor)
        gyro = self._sensor(data, self.gyro_sensor)
        rotation = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(rotation, quat)
        projected_gravity = rotation.reshape(3, 3).T @ np.array(
            [0.0, 0.0, -1.0]
        )

        q = np.array([data.qpos[item.qpos_adr] for item in self.joints])
        dq = np.array([data.qvel[item.dof_adr] for item in self.joints])
        effort = np.array(
            [data.actuator_force[item] for item in self.actuators]
        )
        joint_position = q[:12]  # all default joint angles are zero
        observed_velocity = dq[self.velocity_observation_order]
        observed_effort = effort[self.velocity_observation_order]
        command = np.asarray(command, dtype=float)
        zero_velocity_command = float(np.linalg.norm(command) < 0.05)

        observation = np.concatenate(
            (
                gyro,
                projected_gravity,
                command,
                [zero_velocity_command],
                EE_POSITION_COMMAND,
                self.ee_matrix_command,
                [float(base_height_command)],
                joint_position,
                observed_velocity,
                observed_effort,
                np.asarray(last_actions, dtype=float),
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_SIZE,):
            raise RuntimeError(
                "Observation shape {} != (74,)".format(observation.shape)
            )
        if not np.all(np.isfinite(observation)):
            raise FloatingPointError(self.prefix + " observation is non-finite")
        return observation

    def push_history(self, observation):
        observation = np.asarray(observation, dtype=np.float32)
        if self.history is None:
            self.history = np.tile(observation, HISTORY_LENGTH)
        else:
            self.history[:-OBSERVATION_SIZE] = self.history[
                OBSERVATION_SIZE:
            ]
            self.history[-OBSERVATION_SIZE:] = observation
        return self.history.reshape(1, -1)
