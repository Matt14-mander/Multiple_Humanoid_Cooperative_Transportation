"""Shadow-mode generalized-momentum observers for the two MuJoCo AIRBOT arms."""

from typing import Dict, List

import mujoco
import numpy as np
import pinocchio as pin

from internal_force_suppression.core.force_estimator import (
    CARRY,
    GeneralizedMomentumObserver,
)

from .airbot_observer_validation import (
    AIRBOT_EE_FRAME,
    AIRBOT_JOINTS,
    build_airbot_pinocchio_model,
)
from .model_index import ModelIndex


def _rotation_from_quaternion(quaternion: np.ndarray) -> np.ndarray:
    matrix = np.empty(9)
    mujoco.mju_quat2Mat(matrix, np.asarray(quaternion, dtype=float))
    return matrix.reshape(3, 3)


def _append_mujoco_tool_inertia(pin_model, mj_model, prefix: str) -> Dict[str, object]:
    """Merge the fixed gripper-stub inertia into Pinocchio link6."""
    tool_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, prefix + "gripper_stub"
    )
    link_id = mujoco.mj_name2id(
        mj_model, mujoco.mjtObj.mjOBJ_BODY, prefix + AIRBOT_EE_FRAME
    )
    if tool_id < 0 or link_id < 0:
        raise KeyError("MuJoCo AIRBOT gripper/link6 body is missing")
    if int(mj_model.body_parentid[tool_id]) != link_id:
        raise ValueError("gripper_stub must be a direct fixed child of link6")

    mass = float(mj_model.body_mass[tool_id])
    body_rotation = _rotation_from_quaternion(mj_model.body_quat[tool_id])
    inertial_rotation = _rotation_from_quaternion(mj_model.body_iquat[tool_id])
    com = (
        np.asarray(mj_model.body_pos[tool_id], dtype=float)
        + body_rotation @ np.asarray(mj_model.body_ipos[tool_id], dtype=float)
    )
    inertia_in_inertial_frame = np.diag(mj_model.body_inertia[tool_id])
    rotation = body_rotation @ inertial_rotation
    inertia_in_link = rotation @ inertia_in_inertial_frame @ rotation.T
    terminal_joint = pin_model.getJointId(AIRBOT_JOINTS[-1])
    pin_model.inertias[terminal_joint] = (
        pin_model.inertias[terminal_joint]
        + pin.Inertia(mass, com, inertia_in_link)
    )
    return {
        "mass_kg": mass,
        "com_m": com,
        "inertia_kg_m2": inertia_in_link,
    }


class MujocoDualArmMomentumObserver:
    """Estimate both arm wrenches without feeding them back into control.

    MuJoCo equality-constraint forces are read only to score the estimate.  The
    observer itself receives arm joint position, velocity, and actuator torque.
    This first integration deliberately supports fixed robot bases only.
    """

    def __init__(self, model, config: Dict[str, object]):
        self.model = model
        self.index = ModelIndex(model)
        self.measurement_start_s = float(config.get("measurement_start_s", 0.5))
        self.wrench_regularization = float(
            config.get("wrench_regularization", 1e-3)
        )
        if self.wrench_regularization < 0.0:
            raise ValueError("wrench_regularization must be nonnegative")
        observer_gain = float(config.get("observer_gain", 100.0))
        cutoff = config.get("cutoff_frequency_hz", 20.0)
        cutoff = None if cutoff is None else float(cutoff)
        bias_time_constant = (
            float(config.get("bias_time_constant_s", 0.10))
            if bool(config.get("bias_compensation_enabled", False))
            else None
        )

        self.arms: List[Dict[str, object]] = []
        for prefix in ("r1_", "r2_"):
            pin_model = build_airbot_pinocchio_model()
            tool = _append_mujoco_tool_inertia(pin_model, model, prefix)
            joints = [self.index.joint(prefix + name) for name in AIRBOT_JOINTS]
            actuators = [
                self.index.actuator(prefix + name) for name in AIRBOT_JOINTS
            ]
            equality_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                prefix + "grasp_weld",
            )
            body_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                prefix + AIRBOT_EE_FRAME,
            )
            if equality_id < 0 or body_id < 0:
                raise KeyError("MuJoCo AIRBOT grasp objects are missing")
            self.arms.append(
                {
                    "prefix": prefix,
                    "pin_model": pin_model,
                    "pin_data": pin_model.createData(),
                    "observer": GeneralizedMomentumObserver(
                        pin_model,
                        observer_gain=observer_gain,
                        cutoff_frequency=cutoff,
                        bias_compensation_time_constant_s=bias_time_constant,
                    ),
                    "joints": joints,
                    "actuators": actuators,
                    "dofs": np.array([joint.dof_adr for joint in joints]),
                    "body_id": body_id,
                    "equality_id": equality_id,
                    "tool": tool,
                }
            )

        self.last_estimated_wrenches = np.zeros((2, 6))
        self.last_truth_wrenches = np.zeros((2, 6))
        self.last_estimated_joint_torques = np.zeros((2, 6))
        self.last_truth_joint_torques = np.zeros((2, 6))
        self._wrench_errors = []
        self._joint_errors = []
        self._estimated_samples = []
        self._truth_samples = []
        self._estimated_samples_joint = []
        self._truth_samples_joint = []

    def reset(self) -> None:
        for arm in self.arms:
            arm["observer"].reset()
        self.last_estimated_wrenches.fill(0.0)
        self.last_truth_wrenches.fill(0.0)
        self.last_estimated_joint_torques.fill(0.0)
        self.last_truth_joint_torques.fill(0.0)
        self._wrench_errors.clear()
        self._joint_errors.clear()
        self._estimated_samples.clear()
        self._truth_samples.clear()
        self._estimated_samples_joint.clear()
        self._truth_samples_joint.clear()

    def _constraint_joint_truth(self, data, arm: Dict[str, object]):
        # The observer cannot distinguish the grasp weld from simultaneous
        # gripper/payload contacts: both are external generalized forces on
        # the arm.  MuJoCo's qfrc_constraint contains their complete sum and
        # is therefore the correct simulation-only ground truth.
        return np.asarray(data.qfrc_constraint[arm["dofs"]], dtype=float)

    def _wrench_from_joint_torque(
        self, arm: Dict[str, object], q: np.ndarray, joint_torque: np.ndarray
    ) -> np.ndarray:
        frame_id = arm["pin_model"].getFrameId(AIRBOT_EE_FRAME)
        jacobian = pin.computeFrameJacobian(
            arm["pin_model"],
            arm["pin_data"],
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        normal = jacobian @ jacobian.T + self.wrench_regularization * np.eye(6)
        return np.linalg.solve(normal, jacobian @ joint_torque)

    def update(self, data, contact_phase: str = CARRY) -> np.ndarray:
        estimated = np.zeros((2, 6))
        truth = np.zeros((2, 6))
        estimated_joint = np.zeros((2, 6))
        truth_joint = np.zeros((2, 6))
        dt = float(self.model.opt.timestep)
        for number, arm in enumerate(self.arms):
            q = np.array([data.qpos[j.qpos_adr] for j in arm["joints"]])
            v = np.array([data.qvel[j.dof_adr] for j in arm["joints"]])
            tau = np.array([data.ctrl[a] for a in arm["actuators"]])
            estimated_joint[number] = arm["observer"].estimate_external_torque(
                q, v, tau, dt, contact_phase=contact_phase
            )
            estimated[number] = self._wrench_from_joint_torque(
                arm, q, estimated_joint[number]
            )
            truth_joint[number] = self._constraint_joint_truth(data, arm)
            # Use the same Pinocchio Jacobian and pseudoinverse for estimate
            # and truth.  At the neutral pose the wrench is not unique, while
            # its joint-space projection is; mixing two pseudoinverses creates
            # a large but meaningless Cartesian difference.
            truth[number] = self._wrench_from_joint_torque(
                arm, q, truth_joint[number]
            )

        self.last_estimated_wrenches = estimated
        self.last_truth_wrenches = truth
        self.last_estimated_joint_torques = estimated_joint
        self.last_truth_joint_torques = truth_joint
        if float(data.time) >= self.measurement_start_s:
            self._wrench_errors.append(estimated - truth)
            self._joint_errors.append(estimated_joint - truth_joint)
            self._estimated_samples.append(estimated.copy())
            self._truth_samples.append(truth.copy())
            self._estimated_samples_joint.append(estimated_joint.copy())
            self._truth_samples_joint.append(truth_joint.copy())
        return estimated

    def summary(self) -> Dict[str, object]:
        if not self._wrench_errors:
            return {"sample_count": 0}
        wrench_errors = np.asarray(self._wrench_errors)
        joint_errors = np.asarray(self._joint_errors)
        estimated = np.asarray(self._estimated_samples)
        truth = np.asarray(self._truth_samples)
        return {
            "sample_count": int(len(wrench_errors)),
            "force_rmse_n": np.sqrt(
                np.mean(np.square(wrench_errors[:, :, :3]), axis=(0, 2))
            ),
            "moment_rmse_nm": np.sqrt(
                np.mean(np.square(wrench_errors[:, :, 3:]), axis=(0, 2))
            ),
            "joint_torque_rmse_nm": np.sqrt(
                np.mean(np.square(joint_errors), axis=(0, 2))
            ),
            "estimated_joint_torque_mean_nm": np.mean(
                np.asarray(self._estimated_samples_joint), axis=0
            ),
            "truth_joint_torque_mean_nm": np.mean(
                np.asarray(self._truth_samples_joint), axis=0
            ),
            "estimated_wrench_mean": np.mean(estimated, axis=0),
            "truth_wrench_mean": np.mean(truth, axis=0),
            "tool_mass_kg": np.array(
                [arm["tool"]["mass_kg"] for arm in self.arms]
            ),
            "bias_nm": np.vstack(
                [arm["observer"].estimated_bias for arm in self.arms]
            ),
        }
