"""MuJoCo adapter for the internal-force suppression algorithms."""

from typing import Any, Dict, List, Tuple

import mujoco
import numpy as np

from internal_force_suppression.core.admittance_controller import (
    ResidualAdmittanceController,
)
from internal_force_suppression.core.internal_force_analyzer import (
    ContactWrench,
    InternalForceAnalyzer,
)
from internal_force_suppression.utils.safety_monitor import SafetyMonitor

from .carry_controller import ARM_JOINTS
from .model_index import ModelIndex


class MujocoInternalForceController:
    """Estimate weld loads and generate bounded per-arm suppression wrenches.

    MuJoCo exposes equality-constraint forces in constraint coordinates.  For
    each grasp weld, those rows are projected to generalized force and then
    reconstructed as a world-frame wrench at ``link6``.  The object-side
    wrenches are decomposed into effective and internal components by the
    shared IFSM implementation.
    """

    def __init__(self, model, config: Dict[str, Any]):
        self.model = model
        self.index = ModelIndex(model)
        self.config = config
        self.payload_id = self._body_id("payload_body")
        self.arms: List[Dict[str, Any]] = []
        for prefix in ("r1_", "r2_"):
            equality_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_EQUALITY,
                prefix + "grasp_weld",
            )
            if equality_id < 0:
                raise KeyError("MuJoCo equality not found: " + prefix + "grasp_weld")
            self.arms.append(
                {
                    "prefix": prefix,
                    "body_id": self._body_id(prefix + "link6"),
                    "equality_id": equality_id,
                    "dofs": np.array(
                        [
                            self.index.joint(prefix + name).dof_adr
                            for name in ARM_JOINTS
                        ],
                        dtype=int,
                    ),
                }
            )

        self.force_analyzer = InternalForceAnalyzer(config["force_analyzer"])
        self.admittance = [
            ResidualAdmittanceController(
                config["admittance_robot1"], robot_index=0
            ),
            ResidualAdmittanceController(
                config["admittance_robot2"], robot_index=1
            ),
        ]
        self.safety_monitor = SafetyMonitor(config["safety"])
        adapter = config.get("mujoco_adapter", {})
        adapter_gain = adapter.get("residual_gain")
        if adapter_gain is not None:
            adapter_gain = float(adapter_gain)
            if not 0.0 <= adapter_gain <= 1.0:
                raise ValueError("mujoco_adapter.residual_gain must be between 0 and 1")
            for controller in self.admittance:
                controller.residual_gain = adapter_gain
        self.cutoff_frequency = adapter.get("wrench_cutoff_frequency_hz", 20.0)
        self.max_correction_force = float(
            adapter.get("max_correction_force_n", 5.0)
        )
        self.max_correction_torque = float(
            adapter.get("max_correction_torque_nm", 1.0)
        )
        self.filtered_wrenches = np.zeros((2, 6))
        self.filter_initialized = False
        self.last_contact_wrenches = np.zeros((2, 6))
        self.last_internal_wrenches = np.zeros((2, 6))
        self.last_correction_wrenches = np.zeros((2, 6))
        self.last_force_info: Dict[str, Any] = {}
        self.update_count = 0

    def _body_id(self, name: str) -> int:
        body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name
        )
        if body_id < 0:
            raise KeyError("MuJoCo body not found: " + name)
        return body_id

    def reset(self) -> None:
        for controller in self.admittance:
            controller.reset()
        self.safety_monitor.reset()
        self.force_analyzer.reset_history()
        self.filtered_wrenches.fill(0.0)
        self.filter_initialized = False
        self.last_contact_wrenches.fill(0.0)
        self.last_internal_wrenches.fill(0.0)
        self.last_correction_wrenches.fill(0.0)
        self.last_force_info = {}
        self.update_count = 0

    def _constraint_wrench(self, data, arm: Dict[str, Any]) -> np.ndarray:
        if data.nefc == 0:
            return np.zeros(6)
        equality_rows = np.flatnonzero(
            (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
            & (data.efc_id == arm["equality_id"])
        )
        if equality_rows.size == 0:
            return np.zeros(6)

        constraint_jacobian = data.efc_J.reshape(data.nefc, self.model.nv)
        generalized_force = (
            constraint_jacobian[equality_rows].T @ data.efc_force[equality_rows]
        )
        jacobian_position = np.zeros((3, self.model.nv))
        jacobian_rotation = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(
            self.model,
            data,
            jacobian_position,
            jacobian_rotation,
            arm["body_id"],
        )
        dofs = arm["dofs"]
        wrench_to_joint = np.vstack(
            (jacobian_position[:, dofs], jacobian_rotation[:, dofs])
        ).T
        robot_side_wrench = np.linalg.lstsq(
            wrench_to_joint,
            generalized_force[dofs],
            rcond=None,
        )[0]
        # Constraint generalized force is applied to the robot.  Negating it
        # gives the contact wrench that the robot applies to the payload.
        return -robot_side_wrench

    def estimate_contact_wrenches(self, data, dt: float) -> np.ndarray:
        raw = np.vstack(
            [self._constraint_wrench(data, arm) for arm in self.arms]
        )
        if not self.filter_initialized:
            self.filtered_wrenches = raw
            self.filter_initialized = True
        elif self.cutoff_frequency is None:
            self.filtered_wrenches = raw
        else:
            omega_dt = 2.0 * np.pi * float(self.cutoff_frequency) * dt
            alpha = omega_dt / (1.0 + omega_dt)
            self.filtered_wrenches += alpha * (raw - self.filtered_wrenches)
        self.last_contact_wrenches = self.filtered_wrenches.copy()
        return self.last_contact_wrenches

    @staticmethod
    def _limit_wrench(
        wrench: np.ndarray, max_force: float, max_torque: float
    ) -> np.ndarray:
        limited = np.asarray(wrench, dtype=float).copy()
        force_norm = np.linalg.norm(limited[:3])
        if force_norm > max_force:
            limited[:3] *= max_force / force_norm
        torque_norm = np.linalg.norm(limited[3:])
        if torque_norm > max_torque:
            limited[3:] *= max_torque / torque_norm
        return limited

    def update(self, data, dt: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        contact_wrenches = self.estimate_contact_wrenches(data, dt)
        contacts = [
            ContactWrench(
                force=contact_wrenches[index, :3],
                moment=contact_wrenches[index, 3:],
                contact_point=data.xpos[arm["body_id"]].copy(),
            )
            for index, arm in enumerate(self.arms)
        ]
        force_info = self.force_analyzer.analyze(
            contacts[0],
            contacts[1],
            {"com": data.xpos[self.payload_id].copy()},
        )
        self.safety_monitor.check(force_info, dt)
        corrections = np.zeros((2, 6))
        for robot_index, controller in enumerate(self.admittance):
            internal_wrench = force_info["per_robot"][robot_index]["F_internal"]
            x_adm, v_adm, _ = controller.compute_admittance_dynamics(
                internal_wrench, dt
            )
            restorative_wrench = (
                controller.params.K @ x_adm
                + controller.params.B @ v_adm
            )
            gain = controller.compute_adaptive_gain(
                force_info["internal_magnitude"]
            )
            corrections[robot_index] = self._limit_wrench(
                -gain * restorative_wrench,
                self.max_correction_force,
                self.max_correction_torque,
            )

        self.last_internal_wrenches = np.vstack(
            [item["F_internal"] for item in force_info["per_robot"]]
        )
        self.last_correction_wrenches = corrections
        self.last_force_info = force_info
        self.update_count += 1
        diagnostics = {
            "contact_wrenches": self.last_contact_wrenches.copy(),
            "internal_wrenches": self.last_internal_wrenches.copy(),
            "correction_wrenches": corrections.copy(),
            "internal_magnitude": force_info["internal_magnitude"],
            "internal_ratio": force_info["internal_ratio"],
            "safety_status": self.safety_monitor.get_status(),
        }
        return corrections, diagnostics
