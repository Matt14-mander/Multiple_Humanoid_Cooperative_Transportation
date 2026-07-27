#!/usr/bin/env python3
"""Stacked dynamics for supported dual-robot cooperative carrying.

The three Pinocchio models remain separate trees. The coupled system contains
two support contacts per robot and two rigid grasp contacts on the payload.
Support contacts can be modeled as fixed endpoint positions or as WF wheel
rolling/no-slip contacts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Matrix-vector product without relying on the local BLAS installation."""

    return np.array(
        [
            sum(matrix[row, column] * vector[column] for column in range(matrix.shape[1]))
            for row in range(matrix.shape[0])
        ]
    )


def transpose_matvec(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    return np.array(
        [
            sum(matrix[row, column] * vector[row] for row in range(matrix.shape[0]))
            for column in range(matrix.shape[1])
        ]
    )


def matmat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Dense matrix product without relying on the local BLAS installation."""

    if left.shape[1] != right.shape[0]:
        raise ValueError("Incompatible matrix dimensions")
    return np.array(
        [
            [
                sum(left[row, index] * right[index, column] for index in range(left.shape[1]))
                for column in range(right.shape[1])
            ]
            for row in range(left.shape[0])
        ]
    )


def row_times_matrix(row: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Multiply a 3D row vector by a matrix without local BLAS."""

    return np.array(
        [
            sum(row[index] * matrix[index, column] for index in range(3))
            for column in range(matrix.shape[1])
        ]
    )


def block_diagonal(*matrices: np.ndarray) -> np.ndarray:
    rows = sum(matrix.shape[0] for matrix in matrices)
    columns = sum(matrix.shape[1] for matrix in matrices)
    result = np.zeros((rows, columns))
    row_offset = 0
    column_offset = 0
    for matrix in matrices:
        row_end = row_offset + matrix.shape[0]
        column_end = column_offset + matrix.shape[1]
        result[row_offset:row_end, column_offset:column_end] = matrix
        row_offset = row_end
        column_offset = column_end
    return result


def solve_dense_system(
    matrix: np.ndarray, vector: np.ndarray, tolerance: float = 1e-12
) -> np.ndarray:
    """Solve a small dense system by pivoted elimination.

    This keeps the validation scripts independent of the local Windows LAPACK
    installation. Production MPC code should use a proper sparse solver.
    """

    augmented = np.column_stack(
        (np.array(matrix, dtype=float, copy=True), np.array(vector, dtype=float))
    )
    rows, columns = matrix.shape
    if rows != columns or vector.shape != (rows,):
        raise ValueError("Only square systems with a matching RHS are supported")
    for column in range(columns):
        pivot_row = column + int(
            np.argmax(np.abs(augmented[column:, column]))
        )
        pivot = augmented[pivot_row, column]
        if abs(pivot) <= tolerance:
            raise np.linalg.LinAlgError("Singular dense system")
        if pivot_row != column:
            augmented[[column, pivot_row]] = augmented[[pivot_row, column]]
        augmented[column] /= augmented[column, column]
        for row in range(rows):
            if row == column:
                continue
            factor = augmented[row, column]
            if abs(factor) > tolerance:
                augmented[row] -= factor * augmented[column]
    return augmented[:, -1]


@dataclass
class CoupledTerms:
    """Dynamics and contact terms evaluated at one stacked state."""

    mass_matrix: np.ndarray
    nonlinear_effects: np.ndarray
    actuation_matrix: np.ndarray
    contact_jacobian: np.ndarray
    contact_jacobian_dot_velocity: np.ndarray
    closure_velocity: np.ndarray
    grasp_errors: tuple[np.ndarray, np.ndarray]
    support_force_bases: tuple[np.ndarray, ...]


class CoupledDynamicsModel:
    """Two identical free-flyer robots coupled to one free-flyer payload.

    Stacked ordering is always:

    ``q = [q_robot_1, q_robot_2, q_payload]``
    ``v = [v_robot_1, v_robot_2, v_payload]``
    ``lambda = [lambda_support, lambda_grasp_left, lambda_grasp_right]``

    In fixed-support mode, support multipliers are world-aligned forces. In
    rolling mode they are coordinates in the normal/axle/rolling basis and
    include the wheel rolling moment arm. Grasp wrenches are six-dimensional
    world-aligned force/moment vectors acting on the payload; the robots
    receive the opposite wrench. All contact Jacobians use
    ``pin.LOCAL_WORLD_ALIGNED``.
    """

    def __init__(
        self,
        pin,
        robot_model,
        payload_model,
        robot_frame_name: str = "link6",
        left_payload_frame_name: str = "grasp_left",
        right_payload_frame_name: str = "grasp_right",
        support_frame_names: tuple[str, str] | None = None,
        support_mode: str = "fixed_3d_position",
        support_contact_radius: float = 0.127,
        support_ground_normal: tuple[float, float, float] = (0.0, 0.0, 1.0),
    ) -> None:
        self.pin = pin
        self.robot_model = robot_model
        self.payload_model = payload_model
        self.robot_data_1 = robot_model.createData()
        self.robot_data_2 = robot_model.createData()
        self.payload_data = payload_model.createData()

        self.robot_frame_id = robot_model.getFrameId(robot_frame_name)
        if support_frame_names is None:
            support_frame_names = self._default_support_frame_names(robot_model)
        if len(support_frame_names) != 2:
            raise ValueError("Exactly two support frames are required per robot")
        self.support_frame_names = tuple(support_frame_names)
        supported_modes = ("fixed_3d_position", "rolling_no_slip")
        if support_mode not in supported_modes:
            raise ValueError(
                f"support_mode must be one of {supported_modes}, got {support_mode}"
            )
        self.support_mode = support_mode
        if support_contact_radius <= 0.0:
            raise ValueError("support_contact_radius must be positive")
        self.support_contact_radius = float(support_contact_radius)
        ground_normal = np.asarray(support_ground_normal, dtype=float)
        normal_norm = float(np.sqrt(sum(value * value for value in ground_normal)))
        if normal_norm <= 1e-12:
            raise ValueError("support_ground_normal must be nonzero")
        self.support_ground_normal = ground_normal / normal_norm
        self.support_contact_offset_world = (
            -self.support_contact_radius * self.support_ground_normal
        )
        if support_mode == "rolling_no_slip" and not all(
            name in ("wheel_L_Link", "wheel_R_Link")
            for name in self.support_frame_names
        ):
            raise ValueError(
                "rolling_no_slip currently requires wheel_L_Link and wheel_R_Link"
            )
        self.support_frame_ids = tuple(
            robot_model.getFrameId(name) for name in self.support_frame_names
        )
        self.left_payload_frame_id = payload_model.getFrameId(
            left_payload_frame_name
        )
        self.right_payload_frame_id = payload_model.getFrameId(
            right_payload_frame_name
        )
        if self.robot_frame_id >= robot_model.nframes:
            raise ValueError(f"Missing robot frame: {robot_frame_name}")
        for name, frame_id in zip(self.support_frame_names, self.support_frame_ids):
            if frame_id >= robot_model.nframes:
                raise ValueError(f"Missing support frame: {name}")
        if self.left_payload_frame_id >= payload_model.nframes:
            raise ValueError(
                f"Missing payload frame: {left_payload_frame_name}"
            )
        if self.right_payload_frame_id >= payload_model.nframes:
            raise ValueError(
                f"Missing payload frame: {right_payload_frame_name}"
            )

        self.robot_q_slices = (
            slice(0, robot_model.nq),
            slice(robot_model.nq, 2 * robot_model.nq),
        )
        payload_q_start = 2 * robot_model.nq
        self.payload_q_slice = slice(
            payload_q_start, payload_q_start + payload_model.nq
        )
        self.robot_v_slices = (
            slice(0, robot_model.nv),
            slice(robot_model.nv, 2 * robot_model.nv),
        )
        payload_v_start = 2 * robot_model.nv
        self.payload_v_slice = slice(
            payload_v_start, payload_v_start + payload_model.nv
        )
        self.nq = 2 * robot_model.nq + payload_model.nq
        self.nv = 2 * robot_model.nv + payload_model.nv
        self.support_contact_specs = ((0, 0), (0, 1), (1, 0), (1, 1))
        self.support_active_mask = (True, True, True, True)
        self.grasp_constraint_dim = 2 * 6
        self._update_contact_dimensions()

        self.robot_actuated_velocity_indices = self._actuated_velocity_indices(
            robot_model
        )
        self.nu = 2 * len(self.robot_actuated_velocity_indices)
        self.actuation_matrix = self._build_actuation_matrix()

    def _update_contact_dimensions(self) -> None:
        active_specs = tuple(
            spec
            for spec, active in zip(
                self.support_contact_specs, self.support_active_mask
            )
            if active
        )
        self.active_support_specs = active_specs
        self.support_constraint_dim = 3 * len(active_specs)
        self.contact_wrench_dim = (
            self.support_constraint_dim + self.grasp_constraint_dim
        )
        self.contact_names = tuple(
            "robot_{}/{}".format(robot_index + 1, self.support_frame_names[frame_index])
            for robot_index, frame_index in active_specs
        ) + ("payload/grasp_left", "payload/grasp_right")
        self.contact_block_dimensions = tuple(
            3 for _ in active_specs
        ) + (6, 6)

    def set_support_active_mask(self, active_mask: tuple[bool, bool, bool, bool]) -> None:
        """Enable/disable the four support point constraints for a contact mode."""

        if len(active_mask) != len(self.support_contact_specs):
            raise ValueError("Exactly four support contact states are required")
        self.support_active_mask = tuple(bool(active) for active in active_mask)
        if not any(self.support_active_mask):
            raise ValueError("At least one support contact must remain active")
        self._update_contact_dimensions()

    @staticmethod
    def _default_support_frame_names(model) -> tuple[str, str]:
        """Select the endpoint frames exposed by the staged TRON1 URDF."""

        candidates = (
            ("wheel_L_Link", "wheel_R_Link"),
            ("ankle_L_Link", "ankle_R_Link"),
        )
        for names in candidates:
            if all(model.getFrameId(name) < model.nframes for name in names):
                return names
        raise ValueError(
            "Could not infer support frames; pass support_frame_names explicitly"
        )

    @staticmethod
    def _actuated_velocity_indices(model) -> list[int]:
        indices: list[int] = []
        for joint_id in range(2, model.njoints):
            nv = int(model.nvs[joint_id])
            if nv <= 0:
                continue
            start = int(model.idx_vs[joint_id])
            indices.extend(range(start, start + nv))
        return indices

    def _build_actuation_matrix(self) -> np.ndarray:
        result = np.zeros((self.nv, self.nu))
        for local_index, velocity_index in enumerate(
            self.robot_actuated_velocity_indices
        ):
            result[velocity_index, local_index] = 1.0
            result[self.robot_model.nv + velocity_index, len(self.robot_actuated_velocity_indices) + local_index] = 1.0
        return result

    def split_configuration(self, q: np.ndarray):
        if q.shape != (self.nq,):
            raise ValueError(f"Expected q shape {(self.nq,)}, got {q.shape}")
        return (
            q[self.robot_q_slices[0]],
            q[self.robot_q_slices[1]],
            q[self.payload_q_slice],
        )

    def split_velocity(self, v: np.ndarray):
        if v.shape != (self.nv,):
            raise ValueError(f"Expected v shape {(self.nv,)}, got {v.shape}")
        return (
            v[self.robot_v_slices[0]],
            v[self.robot_v_slices[1]],
            v[self.payload_v_slice],
        )

    def _model_terms(self, model, data, q, v):
        mass_matrix = self.pin.crba(model, data, q)
        mass_matrix = 0.5 * (mass_matrix + mass_matrix.T)
        nonlinear_effects = self.pin.nonLinearEffects(model, data, q, v)

        self.pin.computeJointJacobians(model, data, q)
        self.pin.updateFramePlacements(model, data)
        self.pin.computeJointJacobiansTimeVariation(model, data, q, v)
        self.pin.updateFramePlacements(model, data)
        return mass_matrix, nonlinear_effects

    def _frame_jacobian_terms(self, model, data, frame_id):
        reference = self.pin.LOCAL_WORLD_ALIGNED
        jacobian = self.pin.getFrameJacobian(model, data, frame_id, reference)
        jacobian_dot = self.pin.getFrameJacobianTimeVariation(
            model, data, frame_id, reference
        )
        return jacobian, jacobian_dot

    @staticmethod
    def _skew(vector: np.ndarray) -> np.ndarray:
        x, y, z = vector
        return np.array(
            (
                (0.0, -z, y),
                (z, 0.0, -x),
                (-y, x, 0.0),
            )
        )

    @staticmethod
    def _cross(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        return np.array(
            (
                left[1] * right[2] - left[2] * right[1],
                left[2] * right[0] - left[0] * right[2],
                left[0] * right[1] - left[1] * right[0],
            )
        )

    def _support_jacobian_terms(
        self, model, data, frame_id: int, velocity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the 3D support Jacobian and its velocity derivative.

        ``fixed_3d_position`` constrains the endpoint frame origin. In
        ``rolling_no_slip`` mode the three rows are the nonholonomic wheel
        constraints: no normal velocity, no velocity along the wheel axle, and
        no rolling-direction slip. The rolling direction is computed from the
        wheel axle and the ground normal, so the wheel can move forward.
        """

        jacobian, jacobian_dot = self._frame_jacobian_terms(
            model, data, frame_id
        )
        if self.support_mode == "fixed_3d_position":
            return jacobian[:3], jacobian_dot[:3]

        rotation = data.oMf[frame_id].rotation
        wheel_axis = matvec(rotation, np.array((0.0, 1.0, 0.0)))
        ground_normal = self.support_ground_normal
        rolling_direction = self._cross(wheel_axis, ground_normal)
        rolling_norm = float(np.sqrt(sum(value * value for value in rolling_direction)))
        if rolling_norm <= 1e-12:
            raise ValueError("Wheel axis is parallel to the ground normal")
        rolling_direction /= rolling_norm
        angular_velocity = matvec(jacobian[3:], velocity)
        wheel_axis_dot = self._cross(angular_velocity, wheel_axis)
        rolling_direction_dot = self._cross(wheel_axis_dot, ground_normal)
        linear_jacobian = jacobian[:3]
        angular_jacobian = jacobian[3:]
        linear_jacobian_dot = jacobian_dot[:3]
        angular_jacobian_dot = jacobian_dot[3:]
        support_jacobian = np.vstack(
            (
                row_times_matrix(ground_normal, linear_jacobian),
                row_times_matrix(wheel_axis, linear_jacobian),
                row_times_matrix(rolling_direction, linear_jacobian)
                - self.support_contact_radius
                * row_times_matrix(wheel_axis, angular_jacobian),
            )
        )
        support_jacobian_dot = np.vstack(
            (
                row_times_matrix(ground_normal, linear_jacobian_dot),
                row_times_matrix(wheel_axis, linear_jacobian_dot)
                + row_times_matrix(wheel_axis_dot, linear_jacobian),
                row_times_matrix(rolling_direction, linear_jacobian_dot)
                + row_times_matrix(rolling_direction_dot, linear_jacobian)
                - self.support_contact_radius
                * row_times_matrix(wheel_axis, angular_jacobian_dot)
                - self.support_contact_radius
                * row_times_matrix(wheel_axis_dot, angular_jacobian),
            )
        )
        return support_jacobian, support_jacobian_dot

    def _support_force_basis(self, data, frame_id: int) -> np.ndarray:
        """Return world force directions corresponding to support multipliers."""

        if self.support_mode == "fixed_3d_position":
            return np.eye(3)
        rotation = data.oMf[frame_id].rotation
        wheel_axis = matvec(rotation, np.array((0.0, 1.0, 0.0)))
        rolling_direction = self._cross(
            wheel_axis, self.support_ground_normal
        )
        rolling_norm = float(
            np.sqrt(sum(value * value for value in rolling_direction))
        )
        if rolling_norm <= 1e-12:
            raise ValueError("Wheel axis is parallel to the ground normal")
        rolling_direction /= rolling_norm
        return np.column_stack(
            (self.support_ground_normal, wheel_axis, rolling_direction)
        )

    def support_contact_position(self, data, frame_id: int) -> np.ndarray:
        """Return the modeled instantaneous support point in world coordinates."""

        position = data.oMf[frame_id].translation.copy()
        if self.support_mode == "rolling_no_slip":
            position += self.support_contact_offset_world
        return position

    def evaluate(self, q: np.ndarray, v: np.ndarray) -> CoupledTerms:
        q1, q2, qp = self.split_configuration(q)
        v1, v2, vp = self.split_velocity(v)
        m1, h1 = self._model_terms(
            self.robot_model, self.robot_data_1, q1, v1
        )
        m2, h2 = self._model_terms(
            self.robot_model, self.robot_data_2, q2, v2
        )
        mp, hp = self._model_terms(
            self.payload_model, self.payload_data, qp, vp
        )

        js1_left, djs1_left = self._support_jacobian_terms(
            self.robot_model,
            self.robot_data_1,
            self.support_frame_ids[0],
            v1,
        )
        js1_right, djs1_right = self._support_jacobian_terms(
            self.robot_model,
            self.robot_data_1,
            self.support_frame_ids[1],
            v1,
        )
        js2_left, djs2_left = self._support_jacobian_terms(
            self.robot_model,
            self.robot_data_2,
            self.support_frame_ids[0],
            v2,
        )
        js2_right, djs2_right = self._support_jacobian_terms(
            self.robot_model,
            self.robot_data_2,
            self.support_frame_ids[1],
            v2,
        )
        jr1, djr1 = self._frame_jacobian_terms(
            self.robot_model, self.robot_data_1, self.robot_frame_id
        )
        jr2, djr2 = self._frame_jacobian_terms(
            self.robot_model, self.robot_data_2, self.robot_frame_id
        )
        jp_left, djp_left = self._frame_jacobian_terms(
            self.payload_model, self.payload_data, self.left_payload_frame_id
        )
        jp_right, djp_right = self._frame_jacobian_terms(
            self.payload_model, self.payload_data, self.right_payload_frame_id
        )

        zero_support_rr = np.zeros((3, self.robot_model.nv))
        zero_support_pp = np.zeros((3, self.payload_model.nv))
        zero_grasp_rr = np.zeros((6, self.robot_model.nv))
        support_terms = (
            ((0, 0), js1_left, djs1_left, v1),
            ((0, 1), js1_right, djs1_right, v1),
            ((1, 0), js2_left, djs2_left, v2),
            ((1, 1), js2_right, djs2_right, v2),
        )
        support_blocks = []
        support_bias = []
        support_force_bases = []
        for spec, support_jacobian, support_jacobian_dot, support_velocity in (
            support_terms
        ):
            if spec not in self.active_support_specs:
                continue
            robot_index, _ = spec
            if robot_index == 0:
                support_blocks.append(
                    np.hstack(
                        (support_jacobian, zero_support_rr, zero_support_pp)
                    )
                )
            else:
                support_blocks.append(
                    np.hstack(
                        (zero_support_rr, support_jacobian, zero_support_pp)
                    )
                )
            support_bias.append(
                matvec(support_jacobian_dot, support_velocity)
            )
            support_force_bases.append(
                self._support_force_basis(
                    self.robot_data_1 if robot_index == 0 else self.robot_data_2,
                    self.support_frame_ids[spec[1]],
                )
            )
        grasp_blocks = (
            np.hstack((-jr1, zero_grasp_rr, jp_left)),
            np.hstack((zero_grasp_rr, -jr2, jp_right)),
        )
        contact_jacobian = np.vstack(tuple(support_blocks) + grasp_blocks)
        contact_jacobian_dot_velocity = np.concatenate(
            tuple(support_bias)
            + (
                -matvec(djr1, v1) + matvec(djp_left, vp),
                -matvec(djr2, v2) + matvec(djp_right, vp),
            )
        )
        closure_velocity = matvec(contact_jacobian, v)
        grasp_errors = (
            self.robot_data_1.oMf[self.robot_frame_id]
            .actInv(self.payload_data.oMf[self.left_payload_frame_id]),
            self.robot_data_2.oMf[self.robot_frame_id]
            .actInv(self.payload_data.oMf[self.right_payload_frame_id]),
        )

        return CoupledTerms(
            mass_matrix=block_diagonal(m1, m2, mp),
            nonlinear_effects=np.concatenate((h1, h2, hp)),
            actuation_matrix=self.actuation_matrix,
            contact_jacobian=contact_jacobian,
            contact_jacobian_dot_velocity=contact_jacobian_dot_velocity,
            closure_velocity=closure_velocity,
            grasp_errors=grasp_errors,
            support_force_bases=tuple(support_force_bases),
        )

    def dynamics_residual(
        self,
        terms: CoupledTerms,
        acceleration: np.ndarray,
        actuated_torque: np.ndarray,
        contact_wrench: np.ndarray,
        external_generalized_force: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return ``M*a + h - B*tau - Jc.T*lambda - f_ext``."""

        if acceleration.shape != (self.nv,):
            raise ValueError("acceleration has the wrong dimension")
        if actuated_torque.shape != (self.nu,):
            raise ValueError("actuated_torque has the wrong dimension")
        if contact_wrench.shape != (self.contact_wrench_dim,):
            raise ValueError("contact_wrench has the wrong dimension")
        if external_generalized_force is None:
            external_generalized_force = np.zeros(self.nv)
        if external_generalized_force.shape != (self.nv,):
            raise ValueError("external_generalized_force has the wrong dimension")
        inertia_term = matvec(terms.mass_matrix, acceleration)
        actuation_term = matvec(terms.actuation_matrix, actuated_torque)
        contact_term = transpose_matvec(
            terms.contact_jacobian, contact_wrench
        )
        return (
            inertia_term
            + terms.nonlinear_effects
            - actuation_term
            - contact_term
            - external_generalized_force
        )

    def solve_minimum_norm_inverse_dynamics(
        self, terms: CoupledTerms, acceleration: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for a minimum-norm actuator/contact solution.

        For a prescribed acceleration, this solves

        ``B*tau + Jc.T*lambda = M*a + h``

        using the minimum Euclidean norm solution. It is a feasibility check,
        not a contact-force optimizer: unilateral contact, friction cones,
        torque weighting, and internal-force objectives are not imposed here.
        """

        if acceleration.shape != (self.nv,):
            raise ValueError("acceleration has the wrong dimension")
        generalized_force = matvec(terms.mass_matrix, acceleration) + (
            terms.nonlinear_effects
        )
        force_map = np.hstack(
            (terms.actuation_matrix, terms.contact_jacobian.T)
        )
        gram = matmat(force_map, force_map.T)
        dual_solution = solve_dense_system(gram, generalized_force)
        solution = transpose_matvec(force_map, dual_solution)
        return solution[: self.nu], solution[self.nu :]

    def acceleration_constraint_residual(
        self, terms: CoupledTerms, acceleration: np.ndarray
    ) -> np.ndarray:
        """Return ``Jc*a + Jdotc*v`` for support and grasp constraints."""

        if acceleration.shape != (self.nv,):
            raise ValueError("acceleration has the wrong dimension")
        return matvec(terms.contact_jacobian, acceleration) + (
            terms.contact_jacobian_dot_velocity
        )

    def project_velocity_to_constraints(
        self, terms: CoupledTerms, velocity: np.ndarray
    ) -> np.ndarray:
        """Project a velocity onto the current velocity-level constraints."""

        if velocity.shape != (self.nv,):
            raise ValueError("velocity has the wrong dimension")
        jacobian = terms.contact_jacobian
        gram = matmat(jacobian, jacobian.T)
        dual = solve_dense_system(gram, matvec(jacobian, velocity))
        return velocity - transpose_matvec(jacobian, dual)

    def project_acceleration_to_constraints(
        self, terms: CoupledTerms, desired_acceleration: np.ndarray
    ) -> np.ndarray:
        """Project a desired acceleration onto the acceleration constraints."""

        if desired_acceleration.shape != (self.nv,):
            raise ValueError("desired_acceleration has the wrong dimension")
        jacobian = terms.contact_jacobian
        gram = matmat(jacobian, jacobian.T)
        dual = solve_dense_system(
            gram, -terms.contact_jacobian_dot_velocity
        )
        particular = transpose_matvec(jacobian, dual)
        nullspace_dual = solve_dense_system(
            gram, matvec(jacobian, desired_acceleration)
        )
        nullspace_component = desired_acceleration - transpose_matvec(
            jacobian, nullspace_dual
        )
        return particular + nullspace_component

    def assemble_kkt(
        self,
        terms: CoupledTerms,
        actuated_torque: np.ndarray,
        external_generalized_force: np.ndarray | None = None,
        constraint_rhs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the constrained acceleration/contact-wrench KKT system.

        The unknown is ``[a, lambda]`` and the equation is:

        [ M  -Jc.T ] [ a      ] = [ B*tau - h ]
        [ Jc   0   ] [ lambda ]   [ -Jdotc*v ]
        """

        if actuated_torque.shape != (self.nu,):
            raise ValueError("actuated_torque has the wrong dimension")
        if external_generalized_force is None:
            external_generalized_force = np.zeros(self.nv)
        if external_generalized_force.shape != (self.nv,):
            raise ValueError("external_generalized_force has the wrong dimension")
        if constraint_rhs is None:
            constraint_rhs = -terms.contact_jacobian_dot_velocity
        if constraint_rhs.shape != (self.contact_wrench_dim,):
            raise ValueError("constraint_rhs has the wrong dimension")
        rows = self.nv + self.contact_wrench_dim
        matrix = np.zeros((rows, rows))
        matrix[: self.nv, : self.nv] = terms.mass_matrix
        matrix[: self.nv, self.nv :] = -terms.contact_jacobian.T
        matrix[self.nv :, : self.nv] = terms.contact_jacobian
        rhs = np.concatenate(
            (
                matvec(terms.actuation_matrix, actuated_torque)
                + external_generalized_force
                - terms.nonlinear_effects,
                constraint_rhs,
            )
        )
        return matrix, rhs

    def solve_constrained_dynamics(
        self,
        terms: CoupledTerms,
        actuated_torque: np.ndarray,
        external_generalized_force: np.ndarray | None = None,
        constraint_rhs: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve for acceleration and all support/grasp contact wrenches.

        This is a validation solver only. It imposes the selected equality
        constraints, including fixed support or rolling no-slip rows, but it
        does not impose torque limits, unilateral contact, or friction cones.
        """

        matrix, rhs = self.assemble_kkt(
            terms,
            actuated_torque,
            external_generalized_force,
            constraint_rhs,
        )
        solution = solve_dense_system(matrix, rhs)
        return solution[: self.nv], solution[self.nv :]
