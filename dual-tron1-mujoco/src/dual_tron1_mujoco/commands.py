"""Time-scheduled world-frame commands for arbitrary robot headings."""

from dataclasses import dataclass

import mujoco
import numpy as np


def _wrap_angle(angle: float) -> float:
    return float((angle + np.pi) % (2.0 * np.pi) - np.pi)


@dataclass(frozen=True)
class VelocitySchedule:
    world_vx: float = 0.0
    world_vy: float = 0.0
    yaw_rate: float = 0.0
    start_s: float = 2.0
    stop_s: float = 7.0

    def command_at(self, time_s: float):
        if self.start_s <= time_s < self.stop_s:
            return np.array(
                [self.world_vx, self.world_vy, self.yaw_rate], dtype=float
            )
        return np.zeros(3, dtype=float)


class DualCommandCoordinator:
    """Convert one world velocity into each robot's local forward command."""

    def __init__(self, model, schedule: VelocitySchedule):
        self.schedule = schedule
        self.body_ids = {
            prefix: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_Link"
            )
            for prefix in ("r1_", "r2_")
        }

    def command(self, data, prefix: str):
        world_command = self.schedule.command_at(float(data.time))
        rotation = data.xmat[self.body_ids[prefix]].reshape(3, 3)
        world_velocity = np.array(
            [world_command[0], world_command[1], 0.0], dtype=float
        )
        local_velocity = rotation.T @ world_velocity
        # The upstream WF controller forces lateral command to zero.
        return np.array(
            [float(local_velocity[0]), 0.0, float(world_command[2])],
            dtype=float,
        )


class FormationHoldCoordinator:
    """Low-bandwidth pose feedback around two initial mobile-base poses.

    The upstream policy has no lateral velocity input. Lateral position error
    is therefore converted to yaw correction; forward correction then brings
    each unicycle-like base back toward its initial world pose.
    """

    def __init__(self, model, data, config):
        self.model = model
        self.body_ids = {
            prefix: mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_Link"
            )
            for prefix in ("r1_", "r2_")
        }
        self.position_refs = {
            prefix: data.xpos[body_id, :2].copy()
            for prefix, body_id in self.body_ids.items()
        }
        self.yaw_refs = {
            prefix: self._yaw(data, body_id)
            for prefix, body_id in self.body_ids.items()
        }
        self.forward_kp = float(config.get("forward_kp", 0.08))
        self.forward_kd = float(config.get("forward_kd", 0.12))
        self.lateral_yaw_kp = float(config.get("lateral_yaw_kp", 0.25))
        self.heading_kp = float(config.get("heading_kp", 0.30))
        self.yaw_kd = float(config.get("yaw_kd", 0.08))
        self.max_forward = float(config.get("maximum_forward_command", 0.05))
        self.min_forward = float(config.get("minimum_forward_command", 0.0))
        self.allow_reverse = bool(config.get("allow_reverse", True))
        self.min_forward_by_prefix = {
            "r1_": float(
                config.get("robot_1_minimum_forward_command", self.min_forward)
            ),
            "r2_": float(
                config.get("robot_2_minimum_forward_command", self.min_forward)
            ),
        }
        self.max_forward_by_prefix = {
            "r1_": float(
                config.get("robot_1_maximum_forward_command", self.max_forward)
            ),
            "r2_": float(
                config.get("robot_2_maximum_forward_command", self.max_forward)
            ),
        }
        self.max_yaw = float(config.get("maximum_yaw_command", 0.15))
        self.position_deadband = float(config.get("position_deadband_m", 0.01))
        self.yaw_deadband = float(config.get("yaw_deadband_rad", 0.02))
        self.filter_time_constant = float(config.get("filter_time_constant_s", 0.20))
        self.start_s = float(config.get("start_s", 0.50))
        self.pulse_mode = bool(config.get("pulse_mode", False))
        self.pulse_commands = {
            "r1_": float(config.get("robot_1_pulse_command", 0.08)),
            "r2_": float(config.get("robot_2_pulse_command", 0.03)),
        }
        self.pulse_release_speed = float(
            config.get("pulse_release_speed_mps", 0.03)
        )
        self.pulse_engage_speed = float(
            config.get("pulse_engage_speed_mps", -0.01)
        )
        self.pulse_active = {prefix: False for prefix in self.body_ids}
        self.pulse_started_at = {prefix: 0.0 for prefix in self.body_ids}
        self.pulse_allowed_at = {prefix: self.start_s for prefix in self.body_ids}
        self.pulse_duration = float(config.get("pulse_duration_s", 0.15))
        self.pulse_cooldown = float(config.get("pulse_cooldown_s", 0.75))
        self.filtered_commands = {
            prefix: np.zeros(3) for prefix in self.body_ids
        }
        self.last_times = {prefix: float(data.time) for prefix in self.body_ids}
        self.max_position_errors = {prefix: 0.0 for prefix in self.body_ids}
        self.max_abs_commands = {prefix: np.zeros(3) for prefix in self.body_ids}

    def _yaw(self, data, body_id: int) -> float:
        rotation = data.xmat[body_id].reshape(3, 3)
        return float(np.arctan2(rotation[1, 0], rotation[0, 0]))

    def command(self, data, prefix: str):
        now = float(data.time)
        body_id = self.body_ids[prefix]
        world_error = self.position_refs[prefix] - data.xpos[body_id, :2]
        error_norm = float(np.linalg.norm(world_error))
        self.max_position_errors[prefix] = max(
            self.max_position_errors[prefix], error_norm
        )
        yaw = self._yaw(data, body_id)
        cosine, sine = np.cos(yaw), np.sin(yaw)
        local_forward_error = cosine * world_error[0] + sine * world_error[1]
        local_lateral_error = -sine * world_error[0] + cosine * world_error[1]
        yaw_error = _wrap_angle(self.yaw_refs[prefix] - yaw)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(
            self.model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            body_id,
            velocity,
            0,
        )
        local_forward_velocity = (
            cosine * velocity[3] + sine * velocity[4]
        )
        yaw_rate = float(velocity[2])

        if error_norm < self.position_deadband:
            local_forward_error = 0.0
            local_lateral_error = 0.0
        if abs(yaw_error) < self.yaw_deadband:
            yaw_error = 0.0
        if now < self.start_s:
            raw = np.zeros(3)
        else:
            max_forward = self.max_forward_by_prefix[prefix]
            min_forward = self.min_forward_by_prefix[prefix]
            if self.pulse_mode:
                if (
                    self.pulse_active[prefix]
                    and (
                        now - self.pulse_started_at[prefix]
                        >= self.pulse_duration
                        or local_forward_velocity >= self.pulse_release_speed
                    )
                ):
                    self.pulse_active[prefix] = False
                    self.pulse_allowed_at[prefix] = now + self.pulse_cooldown
                if local_forward_error <= self.position_deadband:
                    self.pulse_active[prefix] = False
                elif (
                    not self.pulse_active[prefix]
                    and local_forward_velocity <= self.pulse_engage_speed
                    and now >= self.pulse_allowed_at[prefix]
                ):
                    self.pulse_active[prefix] = True
                    self.pulse_started_at[prefix] = now
                forward_command = (
                    self.pulse_commands[prefix]
                    if self.pulse_active[prefix]
                    else 0.0
                )
            else:
                forward_lower = -max_forward if self.allow_reverse else 0.0
                forward_command = float(
                    np.clip(
                        self.forward_kp * local_forward_error
                        - self.forward_kd * local_forward_velocity,
                        forward_lower,
                        max_forward,
                    )
                )
                if not self.allow_reverse and forward_command > 0.0:
                    forward_command = max(forward_command, min_forward)
            raw = np.array(
                [
                    forward_command,
                    0.0,
                    np.clip(
                        self.lateral_yaw_kp * local_lateral_error
                        + self.heading_kp * yaw_error
                        - self.yaw_kd * yaw_rate,
                        -self.max_yaw,
                        self.max_yaw,
                    ),
                ]
            )

        dt = max(0.0, now - self.last_times[prefix])
        self.last_times[prefix] = now
        if self.filter_time_constant > 0.0:
            alpha = 1.0 - np.exp(-dt / self.filter_time_constant)
        else:
            alpha = 1.0
        command = self.filtered_commands[prefix]
        command += alpha * (raw - command)
        self.max_abs_commands[prefix] = np.maximum(
            self.max_abs_commands[prefix], np.abs(command)
        )
        return command.copy()
