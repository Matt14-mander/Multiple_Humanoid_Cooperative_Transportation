"""Torch-only batched terms for wiring intent estimation into Isaac Lab."""

import torch


def build_intent_features(q_history, dq_history, previous_action_history):
    """Flatten [N,H,J] histories using the declared channel-major layout."""
    if q_history.shape != dq_history.shape or q_history.shape != previous_action_history.shape:
        raise ValueError("q, dq and action histories must have identical [N,H,J] shapes")
    if q_history.ndim != 3:
        raise ValueError("intent histories must have shape [environments, history, joints]")
    return torch.cat(
        (q_history.flatten(1), dq_history.flatten(1), previous_action_history.flatten(1)),
        dim=-1,
    )


def shift_wrench_reference(force, torque, from_point, to_point):
    """Shift a wrench: M_to = M_from + (p_from - p_to) x F."""
    if force.shape != torque.shape or force.shape[-1] != 3:
        raise ValueError("force and torque require matching [...,3] shapes")
    if from_point.shape != force.shape or to_point.shape != force.shape:
        raise ValueError("wrench points must match force shape")
    return force, torque + torch.linalg.cross(from_point - to_point, force, dim=-1)


def world_wrench_to_base_yaw(force_world, torque_world, base_yaw):
    """Rotate a same-reference wrench into a gravity-aligned base-yaw frame."""
    if force_world.shape != torque_world.shape or force_world.shape[-1] != 3:
        raise ValueError("force and torque require matching [...,3] shapes")
    if base_yaw.shape != force_world.shape[:-1]:
        raise ValueError("base_yaw shape must match wrench batch dimensions")
    cosine, sine = torch.cos(base_yaw), torch.sin(base_yaw)

    def rotate(vector):
        x = cosine * vector[..., 0] + sine * vector[..., 1]
        y = -sine * vector[..., 0] + cosine * vector[..., 1]
        return torch.stack((x, y, vector[..., 2]), dim=-1)

    return rotate(force_world), rotate(torque_world)


def planar_intent_label(force_world, torque_world, base_yaw):
    force, torque = world_wrench_to_base_yaw(force_world, torque_world, base_yaw)
    return torch.stack((force[..., 0], force[..., 1], torque[..., 2]), dim=-1)


class BatchedIntentHistory:
    """GPU-resident history for vectorized Isaac Lab environments."""

    def __init__(self, num_envs, history_steps, joint_count, device):
        if num_envs < 1 or history_steps < 1 or joint_count < 1:
            raise ValueError("history dimensions must be positive")
        shape = (num_envs, history_steps, joint_count)
        self.q = torch.zeros(shape, device=device)
        self.dq = torch.zeros(shape, device=device)
        self.action = torch.zeros(shape, device=device)

    def reset(self, env_ids, q, dq, previous_action):
        self.q[env_ids] = q[:, None, :].expand(-1, self.q.shape[1], -1)
        self.dq[env_ids] = dq[:, None, :].expand(-1, self.dq.shape[1], -1)
        self.action[env_ids] = previous_action[:, None, :].expand(-1, self.action.shape[1], -1)

    def append(self, q, dq, previous_action):
        self.q = torch.roll(self.q, shifts=-1, dims=1)
        self.dq = torch.roll(self.dq, shifts=-1, dims=1)
        self.action = torch.roll(self.action, shifts=-1, dims=1)
        self.q[:, -1] = q
        self.dq[:, -1] = dq
        self.action[:, -1] = previous_action

    def features(self):
        return build_intent_features(self.q, self.dq, self.action)

