"""Causal arm-proprioception history used by training and deployment."""

from collections import deque

import numpy as np


class IntentHistoryBuffer:
    """Store oldest-to-newest q, dq and applied previous arm commands."""

    def __init__(self, spec):
        self.spec = spec
        self._samples = deque(maxlen=spec.history_steps)

    def reset(self, joint_position, joint_velocity=None, previous_action=None):
        q, dq, action = self._validate(joint_position, joint_velocity, previous_action)
        sample = (q, dq, action)
        self._samples.clear()
        self._samples.extend(tuple(value.copy() for value in sample) for _ in range(self.spec.history_steps))

    def append(self, joint_position, joint_velocity, previous_action):
        sample = self._validate(joint_position, joint_velocity, previous_action)
        if not self._samples:
            self.reset(*sample)
        else:
            self._samples.append(tuple(value.copy() for value in sample))

    def arrays(self):
        if len(self._samples) != self.spec.history_steps:
            raise RuntimeError("intent history has not been initialized")
        q, dq, action = zip(*self._samples)
        return np.stack(q), np.stack(dq), np.stack(action)

    def vector(self):
        """Return [q_history, dq_history, previous_action_history]."""
        q, dq, action = self.arrays()
        vector = np.concatenate((q.reshape(-1), dq.reshape(-1), action.reshape(-1)))
        if vector.shape != (self.spec.input_size,):
            raise RuntimeError("history vector does not match IntentEstimatorSpec")
        return vector

    def _validate(self, joint_position, joint_velocity, previous_action):
        n = self.spec.joint_count
        q = np.asarray(joint_position, dtype=np.float32)
        dq = np.zeros(n, dtype=np.float32) if joint_velocity is None else np.asarray(joint_velocity, dtype=np.float32)
        action = np.zeros(n, dtype=np.float32) if previous_action is None else np.asarray(previous_action, dtype=np.float32)
        for name, value in (("joint_position", q), ("joint_velocity", dq), ("previous_action", action)):
            if value.shape != (n,):
                raise ValueError("{} must have shape ({},)".format(name, n))
            if not np.all(np.isfinite(value)):
                raise ValueError(name + " contains non-finite values")
        return q, dq, action

