"""Backend-neutral actor observation construction."""

import numpy as np

from ..hindsight import HistoryBuffer


class ActorObservationBuilder:
    def __init__(self, spec):
        self.spec = spec
        self.proprioception = HistoryBuffer(
            spec.history_steps, (len(spec.proprioception_names),)
        )
        self.actions = HistoryBuffer(spec.history_steps, (spec.action_size,))

    def reset(self, proprioception, previous_action=None):
        action = np.zeros(self.spec.action_size) if previous_action is None else previous_action
        self.proprioception.reset(proprioception)
        self.actions.reset(action)

    def append(self, proprioception, previous_action):
        self.proprioception.append(proprioception)
        self.actions.append(previous_action)

    def build(self, goal_position, compliance):
        goal = np.asarray(goal_position, dtype=float)
        compliance = np.asarray(compliance, dtype=float)
        if goal.shape != (3,) or compliance.shape != (3,):
            raise ValueError("goal_position and compliance must have shape (3,)")
        observation = np.concatenate(
            [self.proprioception.array().reshape(-1),
             self.actions.array().reshape(-1), goal, compliance]
        )
        if observation.size != self.spec.actor_observation_size:
            raise RuntimeError("actor observation does not match PolicySpec")
        return observation

