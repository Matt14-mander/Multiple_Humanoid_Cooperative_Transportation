"""Name-based access to MuJoCo joints and actuators.

The upstream single-robot simulator uses fixed qpos/qvel offsets.  Those offsets
are invalid as soon as a second floating-base robot and a payload are added, so
all accesses in this project go through names resolved at model load time.
"""

from dataclasses import dataclass
from typing import Dict, Iterable

import mujoco


@dataclass(frozen=True)
class JointAddress:
    joint_id: int
    qpos_adr: int
    dof_adr: int


class ModelIndex:
    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self._joints: Dict[str, JointAddress] = {}
        self._actuators: Dict[str, int] = {}

    def joint(self, name: str) -> JointAddress:
        if name not in self._joints:
            joint_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_JOINT, name
            )
            if joint_id < 0:
                raise KeyError("MuJoCo joint not found: {}".format(name))
            self._joints[name] = JointAddress(
                joint_id=joint_id,
                qpos_adr=int(self.model.jnt_qposadr[joint_id]),
                dof_adr=int(self.model.jnt_dofadr[joint_id]),
            )
        return self._joints[name]

    def actuator(self, name: str) -> int:
        if name not in self._actuators:
            actuator_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
            )
            if actuator_id < 0:
                raise KeyError("MuJoCo actuator not found: {}".format(name))
            self._actuators[name] = actuator_id
        return self._actuators[name]

    def require_joints(self, names: Iterable[str]) -> None:
        for name in names:
            self.joint(name)

    def require_actuators(self, names: Iterable[str]) -> None:
        for name in names:
            self.actuator(name)

