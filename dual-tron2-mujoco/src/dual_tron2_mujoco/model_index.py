from dataclasses import dataclass

import mujoco


@dataclass(frozen=True)
class JointAddress:
    qpos_adr: int
    dof_adr: int


class ModelIndex:
    def __init__(self, model):
        self.model = model

    def joint(self, name: str) -> JointAddress:
        joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, name
        )
        if joint_id < 0:
            raise KeyError("MuJoCo joint not found: " + name)
        return JointAddress(
            int(self.model.jnt_qposadr[joint_id]),
            int(self.model.jnt_dofadr[joint_id]),
        )

    def actuator(self, name: str) -> int:
        actuator_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if actuator_id < 0:
            raise KeyError("MuJoCo actuator not found: " + name)
        return actuator_id

