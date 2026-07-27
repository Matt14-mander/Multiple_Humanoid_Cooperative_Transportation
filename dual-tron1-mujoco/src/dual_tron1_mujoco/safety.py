"""Simulation safety checks that mirror future hardware watchdog concerns."""

import math

import mujoco
import numpy as np


class SafetyMonitor:
    def __init__(self, model, minimum_base_height=0.30, maximum_tilt_rad=1.0):
        self.minimum_base_height = float(minimum_base_height)
        self.maximum_tilt_rad = float(maximum_tilt_rad)
        self.body_ids = [
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, prefix + "base_Link"
            )
            for prefix in ("r1_", "r2_")
        ]

    def check(self, data):
        if not np.all(np.isfinite(data.qpos)) or not np.all(
            np.isfinite(data.qvel)
        ):
            raise FloatingPointError("MuJoCo state became non-finite")
        for number, body_id in enumerate(self.body_ids, start=1):
            height = float(data.xpos[body_id, 2])
            body_z = data.xmat[body_id].reshape(3, 3)[:, 2]
            tilt = math.acos(float(np.clip(body_z[2], -1.0, 1.0)))
            if height < self.minimum_base_height:
                raise RuntimeError(
                    "robot {} base height {:.3f} m crossed safety limit".format(
                        number, height
                    )
                )
            if tilt > self.maximum_tilt_rad:
                raise RuntimeError(
                    "robot {} tilt {:.3f} rad crossed safety limit".format(
                        number, tilt
                    )
                )
