import csv
from pathlib import Path

import mujoco
import numpy as np


class CsvRecorder:
    HEADER = [
        "time_s", "payload_x", "payload_y", "payload_z",
        "r1_base_x", "r1_base_y", "r1_base_z",
        "r2_base_x", "r2_base_y", "r2_base_z",
        "r1_grasp_gap_m", "r2_grasp_gap_m", "max_abs_ctrl",
    ]

    def __init__(self, path, model):
        self.path = Path(path)
        self.model = model
        self.rows = []
        self.body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in (
                "payload_body", "r1_base_Link", "r2_base_Link",
                "r1_gripper_pick", "payload_grasp_left",
                "r2_gripper_pick", "payload_grasp_right",
            )
        ]

    def sample(self, data):
        payload, r1, r2, r1_ee, left, r2_ee, right = self.body_ids
        self.rows.append([
            float(data.time), *data.xpos[payload], *data.xpos[r1], *data.xpos[r2],
            float(np.linalg.norm(data.xpos[r1_ee] - data.xpos[left])),
            float(np.linalg.norm(data.xpos[r2_ee] - data.xpos[right])),
            float(np.max(np.abs(data.ctrl))),
        ])

    def write(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(self.HEADER)
            writer.writerows(self.rows)

