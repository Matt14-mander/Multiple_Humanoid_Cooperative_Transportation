"""测试脚本：改善MuJoCo仿真可视化效果"""

# 方法1：运行策略模式且解锁基座（机器人会真正移动）
# python -m dual_tron1_mujoco.run_sim --controller policy --unlock-bases --world-vx 0.5

# 方法2：运行forward测试（专门设计的移动测试）
# python -m dual_tron1_mujoco.run_sim --forward-test

# 方法3：自定义配置文件，添加地板网格纹理
import json
from pathlib import Path

def create_test_config():
    """创建一个解锁基座的测试配置"""
    config = {
        "model": {
            "robot_type": "WF_TRON1A",
            "timestep": 0.001,
            "robot_1_pose": [-0.7, 0.0, 0.92, 0.0],
            "robot_2_pose": [0.7, 0.0, 0.92, 3.141592653589793],
            "payload_pose": [0.0, 0.0, 1.16, 0.0],
            "payload_mass_kg": 10.0,
            "payload_body_size_m": [0.60, 0.24, 0.24],
            "handle_size_m": [0.32, 0.05, 0.05],
            "handle_center_x_m": 0.354,
            "fixed_bases": False,  # 关键：解锁基座
            "grasp_mode": "soft_weld"
        },
        "control": {
            "leg_kp": 40.0,
            "leg_kd": 1.8,
            "wheel_kd": 0.5,
            "arm_j123_kp": 18.0,
            "arm_j123_kd": 1.0,
            "arm_j456_kp": 4.0,
            "arm_j456_kd": 0.5,
            "leg_torque_limit": 80.0,
            "wheel_torque_limit": 40.0,
            "arm_j123_torque_limit": 18.0,
            "arm_j456_torque_limit": 3.0,
            "policy_decimation": 10,
            "action_scale_position": 0.25,
            "action_scale_velocity": 3.0,
            "minimum_base_height_m": 0.30,
            "maximum_tilt_rad": 1.0
        },
        "run": {
            "duration_s": 10.0,
            "realtime": True,  # 开启实时模式便于观察
            "record_csv": "runs/test_movement.csv"
        }
    }

    output_path = Path("configs/test_movement.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"配置文件已创建: {output_path}")
    print("\n运行命令:")
    print(f"python -m dual_tron1_mujoco.run_sim --config {output_path} --controller policy --world-vx 0.5")

if __name__ == "__main__":
    create_test_config()
