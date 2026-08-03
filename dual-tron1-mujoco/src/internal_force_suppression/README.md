# Internal Force Suppression Module (IFSM)

**隐式内力感知 + 残差导纳策略** 模块，用于双机器人协作搬运的内力抑制。

---

## 📋 目录

- [概述](#概述)
- [理论基础](#理论基础)
- [项目结构](#项目结构)
- [安装](#安装)
- [快速开始](#快速开始)
- [配置说明](#配置说明)
- [API文档](#api文档)
- [示例](#示例)
- [测试](#测试)
- [参考文献](#参考文献)

---

## 概述

当两台人形机器人协作搬运物体时，它们形成一个闭链系统。如果协调不完美，会产生**内力**（internal forces）——这些力不会移动物体，但会对机器人和物体施加应力，可能导致：

- 机器人关节过载
- 物体变形或损坏
- 能量浪费
- 系统不稳定

本模块实现了一个模块化的内力抑制系统，可以在预训练运动策略的基础上**透明地**添加内力抑制功能。

### 核心特性

✅ **无力传感器**: 使用动量观测器估计接触力  
✅ **理论支撑**: 基于经典论文的成熟算法  
✅ **模块化设计**: 各组件独立，易于测试和扩展  
✅ **实时性能**: 适合500Hz控制循环  
✅ **安全监控**: 内置安全阈值和紧急停止  
✅ **可配置**: YAML配置文件，运行时可调参数  

---

## 理论基础

### 1. 力估计（Force Estimation）

**方法**: 广义动量观测器（Generalized Momentum Observer）

**理论** (De Luca & Mattone, 2005):
```
p = M(q) * v                    (广义动量)
τ_ext = dp/dt + C(q,v)*v + g(q) - τ_measured
```

通过观测器重构：
```
dr/dt = -K*r + τ_measured - h(q,v)
τ_ext_est = r + K*(p - p_0)
```

### 2. 内力分析（Internal Force Analysis）

**方法**: 零空间投影（Null Space Projection）

**理论** (Erhart & Hirche, 2015):
```
对于双机器人系统:
W_obj = G * F_contact          (抓取矩阵映射)

F_effective = G^+ * W_obj      (有效力，最小范数解)
F_internal = F_contact - F_effective  (内力，零空间分量)

满足: G * F_internal = 0       (不影响物体运动)
```

### 3. 残差导纳控制（Residual Admittance Control）

**方法**: 导纳动力学 + 残差叠加

**理论** (Hogan, 1985):
```
M*a + B*v + K*x = F_internal   (导纳方程)

其中:
M - 期望惯量
B - 期望阻尼
K - 期望刚度
x - 柔顺位移
```

残差动作叠加：
```
action_corrected = action_base + α * action_residual
```

---

## 项目结构

```
dual-tron1-mujoco/
├── src/internal_force_suppression/
│   ├── __init__.py                      # 模块入口
│   ├── integrated_controller.py         # 主控制器（使用这个！）
│   │
│   ├── core/                            # 核心算法
│   │   ├── force_estimator.py          # 力估计器
│   │   ├── internal_force_analyzer.py  # 内力分析器
│   │   └── admittance_controller.py    # 导纳控制器
│   │
│   ├── models/                          # 模型（未来扩展）
│   ├── estimators/                      # 估计器变体（未来扩展）
│   ├── controllers/                     # 控制器变体（未来扩展）
│   │
│   ├── utils/                           # 工具函数
│   │   ├── wrench_decomposition.py     # 力/力矩分解
│   │   └── safety_monitor.py           # 安全监控
│   │
│   └── config/                          # 配置管理
│       └── ifsm_config.py              # 配置加载器
│
├── configs/internal_force_suppression/
│   └── default_ifsm.yaml               # 默认配置（从这里开始）
│
├── examples/internal_force_suppression/
│   └── demo_force_estimation.py        # 力估计演示
│
└── tests/internal_force_suppression/
    └── test_force_estimator.py         # 单元测试
```

---

## 安装

### 依赖项

```bash
# 核心依赖
pip install numpy pinocchio pyyaml

# 可选（用于可视化和测试）
pip install matplotlib pytest
```

### 验证安装

```bash
cd dual-tron1-mujoco
python -m pytest tests/internal_force_suppression/ -v
```

---

## 快速开始

### 基本用法

```python
from internal_force_suppression.integrated_controller import DualRobotCooperativeController
import pinocchio as pin

# 1. 准备机器人模型（Pinocchio）
robot1_model = pin.buildModelFromUrdf("robot1.urdf")
robot2_model = pin.buildModelFromUrdf("robot2.urdf")

# 2. 准备预训练策略
class MyMotionPolicy:
    def predict(self, observation):
        # 你的预训练策略
        action1 = ...
        action2 = ...
        return action1, action2

motion_policy = MyMotionPolicy()

# 3. 创建集成控制器
controller = DualRobotCooperativeController(
    motion_policy=motion_policy,
    robot1_model=robot1_model,
    robot2_model=robot2_model,
    config=None  # 使用默认配置
)

# 4. 控制循环
for step in range(num_steps):
    # 获取观测
    observation = {
        'robot1_state': {'q': q1, 'v': v1, 'tau': tau1},
        'robot2_state': {'q': q2, 'v': v2, 'tau': tau2},
        'object_state': {'com': object_com, 'mass': object_mass},
        'robot1_contact_point': contact_point1,  # 可选
        'robot2_contact_point': contact_point2,  # 可选
    }
    
    # 执行控制步骤
    result = controller.step(observation, dt=0.002)
    
    # 应用动作
    apply_action(robot1, result['robot1_action'])
    apply_action(robot2, result['robot2_action'])
    
    # 查看诊断信息
    diagnostics = result['diagnostics']
    print(f"Internal force: {diagnostics['internal_force']['magnitude']:.2f} N")
```

### 使用自定义配置

```python
# 方法1: 使用配置文件
controller = DualRobotCooperativeController(
    motion_policy=motion_policy,
    robot1_model=robot1_model,
    robot2_model=robot2_model,
    config="configs/internal_force_suppression/my_config.yaml"
)

# 方法2: 使用字典覆盖
config_override = {
    'admittance_robot1': {
        'residual_gain': 0.5,  # 提高抑制强度
        'enable_gain_scheduling': True
    }
}

controller = DualRobotCooperativeController(
    motion_policy=motion_policy,
    robot1_model=robot1_model,
    robot2_model=robot2_model,
    config=config_override
)
```

---

## 配置说明

配置文件位于 `configs/internal_force_suppression/default_ifsm.yaml`

### 关键参数

#### 力估计器
```yaml
force_estimator:
  observer_gain: 100.0          # 观测器增益 (Hz)
                                # 更高 = 更快收敛，但更多噪声
                                # 典型: 50-200
  cutoff_frequency: null        # 低通滤波截止频率 (Hz)
                                # null = 不滤波（仿真推荐）
```

#### 内力分析
```yaml
force_analyzer:
  max_safe_internal_force: 50.0   # 最大安全内力 (N)
  warning_threshold: 35.0         # 警告阈值 (N)
```

#### 导纳控制器
```yaml
admittance_robot1:
  desired_inertia: [10, 10, 10, 1, 1, 1]     # 期望惯量
  desired_damping: [50, 50, 50, 5, 5, 5]     # 期望阻尼
  desired_stiffness: [100, 100, 100, 10, 10, 10]  # 期望刚度
  
  residual_gain: 0.3              # 残差增益 (0-1)
                                  # 更高 = 更强抑制，但可能干扰基础策略
  
  enable_gain_scheduling: false   # 自适应增益调度
```

### 参数调优建议

**保守配置** (优先稳定性):
```yaml
residual_gain: 0.2
desired_damping: [80, 80, 80, 8, 8, 8]  # 高阻尼
```

**激进配置** (优先内力抑制):
```yaml
residual_gain: 0.5
enable_gain_scheduling: true
desired_damping: [30, 30, 30, 3, 3, 3]  # 低阻尼，更柔顺
```

---

## API文档

### `DualRobotCooperativeController`

主控制器类。

#### 构造函数
```python
controller = DualRobotCooperativeController(
    motion_policy,      # 预训练策略，需要 .predict(obs) 方法
    robot1_model,       # Pinocchio Model
    robot2_model,       # Pinocchio Model
    config=None         # 配置（dict 或 path）
)
```

#### 主要方法

**`step(observation, dt)`**

执行一个控制步骤。

**参数**:
- `observation`: 观测字典
  - `robot1_state`: `{'q', 'v', 'tau'}`
  - `robot2_state`: `{'q', 'v', 'tau'}`
  - `object_state`: `{'com', 'mass'}`
  - （可选）`robot1_contact_point`, `robot2_contact_point`
- `dt`: 时间步长（秒）

**返回**:
```python
{
    'robot1_action': np.ndarray,
    'robot2_action': np.ndarray,
    'diagnostics': {
        'internal_force': {
            'magnitude': float,       # 内力大小 (N)
            'ratio': float,           # 内力比率 (0-1)
            'safety_status': str      # "safe", "warning", "danger"
        },
        'robot1': {...},              # 机器人1详细信息
        'robot2': {...},              # 机器人2详细信息
    }
}
```

**`reset()`**

重置所有组件状态。

**`get_statistics()`**

获取性能统计信息。

**`update_config(config_updates)`**

运行时更新配置。

---

## 示例

### 示例1: 力估计演示

```bash
cd examples/internal_force_suppression
python demo_force_estimation.py
```

这个示例：
- 创建简单的2-DOF机器人
- 应用正弦外力
- 使用动量观测器估计
- 绘制真实力 vs 估计力

### 示例2: 完整系统（待实现）

```python
# 即将推出: 完整的双机器人MuJoCo仿真示例
# demo_full_system.py
```

---

## 测试

### 运行所有测试

```bash
cd dual-tron1-mujoco
python -m pytest tests/internal_force_suppression/ -v
```

### 运行特定测试

```bash
# 只测试力估计器
python -m pytest tests/internal_force_suppression/test_force_estimator.py -v

# 运行特定测试
python -m pytest tests/internal_force_suppression/test_force_estimator.py::TestGeneralizedMomentumObserver::test_initialization -v
```

### 测试覆盖率

```bash
pip install pytest-cov
pytest tests/internal_force_suppression/ --cov=src/internal_force_suppression --cov-report=html
```

---

## 开发路线图

### ✅ Phase 1: 基础框架（已完成）
- [x] 项目结构搭建
- [x] 配置系统
- [x] 核心算法实现
- [x] 基础测试

### 🔄 Phase 2: 力估计（进行中）
- [x] 动量观测器实现
- [x] 力估计单元测试
- [ ] 卡尔曼滤波集成
- [ ] 仿真验证

### 📅 Phase 3: 内力分析（计划中）
- [ ] 闭链动力学建模
- [ ] 力分解算法测试
- [ ] 可视化工具

### 📅 Phase 4: 导纳控制（计划中）
- [ ] 导纳控制器测试
- [ ] 参数调优工具
- [ ] 与预训练策略集成

### 📅 Phase 5: 集成与优化（计划中）
- [ ] 完整系统MuJoCo仿真
- [ ] 性能优化
- [ ] 文档完善

---

## 参考文献

### 必读论文

1. **De Luca, A., & Mattone, R. (2005)**  
   "Sensorless robot collision detection and hybrid force/motion control."  
   *IEEE International Conference on Robotics and Automation (ICRA).*  
   📄 DOI: 10.1109/ROBOT.2005.1570465

2. **Erhart, S., & Hirche, S. (2015)**  
   "Internal force analysis and load distribution for cooperative multi-robot manipulation."  
   *IEEE International Conference on Robotics and Automation (ICRA).*  
   📄 DOI: 10.1109/ICRA.2015.7139504

3. **Hogan, N. (1985)**  
   "Impedance control: An approach to manipulation."  
   *Journal of Dynamic Systems, Measurement, and Control*, 107(1), 1-7.  
   📄 经典论文

### 扩展阅读

4. **Lee, J., et al. (2020)**  
   "Learning quadrupedal locomotion over challenging terrain."  
   *Science Robotics*, 5(47).  
   📄 残差控制架构参考

5. **Kumar, V., et al. (2021)**  
   "RMA: Rapid motor adaptation for legged robots."  
   *arXiv preprint arXiv:2107.04034.*  
   📄 隐式感知思想

---

## 常见问题

### Q: 如何选择观测器增益？

**A**: 从100 Hz开始，如果估计有延迟，提高到150-200 Hz。如果噪声太大，降低到50-80 Hz。

### Q: 残差增益应该设多大？

**A**: 从0.2-0.3开始。如果内力抑制不够，逐步提高到0.4-0.5。不要超过0.8，否则会过度干扰基础策略。

### Q: 如何确定最大安全内力？

**A**: 根据机器人和物体的规格。一般：
- 小型机器人 + 轻物体: 20-40 N
- 中型机器人 + 中等物体: 40-80 N
- 大型机器人 + 重物体: 80-150 N

### Q: 导纳参数如何调优？

**A**: 使用临界阻尼作为起点：`B = 2*sqrt(M*K)`。然后根据表现调整：
- 震荡 → 提高阻尼
- 响应慢 → 降低惯量或提高增益
- 太硬 → 降低刚度

---

## 贡献

欢迎贡献！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/amazing-feature`)
3. 提交更改 (`git commit -m 'Add amazing feature'`)
4. 推送到分支 (`git push origin feature/amazing-feature`)
5. 开启 Pull Request

### 代码风格

- 遵循 PEP 8
- 添加类型提示
- 编写docstring（Google风格）
- 添加单元测试

---

## 许可证

[待定]

---

## 联系方式

- 项目负责人: [Your Name]
- Email: [your.email@example.com]
- 项目链接: [GitHub URL]

---

## 致谢

本项目基于以下优秀工作：
- Pinocchio: 快速机器人动力学库
- MuJoCo: 物理仿真引擎
- 感谢De Luca, Erhart, Hogan等研究者的开创性工作

---

**最后更新**: 2026-08-03
