# 内力抑制模块 - 快速入门指南

## 🚀 5分钟快速开始

### 步骤 1: 验证项目结构

```bash
cd dual-tron1-mujoco
python validate_ifsm_structure.py
```

**预期输出**:
```
✓ Found files: 18
✗ Missing files: 0
🎉 SUCCESS! All required files are in place.
```

---

### 步骤 2: 运行力估计演示

```bash
cd examples/internal_force_suppression
python demo_force_estimation.py
```

**这个演示做什么**:
- 创建一个简单的2自由度机器人
- 应用正弦外力
- 使用动量观测器估计力
- 显示真实力 vs 估计力的对比

**预期输出**:
```
✓ Created robot model with 2 DOF
✓ Initialized force estimator
✓ Running simulation for 2.0s at 500 Hz...

Results:
RMSE: ~0.5 N⋅m
Max Error: ~2.0 N⋅m
```

---

### 步骤 3: 运行单元测试

```bash
cd dual-tron1-mujoco
python -m pytest tests/internal_force_suppression/ -v
```

**预期看到**:
```
test_force_estimator.py::TestGeneralizedMomentumObserver::test_initialization PASSED
test_force_estimator.py::TestGeneralizedMomentumObserver::test_reset PASSED
test_force_estimator.py::TestGeneralizedMomentumObserver::test_zero_external_force PASSED
...
```

---

### 步骤 4: 查看配置文件

```bash
cat configs/internal_force_suppression/default_ifsm.yaml
```

**关键参数**:
- `observer_gain`: 100.0 (力估计器增益)
- `residual_gain`: 0.3 (残差控制增益)
- `max_safe_internal_force`: 50.0 N (安全阈值)

---

## 📖 核心概念 (3分钟理解)

### 问题: 什么是内力？

当两台机器人抓住同一个物体时：
```
机器人1 ←→ [物体] ←→ 机器人2
```

如果它们不完美协调：
- 机器人1向左拉 ← 
- 机器人2向右拉 →
- **结果**: 物体不动，但承受应力（内力）

### 解决方案: 三步法

**1. 估计力** (Force Estimator)
```python
# 使用动量观测器，无需力传感器
wrench = force_estimator.estimate_contact_wrench(robot_state)
```

**2. 分解力** (Force Analyzer)
```python
# 分解为有效力（移动物体）和内力（压迫物体）
force_info = force_analyzer.analyze(wrench1, wrench2, object_state)
# force_info['F_effective'] → 移动物体
# force_info['F_internal'] → 内力（需要抑制）
```

**3. 抑制内力** (Admittance Controller)
```python
# 让机器人对内力"柔顺"，自动调整以减少内力
action, diagnostics = admittance_controller.compute_residual_action(
    base_action, force_info, dt
)
# action = base_action + 柔顺修正
```

---

## 💻 代码示例

### 最小可用示例

```python
from internal_force_suppression.integrated_controller import DualRobotCooperativeController
import numpy as np

# 假设你有这些（实际需要替换为真实对象）
class DummyPolicy:
    def predict(self, obs):
        return np.zeros(10), np.zeros(10)  # 返回两个机器人的动作

motion_policy = DummyPolicy()
# robot1_model = ... (你的Pinocchio模型)
# robot2_model = ... (你的Pinocchio模型)

# 创建控制器
controller = DualRobotCooperativeController(
    motion_policy=motion_policy,
    robot1_model=robot1_model,
    robot2_model=robot2_model
)

# 控制循环
dt = 0.002  # 500 Hz
for step in range(1000):
    # 准备观测
    observation = {
        'robot1_state': {
            'q': np.zeros(10),   # 关节位置
            'v': np.zeros(10),   # 关节速度
            'tau': np.zeros(10)  # 关节力矩
        },
        'robot2_state': {
            'q': np.zeros(10),
            'v': np.zeros(10),
            'tau': np.zeros(10)
        },
        'object_state': {
            'com': np.array([0, 0, 1]),  # 物体质心
            'mass': 5.0                   # 物体质量
        }
    }
    
    # 执行控制
    result = controller.step(observation, dt)
    
    # 获取修正后的动作
    action1 = result['robot1_action']
    action2 = result['robot2_action']
    
    # 查看诊断信息
    internal_force = result['diagnostics']['internal_force']['magnitude']
    print(f"Step {step}: Internal force = {internal_force:.2f} N")
```

---

## 🔧 参数调优速查表

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 内力抑制不够 | 残差增益太小 | 提高 `residual_gain` (0.3 → 0.5) |
| 机器人震荡 | 阻尼太低 | 提高 `desired_damping` |
| 响应太慢 | 惯量太大 | 降低 `desired_inertia` |
| 力估计有延迟 | 观测器增益低 | 提高 `observer_gain` (100 → 150) |
| 力估计噪声大 | 观测器增益高 | 降低 `observer_gain` (100 → 70) |
| 基础策略被干扰 | 残差增益太大 | 降低 `residual_gain` (0.3 → 0.2) |

---

## 📚 下一步学习路径

### 新手路径 (1-2周)

**Week 1: 理解理论**
1. 阅读 `src/internal_force_suppression/README.md`
2. 运行所有示例和测试
3. 查看配置文件，理解每个参数
4. 修改参数，观察效果变化

**Week 2: 动手实践**
1. 阅读核心论文（至少 De Luca 2005）
2. 逐行阅读 `force_estimator.py`
3. 尝试修改示例代码
4. 添加自己的测试用例

### 中级路径 (2-4周)

**集成到你的项目**
1. 准备你的机器人Pinocchio模型
2. 实现观测接口（获取 q, v, tau）
3. 集成 `DualRobotCooperativeController`
4. 调优参数
5. 评估内力抑制效果

### 高级路径 (1-3月)

**扩展和改进**
1. 实现自适应参数调度
2. 添加学习型残差控制
3. 支持多物体/多机器人
4. 发论文 📝

---

## 🆘 常见问题 FAQ

### Q1: 必须使用Pinocchio吗？

**A**: 核心算法需要机器人动力学（质量矩阵M、非线性项nle、雅可比J）。Pinocchio是推荐工具，但你也可以：
- 用其他库（Drake, DART）提供相同接口
- 直接提供动力学矩阵

### Q2: 可以用于单臂吗？

**A**: 可以！内力分析部分针对双臂，但力估计和导纳控制是通用的。

### Q3: 实时性能如何？

**A**: 设计目标是500 Hz。大部分计算是矩阵运算（Pinocchio优化过），应该能在2ms内完成。

### Q4: 需要真实力传感器吗？

**A**: 不需要！这就是"隐式感知"的意义——通过关节力矩估计接触力。

### Q5: 与RL策略兼容吗？

**A**: 完全兼容！设计就是为了叠加在任何策略上。只要你的策略有 `.predict(obs)` 方法即可。

---

## 📞 获取帮助

### 自助资源
1. **完整文档**: `src/internal_force_suppression/README.md`
2. **项目总结**: `IFSM_PROJECT_SUMMARY.md`
3. **代码注释**: 每个文件都有详细注释和论文引用
4. **测试代码**: `tests/` 目录有使用示例

### 需要更多帮助？

如果遇到问题，提供以下信息：
- 你在做什么（使用场景）
- 遇到什么问题（错误信息/现象）
- 你尝试了什么
- 相关代码片段

---

## ✅ 检查清单

在开始集成前，确保：

- [ ] 运行过 `validate_ifsm_structure.py` ✅
- [ ] 运行过 `demo_force_estimation.py` ✅
- [ ] 运行过单元测试 ✅
- [ ] 阅读过 README.md（至少浏览）
- [ ] 理解三个核心组件的作用
- [ ] 查看过默认配置文件
- [ ] 准备好机器人模型（Pinocchio）
- [ ] 理解观测接口需求

**全部完成？** 🎉 你已准备好集成IFSM到你的项目！

---

**祝你成功！** 🚀

有问题随时问，我们一起让双机器人协作更安全、更高效！
