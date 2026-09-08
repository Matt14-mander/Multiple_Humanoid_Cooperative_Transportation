# TRON2 + 松灵机械臂 PAINT Intent Estimator 模块规划

## 1. 目标与边界

第一阶段只复现 PAINT 的显式平面意图估计器：从机械臂本体感知历史估计合作方施加在负载上的平面交互 wrench，部署时不使用末端六维力传感器或负载位姿跟踪。

估计目标为

\[
\boldsymbol\beta_t = [F_{x,t},F_{y,t},M_{z,t}]^\mathsf{T}.
\]

本模块暂不同时实现完整 PAINT 高层策略、双机器人协作和全六维 wrench。它应先作为独立、可测试、可导出的学习模块接入 `single-tron2-chip`，随后由 Isaac Lab 的 TRON2_YG 任务调用。

术语上，PAINT 式 (8) 是使用仿真真实 wrench 标签的**监督回归**，不是严格意义上的无标签自监督。其“无力传感器”指部署阶段不再读取真实 wrench。教师策略到学生策略的 KL 蒸馏是另一条训练目标，不能与估计器回归损失混为一谈。

## 2. 论文对应公式

输入采用长度为 \(H\) 的机械臂历史：

\[
\mathbf{o}^{\mathrm{trans}}_t =
\left[
\mathbf q^a_{t-H+1:t},
\dot{\mathbf q}^a_{t-H+1:t},
\mathbf a^a_{t-H:t-1}
\right].
\]

对六轴松灵机械臂，每个时间步包含 18 维；取论文默认的 \(H=4\)、高层控制周期 0.02 s 时，展平输入为 72 维并覆盖 0.08 s 历史。

估计器为

\[
\hat{\boldsymbol\beta}_t=f_\phi(\mathbf{o}^{\mathrm{trans}}_t)
=[\hat F_{x,t},\hat F_{y,t},\hat M_{z,t}]^\mathsf{T}.
\]

基础回归损失为

\[
\mathcal L_{\mathrm{est}}=
\left\|\hat{\mathbf F}^{xy}_t-\mathbf F^{xy}_t\right\|_2^2
+\left\|\hat M_{z,t}-M_{z,t}\right\|_2^2.
\]

实现时应先对三个标签通道分别标准化，避免 N 和 N·m 的数值尺度让损失权重失衡：

\[
\tilde{\boldsymbol\beta}_t=
(\boldsymbol\beta_t-\boldsymbol\mu_\beta)\oslash\boldsymbol\sigma_\beta.
\]

完整学生训练再加入

\[
\mathcal L_{\mathrm{student}}
=\mathcal L_{\mathrm{PPO}}
+\lambda_{\mathrm{KL}}D_{\mathrm{KL}}(\pi_S\|\pi_T)
+\lambda_{\mathrm{est}}\mathcal L_{\mathrm{est}}.
\]

其中 teacher 读取真实 \(\boldsymbol\beta_t\)，student 只能读取机械臂历史和 \(\hat{\boldsymbol\beta}_t\)。

## 3. TRON2_YG 的前置修改

当前官方 TRON2_YG 配置将机械臂高刚度锁定，不能直接用于训练意图估计器。若 \(q^a\) 和 \(\dot q^a\) 几乎不随载荷变化，输入中没有可辨识的交互信息。

机械臂应改为“受控有限阻抗”，而不是完全失能或自由摆动：

\[
\boldsymbol\tau_a=
\mathbf K_p(\mathbf q_a^d-\mathbf q_a)
-\mathbf K_d\dot{\mathbf q}_a
+\boldsymbol\tau_g.
\]

第一版数据采集可保持 \(\mathbf q_a^d\) 为安全搬运姿态，仅降低到能产生可测微小偏移的刚度/阻尼，并保留位置、速度、力矩和碰撞限制。后续高层策略才输出 \(\mathbf q_a^d\)。

## 4. 推荐目录

平台无关部分放入：

```text
single-tron2-chip/src/tron2_chip/intent_estimation/
  __init__.py
  spec.py             # 输入顺序、历史长度、坐标系和归一化元数据
  history.py          # 批量环形历史缓存
  network.py          # 3x128 MLP，72 -> 3
  losses.py           # 标准化 MSE/Huber 与评估指标
  dataset.py          # 离线轨迹数据集和切分
  runtime.py          # 在线推理、复位、反归一化、低通和限幅
  export.py           # TorchScript/ONNX 导出及一致性检查
```

Isaac Lab 适配层放入：

```text
single-tron2-chip/src/tron2_chip/backends/isaaclab/
  intent_observations.py   # 读取 q_a、dq_a、上一拍 arm action
  intent_events.py         # wrench 生成、payload/COM/参数随机化
  intent_labels.py         # 仿真标签和坐标变换
  intent_metrics.py        # batched RMSE、延迟和泛化统计
```

服务器上的 `TRON2_YG_LAB` 任务只负责调用这些模块，不复制网络和历史缓存逻辑。推荐在 Isaac Lab 环境中用 editable install 引用 MHCT：

```bash
pip install -e ~/workspace/MHCT/single-tron2-chip[train]
```

## 5. 坐标系和标签约定

必须在 `IntentEstimatorSpec` 中固定以下契约：

- 机械臂关节顺序为明确列出的 6 个 joint name，禁止依赖 USD 枚举顺序。
- 标签定义为**合作方施加到负载上的外部 wrench**，不能混用抓取约束力、机器人对负载的力或动量观测器残差。
- 仿真先获取世界系真实 wrench，再统一变换到当前机器人 base-yaw frame；只保留 \(F_x,F_y,M_z\)。
- 明确力/力矩作用点。若 torque 参考点变化，需使用 wrench shift 公式后再生成标签。
- action history 保存实际送入机械臂位置控制器的目标，而非网络尚未经过缩放和安全过滤的原始输出。
- episode reset 时历史用当前状态重复填充，避免用全零历史制造伪瞬态。

现有广义动量观测器不能作为第一版训练标签，因为模型误差和工具重力偏置会污染标签。它应作为比较基线；后续可研究混合残差结构

\[
\hat{\boldsymbol\beta}
=\hat{\boldsymbol\beta}_{\mathrm{GMO}}
+f_\phi(\mathbf o^{\mathrm{trans}},\hat{\boldsymbol\beta}_{\mathrm{GMO}}),
\]

但必须在纯 PAINT 基线完成后单独做消融。

## 6. 数据生成

第一版在单机器人、平地、负载耦合场景采集。交互 wrench 直接施加在负载而不是机械臂 link 上，使用逐段线性 profile：10% ramp-up、80% hold、10% ramp-down。方向和幅值正负均匀覆盖。

建议按安全课程逐步扩大范围：

1. 小幅 \(F_x,F_y,M_z\)，固定 payload 和固定 arm pose，验证符号与可辨识性；
2. 扩大到机器人和机械臂可承受范围；
3. 随机 payload mass、固定但不同的 COM、抓取点和机械臂目标姿态；
4. 加入关节噪声、动作延迟、Kp/Kd、摩擦、基座运动和地形随机化；
5. 最后加入 OU/多频连续 wrench，覆盖更接近人类牵引的输入。

训练/验证/测试必须按 episode 或随机种子切分，不能把同一条轨迹的相邻窗口分到不同集合中。

建议同时记录：

```text
timestamp, episode_id,
arm_q[6], arm_dq[6], arm_action_prev[6],
w_partner_world[6], w_partner_base_yaw[3],
payload_mass, payload_com[3], arm_target[6],
base_pose/twist, contact_state, terrain_id
```

## 7. 实施里程碑

### M0：信号与坐标验证

- 解锁为有限阻抗机械臂；
- 单方向施加已知阶跃力/力矩；
- 验证 \(F_x,F_y,M_z\) 的符号、参考点、单位和时间对齐；
- 证明机械臂本体感知对三种输入有可测响应。

### M1：独立离线估计器

- 建立 H=4 的数据集和 3x128 MLP；
- 单独监督训练，不接 PPO；
- 导出 ONNX，并验证 PyTorch/ONNX 输出一致；
- 与零输出、低通启发式和现有 GMO 基线比较。

### M2：闭环 oracle 替换实验

- 在相同高层控制器中分别使用真实 wrench、估计 wrench、无 wrench；
- 确认用估计值替换真实值时性能只发生可接受退化；
- 检查网络误差是否导致底盘命令抖动或机械臂饱和。

### M3：PAINT teacher-student

- 冻结已经训练好的 TRON2 低层 locomotion policy；
- teacher 用真实平面 wrench，输出 \([v_x,v_y,\omega_z]\) 和 6 维 arm target；
- 预训练 estimator 后，student 使用 PPO + KL + regression 联合微调；
- 部署图中删除真实 wrench 和 payload state 的所有 actor 数据路径，critic 特权输入只存在于训练期。

### M4：双机器人迁移

- 两台机器人各自运行同一 estimator，不共享状态；
- 从单机+虚拟 leader 迁移到双机刚性负载；
- 评估外部意图与内部力的混淆；
- 再把质量/COM 估计、内部力抑制和 CHIP 柔顺层接入，而不是让 intent estimator 同时承担这些任务。

## 8. 测试与验收

单元测试：

- 历史窗口时序、reset 和 batch shape；
- 关节重排检测；
- world 到 base-yaw 的 wrench 变换；
- 作用点平移后的 torque；
- 标签标准化/反标准化；
- PyTorch 与 ONNX 推理一致性；
- actor 数据流无 privileged-label 泄漏。

离线指标：

- 每通道 RMSE/MAE：N、N·m；
- 方向余弦相似度、符号准确率和 \(R^2\)；
- wrench 起始响应延迟；
- 零 wrench 时的均值、标准差与漂移；
- 未见质量、COM、姿态和地形上的泛化误差。

闭环消融至少包含：

1. ground-truth wrench oracle；
2. PAINT estimator；
3. GMO；
4. PAINT + GMO residual（后续）；
5. 无 intent；
6. H=1、4、8 的历史长度比较。

闭环验收以运输行为为主，估计 RMSE 只是中间指标：速度/转向意图对齐、负载约束力、base 稳定性、机械臂动作平滑度、饱和比例和跌倒率必须同时报告。

## 9. 实机验证路径

最终部署不使用 FT 传感器，但开发验收仍需要独立真值。可临时使用末端六维力传感器、测力计/已知悬挂载荷或标定牵引装置，只用于记录和对照，不进入 actor 输入。

实机顺序为：固定底座单轴加载 -> 固定底座三通道 -> TRON2 站立加载 -> 低速移动 -> 人机搬运。每一步均保留关节限位、速度限位、估计输出限幅、低通和紧急停止。

## 10. 与 MHCT 现有模块的关系

- `intent estimator`：回答合作方希望向哪里移动，输出显式平面 wrench cue；
- `generalized momentum observer`：用动力学残差估计外部关节力矩/末端 wrench；
- `payload identifier`：估计负载质量和质心；
- `internal-force suppressor`：处理双机器人闭链中不推动物体的内力；
- `CHIP compliance`：规定受到扰动时末端如何柔顺响应。

这些模块互补但不能共享含义不清的“force”变量。所有接口都应带 frame、reference point、unit、timestamp 和 validity mask。

