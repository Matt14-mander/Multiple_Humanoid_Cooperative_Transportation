# TRON + AIRBOT 协同搬运理论推导与实现笔记

> 文档定位：理论模型、公式推导、代码实现和实验验证之间的主索引。  
> 更新日期：2026-09-03  
> 公式约定：所有 wrench 均按 \([\mathbf f^T,\boldsymbol\mu^T]^T\) 排列；除非特别说明，笛卡尔量使用世界坐标系。

## 0. 为什么采用 Markdown + LaTeX

主笔记使用 Markdown，公式使用 Markdown 中的标准 LaTeX：

- 写论文：稳定公式可直接复制到 `equations.tex` 或论文 LaTeX；
- 做 PPT：PowerPoint 公式框支持大部分 LaTeX 输入，复制单个公式最快；
- 对照代码：Markdown 适合加入文件链接、运行命令、状态和实验表格；
- 版本控制：两者都是纯文本，Git 可以准确显示公式和论证的变化；
- 日常记录：Markdown 比纯 `.tex` 更适合快速记录中文 notes、问题和实验结果。

因此不建议只维护一个纯 LaTeX notes 文件，也不建议把公式只放在 Word/PPT 中。论文级排版保留在 `.tex`，知识和实现上下文保留在 `.md`。

---

## 1. 符号、坐标系与系统边界

### 1.1 机器人动力学符号

| 符号 | 含义 | 维度 |
|---|---|---:|
| \(\mathbf q,\mathbf v\) | 广义位置、广义速度 | \(n_q,n_v\) |
| \(\mathbf M(\mathbf q)\) | 质量矩阵 | \(n_v\times n_v\) |
| \(\mathbf C(\mathbf q,\mathbf v)\) | 科氏/离心矩阵 | \(n_v\times n_v\) |
| \(\mathbf g(\mathbf q)\) | 重力广义力 | \(n_v\) |
| \(\boldsymbol\tau\) | 测量或实际执行器力矩 | \(n_v\) |
| \(\boldsymbol\tau_{ext}\) | 外部作用产生的广义力矩 | \(n_v\) |
| \(\mathbf J_e\) | 末端几何雅可比 | \(6\times n_v\) |
| \(\mathbf h_e\) | 末端外部 wrench | \(6\) |

### 1.2 必须固定的系统边界

| 对象 | 是否进入机器人名义动力学 | 原因 |
|---|---|---|
| AIRBOT 固定夹爪、转接件、固定线缆 | 是 | 它们始终属于机器人，漏建模会变成静态虚假外力 |
| 搬运的哑铃、箱子、配重块 | 否 | 它们是待估计的独立物体，不能被偏置补偿吞掉 |
| 两机器人闭链抓持内力 | 否 | 它是真实交互量，需要估计并控制 |

**核心原则：固定工具参数进入 Pinocchio/MuJoCo 机器人模型；未知物体参数进入对象级估计器。**

---

## 2. 无力传感器广义动量观测器

### 2.1 刚体动力学

机械臂关节空间动力学为

$$
\mathbf M(\mathbf q)\dot{\mathbf v}
+\mathbf C(\mathbf q,\mathbf v)\mathbf v
+\mathbf g(\mathbf q)
=\boldsymbol\tau+\boldsymbol\tau_{ext}.
\tag{OBS-01}
$$

若末端是唯一外部作用位置，则

$$
\boldsymbol\tau_{ext}=\mathbf J_e^T\mathbf h_e.
\tag{OBS-02}
$$

定义广义动量

$$
\mathbf p=\mathbf M(\mathbf q)\mathbf v.
\tag{OBS-03}
$$

利用机械系统性质 \(\dot{\mathbf M}=\mathbf C+\mathbf C^T\)，可得

$$
\dot{\mathbf p}
=\boldsymbol\tau+\boldsymbol\tau_{ext}-\boldsymbol\beta,
\qquad
\boldsymbol\beta=\mathbf g-\mathbf C^T\mathbf v.
\tag{OBS-04}
$$

这一步避免直接计算噪声敏感的 \(\dot{\mathbf v}\)。

### 2.2 De Luca 型残差观测器

定义积分状态

$$
\mathbf z(t)=\int_0^t
\left(\boldsymbol\tau-\boldsymbol\beta+\mathbf r\right)d\xi,
\tag{OBS-05}
$$

当前实现的符号约定下，残差为

$$
\mathbf r
=\mathbf K_O\left[
(\mathbf p-\mathbf p_0)-\mathbf z
\right].
\tag{OBS-06}
$$

因此残差动态满足

$$
\dot{\mathbf r}
=\mathbf K_O(\boldsymbol\tau_{ext}-\mathbf r),
\tag{OBS-07}
$$

稳态时

$$
\mathbf r\rightarrow\boldsymbol\tau_{ext}.
\tag{OBS-08}
$$

离散实现采用

$$
\mathbf z_k=\mathbf z_{k-1}
+\Delta t(\boldsymbol\tau_k-\boldsymbol\beta_k+\mathbf r_{k-1}),
\quad
\mathbf r_k=\mathbf K_O[(\mathbf p_k-\mathbf p_0)-\mathbf z_k].
\tag{OBS-09}
$$

**[Implemented]** 对应代码：
`dual-tron1-mujoco/src/internal_force_suppression/core/force_estimator.py`。

### 2.3 一阶低通滤波

观测残差可选一阶低通：

$$
\alpha=\frac{2\pi f_c\Delta t}{1+2\pi f_c\Delta t},
\qquad
\bar{\mathbf r}_k=(1-\alpha)\bar{\mathbf r}_{k-1}+\alpha\mathbf r_k.
\tag{OBS-10}
$$

增大 \(\mathbf K_O\) 或 \(f_c\) 会提高响应速度，但同时放大离散误差和测量噪声。

---

## 3. 为什么模型误差会成为“虚假外力”

真实动力学与名义模型不一致时，观测残差近似为

$$
\mathbf r\approx
\mathbf J_e^T\mathbf h_e
+\Delta\mathbf M\dot{\mathbf v}
+\Delta\mathbf h
+\boldsymbol\tau_{tool}
+\boldsymbol\tau_{fric}
+\boldsymbol\tau_{act},
\tag{ERR-01}
$$

其中 \(\mathbf h=\mathbf C\mathbf v+\mathbf g\)。任何落入 \(\operatorname{col}(\mathbf J_e^T)\) 的误差都会在 wrench 重构时变成虚假末端力。

### 3.1 静止状态的判别

当机器人真正静止，\(\mathbf v=\dot{\mathbf v}=0\)，则

$$
\Delta\mathbf M\dot{\mathbf v}=0.
\tag{ERR-02}
$$

因此“只缩放动量计算中的 \(\mathbf M\)”不应产生稳态静态力。若修改 URDF 的连杆质量，则质量、质心和 \(\mathbf g(\mathbf q)\) 同时变化，静态偏差主要来自重力失配，而非纯惯性项。

### 3.2 已观察问题

- 名义惯性/质量低估 10% 时，零接触出现约 \(0.843\,\mathrm N\) 虚假力；
- 未建模 \(200\,\mathrm g\) 固定工具时，零接触出现约 \(1.134\,\mathrm N\) 虚假力。

上述数值说明观测器本身不能区分真实外力与模型失配，不能单凭“残差非零”宣称检测到接触。

---

## 4. 固定工具建模与双时间尺度偏置补偿

### 4.1 固定工具空间惯量合并

固定工具的空间惯量可写为

$$
\mathcal I_t=(m_t,\mathbf c_t,\mathbf I_t).
\tag{BIAS-01}
$$

将其通过固定变换表达在末端连杆坐标系，再与末端连杆空间惯量相加：

$$
\mathcal I_{last}^{new}
=\mathcal I_{last}\oplus{}^{last}\!\mathcal I_t.
\tag{BIAS-02}
$$

**[Implemented]** MuJoCo 中 `gripper_stub` 的质量、质心和惯量已合并到每个 AIRBOT 的 Pinocchio 末端模型。对应：
`dual-tron1-mujoco/src/dual_tron1_mujoco/mujoco_momentum_observer.py`。

### 4.2 无接触慢变偏置学习

对滤波后残差建立慢变偏置：

$$
\alpha_b=1-e^{-\Delta t/T_b},
\qquad
\hat{\mathbf b}_k=\hat{\mathbf b}_{k-1}
+\alpha_b(\bar{\mathbf r}_k-\hat{\mathbf b}_{k-1}).
\tag{BIAS-03}
$$

补偿后的外力矩估计为

$$
\hat{\boldsymbol\tau}_{ext,k}
=\bar{\mathbf r}_k-\hat{\mathbf b}_k.
\tag{BIAS-04}
$$

状态机约束：

$$
\dot{\hat{\mathbf b}}\ne0
\quad\text{仅在 free-space/release，}\qquad
\dot{\hat{\mathbf b}}=0
\quad\text{在 grasp/carry。}
\tag{BIAS-05}
$$

这解决了两类问题中的不同部分：

1. 固定工具的确定性重力误差由显式惯量建模从根源上消除；
2. 剩余零点、轻微参数误差和积分稳态偏差由无接触偏置学习消除；
3. 抓取后冻结，防止物体重量和持续协同内力被误学成偏置。

**[Implemented]** 对应 `ObserverBiasCompensator`。  
**[Partial]** 当前偏置是关节残差的常量慢变模型，还不是随姿态、温度或步态相位变化的函数。

---

## 5. 从关节外力矩重构末端 wrench

理想关系为

$$
\hat{\boldsymbol\tau}_{ext}=\mathbf J_e^T\hat{\mathbf h}_e.
\tag{WRE-01}
$$

当前采用带 Tikhonov 正则的最小二乘：

$$
\hat{\mathbf h}_e
=\arg\min_{\mathbf h}
\|\mathbf J_e^T\mathbf h-\hat{\boldsymbol\tau}_{ext}\|_2^2
+\lambda\|\mathbf h\|_2^2,
\tag{WRE-02}
$$

闭式解为

$$
\hat{\mathbf h}_e
=(\mathbf J_e\mathbf J_e^T+\lambda\mathbf I)^{-1}
\mathbf J_e\hat{\boldsymbol\tau}_{ext}.
\tag{WRE-03}
$$

**[Implemented]** 双 AIRBOT 的 shadow-mode MuJoCo 集成使用 `LOCAL_WORLD_ALIGNED` Jacobian 和相同的正则逆计算估计值与仿真真值。

**局限：** 奇异位形附近 wrench 不唯一；应同时报告关节空间 RMSE、最小奇异值/条件数，而不能只看笛卡尔 wrench RMSE。

---

## 6. 双机器人对象 wrench、内力与载荷分配

### 6.1 抓取矩阵

第 \(i\) 个抓取点相对物体质心的位置为

$$
\mathbf r_i=\mathbf p_i-\mathbf p_c.
\tag{GRASP-01}
$$

单个接触块为

$$
\mathbf G_i=
\begin{bmatrix}
\mathbf I&\mathbf0\\
[\mathbf r_i]_{\times}&\mathbf I
\end{bmatrix},
\qquad
\mathbf G=[\mathbf G_1\ \mathbf G_2].
\tag{GRASP-02}
$$

对象合 wrench：

$$
\mathbf w_o=\mathbf G\mathbf h,
\qquad
\mathbf h=[\mathbf h_1^T,\mathbf h_2^T]^T.
\tag{GRASP-03}
$$

### 6.2 有效 wrench 与内力分解

最小范数有效分量和内力分量为

$$
\mathbf h_{eff}=\mathbf G^+\mathbf w_o,
\qquad
\mathbf h_{int}=\mathbf h-\mathbf h_{eff}.
\tag{INT-01}
$$

由此

$$
\mathbf G\mathbf h_{int}=\mathbf0.
\tag{INT-02}
$$

**[Implemented]** `internal_force_analyzer.py` 已实现非加权 Moore–Penrose 分解。  
**缺陷：** 非加权最小范数不表达两台机器人力矩容量、稳定裕度和抓取摩擦差异。

### 6.3 Virtual Linkage：具有物理意义的内力坐标

Williams 和 Khatib 提出的 virtual linkage，不只把内力定义为抽象的
\(\ker(\mathbf G)\)，而是用连接各抓取点的虚拟闭链执行器来描述物体内部的
拉伸、压缩和局部扭矩。其意义是：Moore–Penrose 分解回答“哪部分不改变对象合
wrench”，virtual linkage 进一步回答“这部分对应哪一种可控制的物理内力”。

#### 6.3.1 只考虑接触力的多抓取模型

令第 \(i\) 个抓取点相对对象参考点的位置为 \(\mathbf r_i\)，接触力堆叠为

$$
\mathbf f=[\mathbf f_1^T,\ldots,\mathbf f_n^T]^T.
\tag{VL-01}
$$

对象合力与合力矩满足

$$
\begin{bmatrix}\mathbf f_r\\\boldsymbol\mu_r\end{bmatrix}
=\mathbf G_f\mathbf f,
\qquad
\mathbf G_f=
\begin{bmatrix}
\mathbf I&\cdots&\mathbf I\\
[\mathbf r_1]_{\times}&\cdots&[\mathbf r_n]_{\times}
\end{bmatrix}.
\tag{VL-02}
$$

以 \(\mathbf t\) 表示虚拟棱柱构件的轴向内力，以 \(\mathbf E\) 表示由接触拓扑
和连杆单位方向组成的映射。采用“\(\mathbf f_i\) 是机器人施加给物体的力、
\(t>0\) 表示压缩”的本文约定：

$$
\mathbf f=\mathbf B_v\mathbf t+\mathbf f_e,
\qquad
\mathbf G_f\mathbf B_v=\mathbf0.
\tag{VL-03}
$$

博客使用 \(\mathbf f=-\mathbf E\mathbf t+\mathbf f_e\)，因此本文
\(\mathbf B_v=-\mathbf E\)。当 \(\mathbf B_v\) 满列秩且有效分量与该内力基正交时，

$$
\hat{\mathbf t}
=\mathbf B_v^+\mathbf f
=(\mathbf B_v^T\mathbf B_v)^{-1}\mathbf B_v^T\mathbf f.
\tag{VL-04}
$$

符号必须跟“机器人作用于物体”还是“物体反作用于机器人”一起定义，不能只复制
\(\mathbf E\) 矩阵而忽略作用反作用关系。

#### 6.3.2 两机器人抓持的轴向内力

令

$$
\mathbf e_{12}=\frac{\mathbf p_2-\mathbf p_1}
{\|\mathbf p_2-\mathbf p_1\|},
\qquad
\mathbf B_{12}=\begin{bmatrix}\mathbf e_{12}\\-\mathbf e_{12}\end{bmatrix}.
\tag{VL-05}
$$

则正的 \(t_{12}\) 表示两台机器人沿连线向物体内部挤压：

$$
\mathbf f_{int,12}=\mathbf B_{12}t_{12},
\qquad
t_{12}=\frac12\mathbf e_{12}^T(\mathbf f_1-\mathbf f_2).
\tag{VL-06}
$$

可以把期望内力写成虚拟弹簧—阻尼模型：

$$
t_{12}^{*}
=k_t(d_0-d)-d_t\dot d,
\qquad
d=\|\mathbf x_2-\mathbf x_1\|.
\tag{VL-07}
$$

若只需要稳定夹持，更合适的是直接给定有上下限的非零参考值：

$$
t_{min}\le t_{12}^{*}\le t_{max},
\tag{VL-08}
$$

其中 \(t_{min}\) 由防滑条件决定，\(t_{max}\) 由物体允许挤压力、夹爪和关节力矩
上限决定。目标不应永远是 \(t_{12}=0\)，否则可能降低抓持稳定性。

#### 6.3.3 完整 6D 刚性抓取

对于 \(n\) 个非奇异刚性 6D 抓取，接触 wrench 共 \(6n\) 维，对象运动消耗
6 维，因此内力/内力矩空间通常为

$$
\dim\ker(\mathbf G)=6n-6=6(n-1).
\tag{VL-09}
$$

原始 virtual linkage 用 \(3(n-2)\) 个轴向虚拟构件以及 \(n\) 个三自由度球形
执行关节描述这些模式，总执行自由度为

$$
3(n-2)+3n=6(n-1).
\tag{VL-10}
$$

因此“需要 \(3(n-2)\) 个棱柱关节”只是在统计轴向构件，不能解释为完整内力空间
只有 \(3(n-2)\) 维。

双臂刚性抓取是原论文单独处理的特例：用接触力可形成一项沿两抓取点连线的内部
轴向力；再加五项内部力矩模式，共 6 个内部自由度。沿连线方向的两个接触力矩之
和会作用于对象，其差才是内部扭矩；垂直方向力矩还会和接触力偶耦合。因此在工程
实现中，不应直接把五个原始接触力矩分量当成严格的 \(\ker(\mathbf G)\) 基。

可采用以下数值上严格、物理上可解释的构造：先定义物理候选基
\(\mathbf B_{raw}\)，再投影到抓取矩阵零空间并正交化：

$$
\mathbf N_G=\mathbf I-\mathbf G^+\mathbf G,
\qquad
\mathbf B_{VL}=\operatorname{orth}(\mathbf N_G\mathbf B_{raw}),
\qquad
\mathbf G\mathbf B_{VL}=\mathbf0.
\tag{VL-11}
$$

对应的 virtual-linkage 内力坐标为

$$
\boldsymbol\eta_{VL}=\mathbf B_{VL}^+\mathbf h_{int},
\qquad
\mathbf h_{int}=\mathbf B_{VL}\boldsymbol\eta_{VL}.
\tag{VL-12}
$$

#### 6.3.4 与当前项目实现的关系

当前实现使用

$$
\mathbf h_{int}=(\mathbf I-\mathbf G^+\mathbf G)\mathbf h,
\tag{VL-13}
$$

它能精确满足 \(\mathbf G\mathbf h_{int}=0\)，但输出的是 12 维接触 wrench 中的
零空间向量，没有直接给出“夹紧力是多少”。Virtual linkage 应作为这一结果之上的
物理坐标层，而不是未经验证地替换现有零空间投影。

针对 TRON + AIRBOT 双臂箱体搬运，建议第一阶段只增加最清晰、最需要控制的
轴向量：

$$
\hat t_{12}=\frac12\mathbf e_{12}^T
(\hat{\mathbf f}_1-\hat{\mathbf f}_2),
\tag{VL-14}
$$

然后在对象载荷分配中加入 \(t_{12}\) 的参考值和上下限。其余 5 维仍由完整
\(\ker(\mathbf G)\) 投影监测，等坐标系、力矩估计和摩擦模型验证后再逐项控制。

**[Implemented]** 当前已有式 VL-13 的 Moore–Penrose 零空间分解。  
**[Planned]** VL-05～VL-08 的双臂轴向虚拟连杆坐标、非零夹紧力参考和约束尚未接入控制器。  
**[Planned]** VL-11～VL-12 的完整 6D 物理内力坐标需要增加基连续性处理，避免 SVD/QR 基在相邻时刻跳变。

### 6.4 对象阻抗与重力支撑

期望对象力采用

$$
\mathbf f_o^*=
\mathbf K_p(\mathbf x_o^*-\mathbf x_o)
+\mathbf D_p(\dot{\mathbf x}_o^*-\dot{\mathbf x}_o)
-\hat m_o\mathbf g,
\tag{LOAD-01}
$$

期望对象力矩采用

$$
\boldsymbol\mu_o^*=
\mathbf K_R\mathbf e_R
+\mathbf D_R(\boldsymbol\omega_o^*-\boldsymbol\omega_o).
\tag{LOAD-02}
$$

### 6.5 加权载荷分配

在无关节约束时，当前加权最小范数解为

$$
\mathbf h^*=\mathbf W^{-1}\mathbf G^T
(\mathbf G\mathbf W^{-1}\mathbf G^T)^+
\mathbf w_o^*.
\tag{LOAD-03}
$$

若预测关节力矩越界，则求解带约束二次规划：

$$
\min_{\mathbf h}
\frac12\mathbf h^T\mathbf W\mathbf h
+\frac{\rho}{2}\|\mathbf G\mathbf h-\mathbf w_o^*\|_2^2,
\tag{LOAD-04}
$$

满足

$$
\boldsymbol\tau_{min}
\le\boldsymbol\tau_{base}+\mathbf J^T\mathbf h
\le\boldsymbol\tau_{max}.
\tag{LOAD-05}
$$

**[Implemented]** 对象阻抗、COM-aware 抓取矩阵、加权分配和关节力矩限制已经接入 `CooperativeCarryHoldController`。  
**[Partial]** 尚未完整加入抓取摩擦锥、足端稳定裕度和显式期望内力变量。

### 6.6 协同搬运阻抗控制：阅读笔记与重新推导

**阅读入口：** star2dust，[机械臂协同搬运中的阻抗控制](https://star2dust.github.io/post/impedance-control/)，2020-08-17；记录于 2026-09-03。

**博客提要：** 文章比较对象层阻抗和末端分布阻抗，并以平面双臂仿真说明控制过程。前者指定物体对外力的动态响应，再分配接触载荷；后者在各末端设置柔顺动态。内力作为不改变对象合 wrench 的自由分量加入。

**阅读边界：** 下文使用本项目的符号从牛顿–欧拉方程重新推导，不是博客全文转载，也不宣称已逐式复现其所有参考论文。对象阻抗的研究定位已核对 [Schneider–Cannon 的 NASA 文献记录](https://ntrs.nasa.gov/citations/19920068516)；任务空间动力学依据 [Khatib 的 Operational Space Formulation](https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)。博客与下文存在记号和适用条件差异，详见 6.6.6。

#### 6.6.1 先区分三种力与两类控制接口

- \(\mathbf w_{ext}\)：除抓持机器人及已建模重力以外，环境/人施加给物体的合 wrench。
- \(\mathbf h_i\)：机器人 \(i\) 施加给物体的接触 wrench；物体施加给机器人的反作用为 \(-\mathbf h_i\)。
- \(\mathbf h_{int}\)：接触 wrench 的内部分量，满足 \(\mathbf G\mathbf h_{int}=0\)。

这三者不能混用。特别是，动量观测器估计的末端接触反力不是可直接代入对象模型的 \(\mathbf w_{ext}\)；将其同时计入 \(\mathbf G\mathbf h\) 和 \(\mathbf w_{ext}\) 会重复计算。

阻抗控制把运动偏差映射为作用力/力矩；导纳控制把测得的力输入虚拟动力学，积分得到运动参考。两者可以共享质量–阻尼–刚度方程，但控制接口不同。当前 `ResidualAdmittanceController` 属于后者，不能仅凭方程形式把它称为博客的分布阻抗控制器。

#### 6.6.2 期望阻抗及其物理意义

先考虑平移坐标，定义 \(\mathbf e_x=\mathbf x-\mathbf x_d\)，期望动态为

$$
\mathbf M_d\ddot{\mathbf e}_x+\mathbf D_d\dot{\mathbf e}_x
+\mathbf K_d\mathbf e_x=\mathbf f_{ext}.
\tag{IMP-01}
$$

其中 \(\mathbf M_d\succ0\)、\(\mathbf D_d\succ0\)、\(\mathbf K_d\succ0\)，平移部分的单位分别为 kg、N·s/m、N/m。它们是希望外界感受到的动态参数，并不要求等于物体的真实参数。

给定静止参考及恒定外力，稳态满足

$$
\mathbf e_{x,ss}=\mathbf K_d^{-1}\mathbf f_{ext},
\qquad \mathbf C_f=\mathbf K_d^{-1}.
\tag{IMP-02}
$$

这与 CHIP 的目标顺应性相联系，但只是稳态关系，不代表二者动态控制器相同。对于单轴，固有频率与阻尼比为

$$
\omega_n=\sqrt{k_d/m_d},\qquad
\zeta=\frac{d_d}{2\sqrt{m_d k_d}}.
\tag{IMP-03}
$$

固定参考、常参数的理想平移闭环还可定义储能

$$
V=\frac12\dot{\mathbf e}_x^T\mathbf M_d\dot{\mathbf e}_x
+\frac12\mathbf e_x^T\mathbf K_d\mathbf e_x,
\quad
\dot V=\dot{\mathbf e}_x^T\mathbf f_{ext}
-\dot{\mathbf e}_x^T\mathbf D_d\dot{\mathbf e}_x.
\tag{IMP-04}
$$

这说明该理想目标系统的能量耗散性质；不能直接据此宣称存在时延、饱和和行走耦合的完整机器人系统已经得到稳定性证明。

#### 6.6.3 对象动力学到对象阻抗控制律

选物体质心为参考点、使用世界系表达，令 \(\boldsymbol\nu_o=[\mathbf v_c^T,\boldsymbol\omega^T]^T\)。则

$$
\mathbf M_o\dot{\boldsymbol\nu}_o+\mathbf b_o
=\mathbf w_{ext}+\mathbf G\mathbf h,
\quad
\mathbf M_o=\operatorname{diag}(m_o\mathbf I_3,\mathbf I_W),
\quad
\mathbf b_o=\begin{bmatrix}-m_o\mathbf g\\
\boldsymbol\omega\times(\mathbf I_W\boldsymbol\omega)\end{bmatrix}.
\tag{IMP-05}
$$

其中 \(\mathbf I_W=\mathbf R_{WB}\mathbf I_B\mathbf R_{WB}^T\) 是世界系质心惯量。\(\mathbf b_o\) 是 6 维偏置 wrench，含重力与陀螺项，不是一个科氏矩阵。

在平移或一致的小角度局部任务坐标下，以 \(\mathbf e_o\) 表示位姿误差，\(\mathbf e_\nu\) 表示速度误差，设置目标加速度

$$
\mathbf a_{imp}=\mathbf a_{ref}
+\mathbf M_d^{-1}(\mathbf w_{ext}
-\mathbf D_d\mathbf e_\nu-\mathbf K_d\mathbf e_o).
\tag{IMP-06}
$$

消去实际加速度，可得对象控制 wrench

$$
\mathbf w_{cmd}=\mathbf b_o+\mathbf M_o\mathbf a_{ref}
-\mathbf M_o\mathbf M_d^{-1}(\mathbf D_d\mathbf e_\nu+\mathbf K_d\mathbf e_o)
+(\mathbf M_o\mathbf M_d^{-1}-\mathbf I_6)\mathbf w_{ext}.
\tag{IMP-07}
$$

只要抓取映射满行秩、控制 wrench 可实现且模型准确，可以分配

$$
\mathbf h_{cmd}=\mathbf G^+\mathbf w_{cmd}
+\mathbf B_{VL}\boldsymbol\eta^*,
\qquad \mathbf G\mathbf B_{VL}=\mathbf0.
\tag{IMP-08}
$$

第一项负责对象运动/环境柔顺，第二项负责夹紧等内部载荷。带力矩、摩擦等约束时，应回到 LOAD-04 的 QP；截断接触力以后，一般不能继续声称严格实现式 IMP-01。

**对“无力传感器”的启示：** 当 \(\mathbf M_d=\mathbf M_o\) 时，式 IMP-07 中显式的外力反馈项消失；保留自然惯性也可产生一定柔顺响应。若要求任意指定不同的表观惯性，则上述精确模型整形一般需要 \(\mathbf w_{ext}\) 或其估计。不能把所有无力传感器阻抗都等同于完整惯量整形。

#### 6.6.4 分布式末端阻抗与静态参考偏移

考虑固定基座的平移末端任务，\(\mathbf f_i\) 仍定义为机器人输出给物体的力。博客采用的局部目标形式为

$$
\mathbf M_{d,i}\ddot{\mathbf x}_i
+\mathbf D_{d,i}(\dot{\mathbf x}_i-\dot{\mathbf x}_{d,i})
+\mathbf K_{d,i}(\mathbf x_i-\mathbf x_{d,i})=-\mathbf f_i.
\tag{DIMP-01}
$$

该形式对实际加速度建模，没有包含参考加速度前馈。若采用标准误差阻抗，则把加速度项改成 \(\mathbf M_{d,i}(\ddot{\mathbf x}_i-\ddot{\mathbf x}_{d,i})\)，并相应增加前馈。

在固定参考、零初值下，以 \(\boldsymbol\epsilon_i=\mathbf x_{d,i}-\mathbf x_i\) 为误差，完整拉普拉斯关系是

$$
\mathcal F_i(s)=(\mathbf M_{d,i}s^2+\mathbf D_{d,i}s+\mathbf K_{d,i})
\mathcal E_i(s).
\tag{DIMP-02}
$$

恒力下的稳定平衡给出

$$
\mathbf x_{d,i}=\mathbf x_i+\mathbf K_{d,i}^{-1}\mathbf f_i.
\tag{DIMP-03}
$$

这说明接触输出力需要由末端参考偏移产生。规划实现应围绕期望物体位姿给出的名义抓取位置构造

$$
\mathbf x_{d,i}^{cmd}=\mathbf x_{i}^{nom}
+\mathbf K_{d,i}^{-1}\mathbf f_i^*.
\tag{DIMP-04}
$$

不应每一步无条件把当前测量位置作为 \(\mathbf x_i^{nom}\)，否则会丢失对原对象轨迹的恢复作用。当前项目尚未实现此完整分布式控制链。

#### 6.6.5 操作空间动力学与关节力矩：独立核对符号

用 \(\mathbf c_q=\mathbf C_q\dot{\mathbf q}\) 表示关节速度偏置，刚体动力学为
\(\mathbf M_q\ddot{\mathbf q}+\mathbf c_q+\mathbf g_q=\boldsymbol\tau-\mathbf J^T\mathbf f_i\)。在 \(\mathbf J\) 满行秩时定义

$$
\boldsymbol\Lambda=(\mathbf J\mathbf M_q^{-1}\mathbf J^T)^{-1},
\qquad \bar{\mathbf J}=\mathbf M_q^{-1}\mathbf J^T\boldsymbol\Lambda.
\tag{DIMP-05}
$$

由 \(\ddot{\mathbf x}=\mathbf J\ddot{\mathbf q}+\dot{\mathbf J}\dot{\mathbf q}\) 得到

$$
\boldsymbol\Lambda\ddot{\mathbf x}+\boldsymbol\mu_x+\mathbf p_x
=\mathbf F_{cmd}-\mathbf f_i,
\quad
\boldsymbol\mu_x=\boldsymbol\Lambda(\mathbf J\mathbf M_q^{-1}\mathbf c_q-\dot{\mathbf J}\dot{\mathbf q}),
\quad
\mathbf p_x=\boldsymbol\Lambda\mathbf J\mathbf M_q^{-1}\mathbf g_q.
\tag{DIMP-06}
$$

这里 \(\mathbf F_{cmd}=\bar{\mathbf J}^{T}\boldsymbol\tau\)。\(-\boldsymbol\Lambda\dot{\mathbf J}\dot{\mathbf q}\) 的负号由加速度恒等式直接决定。

将 DIMP-01 代入，记 \(\mathbf s_i=\mathbf D_{d,i}(\dot{\mathbf x}_i-\dot{\mathbf x}_{d,i})+\mathbf K_{d,i}(\mathbf x_i-\mathbf x_{d,i})\)，得到

$$
\mathbf F_{cmd,i}=\boldsymbol\mu_{x,i}+\mathbf p_{x,i}
-\boldsymbol\Lambda_i\mathbf M_{d,i}^{-1}\mathbf s_i
+(\mathbf I-\boldsymbol\Lambda_i\mathbf M_{d,i}^{-1})\mathbf f_i.
\tag{DIMP-07}
$$

关节命令可采用

$$
\boldsymbol\tau_i=\mathbf J_i^T\mathbf F_{cmd,i}
+(\mathbf I-\mathbf J_i^T\bar{\mathbf J}_i^T)\boldsymbol\tau_{0,i}.
\tag{DIMP-08}
$$

第二项是动力学一致的力矩零空间项；它不同于当前解析控制器采用的运动学阻尼投影。上述推导假设理想力矩执行、无额外约束接触，不能原样用于带足端接触的浮动基座 TRON。

#### 6.6.6 博客公式用于本项目时的勘误与限制

以下是本笔记的独立检查结论，不代表作者已经发布勘误：

1. **6D 姿态表达：** \(\dot{\mathbf X}\) 若含欧拉角导数，不能直接等同于 twist \([\mathbf v;\boldsymbol\omega]\)。有限旋转需使用一致的姿态运动学映射与几何误差；IMP-06 的线性形式只用于平移或局部近似。
2. **惯量参考点：** IMP-05 的块对角质量矩阵以质心为参考点；若箱体原点与 COM 不重合，需要平移空间惯量和 wrench，不能同时混用两个参考点。
3. **拉普拉斯惯性项：** 在固定参考假设下，完整模型应包含 \(\mathbf M_ds^2\)。省略它属于降阶近似；静态式 DIMP-03 仍可成立。
4. **加速度变换符号：** 博客展示的速度偏置表达含正的 \(\dot{\mathbf J}\) 项；按本节固定基座几何雅可比约定，推导得到 DIMP-06 的负号。实现前应做逆动力学数值对照。
5. **点接触不能保证任意 6D 控制：** 两个不同位置的纯力接触，其 \(6\times6\) 抓取矩阵通常秩为 5，不能产生沿两接触点连线的纯合力矩。博客的平面演示不能直接代表 TRON2 的完整 3D 抓取。
6. **分布式不等于零通信：** 各末端本地执行阻抗，不自动解决对象状态、期望载荷和抓取几何的一致性；需要明确上层规划和消息接口。
7. **不能重复补偿载荷：** 对象重力只在对象层产生支撑需求；机器人固定工具重力在机器人动力学中补偿。不能把未知箱体质量同时加入机器人模型和对象支撑项。

#### 6.6.7 当前实现与建议验证

| 层级 | 当前实现 | 与本节理论的差距 |
|---|---|---|
| 对象阻抗 | LOAD-01/02：位姿 PD + 对象重力支撑 | 尚无完整 IMP-07 的惯量整形及环境 wrench 反馈 |
| 载荷分配 | 加权分配 + 力矩约束 QP | 需将非零内力参考纳入同一个约束问题 |
| AIRBOT 本地控制 | 机器人偏置力矩 + \(\mathbf J^T\) 载荷 | 不是完整 DIMP-07/08 操作空间控制 |
| TRON2 单机基线 | 重力补偿 + 笛卡尔弹簧阻尼 + 运动学姿态投影 | 不等于已训练 CHIP，也不等于任意指定表观惯性 |
| 内力抑制 | 残差导纳及有界修正 | 不等同于博客的末端分布阻抗 |

**[Planned]** 后续验证依次检查：静态 \(\Delta x=K^{-1}f\)；不同质量下的频率/阻尼响应；\(\dot J\dot q\) 偏置符号；零空间修正的 \(G\Delta h=0\)；接触力饱和后对象跟踪误差。本文档新增不改变控制器或测试代码。

---

## 7. 未知物体质量与质心辨识

### 7.1 低动态参数化

不直接估计完整 10 维空间惯性，先估计

$$
\boldsymbol\theta_o=
[m,\ mc_x,\ mc_y,\ mc_z]^T.
\tag{PAY-01}
$$

令物体原点世界系加速度为 \(\mathbf a_o\)，重力为 \(\mathbf g\)，并定义

$$
\mathbf u=\mathbf a_o-\mathbf g.
\tag{PAY-02}
$$

忽略角惯性项的低动态近似为

$$
\mathbf f_o=m\mathbf u,
\qquad
\boldsymbol\mu_o=-[\mathbf u]_{\times}\mathbf R_{WB}(m\mathbf c_B).
\tag{PAY-03}
$$

所以

$$
\mathbf w_o=
\underbrace{
\begin{bmatrix}
\mathbf u&\mathbf0_{3\times3}\\
\mathbf0_{3\times1}&-[\mathbf u]_{\times}\mathbf R_{WB}
\end{bmatrix}}_{\mathbf Y_o}
\boldsymbol\theta_o.
\tag{PAY-04}
$$

### 7.2 当前滑动窗口鲁棒估计

当前代码实际实现的是带岭正则和 Huber 重加权的滑动窗口最小二乘，而不是递归最小二乘：

$$
\hat{\boldsymbol\theta}
=\arg\min_{\boldsymbol\theta\in\Theta}
\sum_{k\in\mathcal W}w_k
\|\mathbf Y_k\boldsymbol\theta-\mathbf w_{o,k}\|_2^2
+\lambda\|\boldsymbol\theta-\boldsymbol\theta_0\|_2^2.
\tag{PAY-05}
$$

投影集合为

$$
m_{min}\le m\le m_{max},
\qquad
\mathbf c_{min}\le\frac{m\mathbf c}{m}\le\mathbf c_{max}.
\tag{PAY-06}
$$

可用条件包括：样本数足够、\(\operatorname{rank}(\mathbf Y)=4\)、条件数和残差 RMS 小于阈值。

### 7.3 冻结与突变监测

辨识完成后冻结参数，只监测创新量

$$
\mathbf e_k=\mathbf w_{o,k}-\mathbf Y_{o,k}\hat{\boldsymbol\theta}_{o}.
\tag{PAY-07}
$$

若 \(\|\mathbf e_k\|\) 连续超阈值，则进入 `reidentification_required`，但不在行走中静默吸收新的载荷。

**[Implemented]** `payload_estimator.py` 已实现滑窗、Huber、物理边界投影、可观性判定、冻结和创新监测。  
**[Partial]** 还需把估计器完整接入 TRON2 演示状态机，并用真实观测 wrench 替代仿真约束真值。

---

## 8. CHIP 风格柔顺学习与解析基线

### 8.1 Hindsight goal

训练时由参考目标、外力和顺应矩阵构造

$$
\mathbf g_{hind}=\mathbf g_{ref}-\mathbf C_f\mathbf f_{ext}.
\tag{CHIP-01}
$$

其中 \(\mathbf C_f\succeq0\)，单位为 \(\mathrm{m/N}\)。若只考虑对角顺应性：

$$
\mathbf C_f=\operatorname{diag}(C_x,C_y,C_z).
\tag{CHIP-02}
$$

训练 actor 输入

$$
\mathbf o_t^{actor}=
[\mathcal H(\mathbf s),\mathcal H(\mathbf a),
\mathbf g_{hind},\mathbf C_f],
\tag{CHIP-03}
$$

critic 可使用真实外力作为 privileged information；奖励仍跟踪原始 \(\mathbf g_{ref}\)。部署时 actor 不再输入外力，目标恢复为 \(\mathbf g_{ref}\)。

### 8.2 当前解析代理控制器

当前 MuJoCo 可见柔顺行为由解析控制器产生：

$$
\boldsymbol\tau=mathbf g(\mathbf q)
+\mathbf J^T\left[
\mathbf K_x(\mathbf g-\mathbf x)-\mathbf D_x\dot{\mathbf x}
\right]
+\mathbf N_J\boldsymbol\tau_{posture},
\tag{CHIP-04}
$$

并取

$$
\mathbf K_x\approx\mathbf C_f^{-1}.
\tag{CHIP-05}
$$

阻尼伪逆对应的运动学零空间投影为

$$
\mathbf N_J=\mathbf I-mathbf J^T
(\mathbf J\mathbf J^T+\lambda_J\mathbf I)^{-1}\mathbf J.
\tag{CHIP-06}
$$

**[Implemented]** `single-tron2-chip` 已有 hindsight 目标、历史缓存、解析阻抗 oracle、部署观测协议、安全过滤、评估和 Isaac Lab 配置骨架。  
**[Not trained]** 目前还没有 PPO 学得的 CHIP policy，不能把解析阻抗结果表述为“已复现 CHIP”。

---

## 9. 理论到代码的映射

| 公式 | 实现文件 | 验证入口 | 状态 |
|---|---|---|---|
| OBS-03～OBS-10 | `dual-tron1-mujoco/src/internal_force_suppression/core/force_estimator.py` | `tests/internal_force_suppression/test_force_estimator.py` | Implemented |
| BIAS-01～BIAS-05 | `mujoco_momentum_observer.py`, `force_estimator.py` | `test_airbot_observer_robustness.py`, `test_mujoco_momentum_observer.py` | Implemented |
| WRE-01～WRE-03 | `mujoco_momentum_observer.py` | `test_mujoco_momentum_observer.py` | Implemented |
| GRASP-01～INT-02 | `internal_force_analyzer.py` | `tests/internal_force_suppression/` | Implemented |
| VL-01～VL-14 | `internal_force_analyzer.py`（零空间部分） | 待增加轴向 virtual-linkage 测试 | Partial/Planned |
| LOAD-01～LOAD-05 | `dual_tron1_mujoco/carry_controller.py` | `test_carry_controller.py` | Implemented |
| IMP-01～IMP-08 / DIMP-01～DIMP-08 | 本文 6.6 节推导；现有对象/解析阻抗仅覆盖部分 | 待补惯量整形与动态阻抗辨识测试 | Partial/Planned |
| PAY-01～PAY-07 | `dual_tron1_mujoco/payload_estimator.py` | `test_payload_estimator.py` | Implemented/Integration pending |
| CHIP-01～CHIP-06 | `single-tron2-chip/` | `test_hindsight.py`, `test_core_deployment.py` | Analytic baseline |

---

## 10. 当前缺陷与下一步推导

### 10.1 高优先级

1. **真实执行力矩定义**：实机观测器必须使用电机侧估计的实际关节力矩，而不能无条件用命令力矩代替；需记录减速器效率、电流—力矩映射和饱和。
2. **浮动基座完整动力学**：当前双 AIRBOT MuJoCo 动量观测器明确限制为固定基座 shadow mode；TRON 行走需要处理基座状态和足端接触 wrench。
3. **外力源不可辨识性**：单姿态下固定工具重量与持续末端接触力可能等价，必须依赖无接触标定、抓取状态机或多姿态激励。
4. **坐标系和符号测试**：为 world、body、LOCAL_WORLD_ALIGNED 建立统一变换与正负号单元测试。
5. **Isaac Lab 真训练**：完成 WFYG_TRON2A USD、manager-based task、reward/event、PPO 和 ONNX 回放；当前只有契约骨架。

### 10.2 中优先级

1. 将常量偏置升级为 \(\hat{\boldsymbol\tau}_u=f_\phi(\mathbf q,\mathbf v,\mathbf a_{IMU},\boldsymbol\omega_{IMU},\boldsymbol\tau,\text{phase})\)。
2. 把摩擦 \(\boldsymbol\tau_{fric}\) 和固定工具误差分别做消融，避免所有误差只由一个 bias 解释。
3. 载荷分配加入抓取摩擦锥、足端支撑多边形、关节功率和热限制。
4. 对质心辨识动作计算 Fisher information 或 \(\lambda_{min}(\sum Y_k^TY_k)\)，按可观性选择动作。
5. 对移动配重或晃动物体使用时变参数/MHE；不要用固定刚体质心模型强行解释。

---

## 11. Notes / 决策日志

### 2026-09-02：文档形式

- 决定使用 `Markdown 主笔记 + equations.tex 公式库`。
- 主笔记负责中文解释、假设、实现状态、实验命令和缺陷；`.tex` 只存稳定公式。
- 公式编号保持稳定，不因章节移动而重编号。

### 2026-09-02：模型边界

- 固定夹爪和转接件属于机器人模型。
- 未知哑铃/箱子属于对象层，不能合并进 AIRBOT 末端惯量。
- 偏置只在无接触/释放阶段学习，抓取和搬运阶段冻结。

### 2026-09-02：CHIP 复现边界

- 当前结果是 CHIP 公式和数据接口的解析 oracle 验证。
- 只有完成 Isaac Lab PPO 训练，并在部署时移除真实外力输入后，才能称为学得的 CHIP-style policy。

### 2026-09-02：Virtual Linkage 内力建模

- 保留当前严格满足 \(\mathbf G\mathbf h_{int}=0\) 的 Moore–Penrose 分解。
- 在它上面增加物理坐标层，优先实现双臂连线方向的轴向挤压力 \(t_{12}\)。
- 夹持任务不以“全部内力为零”为唯一目标，而以满足防滑要求的最小非零轴向内力为目标。
- 博客的 \(-\mathbf E\mathbf t\) 符号来自作用力方向约定；项目统一采用机器人作用于物体的 wrench 后必须重新核对符号。

### 2026-09-03：对象阻抗与分布阻抗

已整理阻抗博客的对象层与末端层思路；补充 twist/欧拉角、惯量参考点、反力符号和完整拉普拉斯惯性项的限制。后续代码接入应先用固定基座验证，不直接把平面例子推广为行走中全身控制。

### 待补实验记录模板

```text
日期：
实验目的：
公式/假设：
代码版本（commit）：
运行命令：
配置：
CSV/图像路径：
主要指标：
结论：
异常与下一步：
```

---

## 12. 参考脉络（待统一 BibTeX）

1. De Luca & Mattone：广义动量残差、无力传感器碰撞检测。
2. Williams, D. & Khatib, O. (1993), *The Virtual Linkage: A Model for Internal Forces in Multi-Grasp Manipulation*, ICRA, pp. 1025--1030, DOI: 10.1109/ROBOT.1993.292110。原文：<https://khatib.stanford.edu/publications/pdfs/Williams_1993_ICRA.pdf>
3. 中文推导参考：star2dust，*机械臂协同搬运中的内力建模*：<https://star2dust.github.io/post/virtual-linkage/>
4. Erhart & Hirche：协同操作中的内力分析与载荷分配。
5. CHIP：基于 hindsight goal 的无力传感器柔顺策略学习。
6. 后续需要补充：模型不确定性补偿、在线载荷辨识、半参数摩擦模型和移动载荷估计相关文献。

本次阻抗阅读新增文献：

- star2dust，*机械臂协同搬运中的阻抗控制*，2020-08-17。[博客原文](https://star2dust.github.io/post/impedance-control/)。
- Schneider, S. A. & Cannon, R. H. (1992). *Object Impedance Control for Cooperative Manipulation: Theory and Experimental Results*. IEEE Transactions on Robotics and Automation, 8(3), 383–394。[NASA 记录](https://ntrs.nasa.gov/citations/19920068516)，DOI: 10.1109/70.143355。此次核对的是摘要/元数据，未取得该文完整正文。
- Szewczyk, J., Plumet, F. & Bidaud, P. (2002). *Planning and controlling cooperating robots through distributed impedance*. Journal of Robotic Systems, 19(6), 283–297。[出版商条目](https://onlinelibrary.wiley.com/doi/abs/10.1002/rob.10041)。此次未取得全文，不将博客公式视为该论文原式。
- Caccavale, F., Chiacchio, P., Marino, A. & Villani, L. (2008). *Six-DOF impedance control of dual-arm cooperative manipulators*. IEEE/ASME Transactions on Mechatronics, 13(5), 576–586。[DOI](https://doi.org/10.1109/TMECH.2008.2002816)。博客列出的后续阅读，此次未核对全文。
- Khatib, O. (1987). *A Unified Approach for Motion and Force Control of Robot Manipulators: The Operational Space Formulation*。[作者网站原文](https://khatib.stanford.edu/publications/pdfs/Khatib_1987_RA.pdf)。用于操作空间建模的理论背景；本节符号计算由动力学恒等式独立给出。

> [Question] 正式写论文前，应逐式核对原论文中的符号、观测器增益矩阵和坐标系定义，并建立 BibTeX；本笔记中的公式优先忠实描述当前代码的符号约定。
