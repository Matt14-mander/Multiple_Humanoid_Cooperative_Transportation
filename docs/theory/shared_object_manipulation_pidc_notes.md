# 多足机器人协同搬运：投影逆动力学、对象阻抗与接触力优化

> 论文：Shengzhi Wang, Niels Dehio, Xuanqi Zeng, Xian Yang, Lingwei Zhang, Yun-Hui Liu, and K. W. Samuel Au, *Shared Object Manipulation with a Team of Collaborative Quadrupeds*, arXiv:2510.00682v1, 2025。  
> 整理日期：2026-09-12。本文保留原论文式 (1)--(26) 的编号，补充标准 QP 形式、逐周期计算顺序和 TRON2 + 松灵机械臂适配说明。  
> 状态：**[Theory note / Planned]**，尚未在本项目的 TRON2 全身力矩控制器中实现。

## 1. 方法解决什么问题

论文把多台浮动基座多足机械臂视为一个受刚性接触约束的聚合系统，同时完成三件事：

1. 对物体、机器人躯干和摆动足施加 6D/3D 笛卡尔阻抗；
2. 用 Projected Inverse Dynamics Control（PIDC）解析处理浮动基座欠驱动和接触等式约束；
3. 用二次规划优化足端和手端接触 wrench，使其满足单边接触、摩擦、接触面力矩和执行器力矩限制。

最终控制结构为

\[
\text{任务参考}
\rightarrow \boldsymbol\tau_M
\rightarrow \mathbf F_c^\star
\rightarrow \boldsymbol\tau^\star,
\]

其中 \(\boldsymbol\tau_M\) 是运动子空间中的任务力矩，\(\mathbf F_c\) 是 QP 中的接触 wrench 决策变量。

---

## 2. 符号与系统边界

| 符号 | 含义 | 维度 |
|---|---|---:|
| \(\mathbf q,\dot{\mathbf q},\ddot{\mathbf q}\) | 聚合系统广义位置、速度、加速度 | \(D\) |
| \(\mathbf M(\mathbf q)\) | 广义惯性矩阵 | \(D\times D\) |
| \(\mathbf h(\mathbf q,\dot{\mathbf q})\) | 重力与科氏/离心项 | \(D\) |
| \(\mathbf S\) | 论文使用的对角驱动关节选择矩阵 | \(D\times D\) |
| \(\boldsymbol\tau\) | 扩展到广义空间的关节力矩 | \(D\) |
| \(\mathbf J_c\) | 全部足端和刚性抓取约束雅可比 | \(K\times D\) |
| \(\boldsymbol\lambda\) | 实际接触 wrench 的纵向堆叠 | \(K\) |
| \(\mathbf J_x\) | 某个运动任务的雅可比 | 通常 \(6\times D\) |
| \(\mathbf F\) | 人或环境作用的外部 6D wrench | \(6\) |
| \(L\) | 当前接触环境的足数 | - |
| \(B\) | 手与物体的接触数 | - |
| \(\mathbf F_c\) | QP 优化的足端/手端命令 wrench | \(3L+6B\) |

论文把各机器人动力学聚合后，\(\mathbf M\) 为按机器人分块的块对角矩阵，\(\mathbf h\) 为各机器人项的堆叠。物体通过刚性抓取运动学、物体任务雅可比和手端接触 wrench 进入控制；Sec. III 没有把物体自由刚体坐标显式追加为一个独立广义坐标块。

---

## 3. 受约束浮动基座动力学

### 3.1 原始动力学

论文式 (1)：

\[
\mathbf M\ddot{\mathbf q}+\mathbf h
=\mathbf S\boldsymbol\tau
+\mathbf J_c^T\boldsymbol\lambda
+\mathbf J_x^T\mathbf F.
\tag{1}
\]

刚性静止接触满足

\[
\mathbf J_c\dot{\mathbf q}=\mathbf 0,
\qquad
\mathbf J_c\ddot{\mathbf q}+\dot{\mathbf J}_c\dot{\mathbf q}=\mathbf 0.
\tag{2}
\]

这里的“静止接触”假设适合不滑动支撑足和刚性手-物体抓取。接触切换时，\(\mathbf J_c\)、维数 \(K\) 和全部投影矩阵都必须重建。

### 3.2 运动/约束子空间投影

定义正交投影

\[
\mathbf P=\mathbf I-\mathbf J_c^T(\mathbf J_c^+)^T.
\]

它投影到 \(\mathcal N(\mathbf J_c)\)，并满足

\[
\mathbf P^2=\mathbf P,\quad
\mathbf P^T=\mathbf P,\quad
\mathbf P\mathbf J_c^T=\mathbf 0,\quad
\mathbf P\dot{\mathbf q}=\dot{\mathbf q}.
\]

因此 \(\mathbf P\) 是允许运动的子空间，\(\mathbf I-\mathbf P\) 是约束力子空间。对式 (1) 分别投影得到

\[
\mathbf P(\mathbf M\ddot{\mathbf q}+\mathbf h)
=\mathbf P\mathbf S\boldsymbol\tau
+\mathbf P\mathbf J_x^T\mathbf F,
\tag{3}
\]

\[
(\mathbf I-\mathbf P)(\mathbf M\ddot{\mathbf q}+\mathbf h)
=(\mathbf I-\mathbf P)\mathbf S\boldsymbol\tau
+\mathbf J_c^T\boldsymbol\lambda
+(\mathbf I-\mathbf P)\mathbf J_x^T\mathbf F.
\tag{4}
\]

由于 \(\mathbf P\mathbf J_c^T=0\)，运动方程 (3) 不显式包含接触 wrench；约束方程 (4) 用于恢复接触 wrench。

### 3.3 广义加速度

\(\mathbf P\mathbf M\) 可能秩亏，不能直接求逆。由

\[
(\mathbf I-\mathbf P)\dot{\mathbf q}=0
\]

求导得到

\[
(\mathbf I-\mathbf P)\ddot{\mathbf q}=\dot{\mathbf P}\dot{\mathbf q}.
\]

定义

\[
\mathbf M_c=\mathbf P\mathbf M+\mathbf I-\mathbf P,
\]

则论文式 (5) 为

\[
\ddot{\mathbf q}
=\mathbf M_c^{-1}
\left(
\mathbf P\mathbf S\boldsymbol\tau
-\mathbf P\mathbf h
+\dot{\mathbf P}\dot{\mathbf q}
+\mathbf P\mathbf J_x^T\mathbf F
\right).
\tag{5}
\]

再定义约束惯性矩阵

\[
\bar{\mathbf M}=\mathbf M\mathbf M_c^{-1}.
\]

将式 (5) 代回式 (4)，得到实际接触 wrench

\[
\begin{aligned}
\boldsymbol\lambda
=&(\mathbf J_c^T)^+
\Big\{
(\mathbf I-\mathbf P)
\left[
\bar{\mathbf M}
(\mathbf P\mathbf S\boldsymbol\tau-\mathbf P\mathbf h
+\dot{\mathbf P}\dot{\mathbf q})
+\mathbf h
\right]\\
&-(\mathbf I-\mathbf P)\mathbf S\boldsymbol\tau
+(\mathbf I-\mathbf P)(\bar{\mathbf M}\mathbf P-\mathbf I)
\mathbf J_x^T\mathbf F
\Big\}.
\end{aligned}
\tag{6}
\]

式 (6) 很关键：QP 的变量 \(\mathbf F_c\) 是产生约束力的命令量，但真正进入摩擦/单边约束的量应是动力学恢复出的 \(\boldsymbol\lambda\)，两者一般不相等。

---

## 4. 笛卡尔阻抗与运动力矩

对任意任务空间 \(x\)，误差定义为

\[
\mathbf e=\mathbf x-\mathbf x_d,
\qquad
\dot{\mathbf e}=\dot{\mathbf x}-\dot{\mathbf x}_d.
\]

约束一致的操作空间惯量为

\[
\boldsymbol\Lambda_c
=\left(
\mathbf J_x\mathbf M_c^{-1}\mathbf P\mathbf J_x^T
\right)^{-1},
\]

非线性补偿项为

\[
\mathbf h_c
=\boldsymbol\Lambda_c\mathbf J_x\mathbf M_c^{-1}
(\mathbf P\mathbf h-\dot{\mathbf P}\dot{\mathbf q})
-\boldsymbol\Lambda_c\dot{\mathbf J}_x\dot{\mathbf q}.
\]

论文式 (7) 的期望任务 wrench：

\[
\mathbf F_{d,x}
=\mathbf h_c
+\boldsymbol\Lambda_c\ddot{\mathbf x}_d
-\mathbf K_d\dot{\mathbf e}
-\mathbf K_p\mathbf e.
\tag{7}
\]

映射回广义关节空间：

\[
\boldsymbol\tau_M=\mathbf J_x^T\mathbf F_{d,x}.
\tag{8}
\]

实际系统同时具有对象、机器人躯干和摆动足任务，故运动力矩按任务叠加：

\[
\boldsymbol\tau_M
=\mathbf J_o^T\mathbf F_{d,o}
+\sum_r\mathbf J_{b,r}^T\mathbf F_{d,b,r}
+\sum_{s\in\mathcal S}\mathbf J_{f,s}^T\mathbf F_{d,f,s}.
\]

论文使用直接叠加，而不是再建立严格的任务优先级投影。因此任务参考必须彼此兼容，否则叠加力矩会发生竞争。

---

## 5. 欠驱动 Projected Inverse Dynamics Control

对浮动基座系统，运动力矩与约束力矩组合为

\[
\boldsymbol\tau
=[\mathbf P\mathbf S]^+\mathbf P\boldsymbol\tau_M
+\mathbf B(\mathbf I-\mathbf P)\boldsymbol\tau_C,
\tag{9}
\]

其中

\[
\mathbf B
=\mathbf I-left[(\mathbf I-\mathbf S)(\mathbf I-\mathbf P)\right]^+.
\tag{10}
\]

解释如下：

- \([\mathbf P\mathbf S]^+\mathbf P\boldsymbol\tau_M\)：在可驱动的运动子空间实现笛卡尔任务；
- \(\mathbf B\)：补偿浮动基座欠驱动，使约束子空间命令可由实际执行器产生；
- \((\mathbf I-\mathbf P)\boldsymbol\tau_C\)：只影响约束 wrench，不改变允许运动。

后文用

\[
(\mathbf I-\mathbf P)\boldsymbol\tau_C=\mathbf J_c^T\mathbf F_c
\]

把约束力矩改写为接触 wrench 决策变量。

---

## 6. 刚性抓取、内力与对象雅可比

### 6.1 抓取矩阵

第 \(i\) 个手端接触到物体质心的部分抓取矩阵为

\[
\mathbf G_i=
\begin{bmatrix}
\mathbf R_i & \mathbf 0\\
\mathbf S(\mathbf r_i)\mathbf R_i & \mathbf R_i
\end{bmatrix},
\tag{11}
\]

其中 \(\mathbf r_i\) 是物体质心到第 \(i\) 个接触坐标系的位置向量，\(\mathbf S(\mathbf r_i)\) 是叉乘反对称矩阵。完整抓取矩阵为

\[
\mathbf G=[\mathbf G_1,\ldots,\mathbf G_B]
\in\mathbb R^{6\times 6B}.
\tag{12}
\]

若手端 wrench 堆叠为 \(\boldsymbol\lambda_{ee}\)，则物体合 wrench 为

\[
\mathbf w_o=\mathbf G\boldsymbol\lambda_{ee}.
\]

满足

\[
\mathbf G\boldsymbol\lambda_{ee}^{int}=0
\]

的分量是内力，不改变物体净运动，但决定挤压、防滑和抓取稳定性。

### 6.2 只约束相对手部运动

手端雅可比纵向堆叠为 \(\mathbf J_{ee}\in\mathbb R^{6B\times D}\)。抓取矩阵零空间投影给出刚性抓取约束雅可比

\[
\mathbf J_{c,ee}
=\left(\mathbf I-\mathbf G^T(\mathbf G^+)^T\right)\mathbf J_{ee}.
\tag{13}
\]

它只禁止各手之间破坏刚性物体几何关系的相对运动；所有手随物体一起平移/转动的绝对运动仍被允许。这就是论文中 virtual linkage 的数学实现。

足端接触雅可比为 \(\mathbf J_{cf}\in\mathbb R^{3L\times D}\)，于是总约束雅可比

\[
\mathbf J_c=
\begin{bmatrix}
\mathbf J_{cf}\\
\mathbf J_{c,ee}
\end{bmatrix}.
\tag{14}
\]

物体质心任务雅可比为

\[
\mathbf J_o=(\mathbf G^+)^T\mathbf J_{ee}.
\tag{15}
\]

论文假定手与物体质心间的相对变换恒定，因此可从手部运动推断物体位姿，而不依赖外部视觉。可用 \(\mathbf J_c\dot{\mathbf q}\neq0\) 检测滑移并更新抓取矩阵，但阈值、滤波与更新算法并未在论文中展开。

---

## 7. 接触力 QP 的目标推导

### 7.1 从关节力矩范数开始

论文用

\[
\min_{\boldsymbol\tau}\ \boldsymbol\tau^T\boldsymbol\tau
\tag{16}
\]

表示减少执行器负担。严格地说这最小化的是力矩二范数，不是机械功率 \(\boldsymbol\tau^T\dot{\mathbf q}\) 或电能。

代入式 (9)，运动项与约束项属于正交子空间，常数项和交叉项可删除，剩下

\[
\min\ \left[\mathbf B(\mathbf I-\mathbf P)\boldsymbol\tau_C\right]^T
\mathbf B(\mathbf I-\mathbf P)\boldsymbol\tau_C.
\tag{18}
\]

令

\[
(\mathbf I-\mathbf P)\boldsymbol\tau_C
=\mathbf J_c^T\mathbf F_c,
\]

则 QP 目标变为

\[
\min_{\mathbf F_c}\quad
\mathbf F_c^T
\underbrace{\mathbf J_c\mathbf B^T\mathbf B\mathbf J_c^T}_{\mathbf Q}
\mathbf F_c.
\tag{19}
\]

标准 QP 写法为

\[
\min_{\mathbf F_c}\quad
\frac12\mathbf F_c^T\mathbf H\mathbf F_c+\mathbf g^T\mathbf F_c,
\qquad
\mathbf H=2(\mathbf Q+\epsilon_Q\mathbf I),\quad \mathbf g=0.
\]

数值实现建议添加很小的 \(\epsilon_Q>0\)，在 \(\mathbf Q\) 半正定或接触冗余时保证求解器条件更好。若要平滑接触力，可再加入

\[
\alpha\|\mathbf F_c-\mathbf F_{c,k-1}\|^2,
\]

相应地更新 \(\mathbf H\) 和 \(\mathbf g\)。这属于工程增强，不是论文原目标。

---

## 8. 物理一致性不等式

### 8.1 摩擦棱锥与单边接触

对第 \(i\) 个接触，\(\mathbf n_{x,i},\mathbf n_{y,i}\) 为两个切向单位向量，\(\mathbf n_{z,i}\) 为法向，摩擦系数为 \(\mu_i\)。令实际接触力为 \(\boldsymbol\lambda_{f,i}\in\mathbb R^3\)，论文式 (20) 写成

\[
\underline{\mathbf d}
\le \mathbf C_i\boldsymbol\lambda_{f,i}
\le \bar{\mathbf d},
\tag{20}
\]

其中

\[
\mathbf C_i=
\begin{bmatrix}
(\mathbf n_{x,i}-\mu_i\mathbf n_{z,i})^T\\
(\mathbf n_{y,i}-\mu_i\mathbf n_{z,i})^T\\
(\mathbf n_{x,i}+\mu_i\mathbf n_{z,i})^T\\
(\mathbf n_{y,i}+\mu_i\mathbf n_{z,i})^T\\
\mathbf n_{z,i}^T
\end{bmatrix},
\]

\[
\underline{\mathbf d}=[-\infty,-\infty,0,0,0]^T,
\qquad
\bar{\mathbf d}=[0,0,+\infty,+\infty,+\infty]^T.
\]

前四行等价于线性化库仑摩擦锥

\[
|f_x|\le\mu f_z,\qquad |f_y|\le\mu f_z,
\]

最后一行是单边条件 \(f_z\ge0\)。

### 8.2 平面接触力矩限制

对手掌/平面足的 6D wrench

\[
\boldsymbol\lambda_i=[f_x,f_y,f_z,m_x,m_y,m_z]^T,
\]

论文式 (21) 使用

\[
\mathbf 0\le\mathbf C_{m,i}\boldsymbol\lambda_i,
\tag{21}
\]

\[
\mathbf C_{m,i}=
\begin{bmatrix}
0&0&\xi&0&0&-1\\
0&0&\xi&0&0& 1\\
0&0&\delta_x&-1&0&0\\
0&0&\delta_x& 1&0&0\\
0&0&\delta_y&0&-1&0\\
0&0&\delta_y&0& 1&0
\end{bmatrix}.
\]

它等价于

\[
|m_z|\le\xi f_z,qquad
|m_x|\le\delta_x f_z,qquad
|m_y|\le\delta_y f_z.
\]

\(\xi\) 是扭转摩擦系数，\(\delta_x,\delta_y\) 是矩形接触面中心到边缘的距离。

### 8.3 执行器力矩限制

用非方阵 \(\mathbf S_{ns}\in\mathbb R^{D\times n_a}\) 取出真实驱动关节，论文式 (22) 为

\[
\boldsymbol\tau_{min}
\le\mathbf S_{ns}^T\boldsymbol\tau
\le\boldsymbol\tau_{max}.
\tag{22}
\]

代入最终力矩表达式后，得到关于 \(\mathbf F_c\) 的线性约束

\[
\underline{\boldsymbol\tau}
\le\boldsymbol\Xi\mathbf F_c
\le\bar{\boldsymbol\tau},
\tag{23}
\]

其中

\[
\boldsymbol\Xi=\mathbf S_{ns}^T\mathbf B\mathbf J_c^T,
\]

\[
\underline{\boldsymbol\tau}
=\boldsymbol\tau_{min}
-\mathbf S_{ns}^T[\mathbf P\mathbf S]^+\mathbf P\boldsymbol\tau_M,
\]

\[
\bar{\boldsymbol\tau}
=\boldsymbol\tau_{max}
-\mathbf S_{ns}^T[\mathbf P\mathbf S]^+\mathbf P\boldsymbol\tau_M.
\]

---

## 9. 从 QP 变量恢复实际接触 wrench

QP 优化的是 \(\mathbf F_c\)，但摩擦和接触面约束必须施加到实际 \(\boldsymbol\lambda\)。将

\[
\boldsymbol\tau
=[\mathbf P\mathbf S]^+\mathbf P\boldsymbol\tau_M
+\mathbf B\mathbf J_c^T\mathbf F_c
\]

代入式 (6)，论文推导出仿射关系

\[
\boldsymbol\lambda=\boldsymbol\rho\mathbf F_c+\boldsymbol\eta,
\tag{26}
\]

其中

\[
\boldsymbol\rho
=(\mathbf J_c^T)^+
(\mathbf I-\mathbf P)
(\bar{\mathbf M}\mathbf P-\mathbf I)
\mathbf S\mathbf B\mathbf J_c^T,
\]

\[
\boldsymbol\eta
=(\mathbf J_c^T)^+
(\mathbf I-\mathbf P)
\left[
\boldsymbol\varepsilon\boldsymbol\tau_M
+\bar{\mathbf M}(-\mathbf P\mathbf h+\dot{\mathbf P}\dot{\mathbf q})
+\mathbf h
+(\bar{\mathbf M}\mathbf P-\mathbf I)\mathbf J_x^T\mathbf F
\right],
\]

\[
\boldsymbol\varepsilon
=(\bar{\mathbf M}\mathbf P-\mathbf I)
\mathbf S[\mathbf P\mathbf S]^+\mathbf P.
\]

因此第 \(i\) 个接触的实际 wrench 可用选择矩阵 \(\mathbf E_i\) 写成

\[
\boldsymbol\lambda_i
=\mathbf E_i(\boldsymbol\rho\mathbf F_c+\boldsymbol\eta).
\]

把它代入式 (20)、(21)，所有约束都成为 \(\mathbf F_c\) 的线性不等式。例如摩擦约束：

\[
\underline{\mathbf d}
-\mathbf C_i\mathbf E_{f,i}\boldsymbol\eta
\le
\mathbf C_i\mathbf E_{f,i}\boldsymbol\rho\mathbf F_c
\le
\bar{\mathbf d}
-\mathbf C_i\mathbf E_{f,i}\boldsymbol\eta.
\]

平面接触力矩约束：

\[
-\mathbf C_{m,i}\mathbf E_i\boldsymbol\eta
\le
\mathbf C_{m,i}\mathbf E_i\boldsymbol\rho\mathbf F_c.
\]

---

## 10. 最终标准 QP

把所有接触约束和力矩约束纵向堆叠，得到

\[
\begin{aligned}
\min_{\mathbf F_c}\quad &
\frac12\mathbf F_c^T\mathbf H\mathbf F_c+\mathbf g^T\mathbf F_c\\
\text{s.t.}\quad &
\mathbf l\le\mathbf A\mathbf F_c\le\mathbf u,
\end{aligned}
\]

其中

\[
\mathbf H=2(\mathbf J_c\mathbf B^T\mathbf B\mathbf J_c^T+\epsilon_Q\mathbf I),
\qquad \mathbf g=0,
\]

\[
\mathbf A=
\begin{bmatrix}
\mathbf A_{friction}\\
\mathbf A_{moment}\\
\boldsymbol\Xi
\end{bmatrix}.
\]

求得 \(\mathbf F_c^\star\) 后：

\[
\boldsymbol\tau^\star
=[\mathbf P\mathbf S]^+\mathbf P\boldsymbol\tau_M
+\mathbf B\mathbf J_c^T\mathbf F_c^\star,
\]

\[
\boldsymbol\lambda^\star
=\boldsymbol\rho\mathbf F_c^\star+\boldsymbol\eta.
\]

执行前必须再次检查 \(\boldsymbol\tau^\star\)、\(\boldsymbol\lambda^\star\)、QP 残差和有限性。若 QP 不可行，工程实现应有分级回退：加入 slack 软化运动任务/接触目标、保持上一拍可行解、降低对象加速度命令，最后进入安全停机；论文没有给出完整回退策略。

---

## 11. 每个控制周期的计算过程

```text
输入：q, dq, 接触集合, 对象/躯干/摆动足参考, 外部 wrench 估计

1. 更新每台机器人的 M(q)、h(q,dq) 和各任务 Jacobian/Jdot*dq
2. 根据支撑足和刚性手-物体约束组装 Jcf、Jee、G
3. 计算 Jc,ee = (I - G^T(G^+)^T) Jee
4. 组装 Jc = [Jcf; Jc,ee]，计算 P、Pdot*dq、Mc、Mbar、B
5. 从刚性抓取关系推断对象位姿/速度，计算 Jo
6. 分别计算对象、各躯干、各摆动足的阻抗 wrench Fd,x
7. 累加 τM = Σ Jx^T Fd,x
8. 构造 QP Hessian H
9. 由式 (26) 计算 rho、eta，将实际接触约束改写为 Fc 的线性约束
10. 加入驱动关节力矩上下界
11. warm-start 求解 Fc*
12. 合成 τ* = [PS]^+PτM + BJc^T Fc*
13. 计算 λ* = rho Fc* + eta，检查摩擦裕度、力矩裕度和残差
14. 向各机器人下发本机对应的关节力矩块
```

数值实现注意：

- 不要每次显式形成普通矩阵逆；对 \(\mathbf M_c\)、\(\boldsymbol\Lambda_c\) 使用线性求解或分解。
- Moore-Penrose 逆要使用一致的截断阈值，监控秩变化和条件数。
- \(\dot{\mathbf P}\dot{\mathbf q}\)、\(\dot{\mathbf J}\dot{\mathbf q}\) 优先由动力学库解析获得；有限差分必须滤波。
- 接触切换会造成 \(\mathbf P\) 跳变，应采用相位过渡、wrench ramp 和 QP warm start。
- 所有 wrench 必须记录 frame、reference point、作用方向和单位。

---

## 12. 仿真与实机验证内容

论文仿真使用两台 ANYmal C + 6DoF Kinova Gen3，共同搬运边长 0.6 m、质量 1 kg 的箱体，在 RaiSim 中测试：

1. 两台躯干与物体同时进行位置/姿态轨迹；
2. 一台机器人静态步态前后移动并进行 \(-30^\circ\) yaw 转向；
3. 与直接多机器人 QP（MQP）比较。

实机使用两台各有 12DoF 腿和 6DoF Sirius-Arm 的自研四足机器人，搬运 1 kg 箱体，测试物体 X-Z 圆周运动、Roll/Pitch/Yaw 姿态运动以及对象/躯干外部扰动恢复。

论文报告的局限包括：高负载下机械臂同步带压迫轴承和传动结构，使摩擦增大；固定 PID 参数导致速度不一致和跟踪延迟。论文还假设对象形状、期望接触点、表面法向和全系统相容的预规划轨迹已知，没有解决完整协同运动规划问题。

---

## 13. 对 TRON2 + 松灵机械臂的适配判断

### 13.1 可以直接复用

- 抓取矩阵 \(\mathbf G\)、内力零空间和 \(\mathbf J_{c,ee}\)；
- 对象/躯干笛卡尔阻抗结构；
- 接触 wrench QP 与关节力矩限制；
- 实际 wrench \(\boldsymbol\lambda=\boldsymbol\rho\mathbf F_c+\boldsymbol\eta\) 后再检查摩擦约束的思想；
- 多机器人动力学按机器人分块组装。

### 13.2 不能直接照搬

1. **轮足非完整约束**：论文足端采用静止点接触 \(\mathbf J_{cf}\dot{\mathbf q}=0\)。TRON2 wheelfoot 在滚动方向允许速度，只约束法向与侧向无滑，因此需要构造非完整滚动约束；只有 solefoot/驻车阶段能直接使用静止接触模型。
2. **低层接口**：论文要求高带宽关节力矩控制。若 TRON2 官方策略输出关节位置或速度，不能直接发送 \(\boldsymbol\tau^\star\)，必须确认电机接口并决定“学习 locomotion + 模型控制机械臂”还是完整 WBC。
3. **手端接触模型**：松灵夹爪抓箱子是否能传递完整 6D wrench 取决于夹持几何和摩擦。若接触更接近点/线接触，应降低 \(\mathbf G_i\) 和 \(\mathbf J_{c,ee}\) 的约束维度，不能强行使用平面 6D 接触。
4. **未知物体参数**：论文已知物体形状、接触点和法向。本项目的质量/质心在线辨识必须先向 \(\mathbf G\)、\(\mathbf J_o\) 和对象参考动力学提供一致参数，并在不可观阶段冻结。
5. **分布式部署**：论文公式按聚合系统集中求解。双 TRON2 若没有低延迟共享状态，需要中央 WBC 或分布式近似；PAINT intent estimator 不能替代动力学状态同步。

### 13.3 推荐实施顺序

1. 固定底座双松灵机械臂 + 刚性箱体，验证式 (11)--(15) 和内力分解；
2. 固定 TRON2 站立支撑集合，验证对象阻抗与手/足接触 QP；
3. solefoot 准静态换步，验证接触切换；
4. wheelfoot 加入滚动非完整约束；
5. 接入对象质量/质心估计、无力传感器 wrench 估计和内部力调节；
6. 最后与预训练 locomotion policy 组合，比较集中 WBC、分层控制和残差策略。

---

## 14. 与本项目现有公式模块的关系

| 本文模块 | MHCT 现有模块 | 关系 |
|---|---|---|
| \(\mathbf G\)、\(\ker\mathbf G\) | `internal_force_analyzer.py`、Virtual Linkage 笔记 | 可复用并扩展到约束雅可比 |
| 对象阻抗 | `model_derivation_notes.md` 的 IMP/DIMP | 本文补充约束一致惯量与 PIDC 映射 |
| 接触 wrench QP | 当前载荷分配/内部力目标 | 新增摩擦、接触面力矩、欠驱动和执行器约束 |
| 外部 wrench \(\mathbf F\) | GMO、PAINT intent estimator | 必须统一 frame/reference/sign 后才能进入式 (1)、(6)、(26) |
| 对象质心 \(\mathbf r_i\) | payload mass/COM estimator | 在线估计值将改变抓取矩阵和对象雅可比 |
| CHIP 柔顺 | 对象/末端参考生成 | 可生成柔顺目标，但不能替代接触可行性 QP |

这篇论文最值得本项目吸收的不是单个阻抗公式，而是完整闭环：**相容运动任务产生 \(\boldsymbol\tau_M\)，欠驱动投影保证动力学一致，接触 QP 保证物理可行，再由实际接触 wrench 反算检查约束。**
