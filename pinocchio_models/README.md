# Pinocchio model workspace

This directory contains the minimum model assets for validating coupled dynamics
of two TRON1 robots with Airbot arms and one carried payload.

## Contents

```text
pinocchio_models/
├── model_manifest.yaml
├── payload/
│   └── payload_box.urdf
└── robot_description/
    └── pointfoot/
        ├── WF_TRON1A/
        │   ├── meshes/
        │   └── urdf/robot_with_arm.urdf
        └── SF_TRON1A/
            ├── meshes/
            └── urdf/robot_with_arm.urdf
```

The two robot instances reuse one robot model. They should be assigned different
floating-base poses in the Pinocchio or simulation code; duplicate URDF files
are not required.

## Model selection

- Use `WF_TRON1A` for the wheel-foot configuration.
- Use `SF_TRON1A` for the sole-foot configuration.
- Do not combine the wheel-foot and sole-foot models in one dynamics experiment.

The `robot_with_arm.urdf` files contain the TRON1 base, the fixed Airbot arm
mount, and arm joints `J1` through `J6`. The base-only `robot.urdf` files are
intentionally not copied here.

## Airbot gripper URDF staging

The downloaded file is staged at:

```text
airbot/urdf/airbot_play_with_gripper.urdf
```

The original downloaded filename was `airbot_play_v2_1_with_gripper.urdf`.
The copy is kept under the SDK-compatible name so later code can use the same
path convention as the real deployment package.

The current file contains `base_link` and `link1` through `link6`, with six
actuated arm joints. It does not contain independent gripper links or gripper
joints. The gripper can still appear in the visual model as fixed geometry
inside the terminal `link6` mesh; no independent gripper degree of freedom is
then available in Pinocchio. Its mesh references still point to the source
computer path `/home/george/Downloads/airbot_play_v6_1/meshes`. Therefore it is
archived as a separate raw asset for now; do not use it for geometry
visualization until the matching mesh files are obtained and the paths are
resolved. The staged file itself has not been modified.

## Pinocchio loading

For a floating-base robot, use a free-flyer root joint:

```python
from pathlib import Path
import pinocchio as pin

ROOT = Path(__file__).resolve().parent
urdf = ROOT / "robot_description/pointfoot/WF_TRON1A/urdf/robot_with_arm.urdf"
model = pin.buildModelFromUrdf(str(urdf), pin.JointModelFreeFlyer())
data = model.createData()
```

The URDF is sufficient for dynamics. The copied STL meshes are only needed for
visualization or geometry loading. For `package://robot_description/...` mesh
paths, use this directory as a package search root.

On Windows, some urdfdom builds cannot open paths containing non-ASCII
characters. `scripts/check_models.py` automatically converts the workspace path
to an ASCII 8.3 path before calling Pinocchio.

## MeshCat visualization

Install the Python MeshCat server package in the active environment:

```bash
pip install meshcat
```

Then run:

```bash
python pinocchio_models/scripts/visualize_meshcat.py --robot-type WF_TRON1A
```

The script opens a browser automatically and also prints the MeshCat URL. Keep
the terminal process alive while inspecting the model. Use `SF_TRON1A` for the
sole-foot model or add `--show-collision` to display collision geometry.

## Two-robot scene

The first two-robot scene is a visualization-only check:

```bash
python pinocchio_models/scripts/visualize_two_robot_payload.py --robot-type WF_TRON1A
```

It loads two instances of the same robot model and one payload model into one
MeshCat viewer. The default layout places the robots on opposite sides of the
payload and rotates the second robot by 180 degrees. This does not enforce a
grasp, close a gripper, or solve contact forces.

## Parameters to replace before hardware conclusions

The current software checks use placeholder geometry and should be calibrated
before drawing hardware conclusions:

1. The actual robot type: `WF_TRON1A` or `SF_TRON1A`.
2. Payload mass, dimensions, center of mass, and inertia in SI units.
3. The two grasp/contact frames and their poses relative to the payload frame.
4. Whether each grasp is rigid, point contact, or compliant contact.
5. The initial poses of both robot bases and the payload.
6. The intended constraints: position only, full pose, or pose plus contact
   wrench limits and friction cones.

These parameters determine the contact assumptions and the numerical values of
the equations below. The current scripts already provide the first analytical
model; the remaining work is parameter calibration and contact-model refinement.

Expected dimensions for one TRON1 plus arm are:

```text
nq = 21
nv = 20
```

This includes a 7-dimensional floating-base configuration, a 6-dimensional
floating-base velocity, and 14 one-dimensional actuated joints.

## Coupled-system setup

The intended first model is:

```text
robot_1: WF_TRON1A or SF_TRON1A, free flyer
robot_2: same model, free flyer
payload: payload/payload_box.urdf, free flyer
```

The payload parameters are placeholders for software validation. Replace its
mass, dimensions, center of mass, and inertia with measured values before using
results for hardware decisions.

The first graspable payload is `payload/payload_with_handles.urdf`. It uses a
0.60 x 0.24 x 0.24 m central body and two fixed rectangular handles of size
0.32 x 0.05 x 0.05 m at x = plus/minus 0.354 m. The handle links are named
`grasp_left` and `grasp_right`
and are the nominal contact frames for the two robots. The total placeholder
mass is 10 kg. These dimensions are selected to match the current neutral
`link6` positions and must be replaced after the real object is measured.

Check the nominal closed-chain geometry with:

```bash
python pinocchio_models/scripts/check_grasp_kinematics.py --robot-type WF_TRON1A
```

Run the first static balance check with:

```bash
python pinocchio_models/scripts/check_static_balance.py --robot-type WF_TRON1A
```

This assigns the payload gravity wrench between the two grasp frames and
reports the resulting arm torque and fixed-base reaction. It is a quasi-static
test and does not include the leg support constraints.

The two-robot MeshCat scene now uses the handle payload by default. Use
`--payload-model box` to return to the original box-only visualization.

Run the first cooperative carry motion demo with:

```bash
python pinocchio_models/scripts/animate_cooperative_carry.py --robot-type WF_TRON1A
```

This demo keeps both bases fixed and uses arm-only damped least-squares IK to
track the two payload grasp frames. It validates synchronized kinematic motion
before adding support contact wrenches and friction. The default motion
is intentionally small; increase `--amplitude-y`, `--amplitude-z`, or
`--amplitude-yaw` only after checking reachability and joint limits.

Run the supported cooperative carry demo with:

```bash
python pinocchio_models/scripts/animate_supported_cooperative_carry.py --robot-type WF_TRON1A
```

This version automatically selects `wheel_L_Link/wheel_R_Link` for WF or
`ankle_L_Link/ankle_R_Link` for SF. Each leg endpoint contributes a fixed
three-dimensional position constraint, so its orientation is left free. This
is a first point-support model: it is useful for validating support reactions,
rigid grasp forces, and synchronized carrying, but it is not yet a rolling
wheel model or a flat-foot friction model. The displayed trajectory uses arm
IK; the final pose also runs the constrained KKT dynamics diagnostic and
prints four support forces plus the two six-dimensional grasp wrenches.

## Validation stages and formulas

The scripts below are ordered from model loading to coupled constrained
dynamics. Each stage has a corresponding equation so that the test result can
be reused directly in a report or presentation.

### 1. Model loading and dimensions

Test command:

```bash
python pinocchio_models/scripts/check_models.py --robot-type WF_TRON1A
```

For each floating-base robot and payload, Pinocchio uses:

$$
q = [q_{base}, q_{joint}], \qquad
v = [v_{base}, v_{joint}], \qquad
\dot q = \mathrm{integrate}(q, v).
$$

The current dimensions are:

```text
one robot: nq = 21, nv = 20
payload:   nq = 7,  nv = 6
```

The single-tree rigid-body dynamics are represented as:

$$
M(q)\dot v + h(q,v) = B\tau + J(q)^T\lambda.
$$

### 2. Nominal grasp geometry

Test command:

```bash
python pinocchio_models/scripts/check_grasp_kinematics.py --robot-type WF_TRON1A
```

For robot $i$ and payload grasp frame $i$, the rigid grasp condition is:

$$
T_{r_i}(q_{r_i}) = T_{p_i}(q_p),
$$

or, using a six-dimensional Lie-algebra error:

$$
e_{g_i} = \log_6\left(T_{r_i}^{-1}T_{p_i}\right) = 0.
$$

The velocity and acceleration-level grasp constraints are:

$$
J_{g_i}(q)v = 0,
$$

$$
J_{g_i}(q)\dot v + \dot J_{g_i}(q,v)v = 0.
$$

The two full-pose grasp constraints contribute 12 rows.

### 3. Quasi-static payload balance

Test command:

```bash
python pinocchio_models/scripts/check_static_balance.py --robot-type WF_TRON1A
```

With payload gravity $g_p$ distributed over the two grasp frames, the
quasi-static payload balance is:

$$
g_p - J_{p,L}^T\lambda_L - J_{p,R}^T\lambda_R = 0.
$$

The robot-side generalized balance checks are:

$$
g_{r_i} - J_{r_i}^T\lambda_{r_i} = 0,
\qquad \lambda_{r_i} = -\lambda_i.
$$

This test treats the robot bases as externally supported. It checks force
distribution and arm torque demand, but is not yet a whole-body contact solve.

### 4. Fixed three-dimensional leg support

Test command:

```bash
python pinocchio_models/scripts/check_coupled_dynamics.py \
  --robot-type WF_TRON1A --support-mode fixed_3d_position
```

The WF support frames are `wheel_L_Link/wheel_R_Link`; the SF support frames
are `ankle_L_Link/ankle_R_Link`. For the first support model, the endpoint
position is fixed while its orientation remains free:

$$
J_{s_i}(q)v_i = 0,
\qquad J_{s_i}(q) \in \mathbb{R}^{3\times nv_i}.
$$

At acceleration level:

$$
J_{s_i}(q)\dot v_i + \dot J_{s_i}(q,v_i)v_i = 0.
$$

Stacking four leg endpoints and two grasps gives:

$$
J_c =
\begin{bmatrix}
J_s \\
J_g
\end{bmatrix},
\qquad
\lambda =
\begin{bmatrix}
\lambda_s \\
\lambda_g
\end{bmatrix},
\qquad
\dim(J_c)=24.
$$

### 5. WF wheel rolling and no-slip support

Test command:

```bash
python pinocchio_models/scripts/check_rolling_constraints.py
```

The URDF wheel collision geometry gives radius $R=0.127$ m. With ground
normal $n=[0,0,1]^T$, the wheel is modeled by three instantaneous constraints:
no penetration, no lateral slip, and no longitudinal rolling slip. The wheel
contact point is not fixed in world coordinates, so the wheel can translate
along its rolling direction.

Let $a_i$ be the wheel axle direction in world coordinates and let $d_i$ be
the rolling direction:

$$
a_i=R_{w_i}e_y,
\qquad
d_i=\frac{a_i\times n}{\|a_i\times n\|}.
$$

Let $J_{v_i}$ and $J_{\omega_i}$ be the linear and angular Jacobians of the
wheel frame. The three rolling rows implemented in
`coupled_dynamics.py` are:

$$
J_{roll,i}=
\begin{bmatrix}
n^T J_{v_i}\\
a_i^T J_{v_i}\\
d_i^T J_{v_i}-R a_i^T J_{\omega_i}
\end{bmatrix}.
$$

The velocity-level nonholonomic constraint is:

$$
J_{roll,i}(q)v_i=0.
$$

For the acceleration-level constraint, the wheel axis and rolling direction
also vary with angular velocity. With $\dot a_i=\omega_i\times a_i$ and
$\dot d_i=\dot a_i\times n$ (the ground normal is fixed), the implemented
constraint is:

$$
J_{roll,i}(q)\dot v_i+\dot J_{roll,i}(q,v_i)v_i=0.
$$

The rolling multiplier coordinates are ordered as
$(\lambda_n,\lambda_a,\lambda_d)$. The physical world-frame support force is:

$$
f_i=\begin{bmatrix}n&a_i&d_i\end{bmatrix}
\begin{bmatrix}\lambda_{n,i}\\\lambda_{a,i}\\\lambda_{d,i}\end{bmatrix}.
$$

This distinction is needed before checking friction cones because the rolling
multiplier rows are not simply world $x/y/z$ force components.

### 6. Coupled constrained dynamics and KKT solve

The complete model is implemented in:

```bash
python pinocchio_models/scripts/check_coupled_dynamics.py \
  --robot-type WF_TRON1A --support-mode fixed_3d_position
```

The stacked state and actuation dimensions are:

```text
q = [q_robot_1, q_robot_2, q_payload], nq = 49
v = [v_robot_1, v_robot_2, v_payload], nv = 46
tau_act = [tau_robot_1, tau_robot_2], dimension = 28
lambda_support = 12, lambda_grasp = 12
```

The block-diagonal unconstrained dynamics are:

$$
M = \mathrm{blockdiag}(M_1,M_2,M_p),
\qquad
h = [h_1^T,h_2^T,h_p^T]^T.
$$

The coupled equations are:

$$
M\dot v + h = B\tau_{act} + J_c^T\lambda,
$$

$$
J_c\dot v + \dot J_c v = 0.
$$

Pinocchio validation solves the KKT system:

$$
\begin{bmatrix}
M & -J_c^T \\
J_c & 0
\end{bmatrix}
\begin{bmatrix}
\dot v \\
\lambda
\end{bmatrix}
=
\begin{bmatrix}
B\tau_{act}-h \\
-\dot J_c v
\end{bmatrix}.
$$

The current fixed-support and rolling-support tests both give a full-rank
`24 x 46` constraint Jacobian and a `70 x 70` KKT system.

### 7. Supported cooperative-carry motion

Test command:

```bash
python pinocchio_models/scripts/animate_supported_cooperative_carry.py \
  --robot-type WF_TRON1A
```

The payload reference follows a small synchronized trajectory:

$$
y_p(t)=A_y\sin(\omega t),
\qquad
z_p(t)=z_0+A_z\sin(\omega t),
\qquad
\psi_p(t)=A_\psi\sin(\omega t).
$$

At each frame, each arm solves a damped least-squares pose tracking problem:

$$
\min_{\Delta q_i}
\left\|e_{g_i}(q_i,q_p)\right\|^2
 + \mu\left\|\Delta q_i\right\|^2.
$$

The displayed motion is kinematic IK. The final configuration is then passed
through the same KKT equations to report support forces and grasp wrenches.

### 8. Base and leg coordinated motion

Test command:

```bash
python pinocchio_models/scripts/animate_leg_supported_cooperative_carry.py \
  --robot-type WF_TRON1A
```

This test prescribes a small floating-base motion while the leg joints
compensate to keep both support endpoint positions fixed. The base reference
is a bounded sinusoidal trajectory:

The default visible amplitudes are `0.02 m` in x, `0.015 m` in y/z, and
`0.05 rad` in yaw. Green markers above the feet indicate the fixed support
targets, so the relative motion of the base/legs can be distinguished from the
arm motion.

$$
x_b(t)=x_0+A_x\sin(\omega t),
\qquad
y_b(t)=y_0+A_y\sin(\omega t),
$$

$$
z_b(t)=z_0+A_z\sin(\omega t),
\qquad
\psi_b(t)=\psi_0+A_\psi\sin(\omega t).
$$

For the eight leg joints, the support target is the initial endpoint position
$p_s^*$:

$$
e_s(q_b,q_l)=p_s^*-p_s(q_b,q_l)=0.
$$

The damped least-squares leg update is:

$$
\Delta q_l = J_l^T
\left(J_lJ_l^T+\mu^2I\right)^{-1}e_s,
\qquad
J_l=\frac{\partial p_s}{\partial q_l}.
$$

The same frame then solves the arm grasp tracking problem and evaluates the
coupled KKT equations. The current WF and SF tests keep support position error
below `1e-7 m`, show nonzero intermediate leg joint motion, and use a default
six-dimensional grasp-error tolerance of `1e-3`. The current default trajectory
reaches approximately `7.4e-4`. A stricter check can be requested with
`--grasp-tolerance 5e-4`, but the current arm IK may reject that threshold at
some intermediate base poses. This remains a kinematic whole-body
coordination test; the dynamic gait and contact switching checks are recorded
below.

### 9. Constrained inverse dynamics and actuator effort check

Test commands:

```bash
python pinocchio_models/scripts/check_inverse_dynamics.py \
  --robot-type WF_TRON1A --support-mode fixed_3d_position

python pinocchio_models/scripts/check_inverse_dynamics.py \
  --robot-type WF_TRON1A --support-mode rolling_no_slip
```

For a prescribed generalized acceleration $a$, the inverse dynamics equation
is:

$$
B\tau + J_c^T\lambda = M(q)a+h(q,v).
$$

The current validation uses the minimum-norm solution. Define:

$$
A = [B\ \ J_c^T],
\qquad
x = [\tau^T\ \lambda^T]^T,
\qquad
b=M(q)a+h(q,v).
$$

Then:

$$
x=A^T(AA^T)^{-1}b.
$$

The actuator effort ratio is checked using the effort limits stored in the
URDF:

$$
\rho_i=\frac{|\tau_i|}{\tau_{i,\max}}.
$$

The current zero-acceleration cooperative-carry check gives an actuator/
contact map rank of `46`, a maximum effort ratio of approximately `0.73`, and
an inverse-dynamics residual below `1e-12`. This is a feasibility and effort
screening test, not yet a force optimizer: friction cones, unilateral contact,
internal-force objectives, torque-rate limits, and dynamic trajectory
acceleration are not imposed by this static check.

### 10. Friction cone, unilateral contact, and contact switching

Test command:

```bash
python pinocchio_models/scripts/check_contact_modes.py
```

For each active rolling contact, the physical support force is reconstructed
from the rolling multiplier basis. The diagnostic inequalities are:

$$
f_{n,i}=n^Tf_i\ge 0,
\qquad
\|f_{t,i}\|\le \mu f_{n,i},
\qquad
f_{t,i}=f_i-f_{n,i}n.
$$

The contact classifier reports `stick`, `slip`, `lift_off`, and `inactive`.
The active support mask changes the stacked contact Jacobian as:

$$
J_c=\begin{bmatrix}J_{s,\mathcal A}\\J_g\end{bmatrix},
\qquad
\mathcal A\subseteq\{1,2,3,4\}.
$$

The script exercises nominal friction-cone satisfaction, a low-$\mu$ slip
case, an upward external wrench that produces negative normal force on robot
1, and support-mask restoration. A one-pass active-set update removes
contacts with $f_n<0$ and resolves the KKT system. This is an active-set
validation and diagnostic, not yet a full friction-constrained QP or
complementarity solver.

### 11. Rolling torque-control integration

Test command:

```bash
python pinocchio_models/scripts/simulate_wheel_torque_control.py \
  --duration 1.0 --time-step 0.005 --wheel-speed 0.2 --kp 20 --kd 4
```

The closed-loop reference uses joint position and velocity errors:

$$
a_d=K_p(q_d-q)+K_d(v_d-v).
$$

Because the floating-base cooperative system is constrained, the desired
acceleration is projected into the acceleration-compatible subspace:

$$
a_c=J_c^T(J_cJ_c^T)^{-1}(-\dot J_cv),
$$

$$
N_c=I-J_c^T(J_cJ_c^T)^{-1}J_c,
\qquad
a=a_c+N_ca_d.
$$

The projected acceleration is passed to the constrained inverse-dynamics
map $B\tau+J_c^T\lambda=Ma+h$. URDF torque limits are applied before
integration. After each Euler step, velocity is projected back onto
$J_cv=0$; a Baumgarte term stabilizes wheel ground height.

The default WF run reaches approximately `0.025 rad` maximum joint tracking
error at `0.2 rad/s`, ground height error `1.3e-6 m`, constraint velocity error
below `3e-17`, minimum normal force about `153 N`, and maximum friction ratio
about `0.02`. These numbers validate the Pinocchio-level closed-loop
integration, not yet motor current dynamics or simulator contact impulses.

### 12. Leg stance/swing gait and hybrid contact modes

Test commands:

```bash
python pinocchio_models/scripts/check_leg_stance_swing.py \
  --robot-type WF_TRON1A

python pinocchio_models/scripts/check_leg_stance_swing.py \
  --robot-type SF_TRON1A
```

在 Windows PowerShell 中不要使用 Bash 的 `\` 换行符；直接写成一行，或
使用 PowerShell 反引号 `` ` `` 换行。

During a swing phase, the corresponding support contact is removed from
$\mathcal A$ and the leg endpoint follows a lifted target trajectory:

$$
p_s^d(\alpha)=p_s^0+
\begin{bmatrix}L\alpha\\0\\H\sin(\pi\alpha)\end{bmatrix},
\qquad 0\le\alpha\le1.
$$

During stance, the endpoint target remains fixed and its three position rows
are active in the KKT system. The test alternates:

```text
both stance -> robot 1 left swing -> touchdown ->
robot 2 right swing -> touchdown -> both stance
```

The WF and SF tests keep stance and swing IK errors below `1e-7 m`. Swing
phases use `21` contact rows (`3` active supports plus `12` grasp rows), while
full stance uses `24` rows. The minimum contact rank is `21` and the KKT
residual remains below `2.1e-12`.

### Current scope

The Pinocchio-only stage now covers model loading, nominal grasp geometry,
quasi-static balance, fixed support, nonholonomic wheel rolling, rigid coupled
dynamics, supported cooperative carrying, base/leg coordination, constrained
inverse dynamics, friction/unilateral diagnostics, hybrid contact masks,
stance/swing leg IK, and projected whole-body torque-control integration.
It does not yet model motor current loops, compliant contact, impact impulses,
torque-rate limits, or a full friction-constrained MPC/QP.

The original SDK, ONNX policies, ROS messages, and Gazebo plugins are not needed
to build the Pinocchio rigid-body models.
