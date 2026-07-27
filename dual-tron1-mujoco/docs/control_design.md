# Dual-arm static carry control design

## Scope

The first control milestone holds a shared payload with both robot bases fixed.
The wheel-leg joints retain their initial-pose PD controller. The twelve AIRBOT
arm actuators are overwritten by an object-level impedance controller at every
MuJoCo step. No locomotion, active internal-force command or force-sensor
feedback is included in this milestone.

## Frames and grasp geometry

All controller vectors are expressed in the MuJoCo world frame. The payload
frame origin is its center of mass. In the parallel formation the nominal
grasp offsets are approximately

```text
r1 = [0, -0.354, 0] m
r2 = [0,  0.354, 0] m
```

The offsets used at runtime come from the simulated body positions rather than
from these nominal constants.

## Object impedance

The desired payload force and moment are

```text
F* = Kp (p_ref - p) - Dp v - m g
M* = KR eR - DR omega
```

where `-m g` is the upward gravity compensation because MuJoCo gravity is
`[0, 0, -9.81]`. The orientation error is computed from the current and desired
rotation-matrix basis vectors. Each force and moment component is limited before
load allocation.

## Grasp matrix and load allocation

For end-effector wrench `fi = [Fi, Mi]`, the payload wrench is

```text
wo = G [f1, f2]

G = [ I       0   I       0 ]
    [ [r1]x   I   [r2]x   I ]
```

The first implementation uses the minimum-norm solution

```text
f* = pinv(G) wo*
```

and therefore commands no deliberate null-space/internal wrench. With a
centered static payload this produces equal vertical load sharing.

## Joint torque command

Each arm command is

```text
tau = Jp.T F* + Jr.T M* + qfrc_bias
      + Kq (q_ref - q) - Dq dq
```

`qfrc_bias` supplies MuJoCo gravity/Coriolis compensation. The posture term is
kept weak so that it does not dominate the object task. Commands are limited by
the configured AIRBOT joint torque limits and by a torque slew-rate limit.

## Current limitations

- The soft weld is a six-DOF numerical grasp, not finger contact.
- MuJoCo equality-row force is a solver diagnostic, not a calibrated load-cell
  measurement.
- The controller assumes torque-controlled arm actuators. Position-only real
  hardware will require an admittance outer loop.
- The pseudoinverse does not account for asymmetric torque margins or optimize
  internal wrench.
- The robot bases are fixed; coupling with the wheel-leg policy is the next
  milestone after static load acceptance.

## Acceptance sequence

Run fixed-base tests at 0.5, 1 and 2 kg. For each test check payload height
loss, tilt, grasp gaps, equality force, arm torque peaks and saturation time.
Only after these pass should the base welds be released at zero commanded
velocity.

## Free-base balance milestone

The free-base mode retains all fourteen ONNX policy outputs so that the policy
sees and actuates the full-body distribution on which it was trained. The arm
controller is changed from absolute torque replacement to an additive load
compensation:

```text
tau_arm = tau_policy + J.T f_payload
```

The additional wrench is ramped in over one second. Its pose reference follows
the centroid translation and common yaw of the two robot bases, so the payload
is held relative to the formation instead of being anchored to the world.
Its damping reference follows the average base linear velocity and common yaw
rate as well; mixing a moving position reference with zero world-velocity
damping creates an artificial drag force and is not a valid relative
impedance.

For the first free-base acceptance test, additive impedance is intentionally
limited to the vertical axis. Horizontal station keeping belongs to the mobile
base formation loop, while payload orientation remains with the learned arm
policy and compliant grasp. Adding horizontal and rotational task wrench on
top of the policy caused frequent 3 Nm wrist saturation.

The initial 0.5 kg run remained upright for 10 seconds but failed zero-command
station keeping: base drift reached about 0.91 m and the maximum payload
relative-position error reached about 0.17 m. Higher payload masses must not be
treated as accepted until an outer formation/centroid feedback controller is
added and the 0.5 kg case passes.

The added formation controller revealed a second limitation of the exported
policy. Its low-level velocity response has a dead zone, delayed forward
response and ineffective reverse response under the closed-chain arm load.
Proportional bidirectional feedback therefore overshoots. The retained
diagnostic implementation uses short, timed, per-robot forward pulses with a
cooldown and keeps yaw correction disabled. This reduced maximum 0.5 kg base
drift from about 0.91 m to about 0.32 m, but payload-relative error remained
about 0.21 m and the acceptance test still failed. Adding horizontal payload
impedance increased closed-chain base drift, so only vertical load compensation
is retained.

The next robust controller milestone is a load-aware locomotion policy or a
wheel-leg policy retrained with randomized arm wrench and shared-payload
dynamics. The pulse controller is a system-identification diagnostic, not a
hardware-ready station-keeping controller.
