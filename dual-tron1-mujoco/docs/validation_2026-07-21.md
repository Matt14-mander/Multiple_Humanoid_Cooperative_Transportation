# Windows baseline validation

Environment used for the first local validation:

- Windows x64
- CPython 3.12 (isolated project-local dependencies)
- MuJoCo 3.2.7
- NumPy 1.26.3
- no ROS and no LIMX runtime

## Verified

- Generated model compiles as `nq=49`, `nv=46`, `nu=28`, `neq=4`.
- Two independent 14-actuator name maps resolve without numeric offsets.
- All referenced TRON1 and AIRBOT mesh files exist.
- AIRBOT URDF fixed-axis RPY is converted to MuJoCo quaternion convention.
- Neutral `link6` positions are within 2 mm of their payload handle frames.
- The model completes 5,000 controlled steps (5 seconds) without NaN/Inf.
- Chinese Windows workspace paths work through the in-memory asset loader.
- WF encoder/policy interfaces are 370->64 and (74+64)->14.

## First dynamic result

With the placeholder 10 kg payload, fixed robot bases, compliant grasp welds,
and deployment-like joint PD gains, the 5 second settling run remained
numerically finite but the payload sagged from 1.160 m to about 0.991 m. Peak
command magnitude was about 9.51 N.m. The raw equality solver rows reached
large values; these are useful comparative diagnostics but are not calibrated
force/torque sensor readings.

This is not a successful carrying controller. It is an actionable finding:
the neutral-pose joint hold controller does not maintain the closed-chain pose
under the placeholder load. Before locomotion is enabled, sweep measured
payload mass/COM, add gravity compensation or task-space load sharing, verify
all AIRBOT joint limits, and inspect wrist torque margin.

## Deployment blockers still open

1. The downloaded AIRBOT Play URDF contains no independent gripper joint or
   collision model; the current grasp is a compliant weld stub.
2. The repository's AIRBOT SDK binaries are Linux `.so` files. A Windows
   AIRBOT hardware API or a separate Linux hardware bridge is still required.
3. The LIMX wheel is Windows x64 but links to `python38.dll`; real LIMX tests
   need a separate CPython 3.8 x64 environment.
4. Two-robot clocks, command deadlines, packet loss, watchdogs and emergency
   stop behavior are absent from pure MuJoCo and need hardware-in-the-loop.
5. The ONNX observation builder and two independent history buffers have not
   yet been ported; current control is deterministic torque PD only.
