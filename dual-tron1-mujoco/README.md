# Dual TRON1 cooperative-carry MuJoCo simulation

Windows-native, ROS-free dynamics simulation for two `WF_TRON1A` robots with
AIRBOT six-axis arms and one payload with two side handles.

## Current milestone

- Reuses the wheel-leg MJCF from `tron1-mujoco-sim`.
- Reads the arm chain, inertias, axes and limits from
  `tron1-rl-deploy-arm/.../robot_with_arm.urdf`.
- Generates two name-prefixed robots, a 10 kg payload and four configurable
  weld constraints in one MJCF.
- Uses name-based joint and actuator lookup; no single-robot qpos offsets.
- Provides a deterministic torque-PD settling controller and CSV recording.
- Provides an ONNX shape inspection entry point. Policy observation/action
  porting is the next control milestone.

The generated free-floating dimensions are:

```text
nq = 49, nv = 46, nu = 28
```

Two base-to-world welds are active by default so the first test isolates the
arm/payload closed chain. Disable `model.fixed_bases` only after this baseline
passes.

## Windows setup (simulation, no ROS)

Install 64-bit CPython 3.10--3.12, then run:

```powershell
cd "E:\Robot\Sustech-多人形协作\dual-tron1-mujoco"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_windows.ps1
.\scripts\run_smoke.ps1
```

This is the complete simulation path. It does not install ROS, LIMX SDK or the
Linux-only AIRBOT SDK. `model_loader.py` loads XML and meshes through memory so
MuJoCo also works when the Windows project path contains Chinese characters.

## Optional LIMX Windows environment

The bundled LIMX wheel contains an x64 extension linked against
`python38.dll`, so use CPython 3.8 x64 for the SDK-compatible environment.

```powershell
cd "E:\Robot\Sustech-多人形协作\dual-tron1-mujoco"
Set-ExecutionPolicy -Scope Process Bypass
.\scripts\setup_limx_windows.ps1
```

For pure MuJoCo development, LIMX SDK is optional. A supported newer Python can
also install the project, but keep `numpy < 1.26.4` if the same environment will
later load the LIMX wheel.

## Build and test

```powershell
.\.venv-sim\Scripts\Activate.ps1
dual-tron1-build
pytest -q
dual-tron1-sim --headless --duration 0.1
dual-tron1-sim --duration 5
```

## AIRBOT sensorless force-observer validation

The first sensorless-estimation stage uses a fixed-base AIRBOT arm. The
validation model is generated from the same deployed `robot_with_arm.urdf`
chain for both dynamics engines: MuJoCo supplies the applied-wrench truth,
while the Pinocchio generalized-momentum observer receives only joint
position, velocity and actuator torque.

Run the zero-force, static-wrench and moving-wrench comparisons with:

```powershell
python -m dual_tron1_mujoco.airbot_observer_validation
python -m dual_tron1_mujoco.airbot_observer_validation --json
```

The MuJoCo truth is used only for error reporting. It is not passed into the
observer. Regression coverage is in `tests/test_airbot_observer_validation.py`.

Run the repeatable robustness matrix with:

```powershell
python -m dual_tron1_mujoco.airbot_observer_robustness
python -m dual_tron1_mujoco.airbot_observer_robustness --json
python -m dual_tron1_mujoco.airbot_observer_robustness --case model-errors
```

It covers encoder/velocity noise, torque calibration error, separately
controlled mass-matrix, link-mass, center-of-mass and gravity mismatch,
combined spatial-inertia mismatch,
unmodeled friction and tool mass, drivetrain loss, delayed and jittered
samples, multiple poses, a near-singular pose, a 40 ms impulse, and a combined
adverse case. `PASS` and `FAIL` use provisional stage-one limits; known failing
cases remain in regression coverage so model calibration and tool/gravity
compensation work cannot accidentally be mistaken for completed robustness.

Two static false-force sources now have explicit compensation paths. Fixed
gripper/tool mass, COM and box inertia are merged into the terminal Pinocchio
link model. Remaining slow joint-residual bias is learned only in
`free_space`/`release` and frozen in `grasp`/`carry`. The task state is supplied
as `robot_state["contact_phase"]` or through
`ImplicitForceEstimator.set_contact_phase()`. Compare the original and fixed
cases with:

```powershell
python -m dual_tron1_mujoco.airbot_observer_robustness --case mass_10pct_low_zero
python -m dual_tron1_mujoco.airbot_observer_robustness --case mass_10pct_low_zero_compensated
python -m dual_tron1_mujoco.airbot_observer_robustness --case unmodeled_tool_200g_zero
python -m dual_tron1_mujoco.airbot_observer_robustness --case modeled_tool_200g_zero
```

## Adaptive payload identification

The object-level payload estimator identifies mass and body-frame center of
mass from a short multi-pose calibration sequence, checks observability, and
then freezes the accepted parameters during transport. The carry controller
uses the frozen estimate for gravity compensation and asymmetric load
allocation without adding the carried object to either arm model.

Run the deterministic centered-dumbbell and offset-box validation with:

```powershell
python -m dual_tron1_mujoco.payload_identification_validation
python -m dual_tron1_mujoco.payload_identification_validation --json
```

See `docs/payload_identification_design.md` for the model, state machine,
limits and repeatable payload-generation command.

## Dual-robot forward test

The mobility test uses `configs/wf_dual_forward.json`: both robots are
side-by-side with the same heading, their base welds are disabled, and the
released payload collision is disabled. Run it from Git Bash with:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./scripts/run_forward.ps1
```

Use a headless run or change the normalized upstream command:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./scripts/run_forward.ps1 \
  -WorldVx 0.1 -Duration 9 -Headless
```

The schedule is stationary for 2 s, commands forward motion from 2--7 s, and
then commands zero until 9 s. `WorldVx` is the original policy command in the
range `[-1, 1]`, not a guaranteed SI velocity. Actual displacement must be
measured from `runs/forward_latest.csv`.

For the carrying scene the robots are arranged side-by-side behind the
payload, with both headings aligned to world +X. The payload handles extend
from its left and right sides along the Y axis, and both robots receive the
same local forward command. Run the preliminary 2 kg cooperative-carry case
from the project root with:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./run_carry.ps1
powershell.exe -ExecutionPolicy Bypass -File ./run_carry.ps1 \
  -WorldVx 0.02 -Duration 9 -Headless
```

This keeps the payload visible and both compliant grasp welds active while
unlocking only the robot bases. Results are written to `runs/carry_latest.csv`.
Treat it as a coupled-dynamics diagnostic rather than a successful transport
benchmark: the deployed single-robot policy was not trained for a mirrored,
closed-chain payload task.

The console summary includes both base displacements, payload displacement,
final heights and the two grasp gaps. A run completing without a numerical
error only proves that MuJoCo integrated the model; large payload height loss,
asymmetric base motion or growing grasp gaps still mean the carry controller
failed dynamically.

## Fixed-base static load control

The first cooperative arm controller keeps both robot bases fixed, holds the
payload pose with object-level Cartesian impedance, distributes the requested
wrench through the dual-grasp matrix, and maps each end-effector wrench to arm
joint torque with `J^T f` plus MuJoCo bias-force compensation. Run the default
0.5 kg acceptance case with:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./run_carry_hold.ps1
```

Select another positive payload mass and optionally run without a viewer:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./run_carry_hold.ps1 \
  -PayloadMass 2.0 -Duration 10 -Headless
```

The result is written to `runs/carry_hold_latest.csv`. This file includes
payload tilt and separate arm torque peaks in addition to the existing pose,
grasp-gap and constraint diagnostics. See `docs/control_design.md` for the
equations, assumptions and limitations.

## Free-base loaded balance test

After the fixed-base test passes, release both base welds while keeping a zero
locomotion command with:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./run_carry_balance.ps1
```

Or run the initial 0.5 kg case without a viewer:

```bash
powershell.exe -ExecutionPolicy Bypass -File ./run_carry_balance.ps1 \
  -PayloadMass 0.5 -Duration 10 -Headless
```

This mode keeps the complete ONNX policy actuation and adds ramped `J^T f`
payload compensation to the arm torques. It records base tilt and planar speed
in `runs/carry_balance_latest.csv` and prints separate payload and balance
acceptance results. Do not increase the payload merely because the process
completes: `balance: FAIL` means the lower-mass stage has not been accepted.

The current exported policy has a delayed, asymmetric command response under
closed-chain load and does not provide reliable reverse station keeping. The
diagnostic formation loop therefore uses short per-robot forward pulses and
vertical-only payload compensation. It substantially reduces drift, but the
0.5 kg case is not yet an accepted controller; do not deploy this pulse logic
as a real-robot station-keeping solution.

The viewer run writes `runs/latest.csv` with payload/base positions,
constraint error, per-grasp constraint force proxies and peak control
magnitude. The force columns are solver-row diagnostics, not calibrated
six-axis load-cell measurements.

Inspect the original deployment networks with:

```powershell
dual-tron1-inspect-onnx
```

The checked-in WF networks were verified as:

```text
encoder: obs_history [1, 370] -> latent [1, 64]
policy:  obs [1, 74] + latent [1, 64] -> actions [1, 14]
```

The intended dual setup therefore runs one 74-D observation/history stack and
one policy session per robot. Sharing one policy's state between robots would
be an implementation error.

## Modeling scope

The current `gripper_stub` is visual only. The two grasps are compliant welds
between each `link6` and the corresponding payload handle. This deliberately
validates the coupled rigid-body model before frictional finger contact is
introduced. The later G2 model will use one commanded opening coordinate in
the documented 0--0.072 m range and two coupled fingers.

## What this milestone can reveal

- closed-chain inconsistency and grasp solver forces;
- payload drift, arm/leg torque saturation and controller instability;
- asymmetry caused by mirrored robot frames;
- sensitivity to payload mass, handle spacing and weld compliance;
- Windows dependency, non-ASCII path and ONNX interface problems.

It cannot yet validate finger contact pressure, backlash, CAN timing, AIRBOT
Windows hardware control, clock synchronization, emergency-stop behavior or
network packet loss. Those are explicit hardware-in-the-loop milestones, not
properties that a rigid weld can prove.
