# Dual TRON2 + arm MuJoCo simulation

Windows-native, ROS-free MuJoCo baseline for two official `WFYG_TRON2A`
robots and one shared payload. The project is parallel to
`dual-tron1-mujoco` and reads the complete wheel-leg, six-axis arm and
two-finger gripper model from `../tron2-robot-description`.

## Current milestone

- Two independently prefixed official `WFYG_TRON2A` model instances.
- 10 wheel-leg, 6 arm and 2 gripper actuators per robot.
- One free payload with configurable mass and COM.
- Fixed-base isolation test and soft welds from both `gripper_pick` bodies.
- Gravity/bias feedforward plus deterministic joint hold control.
- Headless regression, interactive MuJoCo Viewer and CSV recording.

Generated dimensions:

```text
nq = 57, nv = 54, nu = 36
```

This milestone does not reuse the TRON1 ONNX locomotion policy: TRON2 has a
different observation, action and robot-dynamics interface. Walking,
object-level impedance, momentum-observer and internal-force feedback are the
next porting stages.

## Setup

From an environment containing MuJoCo and NumPy:

```powershell
cd "E:\Robot\Sustech-多人形协作\Multiple_Humanoid_Cooperative_Transportation\dual-tron2-mujoco"
python -m pip install -e .
```

Alternatively, without installing, set:

```powershell
$env:PYTHONPATH = "src"
```

## Build and run

```powershell
python -m dual_tron2_mujoco.build_scene
python -m dual_tron2_mujoco.run_sim --headless --duration 1 --rebuild
python -m dual_tron2_mujoco.run_sim --duration 20 --rebuild
```

Run an asymmetric 2 kg payload:

```powershell
python -m dual_tron2_mujoco.run_sim `
  --payload-mass 2 `
  --payload-com 0.05 -0.03 0.02 `
  --duration 20 `
  --rebuild
```

Or use:

```powershell
.\scripts\run_visual.ps1
.\scripts\run_smoke.ps1
```

CSV output is written to `runs/latest.csv`.

