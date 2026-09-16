# Single TRON2 CHIP

Single-robot `WFYG_TRON2A` scaffold for reproducing CHIP-style adaptive
compliance before integration into `dual-tron2-mujoco`.

## Implemented milestone

- Reuses the official model in `../tron2-robot-description`; assets are not copied.
- Builds one fixed-base robot with the `gripper_pick` end-effector site.
- Implements the CHIP training transformation
  `g_hind = g_ref - C f_ext` in the world frame.
- Applies a timed MuJoCo force pulse to the end-effector.
- Provides a 10-step history buffer for the future actor observation.
- Runs gravity-only feedforward plus Cartesian spring-damper impedance as a
  wiring surrogate.
- Records reference, hindsight goal, force and end-effector motion to CSV.

The analytic controller is **not** the learned CHIP policy. Its purpose is to
validate model names, frames, units, force injection and the online hindsight
goal path before PPO training is added. Tracking reward must continue to use
the unmodified `g_ref` when the training environment is implemented.

The fixed-base sanity scene uses a collision-free, well-conditioned arm pose,
raises the base by 2 mm to remove initial wheel penetration, and augments the
source arm model with reflected armature and damping that were absent from the
CAD export. Gravity feedforward is evaluated with zero velocity so Coriolis
terms cannot reinforce transient motion. For the default `C=0.002 m/N`, the
surrogate uses `K=1/C=500 N/m`; its spring force therefore cancels the default
10 N training perturbation at the original reference pose.

## Setup

```powershell
cd "E:\Robot\Sustech-多人形协作\Multiple_Humanoid_Cooperative_Transportation\single-tron2-chip"
python -m pip install -e ".[test]"
```

## Run

Headless formula and MuJoCo smoke test:

```powershell
python -m pytest -v
python -m tron2_chip.run_sanity --headless --rebuild
```

Interactive viewer:

```powershell
python -m tron2_chip.run_sanity --rebuild
```

Deployment-mode compliance test (the actor goal is not shifted by force):

```powershell
python -m tron2_chip.run_deployment `
  --compliance 0.002 0 0 `
  --force 10 0 0 `
  --headless `
  --rebuild
```

In deployment mode the analytic oracle should yield approximately
`C*f = 0.02 m`, whereas the training sanity mode stays near the original goal.

Run a repeatable force/compliance sweep:

```powershell
python -m tron2_chip.evaluation.compliance_sweep `
  --compliances 0.001 0.002 0.004 `
  --forces 2 5 10 `
  --axes x `
  --output-dir runs/deployment_sweep
```

The per-case rollouts and `summary.csv` contain measured compliance, relative
error, peak displacement, recovery time and control-limit usage.

Plot the recorded rollout as a four-panel diagnostic figure:

```powershell
python -m tron2_chip.plot_csv runs/arm_sanity_latest.csv
```

Save to a selected path and open an interactive Matplotlib window:

```powershell
python -m tron2_chip.plot_csv runs/arm_sanity_latest.csv `
  --output runs/arm_sanity_diagnostics.png `
  --axis auto `
  --show
```

The figure compares the actual end-effector, original goal and hindsight goal;
shows the three world-frame force components; plots actual and commanded
offsets on the dominant force axis; and marks the configured actuator limit.

Override directional compliance and force:

```powershell
python -m tron2_chip.run_sanity `
  --compliance 0.002 0 0 `
  --force 10 0 0 `
  --headless `
  --rebuild
```

The default pulse produces the expected training-observation shift
`[-0.02, 0, 0] m`. CSV output is written to
`runs/arm_sanity_latest.csv`.

## Next milestone

1. Add a Gymnasium-compatible motion-tracking environment.
2. Define actor observations from proprioception/action history, `g_hind` and `C`.
3. Give the critic privileged access to the ground-truth perturbation force.
4. Train PPO on fixed-base reach motions and compare stiff, random-force and CHIP ablations.
5. Release the base weld and add the hybrid wheel/leg/arm whole-body action adapter.

## PAINT-style intent estimator scaffold

`tron2_chip.intent_estimation` now contains a simulator-independent baseline
for estimating planar partner intent `[Fx, Fy, Mz]` from four steps of Airbot
joint position, velocity and previously applied arm-position commands. It
includes the versioned 72-to-3 data contract, causal history, normalization,
three-layer PyTorch MLP, regression losses, episode-disjoint dataset splits,
standalone training, runtime filtering and ONNX export.

The input dataset is an NPZ containing `features: [N, 72]`, `targets: [N, 3]`
and `episode_ids: [N]`. Train outside Isaac Lab with:

```bash
pip install -e ".[train,test]"
tron2-intent-train datasets/paint_intent_train.npz \
  --output-dir artifacts/intent_estimator \
  --epochs 100 --batch-size 1024
```

The resulting artifact bundle contains the model checkpoint, hashed signal
specification, input/output normalization and a JSON training report. The
Isaac Lab adapter supplies batched history construction and world-to-base-yaw
wrench-label transforms, but data collection still needs to be connected to
the configured TRON2_YG articulation and payload event manager on Ubuntu.

An OCS2 rollout bundle containing per-episode NPZ files can be adapted without
mixing adjacent samples across dataset splits:

```bash
tron2-intent-prepare-ocs2 /path/to/rollout \
  --output datasets/ocs2_intent_v1.npz \
  --spec-output datasets/ocs2_intent_v1_spec.json \
  --report-output datasets/ocs2_intent_v1_preparation_report.json

tron2-intent-train datasets/ocs2_intent_v1.npz \
  --spec datasets/ocs2_intent_v1_spec.json \
  --output-dir artifacts/ocs2_intent_v1 \
  --epochs 100 --batch-size 1024

tron2-intent-export artifacts/ocs2_intent_v1
```

The OCS2 adapter uses four frames of actual arm position/velocity and the
one-step-shifted arm position reference. Its provisional label is horizon zero
of `future_wrench`, projected to `[Fx, Fy, Mz]`. This is useful for pipeline
validation but is not a substitute for a verified partner-applied payload
wrench label; the generated preparation report records this limitation and
flags target channels with insufficient excitation.

## Cross-platform hand-off

`tron2_chip.core` defines the versioned `PolicySpec`, history-based observation
builder, normalization and action scaling. `tron2_chip.deployment` contains a
backend-independent policy runtime, optional ONNX Runtime backend, normalized
action safety filter and the future robot-interface protocol. These modules do
not depend on MuJoCo or Isaac Lab.

`tron2_chip.backends.isaaclab` currently contains the validated task contract
and batched CHIP tensor terms. The Ubuntu training entry point deliberately
stops until the WFYG_TRON2A USD articulation and Isaac Lab managers have been
registered; it does not claim that PPO training is already available.
