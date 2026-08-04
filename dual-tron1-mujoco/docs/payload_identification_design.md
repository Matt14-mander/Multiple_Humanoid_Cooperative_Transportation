# Payload mass and center-of-mass identification

## System boundary

Fixed grippers, adapters and cables belong in each AIRBOT/Pinocchio model.
The carried object remains an independent rigid body. Its gravity load is
therefore visible in the two arm contact-wrench estimates and can be used for
object-level identification and load allocation.

## Identified parameters

The first implementation estimates four low-dynamic parameters:

```text
theta = [m, m*c_x, m*c_y, m*c_z]
```

For object-origin linear acceleration `a`, gravity `g`, and body-to-world
rotation `R`, define `u = a - g`. Neglecting rotational-inertia terms during
the safe identification motion gives:

```text
F           = m*u
tau_origin  = -skew(u)*R*(m*c)
```

The two contact wrenches are first composed about the object-frame origin.
The resulting six-dimensional wrench is fitted with robust sliding-window
least squares, projected into configured mass and COM bounds.

## State machine

```text
calibrating -> frozen -> reidentification_required
```

- `calibrating`: collect only samples from the stationary identification
  phase. Full rank, bounded condition number, sufficient sample count and a
  low residual are all required before freezing.
- `frozen`: do not change mass or COM during transport. Monitor the dynamics
  innovation only. Isolated impacts do not change the estimate.
- `reidentification_required`: entered after a configured number of
  consecutive innovation violations. The intended response is to stop,
  stabilize and explicitly restart calibration.

A single level pose cannot identify every COM component. The planned physical
sequence must include small, safe pitch and roll poses. Angular-inertia terms
are intentionally outside this first low-dynamic model.

## Controller integration

`CooperativeCarryHoldController.set_payload_estimate()` applies an accepted
mass and body-frame COM. Gravity compensation uses the estimated mass, and the
grasp matrix is formed about the estimated world COM. Contact allocation is
the equality-constrained quadratic solution with high contact-moment cost,
so an offset COM changes the vertical load carried by each robot.

This is the equality-constrained core of the future full QP. Joint-torque,
friction-cone and stability-margin inequalities remain a later integration
stage.

## Commands

Validate the centered dumbbell and fixed-offset box profiles:

```powershell
python -m dual_tron1_mujoco.payload_identification_validation
python -m dual_tron1_mujoco.payload_identification_validation --json
```

Generate a repeatable 6 kg box whose COM is fixed away from its geometry
origin:

```powershell
python -m dual_tron1_mujoco.build_scene `
  --config configs/wf_dual_carry_hold.json `
  --output models/generated/wf_dual_carry_hold_offset.xml `
  --payload-mass 6 `
  --payload-com 0.08 -0.05 0.03
```
