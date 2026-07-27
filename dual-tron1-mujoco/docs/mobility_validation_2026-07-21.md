# Dual-robot mobility validation

Command used on Windows:

```powershell
.\scripts\run_forward.ps1 -WorldVx 0.1 -Duration 9 -Headless
```

The dedicated mobility configuration places both robots side-by-side with the
same heading. The base welds and grasp welds are inactive and the unused
payload is invisible and non-colliding.

The schedule was:

- 0--2 s: zero command;
- 2--7 s: normalized world-frame x command `0.1`;
- 7--9 s: zero command.

Result:

- both simulations completed 9,000 dynamics steps;
- robot 1 x displacement: about 0.7514 m;
- robot 2 x displacement: about 0.7514 m;
- final base height: about 0.7011 m for both robots;
- no NaN, Inf, base-height limit or tilt limit was triggered.

This validates the Windows ONNX execution path, independent history buffers,
action mapping, world/body command conversion and simultaneous 28-actuator
control. It does not yet prove accurate velocity tracking: the robots drifted
about -0.145 m before the nonzero command and coasted about +0.246 m after the
command returned to zero. Standing calibration, command ramping and braking
behavior therefore remain open before payload transport is enabled.
