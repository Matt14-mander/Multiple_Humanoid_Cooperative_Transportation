"""
Example: Force Estimation Demo

Demonstrates the force estimator using a simple single-robot scenario.
"""

import numpy as np
import pinocchio as pin
from pathlib import Path
import sys

# Add src to path
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from internal_force_suppression.core.force_estimator import (
    ImplicitForceEstimator,
    GeneralizedMomentumObserver
)


def create_simple_robot():
    """Create a simple 2-link robot for testing."""
    model = pin.Model()

    # Base joint (fixed)
    parent_id = 0

    # Joint 1 (revolute)
    joint1 = pin.JointModelRY()
    placement1 = pin.SE3.Identity()
    body1_inertia = pin.Inertia.FromSphere(1.0, 0.1)  # 1kg sphere
    joint1_id = model.addJoint(parent_id, joint1, placement1, "joint1")
    model.appendBodyToJoint(joint1_id, body1_inertia, pin.SE3.Identity())

    # Joint 2 (revolute)
    joint2 = pin.JointModelRY()
    placement2 = pin.SE3(np.eye(3), np.array([0, 0, 0.5]))  # 0.5m offset
    body2_inertia = pin.Inertia.FromSphere(0.5, 0.05)  # 0.5kg sphere
    joint2_id = model.addJoint(joint1_id, joint2, placement2, "joint2")
    model.appendBodyToJoint(joint2_id, body2_inertia, pin.SE3.Identity())

    # Add end-effector frame
    ee_placement = pin.SE3(np.eye(3), np.array([0, 0, 0.5]))
    model.addFrame(pin.Frame("end_effector", joint2_id, ee_placement, pin.FrameType.OP_FRAME))

    return model


def simulate_external_force(t, amplitude=10.0, frequency=1.0):
    """Generate a sinusoidal external force for testing."""
    return amplitude * np.sin(2 * np.pi * frequency * t)


def main():
    print("=" * 60)
    print("Force Estimator Demo")
    print("=" * 60)

    # Create robot model
    robot = create_simple_robot()
    print(f"\n✓ Created robot model with {robot.nv} DOF")

    # Create force estimator
    config = {
        'estimator_type': 'momentum_observer',
        'observer_gain': 100.0,
        'cutoff_frequency': None,
        'end_effector_frame': 'end_effector'
    }

    estimator = ImplicitForceEstimator(robot, config)
    print("✓ Initialized force estimator")

    # Simulation parameters
    dt = 0.002  # 500 Hz
    duration = 2.0  # seconds
    steps = int(duration / dt)

    # Initial state
    q = np.array([0.0, 0.0])  # Joint positions
    v = np.array([0.0, 0.0])  # Joint velocities

    # Storage for results
    time_history = []
    force_true_history = []
    force_est_history = []

    print(f"\n✓ Running simulation for {duration}s at {1/dt:.0f} Hz...")
    print(f"  Applying sinusoidal external force to joint 1")

    # Simulation loop
    for step in range(steps):
        t = step * dt

        # Generate true external torque (sinusoidal on joint 1)
        tau_ext_true = np.array([simulate_external_force(t, amplitude=10.0), 0.0])

        # Compute gravity and other terms
        data = robot.createData()
        pin.computeAllTerms(robot, data, q, v)
        tau_gravity = data.g

        # Total measured torque = gravity + external
        tau_measured = tau_gravity + tau_ext_true

        # Estimate external torque
        robot_state = {
            'q': q,
            'v': v,
            'tau': tau_measured,
            'dt': dt
        }

        wrench_info = estimator.estimate_contact_wrench(robot_state)
        tau_ext_estimated = wrench_info['tau_ext']

        # Store results
        time_history.append(t)
        force_true_history.append(tau_ext_true[0])
        force_est_history.append(tau_ext_estimated[0])

        # Simple dynamics integration (for demonstration)
        # In reality, you'd have more complex dynamics
        q += v * dt
        v += 0.0  # No actual motion for this test

    # Compute estimation error
    force_true = np.array(force_true_history)
    force_est = np.array(force_est_history)
    error = force_true - force_est

    rmse = np.sqrt(np.mean(error**2))
    max_error = np.max(np.abs(error))

    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    print(f"RMSE: {rmse:.3f} N⋅m")
    print(f"Max Error: {max_error:.3f} N⋅m")
    print(f"Mean True Force: {np.mean(np.abs(force_true)):.3f} N⋅m")
    print(f"Mean Estimated Force: {np.mean(np.abs(force_est)):.3f} N⋅m")

    # Plot results (if matplotlib available)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 6))

        plt.subplot(2, 1, 1)
        plt.plot(time_history, force_true, 'b-', label='True', linewidth=2)
        plt.plot(time_history, force_est, 'r--', label='Estimated', linewidth=1.5)
        plt.xlabel('Time (s)')
        plt.ylabel('External Torque (N⋅m)')
        plt.title('Force Estimation Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(2, 1, 2)
        plt.plot(time_history, error, 'g-', linewidth=1)
        plt.xlabel('Time (s)')
        plt.ylabel('Estimation Error (N⋅m)')
        plt.title(f'Estimation Error (RMSE: {rmse:.3f} N⋅m)')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        output_path = Path(__file__).parent / "force_estimation_demo.png"
        plt.savefig(output_path, dpi=150)
        print(f"\n✓ Plot saved to: {output_path}")

        plt.show()

    except ImportError:
        print("\n⚠ matplotlib not available, skipping plot")

    print("\n✓ Demo complete!")


if __name__ == "__main__":
    main()
