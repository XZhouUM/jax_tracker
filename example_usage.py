#!/usr/bin/env python3
"""
Example usage of the multi-track tracker in JAX.

This script demonstrates how to:
1. Set up a multi-track tracker with different configurations
2. Simulate measurements from multiple targets
3. Run the tracker and visualize results
"""

from typing import Dict, List

import jax.numpy as jnp
import jax.random
import numpy as np

from measurement_models.radar_measurement_model import RadarMeasurement
from measurement_preprocessing import (MeasurementPreprocessor,
                                       MeasurementSimulator, SensorInfo)
from motion_models.constant_velocity_model import ConstantVelocity
from multi_track_tracker import MultiTrackTracker, TrackerConfig


def create_example_sensors() -> Dict[str, SensorInfo]:
    """Create example sensor configuration."""
    sensors = {
        "radar_1": SensorInfo(
            sensor_id="radar_1",
            sensor_type="radar",
            position=jnp.array([0.0, 0.0, 0.0]),
            orientation=jnp.array([0.0, 0.0, 0.0]),
            measurement_model=RadarMeasurement,
            default_noise_covariance=jnp.diag(
                jnp.array([1.0, 0.5, 0.01])
            ),  # range, range_rate, azimuth
        )
    }
    return sensors


def simulate_target_trajectories(
    num_targets: int = 3, num_steps: int = 50, dt: float = 1.0
) -> List[List[jnp.ndarray]]:
    """
    Simulate target trajectories for testing.

    Returns:
        List of trajectories, where each trajectory is a list of state vectors
    """
    trajectories = []

    # Create different target scenarios
    initial_states = [
        jnp.array([100.0, 50.0, 10.0, 5.0]),  # Target 1: moving northeast
        jnp.array([-80.0, 120.0, -5.0, -8.0]),  # Target 2: moving southwest
        jnp.array([200.0, -100.0, -15.0, 12.0]),  # Target 3: moving northwest
    ]

    for i in range(min(num_targets, len(initial_states))):
        trajectory = []
        state = initial_states[i]

        for step in range(num_steps):
            trajectory.append(state)
            # Propagate using constant velocity model
            state = ConstantVelocity.transition(state, dt)

        trajectories.append(trajectory)

    return trajectories


def run_tracking_example():
    """Run a complete tracking example."""
    print("=== Multi-Track Tracker Example ===")

    # Configuration
    num_targets = 3
    num_steps = 50
    dt = 1.0

    # Create tracker configuration
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        data_association_algorithm="nearest_neighbor",
        gate_threshold=10.0,
        use_ellipsoidal_gate=True,
        confirmation_threshold=3,
        deletion_threshold=5,
        max_tracks=20,
        process_noise_scale=0.1,
        initial_covariance_scale=100.0,
    )

    # Initialize tracker
    tracker = MultiTrackTracker(config)

    # Set up sensors and preprocessor
    sensors = create_example_sensors()
    preprocessor = MeasurementPreprocessor(sensors)
    simulator = MeasurementSimulator(sensors)

    # Generate true trajectories
    true_trajectories = simulate_target_trajectories(num_targets, num_steps, dt)

    # Storage for results
    tracking_results = []
    all_measurements = []

    print(f"Simulating {num_targets} targets for {num_steps} time steps...")

    # Run tracking simulation
    for step in range(num_steps):
        current_time = step * dt

        # Get true states at current time
        true_states = [trajectory[step] for trajectory in true_trajectories]

        # Simulate measurements
        key = jax.random.PRNGKey(step)  # Use step as seed for reproducibility
        raw_measurements = simulator.simulate_measurements(
            true_states=true_states,
            timestamp=current_time,
            detection_probability=0.9,
            false_alarm_rate=0.2,
            key=key,
        )

        # Preprocess measurements
        processed_measurements = preprocessor.preprocess_measurements(
            raw_measurements, current_time
        )

        # Update tracker
        result = tracker.update(
            measurements=processed_measurements, dt=dt, current_time=current_time
        )

        # Store results
        tracking_results.append(result)
        all_measurements.append(processed_measurements)

        # Print progress
        if step % 10 == 0:
            stats = tracker.get_statistics()
            print(
                f"Step {step}: {stats['confirmed_tracks']} confirmed tracks, "
                f"{stats['tentative_tracks']} tentative tracks, "
                f"{len(processed_measurements)} measurements"
            )

    # Print final statistics
    final_stats = tracker.get_statistics()
    print(f"\n=== Final Results ===")
    print(f"Total tracks created: {final_stats['next_track_id']}")
    print(f"Confirmed tracks: {final_stats['confirmed_tracks']}")
    print(f"Tentative tracks: {final_stats['tentative_tracks']}")
    print(f"Deleted tracks: {final_stats['deleted_tracks']}")

    # Get final track states
    final_tracks = tracker.get_confirmed_track_states()
    print(f"\nFinal confirmed track positions:")
    for track_id, track_data in final_tracks.items():
        x, y = track_data["state"][:2]
        print(f"Track {track_id}: position ({x:.1f}, {y:.1f}), age {track_data['age']}")

    return tracker, tracking_results, true_trajectories, all_measurements


def plot_tracking_results(
    tracker, tracking_results, true_trajectories, all_measurements
):
    """Plot tracking results."""
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))

        # Plot true trajectories
        for i, trajectory in enumerate(true_trajectories):
            positions = np.array([[state[0], state[1]] for state in trajectory])
            plt.plot(
                positions[:, 0],
                positions[:, 1],
                "--",
                label=f"True Target {i+1}",
                linewidth=2,
                alpha=0.7,
            )
            plt.plot(positions[0, 0], positions[0, 1], "o", markersize=8)  # Start
            plt.plot(positions[-1, 0], positions[-1, 1], "s", markersize=8)  # End

        # Plot track estimates
        tracker.get_all_track_states()
        track_histories = {}

        # Collect track histories
        for result in tracking_results:
            for track_id in result.get("confirmed_tracks", {}):
                if track_id not in track_histories:
                    track_histories[track_id] = []
                track_data = result["confirmed_tracks"][track_id]
                track_histories[track_id].append(track_data["state"][:2])

        # Plot track histories
        for track_id, history in track_histories.items():
            if len(history) > 2:  # Only plot tracks with sufficient history
                positions = np.array(history)
                plt.plot(
                    positions[:, 0],
                    positions[:, 1],
                    "-",
                    label=f"Track {track_id}",
                    linewidth=2,
                )

        # Plot measurements (sample from first few time steps)
        for step in range(min(5, len(all_measurements))):
            measurements = all_measurements[step]
            for measurement, measurement_model, _ in measurements:
                if measurement_model == RadarMeasurement:
                    # Convert radar measurement to Cartesian
                    range_val, _, azimuth = measurement
                    x = range_val * jnp.cos(azimuth)
                    y = range_val * jnp.sin(azimuth)
                    plt.plot(x, y, "x", color="gray", alpha=0.5, markersize=4)

        plt.xlabel("X Position (m)")
        plt.ylabel("Y Position (m)")
        plt.title("Multi-Track Tracker Results")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.tight_layout()
        plt.show()

    except ImportError:
        print(
            "Matplotlib not available for plotting. Install with: pip install matplotlib"
        )


def run_performance_test():
    """Run a performance test with different configurations."""
    print("\n=== Performance Test ===")

    configurations = [
        ("Nearest Neighbor", "nearest_neighbor"),
        ("Global Nearest Neighbor", "global_nearest_neighbor"),
    ]

    for name, algorithm in configurations:
        print(f"\nTesting {name}...")

        config = TrackerConfig(
            motion_model_class=ConstantVelocity,
            data_association_algorithm=algorithm,
            gate_threshold=10.0,
            confirmation_threshold=2,
            deletion_threshold=3,
        )

        tracker = MultiTrackTracker(config)

        # Simple test with synthetic measurements
        for step in range(20):
            # Create simple measurements
            measurements = [
                (
                    jnp.array([100.0 + step * 5, 50.0 + step * 3, 0.5]),
                    RadarMeasurement,
                    jnp.eye(3),
                ),
                (
                    jnp.array([200.0 - step * 2, 100.0 + step * 4, -0.3]),
                    RadarMeasurement,
                    jnp.eye(3),
                ),
            ]

            result = tracker.update(measurements, dt=1.0)

        stats = tracker.get_statistics()
        print(
            f"  Final tracks: {stats['confirmed_tracks']} confirmed, {stats['tentative_tracks']} tentative"
        )


if __name__ == "__main__":
    # Run the main example
    tracker, results, trajectories, measurements = run_tracking_example()

    # Plot results if matplotlib is available
    plot_tracking_results(tracker, results, trajectories, measurements)

    # Run performance test
    run_performance_test()

    print("\n=== Example Complete ===")
