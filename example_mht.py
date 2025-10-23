#!/usr/bin/env python3
"""
Example demonstrating Multi-Hypothesis Tracking (MHT) in action.

This example shows how MHT handles challenging scenarios like:
- Closely spaced targets
- False alarms
- Missed detections
- Track initialization ambiguity
"""

from typing import List, Tuple

import jax.numpy as jnp
import jax.random as random

from measurement_models.position_measurement_model import PositionMeasurement
from motion_models.constant_velocity_model import ConstantVelocity
from multi_track_tracker import MultiTrackTracker, TrackerConfig


def generate_true_trajectories(n_steps: int = 20) -> List[List[jnp.ndarray]]:
    """Generate ground truth trajectories for multiple targets."""
    trajectories = []

    # Target 1: Moving diagonally
    traj1 = []
    for t in range(n_steps):
        x = 0.0 + t * 0.5
        y = 0.0 + t * 0.3
        traj1.append(jnp.array([x, y]))
    trajectories.append(traj1)

    # Target 2: Moving in opposite direction (creates crossing scenario)
    traj2 = []
    for t in range(n_steps):
        x = 10.0 - t * 0.4
        y = 2.0 + t * 0.2
        traj2.append(jnp.array([x, y]))
    trajectories.append(traj2)

    # Target 3: Starts later (track initialization challenge)
    traj3 = []
    for t in range(n_steps):
        if t < 5:
            traj3.append(None)  # No target present
        else:
            x = 5.0 + (t - 5) * 0.1
            y = 8.0 - (t - 5) * 0.3
            traj3.append(jnp.array([x, y]))
    trajectories.append(traj3)

    return trajectories


def add_measurement_noise(
    true_positions: List[jnp.ndarray], noise_std: float = 0.1, key: jnp.ndarray = None
) -> List[jnp.ndarray]:
    """Add Gaussian noise to true positions."""
    if key is None:
        key = random.PRNGKey(42)

    noisy_measurements = []
    for i, pos in enumerate(true_positions):
        if pos is not None:
            key, subkey = random.split(key)
            noise = random.normal(subkey, shape=pos.shape) * noise_std
            noisy_measurements.append(pos + noise)
        else:
            noisy_measurements.append(None)

    return noisy_measurements


def add_false_alarms(
    measurements: List[jnp.ndarray],
    false_alarm_rate: float = 0.2,
    area_bounds: Tuple[float, float, float, float] = (0, 12, 0, 10),
    key: jnp.ndarray = None,
) -> List[jnp.ndarray]:
    """Add false alarm measurements."""
    if key is None:
        key = random.PRNGKey(123)

    # Determine number of false alarms
    key, subkey = random.split(key)
    n_false_alarms = random.poisson(subkey, false_alarm_rate)

    # Generate false alarm positions
    x_min, x_max, y_min, y_max = area_bounds
    false_alarms = []

    for _ in range(int(n_false_alarms)):
        key, subkey = random.split(key)
        x = random.uniform(subkey, minval=x_min, maxval=x_max)
        key, subkey = random.split(key)
        y = random.uniform(subkey, minval=y_min, maxval=y_max)
        false_alarms.append(jnp.array([x, y]))

    # Combine with real measurements
    all_measurements = [m for m in measurements if m is not None] + false_alarms
    return all_measurements


def simulate_missed_detections(
    measurements: List[jnp.ndarray],
    detection_probability: float = 0.9,
    key: jnp.ndarray = None,
) -> List[jnp.ndarray]:
    """Simulate missed detections."""
    if key is None:
        key = random.PRNGKey(456)

    detected_measurements = []
    for measurement in measurements:
        if measurement is not None:
            key, subkey = random.split(key)
            if random.uniform(subkey) < detection_probability:
                detected_measurements.append(measurement)

    return detected_measurements


def compare_algorithms():
    """Compare MHT with other data association algorithms."""
    print("=== Comparing Data Association Algorithms ===\n")

    # Generate challenging scenario
    true_trajectories = generate_true_trajectories(15)

    algorithms = [
        ("Nearest Neighbor", "nearest_neighbor"),
        ("Global Nearest Neighbor", "global_nearest_neighbor"),
        ("Multi-Hypothesis Tracking", "mht"),
    ]

    results = {}

    for alg_name, alg_type in algorithms:
        print(f"Testing {alg_name}...")

        # Configure tracker
        config = TrackerConfig(
            motion_model_class=ConstantVelocity,
            data_association_algorithm=alg_type,
            gate_threshold=2.0,
            confirmation_threshold=2,
            deletion_threshold=4,
            max_tracks=10,
        )

        tracker = MultiTrackTracker(config)

        # Simulate tracking
        key = random.PRNGKey(789)
        for t in range(15):
            # Get true positions at time t
            true_positions = [
                traj[t] if t < len(traj) else None for traj in true_trajectories
            ]

            # Add noise
            key, subkey = random.split(key)
            noisy_measurements = add_measurement_noise(true_positions, 0.15, subkey)

            # Add false alarms
            key, subkey = random.split(key)
            measurements_with_fa = add_false_alarms(noisy_measurements, 0.3, key=subkey)

            # Simulate missed detections
            key, subkey = random.split(key)
            final_measurements = simulate_missed_detections(
                measurements_with_fa, 0.85, subkey
            )

            # Convert to tracker format
            tracker_measurements = [
                (meas, PositionMeasurement, jnp.eye(2) * 0.1)
                for meas in final_measurements
            ]

            # Update tracker
            tracker.update(tracker_measurements, dt=1.0)

        # Collect results
        stats = tracker.get_statistics()
        track_states = tracker.get_confirmed_track_states()

        results[alg_name] = {
            "total_tracks": stats["total_tracks"],
            "confirmed_tracks": stats["confirmed_tracks"],
            "final_confirmed": len(track_states),
        }

        print(f"  Total tracks created: {stats['total_tracks']}")
        print(f"  Confirmed tracks: {stats['confirmed_tracks']}")
        print(f"  Final confirmed tracks: {len(track_states)}")
        print()

    # Summary
    print("=== Algorithm Comparison Summary ===")
    print("Expected: 3 true targets")
    for alg_name, result in results.items():
        print(f"{alg_name:25}: {result['final_confirmed']} confirmed tracks")

    return results


def demonstrate_mht_hypotheses():
    """Demonstrate MHT hypothesis management."""
    print("=== Demonstrating MHT Hypothesis Management ===\n")

    # Configure MHT tracker
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        data_association_algorithm="mht",
        gate_threshold=3.0,
        confirmation_threshold=2,
        deletion_threshold=3,
        max_tracks=5,
    )

    tracker = MultiTrackTracker(config)

    # Get access to MHT algorithm
    mht = tracker.data_associator

    print("Initial state:")
    print(f"  Hypotheses: {mht.get_hypothesis_count()}")

    # Ambiguous scenario: two measurements, could be one or two targets
    measurements_t1 = [
        (jnp.array([1.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([1.2, 1.1]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    tracker.update(measurements_t1, dt=1.0)
    print(f"\nAfter scan 1 (ambiguous measurements):")
    print(f"  Hypotheses: {mht.get_hypothesis_count()}")
    print(f"  Tracks created: {tracker.get_statistics()['total_tracks']}")

    # Next scan: measurements separate more
    measurements_t2 = [
        (jnp.array([1.5, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([2.0, 1.5]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    tracker.update(measurements_t2, dt=1.0)
    print(f"\nAfter scan 2 (measurements separate):")
    print(f"  Hypotheses: {mht.get_hypothesis_count()}")
    print(f"  Confirmed tracks: {tracker.get_statistics()['confirmed_tracks']}")

    # Third scan: clear separation
    measurements_t3 = [
        (jnp.array([2.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([3.0, 2.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    tracker.update(measurements_t3, dt=1.0)
    print(f"\nAfter scan 3 (clear separation):")
    print(f"  Hypotheses: {mht.get_hypothesis_count()}")
    print(f"  Confirmed tracks: {tracker.get_statistics()['confirmed_tracks']}")

    # Show best hypothesis
    best_hypothesis = mht.get_best_hypothesis()
    if best_hypothesis:
        print(f"\nBest hypothesis likelihood: {best_hypothesis.log_likelihood:.2f}")
        print(f"Track assignments: {best_hypothesis.track_assignments}")
        print(f"New tracks: {best_hypothesis.new_tracks}")
        print(f"False alarms: {best_hypothesis.false_alarms}")


def main():
    """Run MHT demonstration."""
    print("Multi-Hypothesis Tracking (MHT) Demonstration")
    print("=" * 50)
    print()

    # Compare algorithms
    compare_algorithms()
    print()

    # Demonstrate hypothesis management
    demonstrate_mht_hypotheses()

    print("\n" + "=" * 50)
    print("MHT demonstration completed!")
    print("\nKey advantages of MHT:")
    print("- Handles ambiguous data association scenarios")
    print("- Maintains multiple hypotheses until uncertainty resolves")
    print("- Better performance in cluttered environments")
    print("- Principled handling of false alarms and missed detections")


if __name__ == "__main__":
    main()
