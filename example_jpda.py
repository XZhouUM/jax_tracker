#!/usr/bin/env python3
"""
Example demonstrating Joint Probabilistic Data Association (JPDA) in action.

This example shows how JPDA handles uncertain data association scenarios by
calculating association probabilities rather than making hard decisions.
"""

import jax.numpy as jnp
import jax.random as random
from typing import List, Tuple
import numpy as np

from multi_track_tracker import MultiTrackTracker, TrackerConfig
from motion_models.constant_velocity_model import ConstantVelocity
from measurement_models.position_measurement_model import PositionMeasurement
from data_association.joint_probabilistic_data_association import JointProbabilisticDataAssociation


def demonstrate_jpda_probabilities():
    """Demonstrate JPDA association probability calculations."""
    print("=== JPDA Association Probability Demonstration ===\n")
    
    jpda = JointProbabilisticDataAssociation(
        gate_threshold=5.0,
        prob_detection=0.9,
        prob_false_alarm=0.1,
        clutter_density=1e-4,
    )
    
    # Create ambiguous scenario: two tracks, two measurements
    print("Scenario: Two tracks with ambiguous measurements")
    track_states = [
        jnp.array([1.0, 1.0, 0.1, 0.0]),  # Track 0: moving right
        jnp.array([2.0, 2.0, -0.1, 0.0]),  # Track 1: moving left
    ]
    
    measurements = [
        (jnp.array([1.3, 1.2]), PositionMeasurement, jnp.eye(2) * 0.1),  # Ambiguous
        (jnp.array([1.7, 1.8]), PositionMeasurement, jnp.eye(2) * 0.1),  # Ambiguous
    ]
    
    print("Track states:")
    for i, state in enumerate(track_states):
        print(f"  Track {i}: position=({state[0]:.1f}, {state[1]:.1f}), velocity=({state[2]:.1f}, {state[3]:.1f})")
    
    print("\nMeasurements:")
    for i, (meas, _, _) in enumerate(measurements):
        print(f"  Measurement {i}: ({meas[0]:.1f}, {meas[1]:.1f})")
    
    # Calculate associations
    result = jpda.associate(track_states, measurements)
    probs = jpda.get_association_probabilities()
    
    print(f"\nHard associations (for compatibility): {result}")
    
    if probs is not None:
        print("\nAssociation Probability Matrix:")
        print("(Rows = Tracks, Columns = Measurements)")
        prob_matrix = np.array(probs.prob_matrix)
        for i in range(prob_matrix.shape[0]):
            row_str = f"Track {i}: "
            for j in range(prob_matrix.shape[1]):
                row_str += f"M{j}={prob_matrix[i,j]:.3f} "
            print(f"  {row_str}")
        
        print("\nProbability of non-detection:")
        prob_undetected = np.array(probs.prob_undetected)
        for i in range(len(prob_undetected)):
            print(f"  Track {i}: {prob_undetected[i]:.3f}")
        
        print("\nProbability of false alarm:")
        prob_false_alarm = np.array(probs.prob_false_alarm)
        for i in range(len(prob_false_alarm)):
            print(f"  Measurement {i}: {prob_false_alarm[i]:.3f}")
        
        # Verify probability constraints
        print("\nProbability Verification:")
        for i in range(len(track_states)):
            track_sum = np.sum(prob_matrix[i, :]) + prob_undetected[i]
            print(f"  Track {i} total probability: {track_sum:.6f} (should be 1.0)")
        
        for j in range(len(measurements)):
            meas_sum = np.sum(prob_matrix[:, j]) + prob_false_alarm[j]
            print(f"  Measurement {j} total probability: {meas_sum:.6f} (should be 1.0)")


def compare_algorithms_detailed():
    """Compare JPDA with other algorithms in detail."""
    print("\n=== Detailed Algorithm Comparison ===\n")
    
    algorithms = [
        ("Nearest Neighbor", "nearest_neighbor"),
        ("Global Nearest Neighbor", "global_nearest_neighbor"),
        ("Multi-Hypothesis Tracking", "mht"),
        ("Joint Probabilistic Data Association", "jpda"),
    ]
    
    # Create challenging scenario: closely spaced targets with false alarms
    print("Scenario: Closely spaced targets with false alarms")
    
    results = {}
    
    for alg_name, alg_type in algorithms:
        print(f"\nTesting {alg_name}...")
        
        config = TrackerConfig(
            motion_model_class=ConstantVelocity,
            data_association_algorithm=alg_type,
            gate_threshold=3.0,
            confirmation_threshold=2,
            deletion_threshold=4,
            max_tracks=10,
        )
        
        tracker = MultiTrackTracker(config)
        
        # Simulate 10 time steps
        key = random.PRNGKey(42)
        for t in range(10):
            # Two targets moving in parallel
            target1_pos = jnp.array([t * 0.5, 1.0 + t * 0.1])
            target2_pos = jnp.array([t * 0.5, 2.0 + t * 0.1])
            
            # Add noise to measurements
            key, subkey1, subkey2 = random.split(key, 3)
            noise1 = random.normal(subkey1, shape=(2,)) * 0.1
            noise2 = random.normal(subkey2, shape=(2,)) * 0.1
            
            measurements = [
                (target1_pos + noise1, PositionMeasurement, jnp.eye(2) * 0.1),
                (target2_pos + noise2, PositionMeasurement, jnp.eye(2) * 0.1),
            ]
            
            # Add false alarm occasionally
            if t % 3 == 0:
                key, subkey = random.split(key)
                false_alarm_pos = random.uniform(subkey, shape=(2,), minval=-1, maxval=10)
                measurements.append(
                    (false_alarm_pos, PositionMeasurement, jnp.eye(2) * 0.1)
                )
            
            tracker.update(measurements, dt=1.0)
        
        # Collect results
        stats = tracker.get_statistics()
        track_states = tracker.get_confirmed_track_states()
        
        results[alg_name] = {
            'total_tracks': stats['total_tracks'],
            'confirmed_tracks': stats['confirmed_tracks'],
            'final_tracks': len(track_states),
        }
        
        print(f"  Total tracks created: {stats['total_tracks']}")
        print(f"  Confirmed tracks: {stats['confirmed_tracks']}")
        print(f"  Final confirmed tracks: {len(track_states)}")
        
        # JPDA-specific information
        if alg_type == "jpda":
            jpda = tracker.data_associator
            probs = jpda.get_association_probabilities()
            if probs is not None:
                print(f"  Last association probabilities shape: {probs.prob_matrix.shape}")
                avg_uncertainty = np.mean(np.max(probs.prob_matrix, axis=1))
                print(f"  Average association confidence: {avg_uncertainty:.3f}")
    
    # Summary
    print(f"\n=== Algorithm Comparison Summary ===")
    print("Expected: 2 true targets")
    for alg_name, result in results.items():
        print(f"{alg_name:30}: {result['final_tracks']} final tracks")


def demonstrate_jpda_tracking():
    """Demonstrate JPDA tracking over time."""
    print("\n=== JPDA Tracking Demonstration ===\n")
    
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        data_association_algorithm="jpda",
        gate_threshold=4.0,
        confirmation_threshold=2,
        deletion_threshold=3,
    )
    
    tracker = MultiTrackTracker(config)
    jpda = tracker.data_associator
    
    print("Tracking scenario: Two crossing targets")
    
    # Simulate crossing targets
    for t in range(8):
        # Target 1: moving diagonally up-right
        target1_x = t * 0.5
        target1_y = t * 0.3
        
        # Target 2: moving diagonally down-right (crossing path)
        target2_x = t * 0.4
        target2_y = 5.0 - t * 0.4
        
        measurements = [
            (jnp.array([target1_x, target1_y]), PositionMeasurement, jnp.eye(2) * 0.1),
            (jnp.array([target2_x, target2_y]), PositionMeasurement, jnp.eye(2) * 0.1),
        ]
        
        # Add some measurement noise
        key = random.PRNGKey(t)
        for i in range(len(measurements)):
            key, subkey = random.split(key)
            noise = random.normal(subkey, shape=(2,)) * 0.15
            meas, model, R = measurements[i]
            measurements[i] = (meas + noise, model, R)
        
        tracker.update(measurements, dt=1.0)
        
        # Show JPDA probabilities
        probs = jpda.get_association_probabilities()
        stats = tracker.get_statistics()
        
        print(f"Time {t}:")
        print(f"  Active tracks: {stats['confirmed_tracks']}")
        
        if probs is not None and probs.prob_matrix.size > 0:
            prob_matrix = np.array(probs.prob_matrix)
            print(f"  Association probabilities shape: {prob_matrix.shape}")
            
            # Show highest probability associations
            if prob_matrix.shape[0] > 0 and prob_matrix.shape[1] > 0:
                for track_idx in range(prob_matrix.shape[0]):
                    best_meas = np.argmax(prob_matrix[track_idx, :])
                    best_prob = prob_matrix[track_idx, best_meas]
                    print(f"    Track {track_idx} -> Measurement {best_meas} (prob={best_prob:.3f})")
        
        print()
    
    final_stats = tracker.get_statistics()
    print(f"Final results:")
    print(f"  Total tracks created: {final_stats['total_tracks']}")
    print(f"  Confirmed tracks: {final_stats['confirmed_tracks']}")


def main():
    """Run JPDA demonstration."""
    print("Joint Probabilistic Data Association (JPDA) Demonstration")
    print("=" * 60)
    
    demonstrate_jpda_probabilities()
    compare_algorithms_detailed()
    demonstrate_jpda_tracking()
    
    print("\n" + "=" * 60)
    print("JPDA demonstration completed!")
    print("\nKey advantages of JPDA:")
    print("- Calculates association probabilities instead of hard decisions")
    print("- Handles uncertainty in data association gracefully")
    print("- Updates tracks with weighted combinations of measurements")
    print("- Provides probabilistic framework for association decisions")
    print("- Better performance in ambiguous scenarios than hard assignment methods")


if __name__ == "__main__":
    main()
