#!/usr/bin/env python3
"""
Test suite for Joint Probabilistic Data Association (JPDA) algorithm.
"""

import jax.numpy as jnp
import jax.random as random
from typing import List, Tuple

from data_association.joint_probabilistic_data_association import (
    JointProbabilisticDataAssociation,
    AssociationProbabilities,
)
from measurement_models.position_measurement_model import PositionMeasurement
from multi_track_tracker import MultiTrackTracker, TrackerConfig
from motion_models.constant_velocity_model import ConstantVelocity


def test_jpda_creation():
    """Test JPDA algorithm creation and basic properties."""
    print("Testing JPDA creation...")
    
    jpda = JointProbabilisticDataAssociation(
        gate_threshold=5.0,
        use_ellipsoidal_gate=True,
        prob_detection=0.9,
        prob_false_alarm=0.1,
    )
    
    assert jpda.gate_threshold == 5.0
    assert jpda.use_ellipsoidal_gate == True
    assert jpda.prob_detection == 0.9
    assert jpda.prob_false_alarm == 0.1
    assert jpda.last_association_probabilities is None
    
    print("✓ JPDA creation test passed")


def test_jpda_basic():
    """Test basic JPDA functionality with edge cases."""
    print("Testing basic JPDA functionality...")
    
    jpda = JointProbabilisticDataAssociation()
    
    # Test empty inputs
    result = jpda.associate([], [])
    assert result == {}
    assert jpda.get_association_probabilities() is None
    
    # Test empty tracks
    measurements = [
        (jnp.array([1.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]
    result = jpda.associate([], measurements)
    assert result == {}
    
    # Test empty measurements
    track_states = [jnp.array([1.0, 1.0, 0.1, 0.1])]
    result = jpda.associate(track_states, [])
    assert result == {}
    
    print("✓ Basic JPDA test passed")


def test_jpda_single_track_single_measurement():
    """Test JPDA with single track and single measurement."""
    print("Testing JPDA with single track and single measurement...")
    
    jpda = JointProbabilisticDataAssociation(
        gate_threshold=10.0,
        prob_detection=0.9,
        prob_false_alarm=0.1,
    )
    
    # Single track state [x, y, vx, vy]
    track_states = [jnp.array([1.0, 1.0, 0.1, 0.1])]
    
    # Single measurement close to track
    measurements = [
        (jnp.array([1.1, 1.1]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]
    
    result = jpda.associate(track_states, measurements)
    
    # Should associate track 0 with measurement 0
    assert 0 in result
    assert result[0] == 0
    
    # Check association probabilities
    probs = jpda.get_association_probabilities()
    assert probs is not None
    assert probs.prob_matrix.shape == (1, 1)
    assert probs.prob_matrix[0, 0] > 0.5  # High probability of association
    assert probs.prob_undetected[0] < 0.5  # Low probability of non-detection
    
    print("✓ Single track/measurement JPDA test passed")


def test_jpda_multiple_tracks_measurements():
    """Test JPDA with multiple tracks and measurements."""
    print("Testing JPDA with multiple tracks and measurements...")
    
    jpda = JointProbabilisticDataAssociation(
        gate_threshold=5.0,
        prob_detection=0.8,
        prob_false_alarm=0.2,
    )
    
    # Two tracks
    track_states = [
        jnp.array([1.0, 1.0, 0.1, 0.1]),  # Track 0
        jnp.array([5.0, 5.0, -0.1, -0.1]),  # Track 1
    ]
    
    # Three measurements - two close to tracks, one false alarm
    measurements = [
        (jnp.array([1.1, 1.1]), PositionMeasurement, jnp.eye(2) * 0.1),  # Close to track 0
        (jnp.array([5.1, 5.1]), PositionMeasurement, jnp.eye(2) * 0.1),  # Close to track 1
        (jnp.array([10.0, 10.0]), PositionMeasurement, jnp.eye(2) * 0.1),  # False alarm
    ]
    
    result = jpda.associate(track_states, measurements)
    
    # Check that we get reasonable associations
    assert len(result) <= 2  # At most 2 tracks can be associated
    
    # Check association probabilities
    probs = jpda.get_association_probabilities()
    assert probs is not None
    assert probs.prob_matrix.shape == (2, 3)
    
    # Track 0 should have high probability with measurement 0
    assert probs.prob_matrix[0, 0] > probs.prob_matrix[0, 1]
    assert probs.prob_matrix[0, 0] > probs.prob_matrix[0, 2]
    
    # Track 1 should have high probability with measurement 1
    assert probs.prob_matrix[1, 1] > probs.prob_matrix[1, 0]
    assert probs.prob_matrix[1, 1] > probs.prob_matrix[1, 2]
    
    # Measurement 2 should have high false alarm probability
    assert probs.prob_false_alarm[2] > 0.5
    
    print("✓ Multiple tracks/measurements JPDA test passed")


def test_jpda_with_tracker():
    """Test JPDA integration with MultiTrackTracker."""
    print("Testing JPDA integration with tracker...")
    
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        data_association_algorithm="jpda",
        gate_threshold=5.0,
        confirmation_threshold=2,
        deletion_threshold=3,
    )
    
    tracker = MultiTrackTracker(config)
    
    # Verify JPDA is being used
    assert isinstance(tracker.data_associator, JointProbabilisticDataAssociation)
    
    # Test with some measurements
    measurements_t1 = [
        (jnp.array([1.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([5.0, 5.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]
    
    tracker.update(measurements_t1, dt=1.0)
    
    # Should create tracks
    stats = tracker.get_statistics()
    assert stats["total_tracks"] >= 1
    
    # Test second update
    measurements_t2 = [
        (jnp.array([1.1, 1.1]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([5.1, 5.1]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]
    
    tracker.update(measurements_t2, dt=1.0)
    
    # Check that tracks are being maintained
    stats = tracker.get_statistics()
    assert stats["total_tracks"] >= 1
    
    # Access JPDA-specific information
    jpda = tracker.data_associator
    probs = jpda.get_association_probabilities()
    if probs is not None:
        assert probs.prob_matrix.shape[0] > 0  # Should have tracks
        assert probs.prob_matrix.shape[1] > 0  # Should have measurements
    
    print("✓ JPDA tracker integration test passed")


def test_jpda_association_probabilities():
    """Test JPDA association probability calculations."""
    print("Testing JPDA association probability calculations...")
    
    jpda = JointProbabilisticDataAssociation(
        gate_threshold=10.0,
        prob_detection=0.9,
        prob_false_alarm=0.1,
        clutter_density=1e-4,
    )
    
    # Create ambiguous scenario - two tracks, two measurements
    track_states = [
        jnp.array([1.0, 1.0, 0.0, 0.0]),
        jnp.array([2.0, 2.0, 0.0, 0.0]),
    ]
    
    measurements = [
        (jnp.array([1.2, 1.2]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([1.8, 1.8]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]
    
    result = jpda.associate(track_states, measurements)
    probs = jpda.get_association_probabilities()
    
    assert probs is not None
    
    # Check probability matrix properties
    prob_matrix = probs.prob_matrix
    prob_undetected = probs.prob_undetected
    prob_false_alarm = probs.prob_false_alarm
    
    # Probabilities should sum correctly for each track
    for track_idx in range(len(track_states)):
        track_prob_sum = jnp.sum(prob_matrix[track_idx, :]) + prob_undetected[track_idx]
        assert abs(track_prob_sum - 1.0) < 1e-6, f"Track {track_idx} probabilities don't sum to 1: {track_prob_sum}"
    
    # Probabilities should sum correctly for each measurement
    for meas_idx in range(len(measurements)):
        meas_prob_sum = jnp.sum(prob_matrix[:, meas_idx]) + prob_false_alarm[meas_idx]
        assert abs(meas_prob_sum - 1.0) < 1e-6, f"Measurement {meas_idx} probabilities don't sum to 1: {meas_prob_sum}"
    
    # Check combined innovations
    assert len(probs.combined_innovations) == len(track_states)
    assert len(probs.combined_covariances) == len(track_states)
    
    for innovation in probs.combined_innovations:
        assert innovation.shape == (2,)  # Position measurement dimension
    
    for covariance in probs.combined_covariances:
        assert covariance.shape == (2, 2)  # Position measurement covariance
    
    print("✓ JPDA association probability test passed")


def test_jpda_performance_comparison():
    """Test JPDA performance compared to other algorithms."""
    print("Testing JPDA performance comparison...")
    
    # Create challenging scenario with crossing tracks
    algorithms = ["nearest_neighbor", "global_nearest_neighbor", "jpda"]
    results = {}
    
    for alg in algorithms:
        config = TrackerConfig(
            motion_model_class=ConstantVelocity,
            data_association_algorithm=alg,
            gate_threshold=3.0,
            confirmation_threshold=2,
            deletion_threshold=4,
        )
        
        tracker = MultiTrackTracker(config)
        
        # Simulate crossing tracks scenario
        for t in range(5):
            measurements = [
                (jnp.array([t * 0.5, t * 0.3]), PositionMeasurement, jnp.eye(2) * 0.1),
                (jnp.array([5 - t * 0.4, 2 + t * 0.2]), PositionMeasurement, jnp.eye(2) * 0.1),
            ]
            
            # Add some noise and false alarms occasionally
            if t % 2 == 0:
                measurements.append(
                    (jnp.array([10.0, 10.0]), PositionMeasurement, jnp.eye(2) * 0.1)
                )
            
            tracker.update(measurements, dt=1.0)
        
        stats = tracker.get_statistics()
        results[alg] = {
            'total_tracks': stats['total_tracks'],
            'confirmed_tracks': stats['confirmed_tracks'],
        }
    
    # JPDA should handle the scenario reasonably well
    jpda_result = results['jpda']
    assert jpda_result['total_tracks'] >= 2
    
    print(f"Algorithm comparison results:")
    for alg, result in results.items():
        print(f"  {alg}: {result['confirmed_tracks']} confirmed tracks")
    
    print("✓ JPDA performance comparison test passed")


def main():
    """Run all JPDA tests."""
    print("=== Running JPDA Tests ===\n")
    
    test_jpda_creation()
    test_jpda_basic()
    test_jpda_single_track_single_measurement()
    test_jpda_multiple_tracks_measurements()
    test_jpda_with_tracker()
    test_jpda_association_probabilities()
    test_jpda_performance_comparison()
    
    print("\n=== All JPDA Tests Passed! ===")


if __name__ == "__main__":
    main()
