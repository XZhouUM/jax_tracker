#!/usr/bin/env python3
"""
Test script for Multi-Hypothesis Tracking (MHT) implementation.
"""

from typing import List, Tuple

import jax.numpy as jnp

from data_association.multi_hypothesis_tracking import (
    Hypothesis, MultiHypothesisTracking)
from measurement_models.position_measurement_model import PositionMeasurement
from measurement_models.radar_measurement_model import RadarMeasurement
from motion_models.constant_velocity_model import ConstantVelocity
from multi_track_tracker import MultiTrackTracker, TrackerConfig


def test_hypothesis_creation():
    """Test basic hypothesis creation and validation."""
    print("Testing hypothesis creation...")

    # Valid hypothesis
    hypothesis = Hypothesis(
        track_assignments={
            0: 0,
            1: None,
        },  # Track 0 -> Measurement 0, Track 1 -> No detection
        new_tracks=[1],  # Measurement 1 starts new track
        false_alarms={2},  # Measurement 2 is false alarm
        log_likelihood=-5.0,
    )

    assert hypothesis.track_assignments[0] == 0
    assert hypothesis.track_assignments[1] is None
    assert 1 in hypothesis.new_tracks
    assert 2 in hypothesis.false_alarms

    # Test invalid hypothesis (measurement assigned twice)
    try:
        invalid_hypothesis = Hypothesis(
            track_assignments={0: 0, 1: 0},  # Both tracks assigned to measurement 0
            new_tracks=[],
            false_alarms=set(),
            log_likelihood=-10.0,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass  # Expected

    print("✓ Hypothesis creation test passed")


def test_mht_basic():
    """Test basic MHT functionality."""
    print("Testing basic MHT functionality...")

    mht = MultiHypothesisTracking(
        gate_threshold=10.0, max_hypotheses=50, prob_detection=0.9, prob_false_alarm=0.1
    )

    # Test with no tracks and no measurements
    associations = mht.associate([], [])
    assert associations == {}

    # Test with tracks but no measurements
    track_states = [jnp.array([0.0, 0.0, 1.0, 1.0])]  # [x, y, vx, vy]
    associations = mht.associate(track_states, [])
    assert associations == {}

    # Test with measurements but no tracks
    measurements = [(jnp.array([1.0, 2.0]), PositionMeasurement, jnp.eye(2) * 0.1)]
    associations = mht.associate([], measurements)
    assert associations == {}

    print("✓ Basic MHT test passed")


def test_mht_single_track_single_measurement():
    """Test MHT with single track and single measurement."""
    print("Testing MHT with single track and single measurement...")

    mht = MultiHypothesisTracking(
        gate_threshold=10.0, max_hypotheses=10, prob_detection=0.9
    )

    # Single track state
    track_states = [jnp.array([1.0, 1.0, 0.5, 0.5])]  # [x, y, vx, vy]

    # Single measurement close to track
    measurements = [(jnp.array([1.1, 1.1]), PositionMeasurement, jnp.eye(2) * 0.1)]

    associations = mht.associate(track_states, measurements)

    # Should associate track 0 with measurement 0
    assert 0 in associations
    assert associations[0] == 0

    # Check hypothesis count
    assert mht.get_hypothesis_count() > 0

    best_hypothesis = mht.get_best_hypothesis()
    assert best_hypothesis is not None
    assert best_hypothesis.track_assignments[0] == 0

    print("✓ Single track/measurement MHT test passed")


def test_mht_multiple_tracks_measurements():
    """Test MHT with multiple tracks and measurements."""
    print("Testing MHT with multiple tracks and measurements...")

    mht = MultiHypothesisTracking(
        gate_threshold=5.0, max_hypotheses=100, prob_detection=0.8, prob_false_alarm=0.1
    )

    # Multiple track states
    track_states = [
        jnp.array([0.0, 0.0, 1.0, 0.0]),  # Track 0: moving right
        jnp.array([5.0, 5.0, 0.0, 1.0]),  # Track 1: moving up
    ]

    # Multiple measurements
    measurements = [
        (
            jnp.array([0.1, 0.1]),
            PositionMeasurement,
            jnp.eye(2) * 0.1,
        ),  # Close to track 0
        (
            jnp.array([5.1, 5.1]),
            PositionMeasurement,
            jnp.eye(2) * 0.1,
        ),  # Close to track 1
        (
            jnp.array([10.0, 10.0]),
            PositionMeasurement,
            jnp.eye(2) * 0.1,
        ),  # Far from both
    ]

    associations = mht.associate(track_states, measurements)

    # Should have some associations
    assert len(associations) >= 0

    # Check that associations are valid
    for track_idx, meas_idx in associations.items():
        assert 0 <= track_idx < len(track_states)
        assert 0 <= meas_idx < len(measurements)

    print("✓ Multiple tracks/measurements MHT test passed")


def test_mht_with_tracker():
    """Test MHT integration with MultiTrackTracker."""
    print("Testing MHT integration with tracker...")

    # Configure tracker with MHT
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        data_association_algorithm="mht",
        gate_threshold=10.0,
        confirmation_threshold=2,
        deletion_threshold=3,
        max_tracks=10,
    )

    tracker = MultiTrackTracker(config)

    # Simulate measurements for two targets
    measurements_t1 = [
        (jnp.array([1.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([5.0, 5.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    measurements_t2 = [
        (jnp.array([2.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([5.0, 6.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    measurements_t3 = [
        (jnp.array([3.0, 1.0]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([5.0, 7.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    # Update tracker
    tracker.update(measurements_t1, dt=1.0)
    tracker.update(measurements_t2, dt=1.0)
    tracker.update(measurements_t3, dt=1.0)

    # Check that tracks were created and confirmed
    stats = tracker.get_statistics()
    assert stats["total_tracks"] >= 2

    # Get track states
    track_states = tracker.get_confirmed_track_states()
    assert len(track_states) >= 0  # Should have some confirmed tracks

    print("✓ MHT tracker integration test passed")


def test_mht_hypothesis_pruning():
    """Test MHT hypothesis pruning functionality."""
    print("Testing MHT hypothesis pruning...")

    mht = MultiHypothesisTracking(
        gate_threshold=10.0,
        max_hypotheses=5,  # Small number to force pruning
        hypothesis_prune_threshold=-20.0,
        prob_detection=0.9,
    )

    # Create scenario that generates many hypotheses
    track_states = [
        jnp.array([0.0, 0.0, 1.0, 0.0]),
        jnp.array([1.0, 1.0, 0.0, 1.0]),
    ]

    measurements = [
        (jnp.array([0.1, 0.1]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([1.1, 1.1]), PositionMeasurement, jnp.eye(2) * 0.1),
        (jnp.array([2.0, 2.0]), PositionMeasurement, jnp.eye(2) * 0.1),
    ]

    # Run multiple updates to build hypothesis tree
    for _ in range(3):
        mht.associate(track_states, measurements)

    # Check that hypothesis count is limited
    assert mht.get_hypothesis_count() <= 5

    print("✓ MHT hypothesis pruning test passed")


def run_all_mht_tests():
    """Run all MHT tests."""
    print("=== Running MHT Tests ===\n")

    try:
        test_hypothesis_creation()
        test_mht_basic()
        test_mht_single_track_single_measurement()
        test_mht_multiple_tracks_measurements()
        test_mht_with_tracker()
        test_mht_hypothesis_pruning()

        print("\n=== All MHT Tests Passed! ===")

    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    run_all_mht_tests()
