#!/usr/bin/env python3
"""
Basic tests for the multi-track tracker.
"""

import jax.numpy as jnp
import jax.random
from typing import List

from multi_track_tracker import MultiTrackTracker, TrackerConfig
from motion_models.constant_velocity_model import ConstantVelocity
from measurement_models.radar_measurement_model import RadarMeasurement
from data_association.nearest_neighbor import NearestNeighbor, GlobalNearestNeighbor
from track_management import TrackManager
from measurement_preprocessing import MeasurementPreprocessor, SensorInfo


def test_tracker_initialization():
    """Test tracker initialization."""
    print("Testing tracker initialization...")
    
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        data_association_algorithm="nearest_neighbor"
    )
    
    tracker = MultiTrackTracker(config)
    
    assert tracker.current_time == 0.0
    assert len(tracker.track_manager.tracks) == 0
    assert isinstance(tracker.data_associator, NearestNeighbor)
    
    print("✓ Tracker initialization test passed")


def test_track_initialization():
    """Test track initialization from measurements."""
    print("Testing track initialization...")
    
    config = TrackerConfig(motion_model_class=ConstantVelocity)
    tracker = MultiTrackTracker(config)
    
    # Create a simple radar measurement
    measurement = jnp.array([100.0, 5.0, 0.5])  # range, range_rate, azimuth
    measurements = [(measurement, RadarMeasurement, jnp.eye(3))]
    
    # Update tracker
    result = tracker.update(measurements, dt=1.0)
    
    assert len(result['new_tracks']) == 1
    assert result['track_summary']['total'] == 1
    assert result['track_summary']['tentative'] == 1
    assert result['track_summary']['confirmed'] == 0
    
    print("✓ Track initialization test passed")


def test_track_confirmation():
    """Test track confirmation after multiple detections."""
    print("Testing track confirmation...")
    
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        confirmation_threshold=3
    )
    tracker = MultiTrackTracker(config)
    
    # Simulate consistent detections
    for step in range(5):
        # Moving target
        range_val = 100.0 + step * 5.0
        measurement = jnp.array([range_val, 5.0, 0.5])
        measurements = [(measurement, RadarMeasurement, jnp.eye(3))]
        
        result = tracker.update(measurements, dt=1.0)
    
    # Should have one confirmed track
    stats = tracker.get_statistics()
    assert stats['confirmed_tracks'] >= 1
    
    print("✓ Track confirmation test passed")


def test_track_deletion():
    """Test track deletion after missed detections."""
    print("Testing track deletion...")
    
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        confirmation_threshold=2,
        deletion_threshold=3
    )
    tracker = MultiTrackTracker(config)
    
    # Initialize track with detections
    for step in range(3):
        measurement = jnp.array([100.0 + step * 5.0, 5.0, 0.5])
        measurements = [(measurement, RadarMeasurement, jnp.eye(3))]
        tracker.update(measurements, dt=1.0)
    
    # Now miss several detections
    deleted_any = False
    for step in range(5):
        result = tracker.update([], dt=1.0)  # No measurements
        if result.get('deleted_tracks', []):
            deleted_any = True

    # Track should be deleted
    stats = tracker.get_statistics()
    assert deleted_any or stats['total_tracks'] == 0  # Either we saw deletion or no tracks remain
    
    print("✓ Track deletion test passed")


def test_data_association():
    """Test data association algorithms."""
    print("Testing data association...")
    
    # Test with multiple measurements and tracks
    track_states = [
        jnp.array([100.0, 50.0, 5.0, 3.0]),
        jnp.array([200.0, 100.0, -2.0, 4.0])
    ]
    
    measurements = [
        (jnp.array([110.0, 5.0, 0.5]), RadarMeasurement, jnp.eye(3)),
        (jnp.array([205.0, -2.0, 0.8]), RadarMeasurement, jnp.eye(3))
    ]
    
    # Test Nearest Neighbor
    nn_associator = NearestNeighbor(gate_threshold=50.0)
    associations = nn_associator.associate(track_states, measurements)
    
    assert len(associations) <= len(track_states)
    assert len(associations) <= len(measurements)
    
    # Test Global Nearest Neighbor
    gnn_associator = GlobalNearestNeighbor(gate_threshold=50.0)
    associations = gnn_associator.associate(track_states, measurements)
    
    assert len(associations) <= len(track_states)
    assert len(associations) <= len(measurements)
    
    print("✓ Data association test passed")


def test_multi_target_scenario():
    """Test tracking multiple targets simultaneously."""
    print("Testing multi-target scenario...")
    
    config = TrackerConfig(
        motion_model_class=ConstantVelocity,
        confirmation_threshold=2,
        deletion_threshold=4
    )
    tracker = MultiTrackTracker(config)
    
    # Simulate two targets moving in different directions
    for step in range(10):
        measurements = [
            # Target 1: moving northeast
            (jnp.array([100.0 + step * 5.0, 5.0, jnp.arctan2(3.0, 5.0)]), 
             RadarMeasurement, jnp.eye(3)),
            # Target 2: moving southwest  
            (jnp.array([200.0 - step * 3.0, -3.0, jnp.arctan2(-4.0, -3.0)]), 
             RadarMeasurement, jnp.eye(3))
        ]
        
        result = tracker.update(measurements, dt=1.0)
    
    # Should have two confirmed tracks
    stats = tracker.get_statistics()
    assert stats['confirmed_tracks'] >= 2
    
    confirmed_tracks = tracker.get_confirmed_track_states()
    assert len(confirmed_tracks) >= 2
    
    print("✓ Multi-target scenario test passed")


def test_measurement_preprocessing():
    """Test measurement preprocessing."""
    print("Testing measurement preprocessing...")
    
    # Create sensor configuration
    sensors = {
        'radar_1': SensorInfo(
            sensor_id='radar_1',
            sensor_type='radar',
            position=jnp.array([0.0, 0.0, 0.0]),
            orientation=jnp.array([0.0, 0.0, 0.0]),
            measurement_model=RadarMeasurement,
            default_noise_covariance=jnp.diag(jnp.array([1.0, 0.5, 0.01]))
        )
    }
    
    preprocessor = MeasurementPreprocessor(sensors)
    
    # Test valid measurement
    raw_measurements = [
        {
            'sensor_id': 'radar_1',
            'data': [100.0, 5.0, 0.5],
            'timestamp': 1.0
        }
    ]
    
    processed = preprocessor.preprocess_measurements(raw_measurements, 1.0)
    assert len(processed) == 1
    
    measurement, model, covariance = processed[0]
    assert measurement.shape == (3,)
    assert model == RadarMeasurement
    assert covariance.shape == (3, 3)
    
    # Test invalid measurement (should be filtered out)
    raw_measurements = [
        {
            'sensor_id': 'radar_1',
            'data': [-100.0, 5.0, 0.5],  # Negative range
            'timestamp': 1.0
        }
    ]
    
    processed = preprocessor.preprocess_measurements(raw_measurements, 1.0)
    assert len(processed) == 0  # Should be filtered out
    
    print("✓ Measurement preprocessing test passed")


def test_tracker_reset():
    """Test tracker reset functionality."""
    print("Testing tracker reset...")
    
    config = TrackerConfig(motion_model_class=ConstantVelocity)
    tracker = MultiTrackTracker(config)
    
    # Add some tracks
    measurement = jnp.array([100.0, 5.0, 0.5])
    measurements = [(measurement, RadarMeasurement, jnp.eye(3))]
    
    for _ in range(3):
        tracker.update(measurements, dt=1.0)
    
    # Verify tracks exist
    stats_before = tracker.get_statistics()
    assert stats_before['total_tracks'] > 0
    
    # Reset tracker
    tracker.reset()
    
    # Verify reset
    stats_after = tracker.get_statistics()
    assert stats_after['total_tracks'] == 0
    assert tracker.current_time == 0.0
    
    print("✓ Tracker reset test passed")


def run_all_tests():
    """Run all tests."""
    print("=== Running Multi-Track Tracker Tests ===\n")
    
    try:
        test_tracker_initialization()
        test_track_initialization()
        test_track_confirmation()
        test_track_deletion()
        test_data_association()
        test_multi_target_scenario()
        test_measurement_preprocessing()
        test_tracker_reset()
        
        print("\n=== All Tests Passed! ===")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
