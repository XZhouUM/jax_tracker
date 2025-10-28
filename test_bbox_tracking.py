#!/usr/bin/env python3
"""
Test suite for bounding box tracking functionality.
"""

import jax.numpy as jnp

from data_association.iou_association import (IoUNearestNeighbor,
                                              IoUOptimalAssignment)
from measurement_models.bounding_box_measurement_model import \
    BoundingBoxMeasurement
from motion_models.bounding_box_motion_model import BoundingBoxConstantVelocity
from multi_track_tracker import MultiTrackTracker, TrackerConfig


def test_bounding_box_measurement_model():
    """Test bounding box measurement model functionality."""
    print("Testing BoundingBoxMeasurement model...")

    # Test state to measurement prediction
    state = jnp.array(
        [100, 50, 40, 30, 2, 1, 0.5, 0.2]
    )  # [cx, cy, w, h, vcx, vcy, vw, vh]
    measurement = BoundingBoxMeasurement.predict_measurement(state)

    expected_measurement = jnp.array([100, 50, 40, 30])  # [cx, cy, w, h]
    assert jnp.allclose(
        measurement, expected_measurement
    ), f"Expected {expected_measurement}, got {measurement}"

    # Test Jacobian
    jacobian = BoundingBoxMeasurement.jacobian(state)
    expected_shape = (4, 8)
    assert (
        jacobian.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {jacobian.shape}"

    # Test conversion functions
    bbox = BoundingBoxMeasurement.state_to_bbox(state)
    expected_bbox = jnp.array([80, 35, 120, 65])  # [x1, y1, x2, y2]
    assert jnp.allclose(bbox, expected_bbox), f"Expected {expected_bbox}, got {bbox}"

    # Test IoU calculation
    bbox1 = jnp.array([0, 0, 10, 10])
    bbox2 = jnp.array([5, 5, 15, 15])
    iou = BoundingBoxMeasurement.calculate_iou(bbox1, bbox2)
    expected_iou = 25.0 / 175.0  # intersection=25, union=175
    assert abs(iou - expected_iou) < 1e-6, f"Expected IoU {expected_iou}, got {iou}"

    print("✓ BoundingBoxMeasurement tests passed")


def test_bounding_box_motion_model():
    """Test bounding box motion model functionality."""
    print("Testing BoundingBoxConstantVelocity model...")

    # Test state transition
    state = jnp.array(
        [100, 50, 40, 30, 2, 1, 0.5, 0.2]
    )  # [cx, cy, w, h, vcx, vcy, vw, vh]
    dt = 1.0
    next_state = BoundingBoxConstantVelocity.transition(state, dt)

    expected_next_state = jnp.array([102, 51, 40.5, 30.2, 2, 1, 0.5, 0.2])
    assert jnp.allclose(
        next_state, expected_next_state
    ), f"Expected {expected_next_state}, got {next_state}"

    # Test Jacobian
    jacobian = BoundingBoxConstantVelocity.jacobian(state, dt)
    expected_shape = (8, 8)
    assert (
        jacobian.shape == expected_shape
    ), f"Expected shape {expected_shape}, got {jacobian.shape}"

    # Test process noise matrix
    Q = BoundingBoxConstantVelocity.process_noise_matrix(dt)
    assert Q.shape == (8, 8), f"Expected shape (8, 8), got {Q.shape}"

    # Test initial state creation
    measurement = jnp.array([100, 50, 40, 30])
    initial_state = BoundingBoxConstantVelocity.create_initial_state(measurement)
    expected_initial_state = jnp.array([100, 50, 40, 30, 0, 0, 0, 0])
    assert jnp.allclose(
        initial_state, expected_initial_state
    ), f"Expected {expected_initial_state}, got {initial_state}"

    print("✓ BoundingBoxConstantVelocity tests passed")


def test_iou_association():
    """Test IoU-based data association."""
    print("Testing IoU association algorithms...")

    # Create test track states (bounding box format)
    track_states = [
        jnp.array([100, 50, 40, 30, 1, 0, 0, 0]),  # Track 1
        jnp.array([200, 100, 50, 40, -1, 1, 0, 0]),  # Track 2
    ]

    # Create test measurements
    measurements = [
        (
            jnp.array([105, 52, 42, 32]),
            BoundingBoxMeasurement,
            jnp.eye(4),
        ),  # Close to track 1
        (
            jnp.array([195, 105, 48, 38]),
            BoundingBoxMeasurement,
            jnp.eye(4),
        ),  # Close to track 2
        (
            jnp.array([300, 300, 20, 20]),
            BoundingBoxMeasurement,
            jnp.eye(4),
        ),  # Far from both
    ]

    # Test IoU Nearest Neighbor
    iou_nn = IoUNearestNeighbor(min_iou_threshold=0.1)
    associations_nn = iou_nn.associate(track_states, measurements)

    # Should associate track 0 with measurement 0, track 1 with measurement 1
    assert 0 in associations_nn, "Track 0 should be associated"
    assert 1 in associations_nn, "Track 1 should be associated"
    assert (
        associations_nn[0] == 0
    ), f"Track 0 should associate with measurement 0, got {associations_nn[0]}"
    assert (
        associations_nn[1] == 1
    ), f"Track 1 should associate with measurement 1, got {associations_nn[1]}"

    # Test IoU Optimal Assignment
    iou_opt = IoUOptimalAssignment(min_iou_threshold=0.1)
    associations_opt = iou_opt.associate(track_states, measurements)

    # Should give same result for this simple case
    assert (
        associations_opt == associations_nn
    ), f"Optimal assignment should match NN for this case"

    print("✓ IoU association tests passed")


def test_bbox_tracker_integration():
    """Test full bounding box tracker integration."""
    print("Testing bounding box tracker integration...")

    # Configure tracker for bounding box tracking
    config = TrackerConfig(
        motion_model_class=BoundingBoxConstantVelocity,
        data_association_algorithm="iou_nearest_neighbor",
        gate_threshold=0.1,  # IoU threshold
        confirmation_threshold=2,
        deletion_threshold=3,
    )

    tracker = MultiTrackTracker(config)

    # Test with sequence of measurements
    measurements_sequence = [
        # Step 1: Single measurement
        [(jnp.array([100, 50, 40, 30]), BoundingBoxMeasurement, jnp.eye(4))],
        # Step 2: Same object moved slightly
        [(jnp.array([102, 51, 41, 31]), BoundingBoxMeasurement, jnp.eye(4))],
        # Step 3: Add second object
        [
            (jnp.array([104, 52, 42, 32]), BoundingBoxMeasurement, jnp.eye(4)),
            (jnp.array([200, 100, 50, 40]), BoundingBoxMeasurement, jnp.eye(4)),
        ],
        # Step 4: Both objects continue
        [
            (jnp.array([106, 53, 43, 33]), BoundingBoxMeasurement, jnp.eye(4)),
            (jnp.array([198, 102, 52, 42]), BoundingBoxMeasurement, jnp.eye(4)),
        ],
    ]

    for step, measurements in enumerate(measurements_sequence):
        result = tracker.update(measurements, dt=1.0)

        if step == 0:
            # Should have 1 tentative track
            assert (
                result["track_summary"]["tentative"] >= 1
            ), f"Step {step}: Expected at least 1 tentative track"
        elif step == 1:
            # Should still have 1 track (confirmed or tentative)
            assert (
                result["track_summary"]["total"] >= 1
            ), f"Step {step}: Expected at least 1 track"
        elif step == 2:
            # Should have 2 tracks now
            assert (
                result["track_summary"]["total"] >= 2
            ), f"Step {step}: Expected at least 2 tracks"
        elif step == 3:
            # Should still have 2 tracks
            assert (
                result["track_summary"]["total"] >= 2
            ), f"Step {step}: Expected at least 2 tracks"

    final_stats = tracker.get_statistics()
    # next_track_id gives us the number of tracks created (since IDs start from 0)
    tracks_created = final_stats["next_track_id"]
    assert (
        tracks_created >= 2
    ), f"Should have created at least 2 tracks, got {tracks_created}"

    print("✓ Bounding box tracker integration tests passed")


def test_iou_threshold_behavior():
    """Test IoU threshold behavior."""
    print("Testing IoU threshold behavior...")

    # Test with different thresholds
    thresholds = [0.0, 0.3, 0.7, 1.0]

    track_states = [jnp.array([100, 50, 40, 30, 0, 0, 0, 0])]
    measurements = [
        (
            jnp.array([110, 55, 35, 25]),
            BoundingBoxMeasurement,
            jnp.eye(4),
        ),  # Moderate overlap
    ]

    for threshold in thresholds:
        iou_assoc = IoUNearestNeighbor(min_iou_threshold=threshold)
        associations = iou_assoc.associate(track_states, measurements)

        if threshold <= 0.5:  # Should associate at lower thresholds
            assert len(associations) > 0, f"Should associate at threshold {threshold}"
        # At very high thresholds, may not associate

    print("✓ IoU threshold behavior tests passed")


def test_edge_cases():
    """Test edge cases for bounding box tracking."""
    print("Testing edge cases...")

    # Test with no measurements
    iou_assoc = IoUNearestNeighbor()
    track_states = [jnp.array([100, 50, 40, 30, 0, 0, 0, 0])]
    associations = iou_assoc.associate(track_states, [])
    assert (
        len(associations) == 0
    ), "Should return empty associations for no measurements"

    # Test with no tracks
    measurements = [(jnp.array([100, 50, 40, 30]), BoundingBoxMeasurement, jnp.eye(4))]
    associations = iou_assoc.associate([], measurements)
    assert len(associations) == 0, "Should return empty associations for no tracks"

    # Test with non-bounding box measurements (should be ignored)
    from measurement_models.position_measurement_model import \
        PositionMeasurement

    mixed_measurements = [
        (jnp.array([100, 50]), PositionMeasurement, jnp.eye(2)),  # Non-bbox
        (jnp.array([100, 50, 40, 30]), BoundingBoxMeasurement, jnp.eye(4)),  # Bbox
    ]
    associations = iou_assoc.associate(track_states, mixed_measurements)
    # Should only consider the bounding box measurement
    assert (
        len(associations) <= 1
    ), "Should only associate with bounding box measurements"

    print("✓ Edge case tests passed")


if __name__ == "__main__":
    print("Running bounding box tracking tests...\n")

    test_bounding_box_measurement_model()
    test_bounding_box_motion_model()
    test_iou_association()
    test_bbox_tracker_integration()
    test_iou_threshold_behavior()
    test_edge_cases()

    print("\n✅ All bounding box tracking tests passed!")
