#!/usr/bin/env python3
"""
Example demonstrating bounding box tracking with IoU-based data association.

This example shows how to:
1. Set up a tracker for bounding box measurements
2. Simulate object detection measurements as bounding boxes
3. Track multiple objects with IoU-based association
4. Compare different IoU association algorithms
"""

from typing import List, Tuple

import jax.numpy as jnp
import jax.random as random
import numpy as np

from measurement_models.bounding_box_measurement_model import BoundingBoxMeasurement
from motion_models.bounding_box_motion_model import BoundingBoxConstantVelocity
from multi_track_tracker import MultiTrackTracker, TrackerConfig


def simulate_object_detections(step: int, num_objects: int = 3) -> List[Tuple[jnp.ndarray, type, jnp.ndarray]]:
    """
    Simulate object detection measurements as bounding boxes.
    
    Args:
        step: Current time step
        num_objects: Number of objects to simulate
        
    Returns:
        List of (measurement, measurement_model, noise_covariance) tuples
    """
    measurements = []
    
    # Object 1: Moving right and slightly down
    if step < 50:  # Object appears for first 50 steps
        cx1 = 100 + step * 3  # Moving right
        cy1 = 100 + step * 0.5  # Moving slightly down
        w1 = 50 + step * 0.2  # Slightly growing
        h1 = 30 + step * 0.1
        
        measurement1 = jnp.array([cx1, cy1, w1, h1])
        noise_cov1 = jnp.diag(jnp.array([2.0, 2.0, 1.0, 1.0]))  # [cx, cy, w, h] noise
        measurements.append((measurement1, BoundingBoxMeasurement, noise_cov1))
    
    # Object 2: Moving left and up
    if step < 60:  # Object appears for first 60 steps
        cx2 = 300 - step * 2  # Moving left
        cy2 = 200 - step * 1  # Moving up
        w2 = 40 + np.sin(step * 0.1) * 5  # Oscillating size
        h2 = 25 + np.cos(step * 0.1) * 3
        
        measurement2 = jnp.array([cx2, cy2, w2, h2])
        noise_cov2 = jnp.diag(jnp.array([2.0, 2.0, 1.0, 1.0]))
        measurements.append((measurement2, BoundingBoxMeasurement, noise_cov2))
    
    # Object 3: Moving diagonally (appears later)
    if step > 20 and step < 80:  # Object appears from step 20 to 80
        cx3 = 50 + (step - 20) * 4  # Moving right
        cy3 = 50 + (step - 20) * 3  # Moving down
        w3 = 35 - (step - 20) * 0.1  # Shrinking
        h3 = 45 - (step - 20) * 0.15
        
        measurement3 = jnp.array([cx3, cy3, w3, h3])
        noise_cov3 = jnp.diag(jnp.array([2.0, 2.0, 1.0, 1.0]))
        measurements.append((measurement3, BoundingBoxMeasurement, noise_cov3))
    
    # Add some noise to measurements
    key = random.PRNGKey(step)
    for i, (measurement, model, noise_cov) in enumerate(measurements):
        key, subkey = random.split(key)
        noise = random.multivariate_normal(subkey, jnp.zeros(4), noise_cov)
        measurements[i] = (measurement + noise, model, noise_cov)
    
    return measurements


def demonstrate_bbox_tracking():
    """Demonstrate basic bounding box tracking with IoU association."""
    print("=== Bounding Box Tracking Demonstration ===\n")
    
    # Configure tracker for bounding box tracking
    config = TrackerConfig(
        motion_model_class=BoundingBoxConstantVelocity,
        data_association_algorithm="iou_nearest_neighbor",
        gate_threshold=0.1,  # Minimum IoU threshold
        confirmation_threshold=3,
        deletion_threshold=5,
        process_noise_scale=1.0,
        initial_covariance_scale=10.0,
    )
    
    tracker = MultiTrackTracker(config)
    
    print("Tracking Configuration:")
    print(f"  Motion Model: {config.motion_model_class.__name__}")
    print(f"  Data Association: {config.data_association_algorithm}")
    print(f"  IoU Threshold: {config.gate_threshold}")
    print(f"  Confirmation Threshold: {config.confirmation_threshold}")
    print(f"  Deletion Threshold: {config.deletion_threshold}")
    print()
    
    # Run tracking simulation
    num_steps = 100
    track_history = []
    
    for step in range(num_steps):
        # Generate measurements
        measurements = simulate_object_detections(step)
        
        # Update tracker
        result = tracker.update(measurements, dt=1.0)
        
        # Store tracking results
        track_info = {
            'step': step,
            'num_measurements': len(measurements),
            'confirmed_tracks': result['track_summary']['confirmed'],
            'tentative_tracks': result['track_summary']['tentative'],
            'total_tracks': result['track_summary']['total'],
        }
        track_history.append(track_info)
        
        # Print progress every 10 steps
        if step % 10 == 0:
            print(f"Step {step:3d}: {len(measurements)} measurements, "
                  f"{track_info['confirmed_tracks']} confirmed tracks, "
                  f"{track_info['tentative_tracks']} tentative tracks")
    
    # Print final statistics
    final_stats = tracker.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Confirmed tracks: {final_stats['confirmed_tracks']}")
    print(f"  Tentative tracks: {final_stats['tentative_tracks']}")
    print(f"  Total tracks created: {final_stats['next_track_id']}")
    print(f"  Total tracks deleted: {final_stats['deleted_tracks']}")


def compare_iou_algorithms():
    """Compare different IoU-based association algorithms."""
    print("\n=== IoU Algorithm Comparison ===\n")
    
    algorithms = [
        ("IoU Nearest Neighbor", "iou_nearest_neighbor"),
        ("IoU Optimal Assignment", "iou_optimal"),
    ]
    
    results = {}
    
    for name, algorithm in algorithms:
        print(f"Testing {name}...")
        
        config = TrackerConfig(
            motion_model_class=BoundingBoxConstantVelocity,
            data_association_algorithm=algorithm,
            gate_threshold=0.1,  # IoU threshold
            confirmation_threshold=2,
            deletion_threshold=3,
        )
        
        tracker = MultiTrackTracker(config)
        
        # Run shorter simulation for comparison
        for step in range(50):
            measurements = simulate_object_detections(step, num_objects=2)
            tracker.update(measurements, dt=1.0)
        
        stats = tracker.get_statistics()
        results[name] = stats
        
        print(f"  Final tracks: {stats['confirmed_tracks']} confirmed, "
              f"{stats['tentative_tracks']} tentative")
        print(f"  Total created: {stats['next_track_id']}")
        print(f"  Total deleted: {stats['deleted_tracks']}")
        print()
    
    return results


def demonstrate_iou_calculation():
    """Demonstrate IoU calculation between bounding boxes."""
    print("=== IoU Calculation Demonstration ===\n")
    
    # Define some example bounding boxes
    bbox1 = jnp.array([100, 100, 150, 140])  # [x1, y1, x2, y2]
    bbox2 = jnp.array([120, 110, 170, 150])  # Overlapping
    bbox3 = jnp.array([200, 200, 250, 240])  # Non-overlapping
    bbox4 = jnp.array([110, 105, 145, 135])  # Highly overlapping
    
    print("Bounding Box Examples:")
    print(f"  BBox1: {bbox1} (reference)")
    print(f"  BBox2: {bbox2} (overlapping)")
    print(f"  BBox3: {bbox3} (non-overlapping)")
    print(f"  BBox4: {bbox4} (highly overlapping)")
    print()
    
    # Calculate IoU values
    iou12 = BoundingBoxMeasurement.calculate_iou(bbox1, bbox2)
    iou13 = BoundingBoxMeasurement.calculate_iou(bbox1, bbox3)
    iou14 = BoundingBoxMeasurement.calculate_iou(bbox1, bbox4)
    
    print("IoU Calculations:")
    print(f"  IoU(BBox1, BBox2): {iou12:.3f}")
    print(f"  IoU(BBox1, BBox3): {iou13:.3f}")
    print(f"  IoU(BBox1, BBox4): {iou14:.3f}")
    print()
    
    # Show how IoU threshold affects associations
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.7]
    print("Association decisions at different IoU thresholds:")
    for threshold in thresholds:
        assoc2 = "✓" if iou12 >= threshold else "✗"
        assoc3 = "✓" if iou13 >= threshold else "✗"
        assoc4 = "✓" if iou14 >= threshold else "✗"
        print(f"  Threshold {threshold:.1f}: BBox2={assoc2}, BBox3={assoc3}, BBox4={assoc4}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_bbox_tracking()
    compare_iou_algorithms()
    demonstrate_iou_calculation()
    
    print("\n✅ Bounding box tracking demonstration complete!")
