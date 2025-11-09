#!/usr/bin/env python3
"""
Example demonstrating bounding box tracking with IoU-based data association.

This example shows how to:
1. Set up a tracker for bounding box measurements
2. Simulate object detection measurements as bounding boxes
3. Track multiple objects with IoU-based association
4. Compare different IoU association algorithms
"""

from typing import Any, Dict, List, Tuple

import jax.numpy as jnp
import jax.random as random
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from measurement_models.bounding_box_measurement_model import \
    BoundingBoxMeasurement
from motion_models.bounding_box_motion_model import BoundingBoxConstantVelocity
from multi_track_tracker import MultiTrackTracker, TrackerConfig


def simulate_object_detections(
    step: int, num_objects: int = 3
) -> List[Tuple[jnp.ndarray, type, jnp.ndarray]]:
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
            "step": step,
            "num_measurements": len(measurements),
            "confirmed_tracks": result["track_summary"]["confirmed"],
            "tentative_tracks": result["track_summary"]["tentative"],
            "total_tracks": result["track_summary"]["total"],
        }
        track_history.append(track_info)

        # Print progress every 10 steps
        if step % 10 == 0:
            print(
                f"Step {step:3d}: {len(measurements)} measurements, "
                f"{track_info['confirmed_tracks']} confirmed tracks, "
                f"{track_info['tentative_tracks']} tentative tracks"
            )

    # Print final statistics
    final_stats = tracker.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Confirmed tracks: {final_stats['confirmed_tracks']}")
    print(f"  Tentative tracks: {final_stats['tentative_tracks']}")
    print(f"  Total tracks created: {final_stats['next_track_id']}")
    print(f"  Total tracks deleted: {final_stats['deleted_tracks']}")


def demonstrate_bbox_tracking_with_plots():
    """Demonstrate bounding box tracking with visualization."""
    print("=== Bounding Box Tracking with Visualization ===\n")

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
    print()

    # Collect data for plotting
    measurements_history = []
    tracker_history = []

    # Run simulation
    num_steps = 80
    for step in range(num_steps):
        measurements = simulate_object_detections(step)
        result = tracker.update(measurements, dt=1.0)

        # Store data for plotting
        measurements_history.append(measurements)

        # Extract track states for plotting
        tracks = {}
        if hasattr(tracker.track_manager, "tracks"):
            for track_id, track in tracker.track_manager.tracks.items():
                # Get current state vector [cx, cy, w, h, vcx, vcy, vw, vh]
                # track.state is a KalmanFilterTrackState with .x and .P attributes
                state = np.array(track.state.x)
                tracks[track_id] = state

        tracker_result = {
            "tracks": tracks,
            "track_summary": result["track_summary"],
            "step": step,
        }
        tracker_history.append(tracker_result)

        # Print progress every 20 steps
        if step % 20 == 0:
            track_summary = result["track_summary"]
            print(
                f"Step {step:3d}: {len(measurements)} measurements, "
                f"{track_summary['confirmed']} confirmed tracks, "
                f"{track_summary['tentative']} tentative tracks"
            )

    # Print final statistics
    final_stats = tracker.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"  Confirmed tracks: {final_stats['confirmed_tracks']}")
    print(f"  Tentative tracks: {final_stats['tentative_tracks']}")
    print(f"  Total tracks created: {final_stats['next_track_id']}")
    print(f"  Total tracks deleted: {final_stats['deleted_tracks']}")

    # Generate plots
    print(f"\nGenerating visualization plots...")

    # Plot overall tracking results
    plot_tracking_results(
        measurements_history, tracker_history, "IoU-Based Bounding Box Tracking"
    )

    # Plot snapshots at interesting time points
    interesting_steps = [10, 30, 50, 70]
    for step in interesting_steps:
        if step < len(measurements_history):
            measurements = measurements_history[step]
            tracks = tracker_history[step]["tracks"]
            plot_bounding_boxes_snapshot(
                measurements, tracks, step, "Bounding Box Tracking Snapshot"
            )


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

        print(
            f"  Final tracks: {stats['confirmed_tracks']} confirmed, "
            f"{stats['tentative_tracks']} tentative"
        )
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
        print(
            f"  Threshold {threshold:.1f}: BBox2={assoc2}, BBox3={assoc3}, BBox4={assoc4}"
        )


def plot_tracking_results(
    measurements_history: List[List[Tuple[jnp.ndarray, type, jnp.ndarray]]],
    tracker_history: List[Dict[str, Any]],
    title: str = "Bounding Box Tracking Results",
) -> None:
    """
    Plot the tracking results showing measurements and track trajectories.

    Args:
        measurements_history: List of measurements for each time step
        tracker_history: List of tracker results for each time step
        title: Plot title
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Trajectory plot (center positions over time)
    ax1.set_title(f"{title} - Trajectories")
    ax1.set_xlabel("X Position (pixels)")
    ax1.set_ylabel("Y Position (pixels)")
    ax1.grid(True, alpha=0.3)

    # Extract measurement trajectories
    measurement_trajectories = {}
    for step, measurements in enumerate(measurements_history):
        for i, (meas, _, _) in enumerate(measurements):
            if i not in measurement_trajectories:
                measurement_trajectories[i] = {"x": [], "y": [], "steps": []}
            measurement_trajectories[i]["x"].append(float(meas[0]))  # cx
            measurement_trajectories[i]["y"].append(float(meas[1]))  # cy
            measurement_trajectories[i]["steps"].append(step)

    # Extract track trajectories first
    track_trajectories = {}
    for step, result in enumerate(tracker_history):
        if "tracks" in result:
            for track_id, track_state in result["tracks"].items():
                if track_id not in track_trajectories:
                    track_trajectories[track_id] = {"x": [], "y": [], "steps": []}
                # Extract scalar values (track_state is already numpy array)
                track_trajectories[track_id]["x"].append(float(track_state[0]))  # cx
                track_trajectories[track_id]["y"].append(float(track_state[1]))  # cy
                track_trajectories[track_id]["steps"].append(step)

    # Plot track trajectories first (behind measurements)
    colors = ["red", "green", "blue", "orange", "purple", "brown"]
    for track_id, traj in track_trajectories.items():
        color = colors[track_id % len(colors)]
        ax1.plot(
            traj["x"],
            traj["y"],
            "s-",
            color=color,
            alpha=0.6,
            label=f"Track {track_id}",
            markersize=3,
            linewidth=1.5,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot measurement trajectories on top (more visible)
    for i, traj in measurement_trajectories.items():
        color = colors[i % len(colors)]
        ax1.plot(
            traj["x"],
            traj["y"],
            "o-",
            color=color,
            alpha=0.8,
            label=f"Measurements {i}",
            markersize=5,
            linewidth=2,
        )

    ax1.legend()

    # Plot 2: Size evolution over time
    ax2.set_title(f"{title} - Size Evolution")
    ax2.set_xlabel("Time Step")
    ax2.set_ylabel("Bounding Box Area (pixels²)")
    ax2.grid(True, alpha=0.3)

    # Plot track sizes first (behind measurements)
    for track_id, traj in track_trajectories.items():
        areas = []
        for step in traj["steps"]:
            result = tracker_history[step]
            if "tracks" in result and track_id in result["tracks"]:
                track_state = result["tracks"][track_id]
                # Extract scalar values (track_state is already numpy array)
                area = float(track_state[2] * track_state[3])  # w * h
                areas.append(area)
        color = colors[track_id % len(colors)]
        ax2.plot(
            traj["steps"],
            areas,
            "s-",
            color=color,
            alpha=0.6,
            label=f"Track {track_id}",
            markersize=3,
            linewidth=1.5,
            markerfacecolor="none",
            markeredgewidth=1.5,
        )

    # Plot measurement sizes on top (more visible)
    for i, traj in measurement_trajectories.items():
        areas = []
        for step in traj["steps"]:
            meas = measurements_history[step][i][0]
            area = float(meas[2] * meas[3])  # w * h
            areas.append(area)
        color = colors[i % len(colors)]
        ax2.plot(
            traj["steps"],
            areas,
            "o-",
            color=color,
            alpha=0.8,
            label=f"Measurements {i}",
            markersize=5,
            linewidth=2,
        )

    ax2.legend()

    plt.tight_layout()

    # Show plot and auto-close after delay
    plt.show(block=False)
    plt.draw()
    plt.pause(6.0)
    plt.close("all")


def plot_bounding_boxes_snapshot(
    measurements: List[Tuple[jnp.ndarray, type, jnp.ndarray]],
    tracks: Dict[int, jnp.ndarray],
    step: int,
    title: str = "Bounding Box Snapshot",
) -> None:
    """
    Plot a snapshot of bounding boxes at a specific time step.

    Args:
        measurements: List of measurements at this time step
        tracks: Dictionary of track_id -> track_state
        step: Current time step
        title: Plot title
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.set_title(f"{title} - Step {step}")
    ax.set_xlabel("X Position (pixels)")
    ax.set_ylabel("Y Position (pixels)")
    ax.grid(True, alpha=0.3)

    # Plot tracks first (behind measurements) as dashed rectangles
    colors = ["blue", "green", "orange", "purple", "brown", "pink"]
    for track_id, track_state in tracks.items():
        # Extract scalar values (track_state is already numpy array)
        cx, cy, w, h = (
            float(track_state[0]),
            float(track_state[1]),
            float(track_state[2]),
            float(track_state[3]),
        )
        x1, y1 = cx - w / 2, cy - h / 2

        color = colors[track_id % len(colors)]
        rect = patches.Rectangle(
            (x1, y1),
            w,
            h,
            linewidth=2,
            edgecolor=color,
            facecolor="none",
            linestyle="--",
            alpha=0.7,
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            cy - h / 4,
            f"T{track_id}",
            ha="center",
            va="center",
            fontweight="bold",
            color=color,
            fontsize=10,
        )

    # Plot measurements on top as solid rectangles (more visible)
    for i, (meas, _, _) in enumerate(measurements):
        cx, cy, w, h = float(meas[0]), float(meas[1]), float(meas[2]), float(meas[3])
        x1, y1 = cx - w / 2, cy - h / 2

        rect = patches.Rectangle(
            (x1, y1), w, h, linewidth=3, edgecolor="red", facecolor="red", alpha=0.4
        )
        ax.add_patch(rect)
        ax.text(
            cx,
            cy,
            f"M{i}",
            ha="center",
            va="center",
            fontweight="bold",
            color="darkred",
            fontsize=12,
        )

    # Set axis limits
    all_boxes = []
    for meas, _, _ in measurements:
        cx, cy, w, h = float(meas[0]), float(meas[1]), float(meas[2]), float(meas[3])
        all_boxes.extend([cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2])

    for track_state in tracks.values():
        # Extract scalar values (track_state is already numpy array)
        cx, cy, w, h = (
            float(track_state[0]),
            float(track_state[1]),
            float(track_state[2]),
            float(track_state[3]),
        )
        all_boxes.extend([cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2])

    if all_boxes:
        margin = 50
        ax.set_xlim(min(all_boxes) - margin, max(all_boxes) + margin)
        ax.set_ylim(min(all_boxes) - margin, max(all_boxes) + margin)

    # Add legend
    red_patch = patches.Patch(color="red", alpha=0.4, label="Measurements (solid)")
    blue_patch = patches.Patch(color="blue", alpha=0.7, label="Tracks (dashed)")
    ax.legend(handles=[red_patch, blue_patch])

    plt.tight_layout()

    # Check if we can display plots or need to save them
    import matplotlib

    backend = matplotlib.get_backend()

    if backend in ["Agg", "svg", "pdf", "ps"]:
        # Non-interactive backend - save plot
        filename = f"bbox_snapshot_step_{step}.png"
        plt.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close("all")
        print(f"Non-interactive backend detected, snapshot saved as {filename}")
    else:
        # Interactive backend - try to display
        try:
            plt.show(block=False)
            plt.draw()
            plt.pause(0.5)
            plt.close("all")
            print(f"Snapshot plot for step {step} displayed successfully")
        except Exception as e:
            # Fallback: save plot if display fails
            filename = f"bbox_snapshot_step_{step}.png"
            plt.savefig(filename, dpi=150, bbox_inches="tight")
            plt.close("all")
            print(f"Display failed, snapshot saved as {filename}")
            print(f"Exception: {e}")


if __name__ == "__main__":
    # Run all demonstrations
    demonstrate_bbox_tracking()
    compare_iou_algorithms()
    demonstrate_iou_calculation()

    # Ask user if they want to see visualizations
    try:
        import matplotlib.pyplot as plt_check

        print("\n" + "=" * 60)
        print("VISUALIZATION DEMO AVAILABLE")
        print("=" * 60)
        response = (
            input("Would you like to see tracking visualizations? (y/n): ")
            .lower()
            .strip()
        )
        if response in ["y", "yes"]:
            demonstrate_bbox_tracking_with_plots()
        else:
            print("Skipping visualization demo.")
    except ImportError:
        print(
            "\nNote: matplotlib not available. Install with 'pip install matplotlib' to see visualizations."
        )
