# Multi-Track Tracker in JAX

A comprehensive multi-target tracking system implemented in JAX, featuring Kalman filter-based tracking with configurable data association algorithms, motion models, and measurement models.

## Features

- **Multi-target tracking**: Simultaneously track multiple objects
- **Kalman filter-based**: Uses Kalman Filter and its extension for state estimation
- **Configurable data association**: Support several data association algorithms
- **Track management**: Automatic track initialization, confirmation, and deletion
- **Multiple sensor support**: Handle measurements from different sensor types (radar, position, etc.) using different measurement models
- **JAX-powered**: Leverages JAX for efficient numerical computations and potential GPU acceleration

## Architecture

### Core Components

1. **MultiTrackTracker**: Main tracker class that coordinates all components
2. **TrackManager**: Handles track lifecycle (initialization, confirmation, deletion)
3. **Data Association**: Algorithms for associating measurements to tracks
4. **Motion Models**: Predict object motion (currently supports constant velocity)
5. **Measurement Models**: Map object states to expected measurements
6. **Measurement Preprocessing**: Handle multi-sensor measurements and coordinate transformations

### Directory Structure

```
jax_tracker/
├── multi_track_tracker.py          # Main tracker implementation
├── track_management.py             # Track lifecycle management
├── kalman_filter_track.py          # Individual track implementation
├── measurement_preprocessing.py     # Sensor data handling
├── example_usage.py                # Usage examples
├── test_tracker.py                 # Basic tests
├── data_association/
│   ├── data_association.py         # Abstract base class
│   ├── nearest_neighbor.py         # NN and GNN algorithms
│   └── association_gate.py         # Gating functions
├── motion_models/
│   ├── motion_model.py             # Abstract base class
│   └── constant_velocity_model.py  # 2D constant velocity model
└── measurement_models/
    ├── measurement_model.py        # Abstract base class
    ├── radar_measurement_model.py  # Radar measurements (range, range-rate, azimuth)
    └── position_measurement_model.py # Direct position measurements
```

## Quick Start

### Basic Usage

```python
import jax.numpy as jnp
from multi_track_tracker import MultiTrackTracker, TrackerConfig
from motion_models.constant_velocity_model import ConstantVelocity
from measurement_models.radar_measurement_model import RadarMeasurement

# Configure tracker
config = TrackerConfig(
    motion_model_class=ConstantVelocity,
    data_association_algorithm="nearest_neighbor",
    gate_threshold=10.0,
    confirmation_threshold=3,
    deletion_threshold=5
)

# Initialize tracker
tracker = MultiTrackTracker(config)

# Process measurements
measurements = [
    (jnp.array([100.0, 5.0, 0.5]), RadarMeasurement, jnp.eye(3))  # (measurement, model, noise_cov)
]

result = tracker.update(measurements, dt=1.0)
print(f"Confirmed tracks: {result['track_summary']['confirmed']}")
```

### Running Examples

```bash
# Run the main example with visualization
python example_usage.py

# Run basic tests
python test_tracker.py
```

## Configuration Options

The `TrackerConfig` class allows customization of tracker behavior:

- `motion_model_class`: Motion model for track prediction
- `data_association_algorithm`: "nearest_neighbor" or "global_nearest_neighbor"
- `gate_threshold`: Distance threshold for measurement-to-track association
- `confirmation_threshold`: Number of hits needed to confirm a track
- `deletion_threshold`: Number of consecutive misses before track deletion
- `max_tracks`: Maximum number of tracks to maintain
- `process_noise_scale`: Scale factor for motion model uncertainty
- `initial_covariance_scale`: Scale factor for new track uncertainty

## Supported Measurement Models

### Radar Measurements
- **Format**: [range, range_rate, azimuth]
- **Use case**: Radar sensors providing polar coordinates and Doppler information

### Position Measurements
- **Format**: [x, y]
- **Use case**: Direct Cartesian position measurements (e.g., from computer vision)

## Data Association Algorithms

### Nearest Neighbor (NN)
- Assigns each measurement to the closest valid track
- Fast and simple, suitable for sparse target environments
- May struggle with closely spaced targets

### Global Nearest Neighbor (GNN)
- Finds globally optimal assignment using assignment problem formulation
- Better performance with closely spaced targets
- Higher computational cost

## Track Management

The tracker automatically handles:

1. **Track Initialization**: New tracks created from unassociated measurements
2. **Track Confirmation**: Tracks confirmed after sufficient successful associations
3. **Track Deletion**: Tracks deleted after consecutive missed detections
4. **Memory Management**: Automatic cleanup of deleted tracks

## Performance Considerations

- JAX compilation provides significant speedup after initial overhead
- Consider using `jax.jit` for production deployments
- GPU acceleration available through JAX (requires appropriate JAX installation)
- Memory usage scales with number of active tracks and measurements