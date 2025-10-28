# Multi-Track Tracker in JAX

A comprehensive multi-target tracking system implemented in JAX, featuring Kalman filter-based tracking with configurable data association algorithms, motion models, and measurement models.

## Features

- **Multi-target tracking**: Simultaneously track multiple objects
- **Kalman filter-based**: Uses Kalman Filter and its extension for state estimation
- **Configurable data association**: Support several data association algorithms
- **Track management**: Automatic track initialization, confirmation, and deletion
- **Multiple sensor support**: Handle measurements from different sensor types (radar, position, etc.) using different measurement models
- **JAX-powered**: Leverages JAX for efficient numerical computations and potential GPU acceleration

## Get Started

### Installation

```bash
# Clone the repository
git clone https://github.com/XZhouUM/jax_tracker.git
cd jax_tracker

# Run the setup script
./post_clone.sh

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import jax.numpy as jnp
from multi_track_tracker import MultiTrackTracker, TrackerConfig
from motion_models.constant_velocity_model import ConstantVelocity
from measurement_models.radar_measurement_model import RadarMeasurement

# Configure tracker
config = TrackerConfig(
    motion_model_class=ConstantVelocity,
    data_association_algorithm="jpda",  # "nearest_neighbor", "global_nearest_neighbor", "mht", or "jpda"
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

### Examples and Tests

#### Example Scripts
```bash
# Basic tracker example with visualization
python example_usage.py

# Multi-Hypothesis Tracking demonstration and algorithm comparison
python example_mht.py

# Joint Probabilistic Data Association demonstration and probability analysis
python example_jpda.py
```

#### Test Scripts
```bash
# Basic tracker functionality tests (8 tests)
python test_tracker.py

# Multi-Hypothesis Tracking algorithm tests (6 tests)
python test_mht.py

# Joint Probabilistic Data Association algorithm tests (7 tests)
python test_jpda.py

# Run all tests at once
python test_tracker.py && python test_mht.py && python test_jpda.py
```

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
│   ├── multi_hypothesis_tracking.py # MHT algorithm
│   ├── joint_probabilistic_data_association.py # JPDA algorithm
│   └── association_gate.py         # Gating functions
├── motion_models/
│   ├── motion_model.py             # Abstract base class
│   └── constant_velocity_model.py  # 2D constant velocity model
└── measurement_models/
    ├── measurement_model.py        # Abstract base class
    ├── radar_measurement_model.py  # Radar measurements (range, range-rate, azimuth)
    └── position_measurement_model.py # Direct position measurements
```

## Configuration

### TrackerConfig Options

The `TrackerConfig` class allows customization of tracker behavior:

- `motion_model_class`: Motion model for track prediction
- `data_association_algorithm`: "nearest_neighbor", "global_nearest_neighbor", "mht", or "jpda"
- `gate_threshold`: Distance threshold for measurement-to-track association
- `confirmation_threshold`: Number of hits needed to confirm a track
- `deletion_threshold`: Number of consecutive misses before track deletion
- `max_tracks`: Maximum number of tracks to maintain
- `process_noise_scale`: Scale factor for motion model uncertainty
- `initial_covariance_scale`: Scale factor for new track uncertainty

### Supported Measurement Models

#### Radar Measurements
- **Format**: [range, range_rate, azimuth]
- **Use case**: Radar sensors providing polar coordinates and Doppler information

#### Position Measurements
- **Format**: [x, y]
- **Use case**: Direct Cartesian position measurements (e.g., from computer vision)

### Track Management

The tracker automatically handles:

1. **Track Initialization**: New tracks created from unassociated measurements
2. **Track Confirmation**: Tracks confirmed after sufficient successful associations
3. **Track Deletion**: Tracks deleted after consecutive missed detections
4. **Memory Management**: Automatic cleanup of deleted tracks

### Performance Considerations

- JAX compilation provides significant speedup after initial overhead
- Consider using `jax.jit` for production deployments
- GPU acceleration available through JAX (requires appropriate JAX installation)
- Memory usage scales with number of active tracks and measurements

## Data Association Algorithms

The tracker supports four data association algorithms, each with distinct characteristics and trade-offs:

### Algorithm Overview

#### 1. Nearest Neighbor (NN)
- **Algorithm**: Greedy local assignment of measurements to tracks
- **Decision Making**: Hard assignments based on distance
- **Computational Complexity**: O(n*m) where n=tracks, m=measurements
- **Memory Requirements**: Minimal
- **Pros**:
  - Extremely fast execution
  - Simple implementation and debugging
  - Low memory footprint
  - Real-time capable
- **Cons**:
  - Suboptimal in dense target scenarios
  - No global optimization
  - Poor handling of crossing targets
  - Susceptible to measurement noise
- **Best Use Case**: Real-time applications with well-separated targets and low noise

#### 2. Global Nearest Neighbor (GNN)
- **Algorithm**: Optimal global assignment using Hungarian algorithm for current scan
- **Decision Making**: Globally optimal hard assignments
- **Computational Complexity**: O(n³) for Hungarian algorithm
- **Memory Requirements**: Low
- **Pros**:
  - Globally optimal for single scan
  - Handles crossing targets well
  - Better than NN in dense scenarios
  - Deterministic results
- **Cons**:
  - No memory of past associations
  - Computationally expensive for large problems
  - Still makes hard decisions
  - No handling of false alarms
- **Best Use Case**: Moderate number of targets with occasional crossings, when optimality matters

#### 3. Multi-Hypothesis Tracking (MHT)
- **Algorithm**: Maintains multiple hypotheses about track-measurement associations over time
- **Decision Making**: Deferred decisions, maintains multiple possibilities
- **Computational Complexity**: Exponential in worst case, managed by pruning
- **Memory Requirements**: High (stores multiple hypotheses)
- **Pros**:
  - Handles ambiguous scenarios excellently
  - Manages false alarms and missed detections
  - Uses temporal information effectively
  - Theoretically optimal with infinite resources
  - Robust in cluttered environments
- **Cons**:
  - High computational cost
  - Significant memory requirements
  - Complex implementation
  - Requires careful pruning strategies
- **Best Use Case**: Cluttered environments with false alarms, crossing targets, and when accuracy is paramount

#### 4. Joint Probabilistic Data Association (JPDA)
- **Algorithm**: Calculates association probabilities and updates tracks with weighted measurements
- **Decision Making**: Probabilistic, no hard assignments
- **Computational Complexity**: O(2^(n*m)) for enumeration, managed by gating
- **Memory Requirements**: Moderate
- **Pros**:
  - Probabilistic framework handles uncertainty gracefully
  - No premature hard decisions
  - Provides uncertainty quantification
  - Better than NN/GNN in ambiguous scenarios
  - Single-scan processing (no hypothesis management)
- **Cons**:
  - Assumes known number of targets
  - Computational complexity grows exponentially
  - Requires parameter tuning (clutter density, detection probability)
  - May underperform MHT in highly cluttered environments
- **Best Use Case**: Scenarios with measurement uncertainty, moderate target density, and known target count

### Algorithm Comparison Matrix

| Aspect               | NN    | GNN  | MHT   | JPDA  |
| -------------------- | ----- | ---- | ----- | ----- |
| **Speed**            | ⭐⭐⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐    | ⭐⭐⭐   |
| **Memory**           | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐     | ⭐⭐⭐   |
| **Accuracy**         | ⭐⭐    | ⭐⭐⭐  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  |
| **Clutter Handling** | ⭐     | ⭐    | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  |
| **Crossing Targets** | ⭐     | ⭐⭐⭐  | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐  |
| **Implementation**   | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐    | ⭐⭐⭐   |
| **Real-time**        | ⭐⭐⭐⭐⭐ | ⭐⭐⭐  | ⭐⭐    | ⭐⭐⭐   |
| **Uncertainty**      | ⭐     | ⭐    | ⭐⭐⭐⭐  | ⭐⭐⭐⭐⭐ |

### Selection Guide

#### Choose **Nearest Neighbor** when:
- Real-time performance is critical
- Targets are well-separated
- Simple implementation is preferred
- Computational resources are limited
- Tracking quality requirements are moderate

#### Choose **Global Nearest Neighbor** when:
- You need better accuracy than NN
- Targets occasionally cross paths
- You can afford moderate computational cost
- Single-scan optimality is important
- False alarms are rare

#### Choose **Multi-Hypothesis Tracking** when:
- Maximum tracking accuracy is required
- Environment has significant clutter/false alarms
- Targets frequently cross or merge
- Computational resources are available
- Missing a target is costly

#### Choose **Joint Probabilistic Data Association** when:
- You need uncertainty quantification
- Target count is approximately known
- Moderate clutter levels exist
- You want probabilistic rather than hard decisions
- Balance between accuracy and computational cost is needed

### Algorithm-Specific Configuration

#### MHT Configuration
```python
config = TrackerConfig(
    motion_model_class=ConstantVelocity,
    data_association_algorithm="mht",
    gate_threshold=5.0,
    confirmation_threshold=3,
    deletion_threshold=5
)

tracker = MultiTrackTracker(config)

# MHT-specific parameters can be accessed via the data associator
mht = tracker.data_associator
print(f"Active hypotheses: {mht.get_hypothesis_count()}")
best_hypothesis = mht.get_best_hypothesis()
```

**MHT Key Features:**
- **Hypothesis Management**: Maintains multiple possible explanations for observations
- **Deferred Decision Making**: Delays hard decisions until more information is available
- **False Alarm Handling**: Explicitly models and accounts for false alarm measurements
- **Track Initialization**: Handles ambiguous track birth scenarios
- **Pruning**: Automatically removes unlikely hypotheses to maintain computational tractability

#### JPDA Configuration
```python
config = TrackerConfig(
    motion_model_class=ConstantVelocity,
    data_association_algorithm="jpda",
    gate_threshold=5.0,
    confirmation_threshold=3,
    deletion_threshold=5
)

tracker = MultiTrackTracker(config)

# JPDA-specific parameters can be accessed via the data associator
jpda = tracker.data_associator
association_probs = jpda.get_association_probabilities()
if association_probs:
    print(f"Association probability matrix shape: {association_probs.prob_matrix.shape}")
```

**JPDA Key Features:**
- **Probabilistic Associations**: Calculates probabilities for all possible track-measurement pairs
- **Weighted Updates**: Updates tracks using weighted combinations of measurements
- **Uncertainty Handling**: Gracefully handles association uncertainty without hard decisions
- **Probability Constraints**: Ensures association probabilities sum to 1.0 for consistency
- **Combined Innovations**: Provides weighted innovation vectors for Kalman filter updates