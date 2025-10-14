from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Type

import jax.numpy as jnp

from measurement_models.measurement_model import MeasurementModel


@dataclass
class SensorInfo:
    """Information about a sensor."""

    sensor_id: str
    sensor_type: str  # "radar", "lidar", "camera", etc.
    position: jnp.ndarray  # Sensor position [x, y, z]
    orientation: jnp.ndarray  # Sensor orientation [roll, pitch, yaw]
    measurement_model: Type[MeasurementModel]
    default_noise_covariance: jnp.ndarray


class MeasurementPreprocessor:
    """
    Preprocesses measurements from multiple sensors before feeding to the tracker.

    Handles coordinate transformations, noise estimation, and measurement validation.
    """

    def __init__(self, sensors: Dict[str, SensorInfo]):
        """
        Initialize measurement preprocessor.

        Args:
            sensors: Dictionary mapping sensor_id to SensorInfo
        """
        self.sensors = sensors

    def preprocess_measurements(
        self, raw_measurements: List[Dict[str, Any]], timestamp: float
    ) -> List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]]:
        """
        Preprocess raw measurements from multiple sensors.

        Args:
            raw_measurements: List of raw measurement dictionaries
            timestamp: Measurement timestamp

        Returns:
            List of (measurement, measurement_model, noise_covariance) tuples
        """
        processed_measurements = []

        for raw_meas in raw_measurements:
            try:
                processed = self._preprocess_single_measurement(raw_meas, timestamp)
                if processed is not None:
                    processed_measurements.append(processed)
            except Exception as e:
                print(f"Warning: Failed to preprocess measurement {raw_meas}: {e}")
                continue

        return processed_measurements

    def _preprocess_single_measurement(
        self, raw_measurement: Dict[str, Any], timestamp: float
    ) -> Optional[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]]:
        """
        Preprocess a single raw measurement.

        Args:
            raw_measurement: Raw measurement dictionary
            timestamp: Measurement timestamp

        Returns:
            Processed measurement tuple or None if invalid
        """
        # Extract sensor information
        sensor_id = raw_measurement.get("sensor_id")
        if sensor_id not in self.sensors:
            raise ValueError(f"Unknown sensor: {sensor_id}")

        sensor_info = self.sensors[sensor_id]

        # Extract measurement data
        measurement_data = raw_measurement.get("data")
        if measurement_data is None:
            raise ValueError("No measurement data provided")

        # Convert to JAX array
        measurement = jnp.array(measurement_data)

        # Validate measurement
        if not self._validate_measurement(measurement, sensor_info):
            return None

        # Apply coordinate transformation if needed
        measurement = self._transform_measurement(measurement, sensor_info)

        # Estimate noise covariance
        noise_covariance = self._estimate_noise_covariance(raw_measurement, sensor_info)

        return measurement, sensor_info.measurement_model, noise_covariance

    def _validate_measurement(
        self, measurement: jnp.ndarray, sensor_info: SensorInfo
    ) -> bool:
        """
        Validate measurement based on sensor type and expected format.

        Args:
            measurement: Measurement vector
            sensor_info: Sensor information

        Returns:
            True if measurement is valid, False otherwise
        """
        # Check for NaN or infinite values
        if not jnp.all(jnp.isfinite(measurement)):
            return False

        # Sensor-specific validation
        if sensor_info.sensor_type == "radar":
            if measurement.shape[0] != 3:  # [range, range_rate, azimuth]
                return False
            range_val, range_rate, azimuth = measurement

            # Check range is positive
            if range_val <= 0:
                return False

            # Check azimuth is in valid range
            if not (-jnp.pi <= azimuth <= jnp.pi):
                return False

        elif sensor_info.sensor_type == "position":
            if measurement.shape[0] != 2:  # [x, y]
                return False

        return True

    def _transform_measurement(
        self, measurement: jnp.ndarray, sensor_info: SensorInfo
    ) -> jnp.ndarray:
        """
        Transform measurement from sensor coordinate frame to global frame.

        Args:
            measurement: Measurement in sensor frame
            sensor_info: Sensor information

        Returns:
            Measurement in global frame
        """
        # For simplicity, assume measurements are already in global frame
        # In practice, you would apply rotation and translation transformations
        return measurement

    def _estimate_noise_covariance(
        self, raw_measurement: Dict[str, Any], sensor_info: SensorInfo
    ) -> jnp.ndarray:
        """
        Estimate measurement noise covariance.

        Args:
            raw_measurement: Raw measurement dictionary
            sensor_info: Sensor information

        Returns:
            Noise covariance matrix
        """
        # Check if measurement includes uncertainty information
        if "covariance" in raw_measurement:
            return jnp.array(raw_measurement["covariance"])
        elif "uncertainty" in raw_measurement:
            # Convert uncertainty to covariance (assume diagonal)
            uncertainty = jnp.array(raw_measurement["uncertainty"])
            return jnp.diag(uncertainty**2)
        else:
            # Use default sensor noise covariance
            return sensor_info.default_noise_covariance


class MeasurementSimulator:
    """
    Simulates measurements from multiple sensors for testing purposes.
    """

    def __init__(self, sensors: Dict[str, SensorInfo]):
        """
        Initialize measurement simulator.

        Args:
            sensors: Dictionary mapping sensor_id to SensorInfo
        """
        self.sensors = sensors

    def simulate_measurements(
        self,
        true_states: List[jnp.ndarray],
        timestamp: float,
        detection_probability: float = 0.9,
        false_alarm_rate: float = 0.1,
        key: Optional[jnp.ndarray] = None,
    ) -> List[Dict[str, Any]]:
        """
        Simulate measurements from true target states.

        Args:
            true_states: List of true target state vectors
            timestamp: Measurement timestamp
            detection_probability: Probability of detecting each target
            false_alarm_rate: Rate of false alarm measurements
            key: Random key for JAX random number generation

        Returns:
            List of simulated raw measurements
        """
        if key is None:
            key = jnp.array([0, 1])  # Default key

        measurements = []

        for sensor_id, sensor_info in self.sensors.items():
            sensor_measurements = self._simulate_sensor_measurements(
                sensor_id,
                sensor_info,
                true_states,
                timestamp,
                detection_probability,
                false_alarm_rate,
                key,
            )
            measurements.extend(sensor_measurements)

        return measurements

    def _simulate_sensor_measurements(
        self,
        sensor_id: str,
        sensor_info: SensorInfo,
        true_states: List[jnp.ndarray],
        timestamp: float,
        detection_probability: float,
        false_alarm_rate: float,
        key: jnp.ndarray,
    ) -> List[Dict[str, Any]]:
        """Simulate measurements from a single sensor."""
        measurements = []

        # Simulate detections from true targets
        for i, state in enumerate(true_states):
            # Random detection decision
            key, subkey = jax.random.split(key)
            if jax.random.uniform(subkey) < detection_probability:
                # Generate noisy measurement
                true_measurement = sensor_info.measurement_model.predict_measurement(
                    state
                )

                # Add noise
                key, subkey = jax.random.split(key)
                noise = jax.random.multivariate_normal(
                    subkey,
                    jnp.zeros(true_measurement.shape[0]),
                    sensor_info.default_noise_covariance,
                )
                noisy_measurement = true_measurement + noise

                measurements.append(
                    {
                        "sensor_id": sensor_id,
                        "data": noisy_measurement.tolist(),
                        "timestamp": timestamp,
                        "target_id": i,  # For evaluation purposes
                    }
                )

        # Simulate false alarms
        key, subkey = jax.random.split(key)
        num_false_alarms = jax.random.poisson(subkey, false_alarm_rate)

        for _ in range(int(num_false_alarms)):
            # Generate random false alarm measurement
            if sensor_info.sensor_type == "radar":
                # Random radar measurement
                key, subkey = jax.random.split(key)
                range_val = jax.random.uniform(subkey, minval=10.0, maxval=1000.0)
                key, subkey = jax.random.split(key)
                range_rate = jax.random.uniform(subkey, minval=-50.0, maxval=50.0)
                key, subkey = jax.random.split(key)
                azimuth = jax.random.uniform(subkey, minval=-jnp.pi, maxval=jnp.pi)

                false_measurement = jnp.array([range_val, range_rate, azimuth])
            else:
                # Random position measurement
                key, subkey = jax.random.split(key)
                x = jax.random.uniform(subkey, minval=-500.0, maxval=500.0)
                key, subkey = jax.random.split(key)
                y = jax.random.uniform(subkey, minval=-500.0, maxval=500.0)

                false_measurement = jnp.array([x, y])

            measurements.append(
                {
                    "sensor_id": sensor_id,
                    "data": false_measurement.tolist(),
                    "timestamp": timestamp,
                    "target_id": -1,  # Indicates false alarm
                }
            )

        return measurements


# Import jax.random for the simulator
import jax.random
