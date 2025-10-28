import jax.numpy as jnp

from measurement_models.bounding_box_measurement_model import \
    BoundingBoxMeasurement
from measurement_models.measurement_model import MeasurementModel


def cubical_gate(
    state: jnp.ndarray,
    measurement: jnp.ndarray,
    measurement_model: MeasurementModel,
    gate_threshold: float,
) -> bool:
    """
    Cubical gate for association.

    Args:
        state (jnp.ndarray): The state of the track.
        measurement (jnp.ndarray): The measurement.
        measurement_model (MeasurementModel): The measurement model associated with the measurement.
        gate_threshold (float): The gate threshold.

    Returns:
        bool: True if the measurements are within the cubical gate, False otherwise.
    """
    return jnp.all(
        jnp.abs(measurement - measurement_model.predict_measurement(state))
        < gate_threshold
    )


def ellipsoidal_gate(
    state: jnp.ndarray,
    measurement: jnp.ndarray,
    measurement_model: MeasurementModel,
    gate_threshold: float,
) -> bool:
    """
    Ellipsoidal gate for association using Mahalanobis distance.

    Args:
        state (jnp.ndarray): The state of the track.
        measurement (jnp.ndarray): The measurement.
        measurement_model (MeasurementModel): The measurement model associated with the measurement.
        gate_threshold (float): The gate threshold.

    Returns:
        bool: True if the measurement is within the ellipsoidal gate, False otherwise.
    """
    predicted_measurement = measurement_model.predict_measurement(state)
    residual = measurement - predicted_measurement
    return jnp.linalg.norm(residual) < gate_threshold


def iou_gate(
    state: jnp.ndarray,
    measurement: jnp.ndarray,
    measurement_model: MeasurementModel,
    gate_threshold: float,
) -> bool:
    """
    IoU-based gate for bounding box association.

    This gate is specifically designed for bounding box measurements.
    It computes the Intersection over Union (IoU) between the predicted
    bounding box and the measured bounding box, and accepts the association
    if IoU is above the threshold.

    Args:
        state: The state of the track (should be bounding box state)
        measurement: The measurement (should be bounding box measurement)
        measurement_model: The measurement model (should be BoundingBoxMeasurement)
        gate_threshold: The IoU threshold (between 0 and 1)

    Returns:
        True if the IoU is above the threshold, False otherwise
    """
    # Check if this is a bounding box measurement model
    if not issubclass(measurement_model, BoundingBoxMeasurement):
        # Fall back to ellipsoidal gate for non-bounding box measurements
        return ellipsoidal_gate(state, measurement, measurement_model, gate_threshold)

    # Convert state to bounding box format
    predicted_bbox = measurement_model.state_to_bbox(state)

    # Convert measurement to bounding box format
    measured_bbox = measurement_model.measurement_to_bbox(measurement)

    # Calculate IoU
    iou = measurement_model.calculate_iou(predicted_bbox, measured_bbox)

    return iou >= gate_threshold
