import jax.numpy as jnp
from measurement_models.measurement_model import MeasurementModel


def cubical_gate(
    self,
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
    self,
    state: jnp.ndarray,
    measurements: jnp.ndarray,
    measurement_model: MeasurementModel,
    gate_threshold: float,
) -> bool:
    return jnp.linalg.norm(measurements - state.x) < gate_threshold
