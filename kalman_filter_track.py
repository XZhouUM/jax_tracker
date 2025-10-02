import jax.numpy as jnp
from typing import NamedTuple, Dict, Type, Tuple
from motion_models.motion_model import MotionModel
from measurement_models.measurement_model import MeasurementModel


class KalmanFilterTrackState(NamedTuple):
    x: jnp.ndarray  # state vector
    P: jnp.ndarray  # covariance matrix

    @classmethod
    def create(cls, x: jnp.ndarray, P: jnp.ndarray) -> 'KalmanFilterTrackState':
        """Create a new KalmanFilterTrackState instance with dimension check."""
        if x.shape[0] != P.shape[0] or x.shape[0] != P.shape[1]:
            raise ValueError(f"Incompatible dimensions: x {x.shape}, P {P.shape}. "
                             "Expected x.shape[0] == P.shape[0] == P.shape[1].")
        return cls(x, P)


class KalmanFilterTrack:
    def __init__(self,
                 initial_state: KalmanFilterTrackState,
                 motion_model_class: Type[MotionModel]) -> None:
        """
        Initialize the Kalman Filter Track.

        Args:
            initial_state: Initial state of the tracker.
            motion_model_class: Motion model class with static methods.
        """
        self.motion_model_class = motion_model_class
        self.state = initial_state

    def _time_update(self, dt: float, Q: jnp.ndarray) -> None:
        """
        Predict the next state using the motion model.

        Args:
            dt: Time step.
            Q: Process noise covariance matrix.
        """
        self.state.x = self.motion_model_class.transition(self.state.x, dt)
        F = self.motion_model_class.jacobian(self.state.x, dt)
        self.state.P = F @ self.state.P @ F.T + Q

    def _measurement_update(
        self,
        measurements: Dict[jnp.ndarray, Tuple[Type[MeasurementModel], jnp.ndarray]]
    ) -> None:
        """
        Update the state based on measurements.

        The measurements can be empty. In case of empty, no measurement update is performed. Therefore,
        the track is coasting with only the motion prediction.

        Args:
            measurements: Dictionary mapping measurement vectors to
                (MeasurementModel class, measurement noise R) tuples.
        """
        if not measurements:
            return

        for measurement, (measurement_model, R) in measurements.items():
            # Predicted measurement and Jacobian
            measurement_prediction = measurement_model.predict_measurement(self.state.x)
            H = measurement_model.jacobian(self.state.x)

            # Innovation / residual
            y = measurement - measurement_prediction

            # Innovation covariance
            S = H @ self.state.P @ H.T + R

            # Kalman gain
            K = self.state.P @ H.T @ jnp.linalg.inv(S)

            # Update state and covariance
            self.state.x = self.state.x + K @ y
            self.state.P = (jnp.eye(self.state.P.shape[0]) - K @ H) @ self.state.P


    def update(
        self,
        dt: float,
        Q: jnp.ndarray,
        measurements: Dict[jnp.ndarray, Tuple[Type[MeasurementModel], jnp.ndarray]] = None
    ) -> None:
        """
        Perform a full Kalman Filter update: time prediction + measurement update.

        Args:
            dt: Time step.
            Q: Process noise covariance.
            measurements: Optional dictionary of measurements to update the state.
        """
        self._time_update(dt, Q)
        if measurements:
            self._measurement_update(measurements)
