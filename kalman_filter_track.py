import jax.numpy as jnp
from typing import NamedTuple, Dict, Type, Tuple
from motion_models.motion_model import MotionModel
from measurement_models.measurement_model import MeasurementModel


class KalmanFilterTrackState(NamedTuple):
    x: jnp.ndarray  # state vector
    P: jnp.ndarray  # covariance matrix

    @classmethod
    def create(cls, x: jnp.ndarray, P: jnp.ndarray) -> 'KalmanFilterTrackState':
        """Create a new KalmanFilterTrackerState instance."""
        if x.shape[0] != P.shape[0] or x.shape[0] != P.shape[1]:
            raise ValueError(f"Incompatible dimensions: x {x.shape}, P {P.shape}. " 
                             "Expected x.shape[0] == P.shape[0] == P.shape[1].")
        return cls(x, P)


class KalmanFilterTrack:
    def __init__(self,
                 initial_state: KalmanFilterTrackState,
                 motion_model_class: Type[MotionModel]) -> None:
        """
        Initialize the KalmanFilterTracker.

        Args:
            initial_state: The initial state of the tracker.
            motion_model_class: The motion model class to use.
        """
        self.motion_model = motion_model_class()
    
        self.state = initial_state
   
    def _time_update(self, dt: float, Q: jnp.ndarray) -> None:
        """
        Predict the state at the next time step.

        Args:
            dt: The time step.
            Q: The process noise covariance matrix.
        """
        self.state.x = self.motion_model.predict(dt)
        self.state.P = self.motion_model.jacobian() @ self.state.P @ self.motion_model.jacobian().T + Q

    def _measurement_update(self,
                            measurements: Dict[jnp.ndarray, Tuple[MeasurementModel, jnp.ndarray]]) -> None:
        """
        Update the state based on the measurements.

        The measurements can be empty. In case of empty, no measurement update is performed. Therefore,
        the track is coasting with only the motion prediction.

        Args:
            measurements: A dictionary between the measurement, and its measurement
                model and measurement noise covariance matrix.
        """
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


    def update(self, dt: float, measurements: dict) -> None:
        """
        Update the state of the tracker.

        Args:
            dt: The time step.
            measurements: A dictionary between the measurement, and its measurement
                model and measurement noise covariance matrix.
        """
        self._time_update(self, dt)
        self._measurement_update(self, measurements)
