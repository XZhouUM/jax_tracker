import jax.numpy as jnp
from typing import NamedTuple, Dict, Any, Type
from data_association.data_association import DataAssociation
from motion_models.motion_model import MotionModel
from measurement_models.measurement_model import MeasurementModel


class KalmanFilterTrackerState(NamedTuple):
    x: jnp.ndarray  # state vector
    P: jnp.ndarray  # covariance matrix

    @classmethod
    def create(cls, x: jnp.ndarray, P: jnp.ndarray) -> 'KalmanFilterTrackerState':
        """Create a new KalmanFilterTrackerState instance."""
        if x.shape[0] != P.shape[0] or x.shape[0] != P.shape[1]:
            raise ValueError(f"Incompatible dimensions: x {x.shape}, P {P.shape}. " 
                             "Expected x.shape[0] == P.shape[0] == P.shape[1].")
        return cls(x, P)


class KalmanFilterTracker:
    def __init__(self,
                 initial_state: KalmanFilterTrackerState,
                 data_association_class: Type(DataAssociation),
                 data_association_params: Dict{str, Any},
                 motion_model_class: Type(MotionModel),
                 measurement_model_class: Type(MeasurementModel)) -> None:
        """
        Initialize the KalmanFilterTracker.

        Args:
            initial_state: The initial state of the tracker.
            data_association_class: The data association class to use.
            data_association_params: The parameters for the data association class.
            motion_model_class: The motion model class to use.
            measurement_model_class: The measurement model class to use.
        """
        self.data_association = data_association_class(**data_association_params)
        self.motion_model = motion_model_class()
        self.measurement_model = measurement_model_class()
    
        self.state = initial_state

    def _associate(self, state: jnp.ndarray, measurements: jnp.ndarray) -> jnp.ndarray:
        associated_data: jnp.ndarray = None
        for i, measurement in enumerate(measurements):
            if self.data_association.associate(state, measurement):
                if associated_data is None:
                    associated_data = jnp.ndarray([measurement])
                else:
                    associated_data = jnp.concatenate((associated_data, [measurement]))
        return associated_data
   
    def _time_update(self, dt: float) -> None:
        """
        Predict the state at the next time step.
        """
        self.state.x = self.motion_model.predict(dt) 

    def _measurement_update(self, state: jnp.ndarray, measurements: dict) -> None:
        """
        Update the state based on the measurements.
        """
        # associate the measurements to the track.
        associated_data = self._associate(self, state, measurements)

        if associated_data is not None:
            self.state.x = self.measurement_model.update(state, associated_data)

    def update(self, state: jnp.ndarray, dt: float, measurements: dict) -> None:
        self._time_update(self, dt)
        self._measurement_update(self, state, measurements)
