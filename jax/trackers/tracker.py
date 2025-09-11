import jax.numpy as jnp
from typing import NamedTuple, Dict, Any, Type
from data_association.data_association import DataAssociation

class TrackerState(NamedTuple):
    x: jnp.ndarray
    P: jnp.ndarray

class Tracker:
    def __init__(self,
                 data_association: Dict[Type(DataAssociation), Dict{str, Any}],
                 motion_model: Dict[Type(MotionModel), Dict{str, Any}],
                 filter: Dict[Type(Filter), Dict{str, Any}]) -> None:
        self.data_association = data_association[0](**data_association[1])
        self.motion_model = motion_model[0](**motion_model[1])
        self.filter = filter[0](**filter[1])

    def time_update(self, state: jnp.ndarray) -> jnp.ndarray:
        return self.motion_model.predict(state)

    def data_association(self, state: jnp.ndarray, measurements: jnp.ndarray) -> jnp.ndarray:
        associated_data: jnp.ndarray = None
        for i, measurement in enumerate(measurements):
            if self.data_association.associate(state, measurement):
                if associated_data is None:
                    associated_data = jnp.ndarray([measurement])
                else:
                    associated_data = jnp.concatenate((associated_data, [measurement]))
        return associated_data 

    def measurement_update(self, state: jnp.ndarray, measurements: dict) -> jnp.ndarray:
        pass

    def update(self, state: jnp.ndarray, measurements: dict) -> jnp.ndarray:
        state = self.time_update(state)
        measurements = self.data_association(state, measurements)
        return self.measurement_update(state, measurements)
