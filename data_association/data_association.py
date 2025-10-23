import abc
from typing import Dict, List, Tuple, Type

import jax.numpy as jnp

from measurement_models.measurement_model import MeasurementModel


class DataAssociation(abc.ABC):
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def associate(
        self,
        track_states: List[jnp.ndarray],
        measurements: List[Tuple[jnp.ndarray, Type[MeasurementModel], jnp.ndarray]],
    ) -> Dict[int, int]:
        """
        Associate measurements to tracks.

        Args:
            track_states: List of track state vectors
            measurements: List of (measurement, measurement_model, noise_covariance) tuples

        Returns:
            Dictionary mapping track_index -> measurement_index
        """
        pass
