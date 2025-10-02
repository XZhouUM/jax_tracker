from abc import ABC, abstractmethod
import jax.numpy as jnp


class MeasurementModel(ABC):
    """
    Abstract base class for measurement models.
    """

    def __init__(self):
        self.name = self.__class__.__name__

    @staticmethod
    @abstractmethod
    def predict_measurement(state: jnp.ndarray) -> jnp.ndarray:
        """
        Map state to expected measurement (h(x)).
        """
        pass

    @staticmethod
    @abstractmethod
    def jacobian(state: jnp.ndarray) -> jnp.ndarray:
        """
        Jacobian of measurement function wrt state, needed for Kalman filter.
        """
        pass
