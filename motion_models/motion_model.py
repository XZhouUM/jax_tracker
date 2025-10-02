from abc import ABC, abstractmethod
import jax.numpy as jnp


class MotionModel(ABC):
    """Abstract base class for motion models.

    For object tracking problems, the motion model is usually constructed by assuming a
    constant moving state (velocity, acceleration, angular velocity, etc.) of the moving
    object at each time step. In this case, there is no extrogenous input. The key
    components for tracking are the transition function and the Jacobian of the transition
    function.
    """

    def __init__(self):
        """Initialize motion model.

        Args:
            dt: time step
        """
        self.name = self.__class__.__name__

    @staticmethod
    @abstractmethod
    def transition(state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Propagate state forward in time."""
        pass

    @staticmethod
    @abstractmethod
    def jacobian(state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """Jacobian of transition wrt state, needed for Kalman filter."""
        pass
