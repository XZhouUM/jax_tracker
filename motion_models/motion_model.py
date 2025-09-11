from abc import ABC, abstractmethod
import jax.numpy as jnp


class MotionModel(ABC):
    """Abstract base class for motion models.

    For object tracking problem, the motion model is usually constructed by assuming a 
    constant moving state (velocity, acceleration, angular velocity, etc.) of the moving
    object at each time step. In this case, there is no extrogenous input. The key
    componentsfor tracking are the transition function and the Jacobian of the transition
    function.
    """

    def __init__(self, dt: float):
        """Initialize motion model.

        Args:
            dt: time step
        """
        self.dt = dt
        self.name = self.__class__.__name__

    @abstractmethod
    def transition(self, state: jnp.ndarray) -> jnp.ndarray:
        """Propagate state forward in time."""
        pass

    @abstractmethod
    def jacobian(self, state: jnp.ndarray) -> jnp.ndarray:
        """Jacobian of transition wrt state (for EKF)."""
        pass