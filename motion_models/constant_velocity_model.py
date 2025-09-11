from motion_models.motion_model import MotionModel
import jax.numpy as jnp

class ConstantVelocity(MotionModel):
    """Constant velocity model

    Constant velocity model for 2D motion. The state is [x, y, vx, vy].
    """
    def __init__(self, dt: float = 1.0):
        super().__init__(dt)

    @staticmethod
    def _validate_state(state: jnp.ndarray) -> bool:
        """Validate the state length to be 4.

        The constant velocity model assumes the state is [x, y, vx, vy].

        Args:
            state (jnp.ndarray): The state to validate

        Returns:
            bool: True if the state is valid, False otherwise
        """
        return state.shape == (4,)
    
    def transition(self, state: jnp.ndarray) -> jnp.ndarray:
        """Transition the state to the next time step.

        Args:
            state (jnp.ndarray): The state to transition

        Returns:
            jnp.ndarray: The transitioned state
        """
        assert self._validate_state(state)

        x, y, vx, vy = state
        dt = self.dt
        return jnp.array([x + vx * dt, y + vy * dt, vx, vy])

    def jacobian(self, state: jnp.ndarray) -> jnp.ndarray:
        """Calculate the Jacobian of the transition function.

        Args:
            state (jnp.ndarray): The state to calculate the Jacobian

        Returns:
            jnp.ndarray: The Jacobian
        """
        assert self._validate_state(state)
        
        dt = self.dt
        return jnp.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0 ],
            [0, 0, 0, 1 ]
        ])