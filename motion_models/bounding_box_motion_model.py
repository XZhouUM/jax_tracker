import jax.numpy as jnp

from motion_models.motion_model import MotionModel


class BoundingBoxConstantVelocity(MotionModel):
    """
    Constant velocity motion model for bounding box tracking.

    This model assumes that both the center position and the size of the
    bounding box change with constant velocities.

    State format: [cx, cy, w, h, vcx, vcy, vw, vh]
    - cx, cy: center coordinates
    - w, h: width and height
    - vcx, vcy: velocity of center
    - vw, vh: velocity of size (rate of change)

    This model is useful for tracking objects that may be:
    - Moving with constant velocity
    - Changing size at a constant rate (e.g., approaching/receding objects)
    - Undergoing scale changes due to perspective effects
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def _validate_state(state: jnp.ndarray) -> bool:
        """
        Validate the state length to be 8.

        The bounding box constant velocity model assumes the state is
        [cx, cy, w, h, vcx, vcy, vw, vh].

        Args:
            state: The state to validate

        Returns:
            True if the state is valid, False otherwise
        """
        return state.shape == (8,)

    @staticmethod
    def transition(state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Transition the state to the next time step using constant velocity model.

        Args:
            state: Current state [cx, cy, w, h, vcx, vcy, vw, vh]
            dt: Time step

        Returns:
            Next state [cx', cy', w', h', vcx, vcy, vw, vh]
        """
        assert BoundingBoxConstantVelocity._validate_state(state)

        cx, cy, w, h, vcx, vcy, vw, vh = state

        # Update positions and sizes with constant velocities
        new_cx = cx + vcx * dt
        new_cy = cy + vcy * dt
        new_w = w + vw * dt
        new_h = h + vh * dt

        # Ensure width and height remain positive
        new_w = jnp.maximum(new_w, 1.0)  # Minimum width of 1 pixel
        new_h = jnp.maximum(new_h, 1.0)  # Minimum height of 1 pixel

        return jnp.array([new_cx, new_cy, new_w, new_h, vcx, vcy, vw, vh])

    @staticmethod
    def jacobian(state: jnp.ndarray, dt: float) -> jnp.ndarray:
        """
        Calculate the Jacobian of the transition function.

        For the constant velocity model, the Jacobian is mostly identity
        with time step factors for the position-velocity coupling.

        Args:
            state: Current state [cx, cy, w, h, vcx, vcy, vw, vh]
            dt: Time step

        Returns:
            Jacobian matrix (8x8)
        """
        assert BoundingBoxConstantVelocity._validate_state(state)

        # For constant velocity model: x_k+1 = F * x_k
        # where F is the state transition matrix
        F = jnp.array(
            [
                [1, 0, 0, 0, dt, 0, 0, 0],  # cx' = cx + vcx*dt
                [0, 1, 0, 0, 0, dt, 0, 0],  # cy' = cy + vcy*dt
                [0, 0, 1, 0, 0, 0, dt, 0],  # w'  = w  + vw*dt
                [0, 0, 0, 1, 0, 0, 0, dt],  # h'  = h  + vh*dt
                [0, 0, 0, 0, 1, 0, 0, 0],  # vcx' = vcx
                [0, 0, 0, 0, 0, 1, 0, 0],  # vcy' = vcy
                [0, 0, 0, 0, 0, 0, 1, 0],  # vw'  = vw
                [0, 0, 0, 0, 0, 0, 0, 1],  # vh'  = vh
            ]
        )

        return F

    @staticmethod
    def process_noise_matrix(dt: float, noise_scale: float = 1.0) -> jnp.ndarray:
        """
        Generate process noise covariance matrix for the motion model.

        This assumes that the acceleration (change in velocity) is the
        source of uncertainty, following a white noise acceleration model.

        Args:
            dt: Time step
            noise_scale: Scale factor for process noise

        Returns:
            Process noise covariance matrix Q (8x8)
        """
        # White noise acceleration model
        # Assume independent noise for position and size dynamics

        dt2 = dt * dt
        dt3 = dt2 * dt / 2
        dt4 = dt3 * dt / 2

        # Noise covariance for one 2D component (position or size)
        q_block = jnp.array([[dt4, dt3], [dt3, dt2]])

        # Build full 8x8 matrix with 4 independent 2x2 blocks
        Q = jnp.zeros((8, 8))

        # Center position noise (cx, vcx) and (cy, vcy)
        Q = Q.at[0:2, 0:2].set(q_block)  # cx, vcx block
        Q = Q.at[4:6, 4:6].set(q_block)  # cy, vcy block (offset by 4)

        # Size noise (w, vw) and (h, vh)
        Q = Q.at[2:4, 2:4].set(q_block)  # w, vw block
        Q = Q.at[6:8, 6:8].set(q_block)  # h, vh block (offset by 4)

        return Q * noise_scale

    @staticmethod
    def create_initial_state(
        measurement: jnp.ndarray, initial_velocity: jnp.ndarray = None
    ) -> jnp.ndarray:
        """
        Create initial state from first measurement.

        Args:
            measurement: Initial measurement [cx, cy, w, h]
            initial_velocity: Initial velocity [vcx, vcy, vw, vh].
                            If None, assumes zero velocity.

        Returns:
            Initial state [cx, cy, w, h, vcx, vcy, vw, vh]
        """
        if initial_velocity is None:
            initial_velocity = jnp.zeros(4)

        return jnp.concatenate([measurement, initial_velocity])

    @staticmethod
    def create_initial_covariance(
        position_std: float = 10.0, size_std: float = 5.0, velocity_std: float = 1.0
    ) -> jnp.ndarray:
        """
        Create initial state covariance matrix.

        Args:
            position_std: Standard deviation for center position uncertainty
            size_std: Standard deviation for size uncertainty
            velocity_std: Standard deviation for velocity uncertainty

        Returns:
            Initial covariance matrix P0 (8x8)
        """
        P0 = jnp.diag(
            jnp.array(
                [
                    position_std**2,  # cx variance
                    position_std**2,  # cy variance
                    size_std**2,  # w variance
                    size_std**2,  # h variance
                    velocity_std**2,  # vcx variance
                    velocity_std**2,  # vcy variance
                    velocity_std**2,  # vw variance
                    velocity_std**2,  # vh variance
                ]
            )
        )

        return P0
