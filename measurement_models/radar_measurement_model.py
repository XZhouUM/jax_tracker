from measurement_model import MeasurementModel
import jax.numpy as jnp


class RadarMeasurement(MeasurementModel):
    """
    Radar measurement model.

    state = [x, y, vx, vy], measurement = [range, range_rate, azimuth]
    """
    def __init__(self):
        super().__init__()

    def predict_measurement(self, state: jnp.ndarray) -> jnp.ndarray:
        x, y, vx, vy = state
        range = jnp.sqrt(x**2 + y**2)
        range_rate = (x * vx + y * vy) / range
        azimuth = jnp.arctan2(y, x)
        return jnp.array([range, range_rate, azimuth])

    def jacobian(self, state: jnp.ndarray) -> jnp.ndarray:
        """
        Jacobian of the measurement model.
        """
        x, y, vx, vy = state
        # Range and range squared.
        r_sq = x**2 + y**2
        r = jnp.sqrt(x**2 + y**2)
        
        H = jnp.zeros((3, 4))
        
        # d_range/d_state
        H = H.at[0, 0].set(x / r)
        H = H.at[0, 1].set(y / r)
        
        # d_range_rate/d_state
        H = H.at[1, 0].set(vx / r - x * (x*vx + y*vy) / r**3)
        H = H.at[1, 1].set(vy / r - y * (x*vx + y*vy) / r**3)
        H = H.at[1, 2].set(x / r)
        H = H.at[1, 3].set(y / r)
        
        # d_azimuth/d_state
        H = H.at[2, 0].set(-y / r_sq)
        H = H.at[2, 1].set(x / r_sq)
        
        return H
