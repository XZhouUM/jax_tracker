from measurement_models.measurement_model import MeasurementModel
import jax.numpy as jnp


class PositionMeasurement(MeasurementModel):
    """
    Position measurement model for direct Cartesian position measurements.

    Some example use cases:
    - Computer vision systems that directly output object positions
    - GPS measurements (if used in a local coordinate frame)
    
    state = [x, y, vx, vy], measurement = [x, y]
    """
    
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def predict_measurement(state: jnp.ndarray) -> jnp.ndarray:
        """
        Predict measurement from state.
        
        Args:
            state: State vector [x, y, vx, vy]
            
        Returns:
            Predicted measurement [x, y]
        """
        return state[:2]  # Return only position components
    
    @staticmethod
    def jacobian(state: jnp.ndarray) -> jnp.ndarray:
        """
        Jacobian of the measurement model.
        
        Args:
            state: State vector [x, y, vx, vy]
            
        Returns:
            Jacobian matrix H (2x4)
        """
        # Measurement is linear in position, zero for velocity
        H = jnp.array([
            [1.0, 0.0, 0.0, 0.0],  # dx/d[x,y,vx,vy]
            [0.0, 1.0, 0.0, 0.0]   # dy/d[x,y,vx,vy]
        ])
        return H
