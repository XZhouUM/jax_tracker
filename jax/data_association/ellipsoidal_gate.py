from data_association import DataAssociation
import jax.numpy as jnp

class EllipsoidalGate(DataAssociation):
    def __init__(self, gate_threshold: float) -> None:
        super().__init__()
        self.gate_threshold = gate_threshold
    
    def associate(self, state: jnp.ndarray, measurements: dict) -> bool:
        return jnp.linalg.norm(measurements - state.x) < self.gate_threshold

        
