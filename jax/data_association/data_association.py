import abc
import jax.numpy as jnp

class DataAssociation(abc.ABC):
    def __init__(self) -> None:
        self.name = self.__class__.__name__

    @abc.abstractmethod
    def associate(self, state: jnp.ndarray, measurements: dict) -> bool:
        pass
