from abc import ABC
import numpy as np


class IOptimizer(ABC):
    def update(self, params: list[np.ndarray], grads: list[np.ndarray]) -> None:
        pass
