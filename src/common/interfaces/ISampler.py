from abc import ABC
import numpy as np


class ISampler(ABC):
    def do_sample(self, word: np.ndarray) -> np.ndarray:
        pass
