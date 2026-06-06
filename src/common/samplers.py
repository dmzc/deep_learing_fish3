from src.common.vocab import Vocab
from abc import ABC, abstractmethod
import numpy as np


class ISampler(ABC):
    _sample_size: int
    _power: float
    _vocab: Vocab

    def __init__(self, vocab: Vocab, sample_size=5, power=0.75):
        self._vocab = vocab
        self._power = power
        self._sample_size = sample_size

    @abstractmethod
    def do_sample(self, word: np.ndarray) -> np.ndarray:
        pass


class SimpleSampler(ISampler):
    __probability: np.ndarray

    def __init__(self, vocab, sample_size=5, power=0.75):
        super().__init__(vocab, sample_size, power)
        probability = self._vocab.get_probability()
        probability = np.power(probability, power)
        probability = probability / np.sum(probability)
        self.__probability = probability

    def do_sample(self, center_word: np.ndarray):
        batch_size = center_word.shape[0]
        samples = np.zeros((batch_size, self._sample_size), dtype=np.uint32)
        for index in range(batch_size):
            probability = self.__probability.copy()
            c_word = center_word[index]
            probability[c_word] = 0
            probability /= probability.sum()
            samples[index, :] = np.random.choice(
                len(probability), size=self._sample_size, replace=False, p=probability
            )
        return samples
