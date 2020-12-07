import abc
import numpy as np

class Bandit(abc.ABC):
    @abc.abstractmethod
    def __call__(self, action: int) -> int:
        pass

    @abc.abstractproperty
    def k(self) -> int:
        pass



class KArmedBandit(Bandit):
    def __init__(self, k: int, seed: int = 123):
        self._k = k
        self._random_state = np.random.RandomState(seed=seed)
        self._means = self._random_state.normal(0, 1, size=k)

    def __call__(self, action: int):
        if not 0 <= action < self._k:
            raise ValueError(f"action is {action}, should be between 0 and {self._k - 1} (inclusive)")
        return self._random_state.normal(self._means[action], 1)

    @property
    def k(self):
        return self._k 


class NonStationaryKArmedBandit(Bandit):
    def __init__(self, k: int, seed: int = 123):
        self._k = k
        self._random_state = np.random.RandomState(seed=seed)
        self._means = np.zeros(shape=k, dtype=np.float32)

    def __call__(self, action: int):
        if not 0 <= action < self._k:
            raise ValueError(f"action is {action}, should be between 0 and {self._k - 1} (inclusive)")
        self._means += self._random_state.normal(size=self._means.shape)
        return self._random_state.normal(self._means[action], 1)

    @property
    def k(self):
        return self._k 
