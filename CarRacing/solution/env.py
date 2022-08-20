import numpy as np

from gym.envs.box2d.car_racing import CarRacing
from gym.spaces import Discrete

class DiscreteCarRacing(CarRacing):
    actions = np.array([
        [-1, 0, 0],
        [-1, 0, 0.2],
        [-1, 1, 0],
        [0, 0, 0],
        [0, 0, 0.2],
        [0, 1, 0],
        [1, 0, 0],
        [1, 0, 0.2],
        [1, 1, 0],
    ])

    def __init__(self, verbose=1, lap_complete_percent=0.95):
        super().__init__(verbose, lap_complete_percent)
        self.action_space = Discrete(len(self.actions))

    def step(self, action: int):
        if action is None:
            return super().step(None)
        return super().step(self.actions[action])