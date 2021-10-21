import abc
from typing import (
    Optional,
    Tuple,
    Union,
)

import numpy as np
from gym.spaces import Box, Discrete


class Agent(abc.ABC):
    subclasses = {}

    def __init__(self, action_space, observation_space) -> None:
        super().__init__()
        self.action_space = action_space
        self.observation_space = observation_space

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def from_config(cls, config, action_space, observation_space) -> "Agent":
        agent_type = config.pop("__type__")
        klass = cls.subclasses[agent_type]
        return klass(action_space, observation_space, **config)

    @abc.abstractmethod
    def agent_start(self, state: np.ndarray) -> Union[int, float]:
        pass

    @abc.abstractmethod
    def agent_step(self, state: np.ndarray, reward: float) -> Union[int, float]:
        pass

    @abc.abstractmethod
    def agent_end(self, reward: np.ndarray) -> None:
        pass



class SarsaAgent(Agent):
    def __init__(
        self, 
        action_space: Discrete,
        observation_space: Box,
        epsilon: float = 0.05,
        gamma: float = 0.9,
        alpha: float = 0.5,
        n_bins: int = 10,
        action_bins: Optional[int] = None,
        seed: int = 0,
    ): 
        super().__init__(action_space, observation_space)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha

        if not isinstance(observation_space, Box):
            raise TypeError("observation_space should be Box")

        if isinstance(action_space, Discrete):
            self.n_actions = action_space.n
            self.discrete_action = True
            self.action_high = None
            self.action_low = None
            self.actions = None
        elif isinstance(action_space, Box) and action_space.shape == (1,):
            assert action_bins is not None
            self.n_actions = action_bins 
            self.discrete_action = False
            self.action_high = action_space.high[0] 
            self.action_low = action_space.low[0] 
            action_bin_edges = np.linspace(self.action_low, self.action_high, action_bins + 1)
            self.actions = (action_bin_edges[1:] + action_bin_edges[:-1]) / 2

        else: 
            raise NotImplementedError(
                "SarsaAgent is implemented only for Box and Discrete action spaces"
            )

        self.state_mins = observation_space.low
        self.state_maxs = observation_space.high
        self.n_bins = n_bins
        self.n_state_features = n_bins ** len(self.state_mins) 

        self.weights = np.zeros((self.n_actions, self.n_state_features), dtype="float")
        self.last_state = None
        self.last_action = None
        self.last_action_value = None
        self.last_feature_ind = None

        self.rng = np.random.default_rng(seed)
    

    def _get_active_feature(self, state: np.ndarray) -> int:
        scaled_state = (state - self.state_mins) / (self.state_maxs - self.state_mins)
        scaled_state = np.minimum(scaled_state, 1.0 - 1 / (2 * self.n_bins))
        ind = (np.floor(scaled_state * self.n_bins) * self.n_bins ** np.arange(len(state))[::-1]).sum()
        return int(ind)
    
    def _argmax(self, values: np.ndarray):
        values = np.asarray(values)
        max_ = np.max(values)
        max_ind = np.where(values == max_)[0]
        return self.rng.choice(max_ind)


    def select_action(self, feature_ind) -> Tuple[float, float]:
        action_values = self.weights[:, feature_ind]

        if self.rng.random() < self.epsilon:
            action = self.rng.choice(self.n_actions)
        else:
            action = self._argmax(action_values)
        
        return action, action_values[action]

    def to_action_space(self, action):
        if self.discrete_action:
            return action
        return np.array([self.actions[action]])

    def agent_start(self, state):
        feature_ind = self._get_active_feature(state)
        action, value = self.select_action(feature_ind)
        self.last_action = action
        self.last_action_value = value
        self.last_feature_ind = feature_ind
        return self.to_action_space(action)

    def agent_step(self, reward, state):
        feature_ind = self._get_active_feature(state)
        action, action_value = self.select_action(feature_ind)

        self.weights[self.last_action, self.last_feature_ind] += self.alpha * (
            reward + self.gamma * action_value - self.last_action_value
        )  

        self.last_action = action
        self.last_action_value = action_value
        self.last_feature_ind = feature_ind

        return self.to_action_space(action)

    def agent_end(self, reward):
        self.weights[self.last_action, self.last_feature_ind] += self.alpha * (
            reward - self.last_action_value
        )
