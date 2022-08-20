import abc
from turtle import forward

import numpy as np
import torch
from gym import Space
from torch import nn, optim
from torch.distributions import MultivariateNormal



class Agent(abc.ABC):
    subclasses = {}

    def __init__(self, action_space, observation_space) -> None:
        super().__init__()
        self.action_space: Space = action_space
        self.observation_space: Space = observation_space

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.subclasses[cls.__name__] = cls

    @classmethod
    def from_config(cls, config, action_space, observation_space) -> "Agent":
        agent_type = config.pop("__type__")
        klass = cls.subclasses[agent_type]
        return klass(action_space, observation_space, **config)

    @abc.abstractmethod
    def agent_start(self, state: np.ndarray):
        pass

    @abc.abstractmethod
    def agent_step(self, state: np.ndarray, reward: float):
        pass

    @abc.abstractmethod
    def agent_end(self, reward: np.ndarray) -> None:
        pass


class RandomAgent(Agent):
    def agent_start(self, state: np.ndarray):
        return self.action_space.sample()

    def agent_step(self, state: np.ndarray, reward: float):
        return self.action_space.sample()
    
    def agent_end(self, reward: np.ndarray) -> None:
        pass


class DoubleLinear(nn.Module):
    def __init__(self, linear1: nn.Linear, linear2: nn.Linear) -> None:
        super().__init__()
        self._linear1 = linear1
        self._linear2 = linear2
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self._linear1(x), self._linear2(x)


class ReinforceAgent(Agent):
    def __init__(
        self, 
        action_space: Space, 
        observation_space: Space, 
        hidden_sizes: list[int] = [60, 60],
        gamma: float = 0.99,
    ) -> None:
        super().__init__(action_space, observation_space)
        self._n_outputs = self.action_space.shape[0]
        self._n_inputs = self.observation_space.shape[0]
        self._gamma = gamma
        self._policy_network = self._create_network(hidden_sizes)
        self._policy_network.eval()
        self._optimizer = optim.SGD(self._policy_network.parameters(), lr=1e-2)
        self._reset_trajectory()

    def agent_start(self, state: np.ndarray):
        action = self._compute_action(state)
        self._update_trajectory(state, action)
        return action

    def agent_step(self, state: np.ndarray, reward: float):
        action = self._compute_action(state)
        self._update_trajectory(state, action, reward)
        return action
    
    def agent_end(self, reward: np.ndarray) -> None:
        self._update_trajectory(reward=reward)
        self._update_policy()

    def _create_network(self, hidden_sizes: list[int]) -> nn.Module:
        sizes = (self._n_inputs, *hidden_sizes)
        layers = []
        for size_in, size_out in zip(sizes, sizes[1:]): 
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.Tanh())
        layers.append(
            DoubleLinear(
                nn.Linear(hidden_sizes[-1], self._n_outputs),
                nn.Linear(hidden_sizes[-1], self._n_outputs)
            )
        )
        return nn.Sequential(*layers)

    def _compute_action(self, state: np.ndarray):
        state_tensor = torch.from_numpy(state).type(torch.float32)
        mean, precision = self._policy_network(state_tensor)
        mean = mean.detach()
        precision = precision.detach()
        precision = torch.clamp(precision, min=-10.0, max=10.0)
        distribution = MultivariateNormal(mean, covariance_matrix=torch.diag(torch.exp(precision)))
        action = distribution.sample()
        return action.numpy()

    def _update_policy(self):
        returns = self._compute_returns(self._rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        states_tesnor = torch.from_numpy(np.stack(self._states)).type(torch.float32)
        actions_tensor = torch.from_numpy(np.stack(self._actions)).type(torch.float32)

        self._policy_network.train()
        self._optimizer.zero_grad()

        action_means, action_precisions = self._policy_network(states_tesnor)
        action_precisions = torch.clamp(action_precisions, min=-10.0, max=10.0)
        action_distribution = MultivariateNormal(
            action_means, 
            covariance_matrix=torch.diag_embed(torch.exp(action_precisions)),
        )
        action_log_probs = action_distribution.log_prob(actions_tensor)
        loss = -(returns_tensor * action_log_probs).mean()
        loss.backward()

        self._optimizer.step()
        self._policy_network.eval()
        self._reset_trajectory()

    def _update_trajectory(self, state=None, action=None, reward=None):
        if state is not None:
            self._states.append(state)
        if action is not None:
            self._actions.append(action)
        if reward is not None:
            self._rewards.append(reward)
    
    def _reset_trajectory(self):
        self._states = []
        self._actions = []
        self._rewards = []


    def _compute_returns(self, rewards: list[float]) -> list[float]:
        returns = []
        g = 0.0
        for reward in reversed(rewards):
            g = self._gamma * g + reward
            returns.append(g)
        return returns[::-1]









