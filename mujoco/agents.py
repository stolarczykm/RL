import abc

import numpy as np
import torch
from gym import Space
from torch import nn, optim, Tensor
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
        hidden_sizes: list[int] = [60, 80, 60],
        gamma: float = .99,
    ) -> None:
        super().__init__(action_space, observation_space)
        self._n_outputs = self.action_space.shape[0]
        self._n_inputs = self.observation_space.shape[0]
        self._gamma = gamma
        self._policy_network = self._create_actor_network(hidden_sizes)
        self._baseline_network = self._create_baseline_network(hidden_sizes)
        self._policy_network.eval()
        self._baseline_network.eval()
        self._policy_optimizer = optim.SGD(self._policy_network.parameters(), lr=2e-4, momentum=0.9)
        self._baseline_optimizer = optim.SGD(self._baseline_network.parameters(), lr=2e-4, momentum=0.9)
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
        self._update_networks()
        self._reset_trajectory()

    def _create_actor_network(self, hidden_sizes: list[int]) -> nn.Module:
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

    def _create_baseline_network(self, hidden_sizes: list[int]) -> nn.Module:
        sizes = (self._n_inputs, *hidden_sizes)
        layers = []
        for size_in, size_out in zip(sizes, sizes[1:]): 
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(sizes[-1], 1))
        return nn.Sequential(*layers)

    def _compute_action(self, state: np.ndarray):
        state_tensor = torch.from_numpy(state).type(torch.float32)
        mean, precision = self._policy_network(state_tensor)
        mean = mean.detach()
        precision = precision.detach()
        precision = torch.clamp(precision, min=-10.0, max=10.0)
        distribution = MultivariateNormal(mean, covariance_matrix=torch.diag(1/torch.exp(precision)))
        action = distribution.sample()
        return action.numpy()

    def _update_networks(self):
        returns = self._compute_returns(self._rewards)
        returns_tensor = torch.tensor(returns, dtype=torch.float32)
        states_tesnor = torch.from_numpy(np.stack(self._states)).type(torch.float32)
        actions_tensor = torch.from_numpy(np.stack(self._actions)).type(torch.float32)

        self._policy_network_update(returns_tensor, states_tesnor, actions_tensor)
        self._baseline_network_update(returns_tensor, states_tesnor)

    def _baseline_network_update(self, returns: Tensor, states: Tensor):
        self._baseline_network.train()
        self._baseline_optimizer.zero_grad()

        with torch.no_grad():
            targets = returns - self._baseline_network(states)

        loss = -(targets * self._baseline_network(states)).mean()
        loss.backward()

        self._baseline_optimizer.step()
        self._baseline_network.eval()
    
    def _policy_network_update(self, returns: Tensor, states: Tensor, actions: Tensor):
        self._policy_network.train()
        self._policy_optimizer.zero_grad()

        with torch.no_grad():
            baseline = self._baseline_network(states)
        
        targets = returns - baseline
        action_means, action_precisions = self._policy_network(states)
        action_precisions = torch.clamp(action_precisions, min=-10.0, max=10.0)
        action_distribution = MultivariateNormal(
            action_means, 
            covariance_matrix=torch.diag_embed(1/torch.exp(action_precisions)),
        )
        action_log_probs = action_distribution.log_prob(actions)
        gamma_powers = self._gamma_powers(len(targets))
        loss = -(gamma_powers * targets * action_log_probs).mean()
        loss.backward()

        self._policy_optimizer.step()
        self._policy_network.eval()

    def _gamma_powers(self, n: int) -> torch.Tensor:
        powers = []
        power = 1.0
        for _ in range(n):
            powers.append(power)
            power *= self._gamma
        return torch.Tensor(powers)

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


class ActorCriticAgent(Agent):
    def __init__(
        self, 
        action_space: Space, 
        observation_space: Space, 
        hidden_sizes: list[int] = [100, 100, 60],
        gamma: float = .99,
        update_frequency= 128,
    ) -> None:
        super().__init__(action_space, observation_space)
        self._n_outputs = self.action_space.shape[0]
        self._n_inputs = self.observation_space.shape[0]
        self._gamma = gamma
        self._actor_network = self._create_actor_network(hidden_sizes)
        self._critic_network = self._create_critic_network(hidden_sizes)
        self._actor_network.eval()
        self._critic_network.eval()
        self._actor_optimizer = optim.SGD(self._actor_network.parameters(), lr=5e-4, momentum=0.8)
        self._critic_optimizer = optim.SGD(self._critic_network.parameters(), lr=5e-4, momentum=0.8)
        self._reset_trajectory()
        self._last_state = None
        self._last_action = None
        self._steps = 0
        self._update_frequency = update_frequency

    def agent_start(self, state: np.ndarray):
        action = self._compute_action(state)
        self._last_action = action
        self._last_state = state
        return action

    def agent_step(self, state: np.ndarray, reward: float):
        action = self._compute_action(state)
        self._update_trajectory(self._last_state, self._last_action, reward, False, state)
        if (self._steps + 1) % self._update_frequency == 0:
            self._update_networks()
            self._reset_trajectory()
        self._last_action = action
        self._last_state = state
        self._steps += 1
        return action
    
    def agent_end(self, reward: np.ndarray) -> None:
        self._update_trajectory(self._last_state, self._last_action, reward, True, self._last_state)
        if (self._steps + 1) % self._update_frequency == 0:
            self._update_networks()
            self._reset_trajectory()
        self._steps += 1

    def _create_actor_network(self, hidden_sizes: list[int]) -> nn.Module:
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

    def _create_critic_network(self, hidden_sizes: list[int]) -> nn.Module:
        sizes = (self._n_inputs, *hidden_sizes)
        layers = []
        for size_in, size_out in zip(sizes, sizes[1:]): 
            layers.append(nn.Linear(size_in, size_out))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(sizes[-1], 1))
        return nn.Sequential(*layers)

    def _compute_action(self, state: np.ndarray):
        state_tensor = torch.from_numpy(state).type(torch.float32)
        mean, precision = self._actor_network(state_tensor)
        mean = mean.detach()
        precision = precision.detach()
        precision = torch.clamp(precision, min=-10.0, max=10.0)
        distribution = MultivariateNormal(mean, covariance_matrix=torch.diag(1/torch.exp(precision)))
        action = distribution.sample()
        return action.numpy()

    def _update_networks(self):
        rewards = torch.tensor(self._rewards, dtype=torch.float32)
        states = torch.from_numpy(np.stack(self._states)).type(torch.float32)
        actions = torch.from_numpy(np.stack(self._actions)).type(torch.float32)
        next_states = torch.from_numpy(np.stack(self._next_states)).type(torch.float32)
        done = torch.tensor(self._done, dtype=torch.float32)

        self._actor_network_update(rewards, states, actions, done, next_states)
        self._critic_network_update(rewards, states, actions, done, next_states)

    def _critic_network_update(
        self, rewards: Tensor, states: Tensor, actions: Tensor, done: Tensor, next_states: Tensor
    ):
        self._critic_network.train()
        self._critic_optimizer.zero_grad()

        with torch.no_grad():
            next_states_values = self._critic_network(next_states)

        prediction = self._critic_network(states)
        target = rewards + (1-done) * self._gamma * next_states_values
        loss = torch.square(target - prediction).mean()
        loss.backward()

        self._critic_optimizer.step()
        self._critic_network.eval()
    
    def _actor_network_update(self, rewards: Tensor, states: Tensor, actions: Tensor, done: Tensor, next_states: Tensor):
        self._actor_network.train()
        self._actor_optimizer.zero_grad()

        with torch.no_grad():
            states_values = self._critic_network(states)
            next_states_values = self._critic_network(next_states)
        
        advantage = rewards + (1-done) * self._gamma * next_states_values - states_values
        action_means, action_precisions = self._actor_network(states)
        action_precisions = torch.clamp(action_precisions, min=-10.0, max=10.0)
        action_distribution = MultivariateNormal(
            action_means, 
            covariance_matrix=torch.diag_embed(1/torch.exp(action_precisions)),
        )
        action_log_probs = action_distribution.log_prob(actions)
        loss = -(advantage * action_log_probs).mean()
        loss.backward()

        self._actor_optimizer.step()
        self._actor_network.eval()

    def _update_trajectory(
            self, 
            state: np.ndarray, 
            action: np.ndarray, 
            reward: float, 
            done: bool, 
            next_state: np.ndarray,
        ):
        self._states.append(state)
        self._actions.append(action)
        self._rewards.append(reward)
        self._done.append(done)
        self._next_states.append(next_state)
    
    def _reset_trajectory(self):
        self._states = []
        self._actions = []
        self._rewards = []
        self._done = []
        self._next_states = []
