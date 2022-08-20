import random
from collections import deque
from re import S
from turtle import width
from typing import NamedTuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import RMSprop
from torch.nn.functional import mse_loss, one_hot
from torch.utils.tensorboard import SummaryWriter 

from solution.agent import Agent


class ReplayBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    rewards: Tensor
    done: Tensor
    next_states: Tensor


class ReplayBuffer:
    def __init__(self, max_buffer_size: int, device: str) -> None:
        self.device = device
        self.buffer = deque([], maxlen=max_buffer_size) 

    def add(self, state: np.ndarray, action: int, reward: float, done: bool, next_state: np.ndarray) -> None:
        state = torch.from_numpy(state[None]).to(self.device)
        next_state = torch.from_numpy(next_state[None]).to(self.device)
        action = torch.tensor([action], dtype=torch.long).to(self.device)
        done = torch.tensor([done], dtype=torch.float32).to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        self.buffer.append((state, action, reward, done, next_state))

    def sample(self, batch_size: int) -> ReplayBatch:
        sample = random.sample(self.buffer, batch_size)
        return ReplayBatch(*[torch.cat([s[i] for s in sample], dim=0) for i in range(5)])

    
    def __len__(self):
        return len(self.buffer)


class EpsilonSchedule:
    def __init__(self, initial_epsilon: float, final_epsilon: float, n_steps: int) -> None:
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.n_steps = n_steps

    def get_epsilon(self, step):
        if step > self.n_steps:
            return self.final_epsilon
        frac = step / self.n_steps
        return (1 - frac) * self.initial_epsilon + frac * self.final_epsilon


class QNetwork(nn.Module):
    def __init__(self, input_shape, output_dim) -> None:
        super().__init__()
        self.input_shape = input_shape
        self.output_dim = output_dim
        self.network = self._create_network(input_shape, output_dim)

    def _create_network(self, input_shape, output_dim) -> nn.Module:
        height, width, channels = input_shape
        return nn.Sequential(
            nn.Conv2d(channels, 8, (7, 7), stride=(3, 3), padding=(3, 3)),
            nn.ReLU(),
            nn.Conv2d(8, 16, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(16, 16, (3, 3), stride=(2, 2), padding=(1, 1)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear((height // 12) * (width // 12) * 16, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        if len(state.shape) == 3:
            state = torch.unsqueeze(state, 0)
        state = state.permute(0, 3, 1, 2)
        return self.network.forward(state)


class DQNAgent(Agent):
    def __init__(
        self,
        obs_shape: tuple[int],
        n_actions: int,
        epsilon: float,
        gamma: float,
        seed: int = 0,
    ) -> None:
        self.writer = SummaryWriter("../logs")
        self.n_actions = n_actions
        self.obs_shape = obs_shape
        self.epsilon = EpsilonSchedule(0.5, 0.05, 20000)
        self.gamma = gamma
        self.batch_size = 128

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.replay_buffer = ReplayBuffer(10000, self.device)
        self.rng = np.random.default_rng(seed)

        self.policy_network = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_network = QNetwork(obs_shape, n_actions).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = RMSprop(self.policy_network.parameters(), lr=0.001)

        self.update_target_frequency = 2000
        self.update_policy_frequency = 10 
        self.steps = 0
        self.last_state = None
        self.last_action = None
        self.eval = False

    def set_eval(self, eval: bool) -> None:
        self.eval = eval

    def start(self, obs: np.ndarray) -> int:
        obs = self._preprocess_observation(obs)
        action = self._select_action(obs)
        self.last_action = action
        self.last_state = obs
        return action
    
    def step(self, reward: float, done: bool, obs: np.ndarray) -> int:
        obs = self._preprocess_observation(obs)
        self.replay_buffer.add(self.last_state, self.last_action, reward, done, obs)
        action = self._select_action(obs)

        if not self.eval:
            self._update_policy_network()
            self._update_target_network()
        self.last_action = action
        self.last_state = obs
        self.steps += 1
        return action

    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        obs = obs.astype("float32") / 127.5 - 1
        return obs.mean(axis=-1, keepdims=True)
    
    def _select_action(self, obs):
        if not self.eval and self.rng.random() < self.epsilon.get_epsilon(self.steps):
            return self.rng.choice(self.n_actions)

        obs = torch.from_numpy(obs).to(self.device)
        q_values = self.policy_network(obs)
        return np.argmax(q_values.cpu().detach().numpy(), axis=1)[0]

    def _update_policy_network(self):
        if len(self.replay_buffer) < 2048:
            return
        if self.steps % self.update_policy_frequency != 0:
            return
        self.optimizer.zero_grad()
        self.policy_network.train()
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, done, next_states = batch

        target_predictions = self.target_network(next_states)
        target_action_values = target_predictions.max(dim=1).values

        target = rewards + (1 - done) * self.gamma * target_action_values
        policy_predictions = self.policy_network(states)
        actions_one_hot = one_hot(actions, self.n_actions)
        policy_action_values = (actions_one_hot * policy_predictions).sum(dim=1)

        loss = mse_loss(target, policy_action_values)
        loss.backward()
        self.optimizer.step()
        self.writer.add_scalar("mse_loss", loss.item(), self.steps)
        self.writer.add_scalar("epsilon", self.epsilon.get_epsilon(self.steps), self.steps)
        self.writer.add_scalar("avg_q_target", target_predictions.mean().item(), self.steps)
        self.writer.add_scalar("max_q_target", target_predictions.max().item(), self.steps)
        self.writer.add_scalar("avg_batch_reward", rewards.mean().item(), self.steps)

    def _update_target_network(self):
        if self.steps % self.update_target_frequency == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())

    def checkpoint(self, path):
        torch.save(self.policy_network.state_dict(), path)

    def load_state(self, path):
        state_dict = torch.load(path)
        self.policy_network.load_state_dict(state_dict)
        self.target_network.load_state_dict(state_dict)


