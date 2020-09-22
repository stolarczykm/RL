import abc
from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np


class KArmedBandit():
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


class Policy(abc.ABC):
    @abc.abstractmethod
    def choose_action(self, t: int):
        pass

    @abc.abstractmethod
    def update_state(self, t, action, reward):
        pass

    def run(self, bandit: KArmedBandit, iterations=1000):
        rewards = []
        for t in range(1, iterations + 1):
            action  = self.choose_action(t)
            reward = bandit(action)
            rewards.append(reward)
            self.update_state(t, action, reward)
        return rewards
    
    @property
    def name(self):
        return self.__class__.__name__


class SampleAverageValuePolicyMixin:
    def _init_arrays(self, space_size):
        self._rewards = np.zeros(space_size, dtype=np.float)
        self._counts = np.zeros(space_size, dtype=np.int)
        self._q = np.zeros(space_size, dtype=np.float)
    
    def _update_state(self, t, action, reward):
        self._rewards[action] += reward
        self._counts[action] += 1
        self._q[action] = self._rewards[action] / self._counts[action] 


class GreedyMethod(Policy, SampleAverageValuePolicyMixin):
    def __init__(self, space_size: int):
        self._init_arrays(space_size)

    def choose_action(self, t: int):
        return np.argmax(self._q)
    
    def update_state(self, t, action, reward):
        self._update_state(t, action, reward)


class EpsilonGreedyMethod(Policy, SampleAverageValuePolicyMixin):
    def __init__(self, space_size: int, epsilon: float, seed: int = 123):
        self._space_size = space_size
        self._epsilon = epsilon
        self._random_state = np.random.RandomState(seed)
        self._init_arrays(space_size)

    def choose_action(self, t):
        u = self._random_state.uniform()
        return np.argmax(self._q) if u > self._epsilon else self._random_state.randint(self._space_size)
    
    def update_state(self, t, action, reward):
        self._update_state(t, action, reward)


class EpsilonGreedyMethodWithDecay(Policy, SampleAverageValuePolicyMixin):
    def __init__(self, space_size: int, epsilon: float, epsilon_decay: float, seed: int = 123):
        self._space_size = space_size
        self._epsilon = epsilon
        self._random_state = np.random.RandomState(seed)
        self._epsilon_decay = epsilon_decay
        self._init_arrays(space_size)

    def choose_action(self, t: int):
        u = self._random_state.uniform()
        res =  np.argmax(self._q) if u > self._epsilon else self._random_state.randint(self._space_size)
        self._epsilon *= self._epsilon_decay
        return res
    
    def update_state(self, t, action, reward):
        self._update_state(t, action, reward)
    

class UpperConfidenceBoundPolicy(Policy, SampleAverageValuePolicyMixin):
    def __init__(self, space_size: int, c: float, seed: int = 123):
        self._space_size = space_size
        self._c = c
        self._init_arrays(space_size)

    def choose_action(self, t: int):
        if t < 1:
            raise ValueError("t can't be smaller than 1")
        values = self._q + self._c * np.sqrt(np.log(t) / (self._counts + 1e-6))
        return np.argmax(np.where(self._counts == 0, np.inf, values))
    
    def update_state(self, t, action, reward):
        self._update_state(t, action, reward)


def softmax(x: np.ndarray):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
    

class GradientPolicy(Policy):
    def __init__(self, space_size: int, step_size: float, seed: int = 123):
        self._step_size = step_size
        self._space_size = space_size
        self._h = np.zeros(space_size, dtype=np.float)
        self._baseline: float = 0.0
        self._random_state = np.random.RandomState(seed)
    
    def choose_action(self, t: int):
        probs = softmax(self._h)
        return self._random_state.choice(len(probs), p=probs)

    def update_state(self, t, action, reward):
        signs = -np.ones(self._space_size, dtype=np.float)
        signs[action] = 1.0

        probs = softmax(self._h)
        probs[action] = 1 - probs[action]
        
        self._h = self._h + signs * self._step_size * (reward - self._baseline) * probs
        self._baseline = self._baseline + (1 / t) * (reward - self._baseline)


def plot_results(rewards, policies: List[Callable[[int], Policy]]):
    mean_rewards = np.array(rewards).mean(axis=0)

    plt.figure(figsize=(14, 8))
    for rewards, policy in zip(mean_rewards, policies): 
        plt.plot(rewards, alpha=0.4, label=policy(1).name)
    plt.xlabel("Iteration")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()


def run_simulations(space_size, policies, n_seeds, n_iterations):
    rewards = []
    for seed in range(n_seeds): 
        rewards.append([])
        for policy_func in policies:
            bandit = KArmedBandit(space_size, seed=seed)
            res = policy_func(seed).run(bandit, n_iterations)
            rewards[-1].append(res)
    return rewards

def main():
    space_size = 10
    policies: List[Callable[[int], Policy]] = [
        lambda seed: GreedyMethod(space_size),
        lambda seed: EpsilonGreedyMethod(space_size, 0.05, seed=seed),
        lambda seed: EpsilonGreedyMethodWithDecay(space_size, 0.1, 0.998, seed=seed),
        lambda seed: UpperConfidenceBoundPolicy(space_size, 1.0, seed=seed),
        lambda seed: GradientPolicy(space_size, 0.1, seed=seed),
    ]
    rewards = run_simulations(space_size, policies, 100, 1000)
    plot_results(rewards, policies)

if __name__ == "__main__":
    main()



