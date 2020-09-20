import numpy as np
import matplotlib.pyplot as plt


class KArmedBandit():
    def __init__(self, k: int, seed: int = 123):
        self._k = k
        self._random_state = np.random.RandomState(seed=seed)
        self._means = self._random_state.normal(0, 1, size=k)

    def __call__(self, action: int):
        assert 0 <= action < self._k
        return self._random_state.normal(self._means[action], 1)

    @property
    def k(self):
        return self._k 



class GreedyMethod():
    def __init__(self, bandit: KArmedBandit):
        self._bandit = bandit 
        self._rewards = np.zeros(bandit.k, dtype=np.float)
        self._counts = np.zeros(bandit.k, dtype=np.int)
        self._q = np.zeros(bandit.k, dtype=np.float)

    def choose_action(self):
        return np.argmax(self._q)
    
    def update_state(self, action, reward):
        self._rewards[action] += reward
        self._counts[action] += 1
        self._q[action] = self._rewards[action] / self._counts[action] 
    
    def run(self, iterations=1000):
        rewards = []
        for i in range(iterations):
            action = self.choose_action()
            reward = self._bandit(action)
            rewards.append(reward)
            self.update_state(action, reward)
        return rewards


class EpsilonGreedyMethod():
    def __init__(self, bandit: KArmedBandit, epsilon: float, seed: int = 123):
        self._bandit = bandit 
        self._epsilon = epsilon
        self._random_state = np.random.RandomState(seed)
        self._rewards = np.zeros(bandit.k, dtype=np.float)
        self._counts = np.zeros(bandit.k, dtype=np.int)
        self._q = np.zeros(bandit.k, dtype=np.float)

    def choose_action(self):
        u = self._random_state.uniform()
        return np.argmax(self._q) if u > self._epsilon else self._random_state.randint(self._bandit.k)
    
    def update_state(self, action, reward):
        self._rewards[action] += reward
        self._counts[action] += 1
        self._q[action] = self._rewards[action] / self._counts[action] 
    
    def run(self, iterations=1000):
        rewards = []
        for i in range(iterations):
            action = self.choose_action()
            reward = self._bandit(action)
            rewards.append(reward)
            self.update_state(action, reward)
        return rewards


class EpsilonGreedyMethodWithDecay():
    def __init__(self, bandit: KArmedBandit, epsilon: float, epsilon_decay: float, seed: int = 123):
        self._bandit = bandit 
        self._epsilon = epsilon
        self._random_state = np.random.RandomState(seed)
        self._rewards = np.zeros(bandit.k, dtype=np.float)
        self._counts = np.zeros(bandit.k, dtype=np.int)
        self._q = np.zeros(bandit.k, dtype=np.float)
        self._epsilon_decay = epsilon_decay

    def choose_action(self):
        u = self._random_state.uniform()
        res =  np.argmax(self._q) if u > self._epsilon else self._random_state.randint(self._bandit.k)
        self._epsilon *= self._epsilon_decay
        return res
    
    def update_state(self, action, reward):
        self._rewards[action] += reward
        self._counts[action] += 1
        self._q[action] = self._rewards[action] / self._counts[action] 
    
    def run(self, iterations=1000):
        rewards = []
        for i in range(iterations):
            action = self.choose_action()
            reward = self._bandit(action)
            rewards.append(reward)
            self.update_state(action, reward)
        return rewards



if __name__ == "__main__":
    sums = []
    for seed in range(100): 
        bandit = KArmedBandit(10, seed=seed)
        greedy = GreedyMethod(bandit)
        rewards = greedy.run(1000)

        bandit = KArmedBandit(10, seed=seed)
        eps_greedy = EpsilonGreedyMethod(bandit, 0.05, seed=seed)
        rewards_eps = eps_greedy.run(1000)

        bandit = KArmedBandit(10, seed=seed)
        eps_greedy_2 = EpsilonGreedyMethodWithDecay(bandit, 0.1, 0.998, seed=seed)
        rewards_eps_2 = eps_greedy_2.run(1000)
        sums.append([sum(rewards), sum(rewards_eps), sum(rewards_eps_2)])
    
    print(np.mean(sums, axis=0))

