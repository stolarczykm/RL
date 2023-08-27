from typing import Callable, List

import matplotlib.pyplot as plt
import numpy as np

from bandits import (
    Bandit,
    KArmedBandit,
    NonStationaryKArmedBandit
)
from policies import (
    Policy,
    GreedyMethod,
    EpsilonGreedyMethodWithDecay,
    EpsilonGreedyMethod,
    UpperConfidenceBoundPolicy,
    GradientPolicy,
    NonstationaryEpsilonGreedyMethod
)


def plot_results(rewards, policies: List[Callable[[int], Policy]]):
    mean_rewards = np.array(rewards).mean(axis=0)

    plt.figure(figsize=(14, 8))
    for rewards, policy in zip(mean_rewards, policies): 
        plt.plot(rewards, alpha=0.4, label=policy(1).name)
    plt.xlabel("Iteration")
    plt.ylabel("Average reward")
    plt.legend()
    plt.show()


def run_simulations(space_size, policies, bandit_func, n_seeds, n_iterations):
    rewards = []
    for seed in range(n_seeds): 
        rewards.append([])
        for policy_func in policies:
            bandit = bandit_func(space_size, seed)
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
    bandit = lambda space_size, seed: KArmedBandit(space_size, seed=seed)
    rewards = run_simulations(space_size, policies, bandit, 100, 500)
    plot_results(rewards, policies)



if __name__ == "__main__":
    main()