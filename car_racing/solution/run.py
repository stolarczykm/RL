import os

import click
import gym
from gym.core import Env
from torch import as_tensor, ne

from solution.agent import Agent, RandomAgent
from solution.dqn import DQNAgent
from solution.env import DiscreteCarRacing


def run(
    env: Env, agent: DQNAgent, episodes: int = 1, render: bool = True
) -> list[float]:
    obs = env.reset()
    action = agent.start(obs)
    rewards = []
    episode_reward = 0.0
    steps, episode_counter = 0, 0

    while True:
        if render:
            env.render()
        
        obs, reward, done, _ = env.step(action)
        steps += 1
        episode_reward += reward
        action = agent.step(reward, done, obs)

        if not done:
            continue

        rewards.append(episode_reward)
        steps = 0
        episode_reward = 0.0
        episode_counter += 1
        if episode_counter >= episodes:
            break
        obs = env.reset()
        action = agent.start(obs)

    return rewards

@click.command()
@click.argument("checkpoint", type=click.Path(exists=True))
def main(checkpoint) -> None:
    env = DiscreteCarRacing() 
    agent = DQNAgent(
        env.observation_space.shape[:2] + (1,),
        env.action_space.n,
        epsilon=0.5,
        gamma=0.95,
    )
    agent.load_state(checkpoint)
    agent.set_eval(True)
    rewards = run(env, agent, episodes=1, render=True)


if __name__ == "__main__":
    main()
