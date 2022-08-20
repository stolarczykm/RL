import os

import gym
from gym.core import Env
from torch import as_tensor, ne

from solution.agent import Agent, RandomAgent
from solution.dqn import DQNAgent
from solution.env import DiscreteCarRacing


def run(
    env: Env, agent: DQNAgent, episodes: int = 1000, render: bool = False
) -> list[float]:
    obs = env.reset()
    action = agent.start(obs)
    rewards = []
    episode_counter = 0
    episode_reward = 0.0
    steps_per_episode = 100
    steps = 0
    negative_counter = 0

    while True:
        render = episode_counter % 10 == 0
        if render:
            env.render()
        
        obs, reward, done, _ = env.step(action)
        steps += 1
        if reward < 0:
            negative_counter += 1
        else:
            negative_counter = 0
        episode_reward += reward
        action = agent.step(reward, done, obs)

        if not done and negative_counter < steps_per_episode:
            continue

        agent.writer.add_scalar("episode reward", episode_reward, agent.steps)
        agent.writer.add_scalar("episode duration", steps, agent.steps)
        print(episode_counter, episode_reward)

        if episode_counter % 20 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            agent.checkpoint(f"checkpoints/{episode_counter}.ckpt")

        negative_counter = 0
        rewards.append(episode_reward)
        steps = 0
        episode_reward = 0.0
        episode_counter += 1
        if episode_counter >= episodes:
            break
        obs = env.reset()
        action = agent.start(obs)

    return rewards


def main() -> None:
    env = DiscreteCarRacing() 
    agent = DQNAgent(
        env.observation_space.shape[:2] + (1,),
        env.action_space.n,
        epsilon=0.5,
        gamma=0.95,
    )
    rewards = run(env, agent, episodes=1000, render=True)


if __name__ == "__main__":
    main()
