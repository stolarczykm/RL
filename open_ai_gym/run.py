from typing import Optional

import click
import gym
from tqdm import tqdm
import yaml

from agents import Agent
from vizualizer import RewardVizualizer

def run_agent(
    episodes: int, 
    env: gym.Env,
    agent: Agent, 
    vizualizer: RewardVizualizer,
    vizualization_freq: Optional[int] = 10,
):
    obs = env.reset()
    action = agent.agent_start(obs)
    episode_number = 0
    episode_steps = 0
    total_reward = 0
    vizualizer.start()
    rewards = []

    for _ in tqdm(range(episodes)):
        if vizualization_freq and (episode_number % vizualization_freq == 0):
            env.render()
            vizualizer.show(episode_number, episode_steps)
        obs, reward, done, _ = env.step(action)
        total_reward += reward
        episode_steps += 1
        if not done:
            action = agent.agent_step(reward, obs)
        else:
            agent.agent_end(reward)
            obs = env.reset()
            episode_number += 1
            rewards.append(total_reward)
            vizualizer.update(total_reward)
            total_reward, episode_steps = 0, 0
            agent.agent_start(obs)
    env.close()


@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False))
def run(config_path):
    with open(config_path, "r") as config_file:
        config = yaml.safe_load(config_file)
    env = gym.make(config["env"])
    agent = Agent.from_config(
        config["agent"],
        env.action_space,
        env.observation_space,
    )
    vizualizer = RewardVizualizer()
    vizualiztion_freq = config["vizualization_freq"]
    n_episodes = config["n_episodes"]
    run_agent(n_episodes, env, agent, vizualizer, vizualization_freq=vizualiztion_freq)


if __name__ == "__main__":
    run()