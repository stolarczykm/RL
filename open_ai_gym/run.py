import os
from typing import Optional

import click
import gym
from tqdm import tqdm
import yaml

from agents import Agent
from vizualizer import EpisodeVizualizer, RewardVizualizer

_CURRENT_DIR = os.path.dirname(__file__)
print(_CURRENT_DIR)

def run_agent(
    episodes: int, 
    env: gym.Env,
    agent: Agent, 
    vizualizer: RewardVizualizer,
    episode_vizualizer: EpisodeVizualizer,
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
            episode_vizualizer.vizualize_if_idle()
            total_reward, episode_steps = 0, 0
            agent.agent_start(obs)
    env.close()
    return agent


@click.command()
@click.option(
    "--config", "-c", 
    type=click.Path(exists=True, dir_okay=False),
    default=f"{_CURRENT_DIR}/config.yml",
)
@click.argument("experiment", default="pendulum")
def run(config, experiment: str):
    with open(config, "r") as config_file:
        config = yaml.safe_load(config_file)
    config = config["experiments"][experiment]
    env = gym.make(config["env"])

    agent = Agent.from_config(
        config["agent"],
        env.action_space,
        env.observation_space,
    )
    vizualizer = RewardVizualizer()
    episode_vizualizer = EpisodeVizualizer(gym.make(config["env"]), agent)
    vizualiztion_freq = config["vizualization_freq"]
    n_episodes = config["n_episodes"]
    run_agent(
        n_episodes, 
        env, 
        agent, 
        vizualizer,
        vizualization_freq=vizualiztion_freq, 
        episode_vizualizer=episode_vizualizer
    )



if __name__ == "__main__":
    run()