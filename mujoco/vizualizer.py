import matplotlib.pyplot as plt
import numpy as np
from multiprocessing import Process, Queue
from concurrent.futures import ProcessPoolExecutor, Future

import gym
from agents import Agent

class RewardVizualizer():
    def __init__(self, active=True, window=40) -> None:
        self.active = active
        self.window = window


    @classmethod
    def from_config(cls, config) -> "RewardVizualizer":
        return RewardVizualizer(**config)

    def start(self):
        if not self.active:
            return
        self.total_reward = 0
        self.episode_steps = 0
        self.episode_number = 0
        self.rewards = []

        self.fig = plt.figure(figsize=(12, 5))
        self.ax = self.fig.add_subplot(111)
        self.line1 = None
        self.line2 = None
        plt.title("Reward over time")
        plt.grid()
        plt.show(block=False)
    

    def update(self, reward):
        self.rewards.append(reward) 
        
        
    def show(self, episode_number, episode_steps):
        if self.active and episode_number > 0 and episode_steps == 0:
            x, y = np.arange(0, len(self.rewards)), self.rewards
            y_cum = np.cumsum(y)
            moving_avg = (y_cum[self.window:] - y_cum[:-self.window]) / self.window
            
            if self.line1 is None:
                self.line1, = self.ax.plot(x, y, alpha=0.4, label="Reward")
                self.line2, = self.ax.plot(
                    x[self.window:], moving_avg, 
                    color="red", 
                    label=f"Average reward from {self.window} episodes",
                )
            else:
                self.ax.set_xlim(0, len(x))
                self.ax.set_ylim(min(y), max(y))
                self.line1.set_data(x, y)
                self.line2.set_data(x[self.window:], moving_avg)
            plt.legend()
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.draw()


class EpisodeVizualizer:
    def __init__(self, env: gym.Env, agent: Agent) -> None:
        self._env = env
        self._agent = agent
        self._pool = ProcessPoolExecutor(max_workers=1)
        self._future: Future | None = None
        self._queue = Queue()
        self._process = Process(target=self._vizualize, args=(self._queue, env))
        self._process.start()

    def close(self):
        self._queue.put("stop")
        self._process.join()
    
    def update_agent(self, agent: Agent):
        self._agent = agent

    def vizualize_if_idle(self):
        if self._queue.empty():
            self._queue.put(self._agent)

        self._future = self._pool.submit(self._vizualize, self._env, self._agent)


    @staticmethod
    def _vizualize(queue: Queue, env: gym.Env):
        while True:
            agent = queue.get()
            if isinstance(agent, str) and agent == "stop":
                env.close()
                return 
            obs = env.reset()
            action = agent.agent_start(obs)
            done = False

            while not done:
                env.render()
                obs, reward, done, _ = env.step(action)
                if not done:
                    action = agent.agent_step(obs, reward)

