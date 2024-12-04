# import gymnasium as gym
import gym
import numpy as np
import torch
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout


@contextmanager
def suppress_output():
    """
    A context manager that redirects stdout and stderr to devnull
    https://stackoverflow.com/a/52442331
    """
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


with suppress_output():
    ## d4rl prints out a variety of warnings
    import d4rl

# import gymnasium_robotics
#
# gym.register_envs(gymnasium_robotics)


class MazeEnv:
    def __init__(self, agent_cfg, render=False):
        render_mode = "human" if render else None
        self.env = gym.make(
            agent_cfg.task,
            # max_episode_steps=agent_cfg.episode_length * 100,
            # render_mode=render_mode,
        )
        self.obs = None
        self.goal = None
        self.device = agent_cfg.device

        # self.obs_dim = self.env.observation_space["observation"].shape[0]  # type: ignore
        self.obs_dim = self.env.observation_space.shape[0]  # type: ignore
        self.act_dim = self.env.action_space.shape[0]  # type: ignore
        self.num_envs = 1

    def reset(self):
        obs = self.env.reset()
        self.obs = (
            torch.tensor(obs, dtype=torch.float)
            .to(self.device)
            .unsqueeze(0)
        )
        self.goal = torch.tensor(self.env.get_target(), dtype=torch.float).to(self.device)
        return self.obs

    def step(self, action):
        action = action[0].cpu().numpy()

        for _ in range(10):
            obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs = (
            torch.tensor(obs, dtype=torch.float)
            .to(self.device)
            .unsqueeze(0)
        )
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        dones = terminated | truncated
        dones = torch.tensor(dones).to(dtype=torch.long).unsqueeze(0)

        return self.obs, reward, dones, info

    def get_observations(self):
        return self.obs, None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_maze(self):
        maze = np.array(self.env.unwrapped.maze.maze_map)
        return 1 - maze
