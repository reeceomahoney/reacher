import gymnasium as gym
import torch

import gymnasium_robotics

gym.register_envs(gymnasium_robotics)


class MazeEnv:
    def __init__(self, agent_cfg):
        self.env = gym.make(agent_cfg.task, max_episode_steps=agent_cfg.episode_length)
        self.obs = None
        self.device = agent_cfg.device

        self.obs_dim = self.env.observation_space["observation"].shape[0]  # type: ignore
        self.act_dim = self.env.action_space.shape[0]  # type: ignore
        self.num_envs = 1

    def reset(self):
        obs, _ = self.env.reset()
        self.obs = torch.tensor(obs["observation"]).to(self.device)
        return self.obs

    def step(self, action):
        action = action[0, 0].cpu().numpy()
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.obs = torch.tensor(obs["observation"], dtype=torch.float).to(self.device).unsqueeze(0)
        reward = torch.tensor(reward, dtype=torch.float).unsqueeze(0)
        dones = terminated | truncated
        dones = torch.tensor(dones).to(dtype=torch.long).unsqueeze(0)

        return self.obs, reward, dones, info

    def get_observations(self):
        return self.obs, None

    def close(self):
        self.env.close()
