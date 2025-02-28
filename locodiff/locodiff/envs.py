import gymnasium as gym
import gymnasium_robotics
import numpy as np
import torch

gym.register_envs(gymnasium_robotics)


class MazeEnv:
    def __init__(self, agent_cfg, render=False):
        render_mode = "human" if render else None
        self.env = gym.make(
            agent_cfg.task,
            max_episode_steps=int(agent_cfg.episode_length * 100),
            render_mode=render_mode,
        )

        self.num_envs = 1
        self.device = agent_cfg.device
        self.obs_dim = self.env.observation_space["observation"].shape[0]  # type: ignore
        self.act_dim = self.env.action_space.shape[0]  # type: ignore

        self.obs = torch.zeros((1, self.obs_dim), device=self.device)

    def reset(self):
        obs, _ = self.env.reset()
        self.obs = self.to_tensor(obs["observation"])
        goal = self.to_tensor(obs["desired_goal"])
        self.goal = torch.cat([goal, torch.zeros_like(goal)], dim=-1)
        return self.obs

    def step(self, action):
        action = action[0].detach().cpu().numpy()

        obs, reward, terminated, trunacted, info = self.env.step(action)

        self.obs = self.to_tensor(obs["observation"])
        reward = self.to_tensor(reward)
        dones = terminated | trunacted
        dones = torch.tensor(dones).to(dtype=torch.long).unsqueeze(0)

        return self.obs, reward, dones, info

    def get_observations(self) -> tuple[torch.Tensor, None]:
        return self.obs, None

    def render(self):
        self.env.render()

    def close(self):
        self.env.close()

    def get_maze(self):
        maze = np.array(self.env.unwrapped.maze.maze_map)  # type: ignore
        return 1 - maze

    def to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float, device=self.device).unsqueeze(0)

class Environment2D:
    def __init__(self, mass=1.0, damping=0.1, dt=0.1, bounds=(-5, 5)):
        """
        Initialize a 2D environment with fixed boundaries.
        
        Args:
            mass (float): Mass of the object in the environment
            damping (float): Damping coefficient for velocity (simulates friction)
            dt (float): Time step for simulation
            bounds (tuple): Boundaries of the environment (min, max) for both x and y
        """
        self.mass = mass
        self.damping = damping
        self.dt = dt
        self.min_bound, self.max_bound = bounds
        
        # Initial state
        self.position = [0.0, 0.0]  # [x, y]
        self.velocity = [0.0, 0.0]  # [vx, vy]
        
    def step(self, force):
        """
        Update the state of the environment based on applied force.
        
        Args:
            force (list): 2D force vector [fx, fy]
            
        Returns:
            list: New position after the step
        """
        # Calculate acceleration (F = ma)
        ax = force[0] / self.mass
        ay = force[1] / self.mass
        
        # Update velocity with damping (simulates friction or air resistance)
        self.velocity[0] = (1 - self.damping) * self.velocity[0] + ax * self.dt
        self.velocity[1] = (1 - self.damping) * self.velocity[1] + ay * self.dt
        
        # Update position
        self.position[0] += self.velocity[0] * self.dt
        self.position[1] += self.velocity[1] * self.dt
        
        # Enforce boundaries
        self._enforce_boundaries()
        
        return self.position.copy()
    
    def _enforce_boundaries(self):
        """Ensure position stays within the defined boundaries"""
        for i in range(2):
            # Check if position exceeds boundaries
            if self.position[i] < self.min_bound:
                self.position[i] = self.min_bound
                self.velocity[i] = 0  # Stop at boundary
            elif self.position[i] > self.max_bound:
                self.position[i] = self.max_bound
                self.velocity[i] = 0  # Stop at boundary
    
    def get_state(self):
        """Return current state of the environment"""
        return {
            "position": self.position.copy(),
            "velocity": self.velocity.copy()
        }
    
    def set_state(self, position=None, velocity=None):
        """
        Set the state of the environment
        
        Args:
            position (list, optional): New position [x, y]
            velocity (list, optional): New velocity [vx, vy]
        """
        if position is not None:
            self.position = position.copy()
            self._enforce_boundaries()
        
        if velocity is not None:
            self.velocity = velocity.copy()
