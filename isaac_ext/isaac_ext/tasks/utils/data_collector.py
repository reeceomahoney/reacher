from typing import Any, Dict, Optional, Union

import gymnasium as gym
import h5py
import numpy as np


class GymDataCollector:
    """A class to collect and store data from gym environments in HDF5 format."""

    def __init__(
        self,
        env: gym.Env,
        file_path: str,
        buffer_size: int = 10000,
        compression: Optional[str] = "gzip",
    ):
        """
        Initialize the data collector.

        Args:
            env: The gymnasium environment
            file_path: Path where the HDF5 file will be saved
            buffer_size: Size of the buffer before writing to disk
            compression: Compression method for HDF5 datasets
        """
        self.env = env
        self.file_path = file_path
        self.buffer_size = buffer_size
        self.compression = compression

        # Initialize buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.terminal_buffer = []

        # Counter for total steps
        self.total_steps = 0

        # Create HDF5 file and initialize datasets
        self._initialize_h5_file()

    def _initialize_h5_file(self):
        """Initialize the HDF5 file with empty datasets."""
        with h5py.File(self.file_path, "w") as f:
            # Create groups for different data types
            f.create_group("observations")
            f.create_group("actions")
            f.create_group("rewards")
            f.create_group("terminals")

            # Store environment information
            env_group = f.create_group("env_info")
            env_group.attrs["env_id"] = self.env.unwrapped.spec.id
            env_group.attrs["action_space"] = str(self.env.action_space)
            env_group.attrs["observation_space"] = str(self.env.observation_space)

    def add_step(
        self,
        observation: Union[np.ndarray, Dict[str, Any]],
        action: Union[int, float, np.ndarray],
        reward: float,
        terminated: bool,
    ):
        """
        Add a single step of interaction to the buffer.

        Args:
            observation: The observation from the environment
            action: The action taken
            reward: The reward received
            terminated: Whether the episode terminated
        """
        self.obs_buffer.append(observation)
        self.action_buffer.append(action)
        self.reward_buffer.append(reward)
        self.terminal_buffer.append(terminated)

        self.total_steps += 1

        # If buffer is full, write to disk
        if len(self.obs_buffer) >= self.buffer_size:
            self.flush()

    def flush(self):
        """Write current buffer to disk and clear it."""
        if not self.obs_buffer:
            return

        with h5py.File(self.file_path, "a") as f:
            # Convert buffers to numpy arrays
            obs_array = np.array(self.obs_buffer)
            action_array = np.array(self.action_buffer)
            reward_array = np.array(self.reward_buffer)
            terminal_array = np.array(self.terminal_buffer)

            # Calculate start index for this batch
            start_idx = self.total_steps - len(self.obs_buffer)

            # Write data
            self._write_dataset(f["observations"], obs_array, start_idx)
            self._write_dataset(f["actions"], action_array, start_idx)
            self._write_dataset(f["rewards"], reward_array, start_idx)
            self._write_dataset(f["terminals"], terminal_array, start_idx)

        # Clear buffers
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.terminal_buffer = []

    def _write_dataset(self, group: h5py.Group, data: np.ndarray, start_idx: int):
        """Helper method to write or resize and write datasets."""
        dataset_name = f"chunk_{start_idx}"
        group.create_dataset(dataset_name, data=data, compression=self.compression)

    def close(self):
        """Flush remaining data and close the collector."""
        self.flush()


# Example usage:
if __name__ == "__main__":
    # Create a simple environment
    env = gym.make("CartPole-v1")

    # Initialize data collector
    collector = GymDataCollector(env, "cartpole_data.h5", buffer_size=1000)

    # Run some episodes
    for episode in range(5):
        obs, _ = env.reset()
        done = False

        while not done:
            action = env.action_space.sample()  # Random action
            next_obs, reward, terminated, truncated, _ = env.step(action)

            # Collect data
            collector.add_step(obs, action, reward, terminated)

            obs = next_obs
            done = terminated or truncated

    # Close collector and environment
    collector.close()
    env.close()
