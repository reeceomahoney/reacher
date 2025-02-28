import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class PDControlledParticleDataset(Dataset):
    """
    Dataset for particle trajectories controlled by a PD controller.

    Particles start near either bottom-left or bottom-right corner with
    variation in initial position, and are guided to the opposite corner
    via PD control with noise in the dynamics.
    """

    def __init__(
        self,
        num_samples=1000,
        trajectory_length=100,
        grid_size=1.0,
        process_noise=0.02,
        measurement_noise=0.01,
        init_pos_var=0.05,
        kp=1.0,
        kd=0.5,
        dt=0.1,
        seed=42,
    ):
        """
        Args:
            num_samples: Number of trajectory samples in the dataset
            trajectory_length: Number of steps in each trajectory
            grid_size: Size of the grid (normalized to 1.0)
            process_noise: Standard deviation of noise in the dynamics
            measurement_noise: Standard deviation of noise in the measurement
            init_pos_var: Standard deviation of the initial position variation
            kp: Proportional gain for the PD controller
            kd: Derivative gain for the PD controller
            dt: Time step for the simulation
            seed: Random seed for reproducibility
        """
        super().__init__()
        torch.manual_seed(seed)
        np.random.seed(seed)

        self.num_samples = num_samples
        self.trajectory_length = trajectory_length
        self.grid_size = grid_size
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.init_pos_var = init_pos_var
        self.kp = kp
        self.kd = kd
        self.dt = dt

        # Generate the dataset
        self.trajectories = []
        self.start_corners = []
        self.velocities = []  # Store velocities for analysis
        self.accelerations = []  # Store accelerations for analysis
        self.control_signals = []  # Store control signals for analysis

        for i in range(num_samples):
            # Randomly choose starting corner (0: bottom-left, 1: bottom-right)
            start_corner = i % 2

            # Generate the trajectory with PD control
            trajectory, velocity, acceleration, control = self._generate_pd_trajectory(
                start_corner
            )

            self.trajectories.append(trajectory)
            self.start_corners.append(start_corner)
            self.velocities.append(velocity)
            self.accelerations.append(acceleration)
            self.control_signals.append(control)

    def _generate_pd_trajectory(self, start_corner):
        """Generate a single trajectory with PD control"""
        if start_corner == 0:
            # Bottom-left to top-right
            # Add variation to the initial position
            init_pos = torch.tensor([0.0, 0.0]) + torch.randn(2) * self.init_pos_var
            init_pos = torch.clamp(
                init_pos, 0, self.grid_size * 0.2
            )  # Keep it in the corner region
            target_pos = torch.tensor([self.grid_size, self.grid_size])
        else:
            # Bottom-right to top-left
            # Add variation to the initial position
            init_pos = (
                torch.tensor([0.0, self.grid_size]) + torch.randn(2) * self.init_pos_var
            )
            init_pos = torch.clamp(
                init_pos,
                torch.tensor([0.0, self.grid_size * 0.8]),
                torch.tensor([self.grid_size * 0.2, self.grid_size]),
            )
            target_pos = torch.tensor([self.grid_size, 0.0])

        # Initialize trajectory arrays
        trajectory = torch.zeros((self.trajectory_length, 2))
        velocity = torch.zeros((self.trajectory_length, 2))
        acceleration = torch.zeros((self.trajectory_length, 2))
        control_signal = torch.zeros((self.trajectory_length, 2))

        # Set initial state
        trajectory[0] = init_pos
        velocity[0] = torch.zeros(2)  # Start with zero velocity

        # Simulation loop with PD control
        for t in range(1, self.trajectory_length):
            # Current state
            current_pos = trajectory[t - 1]
            current_vel = velocity[t - 1]

            # Error and error derivative
            error = target_pos - current_pos
            error_derivative = -current_vel  # Assuming target velocity is zero

            # PD control law
            control = self.kp * error + self.kd * error_derivative
            control_signal[t] = control

            # Calculate acceleration (F = ma, assuming m = 1)
            acc = control + torch.randn(2) * self.process_noise
            acceleration[t] = acc

            # Update velocity: v = v0 + a * dt
            new_vel = current_vel + acc * self.dt
            velocity[t] = new_vel

            # Update position: p = p0 + v * dt
            new_pos = current_pos + new_vel * self.dt

            # Add measurement noise
            noisy_pos = new_pos + torch.randn(2) * self.measurement_noise

            # Clip position to grid boundaries
            trajectory[t] = torch.clamp(noisy_pos, 0, self.grid_size)

        return trajectory, velocity, acceleration, control_signal

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
        Returns:
            trajectory: Tensor of shape [trajectory_length, 2] containing (x,y) coordinates
            start_corner: 0 for bottom-left, 1 for bottom-right
            velocity: Tensor of shape [trajectory_length, 2] containing velocity vectors
            acceleration: Tensor of shape [trajectory_length, 2] containing acceleration vectors
            control_signal: Tensor of shape [trajectory_length, 2] containing control signals
        """
        return {
            "trajectory": self.trajectories[idx],
            "start_corner": self.start_corners[idx],
            "velocity": self.velocities[idx],
            "acceleration": self.accelerations[idx],
            "control_signal": self.control_signals[idx],
        }

    def visualize_batch(self, batch_size=10):
        """Visualize a batch of trajectories"""
        plt.figure(figsize=(12, 10))

        for i in range(min(batch_size, self.num_samples)):
            sample = self[i]
            trajectory = sample["trajectory"]
            start_corner = sample["start_corner"]

            color = "blue" if start_corner == 0 else "orange"
            plt.plot(trajectory[:, 0], trajectory[:, 1], c=color, alpha=0.6)
            plt.scatter(trajectory[0, 0], trajectory[0, 1], c="green", s=30)
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], c="red", s=30)

        plt.title(f"Batch of {batch_size} PD Controlled Particle Trajectories")
        plt.xlim(-0.1, self.grid_size + 0.1)
        plt.ylim(-0.1, self.grid_size + 0.1)
        plt.grid(True)

        # Add legend
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], color="blue", lw=2, label="Bottom-Left Start"),
            Line2D([0], [0], color="orange", lw=2, label="Bottom-Right Start"),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="green",
                markersize=8,
                label="Start",
            ),
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                markerfacecolor="red",
                markersize=8,
                label="End",
            ),
        ]
        plt.legend(handles=legend_elements)

        plt.show()


# Example usage
if __name__ == "__main__":
    # Create dataset with PD control
    dataset = PDControlledParticleDataset(
        num_samples=100,
        trajectory_length=30,
        grid_size=1.0,
        process_noise=0.03,
        measurement_noise=0.01,
        init_pos_var=0.05,
        kp=2.0,
        kd=1.0,
        dt=0.05,
        seed=42,
    )

    # Create dataloader
    dataloader = DataLoader(dataset, batch_size=100, shuffle=True)

    # Visualize a batch
    dataset.visualize_batch(100)

    # Iterate through the dataloader
    for batch in dataloader:
        # Access batch['trajectory], batch['start_corner'], etc.
        print(f"Batch trajectories shape: {batch['trajectory'].shape}")
        print(f"Batch start corners shape: {batch['start_corner'].shape}")
        break  # Just show the first batch
