import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# Import the Environment2D class from the previous code
# Assuming the class is saved in a file called environment_2d.py
# If you have the class definition in the same file, you can comment out this import
# and include the class definition here
from locodiff.envs import Environment2D


def apply_force_sequence(env, steps, force_fn):
    """
    Roll out the environment for a given number of steps with a force function

    Args:
        env: The environment instance
        steps: Number of steps to simulate
        force_fn: Function that takes step number and returns a force vector

    Returns:
        history: List of positions
    """
    history = [env.get_state()["position"].copy()]

    for i in range(steps):
        force = force_fn(i)
        new_pos = env.step(force)
        history.append(new_pos.copy())

    return history


def test_environment():
    # Create environment
    env = Environment2D(mass=1.0, damping=0.05, dt=0.1)

    # Define different force functions for testing
    force_functions = [
        # Constant force to the right and up
        lambda t: [1.0, 0.5],
        # Circular force (changing direction)
        lambda t: [2.0 * np.cos(t * 0.2), 2.0 * np.sin(t * 0.2)],
        # Oscillating force
        lambda t: [np.sin(t * 0.3) * 3.0, np.cos(t * 0.2) * 2.0],
        # Random force
        lambda t: [np.random.uniform(-2, 2), np.random.uniform(-2, 2)],
    ]

    force_names = ["Constant", "Circular", "Oscillating", "Random"]

    # Number of steps to simulate
    steps = 100

    # Create a figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    axs = axs.flatten()

    # Run simulations and plot results
    for i, (force_fn, name) in enumerate(
        zip(force_functions, force_names, strict=False)
    ):
        # Reset environment
        env.set_state(position=[0.0, 0.0], velocity=[0.0, 0.0])

        # Simulate
        history = apply_force_sequence(env, steps, force_fn)

        # Convert to numpy array for easier plotting
        trajectory = np.array(history)

        # Plot
        ax = axs[i]
        ax.plot(trajectory[:, 0], trajectory[:, 1], "b-", linewidth=1.5)
        ax.plot(trajectory[0, 0], trajectory[0, 1], "go", label="Start")
        ax.plot(trajectory[-1, 0], trajectory[-1, 1], "ro", label="End")

        # Add arrows to show direction
        skip = max(1, steps // 10)
        for j in range(0, steps, skip):
            if j + 1 < len(trajectory):
                dx = trajectory[j + 1, 0] - trajectory[j, 0]
                dy = trajectory[j + 1, 1] - trajectory[j, 1]
                ax.arrow(
                    trajectory[j, 0],
                    trajectory[j, 1],
                    dx,
                    dy,
                    head_width=0.1,
                    head_length=0.2,
                    fc="black",
                    ec="black",
                )

        # Set plot properties
        ax.set_title(f"{name} Force")
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.grid(True)
        ax.set_xlim([-5.5, 5.5])
        ax.set_ylim([-5.5, 5.5])
        ax.legend()

        # Draw boundary
        boundary = plt.Rectangle(
            (-5, -5), 10, 10, fill=False, edgecolor="red", linestyle="--", linewidth=2
        )
        ax.add_patch(boundary)

    plt.tight_layout()
    plt.savefig("environment_trajectories.png")
    plt.show()

    # Create an animation of a single trajectory (circular force)
    def create_animation():
        env.set_state(position=[0.0, 0.0], velocity=[0.0, 0.0])
        history = apply_force_sequence(
            env, 200, lambda t: [2.0 * np.cos(t * 0.2), 2.0 * np.sin(t * 0.2)]
        )
        trajectory = np.array(history)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlim([-5.5, 5.5])
        ax.set_ylim([-5.5, 5.5])
        ax.set_xlabel("X Position")
        ax.set_ylabel("Y Position")
        ax.set_title("Particle Movement with Circular Force")
        ax.grid(True)

        # Draw boundary
        boundary = plt.Rectangle(
            (-5, -5), 10, 10, fill=False, edgecolor="red", linestyle="--", linewidth=2
        )
        ax.add_patch(boundary)

        (line,) = ax.plot([], [], "b-", linewidth=1.5)
        (point,) = ax.plot([], [], "ro", markersize=8)

        def init():
            line.set_data([], [])
            point.set_data([], [])
            return line, point

        def animate(i):
            line.set_data(trajectory[: i + 1, 0], trajectory[: i + 1, 1])
            point.set_data(trajectory[i, 0], trajectory[i, 1])
            return line, point

        anim = FuncAnimation(
            fig, animate, frames=len(trajectory), init_func=init, blit=True, interval=50
        )

        # Save animation (optional)
        # anim.save('particle_movement.gif', writer='pillow', fps=20)

        plt.show()

    # Uncomment the next line to create and show the animation
    # create_animation()


if __name__ == "__main__":
    test_environment()
