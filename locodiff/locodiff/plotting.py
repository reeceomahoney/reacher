import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_maze(maze: torch.Tensor, figsize: tuple = (8, 8)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(maze, cmap="gray", extent=(-4, 4, -4, 4))
    return fig, ax


def plot_obstacle(ax, obstacle: torch.Tensor):
    obstacle_square = Rectangle(
        (obstacle[0, 0].item(), obstacle[0, 1].item()),
        1.0,
        1.0,
        facecolor="red",
        alpha=0.5,
    )
    ax.add_patch(obstacle_square)


def plot_trajectory(
    ax,
    obs_traj: np.ndarray,
    start_pos: tuple,
    goal_pos: tuple,
):
    marker_params = {"markersize": 10, "markeredgewidth": 3}
    # Plot trajectory with color gradient
    gradient = np.linspace(0, 1, len(obs_traj))
    ax.scatter(obs_traj[:, 0], obs_traj[:, 1], c=gradient, cmap="inferno")
    # Plot start and goal positions
    ax.plot(start_pos[0], start_pos[1], "x", color="green", **marker_params)
    ax.plot(goal_pos[0], goal_pos[1], "x", color="red", **marker_params)


def plot_guided_trajectory(
    policy,
    env,
    obs: torch.Tensor,
    goal: torch.Tensor,
    obstacle: torch.Tensor,
    alphas: list,
):
    fig, axes = plt.subplots(1, len(alphas), figsize=(16, 3.5))
    goal_np = goal.cpu().numpy()

    for i, alpha in enumerate(alphas):
        # Compute trajectory
        policy.alpha = alpha
        obs_traj = policy.act({"obs": obs, "obstacle": obstacle, "goal": goal})
        obs_traj = obs_traj["obs_traj"][0].detach().cpu().numpy()

        # Plot maze and trajectory
        axes[i].imshow(env.get_maze(), cmap="gray", extent=(-4, 4, -4, 4))
        plot_obstacle(axes[i], obstacle)
        plot_trajectory(
            axes[i],
            obs_traj,
            start_pos=(obs_traj[0, 0], obs_traj[0, 1]),
            goal_pos=(goal_np[0, 0], goal_np[0, 1]),
        )

        axes[i].set_title(f"alphas={alpha}")
        axes[i].set_axis_off()

    fig.tight_layout()
    return fig


def plot_3d_guided_trajectory(
    policy,
    obs: torch.Tensor,
    goal: torch.Tensor,
    obstacle: torch.Tensor,
    alphas: list,
):
    fig, axes = plt.subplots(
        1, len(alphas), figsize=(16, 3.5), subplot_kw={"projection": "3d"}
    )
    goal_ = goal.cpu().numpy()

    # obstacle
    cuboid_vertices = [
        [0.55, -0.8, 0.0],
        [0.65, -0.8, 0.0],
        [0.65, 0.8, 0.0],
        [0.55, 0.8, 0.0],
        [0.55, -0.8, 0.6],
        [0.65, -0.8, 0.6],
        [0.65, 0.8, 0.6],
        [0.55, 0.8, 0.6],
    ]
    cuboid_faces = [
        [cuboid_vertices[j] for j in [0, 1, 5, 4]],
        [cuboid_vertices[j] for j in [1, 2, 6, 5]],
        [cuboid_vertices[j] for j in [2, 3, 7, 6]],
        [cuboid_vertices[j] for j in [3, 0, 4, 7]],
        [cuboid_vertices[j] for j in [0, 1, 2, 3]],
        [cuboid_vertices[j] for j in [4, 5, 6, 7]],
    ]

    for i, alpha in enumerate(alphas):
        print(alpha)
        # Compute trajectory
        policy.alpha = alpha
        traj = policy.act({"obs": obs, "obstacle": obstacle, "goal": goal})
        traj = traj["obs_traj"][0, :, 18:21].detach().cpu().numpy()

        # Plot trajectory with color gradient
        gradient = np.linspace(0, 1, len(traj))
        axes[i].scatter(traj[:, 0], traj[:, 1], traj[:, 2], c=gradient, cmap="inferno")

        # Plot start and goal positions
        marker_params = {"markersize": 10, "markeredgewidth": 3}
        axes[i].plot(
            traj[0, 0], traj[0, 1], traj[0, 2], "x", color="green", **marker_params
        )
        axes[i].plot(
            goal_[0, 0], goal_[0, 1], goal_[0, 2], "x", color="red", **marker_params
        )
        axes[i].add_collection3d(
            Poly3DCollection(cuboid_faces, alpha=0.5, facecolor="red")
        )
        axes[i].view_init(elev=0, azim=90)
        axes[i].set_title(f"alphas={alpha}")

    fig.tight_layout()
    return fig


def plot_interactive_trajectory(env, runner, obs: torch.Tensor):
    _, ax = plot_maze(env.get_maze())

    # Get current observation and goal positions
    obs_np = obs.cpu().numpy()
    goal_np = env.goal.cpu().numpy()

    # Plot trajectory
    output = runner.policy.act({"obs": obs})
    obs_traj = output["obs_traj"][0].cpu().numpy()

    plot_trajectory(
        ax,
        obs_traj,
        start_pos=(obs_np[0, 0], obs_np[0, 1]),
        goal_pos=(goal_np[0, 0], goal_np[0, 1]),
    )

    plt.draw()
    plt.pause(0.1)

    return output
