import matplotlib.pyplot as plt
import numpy as np
import torch


def plot_maze(maze: torch.Tensor, figsize: tuple = (8, 8)):
    fig, ax = plt.subplots(figsize)
    ax.imshow(maze, cmap="gray", extent=(-4, 4, -4, 4))
    return fig, ax


def plot_obstacle(ax: plt.Axes, obstacle: torch.Tensor):
    obstacle_square = plt.Rectangle(
        (obstacle[0, 0].item(), obstacle[0, 1].item()),
        1.0,
        1.0,
        facecolor="red",
        alpha=0.5,
    )
    ax.add_patch(obstacle_square)


def plot_trajectory(
    ax: plt.Axes,
    obs_traj: np.ndarray,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
):
    marker_params = {"markersize": 10, "markeredgewidth": 3}
    # Plot trajectory with color gradient
    colors = plt.cm.inferno(np.linspace(0, 1, len(obs_traj)))
    ax.scatter(obs_traj[:, 0], obs_traj[:, 1], c=colors)
    # Plot start and goal positions
    ax.plot(start_pos[0], start_pos[1], "x", color="green", **marker_params)
    ax.plot(goal_pos[0], goal_pos[1], "x", color="red", **marker_params)


def plot_cfg_analysis(
    runner,
    env,
    obs: torch.Tensor,
    goal: torch.Tensor,
    obstacle: torch.Tensor,
    cond_lambda: list,
):
    fig, axes = plt.subplots(1, len(cond_lambda), figsize=(16, 6))
    runner.policy.set_goal(goal)
    goal_np = goal.cpu().numpy()

    for i, lam in enumerate(cond_lambda):
        # Compute trajectory
        runner.policy.model.cond_lambda = lam
        obs_traj = runner.policy.act({"obs": obs, "obstacles": obstacle})
        obs_traj = obs_traj["obs_traj"][0].cpu().numpy()

        # Plot maze and trajectory
        axes[i].imshow(env.get_maze(), cmap="gray", extent=(-4, 4, -4, 4))
        plot_obstacle(axes[i], obstacle)
        plot_trajectory(
            axes[i],
            obs_traj,
            start_pos=(obs_traj[0, 0], obs_traj[0, 1]),
            goal_pos=(goal_np[0, 0], goal_np[0, 1]),
        )

        axes[i].set_title(f"cond_lambda={lam}")
        axes[i].set_axis_off()

    fig.tight_layout()
    plt.show()


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
