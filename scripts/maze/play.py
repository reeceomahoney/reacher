import os
import random
import statistics
import sys

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from locodiff.envs import MazeEnv
from locodiff.runner import DiffusionRunner
from vae.utils import get_latest_run

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra.main(
    config_path="../../isaac_ext/isaac_ext/tasks/diffusion/config/maze/",
    config_name="maze_cfg.yaml",
    version_base=None,
)
def main(agent_cfg: DictConfig):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments

    # specify directory for logging experiments
    log_dir = HydraConfig.get().runtime.output_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    # create isaac environment
    env = MazeEnv(agent_cfg, render=True)
    agent_cfg.obs_dim = env.obs_dim
    agent_cfg.act_dim = env.act_dim

    # create runner from rsl-rl
    runner = DiffusionRunner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath("logs/diffusion/maze")
    resume_path = os.path.join(get_latest_run(log_root_path), "models/model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    runner.eval_mode()

    # TODO: make plotting function

    test_type = "cfg"

    if test_type == "mse":
        samplings_steps = [3, 10, 20, 50, 100, 256]
        for steps in samplings_steps:
            test_mse = []
            runner.policy.sampling_steps = steps
            for batch in tqdm(runner.test_loader, desc="Testing...", leave=False):
                mse, obs_mse, act_mse = runner.policy.test(batch, plot=False)
                test_mse.append(mse)
            test_mse = statistics.mean(test_mse)
            print(f"Sampling steps: {steps}, Test MSE: {test_mse}")

    if test_type == "cfg":
        # set up the figure
        cond_lambda = [0, 1, 2, 3, 5, 10, 20]
        fig, axes = plt.subplots(1, len(cond_lambda), figsize=(16, 6))

        # set observation and goal
        obs = torch.tensor([[-2.5, -2.5, 0, 0]]).to(runner.device)
        goal = torch.tensor([[2.5, 2.5]]).to(runner.device)
        obstacle = torch.tensor([[-1, -2]]).to(runner.device)
        runner.policy.set_goal(goal)
        goal = goal.cpu().numpy()

        for i, lam in enumerate(cond_lambda):
            # compute trajectory
            runner.policy.model.cond_lambda = lam
            obs_traj = runner.policy.act({"obs": obs, "obstacles": obstacle})
            obs_traj = obs_traj["obs_traj"][0].cpu().numpy()

            # plot trajectory
            axes[i].imshow(env.get_maze(), cmap="gray", extent=(-4, 4, -4, 4))
            colors = plt.cm.inferno(np.linspace(0, 1, len(obs_traj)))  # type: ignore
            axes[i].scatter(obs_traj[:, 0], obs_traj[:, 1], c=colors)
            # plot current and goal position
            marker_params = {"markersize": 10, "markeredgewidth": 3}
            axes[i].plot(
                obs_traj[0, 0], obs_traj[0, 1], "x", color="green", **marker_params
            )
            axes[i].plot(goal[0, 0], goal[0, 1], "x", color="red", **marker_params)  # type: ignore
            # create title
            axes[i].set_title(f"cond_lambda={lam}")
            axes[i].set_axis_off()

        fig.tight_layout()
        plt.show()

    elif test_type == "play":
        # obtain the trained policy for inference
        policy = runner.get_inference_policy(device=env.device)

        # make figure
        plt.figure(figsize=(8, 8))

        # reset environment
        obs = env.reset()
        runner.policy.set_goal(env.goal)
        timestep = 0
        # simulate environment
        while True:
            # run everything in inference mode
            with torch.inference_mode():
                # agent stepping
                output = policy({"obs": obs})

                # plot trajectory
                plt.clf()
                obs_traj = output["obs_traj"][0].cpu().numpy()
                plt.imshow(env.get_maze(), cmap="gray", extent=(-4, 4, -4, 4))
                colors = plt.cm.inferno(np.linspace(0, 1, len(obs_traj)))  # type: ignore
                plt.scatter(obs_traj[:, 0], obs_traj[:, 1], c=colors)
                # plot current and goal position
                obs_np = obs.cpu().numpy()
                goal_np = env.goal.cpu().numpy()
                marker_params = {"markersize": 10, "markeredgewidth": 3}
                plt.plot(
                    obs_np[0, 0], obs_np[0, 1], "x", color="green", **marker_params
                )  # type: ignore
                plt.plot(
                    goal_np[0, 0], goal_np[0, 1], "x", color="red", **marker_params
                )  # type: ignore
                # draw
                plt.draw()
                plt.pause(0.1)

                # env stepping
                for i in range(runner.policy.T_action):
                    obs, _, dones, _ = env.step(output["action"][:, i])
                    env.render()
                    if i < runner.policy.T_action - 1:
                        runner.policy.update_history({"obs": obs})

                    if dones.any():
                        obs = env.reset()
                        runner.policy.reset(dones)
                        runner.policy.set_goal(env.goal)
                        break

            timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    main()
