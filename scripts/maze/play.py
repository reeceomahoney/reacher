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

from locodiff.classifier_runner import ClassifierRunner
from locodiff.envs import MazeEnv
from locodiff.plotting import plot_guided_trajectory, plot_interactive_trajectory
from locodiff.utils import get_latest_run, get_open_maze_squares

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra.main(
    config_path="../../isaac_ext/isaac_ext/tasks/diffusion/config/maze/",
    config_name="classifier_cfg.yaml",
    version_base=None,
)
def main(agent_cfg: DictConfig):
    """Train with RSL-RL agent."""
    # specify directory for logging experiments
    log_dir = HydraConfig.get().runtime.output_dir
    print(f"[INFO] Logging experiment in directory: {log_dir}")

    # set random seed
    random.seed(agent_cfg.seed)
    np.random.seed(agent_cfg.seed)
    torch.manual_seed(agent_cfg.seed)

    # create isaac environment
    env = MazeEnv(agent_cfg, render=False)
    agent_cfg.obs_dim = env.obs_dim
    agent_cfg.act_dim = env.act_dim

    # create runner from rsl-rl
    # runner = DiffusionRunner(env, agent_cfg, device=agent_cfg.device)
    runner = ClassifierRunner(env, agent_cfg, device=agent_cfg.device)

    # load the checkpoint
    log_root_path = os.path.abspath("logs/classifier/maze")
    resume_path = os.path.join(get_latest_run(log_root_path), "models/model.pt")
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(resume_path)
    runner.eval_mode()

    test_type = "guidance"

    if test_type == "mse":
        samplings_steps = [3, 10, 20, 50, 100, 256]
        for steps in samplings_steps:
            test_mse = []
            runner.policy.sampling_steps = steps
            for batch in tqdm(runner.test_loader, desc="Testing...", leave=False):
                mse = runner.policy.test(batch, plot=False)[0]
                test_mse.append(mse)
            test_mse = statistics.mean(test_mse)
            print(f"Sampling steps: {steps}, Test MSE: {test_mse}")

    elif test_type == "guidance":
        alphas = [0, 200, 300, 500, 700, 1e3]

        obs = torch.tensor([[-2.5, -0.5, 0, 0]]).to(runner.device)
        goal = torch.tensor([[2.5, 2.5, 0, 0]]).to(runner.device)
        obstacle = torch.tensor([[-1, 0]]).to(runner.device)

        plot_guided_trajectory(runner.policy, env, obs, goal, obstacle, alphas)
        plt.show()

    elif test_type == "collisions":

        def sample_maze(size, add_zeros=True):
            open_squares = get_open_maze_squares(env.get_maze())
            x = open_squares[torch.randint(0, len(open_squares), (size,))]
            if add_zeros:
                x = torch.cat([x, torch.zeros(size, 2)], dim=1)
            return x.to(runner.device)

        B = 50
        alphas = [0, 200, 300, 500, 700, 1e3]

        obs = sample_maze(B) + 0.5
        goal = sample_maze(B) + 0.5
        obstacle = sample_maze(B, add_zeros=False)

        total_collisions = []
        for alpha in alphas:
            runner.policy.alpha = alpha
            obs_traj = runner.policy.act(
                {"obs": obs, "obstacle": obstacle, "goal": goal}
            )
            collisions = runner.policy.check_collisions(
                obs_traj["obs_traj"][..., :2], obstacle
            )
            total_collisions.append(collisions.sum().item())

        plt.plot(alphas, total_collisions)

    elif test_type == "play":
        obs = env.reset()
        runner.policy.set_goal(env.goal)
        timestep = 0

        while True:
            with torch.inference_mode():
                # Plot current trajectory
                output = plot_interactive_trajectory(env, runner.policy, obs)

                # Environment stepping
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

    # Close simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    sys.argv.append("hydra/job_logging=disabled")
    sys.argv.append("hydra/hydra_logging=disabled")
    main()
