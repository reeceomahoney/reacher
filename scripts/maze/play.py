import os
import random
import statistics
import sys

import hydra
import numpy as np
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from locodiff.envs import MazeEnv
from locodiff.plotting import plot_cfg_analysis, plot_interactive_trajectory
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

    test_type = "cfg"

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

    elif test_type == "cfg":
        obs = torch.tensor([[-1.5, 0.5, 0, 0]]).to(runner.device)
        goal = torch.tensor([[2.5, 2.5]]).to(runner.device)
        obstacle = torch.tensor([[-1, 0]]).to(runner.device)
        cond_lambda = [0, 1, 2, 3, 5, 10, 20]

        # Generate plots
        plot_cfg_analysis(runner, env, obs, goal, obstacle, cond_lambda)

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
