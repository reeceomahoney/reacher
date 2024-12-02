import matplotlib.pyplot as plt
import os
import sys
import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

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
            obs_traj = output["obs_traj"].cpu().numpy()
            plt.imshow(env.get_maze(), cmap="gray", extent=(-4, 4, -4, 4))
            plt.plot(obs_traj[0, :, 0], obs_traj[0, :, 1])
            # plot current and goal position
            obs_np = obs.cpu().numpy()
            goal_np = env.goal.cpu().numpy()
            plt.plot(obs_np[0, 0], obs_np[0, 1], "go")
            plt.plot(goal_np[0], goal_np[1], "ro")
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
                    env.reset()
                    runner.policy.reset(dones)
                    break

        timestep += 1

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    sys.argv.append("hydra.output_subdir=null")
    sys.argv.append("hydra.run.dir=.")
    main()
