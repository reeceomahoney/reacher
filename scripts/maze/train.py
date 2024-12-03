import torch

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

from locodiff.envs import MazeEnv
from locodiff.runner import DiffusionRunner

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

    # create isaac environment
    env = MazeEnv(agent_cfg)
    agent_cfg.obs_dim = env.obs_dim
    agent_cfg.act_dim = env.act_dim

    # create runner from rsl-rl
    runner = DiffusionRunner(env, agent_cfg, log_dir=log_dir, device=agent_cfg.device)

    # run training
    runner.learn()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
