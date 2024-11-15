import re
import torch
from datetime import datetime
from pathlib import Path

from omni.isaac.lab.envs import DirectRLEnv, ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import RslRlVecEnvWrapper


def get_latest_run(base_path, resume=False):
    """
    Find the most recent directory in a nested structure like Oct-29/13-01-34/
    Returns the full path to the most recent time directory
    """

    def extract_model_number(file_path):
        match = re.search(r"model_(\d+)\.pt", file_path.name)
        return int(match.group(1)) if match else -1

    all_dirs = []
    base_path = Path(base_path)

    # find all dates
    for date_dir in base_path.iterdir():
        if not date_dir.is_dir():
            continue
        # find all times
        for time_dir in date_dir.iterdir():
            if not time_dir.is_dir():
                continue
            try:
                dir_datetime = datetime.strptime(
                    f"{date_dir.name}/{time_dir.name}", "%b-%d/%H-%M-%S"
                )
                all_dirs.append((time_dir, dir_datetime))
            except ValueError:
                continue

    # sort
    sorted_directories = sorted(all_dirs, key=lambda x: x[1], reverse=True)
    target_dir = sorted_directories[1][0] if resume else sorted_directories[0][0]

    # get latest model
    model_files = list(target_dir.glob("model_*.pt"))
    if model_files:
        latest_model_file = max(model_files, key=extract_model_number)
        return latest_model_file
    else:
        return target_dir


class ReacherEnvWrapper(RslRlVecEnvWrapper):
    def __init__(
        self, env: ManagerBasedRLEnv | DirectRLEnv, num_actions: int | None = None
    ):
        super().__init__(env)
        self.num_actions = num_actions

    def step(
        self, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        # add ee command to action
        des_ee_position = self.env.unwrapped.command_manager.get_command("ee_pose")[:, :3]
        actions = torch.cat([actions, des_ee_position], dim=1)
        # record step information
        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)
        # compute dones for compatibility with RSL-RL
        dones = (terminated | truncated).to(dtype=torch.long)
        # move extra observations to the extras dict
        obs = obs_dict["policy"]
        extras["observations"] = obs_dict
        # move time out information to the extras dict
        # this is only needed for infinite horizon tasks
        if not self.unwrapped.cfg.is_finite_horizon:
            extras["time_outs"] = truncated

        # return the step information
        return obs, rew, dones, extras
