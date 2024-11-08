import torch

from omni.isaac.lab.assets import RigidObject
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.utils.math import matrix_from_quat


def ee_pos_rot(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset ee position and orientation in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze() - asset.data.root_pos_w
    quat = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7].squeeze()
    rot_mat = matrix_from_quat(quat)
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([pos, ortho6d], dim=-1)
