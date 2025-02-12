import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def ee_pos_rot(
    env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset ee position and orientation in the environment frame."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    pos = asset.data.body_pos_w[:, asset_cfg.body_ids].squeeze() - asset.data.root_pos_w
    quat = asset.data.body_state_w[:, asset_cfg.body_ids, 3:7].squeeze()
    rot_mat = math_utils.matrix_from_quat(quat)
    ortho6d = rot_mat[..., :2].reshape(-1, 6)
    return torch.cat([pos, ortho6d], dim=-1)
