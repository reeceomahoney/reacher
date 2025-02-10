from omni.isaac.lab.envs.manager_based_env import ManagerBasedEnv
from omni.isaac.lab.envs.mdp.actions import actions_cfg
from omni.isaac.lab.envs.mdp.actions.joint_actions import JointAction
from omni.isaac.lab.managers.action_manager import ActionTerm
from omni.isaac.lab.utils.configclass import configclass


class ClippedJointPositionAction(JointAction):
    """Joint action term that is clipped by the joint limits."""

    cfg: actions_cfg.JointPositionActionCfg
    """The configuration of the action term."""

    def __init__(self, cfg: actions_cfg.JointPositionActionCfg, env: ManagerBasedEnv):
        # initialize the action term
        super().__init__(cfg, env)
        # use default joint positions as offset
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[
                :, self._joint_ids
            ].clone()
        self._min = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 0]
        self._max = self._asset.data.soft_joint_pos_limits[:, self._joint_ids, 1]

    def apply_actions(self):
        # set position targets
        self._asset.set_joint_position_target(
            self.processed_actions.clamp(self._min, self._max),
            joint_ids=self._joint_ids
        )

@configclass
class ClippedJointPositionActionCfg(actions_cfg.JointActionCfg):
    """Configuration for the joint position action term. """

    class_type: type[ActionTerm] = ClippedJointPositionAction

    use_default_offset: bool = True
    """Whether to use default joint positions configured in the articulation asset as offset.
    Defaults to True.

    If True, this flag results in overwriting the values of :attr:`offset` to the default joint positions
    from the articulation asset.
    """
