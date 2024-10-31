from pathlib import Path

from omni.isaac.lab_assets.anymal import ANYDRIVE_3_LSTM_ACTUATOR_CFG

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

usd_path_anymal_d_z1 = str(Path(__file__).parent.absolute()) + "/anymal_d_z1.usd"

ANYMAL_D_Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path_anymal_d_z1,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*HAA": 0.0,  # all HAA
            "joint1": 0.0,  # joint1 arm
            ".*F_HFE": 0.4,  # both front HFE
            ".*H_HFE": -0.4,  # both hind HFE
            "joint2": 0.0,  # joint2 arm
            ".*F_KFE": -0.8,  # both front KFE
            ".*H_KFE": 0.8,  # both hind KFE
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    actuators={
        "legs": ANYDRIVE_3_LSTM_ACTUATOR_CFG,
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            velocity_limit=3.14,
            effort_limit={"joint.*": 30.0},
            stiffness={"joint.*": 20.0},
            damping={"joint.*": 0.5},
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["jointGripper"],
            velocity_limit=100.0,
            effort_limit=0.0,
            stiffness=0,
            damping=0.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
