from pathlib import Path

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.actuators import ImplicitActuatorCfg
from omni.isaac.lab.assets.articulation import ArticulationCfg

usd_path_z1 = str(Path(__file__).parent.absolute()) + "/z1.usd"

Z1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=usd_path_z1,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            velocity_limit=3.14,
            effort_limit={"joint.*": 30.0},
            stiffness={"joint.*": 20.0},
            damping={"joint.*": 0.5},
        ),
        # "gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["jointGripper"],
        #     velocity_limit=100.0,
        #     effort_limit=0.0,
        #     stiffness=0,
        #     damping=0.0,
        # ),
    },
    soft_joint_pos_limit_factor=0.95,
)
