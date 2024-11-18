# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the differential inverse kinematics controller with the simulator.

The differential IK controller can be configured in different modes. It uses the Jacobians computed by
PhysX. This helps perform parallelized computation of the inverse kinematics.

.. code-block:: bash

    # Usage
    ./isaaclab.sh -p source/standalone/tutorials/05_controllers/ik_control.py

"""

"""Launch Isaac Sim Simulator first."""

import argparse
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on using the differential IK controller.")
parser.add_argument("--robot", type=str, default="franka_panda", help="Name of the robot.")
parser.add_argument("--num_envs", type=int, default=128, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.assets import AssetBaseCfg
from omni.isaac.lab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import FRAME_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene, InteractiveSceneCfg
from omni.isaac.lab.utils import configclass
from omni.isaac.lab.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.lab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from omni.isaac.lab_assets import FRANKA_PANDA_HIGH_PD_CFG, UR10_CFG  # isort:skip


@configclass
class TableTopSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # mount
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/Stand/stand_instanceable.usd", scale=(2.0, 2.0, 2.0)
        ),
    )

    # articulation
    if args_cli.robot == "franka_panda":
        robot = FRANKA_PANDA_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    elif args_cli.robot == "ur10":
        robot = UR10_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    else:
        raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")



class JointStatePublisher(Node):
    def __init__(self, sim : sim_utils.SimulationContext, scene : InteractiveScene):
        super().__init__('JointStatePub')

        self.sim = sim 
        self.scene = scene

        self.action_pub = self.create_publisher(JointState, "joint_states", 10)

        self.timer = self.create_timer(0.1, self.run_simulator)
                # Extract scene entities
        # note: we only do this here for readability.
        self.robot = self.scene["robot"]

        # Create controller
        self.diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
        self.diff_ik_controller = DifferentialIKController(self.diff_ik_cfg, num_envs=self.scene.num_envs, device=self.sim.device)

        # Markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

        # Define goals for the arm
        self.ee_goals = [
            [0.5, 0.5, 0.7, 0.707, 0, 0.707, 0],
            [0.5, -0.4, 0.6, 0.707, 0.707, 0.0, 0.0],
            [0.5, 0, 0.5, 0.0, 1.0, 0.0, 0.0],
        ]
        self.ee_goals = torch.tensor(self.ee_goals, device=self.sim.device)
        # Track the given command
        self.current_goal_idx = 0
        # Create buffers to store actions
        self.ik_commands = torch.zeros(self.scene.num_envs, self.diff_ik_controller.action_dim, device=self.robot.device)
        self.ik_commands[:] = self.ee_goals[self.current_goal_idx]

        # Specify robot-specific parameters
        if args_cli.robot == "franka_panda":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
        elif args_cli.robot == "ur10":
            self.robot_entity_cfg = SceneEntityCfg("robot", joint_names=[".*"], body_names=["ee_link"])
        else:
            raise ValueError(f"Robot {args_cli.robot} is not supported. Valid: franka_panda, ur10")
        # Resolving the scene entities
        self.robot_entity_cfg.resolve(self.scene)
        # Obtain the frame index of the end-effector
        # For a fixed base robot, the frame index is one less than the body index. This is because
        # the root body is not included in the returned Jacobians.
        if self.robot.is_fixed_base:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            self.ee_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # Define simulation stepping
        self.sim_dt = self.sim.get_physics_dt()
        self.count = 0



    def run_simulator(self):
        """Runs the simulation loop."""
        # reset
        if self.count % 150 == 0:
            # reset time
            self.count = 0
            # reset joint state
            joint_pos = self.robot.data.default_joint_pos.clone()
            joint_vel = self.robot.data.default_joint_vel.clone()
            self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
            self.robot.reset()
            # reset actions
            self.ik_commands[:] = self.ee_goals[self.current_goal_idx]
            joint_pos_des = joint_pos[:, self.robot_entity_cfg.joint_ids].clone()
            # reset controller
            self.diff_ik_controller.reset()
            self.diff_ik_controller.set_command(self.ik_commands)
            # change goal
            self.current_goal_idx = (self.current_goal_idx + 1) % len(self.ee_goals)
        else:
            # obtain quantities from simulation
            jacobian = self.robot.root_physx_view.get_jacobians()[:, self.ee_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            root_pose_w = self.robot.data.root_state_w[:, 0:7]
            joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            # compute frame in root frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
            )
            # compute the joint commands
            joint_pos_des = self.diff_ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)

        # apply actions
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)
        self.scene.write_data_to_sim()
        # perform step
        self.sim.step()
        # update sim-time
        self.count += 1
        # update buffers
        self.scene.update(self.sim_dt)

        # obtain quantities from simulation
        ee_pose_w = self.robot.data.body_state_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        # update marker positions
        self.ee_marker.visualize(ee_pose_w[:, 0:3], ee_pose_w[:, 3:7])
        self.goal_marker.visualize(self.ik_commands[:, 0:3] + self.scene.env_origins, self.ik_commands[:, 3:7])
        
        action_msg = JointState()
        action_msg.header.frame_id = ''
        action_msg.header.stamp = self.get_clock().now().to_msg()
        joint_name_list = [f"panda_joint{i+1}" for i in self.robot_entity_cfg.joint_ids]
        joint_name_list.append("panda_finger_joint1")
        joint_name_list.append("panda_finger_joint2")

        joint_position_list = joint_pos.tolist()[0]
        joint_position_list.extend([0.0,0.0])
        
        action_msg.name = joint_name_list
        action_msg.position = joint_position_list
        action_msg.effort = ee_pose_w.tolist()[0]

        print(f"Efort_pose w: {ee_pose_w}")
        print()
        # print("JOINT POS")
        # print(joint_pos)
        # print(type(joint_pos))
        # print()

        self.action_pub.publish(action_msg)

        


def main(args=None):
    """Main function."""
    rclpy.init(args=args)


    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = TableTopSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator

    JSPub = JointStatePublisher(sim, scene)
    rclpy.spin(JSPub)

    JSPub.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
