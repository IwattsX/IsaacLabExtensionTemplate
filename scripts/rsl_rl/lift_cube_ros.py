#!/home/iwatts/anaconda3/envs/isaaclab/bin/python3
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to run an environment with a pick and lift state machine.

The state machine is implemented in the kernel function `infer_state_machine`.
It uses the `warp` library to run the state machine in parallel on the GPU.

.. code-block:: bash

    ./isaaclab.sh -p source/standalone/environments/state_machine/lift_cube_sm.py --num_envs 32

"""

"""Launch Omniverse Toolkit first."""

"""
rclpy needs to be here
"""
import rclpy
from rclpy.node import Node
from std_msgs.msg import String, Float64MultiArray
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState
from pprint import pprint

# TODO: Implement custom messages that can work with Isaac Lab
# custom messages with ROS2
# from robot_arm.msg import Actions, PoseOrientation


import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Pick and lift state machine for lift environments.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything else."""

import gymnasium as gym
import torch
from collections.abc import Sequence

import warp as wp

from omni.isaac.lab.assets.rigid_object.rigid_object_data import RigidObjectData

import omni.isaac.lab_tasks  # noqa: F401
from omni.isaac.lab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg
from omni.isaac.lab_tasks.utils.parse_cfg import parse_env_cfg

# initialize warp
wp.init()


class GripperState:
    """States for the gripper."""

    OPEN = wp.constant(1.0)
    CLOSE = wp.constant(-1.0)


class PickSmState:
    """States for the pick state machine."""

    REST = wp.constant(0)
    APPROACH_ABOVE_OBJECT = wp.constant(1)
    APPROACH_OBJECT = wp.constant(2)
    GRASP_OBJECT = wp.constant(3)
    LIFT_OBJECT = wp.constant(4)


class PickSmWaitTime:
    """Additional wait times (in s) for states for before switching."""

    REST = wp.constant(0.2)
    APPROACH_ABOVE_OBJECT = wp.constant(0.5)
    APPROACH_OBJECT = wp.constant(0.6)
    GRASP_OBJECT = wp.constant(0.3)
    LIFT_OBJECT = wp.constant(1.0)


@wp.kernel
def infer_state_machine(
    dt: wp.array(dtype=float),
    sm_state: wp.array(dtype=int),
    sm_wait_time: wp.array(dtype=float),
    ee_pose: wp.array(dtype=wp.transform),
    object_pose: wp.array(dtype=wp.transform),
    des_object_pose: wp.array(dtype=wp.transform),
    des_ee_pose: wp.array(dtype=wp.transform),
    gripper_state: wp.array(dtype=float),
    offset: wp.array(dtype=wp.transform),
):
    # retrieve thread id
    tid = wp.tid()
    # retrieve state machine state
    state = sm_state[tid]
    # decide next state
    if state == PickSmState.REST:
        des_ee_pose[tid] = ee_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.REST:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_ABOVE_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_ABOVE_OBJECT:
        des_ee_pose[tid] = wp.transform_multiply(offset[tid], object_pose[tid])
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.APPROACH_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.APPROACH_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.OPEN
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.APPROACH_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.GRASP_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.GRASP_OBJECT:
        des_ee_pose[tid] = object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.GRASP_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    elif state == PickSmState.LIFT_OBJECT:
        des_ee_pose[tid] = des_object_pose[tid]
        gripper_state[tid] = GripperState.CLOSE
        # TODO: error between current and desired ee pose below threshold
        # wait for a while
        if sm_wait_time[tid] >= PickSmWaitTime.LIFT_OBJECT:
            # move to next state and reset wait time
            sm_state[tid] = PickSmState.LIFT_OBJECT
            sm_wait_time[tid] = 0.0
    # increment wait time
    sm_wait_time[tid] = sm_wait_time[tid] + dt[tid]


class PickAndLiftSm:
    """A simple state machine in a robot's task space to pick and lift an object.

    The state machine is implemented as a warp kernel. It takes in the current state of
    the robot's end-effector and the object, and outputs the desired state of the robot's
    end-effector and the gripper. The state machine is implemented as a finite state
    machine with the following states:

    1. REST: The robot is at rest.
    2. APPROACH_ABOVE_OBJECT: The robot moves above the object.
    3. APPROACH_OBJECT: The robot moves to the object.
    4. GRASP_OBJECT: The robot grasps the object.
    5. LIFT_OBJECT: The robot lifts the object to the desired pose. This is the final state.
    """

    def __init__(self, dt: float, num_envs: int, device: torch.device | str = "cpu"):
        """Initialize the state machine.

        Args:
            dt: The environment time step.
            num_envs: The number of environments to simulate.
            device: The device to run the state machine on.
        """
        # save parameters
        self.dt = float(dt)
        self.num_envs = num_envs
        self.device = device
        # initialize state machine
        self.sm_dt = torch.full((self.num_envs,), self.dt, device=self.device)
        self.sm_state = torch.full((self.num_envs,), 0, dtype=torch.int32, device=self.device)
        self.sm_wait_time = torch.zeros((self.num_envs,), device=self.device)

        # desired state
        self.des_ee_pose = torch.zeros((self.num_envs, 7), device=self.device)
        self.des_gripper_state = torch.full((self.num_envs,), 0.0, device=self.device)

        # approach above object offset
        self.offset = torch.zeros((self.num_envs, 7), device=self.device)
        self.offset[:, 2] = 0.1
        self.offset[:, -1] = 1.0  # warp expects quaternion as (x, y, z, w)

        # convert to warp
        self.sm_dt_wp = wp.from_torch(self.sm_dt, wp.float32)
        self.sm_state_wp = wp.from_torch(self.sm_state, wp.int32)
        self.sm_wait_time_wp = wp.from_torch(self.sm_wait_time, wp.float32)
        self.des_ee_pose_wp = wp.from_torch(self.des_ee_pose, wp.transform)
        self.des_gripper_state_wp = wp.from_torch(self.des_gripper_state, wp.float32)
        self.offset_wp = wp.from_torch(self.offset, wp.transform)

    def reset_idx(self, env_ids: Sequence[int] = None):
        """Reset the state machine."""
        if env_ids is None:
            env_ids = slice(None)
        self.sm_state[env_ids] = 0
        self.sm_wait_time[env_ids] = 0.0

    def compute(self, ee_pose: torch.Tensor, object_pose: torch.Tensor, des_object_pose: torch.Tensor):
        """Compute the desired state of the robot's end-effector and the gripper."""
        # convert all transformations from (w, x, y, z) to (x, y, z, w)
        ee_pose = ee_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        object_pose = object_pose[:, [0, 1, 2, 4, 5, 6, 3]]
        des_object_pose = des_object_pose[:, [0, 1, 2, 4, 5, 6, 3]]

        # convert to warp
        ee_pose_wp = wp.from_torch(ee_pose.contiguous(), wp.transform)
        object_pose_wp = wp.from_torch(object_pose.contiguous(), wp.transform)
        des_object_pose_wp = wp.from_torch(des_object_pose.contiguous(), wp.transform)

        # run state machine
        wp.launch(
            kernel=infer_state_machine,
            dim=self.num_envs,
            inputs=[
                self.sm_dt_wp,
                self.sm_state_wp,
                self.sm_wait_time_wp,
                ee_pose_wp,
                object_pose_wp,
                des_object_pose_wp,
                self.des_ee_pose_wp,
                self.des_gripper_state_wp,
                self.offset_wp,
            ],
            device=self.device,
        )

        # convert transformations back to (w, x, y, z)
        des_ee_pose = self.des_ee_pose[:, [0, 1, 2, 6, 3, 4, 5]]
        # convert to torch
        return torch.cat([des_ee_pose, self.des_gripper_state.unsqueeze(-1)], dim=-1)

class InfoPublisher(Node):
    def __init__(self):
        super().__init__('info_publisher')

        # Dummy test publisher
        self.test_publisher = self.create_publisher(String, 'hello', 10)
        
        # # Create publishers
        # self.action_pub = self.create_publisher(Actions, 'actions', 10)
        self.ee_pose_pub = self.create_publisher(Pose, 'ee_pose_orientation', 10)
        self.object_pose_pub = self.create_publisher(Pose, 'object_pose_orientation', 10)
        self.action_pub = self.create_publisher(JointState, 'joint_states', 10)
        
        self.timer = self.create_timer(0.1, self.publish_info)  # Timer for periodic publishing
        
        # Initialize gym environment
        env_cfg: LiftEnvCfg = parse_env_cfg(
            "Isaac-Lift-Cube-Franka-IK-Abs-v0",
            device=args_cli.device,
            num_envs=args_cli.num_envs,
            use_fabric=not args_cli.disable_fabric,
        )
        self.env = gym.make("Isaac-Lift-Cube-Franka-IK-Abs-v0", cfg=env_cfg)
        self.env.reset()

        self.actions = torch.zeros(self.env.unwrapped.action_space.shape, device=self.env.unwrapped.device)
        self.actions[:, 3] = 1.0
        self.desired_orientation = torch.zeros((self.env.unwrapped.num_envs, 4), device=self.env.unwrapped.device)
        self.desired_orientation[:, 1] = 1.0
        
        self.pick_sm = PickAndLiftSm(env_cfg.sim.dt * env_cfg.decimation, self.env.unwrapped.num_envs, self.env.unwrapped.device)
        self.i = 0
    def publish_info(self):
        # Step the environment
        msg = String()
        msg.data = f"Hello World {self.i}"
        self.i += 1

        self.test_publisher.publish(msg)
        
        with torch.inference_mode():
            dones = self.env.step(self.actions)[-2]

            # Observations
            ee_frame_sensor = self.env.unwrapped.scene["ee_frame"]
            tcp_rest_position = ee_frame_sensor.data.target_pos_w[..., 0, :].clone() - self.env.unwrapped.scene.env_origins
            tcp_rest_orientation = ee_frame_sensor.data.target_quat_w[..., 0, :].clone()
            
            # print("UNWRAPPED SCENE OBJECT DATA DICT")
            # pprint(self.env.unwrapped.scene["object"].__dict__)
            # print("This is the unwrapped scene object")
            # pprint(self.env.unwrapped.scene["object"])
 
            object_data: RigidObjectData = self.env.unwrapped.scene["object"].data
            object_position = object_data.root_pos_w - self.env.unwrapped.scene.env_origins
            desired_position = self.env.unwrapped.command_manager.get_command("object_pose")[..., :3]
            # Compute actions using the state machine
            self.actions = self.pick_sm.compute(
                torch.cat([tcp_rest_position, tcp_rest_orientation], dim=-1),
                torch.cat([object_position, self.desired_orientation], dim=-1),
                torch.cat([desired_position, self.desired_orientation], dim=-1),
            )

            # print(object_position)
            # print(tcp_rest_orientation)
            # print(type(tcp_rest_orientation))
            # Publish the end-effector pose
            ee_pose_msg = Pose()
            ee_pose_msgList = tcp_rest_position.tolist()
            ee_pose_msg.position.x = ee_pose_msgList[0][0]
            ee_pose_msg.position.y = ee_pose_msgList[0][1]
            ee_pose_msg.position.z = ee_pose_msgList[0][2]


            ee_orientation_msgList = tcp_rest_orientation.tolist()

            ee_pose_msg.orientation.x = ee_orientation_msgList[0][0]
            ee_pose_msg.orientation.y = ee_orientation_msgList[0][1]
            ee_pose_msg.orientation.z = ee_orientation_msgList[0][2]
            ee_pose_msg.orientation.w = ee_orientation_msgList[0][3]

            self.ee_pose_pub.publish(ee_pose_msg)

            # print("THESE ARE THE ACTIONS")
            # print(self.actions)

            action_list = self.actions.tolist()

            action_msg = JointState()  # Correct instantiation
            action_msg.header.frame_id = ''
            action_msg.header.stamp = self.get_clock().now().to_msg()
            nameArr = [f"panda_joint{i}" for i in range(1, 8)]
            nameArr.append("panda_finger_joint1")
            action_msg.name = nameArr

            action_msg.position = action_list[0]  # Set the data field
            self.action_pub.publish(action_msg)  # Publish the message

            # Reset state machine if necessary
            if dones.any():
                self.pick_sm.reset_idx(dones.nonzero(as_tuple=False).squeeze(-1))

def main(args=None):
    rclpy.init(args=args)

    info_pub = InfoPublisher()
    rclpy.spin(info_pub)

    # Shutdown
    info_pub.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
