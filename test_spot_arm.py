"""
The program to test the arm control
"""

import argparse
import sys
import time
import numpy as np
import json
import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, BODY_FRAME_NAME,\
    ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
from robot import SPOT
import threading
kill_thread = threading.Event()
def collect_robot_poses(filename, robot):
    try:
        gripper_pose_file_read = open(filename, "r")
        existing_trials = json.load(gripper_pose_file_read)
        gripper_pose_file_read.close()
    except:
        print("Error in reading the record of the experiments")
        existing_trials = None


    gripper_body_poses = []
    while True and not kill_thread.is_set():
        transforms = robot.state_client.get_robot_state().kinematic_state.transforms_snapshot
        # Obtain the transformation of HAND frame in odom frame
        g = get_a_tform_b(transforms, BODY_FRAME_NAME, HAND_FRAME_NAME)
        h = np.array(g.to_matrix())
        b = get_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        c = np.array(b.to_matrix())
        pose = np.hstack((h, c))
        gripper_body_poses.append(pose)

        # Wait for some time so we can drive the robot to a new position.
        time.sleep(0.2)

    # Extend the existing list or Create a new list
    gripper_body_poses = np.array(gripper_body_poses)
    
    if existing_trials is not None:
        print(len(existing_trials))
        gripper_body_poses = gripper_body_poses.tolist()
        print(len(gripper_body_poses))
        existing_trials.append(gripper_body_poses)
        gripper_body_poses = existing_trials
        print(len(gripper_body_poses))
    else:
        print("Creating new trials")
        gripper_body_poses = np.expand_dims(gripper_body_poses, axis=0)
        gripper_body_poses = gripper_body_poses.tolist()
    gripper_pose_file_write = open(filename, "w")
    json.dump(gripper_body_poses, gripper_pose_file_write, indent=4)
    gripper_pose_file_write.close()
def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    robot = SPOT(options)
    robot.lease_alive()
    robot.power_on_stand()
    #robot.arm_control_wasd()

    robot.grasp_obj_camera()
    input('Press enter to exit')
    # TODO: Debug the teleoperatation system for the robot to move the chair backwards
    filename = "./gripper_poses_move_chair.json"

    t1 = threading.Thread(target = collect_robot_poses, args=(filename, robot))
    kill_thread.clear()
    t1.start()
    robot.teleoperation()
    kill_thread.set()
    robot.lease_return()

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)