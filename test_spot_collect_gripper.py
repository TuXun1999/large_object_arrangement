import argparse
import sys
import os
import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.frame_helpers import (
    get_odom_tform_body,
    HAND_FRAME_NAME,
    BODY_FRAME_NAME,
    ODOM_FRAME_NAME,
    GRAV_ALIGNED_BODY_FRAME_NAME,
    get_a_tform_b
)
import cv2
from scipy.spatial.transform import Rotation as R
import numpy as np
import time
import math
import json
import shutil
'''
Collect the gripper poses during the process
'''

def main(argv):
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)


    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('pose_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)


    
    
    try:
        gripper_pose_file_read = open("./gripper_poses_move_chair.json", "r")
        existing_trials = json.load(gripper_pose_file_read)
        gripper_pose_file_read.close()
    except:
        print("Error in reading the record of the experiments")
        existing_trials = None

    try: 
        gripper_poses = []
        while True:
            transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
            # Obtain the transformation of HAND frame in odom frame
            g = get_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME)
            h = np.array(g.to_matrix())

            gripper_poses.append(h)
            # Wait for some time so we can drive the robot to a new position.
            time.sleep(0.2)
    except KeyboardInterrupt:
        # Extend the existing list
        gripper_poses = np.array(gripper_poses)
        
        if existing_trials is not None:
            print(len(existing_trials))
            gripper_poses = gripper_poses.tolist()
            print(len(gripper_poses))
            existing_trials.append(gripper_poses)
            gripper_poses = existing_trials
            print(len(gripper_poses))
        else:
            print("Reading existing trials")
            gripper_poses = np.expand_dims(gripper_poses, axis=0)
            gripper_poses = gripper_poses.tolist()
        gripper_pose_file_write = open("./gripper_poses_move_chair.json", "w")
        json.dump(gripper_poses, gripper_pose_file_write, indent=4)
        gripper_pose_file_write.close()

    return True

if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)