# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Tutorial to show how to use Spot's arm.
"""

import argparse
import sys
import time
import numpy as np

import bosdyn.api.gripper_command_pb2
import bosdyn.client
import bosdyn.client.lease
import bosdyn.client.util
from bosdyn.api import arm_command_pb2, geometry_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME, \
    ODOM_FRAME_NAME, HAND_FRAME_NAME, get_a_tform_b
from bosdyn.client.robot_command import (RobotCommandBuilder, RobotCommandClient,
                                         block_until_arm_arrives, blocking_stand)
from bosdyn.client.robot_state import RobotStateClient
import json
def robot_gripper_move_square(robot, command_client, robot_state_client):
    '''
    The helper function to command the robot to move its gripper along a square
    '''
    waypoints = [
                [0.75, 0, 0.25],
                [0.75, 0, 0.45],
                [0.95, 0, 0.45],
                [0.95, 0, 0.25]
            ]

    i = 0
    gripper_poses_whole = []
    for waypoint in waypoints:
        # Build a position to move the arm to (in meters, relative to and expressed in the gravity aligned body frame).
        x, y, z = waypoint
        hand_ewrt_flat_body = geometry_pb2.Vec3(x=x, y=y, z=z)

        # Rotation as a quaternion
        qw = 1
        qx = 0
        qy = 0
        qz = 0
        flat_body_Q_hand = geometry_pb2.Quaternion(w=qw, x=qx, y=qy, z=qz)

        flat_body_T_hand = geometry_pb2.SE3Pose(position=hand_ewrt_flat_body,
                                                rotation=flat_body_Q_hand)

        robot_state = robot_state_client.get_robot_state()
        odom_T_flat_body = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, GRAV_ALIGNED_BODY_FRAME_NAME)

        odom_T_hand = odom_T_flat_body * math_helpers.SE3Pose.from_proto(flat_body_T_hand)

        # duration in seconds
        seconds = 2

        arm_command = RobotCommandBuilder.arm_pose_command(
            odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
            odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

        # Make the open gripper RobotCommand
        gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

        # Combine the arm and gripper commands into one RobotCommand
        command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

        # Send the request
        cmd_id = command_client.robot_command(command)
        robot.logger.info('Moving arm to position: ' + str(i))

        # Wait until the arm arrives at the goal & record the gripper poses along the trajectory
        gripper_poses = block_until_arm_arrives_with_prints(robot, command_client, cmd_id, record=True)
        gripper_poses_whole = gripper_poses_whole + gripper_poses
        
        # Store the gripper poses inside a file

        i = i + 1
    return gripper_poses_whole
def hello_arm(config):
    """A simple example of using the Boston Dynamics API to command Spot's arm."""

    # See hello_spot.py for an explanation of these lines.
    bosdyn.client.util.setup_logging(config.verbose)

    sdk = bosdyn.client.create_standard_sdk('HelloSpotClient')
    robot = sdk.create_robot(config.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.time_sync.wait_for_sync()

    assert robot.has_arm(), 'Robot requires an arm to run this example.'

    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)

    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Now, we are ready to power on the robot. This call will block until the power
        # is on. Commands would fail if this did not happen. We can also check that the robot is
        # powered at any point.
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. The command service is used to issue commands to a robot.
        # The set of valid commands for a robot depends on hardware configuration. See
        # RobotCommandBuilder for more detailed examples on command building. The robot
        # command service requires timesync between the robot and the client.
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        # Move the arm to a spot in front of the robot, and open the gripper.
        input("==PRESS ENTER TO CONTINUE==")
        # Make the arm pose RobotCommand
        option = "custom"
        iteration = 20
        if option == "square":
            for i in range(iteration):
                gripper_poses = robot_gripper_move_square(robot, command_client, robot_state_client)
                # Record the full trajectory of gripper poses in the file
                try:
                    gripper_pose_file_read = open("./gripper_poses.json", "r")
                    existing_trials = json.load(gripper_pose_file_read)
                    gripper_pose_file_read.close()
                except:
                    print("Error in reading the record of the experiments. Creating new file")
                    existing_trials = None

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
                gripper_pose_file_write = open("./gripper_poses.json", "w")
                json.dump(gripper_poses, gripper_pose_file_write, indent=4)
                gripper_pose_file_write.close()

                # Move the gripper back to the default place
                stow_cmd = RobotCommandBuilder.arm_stow_command()
                command_client.robot_command(stow_cmd)
                time.sleep(1)

        elif option == "custom":
            waypoints = np.load(open("./gripper_pose_sample.npy", 'rb'))
            i = 0
            print("===Test===")
            print(waypoints[5])
            pose_prev = waypoints[0]
            for pose in waypoints[5:]:
                # duration in seconds
                seconds = 0.5
                old_hand_T_hand = math_helpers.SE3Pose.from_matrix(np.linalg.inv(pose_prev)@pose)
                robot_state = robot_state_client.get_robot_state()
                odom_T_old_hand = get_a_tform_b(robot_state.kinematic_state.transforms_snapshot,
                                        ODOM_FRAME_NAME, HAND_FRAME_NAME)

                pose_prev = pose
                odom_T_hand = odom_T_old_hand * old_hand_T_hand
                arm_command = RobotCommandBuilder.arm_pose_command(
                    odom_T_hand.x, odom_T_hand.y, odom_T_hand.z, odom_T_hand.rot.w, odom_T_hand.rot.x,
                    odom_T_hand.rot.y, odom_T_hand.rot.z, ODOM_FRAME_NAME, seconds)

                # Make the open gripper RobotCommand
                gripper_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1.0)

                # Combine the arm and gripper commands into one RobotCommand
                command = RobotCommandBuilder.build_synchro_command(gripper_command, arm_command)

                # Send the request
                cmd_id = command_client.robot_command(command)
                robot.logger.info('Moving arm to position: ' + str(i))

                # Wait until the arm arrives at the goal & Record the state
                block_until_arm_arrives_with_prints(robot, command_client, cmd_id)
                i = i + 1


        robot.logger.info('Done.')
        input("===Wait to continue==")

        # Power the robot off. By specifying "cut_immediately=False", a safe power off command
        # is issued to the robot. This will attempt to sit the robot before powering off.
        robot.power_off(cut_immediately=False, timeout_sec=20)
        assert not robot.is_powered_on(), 'Robot power off failed.'
        robot.logger.info('Robot safely powered off.')

def record_gripper_pose(robot_state_client, gripper_poses):
    """
    The function to record the current gripper pose to the file
    """
    transforms = robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
    # Obtain the transformation of HAND frame in odom frame
    g = get_a_tform_b(transforms, ODOM_FRAME_NAME, HAND_FRAME_NAME)
    h = np.array(g.to_matrix())

    gripper_poses.append(h)
    # Wait for some time so we can drive the robot to a new position.
    time.sleep(0.1)
    
def block_until_arm_arrives_with_prints(robot, command_client, cmd_id, record=False):
    """Block until the arm arrives at the goal and print the distance remaining.
        Note: a version of this function is available as a helper in robot_command
        without the prints.
    """
    gripper_poses = []
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    while True:
        feedback_resp = command_client.robot_command_feedback(cmd_id)
        measured_pos_distance_to_goal = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_pos_distance_to_goal
        measured_rot_distance_to_goal = feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.measured_rot_distance_to_goal
        robot.logger.info('Distance to go: %.2f meters, %.2f radians;',
                          measured_pos_distance_to_goal, measured_rot_distance_to_goal)
        if record:
            robot.logger.info('Gripper pose recorded')
            record_gripper_pose(robot_state_client, gripper_poses)
        if feedback_resp.feedback.synchronized_feedback.arm_command_feedback.arm_cartesian_feedback.status == arm_command_pb2.ArmCartesianCommand.Feedback.STATUS_TRAJECTORY_COMPLETE:
            robot.logger.info('Move complete.')
            break
        time.sleep(0.1)
    return gripper_poses


def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    options = parser.parse_args(argv)
    try:
        hello_arm(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)