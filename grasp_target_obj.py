import argparse
from argparse import Namespace
import sys
import os
import time
from scipy.optimize import fsolve

import bosdyn.client
import bosdyn.client.util
from bosdyn.client.image import ImageClient
from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client import frame_helpers
from bosdyn.api.basic_command_pb2 import RobotCommandFeedbackStatus
import bosdyn.api.basic_command_pb2 as basic_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import GRAV_ALIGNED_BODY_FRAME_NAME
from bosdyn.api import \
    geometry_pb2, arm_command_pb2, robot_command_pb2, synchronized_command_pb2, trajectory_pb2
from bosdyn.client.robot_command import \
    RobotCommandBuilder, RobotCommandClient, \
        block_until_arm_arrives, blocking_stand, blocking_selfright
from bosdyn.util import seconds_to_duration
from bosdyn.client.math_helpers import SE3Pose, Quat

import cv2
from scipy.spatial.transform import Rotation as R

import numpy as np




from torchvision.ops import box_convert

import open3d as o3d

from pose_estimation.pose_estimation import estimate_camera_pose
from grasp_pose_prediction.grasp_sq_mp import predict_grasp_pose_sq

from utils.image_process import point_select_from_custom_image, masked_image_generation

'''
Helper functions to move the robot
'''
def estimate_obj_pose_hand(bbox, image_response, distance):
    ## Estimate the target object pose (indicated by the bounding box) in hand frame
    bbox_center = [int((bbox[0] + bbox[2])/2), int((bbox[1] + bbox[3])/2)]
    pick_x, pick_y = bbox_center
    # Obtain the camera inforamtion
    camera_info = image_response.source
    
    w = camera_info.cols
    h = camera_info.rows
    fl_x = camera_info.pinhole.intrinsics.focal_length.x
    k1= camera_info.pinhole.intrinsics.skew.x
    cx = camera_info.pinhole.intrinsics.principal_point.x
    fl_y = camera_info.pinhole.intrinsics.focal_length.y
    k2 = camera_info.pinhole.intrinsics.skew.y
    cy = camera_info.pinhole.intrinsics.principal_point.y

    pinhole_camera_proj = np.array([
        [fl_x, 0, cx, 0],
        [0, fl_y, cy, 0],
        [0, 0, 1, 0]
    ])
    pinhole_camera_proj = np.float32(pinhole_camera_proj) # Converted into float type
    # Calculate the object's pose in hand camera frame
    initial_guess = [1, 1, 10]
    def equations(vars):
        x, y, z = vars
        eq = [
            pinhole_camera_proj[0][0] * x + pinhole_camera_proj[0][1] * y + pinhole_camera_proj[0][2] * z - pick_x * z,
            pinhole_camera_proj[1][0] * x + pinhole_camera_proj[1][1] * y + pinhole_camera_proj[1][2] * z - pick_y * z,
            x * x + y * y + z * z - distance * distance
        ]
        return eq

    root = fsolve(equations, initial_guess)
    # Correct the frame conventions in hand frame & pinhole model
    # pinhole model: z-> towards object, x-> rightward, y-> downward
    # hand frame in SPOT: x-> towards object, y->rightward
    result = SE3Pose(x=root[2], y=-root[0], z=-root[1], rot=Quat(w=1, x=0, y=0, z=0))
    return result

def compute_stand_location_and_yaw(vision_tform_target, robot_state_client,
                                distance_margin):

    # Compute drop-off location:
    #   Draw a line from Spot to the person
    #   Back up 2.0 meters on that line
    vision_tform_robot = frame_helpers.get_a_tform_b(
        robot_state_client.get_robot_state(
        ).kinematic_state.transforms_snapshot, frame_helpers.VISION_FRAME_NAME,
        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME)


    # Compute vector between robot and person
    robot_rt_person_ewrt_vision = [
        vision_tform_robot.x - vision_tform_target.x,
        vision_tform_robot.y - vision_tform_target.y,
        vision_tform_robot.z - vision_tform_target.z
    ]


    # Compute the unit vector.
    if np.linalg.norm(robot_rt_person_ewrt_vision) < 0.01:
        robot_rt_person_ewrt_vision_hat = vision_tform_robot.transform_point(1, 0, 0)
    else:
        robot_rt_person_ewrt_vision_hat = robot_rt_person_ewrt_vision / np.linalg.norm(
            robot_rt_person_ewrt_vision)


    # Starting at the person, back up meters along the unit vector.
    drop_position_rt_vision = [
        vision_tform_target.x +
        robot_rt_person_ewrt_vision_hat[0] * distance_margin,
        vision_tform_target.y +
        robot_rt_person_ewrt_vision_hat[1] * distance_margin,
        vision_tform_target.z +
        robot_rt_person_ewrt_vision_hat[2] * distance_margin
    ]


    # We also want to compute a rotation (yaw) so that we will face the person when dropping.
    # We'll do this by computing a rotation matrix with X along
    #   -robot_rt_person_ewrt_vision_hat (pointing from the robot to the person) and Z straight up:
    xhat = -robot_rt_person_ewrt_vision_hat
    zhat = [0.0, 0.0, 1.0]
    yhat = np.cross(zhat, xhat)
    mat = np.matrix([xhat, yhat, zhat]).transpose()
    heading_rt_vision = math_helpers.Quat.from_matrix(mat).to_yaw()

    return drop_position_rt_vision, heading_rt_vision

def get_walking_params(max_linear_vel, max_rotation_vel):
    max_vel_linear = geometry_pb2.Vec2(x=max_linear_vel, y=max_linear_vel)
    max_vel_se2 = geometry_pb2.SE2Velocity(linear=max_vel_linear,
                                        angular=max_rotation_vel)
    vel_limit = geometry_pb2.SE2VelocityLimit(max_vel=max_vel_se2)
    params = RobotCommandBuilder.mobility_params()
    params.vel_limit.CopyFrom(vel_limit)
    return params

def block_for_trajectory_cmd(command_client, cmd_id, timeout_sec=None, verbose=False):
    """Helper that blocks until a trajectory command reaches STATUS_AT_GOAL or a timeout is
        exceeded.
       Args:
        command_client: robot command client, used to request feedback
        cmd_id: command ID returned by the robot when the trajectory command was sent
        timeout_sec: optional number of seconds after which we'll return no matter what the
                        robot's state is.
        verbose: if we should print state at 10 Hz.
       Return values:
        True if reaches STATUS_AT_GOAL, False otherwise.
    """
    start_time = time.time()

    if timeout_sec is not None:
        end_time = start_time + timeout_sec
        now = time.time()

    while timeout_sec is None or now < end_time:
        feedback = command_client.robot_command_feedback(cmd_id)
        mobility_feedback = feedback.feedback.synchronized_feedback.mobility_command_feedback
        if mobility_feedback.status != RobotCommandFeedbackStatus.STATUS_PROCESSING:
            print('Failed to reach the goal')
            return False
        traj_feedback = mobility_feedback.se2_trajectory_feedback
        if (traj_feedback.status == traj_feedback.STATUS_AT_GOAL and
                traj_feedback.body_movement_status == traj_feedback.BODY_STATUS_SETTLED):
            print('Arrived at the location')
            return

        time.sleep(0.1)
        now = time.time()

    if verbose:
        print('block_for_trajectory_cmd: timeout exceeded.')

def move_gripper(grasp_pose_gripper, robot_state_client, command_client):
    '''
    Move the gripper to the desired pose, given in the local frame of HAND_frame
    Input:
    grasp_pose_gripper: the pose of the gripper in the current hand frame
    robot_state_client: the state client of the robot
    command_client: the client to send the commands
    '''

    # Transform the desired pose from the moving body frame to the odom frame.
    robot_state = robot_state_client.get_robot_state()
    odom_T_hand = frame_helpers.get_a_tform_b(\
                robot_state.kinematic_state.transforms_snapshot,
                frame_helpers.ODOM_FRAME_NAME, \
                frame_helpers.HAND_FRAME_NAME)
    odom_T_hand = odom_T_hand.to_matrix()

    # Build the SE(3) pose of the desired hand position in the moving body frame.
    odom_T_target = odom_T_hand@grasp_pose_gripper
    odom_T_target = math_helpers.SE3Pose.from_matrix(odom_T_target)
    
    # duration in seconds
    seconds = 5.0

    # Create the arm command.
    arm_command = RobotCommandBuilder.arm_pose_command(
        odom_T_target.x, odom_T_target.y, odom_T_target.z, odom_T_target.rot.w, odom_T_target.rot.x,
        odom_T_target.rot.y, odom_T_target.rot.z, \
            frame_helpers.ODOM_FRAME_NAME, seconds)

    # Tell the robot's body to follow the arm
    follow_arm_command = RobotCommandBuilder.follow_arm_command()

    # Combine the arm and mobility commands into one synchronized command.
    command = RobotCommandBuilder.build_synchro_command(follow_arm_command, arm_command)

    # Send the request
    move_command_id = command_client.robot_command(command)
    print('Moving arm to position.')

    block_until_arm_arrives(command_client, move_command_id, 30.0)


def grasp_target_obj(options):
    if (options.image_source != "hand_color_image"):
        print("Currently Only Support Hand Camera!")
        return True
    ## Fundamental Setup of the robotic platform
    bosdyn.client.util.setup_logging(options.verbose)
    # Authenticate with the robot
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    # Create the clients 
    image_client = robot.ensure_client(ImageClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    # Verification before the formal task
    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    # Start of the formal task
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        ## Part 0: Start the robot and Power it on
        # Power on the robot
        robot.logger.info('Powering on robot... This may take a several seconds.')
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'
        robot.logger.info('Robot powered on.')

        # Tell the robot to stand up. 
        robot.logger.info('Commanding robot to stand...')
        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')

        if options.lower_mode:
            # If the robot needs to grasp a slightly smaller object, 
            # lower down the platform as well
            
            # Move the arm along a simple trajectory.
            x = 0.85  # a reasonable position in front of the robot
            y1 = 0  # centered
            z = 0.35 # at the body's height

            # Use the same rotation as the robot's body.
            rotation = math_helpers.Quat(w=0.9659258, x=0, y=0.258819, z=0)

            # Define times (in seconds) for each point in the trajectory.
            t_first_point = 0  # first point starts at t = 0 for the trajectory.

            # Build the points in the trajectory.
            hand_pose1 = math_helpers.SE3Pose(x=x, y=y1, z=z, rot=rotation)

            # Build the points by combining the pose and times into protos.
            traj_point1 = trajectory_pb2.SE3TrajectoryPoint(
                pose=hand_pose1.to_proto(), time_since_reference=seconds_to_duration(t_first_point))

            # Build the trajectory proto by combining the points.
            hand_traj = trajectory_pb2.SE3Trajectory(points=[traj_point1])

            # Build the command by taking the trajectory and specifying the frame it is expressed
            # in.
            #
            # In this case, we want to specify the trajectory in the body's frame, so we set the
            # root frame name to the flat body frame.
            arm_cartesian_command = arm_command_pb2.ArmCartesianCommand.Request(
                pose_trajectory_in_task=hand_traj, root_frame_name=GRAV_ALIGNED_BODY_FRAME_NAME)

            # Pack everything up in protos.
            arm_command = arm_command_pb2.ArmCommand.Request(
                arm_cartesian_command=arm_cartesian_command)

            synchronized_command = synchronized_command_pb2.SynchronizedCommand.Request(
                arm_command=arm_command)

            robot_command = robot_command_pb2.RobotCommand(synchronized_command=synchronized_command)

            robot.logger.info('Sending ARM trajectory command to lower down the gripper...')

            # Send the trajectory to the robot.
            cmd_id = command_client.robot_command(robot_command)
            time.sleep(4)
        # Command the robot to open its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)
        # Send the trajectory to the robot.
        cmd_id = command_client.robot_command(robot_command)

        ## Create an UI to take the user input & find the target object
        target = input("What do you want to grasp?")
 
        time.sleep(1)
        ## Part 2: Take an image & Estimate Pose
        depth_image_source = "hand_depth_in_hand_color_frame"
        image_responses = image_client.get_image_from_sources(\
            [options.image_source, depth_image_source])
        dtype = np.uint8

        img = np.frombuffer(image_responses[0].shot.image.data, dtype=dtype)
        img = cv2.imdecode(img, -1)
        image_name = os.path.join(options.nerf_model, "pose_estimation.jpg")
        cv2.imwrite(image_name, img)
        
        # Call the function to generate the masked image
        masked_image_generation(options.nerf_model, img, image_name, target)
        
        # Read the preprocessed data
        images_reference_list = \
            ["/images/" + x for x in \
            os.listdir(options.nerf_model + "/images")]# All images under foler "images"
        mesh_filename = os.path.join(options.nerf_model , "target_obj.obj")
        

        mesh = o3d.io.read_triangle_mesh(mesh_filename)
        
        camera_pose_est, camera_proj_img, nerf_scale = \
                estimate_camera_pose("/pose_estimation_masked_rgb.png", options.nerf_model, \
                                     images_reference_list, \
                         mesh, None, \
                         image_type = 'outdoor', visualization = options.visualization)

        # The reference to the file storing the previous predicted superquadric parameters
        suffix = options.nerf_model.split("/")[-1]
        stored_stats_filename = "./grasp_pose_prediction/Marching_Primitives/sq_data/" + suffix + ".p"
        csv_filename = os.path.join(options.nerf_model, "target_obj.csv")

        ## Part 3: Grasp the target object
        ## Specify Gripper Attributes
        # NOTE: These are approximate stats for SPOT robot
        # Gripper Attributes
        gripper_width = 0.09 * nerf_scale
        gripper_length = 0.09 * nerf_scale
        gripper_thickness = 0.089 * nerf_scale
        gripper_attr = {"Type": "Parallel", "Length": gripper_length, \
                        "Width": gripper_width, "Thickness": gripper_thickness}
        
        
        # Correct the frame conventions in camera & gripper of SPOT
        gripper_pose_current = camera_pose_est@np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
        # Correct the frame conventions in camera & gripper of SPOT
        gripper_z_local = (gripper_pose_current[:3, :3].T)[0:3, 2]

        # The gripper's z axis should be vertical
        angle = np.arccos(np.dot([0, 0, 1], gripper_z_local))
        gripper_z_local_cross = np.cross(np.array([0, 0, 1]), gripper_z_local)
        gripper_z_local_cross /= np.linalg.norm(gripper_z_local_cross)
        camera_gripper_correction = R.from_quat(\
            [gripper_z_local_cross[0] * np.sin(angle/2), \
            gripper_z_local_cross[1] * np.sin(angle/2), \
            gripper_z_local_cross[2] * np.sin(angle/2), np.cos(angle/2)]).as_matrix()
        camera_gripper_correction = np.vstack(\
            (np.hstack((camera_gripper_correction, np.array([[0], [0], [0]]))), np.array([0, 0, 0, 1])))
        gripper_pose_current = gripper_pose_current@camera_gripper_correction

        ##############
        # NOTE: 
        # Manually force the height of the estimated gripper to match the measurement
        ##############
        # This is the correction for chair2_real
        gripper_pose_current[2, 3] = 12*0.0254*nerf_scale

        # Optionally, pull the chair backward
        if options.back:
            # Find the gripper pose in body frame
            body_T_hand = frame_helpers.get_a_tform_b(\
                        robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
                        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                        frame_helpers.HAND_FRAME_NAME)
            body_T_hand = body_T_hand.to_matrix()
            gripper_pose_current[0:3, 3] /= nerf_scale
            # Find the chair pose in current body frame
            chair_pose_body = body_T_hand@np.linalg.inv(gripper_pose_current)
            gripper_pose_current[0:3, 3] *= nerf_scale
            print("gripper current")
            # Find the target pose of the chair
            chair_pose_target = np.array([
                [1, 0, 0, 0.5],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]
            ])

            # Find the transformation in body frame (chair frame)
            chair_pose_tran_body = np.linalg.inv(chair_pose_body)@chair_pose_target
            # Assume the gripper will be applied the same transformation
            body_T_hand_goal = body_T_hand@chair_pose_tran_body
            # Convert it into the odom frame
            odom_T_body = frame_helpers.get_a_tform_b(\
                        robot_state_client.get_robot_state().kinematic_state.transforms_snapshot,
                        frame_helpers.ODOM_FRAME_NAME, \
                        frame_helpers.BODY_FRAME_NAME)
            odom_T_body = odom_T_body.to_matrix()
            odom_T_hand_goal = odom_T_body@body_T_hand_goal
            odom_T_hand_goal = math_helpers.SE3Pose.from_matrix(odom_T_hand_goal)
            print("odom")

        point_select = None
        if options.click:
            user_click_point, _ = point_select_from_custom_image(\
                os.path.join(options.nerf_model, "pose_estimation.jpg"),\
                camera_proj_img,\
                camera_pose_est,\
                mesh)
            point_select = user_click_point

        if options.method == "sq_split":
             # Predict the grasp poses
            grasp_pose_options = \
                Namespace(\
                    train=False, normalize = False, store=False, \
                        visualization=options.visualization)
            grasp_poses_world = predict_grasp_pose_sq(gripper_pose_current, \
                                mesh, csv_filename, [1.0, 0.0],\
                                stored_stats_filename, gripper_attr, grasp_pose_options, \
                                point_select=point_select, region_select=options.region_select)

        # Determine the optimal pose based on the minimum rotation
        tran_norm_min = np.Inf
        grasp_pose_gripper = np.eye(4)
        # Further filter out predicted poses to obtain the best one
        for grasp_pose in grasp_poses_world:
            # Correct the frame convention between the gripper of SPOT
            # & the one used in grasp pose prediction
            grasp_pose = grasp_pose@np.array([[-1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])

            # Evaluate the transformation between each predicted grasp pose
            # and the current gripper pose
            rot1 = grasp_pose[:3, :3]
            rot2 = gripper_pose_current[:3, :3]
            tran = rot2.T@rot1

            # Consider the symmetry along x-axis
            r = R.from_matrix(tran[:3, :3])
            r_zyx = r.as_euler('zyx', degrees=True)

            # Consider the symmetry along x-axis
            r = R.from_matrix(tran[:3, :3])
            r_xyz = r.as_euler('xyz', degrees=True)

            # Rotation along x-axis is symmetric
            if r_xyz[0] > 90:
                r_xyz[0] = r_xyz[0] - 180
            elif r_xyz[0] < -90:
                r_xyz[0] = r_xyz[0] + 180
            tran = R.from_euler('xyz', r_xyz, degrees=True).as_matrix()
            
            # Find the one with the minimum rotation
            tran_norm = np.linalg.norm(tran - np.eye(3))
            if tran_norm < tran_norm_min:
                tran_norm_min = tran_norm
                grasp_pose_gripper = grasp_pose
        print("====Selected Grasp Pose=====")
        print(nerf_scale)
        print(grasp_pose_gripper)

        if options.visualization:
            print("Visualize the final grasp result")
            # Create the window to display the grasp
            vis= o3d.visualization.Visualizer()
            vis.create_window()
            vis.add_geometry(mesh)

            # The world frame
            fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
            # Visualize the gripper as well
            arm_end = np.array([-gripper_length, 0, 0])
            center = np.array([0, 0, 0])
            elbow1 = np.array([0, 0, gripper_width/2])
            elbow2 = np.array([0, 0, -gripper_width/2])
            tip1 = np.array([gripper_length, 0, gripper_width/2])
            tip2 = np.array([gripper_length, 0, -gripper_width/2])

            # Construct the gripper
            gripper_points = np.array([
                center,
                arm_end,
                elbow1,
                elbow2,
                tip1,
                tip2
            ])
            gripper_lines = [
                [1, 0],
                [2, 3],
                [2, 4],
                [3, 5]
            ]
            gripper_start = o3d.geometry.TriangleMesh.create_coordinate_frame()
            gripper_start.scale(20/64, [0, 0, 0])
            gripper_start.transform(gripper_pose_current)

            grasp_pose_lineset_start = o3d.geometry.LineSet()
            grasp_pose_lineset_start.points = o3d.utility.Vector3dVector(gripper_points)
            grasp_pose_lineset_start.lines = o3d.utility.Vector2iVector(gripper_lines)
            grasp_pose_lineset_start.transform(gripper_pose_current)
            grasp_pose_lineset_start.paint_uniform_color((1, 0, 0))
            
            gripper_end = o3d.geometry.TriangleMesh.create_coordinate_frame()
            gripper_end.scale(20/64, [0, 0, 0])
            gripper_end.transform(grasp_pose_gripper)

            grasp_pose_lineset_end = o3d.geometry.LineSet()
            grasp_pose_lineset_end.points = o3d.utility.Vector3dVector(gripper_points)
            grasp_pose_lineset_end.lines = o3d.utility.Vector2iVector(gripper_lines)
            grasp_pose_lineset_end.transform(grasp_pose_gripper)
            grasp_pose_lineset_end.paint_uniform_color((0, 1, 0))

            # Optionally, visualize the goal chair pose and goal gripper pose
            if options.back:
                chair_goal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
                
                chair_goal_vis_pose = chair_pose_tran_body
                chair_goal_vis_pose[0:3, 3] *= nerf_scale
                chair_goal_frame.transform(chair_goal_vis_pose)
                chair_goal_frame.paint_uniform_color((0, 1, 0))
                vis.add_geometry(chair_goal_frame)

                hand_goal_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
                hand_goal_frame.scale(20/64, [0, 0, 0])
                hand_goal_vis_pose = np.linalg.inv(chair_pose_body)@body_T_hand_goal
                hand_goal_vis_pose[0:3, 3] *= nerf_scale
                hand_goal_frame.transform(hand_goal_vis_pose)
                hand_goal_frame.paint_uniform_color((0, 0, 1))
                vis.add_geometry(hand_goal_frame)
            
            vis.add_geometry(gripper_start)
            vis.add_geometry(gripper_end)
            vis.add_geometry(grasp_pose_lineset_end)
            vis.add_geometry(grasp_pose_lineset_start)
            vis.add_geometry(fundamental_frame)
            vis.run()

            # Close all windows
            vis.destroy_window()

        # Command the robot to grasp
        print(gripper_pose_current)
        print(grasp_pose_gripper)
        grasp_pose_gripper = np.linalg.inv(gripper_pose_current)@grasp_pose_gripper
        # Consider the symmetry along x-axis
        r = R.from_matrix(grasp_pose_gripper[:3, :3])
        r_xyz = r.as_euler('xyz', degrees=True)

        # Rotation along x-axis is symmetric
        if r_xyz[0] > 90:
            r_xyz[0] = r_xyz[0]- 180
        elif r_xyz[0] < -90:
            r_xyz[0] = r_xyz[0] + 180
        grasp_pose_gripper[:3, :3]= R.from_euler('xyz', r_xyz, degrees=True).as_matrix()
        print(grasp_pose_gripper)

        ################
        # NOTE: 
        # Experiments show that the distance along x-axis might need to be extended or shortened. 
        # You can correct the moving distance of the gripper here
        ################
        # This is the correction used for chair2_real
        grasp_pose_gripper[0, 3] = grasp_pose_gripper[0, 3] + 0.17*nerf_scale



        grasp_pose_gripper[0:3, 3] = grasp_pose_gripper[0:3, 3] / nerf_scale


        # Command the robot to place the gripper at the desired pose
        move_gripper(grasp_pose_gripper, robot_state_client, command_client)
        
        # Close the gripper
        robot_command = RobotCommandBuilder.claw_gripper_close_command()
        cmd_id = command_client.robot_command(robot_command)

        time.sleep(1.0)
        # Whether to go back after the grasp
        if options.back:
            # Command the robot to fix up the relative transformation between
            # its gripper and the body
            # Transform the desired pose from the moving body frame to the odom frame.
            robot_state = robot_state_client.get_robot_state()
            body_T_hand = frame_helpers.get_a_tform_b(\
                        robot_state.kinematic_state.transforms_snapshot,
                        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                        frame_helpers.HAND_FRAME_NAME)

            # duration in seconds
            seconds = 5.0

            # Create the arm command & send it (theoretically, the robot shouldn't move)
            arm_command = RobotCommandBuilder.arm_pose_command(
                body_T_hand.x, body_T_hand.y, body_T_hand.z, body_T_hand.rot.w, body_T_hand.rot.x,
                body_T_hand.rot.y, body_T_hand.rot.z, \
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, seconds)
            command_client.robot_command(arm_command)
            time.sleep(0.5)
            
            # Find the goal gripper pose in body frame
            body_T_odom = frame_helpers.get_a_tform_b(\
                        robot_state.kinematic_state.transforms_snapshot,
                        frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, \
                        frame_helpers.ODOM_FRAME_NAME)
            # Correct the scale
            odom_T_hand_goal.x /= nerf_scale
            odom_T_hand_goal.y /= nerf_scale
            odom_T_hand_goal.z /= nerf_scale

            # The goal hand pose in current body frame
            body_T_hand_goal = body_T_odom.mult(odom_T_hand_goal)
            arm_command = RobotCommandBuilder.arm_pose_command(
                body_T_hand_goal.x, body_T_hand_goal.y, body_T_hand_goal.z, body_T_hand_goal.rot.w, body_T_hand_goal.rot.x,
                body_T_hand_goal.rot.y, body_T_hand_goal.rot.z, \
                    frame_helpers.GRAV_ALIGNED_BODY_FRAME_NAME, seconds)
            command_client.robot_command(arm_command)
            time.sleep(5.0)
        
        input("Waiting for the user to stop")

        # Release the gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_command()
        cmd_id = command_client.robot_command(robot_command)
        return True

def main(argv):
    """Command line interface."""
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--image-source', help='Get image from source(s), \
                        only hand camera is supported so far', default='hand_color_image')
    
    # TODO: Instead of depending on the user, the system should automatically figure
    # out which nerf model to use
    parser.add_argument('--nerf_model', help='The directory containing the preprocessed\
                         NeRF model to use')
    
    # Visualization in open3d
    parser.add_argument(
        '--visualization', action = 'store_true'
    )

    # Whether to lower down the gripper at first (for slightly smaller object)
    parser.add_argument(
        '--lower-mode', action = 'store_true'
    )

    parser.add_argument('--method', default='sq_split', \
                        help='Which method to use in determining grasp poses')
    
    # Whether to go back after the grasp
    parser.add_argument('--back', action = 'store_true', \
                        help="Whether to pull the chair backward for a successful grasp")
    parser.add_argument('--click', action = 'store_true', \
                        help="Whether to allow the user to click a point to grasp (Still under progress; not guaranted to work)")
    parser.add_argument('--region_select', action = 'store_true',\
                        help="Whether to select a region indexed by numbers to grasp")
    options = parser.parse_args(argv)
    try:
        grasp_target_obj(options)
        return True
    except Exception as exc:  # pylint: disable=broad-except
        logger = bosdyn.client.util.get_logger()
        logger.exception('Threw an exception')
        return False
if __name__ == "__main__":
    if not main(sys.argv[1:]):
        sys.exit(1)