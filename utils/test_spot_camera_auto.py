# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

"""Simple image capture tutorial."""

import argparse
import sys
import time
import cv2
import numpy as np
from scipy import ndimage

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import image_pb2
from bosdyn.api import gripper_camera_param_pb2
from bosdyn.client.image import ImageClient, build_image_request
from bosdyn.client.gripper_camera_param import GripperCameraParamClient
from bosdyn.client.robot_command import \
    RobotCommandBuilder, RobotCommandClient, \
        block_until_arm_arrives, blocking_stand, blocking_selfright

ROTATION_ANGLE = {
    'back_fisheye_image': 0,
    'frontleft_fisheye_image': -78,
    'frontright_fisheye_image': -102,
    'left_fisheye_image': 0,
    'right_fisheye_image': 180,
    'hand_color_image': 0
}


def pixel_format_type_strings():
    names = image_pb2.Image.PixelFormat.keys()
    return names[1:]


def pixel_format_string_to_enum(enum_string):
    return dict(image_pb2.Image.PixelFormat.items()).get(enum_string)


def hand_image_resolution(gripper_param_client, resolution):
    camera_mode = None
    if resolution is not None:
        if resolution == '640x480':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_640_480
        elif resolution == '1280x720':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1280_720
        elif resolution == '1920x1080':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_1920_1080
        elif resolution == '3840x2160':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_3840_2160
        elif resolution == '4096x2160':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4096_2160
        elif resolution == '4208x3120':
            camera_mode = gripper_camera_param_pb2.GripperCameraParams.MODE_4208_3120

    request = gripper_camera_param_pb2.GripperCameraParamRequest(
        params=gripper_camera_param_pb2.GripperCameraParams(camera_mode=camera_mode))
    response = gripper_param_client.set_camera_params(request)

def main(argv):
    # Parse args
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    parser.add_argument('--list', help='list image sources', action='store_true')
    parser.add_argument('--auto-rotate', help='rotate right and front images to be upright',
                        action='store_true')
    parser.add_argument('--image-sources', help='Get image from source(s)', action='append')
    parser.add_argument('--image-service', help='Name of the image service to query.',
                        default=ImageClient.default_service_name)
    parser.add_argument(
        '--pixel-format', choices=pixel_format_type_strings(),
        help='Requested pixel format of image. If supplied, will be used for all sources.')
    parser.add_argument(
        '--resolution', default='640x480',
        help='Resolution of the hand camera. Options are 640x480, 1280x720, 1920x1080, 3840x2160, 4096x2160, 4208x3120'
    )
    options = parser.parse_args(argv)

    # Create robot object with an image client.
    sdk = bosdyn.client.create_standard_sdk('image_capture')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()
    robot.time_sync.wait_for_sync()

    gripper_param_client = robot.ensure_client(GripperCameraParamClient.default_service_name)
    # Optionally set the resolution of the hand camera
    if 'hand_color_image' in options.image_sources:
        hand_image_resolution(gripper_param_client, options.resolution)
    image_client = robot.ensure_client(options.image_service)
    
    lease_client = robot.ensure_client(bosdyn.client.lease.LeaseClient.default_service_name)

    # Verification before the formal task
    assert robot.has_arm(), 'Robot requires an arm to run this example.'
    # Verify the robot is not estopped and that an external application has registered and holds
    # an estop endpoint.
    assert not robot.is_estopped(), 'Robot is estopped. Please use an external E-Stop client, ' \
                                    'such as the estop SDK example, to configure E-Stop.'

    # Start of the formal task
    with bosdyn.client.lease.LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        # Power on the robot
        robot.power_on(timeout_sec=20)
        assert robot.is_powered_on(), 'Robot power on failed.'

        command_client = robot.ensure_client(RobotCommandClient.default_service_name)
        blocking_stand(command_client, timeout_sec=10)
        robot.logger.info('Robot standing.')
        # Command the robot to open its gripper
        robot_command = RobotCommandBuilder.claw_gripper_open_fraction_command(1)

        # Send the trajectory to the robot.
        cmd_id = command_client.robot_command(robot_command)

        time.sleep(2)
        # Raise exception if no actionable argument provided
        if not options.list and not options.image_sources:
            parser.error('Must provide actionable argument (list or image-sources).')

        # Optionally list image sources on robot.
        if options.list:
            image_sources = image_client.list_image_sources()
            print('Image sources:')
            for source in image_sources:
                print('\t' + source.name)

        # Optionally capture one or more images.
        if options.image_sources:
            # Capture and save images to disk
            pixel_format = pixel_format_string_to_enum(options.pixel_format)
            image_request = [
                build_image_request(source, pixel_format=pixel_format)
                for source in options.image_sources
            ]
            image_responses = image_client.get_image(image_request)

            for image in image_responses:
                num_bytes = 1  # Assume a default of 1 byte encodings.
                if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_DEPTH_U16:
                    dtype = np.uint16
                    extension = '.png'
                else:
                    if image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGB_U8:
                        num_bytes = 3
                    elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_RGBA_U8:
                        num_bytes = 4
                    elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U8:
                        num_bytes = 1
                    elif image.shot.image.pixel_format == image_pb2.Image.PIXEL_FORMAT_GREYSCALE_U16:
                        num_bytes = 2
                    dtype = np.uint8
                    extension = '.jpg'

                img = np.frombuffer(image.shot.image.data, dtype=dtype)
                if image.shot.image.format == image_pb2.Image.FORMAT_RAW:
                    try:
                        # Attempt to reshape array into a RGB rows X cols shape.
                        img = img.reshape((image.shot.image.rows, image.shot.image.cols, num_bytes))
                    except ValueError:
                        # Unable to reshape the image data, trying a regular decode.
                        img = cv2.imdecode(img, -1)
                else:
                    img = cv2.imdecode(img, -1)

                if options.auto_rotate:
                    img = ndimage.rotate(img, ROTATION_ANGLE[image.source.name])

                # Save the image from the GetImage request to the current directory with the filename
                # matching that of the image source.
                image_saved_path = image.source.name
                image_saved_path = image_saved_path.replace(
                    '/', '')  # Remove any slashes from the filename the image is saved at locally.
                cv2.imwrite(image_saved_path + extension, img)
    return True


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)