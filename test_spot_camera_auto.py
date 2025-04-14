# Copyright (c) 2023 Boston Dynamics, Inc.  All rights reserved.
#
# Downloading, reproducing, distributing or otherwise using the SDK Software
# is subject to the terms and conditions of the Boston Dynamics Software
# Development Kit License (20191101-BDSDK-SL).

import argparse
import sys
import time
import cv2
import os
import numpy as np
from scipy import ndimage
from spot import SPOT, pixel_format_type_strings
import bosdyn
from bosdyn.client.image import ImageClient

from GroundingDINO.groundingdino.util.inference import \
    load_model, load_image, predict, annotate
from segment_anything import sam_model_registry, SamPredictor
from segment_anything import SamPredictor
import gc
import sys
sys.path.append("./samurai/sam2")
from sam2.build_sam import build_sam2_video_predictor
import supervision as sv
import torch
from torchvision.ops import box_convert

import threading
adjust_arm_event = threading.Event()
kill_thread = threading.Event()
def bounding_box_predict(image_name, target, visualization=False):
    ## Predict the bounding box in the current image on the target object
    # Specify the paths to the model
    home_addr = os.path.join(os.getcwd(), "GroundingDINO")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    CONFIG_PATH = home_addr + "/groundingdino/config/GroundingDINO_SwinT_OGC.py"
    WEIGHTS_PATH = home_addr + "/weights/groundingdino_swint_ogc.pth"
    model = load_model(CONFIG_PATH, WEIGHTS_PATH, DEVICE)

    IMAGE_PATH = image_name
    TEXT_PROMPT = target
    BOX_TRESHOLD = 0.3
    TEXT_TRESHOLD = 0.3

    # Load the image & Do the bounding box prediction
    image_source, image = load_image(IMAGE_PATH)
    boxes, logits, phrases = predict(
        model=model, 
        image=image, 
        caption=TEXT_PROMPT, 
        box_threshold=BOX_TRESHOLD, 
        text_threshold=TEXT_TRESHOLD
    )

    # Display the annotated frame
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    if visualization:
        sv.plot_image(annotated_frame, (16, 16))

    # Return the bounding box coordinates
    h, w, _ = image_source.shape
    boxes = boxes * torch.Tensor([w, h, w, h])
    xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy").numpy()
    if boxes.shape[0] == 0: # If there is no prediction
        print("Failed to Detect the target object in current image")
        print("Exiting...")
        return None
    elif boxes.shape[0] != 1: # If there are multiple target objects
        xyxy = xyxy[int(torch.argmax(logits))] 
        # Select the one with the highest confidence  
    else:
        xyxy = xyxy[0]
    return xyxy, logits[0]

'''
Predict bounding box in the target image using Object Tracking module Samurai
'''
def determine_model_cfg(model_path):
    '''
    Function to determine the model configuration based on the model path
    '''
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        raise ValueError("Unknown model size in path!")
def bounding_box_predict_samurai(image_pose_est_path, initial_xyxy, \
            model_path = "./samurai/sam2/checkpoints/sam2.1_hiera_base_plus.pt",\
            visualization = True):
    '''
    Formal function for bounding box prediction
    '''
    model_cfg = determine_model_cfg(model_path)
    predictor = build_sam2_video_predictor(model_cfg, model_path, device="cuda:0")
    frames_or_path = image_pose_est_path
    prompts = {}
    prompts[0] = ((int(initial_xyxy[0]), int(initial_xyxy[1]), int(initial_xyxy[2]), int(initial_xyxy[3])), 0)

    if visualization:
        frames = sorted([os.path.join(image_pose_est_path, f) for f in os.listdir(image_pose_est_path) if f.endswith((".jpg", ".jpeg", ".JPG", ".JPEG"))])
        loaded_frames = [cv2.imread(frame_path) for frame_path in frames]
        height, width = loaded_frames[0].shape[:2]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts[0]
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                if len(non_zero_indices) == 0:
                    bbox = [0, 0, 0, 0]
                else:
                    y_min, x_min = non_zero_indices.min(axis=0).tolist()
                    y_max, x_max = non_zero_indices.max(axis=0).tolist()
                    bbox = [x_min, y_min, x_max, y_max]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if visualization:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    color = [(255, 0, 0)]
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)

                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color[obj_id % len(color)], 2)
                
                cv2.imwrite(os.path.join(os.path.dirname(image_pose_est_path),  "frame" + str(frame_idx) + ".jpg"), img)

    del predictor, state
    gc.collect()
    torch.clear_autocast_cache()
    torch.cuda.empty_cache()

    # Theoretically, there should be only one target object in the only one new image
    return bbox_to_vis[0], mask_to_vis[0]

def collect_images(total_images, robot, options, obj_pose_hand_first, target_obj, \
                   initial_image = "images/obj_pose/0.jpg",  initial_xyxy = None):
    N = total_images
    counter = 0
    while counter <= N and not kill_thread.is_set():
        adjust_arm_event.wait()
        counter += 1
        ## Move the robot around the target object
        # Test a small angle at first
        angle = -np.pi/12 # (2*np.pi)/N
        robot.move_base_arc(obj_pose_hand_first, angle)

        time.sleep(1)
        ## Finetune the robot pose
        # Capture another image
        image_responses, images, image_extensions = robot.capture_images(options)
        # Store the image
        image_pose_est_name = os.path.join(os.path.dirname(initial_image), "1" + image_extensions[0])
        cv2.imwrite(image_pose_est_name, images[0])

        # Find the bounding box of the target in the first image
        # TODO: replace it with Samurai
        xyxy, _ = bounding_box_predict_samurai(os.path.join(os.path.dirname(initial_image)),\
                                 initial_xyxy, visualization=True)
        if xyxy is None:
            # If Samurai failed, still use GroundingDINO
            xyxy, _ = bounding_box_predict(image_pose_est_name, target_obj, False)
        # Estimate the object pose in hand frame
        distance = robot.estimate_obj_distance(image_responses[1], xyxy)
        obj_pose_hand = robot.estimate_obj_pose_hand(xyxy, image_responses[0], distance)

        # Finetune the spot's pose
        robot.correct_body(obj_pose_hand)

        ## Formally store the image
        # Capture another image at the corrected pose
        image_responses, images, image_extensions = robot.capture_images(options)

        # Estimate the object pose in hand frame
        distance = robot.estimate_obj_distance(image_responses[1], xyxy)
        obj_pose_hand = robot.estimate_obj_pose_hand(xyxy, image_responses[0], distance)

        # Store the image
        image_name = "images/image" + str(counter) + image_extensions[0]
        cv2.imwrite(image_name, images[0])
            
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
    if options.image_sources[0] != "hand_color_image" or \
        options.image_sources[1] != "hand_depth_in_hand_color_frame":
        print("Please specify the image sources as hand_color_image and hand_depth_in_hand_color_frame")
        return False
    # Create robot object with an image client.
    robot = SPOT(options)
    target_obj = input("what do you want to grasp?")
    with robot.lease_alive():
        
        robot.power_on_stand()
        # print("==Test wasd control")
        # robot.arm_control_wasd()
        # Capture one image as the start
        image_responses, images, image_extensions = robot.capture_images(options)
        if len(image_extensions) != 2:
            print('Expected one image source, but got {}'.format(len(image_extensions)))
            return False
        # Store the image
        if os.path.exists("images"):
            os.system("rm -r images")
            os.mkdir("images")
            os.mkdir("images/obj_pose")
        else:
            os.mkdir("images")
            os.mkdir("images/obj_pose")
        
        image_name = "images/image0.jpg"
        image_pose_initial_name = "images/obj_pose/0.jpg"
        cv2.imwrite(image_name, images[0])
        cv2.imwrite(image_pose_initial_name, images[0])
        # Find the bounding box of the target in the first image
        xyxy, _ = bounding_box_predict(image_pose_initial_name, target_obj, visualization=False)
        if xyxy is None:
            return False

        # Estimate the object pose in hand frame
        distance = robot.estimate_obj_distance(image_responses[1], xyxy)
        obj_pose_hand = robot.estimate_obj_pose_hand(xyxy, image_responses[0], distance)
        print("===Estimated Distance from Depth sensor===")
        print(distance)
        # Rotate the robot around the object
        N = 5

        t1 = threading.Thread(target = collect_images, \
                args=(N, robot, options, obj_pose_hand, target_obj, image_pose_initial_name, xyxy))
        adjust_arm_event.set()
        kill_thread.clear()
        t1.start()
        while True:
            operation = input("Do you want to pause the process & adjust the arm? [yes|no|q]")
            if operation == "yes":
                adjust_arm_event.clear()
                robot.arm_control_wasd()
                adjust_arm_event.set()
            elif operation == "q":
                break
        kill_thread.set()
        t1.join()
        


        





    return True


if __name__ == '__main__':
    if not main(sys.argv[1:]):
        sys.exit(1)