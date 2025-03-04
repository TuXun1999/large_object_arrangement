import open3d as o3d 
import numpy as np
import json

'''
Visualize the collected gripper poses to see if they are valid
'''
f = open("./gripper_poses_move_chair.json", "r")
gripper_poses_trials = json.load(f)

gripper_poses_all = gripper_poses_trials
gripper_poses_sample = np.load(open("./gripper_pose_sample.npy", 'rb'))
# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()

color_scale = 1/(len(gripper_poses_all) - 1)
for i in range(len(gripper_poses_all)):
    gripper_poses = gripper_poses_all[i]
    for item in gripper_poses:
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(0.01, [0, 0, 0])
        gripper_pose = item
        camera_frame.transform(gripper_pose)
        #camera_frame.paint_uniform_color((1 - i * color_scale, 0, i * color_scale))
        vis.add_geometry(camera_frame)

for pose in gripper_poses_sample:
    camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
    camera_frame.scale(0.01, [0, 0, 0])
    gripper_pose = pose
    camera_frame.transform(gripper_pose)
    # camera_frame.paint_uniform_color((0, 1, 0))
    vis.add_geometry(camera_frame)

fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
fundamental_frame.scale(0.1, [0, 0, 0])
vis.add_geometry(fundamental_frame)
print("===Test==")
print(gripper_poses_sample[0])
print(gripper_poses_sample[30])
print(np.linalg.inv(gripper_poses_sample[0])@gripper_poses_sample[30])
vis.run()
# Close all windows
vis.destroy_window()