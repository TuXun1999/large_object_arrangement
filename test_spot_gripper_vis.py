import open3d as o3d 
import numpy as np
import json

'''
Visualize the collected gripper poses to see if they are valid
'''
f = open("./gripper_poses_move_chair.json", "r")
gripper_poses_trials = json.load(f)

gripper_poses_all = gripper_poses_trials

# Create the window to display everything
vis= o3d.visualization.Visualizer()
vis.create_window()

print(len(gripper_poses_all))
for i in range(len(gripper_poses_all)):
    gripper_poses = gripper_poses_all[i]
    for item in gripper_poses:
        item = np.array(item)
        camera_pose = item[:, 0:4]
        body_pose = item[:, 4:]
        camera_pose = body_pose@camera_pose # Fit the transformation also
        camera_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        camera_frame.scale(0.1, [0, 0, 0])
        body_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
        body_frame.scale(0.1, [0, 0, 0])
        camera_frame.transform(camera_pose)
        body_frame.transform(body_pose)
        camera_frame.paint_uniform_color((1, 0, 0))
        body_frame.paint_uniform_color((0, 1, 0))
        vis.add_geometry(camera_frame)
        vis.add_geometry(body_frame)


# fundamental_frame = o3d.geometry.TriangleMesh.create_coordinate_frame()
# fundamental_frame.scale(0.1, [0, 0, 0])
# vis.add_geometry(fundamental_frame)
vis.run()
# Close all windows
vis.destroy_window()