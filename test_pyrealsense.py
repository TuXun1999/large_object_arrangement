import pyrealsense2 as rs

pipeline = rs.pipeline()
pipeline.start()
frames = pipeline.wait_for_frames()
depth_frame = frames.get_depth_frame()
print("Got depth frame:", depth_frame)