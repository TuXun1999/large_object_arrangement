import open3d as o3d


from GroundingDINO.groundingdino.util.inference import \
    load_model, load_image, predict, annotate





print("???")
vis= o3d.visualization.Visualizer()
print("???")
vis.create_window()
f =  o3d.geometry.TriangleMesh.create_coordinate_frame()
vis.add_geometry(f)
vis.run()
vis.destroy_window()