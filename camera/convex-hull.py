import pyrealsense2 as rs
from scipy.spatial import ConvexHull
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Depth visualization
colorizer = rs.colorizer()

pc = rs.pointcloud()
depth_intrinsics = profile.get_stream(rs.stream.depth).as_video_stream_profile().intrinsics

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        points = pc.calculate(depth_frame)
        vertices = np.asanyarray(points.get_vertices())
        points = np.stack([vertices['f0'], vertices['f1'], vertices['f2']], axis=-1)

        xy_points = points[:, :2]

        # Compute convex hull
        hull = ConvexHull(xy_points)
        hull_points = xy_points[hull.vertices]

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        # Draw hull on image
        for i in range(len(hull_points)):
            p1_3d = np.append(hull_points[i], points[hull.vertices[i], 2])  # get Z
            p2_3d = np.append(hull_points[(i+1)%len(hull_points)], points[hull.vertices[(i+1)%len(hull_points)], 2])

            u1, v1 = rs.rs2_project_point_to_pixel(depth_intrinsics, p1_3d)
            u2, v2 = rs.rs2_project_point_to_pixel(depth_intrinsics, p2_3d)

            cv2.line(depth_colormap, (int(u1), int(v1)), (int(u2), int(v2)), (0, 0, 255), 2)

        cv2.imshow('Depth Hull', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()