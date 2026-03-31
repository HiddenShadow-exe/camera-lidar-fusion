import pyrealsense2 as rs
import numpy as np
import cv2
import time

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Depth visualization
colorizer = rs.colorizer()

last_print_time = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        box_coord = (286, 266)
        chair_coord = (286, 256)

        box_distance = depth_frame.get_distance(box_coord[0], box_coord[1])
        chair_distance = depth_frame.get_distance(chair_coord[0], chair_coord[1])

        current_time = time.time()
        if current_time - last_print_time >= 1:
            print(f"Box distance: {box_distance:.2f} meters")
            print(f"Chair distance: {chair_distance:.2f} meters")
            print(f"Distance difference: {chair_distance - box_distance:.2f} meters")
            last_print_time = current_time

        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        cv2.circle(depth_colormap, box_coord, 4, (0, 0, 255), 2)
        cv2.circle(depth_colormap, chair_coord, 4, (0, 0, 255), 2)
        cv2.imshow('Depth', depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()