import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

align = rs.align(rs.stream.color)

pc = rs.pointcloud()
points = rs.points()

print("Press 's' to save a point cloud")

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        color = np.asanyarray(color_frame.get_data())

        # Show RGB image
        cv2.imshow("Color", color)

        key = cv2.waitKey(1)

        if key == ord('s'):
            print("Saving point cloud...")

            # Map texture to color frame
            pc.map_to(color_frame)
            points = pc.calculate(depth_frame)

            filename = "pointcloud.ply"
            points.export_to_ply(filename, color_frame)

            print("Saved:", filename)

        if key == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()