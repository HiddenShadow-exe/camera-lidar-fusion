import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

# We need to align the depth frame to color frame since the sensors are slightly offset
align = rs.align(rs.stream.color)

# Depth visualization
colorizer = rs.colorizer()

# Filters for depth frames (smooth within frame, smooth across frames, and hole filling)
spatial = rs.spatial_filter()
temporal = rs.temporal_filter()
# hole_filling = rs.hole_filling_filter()

# Filter options
spatial.set_option(rs.option.filter_magnitude, 2)
spatial.set_option(rs.option.filter_smooth_alpha, 0.5)
spatial.set_option(rs.option.filter_smooth_delta, 20)
temporal.set_option(rs.option.filter_smooth_alpha, 0.2)

# Disparity transforms for better smoothing and hole filling results
depth_to_disparity = rs.disparity_transform(True)
disparity_to_depth = rs.disparity_transform(False)

try:
    while True:
        frames = pipeline.wait_for_frames()
        frames = align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        # Convert depth to disparity for better results with filters
        depth_frame = depth_to_disparity.process(depth_frame)

        # Apply filters to depth frame
        depth_frame = spatial.process(depth_frame)
        depth_frame = temporal.process(depth_frame)

        # Convert back to depth and apply hole filling
        depth_frame = disparity_to_depth.process(depth_frame)
        # depth_frame = hole_filling.process(depth_frame)

        if not depth_frame or not color_frame:
            continue

        # Convert frames to numpy arrays for visualization
        color = np.asanyarray(color_frame.get_data())
        depth = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        # Create an overlay of color and depth images
        overlay = cv2.addWeighted(color, 0.6, depth, 0.4, 0)

        cv2.imshow("Overlay", overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()