import pyrealsense2 as rs
import numpy as np
import cv2

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Depth visualization
colorizer = rs.colorizer()

try:
    while True:
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()

        if not depth_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        
        # Remove invalid values
        depth_image = np.where(depth_image == 0, np.nan, depth_image)

        # Simple threshold
        # We assume the box object is the closest thing to the camera
        min_depth = np.nanmin(depth_image)
        mask = depth_image < (min_depth + 100)  # tolerance in mm
        mask = mask.astype(np.uint8) * 255 # black & white bool mask

        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)

        # Detect shapes in the binary mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Skip frame if no object is detected
        if len(contours) == 0:
            continue

        # Select largest contour
        cnt = max(contours, key=cv2.contourArea)

        # Get the smallest rectangle
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        
        # Draw rectangle on image
        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        cv2.drawContours(depth_colormap, [box], 0, (0,255,0), 2)

        for p in box:
            cv2.circle(depth_colormap, tuple(p), 5, (0,0,255), -1)

        cv2.imshow("Simple box detection", depth_colormap)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    pipeline.stop()