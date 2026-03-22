import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

pipeline = rs.pipeline()
config = rs.config()

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

profile = pipeline.start(config)

# Depth visualization
colorizer = rs.colorizer()

def save_pointcloud(depth_frame, box, floor_depth):
    # Build initial point cloud from depth frame
    pc = rs.pointcloud()
    points = pc.calculate(depth_frame)

    # Save raw scene
    filename_raw = "before.ply"
    points.export_to_ply(filename_raw, depth_frame)
    print(f"Saved raw point cloud to {filename_raw}")

    # Extrude top edges down to the floor
    intrinsics = depth_frame.profile.as_video_stream_profile().intrinsics
    h, w = depth_frame.get_height(), depth_frame.get_width()

    # Calculate box depth
    box_center = np.mean(box, axis=0).astype(int)
    cx, cy = int(np.clip(box_center[0], 0, w - 1)), int(np.clip(box_center[1], 0, h - 1))
    box_depth = depth_frame.get_distance(cx, cy)

    # Convert each of the 4 box corners from screen space to 3D
    top_corners_3d = []

    for px, py in box:
        px = int(np.clip(px, 0, w - 1))
        py = int(np.clip(py, 0, h - 1))

        x, y, z_m = rs.rs2_deproject_pixel_to_point(intrinsics, [px, py], box_depth)
        top_corners_3d.append([x, y, z_m])

    top_corners_3d = np.array(top_corners_3d)  # (4, 3)

    # Extrude each edge of the box top down to floor_depth
    extruded_points = []
    num_steps = 50
    for i in range(len(top_corners_3d)):
        p1 = top_corners_3d[i]
        p2 = top_corners_3d[(i + 1) % len(top_corners_3d)]
        for t in np.linspace(0, 1, num_steps):
            x = p1[0] + t * (p2[0] - p1[0])
            y = p1[1] + t * (p2[1] - p1[1])
            for z in np.linspace(p1[2], floor_depth, num_steps):
                extruded_points.append([x, y, z])
    extruded_points = np.array(extruded_points)

    # Merge original scene and extruded sides
    verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    verts = verts[~np.all(verts == 0, axis=1)]
    completed_verts = np.vstack([verts, extruded_points])

    # Build a new rs.points-like export using open3d, same format as before
    pcd_completed = o3d.geometry.PointCloud()
    pcd_completed.points = o3d.utility.Vector3dVector(completed_verts)
    o3d.io.write_point_cloud("after.ply", pcd_completed)
    print("Saved completed point cloud to after.ply")


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

        # Get floor depth by sampling just outside the box edges
        offset = 20  # pixels outside the box to sample, adjust if needed
        floor_samples = []
        h_img, w_img = depth_image.shape
        for i in range(len(box)):
            p1 = box[i]
            p2 = box[(i + 1) % len(box)]
            for t in np.linspace(0, 1, 20):
                # Midpoint along each edge
                mx = p1[0] + t * (p2[0] - p1[0])
                my = p1[1] + t * (p2[1] - p1[1])

                # Normal pointing outward (perpendicular to edge)
                ex, ey = p2[0] - p1[0], p2[1] - p1[1]
                length = np.sqrt(ex**2 + ey**2) + 1e-6
                nx, ny = ey / length, -ex / length
                
                # Sample just outside the box
                sx = int(np.clip(mx + nx * offset, 0, w_img - 1))
                sy = int(np.clip(my + ny * offset, 0, h_img - 1))
                d = depth_image[sy, sx]

                if not np.isnan(d) and d > 0:
                    floor_samples.append(d)

        floor_depth = (np.median(floor_samples) / 1000.0) if floor_samples else np.nanmax(depth_image) / 1000.0

        # Draw rectangle on image
        depth_colormap = np.asanyarray(
            colorizer.colorize(depth_frame).get_data()
        )

        cv2.drawContours(depth_colormap, [box], 0, (0,255,0), 2)

        for p in box:
            cv2.circle(depth_colormap, tuple(p), 5, (0,0,255), -1)

        cv2.putText(depth_colormap, "Press S to save", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Simple box detection", depth_colormap)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            save_pointcloud(depth_frame, box, floor_depth)
        if key == ord('q'):
            break

finally:
    pipeline.stop()