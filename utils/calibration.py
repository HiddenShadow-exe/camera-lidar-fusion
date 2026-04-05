import numpy as np
import cv2
import open3d as o3d

# --- You fill these in during calibration sessions ---
# Each row is one correspondence: a 3D point from the camera, and the 
# matching 3D point you picked from the LiDAR cloud
camera_points = np.array([
    [0.12, 0.41, 1.78],
    [0.12, 0.05, 1.94],
    [-0.22, 0.68, 2.0]
])

lidar_points = np.array([
    [1.0025, -0.327, 0.192],
    [1.042, -0.0546, -0.0866],
    [0.76, -0.54, -0.00545]
])


def solve_rigid_transform(src, dst):
    """
    Find R, t such that dst ≈ R @ src + t
    src: camera points, dst: lidar points
    """
    assert src.shape == dst.shape and len(src) >= 3

    # Center both point sets
    src_centroid = src.mean(axis=0)
    dst_centroid = dst.mean(axis=0)
    src_c = src - src_centroid
    dst_c = dst - dst_centroid

    # SVD
    H = src_c.T @ dst_c
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case (det should be +1, not -1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = dst_centroid - R @ src_centroid
    return R, t


def reprojection_error(src, dst, R, t):
    projected = (R @ src.T).T + t
    errors = np.linalg.norm(projected - dst, axis=1)
    print(f"Per-point errors (m): {errors.round(4)}")
    print(f"Mean error: {errors.mean():.4f}m  |  Max: {errors.max():.4f}m")
    return errors


R, t = solve_rigid_transform(camera_points, lidar_points)
reprojection_error(camera_points, lidar_points, R, t)

print("R =", R)
print("t =", t)

# Save for use in your main script
np.save("calib/calib_R.npy", R)
np.save("calib/calib_t.npy", t)