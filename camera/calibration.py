import numpy as np
import cv2
import socket
import struct
import pickle
import json

# Network Setup
RPI_IP = '192.168.0.10'
PORT = 8485

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to Depth Camera at {RPI_IP}:{PORT}...")
client_socket.connect((RPI_IP, PORT))
print("Connected! Receiving frames...")

data = b""
payload_size = struct.calcsize("Q")

CHESSBOARD_SIZE = (8, 6)  # Inner corners (columns, rows)
SQUARE_SIZE = 0.0325      # Size of a square in meters

# Prepare object points (0,0,0), (1,0,0), ... (scaled by square size)
objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE


# Arrays to store object points and image points
objpoints = []  # 3D points in real world
imgpoints = []  # 2D points in image plane

collected_frames = 0
MAX_FRAMES = 100  # Number of successful chessboard detections before calibration

try:
    while True:
        # Retrieve message size
        while len(data) < payload_size:
            packet = client_socket.recv(65536)
            if not packet: break
            data += packet
            
        if len(data) < payload_size: break
            
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack("Q", packed_msg_size)[0]

        # Retrieve frame data
        while len(data) < msg_size:
            packet = client_socket.recv(65536)
            if not packet: break
            data += packet
            
        if len(data) < msg_size: break
            
        frame_data = data[:msg_size]
        data = data[msg_size:]

        # Deserialize the dictionary
        frame_dict = pickle.loads(frame_data)
        color_image = frame_dict['color']

        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)
        if ret:
            objpoints.append(objp)
            # Refine corner locations
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
            imgpoints.append(corners2)
            collected_frames += 1
            cv2.drawChessboardCorners(color_image, CHESSBOARD_SIZE, corners2, ret)
            print(f"Collected frame {collected_frames}/{MAX_FRAMES}")

        # Display
        cv2.imshow("Calibration Feed", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Once enough frames collected
        if collected_frames >= MAX_FRAMES:
            print("Performing calibration...")
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
            print("Calibration complete!")

            # Save intrinsics
            intrinsics = {
                "fx": float(mtx[0, 0]),
                "fy": float(mtx[1, 1]),
                "ppx": float(mtx[0, 2]),
                "ppy": float(mtx[1, 2]),
                "distortion": dist.flatten().tolist()
            }
            with open("camera_intrinsics.json", "w") as f:
                json.dump(intrinsics, f, indent=4)
            print("Camera intrinsics saved to camera_intrinsics.json")
            break


        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()