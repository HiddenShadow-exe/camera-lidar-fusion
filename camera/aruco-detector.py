import numpy as np
import cv2
import socket
import struct
import pickle
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.camera_intrinsics import load_camera_intrinsics

# Network Setup
RPI_IP = '192.168.0.10'
PORT = 8485

client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
print(f"Connecting to Depth Camera at {RPI_IP}:{PORT}...")
client_socket.connect((RPI_IP, PORT))
print("Connected! Receiving frames...")

data = b""
payload_size = struct.calcsize("Q")

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
        raw_depth = frame_dict['depth']
        color_image = frame_dict['color']

        # --- IMAGE PRE-PROCESSING ---

        hole_mask = (raw_depth == 0).astype(np.uint8)

        # Convert to 8-bit for processing
        depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)

        # Fill holes using inpainting
        depth_8bit = cv2.inpaint(depth_8bit, hole_mask, 5, cv2.INPAINT_TELEA)

        # Generate Colormap for visualization
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # Apply a median blur to remove raw depth sensor noise/speckles
        blurred_depth = cv2.medianBlur(depth_8bit, 7)

        # --- ARUCO DETECTION ---
        
        # Define the dictionary and parameters
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        parameters = cv2.aruco.DetectorParameters()
        
        # Initialize detector
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # Detect markers in the color image
        corners, ids, rejected = detector.detectMarkers(color_image)
        
        # If markers are found, draw them on the color_image
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(color_image, corners, ids)
            
            # Print the ID and center coordinate of the first detected marker
            for i in range(len(ids)):
                c = corners[i][0]
                center_x = int(c[:, 0].mean())
                center_y = int(c[:, 1].mean())
                cv2.putText(color_image, f"ID: {ids[i][0]}", (center_x, center_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                print(f"Detected ArUco Marker ID: {ids[i][0]} at pixel coordinates: ({center_x}, {center_y})")

                # Convert pixel coordinates to 3D point using depth and camera intrinsics
                depth_value = raw_depth[center_y, center_x]
                if depth_value > 0:
                    # Camera intrinsics
                    intristics = load_camera_intrinsics()

                    # Convert to meters
                    z = depth_value / 1000.0
                    x = (center_x - intristics['ppx']) * z / intristics['fx']
                    y = (center_y - intristics['ppy']) * z / intristics['fy']
                    
                    print(f"Marker ID {ids[i][0]} 3D coordinates: ({x:.2f}m, {y:.2f}m, {z:.2f}m)")


        # Show both feeds
        cv2.imshow("Box Detection (Depth)", depth_colormap)
        cv2.imshow("Raw RGB Feed", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()