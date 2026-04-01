import numpy as np
import cv2
import socket
import struct
import pickle

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

        # Generate Colormap
        # Scale the 16-bit depth values to 8-bit (adjust alpha to change contrast)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(raw_depth, alpha=0.03), cv2.COLORMAP_JET)

        # Remove invalid values
        depth_image = np.where(raw_depth == 0, np.nan, raw_depth)

        # Simple threshold
        min_depth = np.nanmin(depth_image)
        mask = depth_image < (min_depth + 100)  # tolerance in mm
        mask = mask.astype(np.uint8) * 255 

        # Clean mask
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.medianBlur(mask, 5)

        # Detect shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            # Select largest contour
            cnt = max(contours, key=cv2.contourArea)

            # Get smallest rectangle
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            
            # Draw on the OpenCV-generated colormap
            cv2.drawContours(depth_colormap, [box], 0, (0,255,0), 2)
            for p in box:
                cv2.circle(depth_colormap, tuple(p), 5, (0,0,255), -1)


        # Show both feeds
        cv2.imshow("Box Detection (Depth)", depth_colormap)
        cv2.imshow("Raw RGB Feed", color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")
    
finally:
    client_socket.close()
    cv2.destroyAllWindows()