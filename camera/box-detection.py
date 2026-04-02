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

# Detection parameters
MIN_AREA = 1000        # Minimum contour area (in pixels) to ignore noise
MIN_EXTENT = 0.75      # How rectangular the object must be (1.0 = perfect rectangle)

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
        depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # IMAGE PROCESSING

        # 1. Apply a median blur to remove raw depth sensor noise/speckles
        blurred_depth = cv2.medianBlur(depth_8bit, 15)

        # 2. Find depth discontinuities (edges)
        # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
        edges = cv2.Canny(blurred_depth, 10, 25)

        # 3. Clean edges and denoise (erosion, then dilation)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        mask = cv2.dilate(mask, kernel, iterations=1)

        # 4. Detect shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Number of boxes
        n = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            
            # Filter 1: Ignore small noise
            if area < MIN_AREA:
                continue

            # Get smallest bounding rectangle
            rect = cv2.minAreaRect(cnt)
            box_width, box_height = rect[1]
            box_area = box_width * box_height

            if box_area == 0:
                continue

            # Filter 2: Rectangularity (Extent)
            # Extent is the ratio of contour area to bounding rectangle area.
            extent = area / box_area

            # If the shape fills out at least MIN_EXTENT of its bounding box, treat it as a rectangle
            if extent > MIN_EXTENT:
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                n += 1
                
                # Draw the valid box on the colormap
                cv2.drawContours(depth_colormap, [box], 0, (0, 255, 0), 2)
                for p in box:
                    cv2.circle(depth_colormap, tuple(p), 5, (0, 0, 255), -1)


        # Show both feeds
        cv2.putText(depth_colormap, f"Number of boxes: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Box Detection (Depth)", depth_colormap)
        cv2.imshow("Raw RGB Feed", color_image)
        cv2.imshow("Edge Mask", mask)

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")

finally:
    client_socket.close()
    cv2.destroyAllWindows()