import numpy as np
import cv2
import socket
import struct
import pickle
from collections import deque

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
MIN_EXTENT = 0.8       # How rectangular the object must be (1.0 = perfect rectangle)
WINDOW_SIZE = 5        # Number of frames to average

# Initialize the frame buffer
frame_buffer = deque(maxlen=WINDOW_SIZE)

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
        current_raw_depth = frame_dict['depth']
        color_image = frame_dict['color']

        # --- SLIDING WINDOW AVERAGING ---
        
        # Add current frame to buffer
        frame_buffer.append(current_raw_depth.astype(np.float32))

        # Compute average depth over the buffer
        stack = np.array(frame_buffer)

        # Create a mask of valid pixels (where depth > 0)
        valid_mask = (stack > 0)

        # Sum only the valid pixels
        sum_valid = np.sum(stack, axis=0)
        
        # Count how many frames had valid data for each pixel
        count_valid = np.sum(valid_mask, axis=0)
        
        # Avoid division by zero: if a pixel was 0 in ALL frames, keep it 0.
        # Otherwise, divide the sum by the count of valid frames.
        raw_depth = np.divide(sum_valid, count_valid, 
                              out=np.zeros_like(sum_valid), 
                              where=count_valid > 0).astype(np.uint16)

        # --- IMAGE PRE-PROCESSING ---

        hole_mask = (raw_depth == 0).astype(np.uint8)

        # Convert to 8-bit for processing
        depth_8bit = cv2.convertScaleAbs(raw_depth, alpha=0.07)

        # Fill holes using inpainting
        depth_8bit = cv2.inpaint(depth_8bit, hole_mask, 5, cv2.INPAINT_TELEA)

        # Generate Colormap for visualization
        depth_colormap = cv2.applyColorMap(depth_8bit, cv2.COLORMAP_JET)

        # --- IMAGE PROCESSING ---

        # 1. Apply a median blur to remove raw depth sensor noise/speckles
        blurred_depth = cv2.medianBlur(depth_8bit, 7)

        # 2. Find depth discontinuities (edges)
        # https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
        edges = cv2.Canny(blurred_depth, 20, 35)

        # 3. Clean edges and denoise (erosion, then dilation)
        kernel = np.ones((11, 11), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

        
        # 4. Detect shapes
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Number of boxes
        n = 0

        for cnt in contours:
            # Bridge gaps using a Convex Hull, this "wraps" the points even if the line is broken
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)
            
            # Ignore small noise
            if area < MIN_AREA:
                continue

            # --- Depth consistency filter ---

            # Create mask for this contour
            mask_local = np.zeros(raw_depth.shape, dtype=np.uint8)
            cv2.drawContours(mask_local, [hull], -1, 255, -1)

            # Extract depth values inside contour
            depth_values = raw_depth[mask_local == 255]
            depth_values = depth_values[depth_values > 0]

            if len(depth_values) < 50:
                continue

            # Compute depth spread
            z_min = np.percentile(depth_values, 10)
            z_max = np.percentile(depth_values, 90)

            depth_range = z_max - z_min

            # Top surface = flat and small depth variation
            if depth_range > 80:   # mm
                continue

            # Simplify the shape (Polygonal Approximation)
            # This helps ignore small wiggles or "missing" chunks of the edge
            epsilon = 0.04 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            # Rectangle Check
            # We look for shapes with roughly 4 corners OR a high "Extent"
            rect = cv2.minAreaRect(hull)
            box_width, box_height = rect[1]
            box_area = box_width * box_height

            if box_area == 0: continue
            extent = area / box_area

            # Extent is the ratio of contour area to bounding rectangle area.
            extent = area / box_area

            # If it has 4-6 vertices (approx) AND it's mostly rectangular (extent)
            if extent > MIN_EXTENT and len(approx) <= 6:
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