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
MIN_AREA = 2000        # Minimum contour area (in pixels) to ignore noise
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

        # 1. Work with RAW depth (mm) for the height mask
        # Apply a light blur to raw depth to handle sensor jitter
        raw_depth_float = raw_depth.astype(np.float32)
        raw_depth_blur = cv2.medianBlur(raw_depth_float, 5)

        # 2. Find the floor in Millimeters
        valid_depths_mm = raw_depth_blur[raw_depth_blur > 500].flatten() # Ignore camera noise < 50cm
        
        if len(valid_depths_mm) > 0:
            hist, bins = np.histogram(valid_depths_mm, bins=100)
            floor_mm = bins[np.argmax(hist)]
            
            # 1. Height Mask
            height_mask = ((raw_depth_blur > 100) & (raw_depth_blur < (floor_mm - 150))).astype(np.uint8) * 255

            # Flatness Mask
            dzdx = cv2.Sobel(raw_depth_blur, cv2.CV_32F, 1, 0, ksize=1)
            dzdy = cv2.Sobel(raw_depth_blur, cv2.CV_32F, 0, 1, ksize=1)
            mag = np.sqrt(dzdx**2 + dzdy**2)
            
            # Binary mask: Black where flat, White where steep (the walls)
            # Raise this number if the box tops are too "holey"
            flat_mask = (mag > 30.0).astype(np.uint8) * 255

            # Erode then dilate to fill small holes in the flat_mask (the box tops)
            flat_mask = cv2.morphologyEx(flat_mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

            # Then erode to get rid of edge noise
            flat_mask = cv2.erode(flat_mask, np.ones((11, 11), np.uint8), iterations=1)

            # 4. COMBINE: Subtract non_flat_mask points from height_mask to get our final box mask
            final_box_mask = cv2.subtract(height_mask, flat_mask)

            # 5. FINAL POLISH
            # A quick 'Open' to remove any tiny stray pixels
            final_box_mask = cv2.morphologyEx(final_box_mask, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

        # Find contours on the cleaned mask
        contours, _ = cv2.findContours(final_box_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        n = 0

        for cnt in contours:
            # Bridge gaps using a Convex Hull, this "wraps" the points even if the line is broken
            hull = cv2.convexHull(cnt)
            area = cv2.contourArea(hull)

            if area < MIN_AREA:
                continue

            # Fit a minimum area rectangle to the contour and compute its extent
            rect = cv2.minAreaRect(hull)
            box_w, box_h = rect[1]
            if box_w * box_h == 0:
                continue

            extent = area / (box_w * box_h)

            epsilon = 0.04 * cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, epsilon, True)

            if extent > MIN_EXTENT and len(approx) <= 6:
                box = cv2.boxPoints(rect)
                box = np.int32(box)

                # If any of the corners are near the image edge, it's likely a false detection
                if np.any(box < 10) or np.any(box[:, 0] > (raw_depth.shape[1] - 10)) or np.any(box[:, 1] > (raw_depth.shape[0] - 10)):
                    continue

                n += 1
                cv2.drawContours(depth_colormap, [box], 0, (0, 255, 0), 2)
                cv2.drawContours(color_image,    [box], 0, (0, 255, 0), 2)
                for p in box:
                    cv2.circle(depth_colormap, tuple(p), 5, (0, 0, 255), -1)
                    cv2.circle(color_image,    tuple(p), 5, (0, 0, 255), -1)


        # Show both feeds
        cv2.putText(depth_colormap, f"Number of boxes: {n}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        cv2.imshow("Box Detection (Depth)", depth_colormap)
        cv2.imshow("Raw RGB Feed", color_image)
        cv2.imshow("Combined Mask", final_box_mask)
        cv2.imshow("1. Height Mask Only", height_mask)
        cv2.imshow("2. Flatness Mask Only", flat_mask)  

        if cv2.waitKey(1) & 0xFF == ord('q'): break

except Exception as e:
    print(f"Network stream ended or error: {e}")
    raise e

finally:
    client_socket.close()
    cv2.destroyAllWindows()