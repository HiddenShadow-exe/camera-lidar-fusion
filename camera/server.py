import pyrealsense2 as rs
import numpy as np
import socket
import struct
import pickle

# Network Setup
HOST = '0.0.0.0'
PORT = 8485

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST, PORT))
server_socket.listen(1)

# RealSense Setup
pipeline = rs.pipeline()
config = rs.config()

# Enable both Depth and RGB Color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

print(f"Server is running and listening on port {PORT}...")
print("Press Ctrl+C to stop the server.")

try:
    while True:
        print("\nWaiting for a new connection...")
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")
        
        try:
            while True:
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()

                if not depth_frame or not color_frame:
                    continue

                # Convert to raw numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                # Package the raw arrays
                frame_dict = {
                    'depth': depth_image,
                    'color': color_image
                }

                # Serialize and send
                data_bytes = pickle.dumps(frame_dict)
                message_size = struct.pack("Q", len(data_bytes))
                conn.sendall(message_size + data_bytes)

        except (ConnectionResetError, BrokenPipeError, socket.error):
            print(f"Client {addr} disconnected.")

        except Exception as e:
            print(f"An error occurred during streaming: {e}")

        finally:
            pipeline.stop()
            conn.close()
            server_socket.close()

except KeyboardInterrupt:
    print("\nServer stopped by user...")

finally:
    pipeline.stop()
    server_socket.close()
    print("Resources released.")