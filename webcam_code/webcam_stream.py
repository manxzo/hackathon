#!/usr/bin/env python3
"""
Webcam Streaming Server
Captures video from USB webcam and streams it over the local network
"""

from flask import Flask, Response
import cv2
import socket
import argparse
import threading
import time
import numpy as np
import sys
import os

# Add the my_hack_test directory to the path to import arena_transform
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'my_hack_test'))
from arena_transform import detect_red_corners

app = Flask(__name__)

# Global variables for frame sharing
camera = None
camera_index = 0  # Default camera index
output_frame = None
lock = threading.Lock()
camera_thread = None

# Arena transformation globals
transform_matrix = None
transform_enabled = False
output_width = 480
output_height = 640

class CameraStream:
    """Background thread to continuously capture frames"""
    def __init__(self, camera_index, width=640, height=480, enable_transform=False):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        self.stopped = False
        self.enable_transform = enable_transform

    def start(self):
        """Start the camera capture thread"""
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def detect_and_setup_transformation(self, frame):
        """Detect corners and setup transformation matrix"""
        global transform_matrix, transform_enabled

        print("\n🔍 Detecting arena corners for transformation...")
        corner_points, red_mask = detect_red_corners(frame)

        if len(corner_points) != 4:
            print(f"⚠️ Warning: Found {len(corner_points)} corners instead of 4")
            print("   Transformation disabled. Streaming original frames.")
            return False

        print(f"✅ Found 4 corner markers!")
        print("Detected corner coordinates:")
        labels = ["Top-left", "Top-right", "Bottom-right", "Bottom-left"]
        for i, (x, y) in enumerate(corner_points):
            print(f"  {labels[i]}: ({x:.0f}, {y:.0f})")

        # Convert to float32
        src_points = np.float32(corner_points)

        # Define destination points (perfect rectangle)
        dst_points = np.float32([
            [0, 0],                        # Top-left
            [output_width, 0],             # Top-right
            [output_width, output_height], # Bottom-right
            [0, output_height]             # Bottom-left
        ])

        # Calculate perspective transform
        transform_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transform_enabled = True

        print(f"✅ Transformation matrix calculated!")
        print(f"   Output resolution: {output_width}x{output_height}")
        print("   Arena transformation is now active!\n")

        return True

    def update(self):
        """Continuously capture frames from camera"""
        global output_frame, lock, transform_matrix, transform_enabled

        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        # Wait for first frame and setup transformation if enabled
        if self.enable_transform:
            print("Waiting for first frame to setup transformation...")
            for _ in range(30):  # Try up to 30 frames
                success, frame = self.camera.read()
                if success:
                    # Rotate frame 90 degrees clockwise
                    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

                    if self.detect_and_setup_transformation(frame):
                        break
                    else:
                        print("⚠️ Retrying corner detection...")
                        time.sleep(0.5)
                time.sleep(0.1)

        while not self.stopped:
            success, frame = self.camera.read()
            if not success:
                continue

            # Rotate frame 90 degrees clockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Apply transformation if enabled
            if transform_enabled and transform_matrix is not None:
                frame = cv2.warpPerspective(frame, transform_matrix, (output_width, output_height))

            # Acquire lock and update the output frame
            with lock:
                output_frame = frame.copy()

            # Small delay to prevent excessive CPU usage
            time.sleep(0.01)

        self.camera.release()

    def stop(self):
        """Stop the camera capture thread"""
        self.stopped = True

def generate_frames():
    """Generate frames for clients from the shared buffer"""
    global output_frame, lock

    while True:
        # Wait for a frame to be available
        with lock:
            if output_frame is None:
                continue
            frame = output_frame.copy()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Yield frame in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

        # Small delay to prevent excessive bandwidth usage
        time.sleep(0.033)  # ~30 FPS

@app.route('/')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stream')
def stream():
    """Alternative video streaming route"""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def get_local_ip():
    """Get the local IP address"""
    try:
        # Create a socket to determine the local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return socket.gethostbyname(socket.gethostname())

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Webcam Streaming Server')
    parser.add_argument('-c', '--camera', type=int, default=0,
                        help='Camera index (default: 0). Use check_cameras.py to find available cameras.')
    parser.add_argument('-p', '--port', type=int, default=8000,
                        help='Port number (default: 5000)')
    parser.add_argument('-r', '--resolution', type=str, default='640x480',
                        choices=['320x240', '640x480', '800x600', '1280x720', '1920x1080'],
                        help='Video resolution (default: 640x480). Options: 320x240, 640x480, 800x600, 1280x720, 1920x1080')
    parser.add_argument('-t', '--transform', action='store_true',
                        help='Enable arena transformation (detects red corners and applies perspective correction)')
    args = parser.parse_args()

    camera_index = args.camera
    port = args.port
    enable_transform = args.transform

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    # Start the camera capture thread
    camera_thread = CameraStream(camera_index, width, height, enable_transform=enable_transform).start()

    # Give the camera time to initialize (longer if transformation is enabled)
    if enable_transform:
        time.sleep(4)  # Extra time for corner detection
    else:
        time.sleep(2)

    local_ip = get_local_ip()
    print("=" * 60)
    print("🎥 Webcam Streaming Server Starting...")
    print("=" * 60)
    print(f"\n📹 Using Camera Index: {camera_index}")
    print(f"\n📐 Input Resolution: {width}x{height}")
    if enable_transform and transform_enabled:
        print(f"\n🔄 Arena Transformation: ENABLED")
        print(f"   Output Resolution: {output_width}x{output_height}")
    elif enable_transform:
        print(f"\n⚠️  Arena Transformation: FAILED (streaming original frames)")
    else:
        print(f"\n🔄 Arena Transformation: DISABLED")
    print(f"\n📍 Local IP Address: {local_ip}")
    print(f"\n🌐 Stream URLs:")
    print(f"   http://{local_ip}:{port}/")
    print(f"   http://{local_ip}:{port}/stream")
    print(f"\n💡 Use either URL in your Python client to receive frames")
    print(f"\n💡 Multiple clients can now connect simultaneously!")
    print(f"\n⚠️  Press Ctrl+C to stop the server\n")
    print("=" * 60)

    # Run the Flask app
    # host='0.0.0.0' makes it accessible on the local network
    try:
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        if camera_thread is not None:
            camera_thread.stop()
