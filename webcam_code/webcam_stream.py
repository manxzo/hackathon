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

app = Flask(__name__)

# Global variables for frame sharing
camera = None
camera_index = 0  # Default camera index
output_frame = None
lock = threading.Lock()
camera_thread = None

class CameraStream:
    """Background thread to continuously capture frames"""
    def __init__(self, camera_index, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None
        self.stopped = False

    def start(self):
        """Start the camera capture thread"""
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        """Continuously capture frames from camera"""
        global output_frame, lock

        self.camera = cv2.VideoCapture(self.camera_index)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_FPS, 30)

        while not self.stopped:
            success, frame = self.camera.read()
            if not success:
                continue

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
    args = parser.parse_args()

    camera_index = args.camera
    port = args.port

    # Parse resolution
    width, height = map(int, args.resolution.split('x'))

    # Start the camera capture thread
    camera_thread = CameraStream(camera_index, width, height).start()

    # Give the camera time to initialize
    time.sleep(2)

    local_ip = get_local_ip()
    print("=" * 60)
    print("🎥 Webcam Streaming Server Starting...")
    print("=" * 60)
    print(f"\n📹 Using Camera Index: {camera_index}")
    print(f"\n📐 Resolution: {width}x{height}")
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
