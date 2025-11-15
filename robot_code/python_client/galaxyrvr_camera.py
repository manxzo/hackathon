"""
GalaxyRVR Camera Helper

This module provides utilities for accessing the robot's camera feed.

Example usage:
    from galaxyrvr_camera import CameraStream

    camera = CameraStream("192.168.1.216")
    camera.start()

    while True:
        frame = camera.get_frame()
        if frame is not None:
            # Process frame
            cv2.imshow('Camera', frame)
            cv2.waitKey(1)

    camera.stop()
"""

import cv2
import numpy as np
import requests
import threading
import time


class CameraStream:
    """Helper class to stream camera feed from ESP32-CAM"""

    def __init__(self, robot_ip="192.168.1.216", port=9000, display=False):
        """
        Initialize camera stream

        Args:
            robot_ip: IP address of the ESP32-CAM
            port: Camera stream port (default: 9000)
            display: If True, automatically display camera in separate window (default: False)
        """
        self.camera_url = f"http://{robot_ip}:{port}/mjpg"
        self.latest_frame = None
        self.camera_error = None
        self.running = False
        self.display = display
        self._thread = None
        self._display_thread = None

    def start(self):
        """Start camera streaming in background thread"""
        if self.running:
            return

        self.running = True
        self._thread = threading.Thread(target=self._stream_thread, daemon=True)
        self._thread.start()

        # Start display thread if display mode is enabled
        if self.display:
            self._display_thread = threading.Thread(target=self._display_thread_func, daemon=True)
            self._display_thread.start()

        print(f"✓ Camera stream started from {self.camera_url}")

    def stop(self):
        """Stop camera streaming"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._display_thread:
            self._display_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print("✓ Camera stream stopped")

    def get_frame(self):
        """
        Get the latest camera frame

        Returns:
            numpy.ndarray: Latest frame or None if not available
        """
        return self.latest_frame

    def get_error(self):
        """
        Get the last camera error if any

        Returns:
            str: Error message or None
        """
        return self.camera_error

    def _display_thread_func(self):
        """Separate thread for displaying camera feed (runs independently)"""
        print("✓ Camera display window started")
        while self.running:
            if self.latest_frame is not None:
                cv2.imshow('GalaxyRVR Camera Feed', self.latest_frame)
                # Use a short waitKey to keep window responsive
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.running = False
                    break
            time.sleep(0.01)  # 100 Hz display refresh
        cv2.destroyAllWindows()
        print("✓ Camera display window closed")

    def _stream_thread(self):
        """Background thread to continuously fetch camera frames"""
        while self.running:
            try:
                response = requests.get(self.camera_url, stream=True, timeout=10)

                if response.status_code == 200:
                    self.camera_error = None
                    bytes_data = bytes()

                    for chunk in response.iter_content(chunk_size=1024):
                        if not self.running:
                            break

                        bytes_data += chunk

                        # Find JPEG boundaries
                        a = bytes_data.find(b'\xff\xd8')
                        b = bytes_data.find(b'\xff\xd9')

                        if a != -1 and b != -1:
                            jpg = bytes_data[a:b+2]
                            bytes_data = bytes_data[b+2:]

                            # Decode frame
                            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                            if frame is not None:
                                self.latest_frame = frame
                else:
                    self.camera_error = f"HTTP {response.status_code}"
                    time.sleep(1)

            except Exception as e:
                self.camera_error = str(e)
                time.sleep(2)
