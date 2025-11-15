"""
GalaxyRVR Simple Keyboard Control with Camera Stream (Refactored)

This script uses the GalaxyRVR library for keyboard control with camera display.
Uses non-blocking keyboard input with periodic camera updates.

Requirements:
    pip install websockets opencv-python numpy requests

Usage:
    python simple_control_stream_refactored.py
"""

import asyncio
import cv2
import sys
import tty
import termios
import select
from galaxyrvr import GalaxyRVR
from galaxyrvr_camera import CameraStream


class NonBlockingKeyboardController:
    """Lightweight keyboard controller that doesn't block camera display"""

    def __init__(self, robot, camera, normal_speed=60, boost_speed=90):
        self.robot = robot
        self.camera = camera
        self.normal_speed = normal_speed
        self.boost_speed = boost_speed
        self.running = True

    def get_key(self):
        """Non-blocking keyboard input"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    async def run(self):
        """Main control loop with camera display"""
        print("\n" + "=" * 60)
        print("KEYBOARD CONTROLS:")
        print("=" * 60)
        print("  'w' or 'W'    - Move forward (Shift+W for boost)")
        print("  's' or 'S'    - Move backward (Shift+S for boost)")
        print("  'a' or 'A'    - Rotate left (Shift+A for boost)")
        print("  'd' or 'D'    - Rotate right (Shift+D for boost)")
        print("  SPACE         - Stop")
        print("  'q'           - Quit")
        print("=" * 60 + "\n")

        old_settings = termios.tcgetattr(sys.stdin)

        try:
            tty.setraw(sys.stdin.fileno())
            last_key = None

            while self.running and self.robot.running:
                # Display camera frame (non-blocking)
                frame = self.camera.get_frame()
                if frame is not None:
                    cv2.imshow('GalaxyRVR Camera Feed', frame)
                    cv2.waitKey(1)

                # Get keyboard input (non-blocking)
                key = self.get_key()

                if key:
                    if key == 'q':
                        print("\n\nQuitting...")
                        self.running = False
                        break
                    elif key == ' ':
                        self.robot.stop()
                        await self.robot.send()
                    elif key in ['w', 'W', 's', 'S', 'a', 'A', 'd', 'D']:
                        speed = self.boost_speed if key.isupper() else self.normal_speed

                        if key.lower() == 'w':
                            self.robot.forward(speed)
                        elif key.lower() == 's':
                            self.robot.backward(speed)
                        elif key.lower() == 'a':
                            self.robot.turn_left(speed)
                        elif key.lower() == 'd':
                            self.robot.turn_right(speed)

                        await self.robot.send()

                    last_key = key
                else:
                    # No key pressed - stop robot
                    if last_key is not None:
                        self.robot.stop()
                        await self.robot.send()
                        last_key = None

                # Small delay - this is key to not blocking
                await asyncio.sleep(0.02)  # 50 Hz

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


async def main():
    """Main entry point"""

    # Configuration
    robot_ip = "192.168.1.216"
    # robot_ip = "192.168.4.1"
    port = 8765

    print("=" * 60)
    print("GalaxyRVR Simple Keyboard Control with Camera Stream")
    print("=" * 60)

    # Create and connect to robot
    robot = GalaxyRVR(robot_ip, port)
    if not await robot.connect():
        print("Failed to connect. Exiting...")
        return

    # Start camera stream (fetch only, display in main loop)
    camera = CameraStream(robot_ip, display=False)
    camera.start()

    # Give camera time to connect
    await asyncio.sleep(2)

    # Create and run keyboard controller with integrated camera display
    controller = NonBlockingKeyboardController(robot, camera, normal_speed=60, boost_speed=90)
    await controller.run()

    # Cleanup
    camera.stop()
    cv2.destroyAllWindows()
    await robot.disconnect()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
