"""
GalaxyRVR Keyboard Control Helper

This module provides utilities for keyboard control of the robot.

Example usage:
    from galaxyrvr import GalaxyRVR
    from galaxyrvr_keyboard import KeyboardController

    async def main():
        robot = GalaxyRVR("192.168.1.216")
        await robot.connect()

        keyboard = KeyboardController(robot, normal_speed=60, boost_speed=90)
        await keyboard.run()

        await robot.disconnect()
"""

import asyncio
import sys
import tty
import termios
import select


class KeyboardController:
    """Helper class for keyboard control of GalaxyRVR"""

    def __init__(self, robot, normal_speed=60, boost_speed=90):
        """
        Initialize keyboard controller

        Args:
            robot: GalaxyRVR instance
            normal_speed: Normal movement speed (0-100)
            boost_speed: Boosted speed when Shift is pressed (0-100)
        """
        self.robot = robot
        self.normal_speed = normal_speed
        self.boost_speed = boost_speed
        self.running = True

    def get_key(self):
        """Non-blocking keyboard input"""
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def print_controls(self):
        """Print keyboard controls"""
        print("\n" + "=" * 60)
        print("KEYBOARD CONTROLS:")
        print("=" * 60)
        print("  'w' or 'W'    - Move forward (Shift+W for boost)")
        print("  's' or 'S'    - Move backward (Shift+S for boost)")
        print("  'a' or 'A'    - Rotate left (Shift+A for boost)")
        print("  'd' or 'D'    - Rotate right (Shift+D for boost)")
        print("  SPACE         - Stop")
        print("  'q'           - Quit")
        print("=" * 60)
        print(f"Normal Speed: {self.normal_speed}% | Boost Speed: {self.boost_speed}%")
        print("=" * 60 + "\n")

    async def run(self):
        """Main keyboard control loop"""
        self.print_controls()

        # Save terminal settings
        old_settings = termios.tcgetattr(sys.stdin)

        try:
            # Set terminal to raw mode
            tty.setraw(sys.stdin.fileno())

            last_key = None

            while self.running and self.robot.running:
                # Get keyboard input
                key = self.get_key()

                if key:
                    # Process key press
                    if key == 'q':
                        print("\n\nQuitting...")
                        self.running = False
                        break
                    elif key == ' ':  # Space bar
                        self.robot.stop()
                        await self.robot.send()
                        print("■ STOP")
                    elif key in ['w', 'W', 's', 'S', 'a', 'A', 'd', 'D']:
                        # Check for diagonal movements
                        if last_key and last_key != key:
                            keys = {last_key.lower(), key.lower()}
                            boost = 'W' in [last_key, key] or 'S' in [last_key, key] or \
                                   'A' in [last_key, key] or 'D' in [last_key, key]
                            speed = self.boost_speed if boost else self.normal_speed

                            if keys == {'w', 'd'}:
                                self.robot.forward_right(speed)
                                print(f"↗ FORWARD-RIGHT: {speed}%")
                            elif keys == {'w', 'a'}:
                                self.robot.forward_left(speed)
                                print(f"↖ FORWARD-LEFT: {speed}%")
                            elif keys == {'s', 'd'}:
                                self.robot.backward_right(speed)
                                print(f"↘ BACKWARD-RIGHT: {speed}%")
                            elif keys == {'s', 'a'}:
                                self.robot.backward_left(speed)
                                print(f"↙ BACKWARD-LEFT: {speed}%")
                            else:
                                self._process_single_key(key)
                        else:
                            self._process_single_key(key)

                        await self.robot.send()

                    last_key = key
                else:
                    # No key pressed - stop robot
                    if last_key is not None:
                        self.robot.stop()
                        await self.robot.send()
                        last_key = None

                # Small delay for responsive key detection
                await asyncio.sleep(0.02)  # 50 Hz

        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
        finally:
            # Restore terminal settings
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

    def _process_single_key(self, key):
        """Process single key press"""
        if key == 'w':
            self.robot.forward(self.normal_speed)
            print(f"⬆ FORWARD: {self.normal_speed}%")
        elif key == 'W':
            self.robot.forward(self.boost_speed)
            print(f"⬆ FORWARD: {self.boost_speed}% (BOOST)")
        elif key == 's':
            self.robot.backward(self.normal_speed)
            print(f"⬇ BACKWARD: {self.normal_speed}%")
        elif key == 'S':
            self.robot.backward(self.boost_speed)
            print(f"⬇ BACKWARD: {self.boost_speed}% (BOOST)")
        elif key == 'a':
            self.robot.turn_left(self.normal_speed)
            print(f"↺ ROTATE LEFT: {self.normal_speed}%")
        elif key == 'A':
            self.robot.turn_left(self.boost_speed)
            print(f"↺ ROTATE LEFT: {self.boost_speed}% (BOOST)")
        elif key == 'd':
            self.robot.turn_right(self.normal_speed)
            print(f"↻ ROTATE RIGHT: {self.normal_speed}%")
        elif key == 'D':
            self.robot.turn_right(self.boost_speed)
            print(f"↻ ROTATE RIGHT: {self.boost_speed}% (BOOST)")
