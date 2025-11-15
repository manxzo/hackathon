"""
GalaxyRVR Simple Keyboard Control (Refactored)

This script uses the GalaxyRVR library for simple keyboard control.

Requirements:
    pip install websockets

Usage:
    python simple_control_refactored.py
"""

import asyncio
from galaxyrvr import GalaxyRVR
from galaxyrvr_keyboard import KeyboardController


async def main():
    """Main entry point"""

    # Configuration
    robot_ip = "192.168.1.216"
    robot_ip = "192.168.1.113" ## 2nd robot
    # robot_ip = "192.168.4.1"
    port = 8765

    print("=" * 60)
    print("GalaxyRVR Simple Keyboard Control")
    print("=" * 60)

    # Create and connect to robot
    robot = GalaxyRVR(robot_ip, port)
    if not await robot.connect():
        print("Failed to connect. Exiting...")
        return

    # Create keyboard controller
    keyboard = KeyboardController(robot, normal_speed=60, boost_speed=90)

    # Run keyboard control
    await keyboard.run()

    # Cleanup
    await robot.disconnect()
    print("Done!")


if __name__ == "__main__":
    asyncio.run(main())
