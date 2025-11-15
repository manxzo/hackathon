#!/usr/bin/env python3
"""
Webcam Client - Receives and displays frames from the streaming server
"""

import cv2
import numpy as np
import requests
import argparse
import os
from datetime import datetime

def receive_stream(url, save_dir=None):
    """
    Receive and display video stream from the server

    Args:
        url: Full URL to the video stream (e.g., http://192.168.1.109:5000/)
        save_dir: Directory to save captured frames (optional)
    """
    print(f"Connecting to stream: {url}")

    # List to store captured frames in memory
    captured_frames = []

    # Create save directory if specified
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        print(f"📁 Frames will be saved to: {save_dir}")

    try:
        # Open stream with requests
        stream = requests.get(url, stream=True, timeout=5)

        if stream.status_code != 200:
            print(f"Error: Server returned status code {stream.status_code}")
            return captured_frames

        print("✅ Connected!")
        print("\n🎮 Controls:")
        print("  'x' - Capture current frame")
        print("  'q' - Quit")
        print()

        # Buffer to accumulate bytes
        bytes_data = bytes()
        frame_count = 0

        for chunk in stream.iter_content(chunk_size=1024):
            bytes_data += chunk

            # Find start and end of JPEG frame
            a = bytes_data.find(b'\xff\xd8')  # JPEG start
            b = bytes_data.find(b'\xff\xd9')  # JPEG end

            if a != -1 and b != -1:
                jpg = bytes_data[a:b+2]
                bytes_data = bytes_data[b+2:]

                # Decode and display frame
                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)

                if frame is not None:
                    # Display frame count on screen
                    display_frame = frame.copy()
                    cv2.putText(display_frame, f"Captured: {len(captured_frames)}",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow('Webcam Stream', display_frame)

                    # Handle key presses
                    key = cv2.waitKey(1) & 0xFF

                    # Press 'x' to capture frame
                    if key == ord('x'):
                        frame_count += 1
                        captured_frames.append(frame.copy())

                        # Save to disk if directory specified
                        if save_dir:
                            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                            filename = os.path.join(save_dir, f"frame_{timestamp}.jpg")
                            cv2.imwrite(filename, frame)
                            print(f"📸 Captured frame #{len(captured_frames)} → {filename}")
                        else:
                            print(f"📸 Captured frame #{len(captured_frames)} (in memory)")

                    # Press 'q' to quit
                    elif key == ord('q'):
                        break

    except requests.exceptions.RequestException as e:
        print(f"❌ Connection error: {e}")
    except KeyboardInterrupt:
        print("\n\n🛑 Client stopped")
    finally:
        cv2.destroyAllWindows()
        print(f"\n✅ Total frames captured: {len(captured_frames)}")
        return captured_frames

def main():
    parser = argparse.ArgumentParser(description='Webcam Stream Client')
    parser.add_argument('url', help='Stream URL (e.g., http://192.168.1.109:5000/)')
    parser.add_argument('-s', '--save-dir', type=str, default=None,
                        help='Directory to save captured frames (optional). If not specified, frames are kept in memory only.')
    args = parser.parse_args()

    # Receive stream and get captured frames
    captured_frames = receive_stream(args.url, args.save_dir)

    # Example: Process captured frames here
    if captured_frames:
        print("\n🔧 Processing captured frames...")
        for i, frame in enumerate(captured_frames):
            # Your processing code here
            # Example: Print frame shape
            print(f"  Frame {i+1}: shape={frame.shape}, dtype={frame.dtype}")

        print("\n💡 Tip: Use the 'captured_frames' list for further processing")
        print("   Each frame is a numpy array (height, width, channels)")

if __name__ == "__main__":
    main()
