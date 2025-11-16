import asyncio
import base64
import os
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import dotenv
import numpy as np
import requests
from openai import OpenAI


dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


HERE = Path(__file__).resolve().parent
ROBOT_CLIENT_DIR = HERE / "hackathon" / "robot_code" / "python_client"
if str(ROBOT_CLIENT_DIR) not in sys.path:
    sys.path.append(str(ROBOT_CLIENT_DIR))

from galaxyrvr import GalaxyRVR  # type: ignore  # noqa: E402
from galaxyrvr_camera import CameraStream as RobotCameraStream  # type: ignore  # noqa: E402


def _frame_to_data_url(frame: np.ndarray) -> str:
    ok, buffer = cv2.imencode(".jpg", frame)
    if not ok:
        raise RuntimeError("Failed to encode frame to JPEG")
    b64 = base64.b64encode(buffer.tobytes()).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}"


class OverheadWebcamStream:
    def __init__(self, url: str) -> None:
        self.url = url
        self.latest_frame: Optional[np.ndarray] = None
        self.error: Optional[str] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2)

    def get_frame(self) -> Optional[np.ndarray]:
        return self.latest_frame

    def _loop(self) -> None:
        while self._running:
            try:
                stream = requests.get(self.url, stream=True, timeout=5)
                if stream.status_code != 200:
                    self.error = f"HTTP {stream.status_code}"
                    time.sleep(1)
                    continue

                bytes_data = bytes()
                for chunk in stream.iter_content(chunk_size=1024):
                    if not self._running:
                        break

                    bytes_data += chunk
                    a = bytes_data.find(b"\xff\xd8")
                    b = bytes_data.find(b"\xff\xd9")
                    if a != -1 and b != -1:
                        jpg = bytes_data[a : b + 2]
                        bytes_data = bytes_data[b + 2 :]
                        frame = cv2.imdecode(
                            np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR
                        )
                        if frame is not None:
                            self.latest_frame = frame
            except Exception as e:  # pragma: no cover - network errors
                self.error = str(e)
                time.sleep(2)


AGENT_SYSTEM_PROMPT = """You are a real-time robot navigation controller for a small rover in a physical arena.\n\nEnvironment:\n- The rover drives on a white floor with various obstacles (dark crumpled blobs, beige blocks, white packages).\n- At the far end of the arena there are three horizontal blue finish lines.\n- From the overhead camera perspective, the lanes are numbered left to right as 1, 2, 3.\n- The goal is to move the robot so that its middle wheel crosses the blue finish line corresponding to target_lane.\n\nObservations you get on each step:\n- Robot sensors:\n  - ultrasonic_distance_cm (float or null): approximate distance to obstacle in front.\n  - ir_left, ir_right (0 or 1, or null): 1 means very close obstacle on that side, 0 means clear.\n- Robot forward-facing camera image.\n- Overhead webcam image (if provided).\n\nControls available:\n- You do NOT directly control individual wheels.\n- Instead you output high-level continuous commands:\n  - linear_speed in [-1.0, 1.0]: forward/backward speed (1.0 is full forward, -1.0 full reverse).\n  - angular_speed in [-1.0, 1.0]: rotation (-1.0 is turn left on the spot, 1.0 is turn right on the spot).\n- The controller will map these to differential wheel speeds.\n\nSafety and navigation requirements:\n- Never intentionally collide with obstacles.\n- Use ultrasonic and IR sensors as hard safety: if very close to an obstacle, slow down or stop and steer away.\n- Use the cameras to choose a path around obstacles and to align with the correct blue lane at the end.\n- Prefer smooth, incremental movements; avoid oscillating rapidly.\n\nOutput format:\n- You MUST respond with a single JSON object, no extra text, markdown, or explanation.\n- The JSON schema is:\n  {\n    "action": "MOVE" | "STOP" | "FINISH",\n    "linear_speed": float,   // between -1.0 and 1.0\n    "angular_speed": float,  // between -1.0 and 1.0\n    "done": bool\n  }\n\nGuidance:\n- Use "MOVE" with non-zero speeds for normal motion.\n- Use "STOP" with small or zero speeds when you need to pause or re-evaluate near obstacles.\n- Use "FINISH" when you believe the robot's center is aligned with the correct blue line and it should stop permanently (done=true).\n- Keep linear_speed modest (e.g. around 0.3–0.6) unless the path is very clear.\n- If sensors report dangerously small distance or IR=1, reduce linear_speed and apply angular_speed to steer away.\n"""


@dataclass
class RobotState:
    ultrasonic_distance_cm: Optional[float]
    ir_left: Optional[int]
    ir_right: Optional[int]
    battery_voltage: Optional[float]


class RobotNavigationAgent:
    def __init__(
        self,
        robot_ip: str,
        target_lane: int,
        ws_port: int = 8765,
        overhead_webcam_url: Optional[str] = None,
    ) -> None:
        self.robot_ip = robot_ip
        self.ws_port = ws_port
        self.target_lane = int(target_lane)
        self.overhead_webcam_url = overhead_webcam_url

        self.robot: Optional[GalaxyRVR] = None
        self.robot_cam: Optional[RobotCameraStream] = None
        self.overhead_cam: Optional[OverheadWebcamStream] = None

    async def _connect(self) -> None:
        robot = GalaxyRVR(self.robot_ip, self.ws_port)
        ok = await robot.connect()
        if not ok:
            raise RuntimeError(f"Failed to connect to robot at {self.robot_ip}:{self.ws_port}")
        self.robot = robot

        self.robot_cam = RobotCameraStream(self.robot_ip, display=False)
        self.robot_cam.start()

        if self.overhead_webcam_url is not None:
            self.overhead_cam = OverheadWebcamStream(self.overhead_webcam_url)
            self.overhead_cam.start()

        await asyncio.sleep(1.5)

    async def _disconnect(self) -> None:
        if self.overhead_cam is not None:
            self.overhead_cam.stop()
            self.overhead_cam = None

        if self.robot_cam is not None:
            self.robot_cam.stop()
            self.robot_cam = None

        if self.robot is not None:
            try:
                self.robot.stop()
                await self.robot.send()
            except Exception:
                pass
            await self.robot.disconnect()
            self.robot = None

    def _get_robot_state(self) -> RobotState:
        if self.robot is None:
            return RobotState(None, None, None, None)
        return RobotState(
            ultrasonic_distance_cm=self.robot.ultrasonic_distance,
            ir_left=self.robot.ir_left,
            ir_right=self.robot.ir_right,
            battery_voltage=self.robot.battery_voltage,
        )

    def _get_images(self) -> Dict[str, np.ndarray]:
        images: Dict[str, np.ndarray] = {}
        if self.robot_cam is not None:
            frame = self.robot_cam.get_frame()
            if frame is not None:
                images["robot_camera"] = frame
        if self.overhead_cam is not None:
            frame2 = self.overhead_cam.get_frame()
            if frame2 is not None:
                images["overhead_camera"] = frame2
        return images

    def _build_text_observation(self, state: RobotState, step: int) -> str:
        return (
            f"Step: {step}\n"
            f"Target lane (blue line index): {self.target_lane} (1=left, 3=right).\n"
            f"Ultrasonic distance (cm): {state.ultrasonic_distance_cm}.\n"
            f"IR left (1=obstacle, 0=clear): {state.ir_left}.\n"
            f"IR right (1=obstacle, 0=clear): {state.ir_right}.\n"
            f"Battery voltage (if available): {state.battery_voltage}.\n"
            "Use the images plus these readings to decide a safe motion command towards the desired lane."
        )

    def _call_model_for_action(
        self,
        state: RobotState,
        images: Dict[str, np.ndarray],
        step: int,
    ) -> Dict[str, Any]:
        text_obs = self._build_text_observation(state, step)

        content = [
            {"type": "input_text", "text": text_obs},
        ]

        if "robot_camera" in images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": {"url": _frame_to_data_url(images["robot_camera"])},
                }
            )
        if "overhead_camera" in images:
            content.append(
                {
                    "type": "input_image",
                    "image_url": {"url": _frame_to_data_url(images["overhead_camera"])},
                }
            )

        resp = client.responses.create(
            model="gpt-5.1-o",
            instructions=AGENT_SYSTEM_PROMPT,
            input=[{"role": "user", "content": content}],
        )

        raw = ""
        try:
            first_output = resp.output[0]
            if getattr(first_output, "content", None):
                for item in first_output.content:
                    text_obj = getattr(item, "text", None)
                    if text_obj is not None:
                        value = getattr(text_obj, "value", None)
                        raw += value if isinstance(value, str) else str(text_obj)
        except Exception:
            raw = str(resp)

        raw = (raw or "").strip()
        if raw.startswith("```"):
            parts = raw.split("```", 2)
            if len(parts) >= 2:
                raw = parts[1]
        raw = raw.strip()

        import json

        try:
            data = json.loads(raw)
        except Exception:
            data = {"action": "STOP", "linear_speed": 0.0, "angular_speed": 0.0, "done": False}
        return data

    async def _apply_action(self, action: Dict[str, Any]) -> bool:
        if self.robot is None:
            return True

        act = str(action.get("action", "MOVE")).upper()
        try:
            linear = float(action.get("linear_speed", 0.0))
        except Exception:
            linear = 0.0
        try:
            angular = float(action.get("angular_speed", 0.0))
        except Exception:
            angular = 0.0

        linear = max(-1.0, min(1.0, linear))
        angular = max(-1.0, min(1.0, angular))

        done = bool(action.get("done", False))

        if act in {"STOP", "FINISH"}:
            self.robot.stop()
            await self.robot.send()
        else:
            max_speed = 70.0
            left = max_speed * (linear - angular)
            right = max_speed * (linear + angular)
            self.robot.set_motors(left, right)
            await self.robot.send()

        return done

    async def run(self, max_steps: int = 300, step_delay: float = 0.4) -> None:
        await self._connect()
        try:
            for step in range(max_steps):
                state = self._get_robot_state()
                images = self._get_images()
                action = self._call_model_for_action(state, images, step)
                done = await self._apply_action(action)
                if done:
                    break
                await asyncio.sleep(step_delay)
        finally:
            await self._disconnect()


async def _main_cli() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="LLM-based GalaxyRVR navigation agent")
    parser.add_argument("lane", type=int, help="Target finish lane (1, 2, or 3)")
    parser.add_argument("--robot-ip", dest="robot_ip", default="192.168.1.216")
    parser.add_argument("--ws-port", dest="ws_port", type=int, default=8765)
    parser.add_argument(
        "--overhead-url",
        dest="overhead_url",
        default=None,
        help="Optional MJPEG stream URL from overhead webcam server",
    )
    args = parser.parse_args()

    agent = RobotNavigationAgent(
        robot_ip=args.robot_ip,
        target_lane=args.lane,
        ws_port=args.ws_port,
        overhead_webcam_url=args.overhead_url,
    )
    await agent.run()


if __name__ == "__main__":
    asyncio.run(_main_cli())

