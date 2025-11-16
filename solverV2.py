import os
import random
import urllib.request
from typing import Optional

from openai import OpenAI
import dotenv


dotenv.load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


HACK_QUIZ_ENDPOINT = "https://drift-quiz-267976266267.us-central1.run.app"
HACK_QUIZ_S3_BASE = "https://drift-hack.s3.ap-south-1.amazonaws.com"


SYSTEM_PROMPT = """You are a puzzle-solving agent.
You receive a puzzle in the form of an image.
Your job is to analyze the puzzle and decide which of the three choices is correct:
1, 2, or 3.

Return ONLY a single character: "1", "2", or "3".
No explanation, no extra text, no spaces.
"""


def _fetch_hack_quiz_image_url() -> str:
    """Replicate the website logic to determine which puzzle image to load.

    The frontend does:
    - GET HACK_QUIZ_ENDPOINT -> text "mode"
    - if mode == "0": use defualt.png
    - else: mode is max random index; choose random 1..mode and load that PNG.
    """

    try:
        with urllib.request.urlopen(HACK_QUIZ_ENDPOINT, timeout=5) as resp:
            mode = resp.read().decode("utf-8").strip()
    except Exception:
        # Fallback if the endpoint is unreachable
        mode = "10"

    if mode == "0":
        return f"{HACK_QUIZ_S3_BASE}/defualt.png"

    try:
        max_num = int(mode)
    except ValueError:
        max_num = 10

    if max_num <= 0:
        max_num = 10

    random_num = random.randint(1, max_num)
    return f"{HACK_QUIZ_S3_BASE}/{random_num}.png"


def _call_model_with_image_url(image_url: str, extra_text: Optional[str] = None) -> int:
    """Send the puzzle image URL to an OpenAI vision-capable model and return 1, 2, or 3."""

    text_prompt = extra_text or (
        "You are given a puzzle shown in this image. "
        "Carefully read any text and analyze the picture, then determine which answer option 1, 2, or 3 is correct. "
        "Return ONLY a single character: '1', '2', or '3'."
    )

    # Use the Omni model gpt-4.2-o via the Responses API so it can handle any input type.
    resp = client.responses.create(
        model="gpt-4.2-o",
        instructions=SYSTEM_PROMPT,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": text_prompt},
                    {"type": "input_image", "image_url": image_url},
                ],
            }
        ],
    )

    # Extract text output from the Responses API structure.
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
        # Fallback: best-effort stringification
        raw = str(resp)

    raw = (raw or "").strip()
    for ch in raw:
        if ch in {"1", "2", "3"}:
            return int(ch)

    raise ValueError(f"Model did not return a valid answer: {raw!r}")


def solve_hack_quiz_puzzle(extra_text: Optional[str] = None) -> int:
    """Fetch a puzzle from the Drift hack-quiz and return the answer 1, 2, or 3."""

    image_url = _fetch_hack_quiz_image_url()
    return _call_model_with_image_url(image_url, extra_text=extra_text)


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Solver for Drift hack-quiz puzzles that outputs 1, 2, or 3.",
    )
    parser.add_argument(
        "--extra",
        help="Optional extra textual instruction to append to the puzzle.",
        default=None,
    )
    args = parser.parse_args()

    answer = solve_hack_quiz_puzzle(extra_text=args.extra)
    sys.stdout.write(str(answer) + "\n")

