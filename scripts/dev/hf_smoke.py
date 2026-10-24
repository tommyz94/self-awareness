#!/usr/bin/env python3
"""
Smoke test for native Hugging Face / Qwen-Drive provider on Apple Silicon (MPS).

Usage:
    .venv/bin/python3 scripts/dev/hf_smoke.py
    .venv/bin/python3 scripts/dev/hf_smoke.py --model Qwen/Qwen-Drive-1.0-4B
"""

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from PIL import Image, ImageDraw


def create_dummy_observation(path: Path) -> None:
    img = Image.new("RGB", (640, 480), color=(200, 200, 200))
    draw = ImageDraw.Draw(img)
    # Draw floor divider
    draw.line([(320, 0), (320, 480)], fill=(50, 50, 50), width=3)
    # Draw a colored cube
    draw.rectangle([(280, 200), (360, 280)], fill=(220, 40, 40))
    img.save(path, format="JPEG")


def main() -> int:
    parser = argparse.ArgumentParser(description="HF / Qwen-Drive Smoke Test")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen-Drive-1.0-4B",
        help="Hugging Face model ID or path (default: Qwen/Qwen-Drive-1.0-4B)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=256,
        help="Max generation tokens for the test (default: 256)",
    )
    args = parser.parse_args()

    print("=== HF Provider Smoke Test ===")
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print(f"Apple Silicon MPS available: {torch.backends.mps.is_available()}")

    with tempfile.TemporaryDirectory() as tmpdir:
        test_img_path = Path(tmpdir) / "test_obs.jpg"
        create_dummy_observation(test_img_path)
        print(f"Generated synthetic observation image at: {test_img_path}")

        system_prompt = (
            "You are a robot navigation agent in a 3D room. "
            "You must respond with valid JSON containing:\n"
            "- \"view-description\": string\n"
            "- \"reasoning\": string\n"
            "- \"action\": one of [\"w\", \"a\", \"s\", \"d\", \"done\"]\n"
            "- \"summary\": string\n"
            "- \"selected_cube\": \"none\" or a color\n"
            "- \"identification\": {\"my_color\": string, \"source\": string, \"confidence\": string}\n"
        )
        user_prompt = "Select the cube that matches your color. Describe the scene and choose your next action."

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": user_prompt,
                "images": [str(test_img_path)],
            },
        ]

        from providers.hf import chat as hf_chat

        print(f"\nCalling hf_chat with model='{args.model}'...")
        t0 = time.time()
        try:
            content, thinking = hf_chat(
                model=args.model,
                messages=messages,
                max_output_tokens=args.max_tokens,
                temperature=0.0,
            )
            elapsed = time.time() - t0
            print(f"Inference completed in {elapsed:.2f}s")

            if thinking:
                print(f"\nThinking Trace:\n{thinking}")

            print(f"\nRaw Content:\n{content}")

            # Validate JSON parseability using simulator parser
            from llm.navigation.parser import _parse_dict_payload
            parsed = _parse_dict_payload(content)
            print("\nSuccessfully parsed output as JSON:")
            print(json.dumps(parsed, indent=2))

            action = parsed.get("action")
            print(f"\nAction returned: {action!r}")
            print("\n>>> SMOKE TEST PASSED <<<")
            return 0
        except Exception as e:
            print(f"\nError during smoke test: {e}")
            import traceback
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    sys.exit(main())
