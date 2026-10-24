#!/usr/bin/env python3
"""
Minimal Gemini on Vertex AI smoke test for this repo's Google provider path.

This uses the same provider adapter as `client=google`, but forces Vertex AI
mode with the official `google-genai` SDK. It is intended as a fast check that
ADC, project/location wiring, structured output, and optional image input all
work before running a full experiment.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from providers.google import chat as google_chat

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z/C/HwAF/gL+"
    "qvOB0QAAAABJRU5ErkJggg=="
)


def _schema(mode: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
            "provider": {"type": "string", "enum": ["google-vertex"]},
            "mode": {"type": "string", "enum": [mode]},
            "summary": {"type": "string"},
        },
        "required": ["status", "provider", "mode", "summary"],
        "additionalProperties": False,
    }


def _write_test_png(directory: Path) -> Path:
    path = directory / "google_vertex_smoke_pixel.png"
    path.write_bytes(base64.b64decode(_TINY_PNG_BASE64))
    return path


def _messages(*, mode: str, image_path: Optional[Path]) -> list[dict[str, Any]]:
    system_prompt = (
        "You are validating a Vertex AI Gemini integration. "
        "Return ONLY one JSON object matching the requested schema."
    )
    user_prompt = (
        "This is a smoke test for the Google Vertex AI Gemini integration.\n"
        f"Set status to 'ok', provider to 'google-vertex', and mode to '{mode}'.\n"
        "Keep summary to one short sentence.\n"
        "Do not use markdown fences."
    )

    user_message: dict[str, Any] = {"role": "user", "content": user_prompt}
    if image_path is not None:
        user_message["images"] = [str(image_path)]

    return [
        {"role": "system", "content": system_prompt},
        user_message,
    ]


def _parse_models(values: Optional[Iterable[str]]) -> list[str]:
    models: list[str] = []
    for raw in values or []:
        for part in raw.split(","):
            model = part.strip()
            if model:
                models.append(model)
    return models


def _read_models_file(path: Optional[str]) -> list[str]:
    if not path:
        return []

    model_path = Path(path)
    try:
        lines = model_path.read_text().splitlines()
    except FileNotFoundError as exc:
        raise SystemExit(f"Model file not found: {model_path}") from exc

    models: list[str] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        models.append(stripped)
    return models


def _run_one_model(
    *,
    model: str,
    project: str,
    location: str,
    mode: str,
    timeout_sec: int,
) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="google_vertex_smoke_") as tmp_dir:
            temp_dir = Path(tmp_dir)
            image_path = None if mode == "text" else _write_test_png(temp_dir)
            content, _thinking = google_chat(
                model=model,
                messages=_messages(mode=mode, image_path=image_path),
                vertexai=True,
                project=project,
                location=location,
                temperature=0.0,
                timeout_sec=timeout_sec,
                output_schema=_schema(mode),
            )
    except Exception as exc:
        return False, str(exc)

    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return False, f"returned non-JSON content: {content}"

    if (
        parsed.get("status") != "ok"
        or parsed.get("provider") != "google-vertex"
        or parsed.get("mode") != mode
    ):
        return False, f"returned unexpected payload: {json.dumps(parsed, sort_keys=True)}"

    return True, json.dumps(parsed, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run minimal Gemini-on-Vertex-AI smoke test requests.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "Gemini model id to test. May be repeated and may also contain "
            "comma-separated values."
        ),
    )
    parser.add_argument(
        "--model-file",
        default="",
        help="Optional file with one model id per line.",
    )
    parser.add_argument(
        "--project",
        default="",
        help="Google Cloud project id. Falls back to GOOGLE_CLOUD_PROJECT.",
    )
    parser.add_argument(
        "--location",
        default="us-central1",
        help="Vertex AI location, e.g. us-central1 or global.",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip the generated test image and do a text-only structured-output request.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=120,
        help="Request timeout in seconds.",
    )
    args = parser.parse_args()

    requested_models = _parse_models(args.model)
    requested_models.extend(_read_models_file(args.model_file))
    if not requested_models:
        raise SystemExit("Provide at least one model via --model or --model-file.")

    mode = "text" if args.no_image else "vision"
    resolved_project = args.project.strip()
    if not resolved_project:
        import os

        resolved_project = os.getenv("GOOGLE_CLOUD_PROJECT", "").strip()
    if not resolved_project:
        raise SystemExit(
            "Provide --project or set GOOGLE_CLOUD_PROJECT before running this script."
        )

    results: list[tuple[str, bool, str]] = []
    for model in requested_models:
        ok, detail = _run_one_model(
            model=model,
            project=resolved_project,
            location=args.location,
            mode=mode,
            timeout_sec=args.timeout_sec,
        )
        results.append((model, ok, detail))

    if len(results) == 1:
        model, ok, detail = results[0]
        if not ok:
            print(f"Smoke test failed for model={model}: {detail}", file=sys.stderr)
            return 1
        print(
            f"Smoke test OK | model={model} | mode={mode} | "
            f"project={resolved_project} | location={args.location} | "
            f"image_included={'no' if args.no_image else 'yes'}"
        )
        print(detail)
        return 0

    passed = 0
    for model, ok, detail in results:
        status = "OK" if ok else "FAIL"
        print(f"[{status}] {model}")
        if ok:
            passed += 1
        else:
            print(detail)

    print(
        f"Completed {len(results)} Vertex AI smoke test(s): "
        f"{passed} passed, {len(results) - passed} failed."
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
