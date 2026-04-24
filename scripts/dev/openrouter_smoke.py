#!/usr/bin/env python3
"""
Minimal OpenRouter smoke test for structured outputs and optional image input.

This is intended as a fast integration check before launching full experiment
batches. It exercises the real OpenRouter adapter with one tiny JSON-schema
request and, by default, includes a generated 1x1 PNG to test the multimodal
request path used by the experiments.

It can test one model or a list of models in sequence, which is useful for
preflighting whether specific ``:nitro`` model ids actually work with this
repo's request shape.
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

from providers.openrouter import chat as openrouter_chat

_TINY_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR42mP8z/C/HwAF/gL+"
    "qvOB0QAAAABJRU5ErkJggg=="
)


def _schema(mode: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["ok"]},
            "provider": {"type": "string", "enum": ["openrouter"]},
            "mode": {"type": "string", "enum": [mode]},
            "summary": {"type": "string"},
        },
        "required": ["status", "provider", "mode", "summary"],
        "additionalProperties": False,
    }


def _write_test_png(directory: Path) -> Path:
    path = directory / "openrouter_smoke_pixel.png"
    path.write_bytes(base64.b64decode(_TINY_PNG_BASE64))
    return path


def _messages(*, mode: str, image_path: Optional[Path]) -> list[dict[str, Any]]:
    system_prompt = (
        "You are validating an API integration. "
        "Return ONLY one JSON object matching the requested schema."
    )
    user_prompt = (
        "This is a smoke test for the OpenRouter integration.\n"
        f"Set status to 'ok', provider to 'openrouter', and mode to '{mode}'.\n"
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


def _with_variant(model: str, variant: str) -> str:
    if not variant or ":" in model:
        return model
    return f"{model}:{variant}"


def _run_one_model(
    *,
    model: str,
    mode: str,
    api_key: str,
    base_url: str,
    http_referer: str,
    app_title: str,
    require_parameters: bool,
    timeout_sec: int,
) -> tuple[bool, str]:
    try:
        with tempfile.TemporaryDirectory(prefix="openrouter_smoke_") as tmp_dir:
            temp_dir = Path(tmp_dir)
            image_path = None if mode == "text" else _write_test_png(temp_dir)
            content, _thinking = openrouter_chat(
                model=model,
                messages=_messages(mode=mode, image_path=image_path),
                api_key=api_key,
                base_url=base_url,
                http_referer=http_referer,
                app_title=app_title,
                require_parameters=require_parameters,
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
        or parsed.get("provider") != "openrouter"
        or parsed.get("mode") != mode
    ):
        return False, f"returned unexpected payload: {json.dumps(parsed, sort_keys=True)}"

    return True, json.dumps(parsed, indent=2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run minimal OpenRouter smoke test requests.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help=(
            "OpenRouter model id to test. May be repeated and may also contain "
            "comma-separated values."
        ),
    )
    parser.add_argument(
        "--model-file",
        default="",
        help="Optional file with one model id per line.",
    )
    parser.add_argument(
        "--variant",
        choices=["", "nitro"],
        default="",
        help="Optional variant suffix to append to bare model ids, e.g. ':nitro'.",
    )
    parser.add_argument(
        "--no-image",
        action="store_true",
        help="Skip the generated test image and do a text-only structured-output request.",
    )
    parser.add_argument(
        "--api-key",
        default="",
        help="Optional API key fallback. OPENROUTER_API_KEY still takes precedence.",
    )
    parser.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="OpenRouter-compatible base URL.",
    )
    parser.add_argument(
        "--http-referer",
        default="",
        help="Optional HTTP-Referer header for OpenRouter attribution.",
    )
    parser.add_argument(
        "--app-title",
        default="",
        help="Optional X-OpenRouter-Title header for OpenRouter attribution.",
    )
    parser.add_argument(
        "--allow-fallback-routing",
        action="store_true",
        help="Disable provider.require_parameters for this smoke test.",
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
    models = [_with_variant(model, args.variant) for model in requested_models]
    if not models:
        raise SystemExit("Provide at least one model via --model or --model-file.")

    mode = "text" if args.no_image else "vision"
    require_parameters = not args.allow_fallback_routing

    results: list[tuple[str, bool, str]] = []
    for model in models:
        ok, detail = _run_one_model(
            model=model,
            mode=mode,
            api_key=args.api_key,
            base_url=args.base_url,
            http_referer=args.http_referer,
            app_title=args.app_title,
            require_parameters=require_parameters,
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
            print(f"  {detail}")

    print(
        f"Summary: {passed}/{len(results)} passed | mode={mode} | "
        f"image_included={'no' if args.no_image else 'yes'} | "
        f"require_parameters={'yes' if require_parameters else 'no'}"
    )
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
