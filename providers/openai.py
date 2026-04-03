"""
OpenAI provider adapter for the navigation VLM.

Uses the Responses API (openai>=1.0.0).  Structured output is requested via
the ``text`` parameter so the model returns valid JSON matching the schema.

API key precedence (highest first):
  1. OPENAI_API_KEY environment variable
  2. ``api_key`` argument passed by the caller
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _mime_type_for_path(image_path: str) -> str:
    ext = Path(image_path).suffix.lstrip(".").lower()
    if ext in {"jpg", "jpeg"}:
        return "image/jpeg"
    if ext == "png":
        return "image/png"
    if ext == "webp":
        return "image/webp"
    if ext == "gif":
        return "image/gif"
    return "image/jpeg"


def _encode_image(image_path: str) -> str:
    """Read a local image file and return a base64-encoded string."""
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _build_input_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert internal Ollama-format messages to OpenAI Responses API messages.

    This preserves the original message roles so the system prompt keeps its
    higher-priority instruction semantics in the Responses API.
    """
    input_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        images: List[str] = msg.get("images", [])
        content_items: List[Dict[str, Any]] = []

        # Images first, then text
        for image_path in images:
            try:
                b64 = _encode_image(image_path)
                media_type = _mime_type_for_path(image_path)
                content_items.append({
                    "type": "input_image",
                    "image_url": f"data:{media_type};base64,{b64}",
                })
            except OSError:
                pass  # skip unreadable images

        if content:
            content_items.append({"type": "input_text", "text": content})

        if content_items:
            input_messages.append({"role": role, "content": content_items})

    return input_messages


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: str = "",
    temperature: float = 0.2,
    output_schema: Any,
    stream_thinking: bool = False,
    timeout_sec: int = 180,
) -> Tuple[str, Optional[str]]:
    """
    Send messages to the OpenAI Responses API and return (content, None).

    ``stream_thinking`` is accepted for interface compatibility but ignored;
    OpenAI reasoning tokens are not surfaced here.
    """
    import openai  # lazy import — optional dependency

    resolved_key = os.getenv("OPENAI_API_KEY") or api_key
    if not resolved_key:
        raise ValueError(
            "OpenAI API key not found. Set OPENAI_API_KEY or pass openai.api_key in config."
        )

    client = openai.OpenAI(api_key=resolved_key, timeout=timeout_sec)

    input_messages = _build_input_messages(messages)

    response = client.responses.create(
        model=model,
        input=input_messages,
        temperature=temperature,
        text={
            "format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "navigation_response",
                    "schema": output_schema,
                    "strict": True,
                },
            }
        },
    )

    content = response.output_text
    return content, None
