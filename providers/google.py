"""
Google Gemini provider adapter for the navigation VLM.

Uses the ``google-genai`` SDK (google-genai>=1.0.0).  Structured output is
requested via ``response_mime_type="application/json"`` and
``response_schema``.  A ``_add_property_ordering`` helper recursively injects
``propertyOrdering`` so the schema works correctly with Gemini 2.0 models.

API key precedence (highest first):
  1. GEMINI_API_KEY environment variable
  2. GOOGLE_API_KEY environment variable (backward-compatible fallback)
  3. ``api_key`` argument passed by the caller
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _add_property_ordering(schema: Any) -> Any:
    """
    Recursively inject ``propertyOrdering`` into every object node of the
    JSON schema so that Gemini 2.0 structured-output mode works correctly.
    Returns a deep copy — the original schema is never mutated.
    """
    schema = copy.deepcopy(schema)

    def _visit(node: Any) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object" and "properties" in node:
            node["propertyOrdering"] = list(node["properties"].keys())
            for child in node["properties"].values():
                _visit(child)
        for value in node.values():
            if isinstance(value, (dict, list)):
                _visit(value)

    _visit(schema)
    return schema


def _resolve_api_key(api_key: str) -> str:
    return (
        os.getenv("GEMINI_API_KEY")
        or os.getenv("GOOGLE_API_KEY")
        or api_key
    )


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
    Send messages to the Google Gemini API and return (content, None).

    ``stream_thinking`` is accepted for interface compatibility but ignored.
    """
    import google.genai as genai  # lazy import — optional dependency
    from google.genai import types

    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "Google API key not found. "
            "Set GEMINI_API_KEY (or GOOGLE_API_KEY) or pass google.api_key in config."
        )

    client = genai.Client(api_key=resolved_key)

    # Extract system instruction from the first system message (if any)
    system_instruction: Optional[str] = None
    user_messages: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            system_instruction = msg.get("content", "")
        else:
            user_messages.append(msg)

    # Build content parts for user turn
    parts: List[Any] = []
    for msg in user_messages:
        content = msg.get("content", "")
        images: List[str] = msg.get("images", [])

        for image_path in images:
            try:
                image_bytes = Path(image_path).read_bytes()
                mime_type = _mime_type_for_path(image_path)
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
            except OSError:
                pass  # skip unreadable images

        if content:
            parts.append(types.Part.from_text(text=content))

    contents = [types.Content(role="user", parts=parts)]

    generate_config = types.GenerateContentConfig(
        temperature=temperature,
        response_mime_type="application/json",
        response_json_schema=_add_property_ordering(output_schema),
        system_instruction=system_instruction,
    )

    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=generate_config,
    )

    return response.text, None
