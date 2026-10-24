"""
Anthropic Claude provider adapter for the navigation VLM.

Uses the native Anthropic Messages API (anthropic>=0.68.0). Structured output
is requested via ``output_config={"format": {"type": "json_schema", ...}}``.

Supports both:
  - the direct Anthropic API via ``ANTHROPIC_API_KEY``
  - Claude on Vertex AI via ``anthropic[vertex]`` with ADC / project / region

API key precedence (highest first):
  1. ANTHROPIC_API_KEY environment variable
  2. ``api_key`` argument passed by the caller
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_MAX_OUTPUT_TOKENS = 4096
_VERTEX_ENV_KEYS_TRUE = {"1", "true", "yes", "on"}


def _resolve_api_key(api_key: str) -> str:
    return os.getenv("ANTHROPIC_API_KEY") or api_key


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in _VERTEX_ENV_KEYS_TRUE


def _resolve_vertex_settings(
    *,
    vertexai: bool,
    project: str,
    location: str,
) -> tuple[bool, str, str]:
    use_vertexai = vertexai or _truthy_env("ANTHROPIC_USE_VERTEXAI")
    resolved_project = (
        project
        or os.getenv("GOOGLE_CLOUD_PROJECT")
        or os.getenv("GCLOUD_PROJECT")
    )
    resolved_location = (
        location
        or os.getenv("GOOGLE_CLOUD_LOCATION")
        or os.getenv("VERTEX_AI_LOCATION")
    )
    return use_vertexai, resolved_project, resolved_location


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
    return base64.b64encode(Path(image_path).read_bytes()).decode("utf-8")


def _build_request(
    messages: List[Dict[str, Any]],
) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    anthropic_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        images: List[str] = msg.get("images", [])

        if role == "system":
            if content:
                system_parts.append(content)
            continue

        content_blocks: List[Dict[str, Any]] = []
        for image_path in images:
            content_blocks.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": _mime_type_for_path(image_path),
                        "data": _encode_image(image_path),
                    },
                }
            )

        if content:
            content_blocks.append({"type": "text", "text": content})

        if content_blocks:
            anthropic_messages.append({"role": role, "content": content_blocks})

    return "\n\n".join(system_parts), anthropic_messages


def _response_text(response: Any) -> str:
    content_blocks = getattr(response, "content", []) or []
    text_parts = [
        getattr(block, "text", "")
        for block in content_blocks
        if getattr(block, "type", None) == "text" and getattr(block, "text", "")
    ]
    if not text_parts:
        raise ValueError("Anthropic response did not contain any text content blocks.")
    return "".join(text_parts)


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: str = "",
    vertexai: bool = False,
    project: str = "",
    location: str = "",
    max_output_tokens: Optional[int] = None,
    max_retries: Optional[int] = None,
    temperature: float = 0.2,
    output_schema: Any,
    stream_thinking: bool = False,
    timeout_sec: int = 180,
) -> Tuple[str, Optional[str]]:
    """
    Send messages to the Anthropic Messages API and return (content, None).

    ``stream_thinking`` is accepted for interface compatibility but ignored.
    """
    import anthropic  # lazy import — optional dependency

    use_vertexai, resolved_project, resolved_location = _resolve_vertex_settings(
        vertexai=vertexai,
        project=project,
        location=location,
    )

    if max_output_tokens is None:
        resolved_max_output_tokens = _MAX_OUTPUT_TOKENS
    else:
        resolved_max_output_tokens = int(max_output_tokens)
        if resolved_max_output_tokens <= 0:
            raise ValueError(
                f"max_output_tokens must be positive when set, got {max_output_tokens}."
            )

    resolved_max_retries = 2 if max_retries is None else int(max_retries)
    if resolved_max_retries < 0:
        raise ValueError(f"max_retries must be non-negative, got {max_retries}.")

    if use_vertexai:
        if not resolved_project:
            raise ValueError(
                "Claude on Vertex AI requires a Google Cloud project. "
                "Set anthropic.project, GOOGLE_CLOUD_PROJECT, or pass project=..."
            )
        if not resolved_location:
            raise ValueError(
                "Claude on Vertex AI requires a location. "
                "Set anthropic.location, GOOGLE_CLOUD_LOCATION, or pass location=..."
            )
        client = anthropic.AnthropicVertex(
            region=resolved_location,
            project_id=resolved_project,
            timeout=timeout_sec,
            max_retries=resolved_max_retries,
        )
    else:
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "Anthropic API key not found. "
                "Set ANTHROPIC_API_KEY or pass anthropic.api_key in config."
            )

        client = anthropic.Anthropic(
            api_key=resolved_key,
            timeout=timeout_sec,
            max_retries=resolved_max_retries,
        )
    system_prompt, anthropic_messages = _build_request(messages)

    response = client.messages.create(
        model=model,
        max_tokens=resolved_max_output_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=anthropic_messages,
        output_config={
            "format": {
                "type": "json_schema",
                "schema": output_schema,
            }
        },
    )

    return _response_text(response), None
