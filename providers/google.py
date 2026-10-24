"""
Google Gemini provider adapter for the navigation VLM.

Supports both:
  - the standalone Gemini API via ``GEMINI_API_KEY`` / ``GOOGLE_API_KEY``
  - Gemini on Vertex AI via ADC or other Google Cloud auth configured for the
    ``google-genai`` SDK

Structured output is requested via ``response_mime_type="application/json"``
and ``response_schema``. A ``_add_property_ordering`` helper recursively
injects ``propertyOrdering`` so the schema works correctly with Gemini 2.0
models.
"""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_VERTEX_ENV_KEYS_TRUE = {"1", "true", "yes", "on"}
_LOW_THINKING_BUDGET_MODELS = ("gemini-2.5-pro",)
_RETRYABLE_STATUS_CODES = {429, 504}
_MAX_GENERATE_RETRIES = 5
_INITIAL_RETRY_DELAY_SEC = 2.0
_MAX_RETRY_DELAY_SEC = 30.0


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


def _truthy_env(name: str) -> bool:
    value = os.getenv(name, "")
    return value.strip().lower() in _VERTEX_ENV_KEYS_TRUE


def _resolve_vertex_settings(
    *,
    vertexai: bool,
    project: str,
    location: str,
) -> tuple[bool, str, str]:
    use_vertexai = vertexai or _truthy_env("GOOGLE_GENAI_USE_VERTEXAI")
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


def _serialize_structured_response(response: Any) -> Optional[str]:
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        if hasattr(parsed, "model_dump"):
            try:
                parsed = parsed.model_dump(mode="json")
            except TypeError:
                parsed = parsed.model_dump()
        try:
            return json.dumps(parsed)
        except TypeError:
            pass

    text = getattr(response, "text", None)
    if text is None:
        return None
    return str(text)


def _thinking_budget_for_model(model: str) -> Optional[int]:
    normalized = str(model).strip().lower()
    if any(name in normalized for name in _LOW_THINKING_BUDGET_MODELS):
        return 128
    return None


def _is_retryable_google_error(exc: Exception) -> bool:
    code = getattr(exc, "code", None)
    if isinstance(code, int) and code in _RETRYABLE_STATUS_CODES:
        return True

    status = str(getattr(exc, "status", "") or "").strip().upper()
    if status in {"RESOURCE_EXHAUSTED", "DEADLINE_EXCEEDED"}:
        return True

    message = str(getattr(exc, "message", "") or exc).upper()
    return (
        "RESOURCE_EXHAUSTED" in message
        or "DEADLINE_EXCEEDED" in message
        or "TOO MANY REQUESTS" in message
        or "GATEWAY TIMEOUT" in message
        or " 429 " in f" {message} "
        or " 504 " in f" {message} "
    )


def _generate_content_with_retries(
    *,
    client: Any,
    model: str,
    contents: List[Any],
    config: Any,
    max_retries: int,
    initial_retry_delay_sec: float,
    max_retry_delay_sec: float,
) -> Any:
    last_exc: Optional[Exception] = None

    for attempt in range(1, max_retries + 1):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except Exception as exc:
            last_exc = exc
            if not _is_retryable_google_error(exc) or attempt >= max_retries:
                raise

            delay_sec = min(
                initial_retry_delay_sec * (2 ** (attempt - 1)),
                max_retry_delay_sec,
            )
            code = getattr(exc, "code", "unknown")
            status = getattr(exc, "status", "") or "retryable_error"
            print(
                f"  [Google] Attempt {attempt}/{max_retries} failed "
                f"({code} {status}). Retrying in {delay_sec:.0f}s...",
                flush=True,
            )
            time.sleep(delay_sec)

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Google provider made no generate_content attempts.")


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: str = "",
    vertexai: bool = False,
    project: str = "",
    location: str = "",
    max_output_tokens: Optional[int] = None,
    max_retries: int = _MAX_GENERATE_RETRIES,
    initial_retry_delay_sec: float = _INITIAL_RETRY_DELAY_SEC,
    max_retry_delay_sec: float = _MAX_RETRY_DELAY_SEC,
    temperature: float = 0.2,
    output_schema: Any,
    stream_thinking: bool = False,
    timeout_sec: int = 180,
) -> Tuple[str, Optional[str]]:
    """
    Send messages to Gemini and return (content, None).

    ``stream_thinking`` is accepted for interface compatibility but ignored.
    """
    import google.genai as genai  # lazy import — optional dependency
    from google.genai import types

    if int(max_retries) <= 0:
        raise ValueError(f"max_retries must be positive, got {max_retries}.")
    if float(initial_retry_delay_sec) <= 0:
        raise ValueError(
            f"initial_retry_delay_sec must be positive, got {initial_retry_delay_sec}."
        )
    if float(max_retry_delay_sec) <= 0:
        raise ValueError(
            f"max_retry_delay_sec must be positive, got {max_retry_delay_sec}."
        )

    max_retries = int(max_retries)
    initial_retry_delay_sec = float(initial_retry_delay_sec)
    max_retry_delay_sec = float(max_retry_delay_sec)

    http_options = types.HttpOptions(timeout=int(timeout_sec * 1000))

    use_vertexai, resolved_project, resolved_location = _resolve_vertex_settings(
        vertexai=vertexai,
        project=project,
        location=location,
    )

    if use_vertexai:
        if not resolved_project:
            raise ValueError(
                "Vertex AI Gemini requires a Google Cloud project. "
                "Set google.project, GOOGLE_CLOUD_PROJECT, or pass project=..."
            )
        if not resolved_location:
            raise ValueError(
                "Vertex AI Gemini requires a location. "
                "Set google.location, GOOGLE_CLOUD_LOCATION, or pass location=..."
            )
        client = genai.Client(
            vertexai=True,
            project=resolved_project,
            location=resolved_location,
            http_options=http_options,
        )
    else:
        resolved_key = _resolve_api_key(api_key)
        if not resolved_key:
            raise ValueError(
                "Google API key not found. "
                "Set GEMINI_API_KEY (or GOOGLE_API_KEY), or enable Vertex AI "
                "with google.use_vertexai / GOOGLE_GENAI_USE_VERTEXAI."
            )

        client = genai.Client(api_key=resolved_key, http_options=http_options)

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
            image_bytes = Path(image_path).read_bytes()
            mime_type = _mime_type_for_path(image_path)
            parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))

        if content:
            parts.append(types.Part.from_text(text=content))

    contents = [types.Content(role="user", parts=parts)]

    generate_config_kwargs: Dict[str, Any] = {
        "temperature": temperature,
        "response_mime_type": "application/json",
        "response_json_schema": _add_property_ordering(output_schema),
        "system_instruction": system_instruction,
    }
    thinking_budget = _thinking_budget_for_model(model)
    if thinking_budget is not None:
        generate_config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=False,
        )
    if max_output_tokens is not None:
        if int(max_output_tokens) <= 0:
            raise ValueError(
                f"max_output_tokens must be positive when set, got {max_output_tokens}."
            )
        generate_config_kwargs["max_output_tokens"] = int(max_output_tokens)

    generate_config = types.GenerateContentConfig(**generate_config_kwargs)

    response = _generate_content_with_retries(
        client=client,
        model=model,
        contents=contents,
        config=generate_config,
        max_retries=max_retries,
        initial_retry_delay_sec=initial_retry_delay_sec,
        max_retry_delay_sec=max_retry_delay_sec,
    )

    serialized = _serialize_structured_response(response)
    if serialized is None:
        raise ValueError("Google provider returned no parsable text or structured payload.")

    return serialized, None
