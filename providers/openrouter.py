"""
OpenRouter provider adapter for the navigation VLM.

Uses the OpenAI-compatible Chat Completions API via the ``openai`` Python SDK.
Structured output is requested through ``response_format={"type": "json_schema", ...}``.

API key precedence (highest first):
  1. OPENROUTER_API_KEY environment variable
  2. ``api_key`` argument passed by the caller
"""

from __future__ import annotations

import base64
import json
import os
import tempfile
import time
from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


def _bound_unconstrained_strings(schema: Any, max_length: Optional[int]) -> Any:
    if max_length is None:
        return schema
    bounded = deepcopy(schema)

    def _visit(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "string" and "enum" not in node:
                node.setdefault("maxLength", int(max_length))
            for value in node.values():
                _visit(value)
        elif isinstance(node, list):
            for value in node:
                _visit(value)

    _visit(bounded)
    return bounded
_RATE_LIMIT_LOCK_PATH = Path(tempfile.gettempdir()) / "self_awareness_openrouter_rate_limit.lock"
_RATE_LIMIT_STATE_PATH = Path(tempfile.gettempdir()) / "self_awareness_openrouter_rate_limit.json"


def _resolve_api_key(api_key: str) -> str:
    return os.getenv("OPENROUTER_API_KEY") or api_key


def _normalize_provider_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [part.strip() for part in stripped.split(",") if part.strip()]

    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        try:
            items = list(value)
        except TypeError:
            text = str(value).strip()
            return [text] if text else []

    return [str(item).strip() for item in items if str(item).strip()]


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


def _enforce_global_min_interval(min_interval_sec: float) -> None:
    if min_interval_sec <= 0:
        return

    import fcntl

    _RATE_LIMIT_LOCK_PATH.touch(exist_ok=True)
    with _RATE_LIMIT_LOCK_PATH.open("r+", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            last_request_ts = 0.0
            try:
                state = json.loads(_RATE_LIMIT_STATE_PATH.read_text())
                last_request_ts = float(state.get("last_request_ts", 0.0))
            except (FileNotFoundError, json.JSONDecodeError, TypeError, ValueError):
                last_request_ts = 0.0

            now = time.time()
            wait_sec = max(0.0, last_request_ts + min_interval_sec - now)
            if wait_sec > 0:
                time.sleep(wait_sec)

            _RATE_LIMIT_STATE_PATH.write_text(
                json.dumps({"last_request_ts": time.time()}),
                encoding="utf-8",
            )
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _build_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Convert internal Ollama-format messages to OpenRouter chat-completions
    messages. Text is emitted before images to match OpenRouter guidance.
    """
    openrouter_messages: List[Dict[str, Any]] = []

    for msg in messages:
        role = str(msg.get("role", "user"))
        content = str(msg.get("content", ""))
        images: List[str] = msg.get("images", [])

        if not images:
            if content:
                openrouter_messages.append({"role": role, "content": content})
            continue

        content_blocks: List[Dict[str, Any]] = []
        if content:
            content_blocks.append({"type": "text", "text": content})

        for image_path in images:
            content_blocks.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": (
                            f"data:{_mime_type_for_path(image_path)};base64,"
                            f"{_encode_image(image_path)}"
                        )
                    },
                }
            )

        if content_blocks:
            openrouter_messages.append({"role": role, "content": content_blocks})

    return openrouter_messages


def _response_text(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise ValueError("OpenRouter response did not contain any choices.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise ValueError("OpenRouter response did not contain a message payload.")

    content = getattr(message, "content", None)
    if isinstance(content, str) and content:
        return content

    if isinstance(content, list):
        text_parts: List[str] = []
        for part in content:
            part_type = getattr(part, "type", None)
            if isinstance(part, dict):
                part_type = part.get("type")
                if part_type == "text" and part.get("text"):
                    text_parts.append(part["text"])
            elif part_type == "text" and getattr(part, "text", ""):
                text_parts.append(part.text)
        if text_parts:
            return "".join(text_parts)

    refusal = getattr(message, "refusal", None)
    if refusal:
        raise ValueError(f"OpenRouter model refused the request: {refusal}")

    raise ValueError("OpenRouter response did not contain any text content.")


def chat(
    model: str,
    messages: List[Dict[str, Any]],
    *,
    api_key: str = "",
    temperature: float = 0.2,
    output_schema: Any,
    stream_thinking: bool = False,
    timeout_sec: int = 180,
    base_url: str = _DEFAULT_BASE_URL,
    http_referer: str = "",
    app_title: str = "",
    require_parameters: bool = True,
    allow_fallbacks: Optional[bool] = None,
    provider_order: Any = None,
    provider_only: Any = None,
    provider_ignore: Any = None,
    min_request_interval_sec: float = 0.0,
    max_output_tokens: Optional[int] = None,
    structured_string_max_length: Optional[int] = None,
) -> Tuple[str, Optional[str]]:
    """
    Send messages to OpenRouter and return (content, None).

    ``stream_thinking`` is accepted for interface compatibility but ignored.
    """
    import openai  # lazy import — optional dependency

    resolved_key = _resolve_api_key(api_key)
    if not resolved_key:
        raise ValueError(
            "OpenRouter API key not found. "
            "Set OPENROUTER_API_KEY or pass openrouter.api_key in config."
        )

    client = openai.OpenAI(
        api_key=resolved_key,
        base_url=base_url or _DEFAULT_BASE_URL,
        timeout=timeout_sec,
    )

    extra_headers = {}
    if http_referer:
        extra_headers["HTTP-Referer"] = http_referer
    if app_title:
        extra_headers["X-OpenRouter-Title"] = app_title

    provider_config: Dict[str, Any] = {}
    if require_parameters:
        provider_config["require_parameters"] = True
    if allow_fallbacks is not None:
        provider_config["allow_fallbacks"] = bool(allow_fallbacks)

    normalized_order = _normalize_provider_list(provider_order)
    normalized_only = _normalize_provider_list(provider_only)
    normalized_ignore = _normalize_provider_list(provider_ignore)
    if normalized_order:
        provider_config["order"] = normalized_order
    if normalized_only:
        provider_config["only"] = normalized_only
    if normalized_ignore:
        provider_config["ignore"] = normalized_ignore

    extra_body = {}
    if provider_config:
        extra_body["provider"] = provider_config

    _enforce_global_min_interval(min_request_interval_sec)

    request_kwargs: Dict[str, Any] = {
        "model": model,
        "messages": _build_messages(messages),
        "temperature": temperature,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "navigation_response",
                "strict": True,
                "schema": _bound_unconstrained_strings(
                    output_schema,
                    structured_string_max_length,
                ),
            },
        },
        "extra_headers": extra_headers or None,
        "extra_body": extra_body or None,
    }
    if max_output_tokens is not None:
        request_kwargs["max_tokens"] = int(max_output_tokens)

    response = client.chat.completions.create(**request_kwargs)

    return _response_text(response), None
