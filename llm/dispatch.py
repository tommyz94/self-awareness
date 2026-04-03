"""
Multi-provider chat dispatcher for the navigation VLM.

Routes calls to the appropriate backend based on ``client_type``:
  - ``ollama``  — local Ollama server (default)
  - ``openai``  — OpenAI Responses API  (requires openai>=1.0.0)
  - ``google``  — Google Gemini API     (requires google-genai>=1.0.0)
  - ``anthropic`` — Anthropic Claude API (requires anthropic>=0.68.0)

Usage
-----
    from llm.dispatch import chat

    content, thinking = chat(
        client_type="ollama",
        model="gemma3:4b",
        messages=messages,
        output_schema=RESPONSE_SCHEMA,
        host="http://localhost:11434",
    )
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_CLIENT_TYPE_ALIASES = {"claude": "anthropic"}
_CANONICAL_CLIENTS = {"ollama", "openai", "google", "anthropic"}
_VALID_CLIENTS = _CANONICAL_CLIENTS | set(_CLIENT_TYPE_ALIASES)


def canonicalize_client_type(client_type: str) -> str:
    normalized = client_type.strip().lower()
    return _CLIENT_TYPE_ALIASES.get(normalized, normalized)


def chat(
    *,
    client_type: str,
    model: str,
    messages: List[Dict[str, Any]],
    output_schema: Any,
    stream_thinking: bool = False,
    host: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.2,
    timeout_sec: int = 180,
) -> Tuple[str, Optional[str]]:
    """
    Dispatch a chat request to the configured provider and return
    ``(content_json_string, thinking_or_None)``.

    Parameters
    ----------
    client_type:
        One of ``"ollama"``, ``"openai"``, ``"google"``, or ``"anthropic"``.
        ``"claude"`` is accepted as an alias for ``"anthropic"``.
    model:
        Model name understood by the chosen provider.
    messages:
        Internal Ollama-format message list:
        ``[{"role": "system"|"user", "content": "...", "images": ["path.jpg"]}]``
    output_schema:
        JSON Schema dict describing the expected response structure.
    stream_thinking:
        Ollama only — stream pre-output reasoning tokens to stdout.
        Silently ignored for OpenAI and Google.
    host:
        Ollama only — server URL (default ``http://localhost:11434``).
    api_key:
        Config-level API key fallback.  Each provider first checks its
        canonical environment variable before using this value.
    temperature:
        Sampling temperature forwarded to the provider.
    timeout_sec:
        Request timeout in seconds.
    """
    client_type = canonicalize_client_type(client_type)
    if client_type not in _CANONICAL_CLIENTS:
        raise ValueError(
            f"Unknown client_type={client_type!r}. "
            f"Valid options: {sorted(_VALID_CLIENTS)}"
        )

    if client_type == "ollama":
        from providers.ollama import chat as ollama_chat

        return ollama_chat(
            host=host or "http://localhost:11434",
            model=model,
            messages=messages,
            temperature=temperature,
            timeout_sec=timeout_sec,
            output_format=output_schema,
            stream_thinking=stream_thinking,
        )

    if client_type == "openai":
        from providers.openai import chat as openai_chat

        return openai_chat(
            model=model,
            messages=messages,
            api_key=api_key or "",
            temperature=temperature,
            output_schema=output_schema,
            stream_thinking=stream_thinking,
            timeout_sec=timeout_sec,
        )

    if client_type == "google":
        from providers.google import chat as google_chat

        return google_chat(
            model=model,
            messages=messages,
            api_key=api_key or "",
            temperature=temperature,
            output_schema=output_schema,
            stream_thinking=stream_thinking,
            timeout_sec=timeout_sec,
        )

    from providers.anthropic import chat as anthropic_chat

    return anthropic_chat(
        model=model,
        messages=messages,
        api_key=api_key or "",
        temperature=temperature,
        output_schema=output_schema,
        stream_thinking=stream_thinking,
        timeout_sec=timeout_sec,
    )
