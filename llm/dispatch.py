"""
Multi-provider chat dispatcher for the navigation VLM.

Routes calls to the appropriate backend based on ``client_type``:
  - ``ollama``  — local Ollama server (default)
  - ``openai``  — OpenAI Responses API  (requires openai>=1.0.0)
  - ``openrouter`` — OpenRouter Chat Completions API (requires openai>=1.0.0)
  - ``google``  — Google Gemini API or Vertex AI (requires google-genai>=1.0.0)
  - ``anthropic`` — Anthropic Claude API or Claude on Vertex AI
    (requires anthropic>=0.68.0, or anthropic[vertex] for Vertex AI)

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

_CLIENT_TYPE_ALIASES = {"claude": "anthropic", "huggingface": "hf"}
_CANONICAL_CLIENTS = {"ollama", "openai", "openrouter", "google", "anthropic", "hf"}
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
    base_url: Optional[str] = None,
    http_referer: Optional[str] = None,
    app_title: Optional[str] = None,
    require_parameters: Optional[bool] = None,
    allow_fallbacks: Optional[bool] = None,
    provider_order: Optional[Any] = None,
    provider_only: Optional[Any] = None,
    provider_ignore: Optional[Any] = None,
    min_request_interval_sec: Optional[float] = None,
    vertexai: Optional[bool] = None,
    project: Optional[str] = None,
    location: Optional[str] = None,
    hf_device: str = "auto",
    hf_dtype: str = "auto",
    max_output_tokens: Optional[int] = None,
    structured_string_max_length: Optional[int] = None,
    max_retries: Optional[int] = None,
    initial_retry_delay_sec: Optional[float] = None,
    max_retry_delay_sec: Optional[float] = None,
    temperature: float = 0.2,
    timeout_sec: int = 600,
) -> Tuple[str, Optional[str]]:
    """
    Dispatch a chat request to the configured provider and return
    ``(content_json_string, thinking_or_None)``.

    Parameters
    ----------
    client_type:
        One of ``"ollama"``, ``"openai"``, ``"openrouter"``,
        ``"google"``, or ``"anthropic"``.
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
        Silently ignored for OpenAI, Google, and Anthropic.
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

    if client_type == "openrouter":
        from providers.openrouter import chat as openrouter_chat

        return openrouter_chat(
            model=model,
            messages=messages,
            api_key=api_key or "",
            base_url=base_url or "https://openrouter.ai/api/v1",
            http_referer=http_referer or "",
            app_title=app_title or "",
            require_parameters=True if require_parameters is None else require_parameters,
            allow_fallbacks=allow_fallbacks,
            provider_order=provider_order,
            provider_only=provider_only,
            provider_ignore=provider_ignore,
            min_request_interval_sec=(
                0.0 if min_request_interval_sec is None else float(min_request_interval_sec)
            ),
            temperature=temperature,
            output_schema=output_schema,
            max_output_tokens=max_output_tokens,
            structured_string_max_length=structured_string_max_length,
            stream_thinking=stream_thinking,
            timeout_sec=timeout_sec,
        )

    if client_type == "google":
        from providers.google import chat as google_chat

        return google_chat(
            model=model,
            messages=messages,
            api_key=api_key or "",
            vertexai=bool(vertexai),
            project=project or "",
            location=location or "",
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            initial_retry_delay_sec=initial_retry_delay_sec,
            max_retry_delay_sec=max_retry_delay_sec,
            temperature=temperature,
            output_schema=output_schema,
            stream_thinking=stream_thinking,
            timeout_sec=timeout_sec,
        )

    if client_type == "hf":
        from providers.hf import chat as hf_chat

        return hf_chat(
            device=hf_device,
            dtype=hf_dtype,
            model=model,
            messages=messages,
            output_schema=output_schema,
            stream_thinking=stream_thinking,
            max_output_tokens=max_output_tokens,
            temperature=temperature,
            timeout_sec=timeout_sec,
        )

    from providers.anthropic import chat as anthropic_chat

    return anthropic_chat(
        model=model,
        messages=messages,
        api_key=api_key or "",
        vertexai=bool(vertexai),
        project=project or "",
        location=location or "",
        max_output_tokens=max_output_tokens,
        max_retries=(
            None if max_retries is None else int(max_retries)
        ),
        temperature=temperature,
        output_schema=output_schema,
        stream_thinking=stream_thinking,
        timeout_sec=timeout_sec,
    )
