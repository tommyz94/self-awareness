# `providers/`

Canonical provider adapters for local and cloud model backends.

Each provider module exposes the same `chat(...)`-style interface so the
navigation layer can switch providers from config.

## Files

### `ollama.py`

Local Ollama HTTP adapter, including model listing support.

### `openai.py`

OpenAI Responses API adapter for image + prompt navigation calls.

### `google.py`

Gemini API adapter for multimodal navigation calls.

### `anthropic.py`

Anthropic Messages API adapter for Claude multimodal navigation calls.

## Notes

- `providers/` is canonical.
- `providers/` is the only supported provider adapter namespace.
