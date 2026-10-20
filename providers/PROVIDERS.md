# `providers/`

Canonical provider adapters for local and cloud model backends.

Each provider module exposes the same `chat(...)`-style interface so the
navigation layer can switch providers from config.

## Files

### `ollama.py`

Local Ollama HTTP adapter, including model listing support.

### `openai.py`

OpenAI Responses API adapter for image + prompt navigation calls.

### `openrouter.py`

OpenRouter chat-completions adapter for OpenAI-compatible model routing.

### `google.py`

Gemini API adapter for multimodal navigation calls.

### `anthropic.py`

Anthropic Messages API adapter for Claude multimodal navigation calls.

## Notes

- `providers/` is canonical.
- `providers/` is the only supported provider adapter namespace.

## Paper runs and local research

Use `client=ollama|openai|openrouter|google|anthropic` and an exact model ID
in `nav`. Keep tracked API-key fields empty; set credentials in the environment.
The OpenRouter-compatible adapter can use an explicitly configured
`openrouter.base_url`, so its client name does not identify the inference host.
Preserve full model IDs and record the server separately for local models.

Additional HF and Vertex adapters in the local research checkout are outside
this alignment commit and require separate review and optional dependencies.
