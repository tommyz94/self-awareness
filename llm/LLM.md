# `llm/`

Canonical package for navigation-VLM inference.

This package owns:

- provider dispatch
- navigation prompt/message assembly
- structured output schemas
- JSON parsing and action normalization

## Canonical modules

### `dispatch.py`

Routes `chat(...)` calls to the selected provider in `providers/`.

### `navigation/`

Canonical navigation stack:

- `service.py` builds messages and runs one navigation step
- `schema.py` defines the structured output schema
- `parser.py` normalizes and parses model replies

## Notes

- All new code must import from `llm/dispatch.py` and `llm/navigation/`.
- `dispatch.py` routes directly to `providers/`.
