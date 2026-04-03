# `llm/navigation/`

Canonical navigation-VLM stack.

## Files

### `service.py`

Builds the navigation request for one step, calls the dispatcher, and returns
the parsed model action.

### `schema.py`

Defines the structured output schema expected from the navigation model.

### `parser.py`

Normalizes and parses model replies into validated navigation actions and
task-specific fields.

## Notes

- This folder is canonical.
- `llm/navigation_vlm.py`, `llm/schemas.py`, and `llm/vlm_parser.py` are
  compatibility wrappers around this functionality.
