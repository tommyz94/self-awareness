# `scripts/dev/`

Developer-local shell helpers.

## Files

### `keys.sh`

Convenience shell snippet for exporting local API keys or developer-specific
environment variables.

### `openrouter_matrix.py`

JSON-driven batch launcher for OpenRouter experiment matrices. It can preview
the full model x experiment grid or open one macOS Terminal window per job.
Supports repeated `--set key=value` overrides for one-off Hydra params, while
refusing keys locked by experiment presets.

### `openrouter_smoke.py`

Minimal OpenRouter connectivity smoke test. It sends one structured-output
request through the real OpenRouter adapter and, by default, includes a tiny
generated PNG so you can verify the multimodal request path before starting a
full matrix. It can also test multiple model ids in sequence, which is useful
for checking whether `:nitro` routes actually work for this repo's strict
multimodal JSON-schema request shape.

### `openrouter_matrix.example.json`

Example matrix config for `openrouter_matrix.py`.

### `openrouter_matrix.requested.json`

Full requested OpenRouter sweep config using the user's selected model list
across `e1/e2/e3/e4/e5/e7`, with `seed`, `print_reasoning`, and a shared
OpenRouter request throttle preconfigured.

### `openrouter_matrix.smoke.json`

Small 2x2 Terminal-launch smoke config for the matrix launcher. It uses two
OpenRouter multimodal models across `e1` and `e2`, with `max_steps: 1`, so
you can verify Terminal spawning and log routing before a
full batch.

### `openrouter_matrix.smoke.safe.json`

Safer 1x2 smoke config for the matrix launcher. It uses
`google/gemma-4-26b-a4b-it:nitro` across `e1` and `e2` with `max_steps: 1`,
which is useful when you want a minimal Terminal-launch check using a model
that OpenRouter explicitly advertises as multimodal with structured output
support.

### `run_openrouter_matrix_requested.sh`

Convenience wrapper that launches `openrouter_matrix.requested.json` directly.
Run it as-is to open Terminal windows, or pass `--dry-run` to print commands
without launching.

## Notes

- Treat this folder as developer-local workflow support.
- Do not store real secrets in committed helper files.
