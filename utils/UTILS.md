# `utils/`

Shared helper modules for logging, JSON export, capture, visualization, and
external integrations.

The canonical layout is now split by responsibility:

- `io/` for exported files and runtime logging
- `viz/` for plots and visual summaries
- `capture/` for observation/image capture helpers
- `integrations/` for optional external services such as W&B

## Canonical subfolders

### `io/`

File logging and JSON export helpers used by the runtime controller and
evaluation pipeline.

### `viz/`

Visualization helpers such as trajectory heatmaps.

### `capture/`

Capture helpers placeholder (currently no shared helper module in use).

### `integrations/`

Optional third-party integrations, currently the W&B logger used by judge
evaluation.

Use the canonical subpackages under `utils/`.
