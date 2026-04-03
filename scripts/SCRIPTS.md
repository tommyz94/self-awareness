# `scripts/`

Shell entrypoints for common simulator workflows.

The canonical layout is split by backend or workflow:

- `mujoco/` for MuJoCo run scripts
- `pipelines/` for multi-step end-to-end workflows
- `dev/` for local developer helpers

## Top-level wrapper scripts

The old flat script names remain as convenience wrappers:

- `mujoco.sh`
- `mujoco_headless.sh`
- `mujoco_eval.sh`
- `pipeline.sh`

New references should point to the canonical subfolders.
