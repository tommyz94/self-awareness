# `scripts/mujoco/`

Canonical MuJoCo run scripts.

## Files

### `run.sh`

Standard MuJoCo launcher using the canonical script layout.

### `headless.sh`

MuJoCo launcher specialized for offscreen/headless runs.

### `eval.sh`

MuJoCo experiment/evaluation helper script.

It now defaults to the current output layout under
`outputs/mujoco/<client>/<nav_slug>/runs/<experiment>/...` and calls the
canonical judge entrypoint `python -m experiments.evaluation.judge`.

## Notes

- These scripts are the canonical MuJoCo entrypoints.
- The old top-level `scripts/mujoco*.sh` files are wrappers.
