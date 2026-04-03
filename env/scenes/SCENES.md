# `env/scenes/`

Scene-building helpers shared by the MuJoCo backend.

## Files

### `mujoco_builder.py`

Builds MuJoCo XML fragments for room/corridor layouts, mirror surfaces,
humanoid insertion, and scene objects.

## Notes

- `scenes/` owns layout construction.
- `backends/` owns runtime stepping, capture, and backend-specific state.
