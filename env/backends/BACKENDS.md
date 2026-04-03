# `env/backends/`

Concrete environment backend used by the runtime controller.

## Files

### `mujoco.py`

MuJoCo backend for offscreen or live-window rendering.

It owns:

- robot pose updates
- `e4`-specific forward-diagonal handling for `a`/`d` (`w/a/d/done`, no backward)
- `e4` random mirror-vs-long corridor variant behavior with secondary agent support
- object resets
- RGB capture
- mirrored humanoid synchronization

### `rest_pose.py`

Shared MuJoCo resting-pose definitions.

It owns:

- the canonical MuJoCo arm-rest joint pose
