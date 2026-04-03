# `env/`

Environment abstraction layer for the simulator backend.

The canonical layout is split between backend wrappers and scene builders:

- `base.py` defines the backend contract used by the runtime controller
- `backends/` contains the concrete MuJoCo environment class
- `scenes/` contains MuJoCo scene-building helpers

## Canonical modules

### `base.py`

Defines `BaseEnv`, the shared interface expected by `runtime/controller.py`.

It covers:

- reset/setup for a new run
- movement and orientation updates
- observation capture
- optional human input binding
- frame/tick lifecycle methods

### `backends/`

Concrete backend implementations:

- `mujoco.py` for headless or windowed MuJoCo rendering
- `rest_pose.py` for shared humanoid arm-rest pose definitions

### `scenes/`

Scene-construction helpers owned by the environment layer:

- `mujoco_builder.py` builds MuJoCo XML fragments and scene bodies

For mirror-enabled two-room scenes, reflectable objects (for example cubes)
render as aperture-gated visual mirror clones.

For mirror-enabled corridor scenes, the backend builds a static symmetric
second corridor beyond the mirror. That right-side corridor is visual-only;
navigation and collision remain confined to the playable side.

## Imports

Only the canonical module layout is supported:

- `env/base.py`
- `env/backends/`
- `env/scenes/`
