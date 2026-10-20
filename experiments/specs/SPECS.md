# `experiments/specs/`

Canonical home for experiment plugin definitions and shared experiment-spec
types. Folder IDs are historical implementation names. Public paper presets
are `paper/e1` through `paper/e5`; the mapping is in `experiments/naming.py`
and `docs/EXPERIMENTS.md`.

## Canonical files

### `types.py`

Shared low-level dataclasses used by the experiment system:

- `SceneObjectSpec`
- `SceneSpec`
- `RunSetup`
- `TaskCapabilities`
- `TaskSpec`
- `ExperimentSpec`
- backend override/result types

Plugin packages import these types directly to avoid discovery/import cycles.

### `loader.py`

Lazy plugin discovery for `experiments/specs/<id>/experiment.py`.

Discovery only imports `experiment.py` modules and caches discovered
`EXPERIMENT` objects on first lookup.

### `registry.py`

Compatibility bridge that re-exports the shared types and delegates
`get_experiment(...)` to the loader.

## Experiment packages

Each experiment lives in its own subfolder, for example:

- `e0/`
- `e1/`
- `e2/`
- `e3/`
- `e4/`
- `e5/`
- `e6/`
- `e7/`
- `e8/`

The only required file is `experiment.py`, which must export exactly one
public symbol:

- `EXPERIMENT: ExperimentSpec`

Optional per-experiment helpers such as `scene.py`, `task.py`, and
`mujoco.py` can be added for larger experiments.

## Notes

- `conf/experiment/<id>.yaml` remains the thin Hydra selector for each
  experiment id.
- Runtime options such as `distractors` and `random_wall_color` are resolved by the experiment plugin
  before the controller creates the backend, so experiments can keep geometry
  static for the life of one controller run.
- Reflectable scene objects are currently expressed through
  `SceneObjectSpec.metadata["reflect_in_mirror"] = True`, which the backends
  use to decide whether a visual mirror clone should be rendered.
- `e5`, `e6`, `e7`, and `e8` are current examples of experiment-specific scene
  variants with mirrored humanoid distractors.
- New experiments should be added here instead of editing the runtime
  controller or maintaining a hand-written central registry.
