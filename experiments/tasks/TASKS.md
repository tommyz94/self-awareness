# `experiments/tasks/`

Task-specific navigation behavior layered on top of the scene/backends.

## Canonical files

### `cube.py`

Cube-selection task used by the original mirror experiments.

It owns:

- cube-task prompt wording
- cube-task action semantics (`a`/`d` turn by 30° in `e1`/`e2`/`e3`/`e5`)
- cube-task structured output schema
- cube-task reply parsing/finalization
- cube-task capability metadata (heatmap, judge, summary metrics, task output)

### `corridor.py`

Corridor navigation task used by `e4`.

It owns:

- the corridor navigation instruction and stop rules
- the `e4` action wording where `a`/`d` move forward-diagonal without rotating
- the `e4` stop semantics (`done@10` in mirror mode, `done@20` in long mode)
- the corridor task schema
- any corridor-specific parsing/finalization behavior
- corridor-task capability metadata

### `explore.py`

Room-exploration task used by `e6`.

It owns:

- the distractor-room exploration instruction
- 30° turn wording for `a`/`d`
- non-cube response schema and parsing behavior
- explore-task capability metadata

## Notes

- Shared tasks are the default.
- Experiment plugins can still provide a local task factory override when a new
  experiment needs custom task behavior that should not live in the shared task
  registry.

## Compatibility wrappers

The old flat task module names remain as thin wrappers:

- `cube_task.py`
- `corridor_task.py`
- `explore_task.py`
