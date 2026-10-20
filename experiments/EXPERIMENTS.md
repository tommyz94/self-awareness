# `experiments/`

Experiment definitions and evaluation tooling.

`naming.py` maps manuscript-facing `paper/e1` through `paper/e5` and named
research extras to stable implementation IDs. Existing data are not renamed.
See `docs/EXPERIMENTS.md` for the mapping.

The canonical layout is now split by responsibility:

- `specs/` defines experiment plugins, scene templates, and run-setup wiring
- `tasks/` defines reusable task behavior plus task capability metadata
- `evaluation/` contains metrics, judge logic, and judge reliability tooling

## Canonical subfolders

### `specs/`

Auto-discovered experiment plugin packages such as `e1` through `e7`.

Each experiment package exports a canonical `EXPERIMENT` object from
`experiment.py`. The runtime resolves `experiment_id` through this plugin
layer instead of a controller-level branch table.

Mirror-enabled two-room experiments currently share the same visual reflection
model: the main humanoid and reflectable cubes render as aperture-gated visual
clones, and `e5`/`e6`/`e7` extend that model with mirrored distractor humanoids.

### `tasks/`

Task-specific behavior for cube selection, corridor navigation, and
distractor-room exploration tasks.

Tasks also carry capability metadata used by the runtime and evaluation layer,
for example:

- whether the task supports heatmaps
- whether the task supports the judge pipeline
- whether experiment-level summary metrics exist
- how to extract task-specific run output from step records

### `evaluation/`

Offline analysis and evaluation helpers:

- behavioral metrics
- LLM-as-judge scoring
- judge prompt variants and metric definitions
- judge reliability analysis

## Imports

Only the canonical package layout is supported:

- `experiments/specs/`
- `experiments/tasks/`
- `experiments/evaluation/`
