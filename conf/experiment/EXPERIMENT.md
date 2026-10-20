# `conf/experiment/`

## Paper presets

Use `experiment=paper/e1` through `paper/e5`. These IDs match the manuscript;
they select stable implementations e1, e2, e3, e5, and e7 respectively.
Each preset sets `paper_experiment_id` so exports record both vocabularies.
Paper E5 uses the exploration prompt, not cube selection.

Paper presets run seven episodes for one seed, a 100-step budget, one-step
history, first-person rendering, and the default humanoid. Select a model
explicitly with `nav=...`. Outputs go under `outputs/paper/`.

## Research extras

Named selectors under `extras/` preserve direct perception, front identity,
the original corridor task, exploration, embodiment, and mirror-side choice.
See `extras/EXTRAS.md`.

## Historical selectors

Bare e0-e8 and e4_1 retain their original implementation semantics and defaults.
In particular, bare e4 is front identity, bare e5 is the paper's distractor
condition, and bare e7 is cluttered exploration with automatic navigation by
default. Their source packages and existing output folders are not renamed.

Cube colors are sampled from the shared ten-color pool. Distractor color
policies remain `allow_agent_color`, `exclude_agent_color`, and
`allow_duplicates`. `e7_random_cube_heights` is a historical runtime-option
name retained for compatibility.

The full mapping, controls, and preservation policy are in
`docs/EXPERIMENTS.md` and `docs/REPRODUCIBILITY.md`.
