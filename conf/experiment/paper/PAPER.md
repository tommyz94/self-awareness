# `conf/experiment/paper/`

Manuscript-facing presets selected with `experiment=paper/e1` through
`paper/e5`. Each seed runs seven episodes, with a 100-step budget, first-person
RGB, one-step history, and the default humanoid. Provide a model with `nav=...`.

E1/E2/E3 use implementations e1/e2/e3; paper E4 uses e5; paper E5 uses e7.
The latter is an exploration task, not cube selection. These presets do not
inherit the automatic-navigation default of legacy e7.

JSON and directory implementation IDs remain stable. New JSON files also carry
`paper_experiment_id`; original files are never renamed or rewritten.
See `docs/EXPERIMENTS.md` for the mapping and `docs/REPRODUCIBILITY.md` for counts.
