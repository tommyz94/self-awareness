# Experiment naming and compatibility

## Public names

Use `experiment=paper/e1` through `paper/e5` for the five manuscript conditions.
The implementation mapping is centralized in `experiments/naming.py`.

| Paper | Implementation | Scene/task |
| --- | --- | --- |
| E1 | e1 | Mirror baseline, cube selection |
| E2 | e2 | No mirror, cube selection |
| E3 | e3 | Misleading color cue, cube selection |
| E4 | e5 | Moving distractor robots, cube selection |
| E5 | e7 | Distractors and clutter, open-ended exploration |

These presets select the existing scene builders and task prompts. No color
sampling, rendering logic, action semantics, or scoring formula is changed.

## Stable data identities

Bare numeric IDs keep the implementation installed in that revision; `e4`
is never paper E4. The committed historical e4 is the corridor experiment;
local research also has a front-identity replacement under that number.
This version-dependent ambiguity is why new paper runs use explicit namespaces.
Bare `e5` and `e7` retain their historical meanings.
Never globally replace these strings in JSON files, manifests, or notebooks.

New paper outputs retain `experiment_id` and directory names as implementation
IDs, and add `paper_experiment_id`. The run context also records the public
preset and implementation. This keeps historical analysis scripts compatible.
Data that predate this addition are interpreted using the table above.

## Extras

Shared named presets preserve direct perception (e0), exploration without
cubes (e6), clothed embodiment (e8), and the original mirror-versus-long
corridor as `extras/corridor`. Local front-identity (e4) and mirror-side-choice
(e4_1) prototypes remain outside this commit, pending a separate review.
Their local presets must not be assumed to exist in a fresh code checkout.

Cube/exploration scenes use W/S and 30-degree A/D turns. Corridor tasks use W
and forward-diagonal A/D with fixed yaw; backward movement is disabled.
Front identity asks for a yes/no decision. The restored corridor task instead
scores stopping on the terminal row at the expected step.

Some implementation helpers and saved fields retain names such as
`e7_random_cube_heights` or `e4_variant`. They are compatibility names,
not manuscript condition labels.
