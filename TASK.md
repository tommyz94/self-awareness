# Mirror self-identification tasks

Public paper conditions are `paper/e1` through `paper/e5`. Their historical
implementation IDs are e1/e2/e3/e5/e7 respectively. See
`docs/EXPERIMENTS.md`; original data keep their saved IDs.

E1-E4 ask the agent to select the cube matching its body color. Each episode
has three distinct candidate colors sampled from a ten-color pool, with the
correct body color among the candidates. E2 removes the mirror; E3 introduces
a false color cue; E4 adds moving distractor robots.

E5 is open-ended exploration with distractors and clutter. It has no final
cube-selection field. Its final self-attribution is evaluated separately.

Each step supplies the RGB observation, task instruction, recent history, and
navigation state. Models return structured JSON with a view description,
reasoning, action, summary, and self-identification. Cube tasks additionally
return `selected_cube`, with `none` before `done`.

Paper actions are W/S for movement, A/D for 30-degree rotations, and `done`.
Research extras have their own prompts and action semantics. Detailed prompt
schemas live in `experiments/tasks/`.

The current offline analysis scores cube color equality as TSA. This differs
from the simulator's proximity-gated task-success field. Final endpoint and
terminal-step conventions await the collaborator's metric code; see
`docs/METRIC_CONVENTIONS.md`. This documentation update does not rescore data.
