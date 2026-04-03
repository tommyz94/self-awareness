# `conf/experiment/`

Hydra experiment preset overrides.

## Files

### `e1.yaml`

Mirror-room cube task with the mirror enabled and 30° `a`/`d` turns. Cube and
agent colors are sampled from the shared 15-color pool.

### `e2.yaml`

Mirror-room cube task with the mirror disabled and 30° `a`/`d` turns. Cube and
agent colors are sampled from the shared 15-color pool.

### `e3.yaml`

Mirror-room cube task with the mirror enabled plus misleading color text in the
prompt, with 30° `a`/`d` turns. Cube and agent colors are sampled from the
shared 15-color pool.

### `e4.yaml`

Checkerboard corridor preset with the corridor task. Each run samples a latent
variant uniformly: `mirror` mode (stop at step `10`) or `long` mode (stop at
step `20`, with the second active agent moving on the opposite side). Actions
are `w/a/d/done`, where `a/d` are forward-diagonal moves with fixed yaw.
Grid semantics: `10x3` rows/lanes in mirror mode, `20x3` in long mode (`10+10`
across the mirror plane). On terminal row, `done` is the only successful
decision; `w/a/d` ends the episode as failure.

### `e5.yaml`

Mirrored cube-task preset with colored humanoid distractor robots. `e5` keeps a
wider centered mirror opening than the baseline room and places the
main agent plus distractors on two parallel rows, while cube slots are sampled
from room-corner positions. Distractor count is sampled per run in `[1, 6]`,
with random per-distractor colors (duplicates allowed). `e5` also uses 30°
`a`/`d` turns, and cube/agent colors come from the shared 15-color pool.

### `e6.yaml`

Mirrored distractor-room preset derived from `e5`, but without cubes in the run
scene. It keeps the same widened mirror opening, random distractor count per
run in `[1, 6]`, random per-distractor colors (duplicates allowed), and 30°
`a`/`d` turns. `e6` uses the `explore_task`, defaults to `nav: auto`, and runs
100 automatic ego-agent actions per episode (with 0.5s pause between actions)
before advancing to the next episode. `e6` also exposes `save_dataset` (default
`false`) to export `Mirror_Video_Dataset/` with per-episode RGB frames and
metadata JSON.

### `e7.yaml`

Mirrored distractor-room preset derived from `e6`, with additional random
distractor cubes. It keeps random distractor count per run in `[1, 6]` and
adds `5..15` random cubes (pool colors, duplicates allowed) distributed across
both room sides. By default, `e7_random_cube_heights: true` makes each sampled
cube a same-color vertical stack with random height in `[1, 5]`; set
`e7_random_cube_heights=false` to disable stacking. `e7` uses `explore_task`, defaults to `nav: auto`, runs
100 automatic ego-agent actions per episode, and supports
`save_dataset` (default `false`) with the same schema used in `e6`.

## Notes

- These files are thin preset selectors, not full standalone configs.
- They are loaded with `experiment=<id>`.
- The real experiment logic lives under `experiments/specs/<id>/`.
- `random_wall_color=true` enables one random shared wall color per run for `e1/e2/e3/e5/e6/e7` (default stays white/off).
