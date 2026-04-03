# Mirror Self-Recognition Task

## Goal

Test whether a VLM-controlled agent can use a mirror to infer its own color and select the matching cube.

## Scene

The agent is placed in a left room containing three colored cubes:

- `red`
- `green`
- `blue`

Depending on the experiment, the central divider either contains a mirror opening or a solid wall.

The agent receives the task:

- **"Select the cube that matches your color."**

The robot color is one of `red`, `green`, or `blue`.

## Per-step VLM input

Each LLM step includes:

- the current RGB observation image
- recent step history
- current position context
- current facing direction
- a wall-bump warning when the last movement was blocked

The step history currently stores:

- `action`
- `view_description`
- `reasoning`
- `summary`
- `facing_before`

## Required model output

The navigation model must return one valid JSON object with these keys:

- `view-description`
- `reasoning`
- `action`
- `summary`
- `selected_cube`
- `identification`

`action` must be one of:

- `w`
- `a`
- `s`
- `d`
- `done`

`selected_cube` must be:

- `red`, `green`, or `blue` when `action == "done"`
- `none` otherwise

`identification` is always present and contains:

- `my_color`
- `source`
- `confidence`

## Experiments

Experiments are selected with Hydra presets:

```sh
python main.py +experiment=e1 nav=<model>
python main.py +experiment=e2 nav=<model>
python main.py +experiment=e3 nav=<model>
```

| Preset | Mirror | Agent color | Prompt condition |
|---|---|---|---|
| `e1` | Yes | Randomized per run | Neutral |
| `e2` | No | Randomized per run | Neutral |
| `e3` | Yes | Randomized per run | Misleading color injected in the system prompt |

## Success criterion

A run is successful when the agent returns `done` and the selected cube color matches the robot's true body color.
