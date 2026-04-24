# `conf/`

Hydra configuration root for the simulator.

This folder contains the shared runtime config plus the experiment preset
overrides loaded with `+experiment=...`.

## Files

### `config.yaml`

Base config loaded by `main.py`.

It defines the main runtime knobs:

- environment backend selection
- navigation model and provider
- prompt/history settings
- camera/view controls such as `third_person`, `camera_height`, `body_style`, and `self_perception_mode`
- scene appearance controls such as `random_wall_color` and `e7_random_cube_heights`
- experiment mode and run count
- backend-specific headless options
- output/logging toggles

### `experiment/`

Preset overrides for `e0` through `e8`.

Each preset keeps the public CLI simple by setting:

- `experiment_id`
- default `num_runs`
- any experiment-specific override that should not live in the base config

## Notes

- `conf/` is a canonical folder, not a wrapper.
- See `experiment/EXPERIMENT.md` for the preset-by-preset overview.
