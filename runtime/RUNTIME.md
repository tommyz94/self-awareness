# `runtime/`

Canonical home for the simulator runtime coordinator.

## Files

### `controller.py`

Main orchestrator for:

- backend creation
- experiment/plugin resolution
- run setup and experiment loops
- human-navigation input handling
- human cube-task finish flow
- image capture
- navigation-model calls
- JSON/log/heatmap export

In human cube-task runs, `Controller` now:

- records movement steps directly from live key input
- captures observation metadata for those human steps
- opens the final selection dialog when `f` is pressed
- appends a synthetic `done` step with the selected cube and reasoning
- writes `runNN.json` through the same export path as model-driven runs
- advances to the next run when `num_runs > 1` and the human submits the
  finish dialog

### `human_finish_dialog.py`

Shared human finish-dialog helper used by the runtime controller.

- MuJoCo uses an OpenCV dialog backend
- the dialog collects the selected cube and free-text reasoning before the run
  is finalized

## Notes

- New code should import `Controller` from here.
- The controller resolves `experiment_id` through the experiment plugin layer
  and asks each `ExperimentSpec` to resolve its task and backend overrides.
- `env=mujoco nav=human` requires `mujoco.headless=false`.
- For MuJoCo human cube-task runs, press `f` to open the finish dialog, use
  `1/2/3` to choose the cube, type reasoning directly, then press `Enter` to
  submit.
