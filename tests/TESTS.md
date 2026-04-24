# `tests/`

Repository test suite for compatibility wrappers, provider adapters, docs,
human-navigation flow, and the higher-risk refactors around evaluation and the
`e4` scene.

## Files

### `conftest.py`

Test bootstrap and path setup.

### `test_provider_adapters.py`

Mocked coverage for the provider adapters and the shared navigation parser.

### `test_wandb_logger.py`

Mocked coverage for the W&B logging layer used by judge evaluation.

### `test_module_layout.py`

Coverage for the canonical module layout and removal of retired compatibility
surfaces:

- canonical import paths
- removed legacy-path regressions
- moved asset resolution

### `test_e4_corridor_scene.py`

Coverage for the corridor task, prompt building, and the `e4` scene-specific
behavior.

### `test_human_nav_initialization.py`

Coverage for human-navigation controller setup, including single-run startup
and experiment-scoped scene initialization.

### `test_human_nav_finish_dialog.py`

Coverage for the human cube-task finish flow:

- submit path writes `runNN.json`
- cancel path resumes navigation
- multi-run progression continues after submit
- MuJoCo uses the OpenCV finish dialog backend
- OpenCV dialog uses `1/2/3` for cube selection and treats letters as
  reasoning input

### `test_folder_docs.py`

Coverage that every in-scope folder has its matching uppercase folder doc.

## Notes

- Tests are mostly mocked and import-level; they are designed to catch layout,
  adapter, parser, controller, and doc regressions without requiring live
  model calls.
- The live simulator still needs occasional manual smoke tests in MuJoCo.

## Manual smoke

Use these when changing experiment flow, provider dispatch, or judge tooling:

```sh
python main.py +experiment=e1 env=mujoco nav=<local-ollama-model> num_runs=1
python main.py +experiment=e2 env=mujoco nav=<local-ollama-model> num_runs=1
python main.py +experiment=e3 env=mujoco nav=<local-ollama-model> num_runs=1
python main.py env=mujoco nav=human mujoco.headless=false experiment=e1 num_runs=2
OPENROUTER_API_KEY=... python3 scripts/dev/openrouter_smoke.py --model qwen/qwen-vl-max:nitro
python -m experiments.evaluation.judge outputs/.../run01.json
```
