# Mirror, Mirror on the Wall

Code for *Mirror, Mirror on the Wall: Can VLM Agents Tell Who They Are at All?*
(arXiv: [2605.08816](https://arxiv.org/abs/2605.08816)).

Citation metadata: [CITATION.cff](CITATION.cff).
Scope, data access, and release status: [benchmark card](docs/BENCHMARK_CARD.md).

A first-person MuJoCo benchmark for functional mirror-guided self-identification.
The agent reports a body color and, in E1-E4, selects a matching cube. This
evaluates grounded behavior; it does not establish philosophical self-awareness.

## Install

Python 3.11 is the tested development version.

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -r requirements-dev.txt
```

Provider SDKs are used only for the selected backend. Local experimental inference adapters may have additional optional dependencies;
see [providers](providers/PROVIDERS.md).
Do not put API keys in tracked configuration files.

## Paper experiments

Use the explicit `paper/` namespace. Bare numeric IDs retain their historical
meaning so existing scripts and datasets remain readable.

| Paper condition | Preset | Historical implementation | Task |
| --- | --- | --- | --- |
| E1: Mirror baseline | `paper/e1` | `e1` | Matching cube |
| E2: No mirror | `paper/e2` | `e2` | Matching cube |
| E3: Misleading language | `paper/e3` | `e3` | Matching cube |
| E4: Distractor robots | `paper/e4` | `e5` | Matching cube |
| E5: Cluttered exploration | `paper/e5` | `e7` | Final self-attribution |

Check a preset without rendering or contacting a model:

```sh
python main.py experiment=paper/e4 --cfg job
```

Run one seed of seven episodes with a model:

```sh
python main.py experiment=paper/e1 client=google nav=gemini-2.5-pro seed=123 mujoco.headless=true
```

For an OpenRouter matrix, preview the commands before launching:

```sh
python scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.paper.json
```

This preview makes no model calls. The example covers two OpenRouter models
and one seed; it is not the complete nine-model evaluation.

The presets fix a 100-step budget, one-step textual history, first-person
640x480 observations, the standard humanoid, and seven episodes per seed.
The original eight-model evaluation used three seeds per condition. The
additional Cosmos set has one seed per condition; see [reproducibility](docs/REPRODUCIBILITY.md).
The shared pool contains **ten colors**, with three distinct candidate cubes
in E1-E4. E5 has no target-cube decision.

Inspect the scene interactively:

```sh
python main.py experiment=paper/e1 nav=human mujoco.headless=false
```

Human controls are W/S for movement, A/D for 30-degree turns, C for camera,
and F for the cube-selection dialog. The corridor extras use different controls.

## Preserved research extras

Original research files remain in place. The shared presets include:

| Extra preset | Historical ID | Purpose |
| --- | --- | --- |
| `extras/direct_perception` | `e0` | Direct mirror-view sanity check |
| `extras/corridor` | `corridor` | Original mirror/long-corridor stopping task |
| `extras/exploration` | `e6` | Exploration and optional video export |
| `extras/embodiment` | `e8` | Clothed-avatar self-description |

Additional local front-identity and mirror-side-choice prototypes are preserved
but are not included in this alignment commit; their publication and integration
need a separate code review. See [development](docs/DEVELOPMENT.md).

```sh
python main.py experiment=extras/direct_perception nav=human mujoco.headless=false
python main.py experiment=extras/exploration save_dataset=true mujoco.headless=true
```

The numeric `e7` preset retains automatic navigation; `paper/e5` does not
inherit that default. Always provide `nav=<model>` for model evaluations.
For non-cube tasks the simulator skips room-style heatmaps when the task does
not support them. Full mapping and controls: [experiments](docs/EXPERIMENTS.md).

## Data and evaluation status

Original run JSONs, observations, manifests, and notebooks are preserved in
place. New paper runs go to `outputs/paper/`; extras go to `outputs/extras/`.
Subdirectories and `experiment_id` keep the implementation ID. Paper-run JSONs
also include `paper_experiment_id`, and their `run_context` records both IDs.
Thus paper E4 is unambiguously labeled even though its stable directory is `e5`.

**Metric implementations and reported results are unchanged by this alignment.**
The final evaluation code is being obtained from a collaborator. Endpoint,
terminal-step, and undefined-value conventions must be reconciled before a
new metrics release. [Outstanding metric questions](docs/METRIC_CONVENTIONS.md).

Historical analysis commands, selected counts, and data-preservation checks are
documented in [reproducibility](docs/REPRODUCIBILITY.md). Generated data remain
local; a public code checkout is not a download of the evaluation dataset.

## Tests and development

```sh
python -m pytest -q
```

Tests use mocked providers and do not require paid model calls. Development
checks, optional assets, and the Git inclusion policy are described in
[development](docs/DEVELOPMENT.md). Provider/API options remain documented in
[providers](providers/PROVIDERS.md) and [developer scripts](scripts/dev/DEV.md).
