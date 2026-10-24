# Mirror, Mirror on the Wall

Can a vision-language model distinguish its own reflection from another robot,
identify its body color, and act on that information? This repository provides
the MuJoCo environments, model interfaces and evaluation tools for
*Mirror, Mirror on the Wall: Can VLM Agents Tell Who They Are at All?*

[Paper](https://arxiv.org/abs/2605.08816) · [Citation](#citation)

You can inspect the environments yourself, run a local or API-hosted model,
and calculate benchmark scores from saved episodes. The repository also
includes the scripts used to produce the paper's comparison figures.

## What the benchmark does

A model controls a robot through a first-person camera. At each step it receives
an RGB image, recent history and position information, then returns a structured
response containing its next action and self-identification. It can move
forward/backward, turn left/right, or finish with `done`.

The five official conditions match E1–E5 in the paper:

| Preset | What changes | What the agent must do |
| --- | --- | --- |
| [e1](experiments/paper/e1/experiment.py) | Baseline room with a mirror | Identify its color and select the matching cube from three candidates |
| [e2](experiments/paper/e2/experiment.py) | The mirror is removed | Perform the same cube-selection task without mirror evidence |
| [e3](experiments/paper/e3/experiment.py) | The prompt supplies an incorrect body color | Resolve the conflict between language and vision, then select a cube |
| [e4](experiments/paper/e4/experiment.py) | Other robots move through the mirrored scene | Distinguish itself from distractors and select a cube |
| [e5](experiments/paper/e5/experiment.py) | Cluttered exploration with moving robots | Explore and report its own color at the final decision; no cube selection |

Each official preset runs **seven episodes per seed**, with a **100-step limit**,
640×480 observations and one-step textual history. The primary paper evaluation
uses three seeds per model and condition. The [presets](conf/experiment/)
define these settings; [task modules](experiments/tasks/) contain the exact
prompts and response schemas.

## Run an experiment

Use **Python 3.11** and run commands from the repository root. First install
the simulator, metric calculator, plotting tools and notebooks:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/dev/check_installation.py
```

The check validates all five scene presets without calling a model. Add
`--render` to check image rendering too. On Linux without working EGL, install
OSMesa (for example, `libosmesa6` on Ubuntu) and set `MUJOCO_GL=osmesa`.

### Choose a model backend

Start with **OpenRouter for hosted models** or **Ollama for local models**.
The same experiment runner also supports direct provider APIs as alternatives.
`client` selects the backend; `nav` selects the exact model ID understood by
that backend. Choose a model that accepts images and can return the required
structured response.

| Backend | `client` | Optional Python SDK | Setup |
| --- | --- | --- | --- |
| OpenRouter | `openrouter` | `openai==2.2.0` | `OPENROUTER_API_KEY` |
| Ollama | `ollama` | None | Running Ollama server with a vision model installed |
| OpenAI | `openai` | `openai==2.2.0` | `OPENAI_API_KEY` |
| Google Gemini | `google` | `google-genai==1.67.0` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| Anthropic Claude | `anthropic` | `anthropic==0.97.0` | `ANTHROPIC_API_KEY` |

**Hosted example (OpenRouter).** Install the SDK, set your API key,
and replace `YOUR_VISION_MODEL` with its OpenRouter model ID:

```sh
python -m pip install "openai==2.2.0"
export OPENROUTER_API_KEY="YOUR_API_KEY"
python main.py experiment=e1 client=openrouter nav=YOUR_VISION_MODEL \
  seed=123 mujoco.headless=true
```

For other APIs, use the SDK, `client` and credential variable from the table
above. To install all supported API SDKs, use
`python -m pip install -r requirements-providers.txt`.
All provider settings are in [conf/config.yaml](conf/config.yaml).
The [Google](providers/google.py) and [Anthropic](providers/anthropic.py)
adapters also support Vertex AI through application-default credentials and
their `use_vertexai`, `project` and `location` settings.

**Local example.** With Ollama running and a vision model installed, replace
`YOUR_OLLAMA_MODEL` with its installed name and set the server address:

```sh
python main.py experiment=e1 client=ollama nav=YOUR_OLLAMA_MODEL \
  ollama.host=http://localhost:11434 seed=123 mujoco.headless=true
```

Both commands run seven E1 episodes without opening a simulator window.
`seed=123` fixes the random seed; choose a different seed for another batch.
Change `experiment=e1` to `e2`, `e3`, `e4` or `e5` to choose another condition.
Append `--cfg job` to inspect the configuration without starting a run.
For a short setup check, append `num_runs=1 max_steps=5`; this changes the
benchmark protocol and is only a smoke test.

### Explore the scene yourself

Human control needs neither a model nor API credentials:

```sh
python main.py experiment=e1 nav=human mujoco.headless=false
```

Use W/S to move and A/D to turn. In cube tasks, press F to open the final
selection dialog, choose a cube with 1/2/3, enter reasoning and press Enter.
Q or Esc closes the scene.

### Find the outputs

Official runs are saved under `outputs/paper/mujoco/`, grouped by experiment,
provider, model, level and session:

```text
outputs/paper/mujoco/<experiment>/<provider>/<model>/level_2/<timestamp>/run_01/
```

Each model-driven episode saves `runNN.json`
(setup, actions and identifications), observation images, and a text log.
The session also contains `experiment_summary.json`. These runtime summaries
are useful for inspection; calculate the paper's aggregate metrics with the
evaluator below.
Generated runs and `reports/` are ignored by Git.

## Turn saved episodes into scores and figures

The workflow is **run episodes → select logs → calculate metrics → plot a CSV**.
Evaluation and plotting run offline and do not require a model or simulator.

Create a JSON manifest listing the episodes you want to score. For example,
save this as `selected_runs.json`, replacing the path with a real run JSON
and adding one row per episode:

```json
{
  "experiment_id_scheme": "paper-v1",
  "manifest_rows": [
    {"experiment_id": "e1", "path": "PATH_TO_YOUR_E1_RUN_JSON"}
  ]
}
```

Use `e1`–`e5` for `experiment_id`, matching each episode's condition, and keep
`experiment_id_scheme` set to `paper-v1`. Paths are relative to the repository
root, or absolute. Keep the output folder structure intact: the evaluator
reads the experiment, provider and model from
`<experiment>/<provider>/<model>/.../runNN.json`.

```sh
python scripts/analysis/evaluate_selected_runs.py \
  --manifest selected_runs.json --dry-run
python scripts/analysis/evaluate_selected_runs.py \
  --manifest selected_runs.json \
  --out-dir reports/evaluation
```

The dry run checks the selection. Evaluation writes `paper_metrics.csv`, input
hashes and calculation provenance to a **new** report directory. Choose a
different `--out-dir` for each evaluation; existing directories are refused.
Use `paper_metrics.csv` for scores labeled E1–E5. The evaluator handles the
calculator's internal identifiers automatically and scores only the episodes
listed in your manifest.

The six core scores describe task success (TSA), steps to a decision (TTD),
mirror exposure overall and among successful episodes (MCR and MTATO), correct
color claims (CAAL), and claims preceding mirror evidence (CR). All models use
the same [calculator](experiment.py): success requires a correct `done`, MCR
and CAAL exclude that step, and CR searches the full trajectory. Timeouts are
failures; MTATO is undefined without successful episodes. The calculator
defines the exact denominators and aggregation rules.

### Plot the results

Open [results_analysis.ipynb](notebooks/results_analysis.ipynb), set `CSV_PATH`
to your evaluator's `paper_metrics.csv`, and run the cells in order:

```sh
jupyter lab notebooks/results_analysis.ipynb
```

The notebook displays the five comparison figures supported by the available
conditions and statistics. Set `SAVE_FIGURES = True` in its last cell to export
PDF, SVG and PNG files to `reports/figures/`. Missing statistics stay N/A;
models are discovered from the CSV, including models outside the paper.
The notebook starts without saved outputs.

The same [plotting code](notebooks/plot_paper_results.py) also runs directly:

```sh
python notebooks/plot_paper_results.py \
  --csv reports/evaluation/paper_metrics.csv \
  --output-dir reports/figures
```

Both routes use the same functions. Exports include a `figure_manifest.json`
with input hashes. Multiple CSVs are accepted after `--csv` when their
model/condition pairs do not overlap. Generated figures are ignored by Git.

**Reproducing the paper:** original evaluation logs and aggregate inputs are
not included in a code checkout. The primary set contains 840 episodes across
eight models; Cosmos adds 35 episodes with a smaller sample per condition.
You can run new episodes with the code and bundled assets. Reproducing the
reported numbers requires the original logs and selection manifest; recreating
the figures requires the aggregate CSV. Download instructions for these paper
inputs are not yet provided.

## Where to find things

| Location | Responsibility |
| --- | --- |
| [main.py](main.py), [conf/config.yaml](conf/config.yaml) | Entry point and configuration |
| [experiments/paper/](experiments/paper/), [loader.py](experiments/loader.py) | Official scene definitions and experiment selection |
| [tasks/](experiments/tasks/), [shared/](experiments/shared/) | Prompts, response schemas, shared geometry and distractor mechanics |
| [runtime/controller.py](runtime/controller.py) | Episode loop: capture an image, query the model, apply an action, save logs |
| [env/backends/mujoco.py](env/backends/mujoco.py) | Simulation, rendering and movement |
| [llm/navigation/](llm/navigation/), [providers/](providers/) | Model messages, response parsing and backend adapters |
| [evaluate_selected_runs.py](scripts/analysis/evaluate_selected_runs.py), [experiment.py](experiment.py) | Input selection and core metric calculation |
| [scripts/dev/](scripts/dev/) | Optional diagnostics and batch utilities; not needed for the workflow above |
| [notebooks/](notebooks/), [tests/](tests/) | Analysis code and automated checks |

Start with `main.py` and `runtime/controller.py` to follow an episode.
Research extensions outside the official E1–E5 benchmark are in
[experiments/experimental/](experiments/experimental/) and require an explicit
`experiment=experimental/<name>` selector. These extensions are under
development: `direct_perception`, `corridor`, `exploration`, `embodiment`,
`front_identity` and `mirror_side_choice`. The embodiment extension requires
additional avatar assets in `clothes/objs/` that are not distributed with this
repository. All assets needed for E1–E5 are bundled.

## Tests

```sh
python -m pip install -r requirements.txt "pytest>=8,<10"
python -m pytest -q
```

The suite includes metric and plotting checks. Tests use mocked providers
and make no paid model calls. Use
`python scripts/dev/check_installation.py --providers` after installing all
API SDKs to check their imports without making requests.

## Citation

```bibtex
@misc{ziliotto2026mirrormirrorwallvlm,
  title={Mirror, Mirror on the Wall: Can VLM Agents Tell Who They Are at All?},
  author={Filippo Ziliotto and Ciro Beneduce and Bruno Lepri and Luciano Serafini and Massimiliano Luca and Tommaso Campari},
  year={2026},
  eprint={2605.08816},
  archivePrefix={arXiv},
  primaryClass={cs.AI},
  url={https://arxiv.org/abs/2605.08816},
}
```
