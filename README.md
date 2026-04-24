# Mirror AI — Robot Self-Recognition Simulator

A simulator for testing whether a vision-language model can recognize itself in a mirror and use that information to complete a goal-directed task.

## The Experiment

The main self-recognition scene contains:

- a left room where the main robot starts
- three colored cubes sampled per run from a 15-color pool
- a central divider with either a mirror opening or a solid wall
- mirrored visual clones of the robot and cubes on the other side of the divider when the mirror is enabled
- optional experiment-specific distractor robots in supported scenes

The cube-task objective is:

> **Select the cube that matches your color.**

A successful run requires the agent to:

1. navigate to gather evidence
2. infer its own color from the mirror view
3. call `done` with the matching `selected_cube`

## Running

Configuration is managed by Hydra. Override values with `key=value` syntax.
`env` defaults to `mujoco`, so `env=mujoco` is optional in commands unless you want to make it explicit.

**Human navigation:**

```sh
python main.py nav=human mujoco.headless=false
```

Keys: `W` forward, `S` backward, `A`/`D` rotate by 30° in `e1/e2/e3/e5/e6/e7`; in `e4`, `A`/`D` are forward diagonals and `S` is disabled. `e0` is a direct mirror-view sanity check, so human mode is mainly for visual inspection. `C` toggles camera.

**LLM navigation via Ollama:**

```sh
python main.py nav=gemma3:4b
```

Each step captures a JPG observation, sends it to the VLM, parses one JSON action, executes it, and records the run.

**Controlled experiments (quick commands):**

```sh
# LLM-driven
python main.py experiment=e0 nav=gemma3:4b
python main.py experiment=e1 nav=gemma3:4b
python main.py experiment=e2 nav=gemma3:4b
python main.py experiment=e3 nav=gemma3:4b
python main.py experiment=e4 nav=gemma3:4b
python main.py experiment=e5 nav=gemma3:4b

# Auto presets (e6/e7 default to nav=auto, 100 runs, 100 steps)
python main.py experiment=e6
python main.py experiment=e7

# Auto + dataset export
python main.py experiment=e6 save_dataset=true
python main.py experiment=e7 save_dataset=true

# Human mode examples
python main.py nav=human mujoco.headless=false experiment=e0
python main.py nav=human mujoco.headless=false experiment=e1
python main.py nav=human mujoco.headless=false experiment=e4
python main.py nav=human mujoco.headless=false experiment=e7
```

**Human mode (`mujoco.headless=false`) for every experiment:**

```sh
python main.py nav=human mujoco.headless=false experiment=e0
python main.py nav=human mujoco.headless=false experiment=e1
python main.py nav=human mujoco.headless=false experiment=e2
python main.py nav=human mujoco.headless=false experiment=e3
python main.py nav=human mujoco.headless=false experiment=e4
python main.py nav=human mujoco.headless=false experiment=e5
python main.py nav=human mujoco.headless=false experiment=e6
python main.py nav=human mujoco.headless=false experiment=e7
```

## Experiment Reference (`e0`-`e8`)

General note on color sampling:
- Agent, cubes, and distractors are sampled from a shared 15-color pool.
- There is currently no CLI parameter to force a specific fixed agent color for a run.

### `e0` — Direct Mirror Self-Perception (sanity check)

- What it does: places the agent directly in front of a wide, unobstructed mirror and asks it to report its own body color immediately. No navigation, cube selection, or scene disambiguation is required.
- Metrics currently computed: no aggregate summary metric. Run JSON contains `task_output.success`, `reported_color`, and the structured `identification` payload. Mirror-exposure fields are also exported per run.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e0/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; heatmaps disabled.
- Key parameters: preset fixes `num_runs=10` and `max_steps=1`; runtime options still include `random_wall_color=true|false`, `third_person=true|false`, `camera_height=<0..1>`, `body_style=default|clothes`, and `self_perception_mode=color|description`.
- Command examples:
- `python main.py experiment=e0 nav=gemma3:4b`
- `python main.py experiment=e0 body_style=clothes nav=gemma3:4b`
- `python main.py experiment=e0 body_style=clothes self_perception_mode=description nav=gemma3:4b`
- `python main.py experiment=e0 client=openrouter nav=meta-llama/llama-3.2-11b-vision-instruct`

### `e1` — Mirror Cube Task (baseline mirror)

- What it does: mirrored two-room cube task; agent must identify its own color and select the matching cube.
- Metrics currently computed: `SIA` (aggregate). `Δ_mirror` is obtained by comparing `SIA(e1)` vs `SIA(e2)`. Run-level mirror exposure fields are also exported in run JSON.
- Saved outputs: per-run JSON + experiment summary JSON (if `outputs.json_export=true`) under `outputs/mujoco/e1/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; heatmaps supported (`outputs.heatmap=true`).
- Key parameters: `num_runs`, `random_wall_color=true|false`, `third_person=true|false`, `camera_height=<0..1>`, `outputs.save_obs=true|false` (human mode).
- Command examples:
- `python main.py experiment=e1 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e1 random_wall_color=true`

### `e2` — No-Mirror Control

- What it does: same cube task layout as `e1`, but mirror disabled.
- Metrics currently computed: `SIA` (aggregate). Used as control for `Δ_mirror`.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e2/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; heatmaps supported.
- Key parameters: same as `e1`.
- Command examples:
- `python main.py experiment=e2 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e2`

### `e3` — Mirror + Misleading Prompt Color

- What it does: same mirror cube task, but prompt includes a misleading color cue (`wrong_color`).
- Metrics currently computed: `SIA` and `GI` (Grounding Index) for robustness to textual bias.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e3/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; heatmaps supported.
- Key parameters: same as `e1`, plus all standard provider/model parameters.
- Command examples:
- `python main.py experiment=e3 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e3`

### `e4` — Corridor Mirror-vs-Long Disambiguation

- What it does: each episode samples a hidden variant, either `mirror` corridor (`10x3`, target `done@10`) or `long` corridor (`20x3`, target `done@20`).
- Metrics currently computed: no aggregate cube metrics (`SIA/GI` not applicable). Run JSON contains task outcome fields such as `success`, `e4_variant`, `expected_done_step`, `actual_done_step`, `step_match`, `position_match`, and terminal-decision reason.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e4/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; heatmaps disabled for corridor layout.
- Experiment behavior: actions are `w/a/d/done` (no backward); `a/d` are forward diagonals with fixed yaw. In human mode, `F` maps to `done`.
- Key parameters: `num_runs`, `max_steps`, `third_person`, `camera_height`.
- Command examples:
- `python main.py experiment=e4 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e4 num_runs=10 max_steps=25`

### `e5` — Mirror Cube Task + Moving Distractors

- What it does: cube task like `e1`, with `1..6` moving distractor humanoids.
- Metrics currently computed: cube-task metrics (`SIA`; `GI` display is not informative here because no misleading color is injected).
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/e5/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; heatmaps supported.
- Experiment-specific parameters: `distractor_color_mode=allow_agent_color|exclude_agent_color|allow_duplicates`, `random_wall_color=true|false`.
- Command examples:
- `python main.py experiment=e5 nav=gemma3:4b`
- `python main.py experiment=e5 nav=gemma3:4b distractor_color_mode=exclude_agent_color`
- `python main.py nav=human mujoco.headless=false experiment=e5 distractor_color_mode=allow_duplicates`

### `e6` — Auto Explore with Distractors (No Cubes)

- What it does: mirrored room exploration with `1..6` moving distractor humanoids, no cubes. Preset defaults to auto-navigation.
- Metrics currently computed: no cube aggregate metrics (`SIA/GI` not applicable). Per-run task output includes completion/termination and identification payload.
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/e6/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; optional dataset export (`save_dataset=true` in `nav=auto`) to `Mirror_Video_Dataset/`.
- Experiment-specific parameters: `nav=auto|human|<model>` (preset default `auto`), `max_steps` (preset `100`), `save_dataset=true|false` (active for `nav=auto`), `distractor_color_mode=allow_agent_color|exclude_agent_color|allow_duplicates`, `random_wall_color=true|false`.
- Command examples:
- `python main.py experiment=e6`
- `python main.py experiment=e6 save_dataset=true`
- `python main.py nav=human mujoco.headless=false experiment=e6 distractor_color_mode=allow_agent_color`

### `e7` — `e6` + Random Distractor Cubes

- What it does: same as `e6` plus `5..15` distractor cube bases; optional vertical stacks (`1..5` height).
- Metrics currently computed: no cube aggregate metrics (`SIA/GI` not applicable for this explore task). Run JSON and optional dataset export are aligned with `e6`.
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/e7/<client>/<nav_slug>/<msr_tag>/<session_id>/run_01/`; optional dataset export to `Mirror_Video_Dataset/` when `save_dataset=true` and `nav=auto`.
- Experiment-specific parameters: `e7_random_cube_heights=true|false` (preset default `true`), `save_dataset=true|false` (auto mode), `distractor_color_mode=allow_agent_color|exclude_agent_color|allow_duplicates`, `random_wall_color=true|false`.
- Command examples:
- `python main.py experiment=e7`
- `python main.py experiment=e7 save_dataset=true`
- `python main.py experiment=e7 e7_random_cube_heights=false`

### `e8` — Embodiment Prior Bias

- What it does: mirrors `e5` scene complexity and moving-distractor setup, but replaces the cube-color objective with an exploration-based self-description task. The ego agent, mirror reflection, and distractors default to clothed human-mesh avatars when available.
- Metrics currently computed: no cube aggregate metrics (`SIA/GI` not applicable). Run JSON includes `task_output.success`, `task_output.self_description`, and mirror-exposure fields.
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/e8/<client>/<nav_slug>/clothes/<msr_tag>/<session_id>/run_01/`; heatmaps supported.
- Experiment-specific parameters: inherits `e5` scene controls plus `body_style=default|clothes`. The `e8` preset sets `body_style=clothes`.
- Command examples:
- `python main.py experiment=e8 nav=gemma3:4b`
- `python main.py experiment=e8 client=openrouter nav=meta-llama/llama-3.2-11b-vision-instruct`

In mirror-enabled two-room scenes, mirror content is visual-only: reflected cubes,
the mirrored humanoid, and `e5`/`e6`/`e7`/`e8` mirrored distractors appear only when the real
object lies inside the active mirror opening (except experiment-specific overrides, where configured).

## MuJoCo backend

MuJoCo renders the default articulated humanoid by default, while movement stays discrete and kinematic. With `body_style=clothes`, it instead uses clothed avatar meshes discovered under `clothes/objs/` together with their preferred color textures when available (falling back to the default humanoid if no usable OBJ is found).

**Headless MuJoCo (optional, any platform):**

```sh
pip install mujoco pillow
python main.py nav=gemma3:4b
```

**MuJoCo with live RGB window:**

```sh
pip install mujoco pillow opencv-python
python main.py nav=gemma3:4b mujoco.headless=false
python main.py nav=human mujoco.headless=false
```

When `mujoco.headless=false`, MuJoCo can show a live RGB window during both human and LLM runs. When `mujoco.headless=true`, MuJoCo stays fully offscreen — no browser, virtual display, or GPU required.

On Linux, headless MuJoCo defaults to `MUJOCO_GL=egl`. If a cluster node lacks EGL support, override it with `MUJOCO_GL=osmesa`.

## Provider selection

The navigation VLM can use Ollama (default), OpenAI, OpenRouter, Google Gemini, or Anthropic Claude.
For Ollama, you can inspect locally installed models with `ollama list`.

| Provider (`client`) | SDK / install | API key env var | Command example |
|---|---|---|---|
| `ollama` | local Ollama server | none | `python main.py client=ollama nav=gemma3:4b` |
| `openai` | `pip install "openai>=1.0.0"` | `OPENAI_API_KEY` | `OPENAI_API_KEY=sk-... python main.py client=openai nav=gpt-5` |
| `openrouter` | `pip install "openai>=1.0.0"` | `OPENROUTER_API_KEY` | `OPENROUTER_API_KEY=... python main.py client=openrouter nav=meta-llama/llama-3.2-11b-vision-instruct` |
| `google` | `pip install google-genai` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `GEMINI_API_KEY=... python main.py client=google nav=gemini-2.5-flash` |
| `anthropic` | `pip install "anthropic>=0.68.0"` | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY=... python main.py client=anthropic nav=claude-sonnet-4-5` |

`client=claude` is accepted as an alias, but `anthropic` is the canonical provider
name used internally and in output paths.

API keys can also be set in `conf/config.yaml` under `openai.api_key`,
`openrouter.api_key`, `google.api_key`, or `anthropic.api_key`.

For `client=openrouter`, `nav` must be the OpenRouter model id, not the local
Ollama tag. For example, local `llama3.2-vision:11b` maps to OpenRouter model
id `meta-llama/llama-3.2-11b-vision-instruct`. The OpenRouter adapter requests
strict JSON-schema output, so pick a model/provider route that supports
structured outputs.

For a quick connectivity check before a full run, use the smoke script in
`scripts/dev/openrouter_smoke.py`. By default it sends one minimal structured
request with a generated 1x1 PNG so you can verify API key, model routing,
JSON-schema output, and the image path:

`OPENROUTER_API_KEY=... python3 scripts/dev/openrouter_smoke.py --model qwen/qwen-vl-max:nitro`

For text-only models, add `--no-image`.

To check several model ids at once, repeat `--model` or use comma-separated
values. If you want to test the Nitro route for bare model ids without editing
each one manually, add `--variant nitro`. This is the most reliable way to
verify whether a given model id actually works for this repo's exact OpenRouter
request shape:

`OPENROUTER_API_KEY=... python3 scripts/dev/openrouter_smoke.py --variant nitro --model google/gemma-4-26b-a4b-it --model google/gemma-4-31b-it --model mistralai/ministral-14b-2512 --model qwen/qwen-vl-max --model qwen/qwen3.6-plus`

If you launch many OpenRouter jobs in parallel and see bursts of `429 Too Many Requests`,
set `openrouter.min_request_interval_sec=<seconds>`. This enforces a shared
cross-process delay between OpenRouter requests on the same machine, so matrix
jobs started in separate Terminal windows do not all hit the API at once.
Start with `0.5` or `1.0` and increase if rate limits persist.

For large OpenRouter sweeps, use the matrix launcher in
`scripts/dev/openrouter_matrix.py`. Define models, experiments, and shared args
in `scripts/dev/openrouter_matrix.example.json`, preview with:

`python3 scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.example.json`

Then launch one macOS Terminal window per model x experiment job with:

`python3 scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.example.json --launch`

For a minimal 2x2 Terminal-launch smoke check, use:

`python3 scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.smoke.json`

and then:

`python3 scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.smoke.json --launch`

You can also restrict the launch set from the CLI:

`python3 scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.example.json --models meta-llama/llama-3.2-11b-vision-instruct --experiments e1,e2 --launch`

To override normal Hydra flags without editing the JSON, pass repeated
`--set key=value` arguments. These apply to every generated job and override
values from the matrix file:

`python3 scripts/dev/openrouter_matrix.py --config scripts/dev/openrouter_matrix.example.json --set seed=123 --set history_steps=2 --set mujoco.headless=true --launch`

The launcher intentionally refuses to override keys owned by the selected
experiment presets in `conf/experiment/<id>.yaml`. That protects fixed setup
choices such as `num_runs` for all experiments and `max_steps` for `e6` and
`e7`. The matrix still controls `nav` itself through the selected OpenRouter
model list, so `e6` and `e7` can be launched from the matrix even though their
presets default to `nav: auto`.

## Config keys

| Key | Default | Description |
|---|---|---|
| `env` | `mujoco` | Backend. Supported value: `mujoco`. |
| `nav` | `human` | `human`, `auto` (currently for `e6/e7`), or a model name for the selected `client`. |
| `client` | `ollama` | Navigation provider: `ollama`, `openai`, `openrouter`, `google`, or `anthropic`. `claude` is accepted as an alias for `anthropic`. |
| `print_reasoning` | `false` | Stream thinking tokens (Ollama only). |
| `distractors` | `0` | Global distractor knob used by experiments that opt into it. `e5`/`e6`/`e7` randomize distractor count per run in `[1, 6]`. |
| `max_steps` | `20` | Maximum LLM steps per run. |
| `third_person` | `false` | Use third-person camera instead of first-person. |
| `camera_height` | `1.0` | Scale the first-person camera eye height from `0.0` (ground level) to `1.0` (current default). Third-person view is unaffected. |
| `body_style` | `default` | Visual embodiment style. `default` uses the built-in humanoid renderer; `clothes` uses clothed avatar OBJ meshes and their texture maps when available. |
| `self_perception_mode` | `color` | `e0` self-report task variant. `color` asks for body color; `description` asks for a detailed appearance description from the mirror. |
| `random_wall_color` | `false` | For `e0/e1/e2/e3/e5/e6/e7`, sample one non-agent wall color per run and apply it to all room walls. |
| `distractor_color_mode` | `allow_duplicates` | `e5/e6/e7` distractor color policy. `allow_agent_color`: distractors may use ego color (no duplicates). `exclude_agent_color`: distractors must differ from ego (no duplicates). `allow_duplicates`: distractors are sampled with replacement (duplicates allowed, ego color allowed). |
| `e7_random_cube_heights` | `false` | `e7`-only behavior toggle. When `true`, each sampled cube base becomes a same-color vertical stack with random height in `[1, 5]`; when `false`, each base is a single cube. The `e7` preset sets this to `true` by default. |
| `save_dataset` | `false` | Dataset export switch. Currently active for `experiment=e6/e7` with `nav=auto`; writes `Mirror_Video_Dataset/` with RGB frames plus per-episode JSON metadata. |
| `history_steps` | `1` | Number of previous steps included in the prompt. |
| `experiment_id` | `null` | Internal experiment id set by `+experiment=e0/e1/e2/e3/e4/e5/e6/e7/e8`. |
| `num_runs` | `1` | Number of runs. The `e0` preset sets this to `10`; `e6/e7` set it to `100`. |
| `ollama.host` | `http://localhost:11434` | Ollama endpoint. |
| `openai.api_key` | `""` | OpenAI API key fallback (prefer `OPENAI_API_KEY` env var). |
| `openrouter.api_key` | `""` | OpenRouter API key fallback (prefer `OPENROUTER_API_KEY` env var). |
| `openrouter.base_url` | `https://openrouter.ai/api/v1` | OpenRouter API base URL. |
| `openrouter.http_referer` | `""` | Optional `HTTP-Referer` header for OpenRouter app attribution. |
| `openrouter.app_title` | `""` | Optional `X-OpenRouter-Title` header for OpenRouter app attribution. |
| `openrouter.require_parameters` | `true` | Restrict routing to providers that support all requested parameters (recommended for structured output). |
| `openrouter.min_request_interval_sec` | `0.0` | Shared machine-local throttle for OpenRouter requests. Useful when parallel Terminal jobs trigger `429` bursts. |
| `google.api_key` | `""` | Google API key fallback (prefer `GEMINI_API_KEY` env var). |
| `anthropic.api_key` | `""` | Anthropic API key fallback (prefer `ANTHROPIC_API_KEY` env var). |
| `mujoco.headless` | `false` | Offscreen MuJoCo when `true`; live OpenCV window when `false` (default). |
| `outputs.dir` | `outputs` | Root directory for all generated artifacts. |
| `outputs.log_file` | `llm_history.txt` | Per-run log filename written inside the run directory. |
| `outputs.json_export` | `true` | Write per-run JSON and experiment summary JSON. |
| `outputs.heatmap` | `true` | Write per-run heatmap PNGs for supported room-style scenes. |
| `outputs.save_obs` | `false` | Save JPG observations after each manual key press. |

## Output layout

Generated files are grouped by backend, experiment id, provider, navigation
mode/model, MSR level, and session timestamp. Built-in experiments (`e0`
through `e7`) all use experiment-scoped roots. Ad-hoc single runs without
`experiment=<id>` keep the legacy provider-first layout.

```text
outputs/
  <env>/
    <experiment_id>/         # built-in experiments e0-e7
      <client>/
        <nav_slug>/
          <msr_tag>/         # e.g. level_2
            <session_id>/    # timestamped controller session
              run_01/
                obs/
                llm_history.txt
                run01.json
    <client>/                # ad-hoc single runs without experiment=
      <nav_slug>/
        <msr_tag>/
          <session_id>/
            run_01/
```

Examples:

- `outputs/mujoco/ollama/gemma3_4b/runs/run01.json`
- `outputs/mujoco/e0/openrouter/meta-llama_llama-3.2-11b-vision-instruct/level_2/20260423_120000/run_01/run01.json`
- `outputs/mujoco/e1/ollama/gemma3_4b/level_2/20260423_120000/run_01/run01.json`
- `outputs/mujoco/openai/gpt-5/level_2/20260423_120000/run_01/run01.json`
- `outputs/mujoco/e4/google/gemini-2.5-flash/level_2/20260423_120000/experiment_summary.json`
- `outputs/mujoco/e5/ollama/gemma3_4b/level_2/20260423_120000/run_01/run01.json`
- `outputs/mujoco/e6/ollama/auto/level_2/20260423_120000/run_01/run01.json`
- `outputs/mujoco/e7/anthropic/claude-sonnet-4-5/level_2/20260423_120000/run_01/run01.json`
- `outputs/mujoco/ollama/gemma3_4b/level_2/20260423_120000/run_01/run01.json`
- `Mirror_Video_Dataset/episodes/episode_0001/rgb/000000.jpg` (`experiment=e6 save_dataset=true` or `experiment=e7 save_dataset=true`)
- `Mirror_Video_Dataset/episodes/episode_0001/episode.json` (`experiment=e6 save_dataset=true` or `experiment=e7 save_dataset=true`)

`nav_slug` is the `nav` value with `:`, `/`, and `\` replaced by `_`.

## Project Structure

```text
simulator/
├── main.py
├── conf/
├── providers/
├── llm/
├── runtime/
├── env/
├── assets/
├── experiments/
├── utils/
├── outputs/
└── images/
```
