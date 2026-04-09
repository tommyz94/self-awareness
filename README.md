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

Keys: `W` forward, `S` backward, `A`/`D` rotate by 30° in `e1/e2/e3/e5/e6/e7`; in `e4`, `A`/`D` are forward diagonals and `S` is disabled. `C` toggles camera.

**LLM navigation via Ollama:**

```sh
python main.py nav=gemma3:4b
```

Each step captures a JPG observation, sends it to the VLM, parses one JSON action, executes it, and records the run.

**Controlled experiments (quick commands):**

```sh
# LLM-driven
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
python main.py nav=human mujoco.headless=false experiment=e1
python main.py nav=human mujoco.headless=false experiment=e4
python main.py nav=human mujoco.headless=false experiment=e7
```

**Human mode (`mujoco.headless=false`) for every experiment:**

```sh
python main.py nav=human mujoco.headless=false experiment=e1
python main.py nav=human mujoco.headless=false experiment=e2
python main.py nav=human mujoco.headless=false experiment=e3
python main.py nav=human mujoco.headless=false experiment=e4
python main.py nav=human mujoco.headless=false experiment=e5
python main.py nav=human mujoco.headless=false experiment=e6
python main.py nav=human mujoco.headless=false experiment=e7
```

## Experiment Reference (`e1`-`e7`)

General note on color sampling:
- Agent, cubes, and distractors are sampled from a shared 15-color pool.
- There is currently no CLI parameter to force a specific fixed agent color for a run.

### `e1` — Mirror Cube Task (baseline mirror)

- What it does: mirrored two-room cube task; agent must identify its own color and select the matching cube.
- Metrics currently computed: `SIA` (aggregate). `Δ_mirror` is obtained by comparing `SIA(e1)` vs `SIA(e2)`. Run-level mirror exposure fields are also exported in run JSON.
- Saved outputs: per-run JSON + experiment summary JSON (if `outputs.json_export=true`) under `outputs/mujoco/e1/<client>/<nav_slug>/runs/`; heatmaps supported (`outputs.heatmap=true`).
- Key parameters: `num_runs`, `random_wall_color=true|false`, `third_person=true|false`, `camera_height=<0..1>`, `outputs.save_obs=true|false` (human mode).
- Command examples:
- `python main.py experiment=e1 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e1 random_wall_color=true`

### `e2` — No-Mirror Control

- What it does: same cube task layout as `e1`, but mirror disabled.
- Metrics currently computed: `SIA` (aggregate). Used as control for `Δ_mirror`.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e2/<client>/<nav_slug>/runs/`; heatmaps supported.
- Key parameters: same as `e1`.
- Command examples:
- `python main.py experiment=e2 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e2`

### `e3` — Mirror + Misleading Prompt Color

- What it does: same mirror cube task, but prompt includes a misleading color cue (`wrong_color`).
- Metrics currently computed: `SIA` and `GI` (Grounding Index) for robustness to textual bias.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e3/<client>/<nav_slug>/runs/`; heatmaps supported.
- Key parameters: same as `e1`, plus all standard provider/model parameters.
- Command examples:
- `python main.py experiment=e3 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e3`

### `e4` — Corridor Mirror-vs-Long Disambiguation

- What it does: each episode samples a hidden variant, either `mirror` corridor (`10x3`, target `done@10`) or `long` corridor (`20x3`, target `done@20`).
- Metrics currently computed: no aggregate cube metrics (`SIA/GI` not applicable). Run JSON contains task outcome fields such as `success`, `e4_variant`, `expected_done_step`, `actual_done_step`, `step_match`, `position_match`, and terminal-decision reason.
- Saved outputs: per-run JSON + experiment summary JSON in `outputs/mujoco/e4/<client>/<nav_slug>/runs/`; heatmaps disabled for corridor layout.
- Experiment behavior: actions are `w/a/d/done` (no backward); `a/d` are forward diagonals with fixed yaw. In human mode, `F` maps to `done`.
- Key parameters: `num_runs`, `max_steps`, `third_person`, `camera_height`.
- Command examples:
- `python main.py experiment=e4 nav=gemma3:4b`
- `python main.py nav=human mujoco.headless=false experiment=e4 num_runs=10 max_steps=25`

### `e5` — Mirror Cube Task + Moving Distractors

- What it does: cube task like `e1`, with `1..6` moving distractor humanoids.
- Metrics currently computed: cube-task metrics (`SIA`; `GI` display is not informative here because no misleading color is injected).
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/<client>/<nav_slug>/runs/e5/`; heatmaps supported.
- Experiment-specific parameters: `distractor_color_mode=allow_agent_color|exclude_agent_color|allow_duplicates`, `random_wall_color=true|false`.
- Command examples:
- `python main.py experiment=e5 nav=gemma3:4b`
- `python main.py experiment=e5 nav=gemma3:4b distractor_color_mode=exclude_agent_color`
- `python main.py nav=human mujoco.headless=false experiment=e5 distractor_color_mode=allow_duplicates`

### `e6` — Auto Explore with Distractors (No Cubes)

- What it does: mirrored room exploration with `1..6` moving distractor humanoids, no cubes. Preset defaults to auto-navigation.
- Metrics currently computed: no cube aggregate metrics (`SIA/GI` not applicable). Per-run task output includes completion/termination and identification payload.
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/<client>/<nav_slug>/runs/e6/`; optional dataset export (`save_dataset=true` in `nav=auto`) to `Mirror_Video_Dataset/`.
- Experiment-specific parameters: `nav=auto|human|<model>` (preset default `auto`), `max_steps` (preset `100`), `save_dataset=true|false` (active for `nav=auto`), `distractor_color_mode=allow_agent_color|exclude_agent_color|allow_duplicates`, `random_wall_color=true|false`.
- Command examples:
- `python main.py experiment=e6`
- `python main.py experiment=e6 save_dataset=true`
- `python main.py nav=human mujoco.headless=false experiment=e6 distractor_color_mode=allow_agent_color`

### `e7` — `e6` + Random Distractor Cubes

- What it does: same as `e6` plus `5..15` distractor cube bases; optional vertical stacks (`1..5` height).
- Metrics currently computed: no cube aggregate metrics (`SIA/GI` not applicable for this explore task). Run JSON and optional dataset export are aligned with `e6`.
- Saved outputs: per-run JSON + experiment summary JSON under `outputs/mujoco/<client>/<nav_slug>/runs/e7/`; optional dataset export to `Mirror_Video_Dataset/` when `save_dataset=true` and `nav=auto`.
- Experiment-specific parameters: `e7_random_cube_heights=true|false` (preset default `true`), `save_dataset=true|false` (auto mode), `distractor_color_mode=allow_agent_color|exclude_agent_color|allow_duplicates`, `random_wall_color=true|false`.
- Command examples:
- `python main.py experiment=e7`
- `python main.py experiment=e7 save_dataset=true`
- `python main.py experiment=e7 e7_random_cube_heights=false`

In mirror-enabled two-room scenes, mirror content is visual-only: reflected cubes,
the mirrored humanoid, and `e5`/`e6`/`e7` mirrored distractors appear only when the real
object lies inside the active mirror opening (except experiment-specific overrides, where configured).

## MuJoCo backend

MuJoCo renders an articulated humanoid loaded from `assets/mujoco/humanoid.xml`, while movement stays discrete and kinematic.

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

The navigation VLM can use Ollama (default), OpenAI, Google Gemini, or Anthropic Claude.
For Ollama, you can inspect locally installed models with `ollama list`.

| Provider (`client`) | SDK / install | API key env var | Command example |
|---|---|---|---|
| `ollama` | local Ollama server | none | `python main.py client=ollama nav=gemma3:4b` |
| `openai` | `pip install "openai>=1.0.0"` | `OPENAI_API_KEY` | `OPENAI_API_KEY=sk-... python main.py client=openai nav=gpt-5` |
| `google` | `pip install google-genai` | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `GEMINI_API_KEY=... python main.py client=google nav=gemini-2.5-flash` |
| `anthropic` | `pip install "anthropic>=0.68.0"` | `ANTHROPIC_API_KEY` | `ANTHROPIC_API_KEY=... python main.py client=anthropic nav=claude-sonnet-4-5` |

`client=claude` is accepted as an alias, but `anthropic` is the canonical provider
name used internally and in output paths.

API keys can also be set in `conf/config.yaml` under `openai.api_key`,
`google.api_key`, or `anthropic.api_key`.

## Config keys

| Key | Default | Description |
|---|---|---|
| `env` | `mujoco` | Backend. Supported value: `mujoco`. |
| `nav` | `human` | `human`, `auto` (currently for `e6/e7`), or a model name for the selected `client`. |
| `client` | `ollama` | Navigation provider: `ollama`, `openai`, `google`, or `anthropic`. `claude` is accepted as an alias for `anthropic`. |
| `print_reasoning` | `false` | Stream thinking tokens (Ollama only). |
| `distractors` | `0` | Global distractor knob used by experiments that opt into it. `e5`/`e6`/`e7` randomize distractor count per run in `[1, 6]`. |
| `max_steps` | `20` | Maximum LLM steps per run. |
| `third_person` | `false` | Use third-person camera instead of first-person. |
| `camera_height` | `1.0` | Scale the first-person camera eye height from `0.0` (ground level) to `1.0` (current default). Third-person view is unaffected. |
| `random_wall_color` | `false` | For `e1/e2/e3/e5/e6/e7`, sample one non-agent wall color per run and apply it to all room walls. |
| `distractor_color_mode` | `allow_duplicates` | `e5/e6/e7` distractor color policy. `allow_agent_color`: distractors may use ego color (no duplicates). `exclude_agent_color`: distractors must differ from ego (no duplicates). `allow_duplicates`: distractors are sampled with replacement (duplicates allowed, ego color allowed). |
| `e7_random_cube_heights` | `false` | `e7`-only behavior toggle. When `true`, each sampled cube base becomes a same-color vertical stack with random height in `[1, 5]`; when `false`, each base is a single cube. The `e7` preset sets this to `true` by default. |
| `save_dataset` | `false` | Dataset export switch. Currently active for `experiment=e6/e7` with `nav=auto`; writes `Mirror_Video_Dataset/` with RGB frames plus per-episode JSON metadata. |
| `history_steps` | `1` | Number of previous steps included in the prompt. |
| `experiment_id` | `null` | Internal experiment id set by `+experiment=e1/e2/e3/e4/e5/e6/e7`. |
| `num_runs` | `1` | Number of runs. `e6/e7` presets set this to `100`. |
| `ollama.host` | `http://localhost:11434` | Ollama endpoint. |
| `openai.api_key` | `""` | OpenAI API key fallback (prefer `OPENAI_API_KEY` env var). |
| `google.api_key` | `""` | Google API key fallback (prefer `GEMINI_API_KEY` env var). |
| `anthropic.api_key` | `""` | Anthropic API key fallback (prefer `ANTHROPIC_API_KEY` env var). |
| `mujoco.headless` | `false` | Offscreen MuJoCo when `true`; live OpenCV window when `false` (default). |
| `outputs.dir` | `outputs` | Root directory for all generated artifacts. |
| `outputs.log_file` | `llm_history.txt` | Per-run log filename written inside the run directory. |
| `outputs.json_export` | `true` | Write per-run JSON and experiment summary JSON. |
| `outputs.heatmap` | `true` | Write per-run heatmap PNGs for supported room-style scenes. |
| `outputs.save_obs` | `false` | Save JPG observations after each manual key press. |

## Output layout

Generated files are grouped by backend, provider, and navigation model. For
`e1` through `e4`, the experiment id is part of the root output path. Single
runs plus `e5`/`e6`/`e7` keep the legacy layout.

```text
outputs/
  <env>/
    <experiment_id>/         # e1-e4 only
      <client>/
        <nav_slug>/
          obs/
          runs/
    <client>/                # single-run and e5/e6/e7
      <nav_slug>/
        obs/
        runs/
          <experiment_id>/   # e5/e6/e7 only
```

Examples:

- `outputs/mujoco/ollama/gemma3_4b/runs/run01.json`
- `outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json`
- `outputs/mujoco/openai/gpt-5/runs/run01.json`
- `outputs/mujoco/e4/google/gemini-2.5-flash/runs/experiment_summary.json`
- `outputs/mujoco/ollama/gemma3_4b/runs/e5/run01.json`
- `outputs/mujoco/ollama/gemma3_4b/runs/e6/run01.json`
- `outputs/mujoco/ollama/gemma3_4b/runs/e7/run01.json`
- `outputs/mujoco/anthropic/claude-sonnet-4-5/runs/run01.json`
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
