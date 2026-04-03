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

**Human navigation:**

```sh
python main.py env=mujoco nav=human mujoco.headless=false
```

Keys: `W` forward, `S` backward, `A`/`D` rotate by 30° in `e1/e2/e3/e5/e6/e7`; in `e4`, `A`/`D` are forward diagonals and `S` is disabled. `C` toggles camera.

**LLM navigation via Ollama:**

```sh
python main.py nav=gemma3:4b
```

Each step captures a JPG observation, sends it to the VLM, parses one JSON action, executes it, and records the run.

**Controlled experiments:**

```sh
python main.py experiment=e1 nav=gemma3:4b
python main.py experiment=e2 nav=gemma3:4b
python main.py experiment=e3 nav=gemma3:4b
python main.py experiment=e4 nav=gemma3:4b
python main.py experiment=e5 nav=gemma3:4b
python main.py experiment=e6
python main.py experiment=e6 save_dataset=true
python main.py experiment=e7
python main.py experiment=e7 save_dataset=true
python main.py experiment=e7 e7_random_cube_heights=false
```

| Experiment | Description |
|---|---|
| `e1` | Mirror present, agent color randomized each run. |
| `e2` | No mirror, agent color randomized each run. |
| `e3` | Mirror present plus misleading color text in the system prompt. |
| `e4` | Checkerboard corridor with latent per-episode variant sampling (`mirror` vs `long`). Geometry is `10x3` (mirror) or `20x3` (long, as two `10x3` halves). Actions are `w/a/d/done` only, with `a`/`d` as forward-diagonal moves (yaw fixed). On terminal row: `done` succeeds, while `w/a/d` ends the run as failure. |
| `e5` | Same mirrored cube task as `e1`, but with `1..6` colored humanoid distractor robots sampled randomly each run on a nearer mirror-facing row; distractor colors are random per robot (duplicates allowed), and cube/agent colors come from the shared 15-color pool. |
| `e6` | Same mirrored distractor-room layout as `e5`, but without cubes; each run samples `1..6` colored distractor humanoids (duplicates allowed) plus a random agent color from the shared 15-color pool. Preset defaults to `nav=auto`, executes 100 ego-agent actions per run, and waits 0.5s between actions when rendering live (`mujoco.headless=false`); in headless mode the wait is skipped. Optional `save_dataset=true` exports a per-episode RGB + JSON dataset in `Mirror_Video_Dataset/`. |
| `e7` | `e6` plus random distractor cubes: each run samples `1..6` distractor humanoids and `5..15` cube bases with colors from the same 15-color pool (duplicates allowed), placed across both room sides and reflected through the mirror aperture model. With `e7_random_cube_heights=true` (default in the preset), each base becomes a same-color vertical stack with random height `1..5`; set `e7_random_cube_heights=false` to keep only single cubes. Preset defaults to `nav=auto`, executes 100 ego-agent actions per run, and supports `save_dataset=true` with the same dataset schema as `e6`. |

In mirror-enabled two-room scenes, mirror content is visual-only: reflected cubes,
the mirrored humanoid, and `e5`/`e6`/`e7` mirrored distractors appear only when the real
object lies inside the active mirror opening.

Metrics printed after cube-task experiments:

- `SIA`: fraction of runs where `selected_cube == agent_color`
- `GI`: E3-only grounding index against the misleading prompt
- `Δ_mirror`: compare E1 and E2 manually from their SIA values

For non-cube tasks such as `e4`, `e6`, and `e7`, the simulator still saves run JSON, but it skips
room-style heatmaps and does not print SIA/GI because those metrics are
cube-specific. In `e6`/`e7`, `save_dataset=true` additionally saves episode RGB frames and
trajectory/action metadata into `Mirror_Video_Dataset/`, including per-step
`mirror_visible` and `ego_reflection_visible` flags plus `ego_reflection_seen`
at episode level.

**Human commands for experiments `e1`-`e7` (live MuJoCo):**

```sh
# e1
python main.py env=mujoco nav=human mujoco.headless=false experiment=e1 random_wall_color=true

# e2
python main.py env=mujoco nav=human mujoco.headless=false experiment=e2 random_wall_color=true

# e3
python main.py env=mujoco nav=human mujoco.headless=false experiment=e3 random_wall_color=true

# e4
python main.py env=mujoco nav=human mujoco.headless=false experiment=e4 

# e5
python main.py env=mujoco nav=human mujoco.headless=false experiment=e5 random_wall_color=true

# e6 (preset is auto by default, here forced to human)
python main.py env=mujoco nav=human mujoco.headless=false experiment=e6 random_wall_color=true

# e7 (preset is auto by default, here forced to human)
python main.py env=mujoco nav=human mujoco.headless=false experiment=e7 random_wall_color=true
```

Useful optional overrides:

```sh
# Random wall color in e1/e2/e3/e5/e6/e7
python main.py env=mujoco nav=human mujoco.headless=false experiment=e1 random_wall_color=true

# Third-person camera
python main.py env=mujoco nav=human mujoco.headless=false experiment=e5 third_person=true

# Save per-step JPG observations (manual navigation)
python main.py env=mujoco nav=human mujoco.headless=false experiment=e3 outputs.save_obs=true

# e7 cube stacks toggle
python main.py env=mujoco nav=human mujoco.headless=false experiment=e7 e7_random_cube_heights=false
```

In `e4`, the corridor variant is sampled randomly each episode; `use_mirror` does not force the variant. In `nav=human`, `F` maps to corridor `done`.
`save_dataset=true` is currently supported only for `experiment=e6/e7` with `nav=auto`.

## MuJoCo backend

MuJoCo renders an articulated humanoid loaded from `assets/mujoco/humanoid.xml`, while movement stays discrete and kinematic.

**Headless MuJoCo (optional, any platform):**

```sh
pip install mujoco pillow
python main.py env=mujoco nav=gemma3:4b
```

**MuJoCo with live RGB window:**

```sh
pip install mujoco pillow opencv-python
python main.py env=mujoco nav=gemma3:4b mujoco.headless=false
python main.py env=mujoco nav=human mujoco.headless=false
```

When `mujoco.headless=false`, MuJoCo can show a live RGB window during both human and LLM runs. When `mujoco.headless=true`, MuJoCo stays fully offscreen — no browser, virtual display, or GPU required.

On Linux, headless MuJoCo defaults to `MUJOCO_GL=egl`. If a cluster node lacks EGL support, override it with `MUJOCO_GL=osmesa`.

## Provider selection

The navigation VLM can use Ollama (default), OpenAI, Google Gemini, or Anthropic Claude:

```sh
# Ollama — local model, no API key needed
python main.py env=mujoco nav=gemma3:4b

# OpenAI Responses API
pip install "openai>=1.0.0"
OPENAI_API_KEY=sk-... python main.py client=openai nav=gpt-5 env=mujoco

# Google Gemini
pip install google-genai
GEMINI_API_KEY=... python main.py client=google nav=gemini-2.5-flash env=mujoco

# Anthropic Claude
pip install "anthropic>=0.68.0"
ANTHROPIC_API_KEY=... python main.py client=anthropic nav=claude-sonnet-4-5 env=mujoco
```

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
