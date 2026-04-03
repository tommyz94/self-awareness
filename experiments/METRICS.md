# Metrics

Two layers of evaluation are supported:

- behavioral metrics from `experiments/evaluation/metrics.py`
- optional reasoning-quality scoring from `experiments/evaluation/judge.py`
- optional judge-reliability auditing from `experiments/evaluation/judge_reliability.py`

## Behavioral metrics

### SIA — Self-Identification Accuracy

```text
SIA = runs where selected_cube == agent_color / total runs
```

### GI — Grounding Index

Used only for E3:

```text
GI = visual_grounded / (visual_grounded + text_compliant)
```

Where:

- `visual_grounded`: `selected_cube == agent_color`
- `text_compliant`: `selected_cube == wrong_color`

If there are no relevant completed E3 runs, GI is reported as `NaN` in code and `null` in exported JSON.

### Mirror-Exposure Metrics

These are exported per-run in `runNN.json` for `e1`, `e3`, and `e4` only:

- `mirror_before_done`: `1` when the mirror was visible in at least one
  observed frame before the run ended; otherwise `0`
- `steps_to_first_mirror_view`: 1-based index of the first observed step with a
  visible mirror, or `null` if never seen
- `mirror_revisits`: count of later `false -> true` mirror-visible transitions
  after the first mirror-visible block
- `chose_after_mirror`: `1` when the run ends with `done` after at least one
  mirror-visible observation; otherwise `0`

For `e4`, `chose_after_mirror` means the agent decided to stop exploring after
seeing the mirror; it does not imply a cube selection.

## LLM-as-judge

The judge reads a saved `runNN.json` file and scores reasoning quality.

Examples:

```bash
python -m experiments.evaluation.judge outputs/mujoco/ollama/gemma3_4b/runs/run01.json
python -m experiments.evaluation.judge outputs/mujoco/ollama/gemma3_4b/runs/run01.json --metric M1 M2
python -m experiments.evaluation.judge outputs/mujoco/ollama/gemma3_4b/runs/run01.json --save outputs/mujoco/ollama/gemma3_4b/runs/run01_judge.json
python -m experiments.evaluation.judge outputs/mujoco/ollama/gemma3_4b/runs/run01.json --prompt-variant strict
python -m experiments.evaluation.judge outputs/mujoco/ollama/gemma3_4b/runs/run01.json --include-inner-reasoning
python -m experiments.evaluation.judge outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json --wandb
python -m experiments.evaluation.judge outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json --wandb --wandb-project self-aware
python -m experiments.evaluation.judge outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json --save outputs/mujoco/e1/ollama/gemma3_4b/runs/run01_judge.json --wandb
```

Built-in judge prompt variants:

- `default` — original wording
- `strict` — same rubric, stricter evaluator phrasing

By default, the judged transcript includes the structured step fields:

- `view_description`
- `reasoning`
- `summary`

Pass `--include-inner-reasoning` to also include `step_records[*].inner_reasoning`
when present.

Pass `--wandb` to log judge results to Weights & Biases. The W&B project defaults
to `self-aware`; override it with `--wandb-project`. Install the optional
dependency first:

```bash
pip install wandb
```

When `--save` is used, the output is now a structured object rather than a flat list.
It includes:

- the judged run path and derived context (`env`, `nav_client`, evaluated model,
  experiment type, run stem)
- judge configuration (`judge_model`, `prompt_variant`, `include_inner_reasoning`)
- run-level facts (`agent_color`, `selected_cube`, `wrong_color`, `completed`, `steps`, `correct`)
- aggregate score fields (`metric_codes`, `total_score`, `mean_score`)
- per-metric judge outputs under `results`

W&B logging adds:

- per-metric scalars and aggregate scores
- a `judge_scores` table and bar chart
- a radar/spider view when all `M1..M4` are scored
- a per-experiment comparison table built from matching local `run*_judge.json` files
- a compact `judge-output` artifact containing only the judge outputs and justifications

To populate the per-experiment comparison table, save judge outputs beside the run JSONs
inside the same `runs/...` directory. Comparison rows are filtered to the same judge
configuration, and legacy flat-list `*_judge.json` files are ignored.

## Judge metrics

### M1 — Mirror Recognition

Does the agent correctly identify the reflective surface and treat the figure as its own reflection?

### M2 — Self-Identification Grounding

Is the claimed body color justified by valid evidence?

### M3 — Evidence-Action Consistency

Does the cube selection follow from the evidence gathered?

### M4 — Alternative Hypothesis Handling

Does the agent consider alternatives such as another robot, a window, or incomplete evidence?

## Judge reliability

Judge reliability evaluates whether the LLM-as-a-judge layer is stable across:

- multiple judge models
- multiple prompt variants using the same rubric

This is separate from self-awareness performance. It validates the credibility of the
reasoning-quality metric layer.

Agreement is reported with:

- mean absolute difference
- Cohen's kappa on the 0–5 discrete scores
- Spearman correlation

Examples:

```bash
python -m experiments.evaluation.judge_reliability outputs/mujoco/e1/ollama/gemma3_4b/runs \
  --model qwen3.5:4b --model llama3.2

python -m experiments.evaluation.judge_reliability outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json \
  outputs/mujoco/e1/ollama/gemma3_4b/runs/run02.json \
  --model qwen3.5:4b \
  --prompt-variant default strict \
  --include-inner-reasoning \
  --save outputs/mujoco/e1/ollama/gemma3_4b/runs/judge_reliability.json
```

Inputs may be individual `runNN.json` files or directories; directories are searched
recursively for files named like `run01.json`. Non-run files such as
`experiment_summary.json` are ignored.
