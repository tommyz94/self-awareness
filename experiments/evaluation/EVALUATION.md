# `experiments/evaluation/`

Offline evaluation and analysis tools for completed runs.

## Files

### `metrics.py`

Behavioral metrics and experiment-summary helpers.

### `judge.py`

LLM-as-judge entrypoint for scoring a saved run transcript.

### `judge_reliability.py`

Repeated-judge analysis for checking how stable the judge outputs are across
multiple evaluations.

### `judge_metrics.py`

Definitions for the judge scoring rubric items.

### `judge_prompts.py`

Prompt variants used by the judge pipeline.

## Notes

- This folder is canonical.
- The old flat files in `experiments/` are compatibility wrappers around these
  modules.
