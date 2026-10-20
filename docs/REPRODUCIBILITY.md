# Reproducibility and original data

## Evaluation sets

The original authoritative selection is
`final_outputs/reports/metric_manifest.json`: 840 episodes, eight models,
five conditions, three seeds per model-condition and seven episodes per seed.
Its condition IDs are historical implementation IDs: e1/e2/e3/e5/e7.

The separate additional Cosmos set is
`outputs_cosmos_reason2_q4_reviewer_seed_3315837970_parallel2/`:
35 episodes, one seed, seven episodes per condition. Its recorded model ID is
`mradermacher/Cosmos-Reason2-8B-GGUF:Q4_K_M`. The adapter name `openrouter`
does not by itself establish where the inference server ran.

The primary manifest does not include Cosmos. Do not regenerate it over a
mixed directory or silently add the additional runs to the primary cohort.
Exact model IDs/routes are recorded in the logs and manuscript; primary
families are Claude Sonnet 4.6, GPT-5.1, Gemini 2.5 Pro/Flash,
Gemma 4 26B/31B, Qwen 3.6 Plus, and Ministral 3 14B.

## Running new episodes

```sh
python main.py experiment=paper/e4 client=google nav=gemini-2.5-pro seed=123 mujoco.headless=true
```

This runs seven episodes for one seed. Replication of the primary protocol
requires three explicit seeds for each model-condition, not one arbitrary
21-episode invocation. API/model availability and stochastic decoding can
prevent bitwise reproduction of fresh trajectories.

The presets use first-person 640x480 RGB, one-step textual history,
100 maximum steps, standard humanoids, and the ten-color pool.
Scene builders retain the 110-degree egocentric field of view.
The providers currently default to temperature 0.2; the final evaluator and
provider settings still require reconciliation with the collaborator.

## Historical analysis

These local research tools retain the old implementation vocabulary. They
are not shipped by this alignment commit; review and publish them separately
with the final evaluator:

```sh
python scripts/dev/run_inventory_report.py final_outputs
python scripts/dev/metric_readiness_report.py final_outputs
python experiment.py final_outputs --csv-out /tmp/self-awareness-check.csv
```

The last command scans directories, not the authoritative manifest. It is
only a historical diagnostic; its counts need not equal the selected 840.
For exact checks, use a disposable view containing only manifest-selected
files. Never run `apply_metric_manifest.py` against original research data:
it deletes unselected run directories and may rewrite seed metadata.

The original manifest contains absolute local paths. A portable public
manifest and a documented dataset download are still needed for a final
data release. Current Git ignores prevent accidental inclusion of raw
runs, frames, and the private handoff archive; they do not delete them.

## Release gates

- Obtain and review the collaborator's final metric code.
- Reconcile the 840 primary and 35 additional run counts and metric rules.
- Publish a portable manifest and hashes, and define dataset access.
- Regenerate final figures/tables from one approved evaluator.
- Make the benchmark card and referenced causal audit available.
- Confirm author-approved licensing, the citation metadata, and a release tag.

The code alignment alone does not assert that these scientific release gates
have been completed.
