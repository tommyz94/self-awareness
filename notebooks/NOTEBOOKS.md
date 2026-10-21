# Paper figure sources

`results_analysis.ipynb` retains the original notebook and adds Cosmos to the
five final figures (cells 9–13). Run the setup cell (cell 2) and those five cells
for the final plots. Other cells are historical exploratory/table code, not
part of the batch command below. Source notebooks contain no saved outputs.

## Installation and inputs

Python 3.11 is the tested version. Plotting has a standalone dependency profile;
no simulator, provider SDK, model call or evaluator is needed:

```sh
python -m pip install -r requirements-analysis.txt
```

Supply your aggregate CSV separately. A code checkout does not include raw
runs, aggregate results, the provisional Cosmos JSON, or exported figures.
The current local inputs are the original 40-row aggregate CSV (eight models,
21 episodes per model/condition) and a separate Cosmos transcription from
Table 1, page 10 of `Self_Awereness (3).pdf` (seven episodes per condition).
The latter holds rounded, provisional means and standard errors, not new
metric calculations. Original inputs are never modified.

With a complete CSV containing all nine models, from the repository root:

```sh
python notebooks/plot_paper_results.py --csv /path/to/aggregates.csv
```

While using the existing local eight-model CSV and the local Cosmos overlay:

```sh
python notebooks/plot_paper_results.py \
  --csv final_outputs/reports/result_trimmed.csv \
  --reported-cosmos notebooks/plot_sources/cosmos_reported_metrics.json
```

Alternatively pass the original CSV and a Cosmos-only aggregate CSV together
after `--csv`. Duplicate model/condition rows are rejected. A precise CSV row
takes precedence over a matching rounded overlay row.

CSV identity columns are `model_name`, `experiment_id`, and `total_runs`.
Core metric columns are `TSA_mean`, `TTD_mean`, `MCR_mean`, `MTATO_mean`,
`CAAL_mean`, and `CONFABULATION_mean`. Conditions retain implementation IDs
`e1/e2/e3/e5/e7`, displayed as paper E1–E5. The two auxiliary plots also use
`TSA-COMPLETE_mean` and `ident_confusion_counts_json` from the original CSV.
For Cosmos these auxiliary fields may be empty until approved aggregates
arrive. The optional overlay has a `source` description and a `rows` list with
the same identity/core metric fields; JSON null preserves undefined values.

For notebook use, set `PAPER_PLOT_CSV` and optionally `PAPER_PLOT_COSMOS` in the
kernel environment, or edit the setup cell. Defaults use the local paths above
when present. Paths are resolved from the repository, not a user's home folder.
Run the kernel from within the checkout. The source-only archive is retained
for comparison at `archive/results_analysis_legacy_source.ipynb`.

## Exports and original appearance

The batch command executes the notebook's final plotting cells directly and
writes five figures (radar, TSA/CAAL scatter, core heatmap, completion gap,
and E5 attribution decomposition), each as PDF, SVG and PNG, beneath
`notebooks/paper_assets/all_models/`. `--output-dir` changes that destination.
The adjacent generated `figure_manifest.json` records input hashes, model
cohorts and plotting policy. Generated exports remain ignored by Git.

The original eight models retain their colors, markers, relative order,
fonts and figure layouts. Cosmos is appended with its own color and marker.
Original display transforms remain: inverse min–max TTD on radars,
per-condition inverse min–max TTD on the core heatmap, and percentile-based
TTD marker sizes on the scatter. Adding observations can affect these
data-dependent scales. The original CR transform and chance references
remain unchanged. No new error bars or reference lines have been added.

## Metric boundary and missing values

No evaluator or raw-episode scan runs during plotting. Undefined MTATO for
Cosmos is retained as undefined, not replaced with zero. Its unavailable
completion-conditioned success and E5 attribution counts are shown as N/A;
they cannot be inferred from the six core means. Some N/A values can remain
mathematically undefined even after the final evaluator arrives, such as
completion-conditioned success when no episodes emit `done`.

The collaborator's final evaluator and conventions are still pending. Once
approved aggregates arrive, replace the provisional inputs and regenerate
figures; do not change metric definitions inside plotting code.

## Offline checks

After installing the plotting profile and pytest:

```sh
python -m pytest -q tests/test_paper_plots.py
```

Checks use synthetic aggregates, retain original style/transform definitions,
and render all five figures without original data. Full local notebook
previews and the unchanged original notebook with outputs are preserved
locally, outside the source commit.
