# `conf/`

Hydra configuration root. Select presets with `experiment=paper/e1` through
`paper/e5`, or `experiment=extras/<name>`. Bare numeric presets retain the
historical implementation IDs for compatibility.

`config.yaml` defines provider settings, observation and navigation controls,
rendering, outputs, and default runtime options. It must not contain API keys.
`experiment/` contains thin preset selectors; experiment logic remains in
`experiments/specs/`.

The public paper presets set seven episodes per seed and the benchmark runtime
settings. Extras retain their own defaults. See
`experiment/EXPERIMENT.md` and `docs/EXPERIMENTS.md`.
