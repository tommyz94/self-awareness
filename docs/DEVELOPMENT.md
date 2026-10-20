# Development and repository hygiene

## Local checks

Use Python 3.11 and install `requirements.txt` plus `requirements-dev.txt`.
Run `python -m pytest -q` for the offline suite. Provider calls in tests are
mocked. GPU rendering tests may skip when the optional platform support is
unavailable; live paid model sweeps are not a unit-test requirement.

Validate a public preset with `python main.py experiment=paper/e4 --cfg job`.
This composes configuration without creating a simulator or calling an API.
Run matrix launchers without `--launch` to preview commands.

## What belongs in Git

Track source, tests, presets, documentation, analysis notebooks and small
curated manifests/reports. Generated CSV/PDF files are not globally ignored:
a reproducibility table or reference figure can be deliberately versioned.

Raw observations, model outputs, local run roots, caches, credentials,
virtual environments, and large generated figure directories are ignored.
Research extras remain source code, with named presets under
`conf/experiment/extras/`. Use `.git/info/exclude` for personal exclusions;
do not hide shared experiments or tests in a shared ignore file.

Ignoring a path does not untrack an already committed file. Existing tracked
avatar assets and caches are preserved by this change. Review any future
index cleanup separately; do not delete original research files to prepare
a code release.

## Paper naming

Keep implementation IDs stable. Public presets and new JSON metadata provide
the paper mapping; never bulk-replace old e4/e5/e7 strings in saved data.
See `EXPERIMENTS.md` and `METRIC_CONVENTIONS.md` before changing evaluation.

## Optional assets and dependencies

The standard paper scenes use MuJoCo humanoids. Clothed avatars are required
only for embodiment experiments. HF/Qwen-Drive support is experimental and
requires its upstream model-specific packages; installing the standard
benchmark does not require those weights.

This repository currently contains pre-existing local research changes.
The paper-alignment commit intentionally excludes local provider extensions,
front-identity/mirror-side-choice prototypes, metric tooling and generated
reports. Its four shared extra presets work without these local additions.
Review the actual staged diff before committing; do not use a blanket
`git add .` to prepare a release with unreviewed data and assets.
