from __future__ import annotations

import random as _random

from experiments.shared.scenes import (
    build_corridor_run_setup as _build_corridor_run_setup,
    build_corridor_scene as _build_corridor_scene,
)
from experiments.shared.types import (
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
    ExperimentSpec as _ExperimentSpec,
    SceneSpec as _SceneSpec,
)

_VARIANT_MIRROR = "mirror"
_VARIANT_LONG = "long"


def _sample_variant() -> str:
    return _random.choice((_VARIANT_MIRROR, _VARIANT_LONG))


def _expected_done_step(variant: str) -> int:
    return 10 if variant == _VARIANT_MIRROR else 20


def _build_scene_template(options: _ExperimentRuntimeOptions) -> _SceneSpec:
    del options
    # Keep a canonical template for environment boot; per-episode variants are
    # sampled in _build_run_setup.
    return _build_corridor_scene(use_mirror=True)


def _build_run_setup(
    run_id: int,
    _options: _ExperimentRuntimeOptions,
    _scene_template: _SceneSpec,
):
    del run_id, _options, _scene_template
    variant = _sample_variant()
    scene_spec = _build_corridor_scene(use_mirror=(variant == _VARIANT_MIRROR))
    run_setup = _build_corridor_run_setup(scene_spec)
    run_setup.metadata.update({
        "e4_variant": variant,
        "expected_done_step": _expected_done_step(variant),
    })
    return run_setup


EXPERIMENT = _ExperimentSpec(
    experiment_id="experimental/corridor",
    scene_id=_build_corridor_scene().scene_id,
    task_id="corridor_task",
    num_runs_default=10,
    scene_template_factory=_build_scene_template,
    run_setup_factory=_build_run_setup,
)
