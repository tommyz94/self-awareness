"""Discover the five official experiments; import prototypes only on request."""
from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

from experiments.naming import resolve_experiment_selection
from experiments.shared.types import ExperimentSpec

_PAPER_DIR = Path(__file__).resolve().parent / "paper"
_EXPERIMENTAL_DIR = Path(__file__).resolve().parent / "experimental"
_CACHE: dict[str, ExperimentSpec] | None = None
_EXPERIMENTAL_CACHE: dict[str, ExperimentSpec] = {}


def invalidate_experiment_cache() -> None:
    global _CACHE
    _CACHE = None
    _EXPERIMENTAL_CACHE.clear()


def _load_experiment_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def _load_plugin(module_name: str) -> ExperimentSpec:
    module = _load_experiment_module(module_name)
    exports = [name for name in vars(module)
               if not name.startswith("_") and name != "annotations"]
    if "EXPERIMENT" not in exports:
        raise ValueError(f"{module_name} must export EXPERIMENT: ExperimentSpec")
    if exports != ["EXPERIMENT"]:
        raise ValueError(f"{module_name} must export exactly one public symbol: EXPERIMENT")
    experiment = module.EXPERIMENT
    if not isinstance(experiment, ExperimentSpec):
        raise TypeError(f"{module_name}.EXPERIMENT must be an ExperimentSpec")
    return experiment


def discover_experiments(*, force_reload=False, include_experimental=False) -> dict[str, ExperimentSpec]:
    global _CACHE
    if force_reload:
        invalidate_experiment_cache()
    if _CACHE is None:
        discovered = {}
        for candidate in sorted(_PAPER_DIR.iterdir()):
            if candidate.name.startswith("_") or not (candidate / "experiment.py").is_file():
                continue
            experiment = _load_plugin(f"experiments.paper.{candidate.name}.experiment")
            if experiment.experiment_id in discovered:
                raise ValueError(f"Duplicate experiment_id={experiment.experiment_id!r}")
            discovered[experiment.experiment_id] = experiment
        _CACHE = discovered
    result = dict(_CACHE)
    if include_experimental:
        for candidate in sorted(_EXPERIMENTAL_DIR.iterdir()):
            if not candidate.name.startswith("_") and (candidate / "experiment.py").is_file():
                key = f"experimental/{candidate.name}"
                result[key] = get_experiment(key)
    return result


def get_experiment(experiment_id: str) -> ExperimentSpec:
    canonical, _ = resolve_experiment_selection(experiment_id)
    if canonical and canonical.startswith("experimental/"):
        if canonical not in _EXPERIMENTAL_CACHE:
            name = canonical.removeprefix("experimental/")
            experiment = _load_plugin(f"experiments.experimental.{name}.experiment")
            if experiment.experiment_id != canonical:
                raise ValueError(f"Experimental plugin ID disagrees with {canonical!r}")
            _EXPERIMENTAL_CACHE[canonical] = experiment
        return _EXPERIMENTAL_CACHE[canonical]
    experiments = discover_experiments()
    if canonical not in experiments:
        raise ValueError(f"Unknown experiment_id={experiment_id!r}. Known: {', '.join(experiments)}")
    return experiments[canonical]


def get_registry(*, include_experimental=False) -> dict[str, ExperimentSpec]:
    return discover_experiments(include_experimental=include_experimental)
