from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType
from typing import Dict, Optional

from experiments.specs.types import ExperimentSpec

_CACHE: Optional[Dict[str, ExperimentSpec]] = None
_SPECS_DIR = Path(__file__).resolve().parent


def invalidate_experiment_cache() -> None:
    global _CACHE
    _CACHE = None


def _load_experiment_module(module_name: str) -> ModuleType:
    return importlib.import_module(module_name)


def discover_experiments(*, force_reload: bool = False) -> Dict[str, ExperimentSpec]:
    global _CACHE
    if _CACHE is not None and not force_reload:
        return dict(_CACHE)

    discovered: Dict[str, ExperimentSpec] = {}
    for candidate in sorted(_SPECS_DIR.iterdir()):
        if not candidate.is_dir() or candidate.name.startswith("_"):
            continue
        experiment_file = candidate / "experiment.py"
        if not experiment_file.exists():
            continue

        module_name = f"experiments.specs.{candidate.name}.experiment"
        module = _load_experiment_module(module_name)
        exported_names = [
            name
            for name in vars(module)
            if not name.startswith("_")
        ]
        if "EXPERIMENT" not in vars(module):
            raise ValueError(
                f"{module_name} must export EXPERIMENT: ExperimentSpec"
            )
        meaningful_exports = [name for name in exported_names if name != "annotations"]
        if any(name != "EXPERIMENT" for name in meaningful_exports):
            raise ValueError(
                f"{module_name} must export exactly one public symbol: EXPERIMENT"
            )

        experiment = module.EXPERIMENT
        if not isinstance(experiment, ExperimentSpec):
            raise TypeError(
                f"{module_name}.EXPERIMENT must be an ExperimentSpec, got {type(experiment)!r}"
            )
        if experiment.experiment_id in discovered:
            previous = discovered[experiment.experiment_id]
            raise ValueError(
                f"Duplicate experiment_id={experiment.experiment_id!r} discovered in "
                f"{module_name} and {previous!r}"
            )
        discovered[experiment.experiment_id] = experiment

    _CACHE = discovered
    return dict(discovered)


def get_discovered_experiment(experiment_id: str) -> ExperimentSpec:
    experiments = discover_experiments()
    if experiment_id not in experiments:
        known = ", ".join(sorted(experiments))
        raise ValueError(
            f"Unknown experiment_id={experiment_id!r}. Known: {known}"
        )
    return experiments[experiment_id]
