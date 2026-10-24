from experiments.shared.types import TaskSpec
from experiments.tasks.cube import CUBE_TASK_SPEC
from experiments.tasks.explore import EXPLORE_TASK_SPEC

_TASK_REGISTRY = {
    "cube_task": CUBE_TASK_SPEC,
    "explore_task": EXPLORE_TASK_SPEC,
}
_EXPERIMENTAL_TASKS = {
    "corridor_task": ("corridor", "CORRIDOR_TASK_SPEC"),
    "front_identity_task": ("front_identity", "FRONT_IDENTITY_TASK_SPEC"),
    "mirror_side_choice_task": ("mirror_side_choice", "MIRROR_SIDE_CHOICE_TASK_SPEC"),
    "self_description_explore_task": ("self_description_explore", "SELF_DESCRIPTION_EXPLORE_TASK_SPEC"),
    "self_perception_task": ("self_perception", "SELF_PERCEPTION_TASK_SPEC"),
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in _TASK_REGISTRY and task_id in _EXPERIMENTAL_TASKS:
        from importlib import import_module
        module, symbol = _EXPERIMENTAL_TASKS[task_id]
        _TASK_REGISTRY[task_id] = getattr(import_module(f"experiments.tasks.{module}"), symbol)
    if task_id not in _TASK_REGISTRY:
        known = ", ".join(sorted(_TASK_REGISTRY))
        raise ValueError(f"Unknown task_id={task_id!r}. Known: {known}")
    return _TASK_REGISTRY[task_id]


def resolve_task(*, task_id: str | None = None, experiment_id: str | None = None) -> TaskSpec:
    if experiment_id is not None:
        from experiments import get_experiment

        return get_experiment(experiment_id).resolve_task()
    if task_id is None:
        raise ValueError("resolve_task() requires task_id or experiment_id")
    return get_task(task_id)
