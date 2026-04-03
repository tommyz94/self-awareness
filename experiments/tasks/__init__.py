from experiments.specs.types import TaskSpec
from experiments.tasks.cube import CUBE_TASK_SPEC
from experiments.tasks.corridor import CORRIDOR_TASK_SPEC
from experiments.tasks.explore import EXPLORE_TASK_SPEC

_TASK_REGISTRY = {
    "cube_task": CUBE_TASK_SPEC,
    "corridor_task": CORRIDOR_TASK_SPEC,
    "explore_task": EXPLORE_TASK_SPEC,
}


def get_task(task_id: str) -> TaskSpec:
    if task_id not in _TASK_REGISTRY:
        known = ", ".join(sorted(_TASK_REGISTRY))
        raise ValueError(f"Unknown task_id={task_id!r}. Known: {known}")
    return _TASK_REGISTRY[task_id]


def resolve_task(*, task_id: str | None = None, experiment_id: str | None = None) -> TaskSpec:
    if experiment_id is not None:
        from experiments.specs import get_experiment

        return get_experiment(experiment_id).resolve_task()
    if task_id is None:
        raise ValueError("resolve_task() requires task_id or experiment_id")
    return get_task(task_id)
