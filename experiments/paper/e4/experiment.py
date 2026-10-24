"""Paper E4: cube selection with moving distractor robots."""
from experiments.shared.types import (
    ExperimentSpec as _ExperimentSpec,
    ExperimentRuntimeOptions as _ExperimentRuntimeOptions,
)
from experiments.shared import distractors as _distractors

EXPERIMENT = _ExperimentSpec(
    experiment_id="e4",
    scene_id="mirror_room_robot_distractors",
    task_id="cube_task",
    num_runs_default=7,
    default_runtime_options=_ExperimentRuntimeOptions(),
    scene_template=_distractors._resolve_scene_template(_ExperimentRuntimeOptions()),
    scene_template_factory=_distractors._resolve_scene_template,
    run_setup_factory=_distractors._initialize_run,
    backend_overrides_factory=_distractors._resolve_backend_overrides,
)
