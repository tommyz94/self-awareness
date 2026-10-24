from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from experiments.evaluation.metrics import RunOutcome


def _extract_task_output_default(
    step_records: list[dict[str, Any]],
    completed: bool,
    run_setup: Optional["RunSetup"] = None,
) -> dict[str, Any]:
    del step_records, completed, run_setup
    return {}


@dataclass(frozen=True)
class TaskCapabilities:
    supports_heatmap: bool = False
    supports_judge: bool = False
    supports_summary_metrics: bool = False
    extract_task_output: Callable[[list[dict[str, Any]], bool, Optional["RunSetup"]], dict[str, Any]] = _extract_task_output_default
    summarize_experiment: Optional[Callable[[list["RunOutcome"], str], Optional[dict[str, Any]]]] = None


@dataclass
class SceneObjectSpec:
    object_id: str
    kind: str
    position: Tuple[float, float]
    color: Optional[str] = None
    size: Optional[Tuple[float, float, float]] = None
    asset_id: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def type(self) -> str:
        return self.kind

    def to_legacy_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "object_id": self.object_id,
            "type": self.kind,
            "position": self.position,
        }
        if self.color is not None:
            data["color"] = self.color
        if self.size is not None:
            data["size"] = self.size
        if self.asset_id is not None:
            data["asset_id"] = self.asset_id
        data.update(self.metadata)
        return data

    def __getitem__(self, key: str) -> Any:
        return self.to_legacy_dict()[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.to_legacy_dict().get(key, default)


def coerce_scene_object(value: SceneObjectSpec | dict[str, Any]) -> SceneObjectSpec:
    if isinstance(value, SceneObjectSpec):
        return value

    metadata = {
        k: v
        for k, v in value.items()
        if k not in {"object_id", "type", "kind", "position", "color", "size", "asset_id"}
    }
    object_id = str(
        value.get("object_id")
        or value.get("id")
        or f"{value.get('type', value.get('kind', 'object'))}_{value.get('color', 'item')}"
    )
    return SceneObjectSpec(
        object_id=object_id,
        kind=str(value.get("kind", value.get("type", "object"))),
        position=tuple(value["position"]),
        color=value.get("color"),
        size=tuple(value["size"]) if value.get("size") is not None else None,
        asset_id=value.get("asset_id"),
        metadata=metadata,
    )


@dataclass
class SceneSpec:
    scene_id: str
    mirror_enabled: bool
    layout_kind: str = "two_room"
    length: float = 15.0
    width: float = 15.0
    height: float = 7.0
    wall_thickness: float = 0.2
    spawn_pos: Tuple[float, float] = (-10.0, 2.5)
    spawn_yaw: float = 0.0
    mirror_plane_x: Optional[float] = None
    mirror_aperture_mode: str = "window"
    mirror_window_z: Optional[Tuple[float, float]] = (1.0, 3.0)
    turn_step_deg: float = 90.0
    wall_color_rgba: Optional[Tuple[float, float, float, float]] = None
    objects: List[SceneObjectSpec | dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.objects = [coerce_scene_object(obj) for obj in self.objects]
        if self.mirror_enabled and self.mirror_plane_x is None:
            self.mirror_plane_x = 12.0 if self.layout_kind == "corridor" else -self.wall_thickness / 2
        if self.mirror_enabled and self.mirror_aperture_mode == "window" and self.mirror_window_z is None:
            self.mirror_window_z = (1.0, 3.0)
        if not self.mirror_enabled:
            self.mirror_aperture_mode = "none"
            self.mirror_window_z = None
        if self.turn_step_deg <= 0:
            raise ValueError("turn_step_deg must be > 0.")
        if self.wall_color_rgba is not None:
            self.wall_color_rgba = tuple(float(v) for v in self.wall_color_rgba)
            if len(self.wall_color_rgba) != 4:
                raise ValueError("wall_color_rgba must contain 4 values (r, g, b, a).")
            if any(v < 0.0 or v > 1.0 for v in self.wall_color_rgba):
                raise ValueError("wall_color_rgba values must be in [0, 1].")


def scene_geometry_signature(scene_spec: SceneSpec) -> tuple[Any, ...]:
    return (
        scene_spec.scene_id,
        scene_spec.layout_kind,
        scene_spec.mirror_enabled,
        scene_spec.length,
        scene_spec.width,
        scene_spec.height,
        scene_spec.wall_thickness,
        scene_spec.mirror_plane_x,
        scene_spec.mirror_aperture_mode,
        scene_spec.turn_step_deg,
    )


@dataclass
class RunSetup:
    agent_color_name: str
    scene_spec: SceneSpec
    wrong_color: Optional[str] = None
    cube_positions: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.cube_positions:
            self.cube_positions = {
                obj.color: tuple(obj.position)
                for obj in self.scene_spec.objects
                if obj.kind == "cube" and obj.color is not None
            }


@dataclass(frozen=True)
class ExperimentRuntimeOptions:
    distractors: int = 0
    camera_height: float = 1.0
    use_mirror: bool = True
    random_wall_color: bool = False
    distractor_color_mode: str = "allow_duplicates"
    random_cube_heights: bool = False
    body_style: str = "default"
    self_perception_mode: str = "color"


@dataclass
class MuJoCoSceneBuildResult:
    xml: str
    robot_geom_names: list[str]
    mirror_geom_names: list[str]


@dataclass(frozen=True)
class BackendOverrides:
    mujoco_scene_builder: Optional[Callable[[SceneSpec, int, int], MuJoCoSceneBuildResult]] = None
    mujoco_reset_applier: Optional[Callable[[Any, RunSetup], None]] = None
    mujoco_step_applier: Optional[Callable[[Any, str], None]] = None


@dataclass
class TaskSpec:
    task_id: str
    build_instruction: Callable[[RunSetup], str]
    response_schema: Dict[str, Any]
    parse_reply: Callable[[str, Optional[str]], tuple]
    build_system_prompt: Optional[Callable[..., str]] = None
    build_user_prompt: Optional[Callable[..., str]] = None
    capabilities: TaskCapabilities = field(default_factory=TaskCapabilities)


@dataclass
class ExperimentSpec:
    experiment_id: str
    scene_id: str
    task_id: str
    num_runs_default: int
    default_runtime_options: ExperimentRuntimeOptions = field(default_factory=ExperimentRuntimeOptions)
    scene_template: Optional[SceneSpec] = None
    build_run_setup: Optional[Callable[[int], RunSetup]] = None
    scene_template_factory: Optional[Callable[[ExperimentRuntimeOptions], SceneSpec]] = None
    run_setup_factory: Optional[Callable[[int, ExperimentRuntimeOptions, SceneSpec], RunSetup]] = None
    task_factory: Optional[Callable[[], TaskSpec]] = None
    backend_overrides_factory: Optional[Callable[..., BackendOverrides]] = None

    def resolve_task(self) -> TaskSpec:
        if self.task_factory is not None:
            return self.task_factory()
        from experiments.tasks import get_task
        return get_task(self.task_id)

    def resolve_scene_template(
        self,
        runtime_options: Optional[ExperimentRuntimeOptions] = None,
    ) -> SceneSpec:
        options = runtime_options or self.default_runtime_options
        if self.scene_template_factory is not None:
            return self.scene_template_factory(options)
        if self.scene_template is None:
            raise ValueError(
                f"Experiment {self.experiment_id!r} does not define a scene template."
            )
        return self.scene_template

    def initialize_run(
        self,
        run_id: int,
        runtime_options: Optional[ExperimentRuntimeOptions] = None,
    ) -> RunSetup:
        options = runtime_options or self.default_runtime_options
        scene_template = self.resolve_scene_template(options)
        if self.run_setup_factory is not None:
            return self.run_setup_factory(run_id, options, scene_template)
        if self.build_run_setup is not None:
            return self.build_run_setup(run_id)
        raise ValueError(
            f"Experiment {self.experiment_id!r} does not define a run initializer."
        )

    def resolve_backend_overrides(
        self,
        runtime_options: Optional[ExperimentRuntimeOptions] = None,
        scene_template: Optional[SceneSpec] = None,
    ) -> BackendOverrides:
        if self.backend_overrides_factory is not None:
            options = runtime_options or self.default_runtime_options
            resolved_scene = scene_template or self.resolve_scene_template(options)
            try:
                return self.backend_overrides_factory(options, resolved_scene)
            except TypeError:
                return self.backend_overrides_factory()
        return BackendOverrides()

    @property
    def scene_spec(self) -> SceneSpec:
        return self.resolve_scene_template()

    @property
    def run_initializer(self) -> Callable[[int], RunSetup]:
        return lambda run_id: self.initialize_run(run_id, self.default_runtime_options)

    @property
    def mirror_enabled(self) -> bool:
        return self.resolve_scene_template().mirror_enabled
