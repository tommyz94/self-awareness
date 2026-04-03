from __future__ import annotations

from pathlib import Path
from types import ModuleType

import pytest

from experiments.specs.loader import discover_experiments, invalidate_experiment_cache
from experiments.specs.types import (
    BackendOverrides,
    ExperimentSpec,
    MuJoCoSceneBuildResult,
    RunSetup,
    SceneObjectSpec,
    SceneSpec,
    TaskCapabilities,
    TaskSpec,
)


@pytest.fixture(autouse=True)
def _reset_experiment_loader_cache():
    invalidate_experiment_cache()
    yield
    invalidate_experiment_cache()


def _simple_scene(scene_id: str = "test_scene") -> SceneSpec:
    return SceneSpec(scene_id=scene_id, mirror_enabled=True)


def _simple_run_setup(run_id: int) -> RunSetup:
    del run_id
    return RunSetup(agent_color_name="red", scene_spec=_simple_scene())


def _plugin_module(experiment_id: str) -> ModuleType:
    module = ModuleType(f"{experiment_id}.experiment")
    module.EXPERIMENT = ExperimentSpec(
        experiment_id=experiment_id,
        scene_id=f"{experiment_id}_scene",
        task_id="cube_task",
        num_runs_default=1,
        scene_template=_simple_scene(f"{experiment_id}_scene"),
        build_run_setup=_simple_run_setup,
    )
    return module


def test_builtin_experiment_plugins_are_discovered():
    experiments = discover_experiments(force_reload=True)

    assert {"e1", "e2", "e3", "e4"} <= set(experiments)


def test_loader_only_imports_experiment_modules(tmp_path, monkeypatch):
    root = tmp_path / "specs"
    alpha = root / "alpha"
    alpha.mkdir(parents=True)
    (alpha / "experiment.py").write_text("# stub\n")
    (alpha / "mujoco.py").write_text("# should not be imported\n")
    (alpha / "vpython.py").write_text("# should not be imported\n")

    imported: list[str] = []

    def fake_load(name: str):
        imported.append(name)
        return _plugin_module("alpha")

    monkeypatch.setattr("experiments.specs.loader._SPECS_DIR", root)
    monkeypatch.setattr("experiments.specs.loader._load_experiment_module", fake_load)
    invalidate_experiment_cache()

    experiments = discover_experiments(force_reload=True)

    assert imported == ["experiments.specs.alpha.experiment"]
    assert list(experiments) == ["alpha"]


def test_loader_rejects_duplicate_experiment_ids(tmp_path, monkeypatch):
    root = tmp_path / "specs"
    for name in ("alpha", "beta"):
        pkg = root / name
        pkg.mkdir(parents=True)
        (pkg / "experiment.py").write_text("# stub\n")

    def fake_load(name: str):
        del name
        return _plugin_module("duplicate")

    monkeypatch.setattr("experiments.specs.loader._SPECS_DIR", root)
    monkeypatch.setattr("experiments.specs.loader._load_experiment_module", fake_load)
    invalidate_experiment_cache()

    with pytest.raises(ValueError, match="Duplicate experiment_id"):
        discover_experiments(force_reload=True)


def test_loader_rejects_malformed_experiment_export(tmp_path, monkeypatch):
    root = tmp_path / "specs"
    pkg = root / "alpha"
    pkg.mkdir(parents=True)
    (pkg / "experiment.py").write_text("# stub\n")

    def fake_load(name: str):
        module = _plugin_module("alpha")
        module.EXTRA = "not allowed"
        return module

    monkeypatch.setattr("experiments.specs.loader._SPECS_DIR", root)
    monkeypatch.setattr("experiments.specs.loader._load_experiment_module", fake_load)
    invalidate_experiment_cache()

    with pytest.raises(ValueError, match="exactly one public symbol"):
        discover_experiments(force_reload=True)


def test_experiment_spec_resolve_task_prefers_local_factory():
    local_task = TaskSpec(
        task_id="local_task",
        build_instruction=lambda run_setup: f"local:{run_setup.agent_color_name}",
        response_schema={},
        parse_reply=lambda raw, previous_action: (raw, previous_action),
        capabilities=TaskCapabilities(),
    )
    spec = ExperimentSpec(
        experiment_id="local",
        scene_id="local_scene",
        task_id="cube_task",
        num_runs_default=1,
        scene_template=_simple_scene("local_scene"),
        build_run_setup=_simple_run_setup,
        task_factory=lambda: local_task,
    )

    assert spec.resolve_task() is local_task


def test_tasks_resolve_task_can_follow_experiment_local_override(monkeypatch):
    from experiments.tasks import resolve_task

    local_task = TaskSpec(
        task_id="local_task",
        build_instruction=lambda run_setup: f"local:{run_setup.agent_color_name}",
        response_schema={},
        parse_reply=lambda raw, previous_action: (raw, previous_action),
        capabilities=TaskCapabilities(),
    )
    spec = ExperimentSpec(
        experiment_id="local",
        scene_id="local_scene",
        task_id="cube_task",
        num_runs_default=1,
        scene_template=_simple_scene("local_scene"),
        build_run_setup=_simple_run_setup,
        task_factory=lambda: local_task,
    )

    monkeypatch.setattr(
        "experiments.specs.get_experiment",
        lambda experiment_id: spec if experiment_id == "local" else None,
    )

    assert resolve_task(experiment_id="local") is local_task


def test_backend_overrides_default_to_empty():
    spec = ExperimentSpec(
        experiment_id="local",
        scene_id="local_scene",
        task_id="cube_task",
        num_runs_default=1,
        scene_template=_simple_scene("local_scene"),
        build_run_setup=_simple_run_setup,
    )

    overrides = spec.resolve_backend_overrides()

    assert overrides == BackendOverrides()


def test_backend_overrides_factory_can_supply_typed_mujoco_builder():
    def build_scene(scene_spec: SceneSpec, render_width: int, render_height: int) -> MuJoCoSceneBuildResult:
        assert scene_spec.scene_id == "local_scene"
        assert render_width == 640
        assert render_height == 480
        return MuJoCoSceneBuildResult(
            xml="<mujoco/>",
            robot_geom_names=["robot_geom"],
            mirror_geom_names=["mirror_geom"],
        )

    spec = ExperimentSpec(
        experiment_id="local",
        scene_id="local_scene",
        task_id="cube_task",
        num_runs_default=1,
        scene_template=_simple_scene("local_scene"),
        build_run_setup=_simple_run_setup,
        backend_overrides_factory=lambda: BackendOverrides(
            mujoco_scene_builder=build_scene,
        ),
    )

    overrides = spec.resolve_backend_overrides()
    result = overrides.mujoco_scene_builder(spec.scene_template, 640, 480)

    assert result.xml == "<mujoco/>"
    assert result.robot_geom_names == ["robot_geom"]
    assert result.mirror_geom_names == ["mirror_geom"]


def test_scene_object_spec_coerces_legacy_dicts():
    scene = SceneSpec(
        scene_id="with_objects",
        mirror_enabled=True,
        objects=[
            {"type": "cube", "color": "red", "position": (-10.0, -4.0)},
            SceneObjectSpec(
                object_id="cube_green",
                kind="cube",
                color="green",
                position=(-8.0, -4.0),
            ),
        ],
    )

    assert all(isinstance(obj, SceneObjectSpec) for obj in scene.objects)
    assert scene.objects[0].kind == "cube"
    assert scene.objects[0]["color"] == "red"
