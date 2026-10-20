"""Paper names must select the original tasks without changing their scenes."""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
from types import SimpleNamespace

from hydra import compose, initialize_config_dir
import pytest

from experiments.evaluation.metrics import RunOutcome
from experiments.naming import EXTRA_EXPERIMENTS, PAPER_EXPERIMENTS, resolve_experiment_selection
from experiments.specs import ExperimentRuntimeOptions, get_experiment
from utils.io.json_export import save_experiment_json, save_run_json

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("paper_id,implementation_id", PAPER_EXPERIMENTS.items())
def test_paper_preset_matches_historical_scene_and_task(paper_id, implementation_id):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment=paper/{paper_id}"])

    assert cfg.paper_experiment_id == paper_id
    assert cfg.experiment_id == implementation_id
    assert (cfg.num_runs, cfg.max_steps, cfg.history_steps) == (7, 100, 1)
    assert (cfg.mujoco.render_width, cfg.mujoco.render_height) == (640, 480)
    assert cfg.nav != "auto"
    assert cfg.outputs.dir == "outputs/paper"
    assert cfg.third_person is False
    assert cfg.body_style == "default"

    public = get_experiment(f"paper/{paper_id}")
    historical = get_experiment(implementation_id)
    options = ExperimentRuntimeOptions(e7_random_cube_heights=cfg.e7_random_cube_heights)
    state = random.getstate()
    try:
        random.seed(123)
        actual = asdict(public.initialize_run(1, options))
        random.seed(123)
        expected = asdict(historical.initialize_run(1, options))
    finally:
        random.setstate(state)
    assert actual == expected
    assert public.resolve_task() is historical.resolve_task()
    assert public.task_id == ("explore_task" if paper_id == "e5" else "cube_task")


@pytest.mark.parametrize("name,task_id", [(name, task_id) for name, task_id in [
    ("direct_perception", "self_perception_task"),
    ("front_identity", "front_identity_task"),
    ("corridor", "corridor_task"),
    ("exploration", "explore_task"),
    ("embodiment", "self_description_explore_task"),
    ("mirror_side_choice", "mirror_side_choice_task"),
] if name in EXTRA_EXPERIMENTS])
def test_extra_presets_remain_resolvable(name, task_id):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment=extras/{name}"])
    assert cfg.paper_experiment_id is None
    assert get_experiment(f"extras/{name}").resolve_task().task_id == task_id
    assert get_experiment(cfg.experiment_id).task_id == task_id


def test_numeric_ids_are_not_reinterpreted_and_conflicts_are_rejected():
    assert get_experiment("e4").experiment_id == "e4"
    assert get_experiment("e5").task_id == "cube_task"
    assert get_experiment("e7").task_id == "explore_task"
    with pytest.raises(ValueError, match="requires implementation"):
        resolve_experiment_selection("e4", "e4")
    with pytest.raises(ValueError, match="Unknown paper"):
        resolve_experiment_selection("paper/e8")


def test_paper_export_adds_identity_without_changing_legacy_fields(tmp_path):
    context = {"paper_experiment_id": "e4", "implementation_experiment_id": "e5"}
    outcome = RunOutcome(
        run_id=1, agent_color="red", wrong_color=None, selected_cube="red",
        completed=True, steps=1, experiment_id="e5", task_id="cube_task",
    )
    run_path = save_run_json(1, outcome, [], str(tmp_path), run_context=context)
    summary_path = save_experiment_json([outcome], "e5", str(tmp_path), run_context=context)
    for path in (run_path, summary_path):
        data = json.loads(Path(path).read_text())
        assert data["experiment_id"] == "e5"
        assert data["paper_experiment_id"] == "e4"
    assert context == {"paper_experiment_id": "e4", "implementation_experiment_id": "e5"}


def test_controller_resolves_public_preset_and_records_both_ids(monkeypatch, tmp_path):
    from runtime.controller import Controller

    fake_env = SimpleNamespace(position=(-10.0, 0.0), direction="N")
    monkeypatch.setattr("runtime.controller._create_env", lambda **kwargs: fake_env)
    controller = Controller(
        nav_mode="test-model", experiment_id="paper/e4",
        output_dir=str(tmp_path), heatmap=False,
    )
    context = controller._build_run_context()
    assert controller.experiment_id == "e5"
    assert controller.task_spec.task_id == "cube_task"
    assert context["paper_experiment_id"] == "e4"
    assert context["implementation_experiment_id"] == "e5"
    assert context["experiment_preset"] == "paper/e4"


def test_paper_matrix_keeps_preset_namespace_out_of_log_filenames(monkeypatch):
    from scripts.dev.openrouter_matrix import _prepare_jobs

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    jobs = _prepare_jobs(
        ROOT / "scripts/dev/openrouter_matrix.paper.json",
        model_filter=["qwen/qwen3.6-plus"],
        experiment_filter=["paper/e4", "paper/e5"],
        batch_id="offline-check",
        cli_overrides={},
    )
    assert len(jobs) == 2
    for job in jobs:
        assert "/" not in job["name"]
        assert job["log_file"].parent.name == "offline-check"
        assert f"experiment={job['experiment']}" in job["python_command"]


def test_legacy_export_omits_paper_identity(tmp_path):
    outcome = RunOutcome(1, "red", None, "red", True, 1, experiment_id="e1", task_id="cube_task")
    run_path = save_run_json(1, outcome, [], str(tmp_path))
    summary_path = save_experiment_json([outcome], "e1", str(tmp_path))
    for path in (run_path, summary_path):
        assert "paper_experiment_id" not in json.loads(Path(path).read_text())


def test_controller_finalize_propagates_paper_context_without_changing_outcome(tmp_path):
    from runtime.controller import Controller

    controller = Controller.__new__(Controller)
    controller.paper_experiment_id = "e4"
    controller.experiment_id = "e5"
    controller._build_run_context = lambda: {
        "paper_experiment_id": "e4", "experiment_preset": "paper/e4",
        "implementation_experiment_id": "e5",
    }
    controller.current_run_dir = tmp_path
    controller.current_run_setup = None
    controller.trajectory = [(0.0, 0.0)]
    controller.current_object_positions_2d = {}
    controller.step_records = []
    controller.json_export = True
    controller.heatmap = False
    outcome = RunOutcome(1, "red", None, "red", True, 1, experiment_id="e5", task_id="cube_task")
    assert controller._finalize_episode(1, outcome) is outcome
    data = json.loads((tmp_path / "run01.json").read_text())
    assert data["experiment_id"] == "e5"
    assert data["paper_experiment_id"] == "e4"
    assert data["run_context"]["experiment_preset"] == "paper/e4"
    assert data["completed"] is True
    assert outcome.selected_cube == "red"
