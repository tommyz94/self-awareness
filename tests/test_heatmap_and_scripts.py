from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import subprocess
import sys

from experiments.specs.registry import RunSetup, SceneSpec


ROOT = Path(__file__).resolve().parent.parent


def test_importing_runtime_controller_does_not_load_matplotlib():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                "import runtime.controller; "
                "print(any(name.startswith('matplotlib') for name in sys.modules))"
            ),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert result.stdout.strip() == "False"


def test_save_heatmap_if_supported_skips_corridor_task_without_import(monkeypatch, capsys):
    import runtime.controller as controller

    fake_module = ModuleType("utils.viz.heatmap")
    called = {"value": False}

    def _fake_save(*args, **kwargs):
        called["value"] = True
        return "should-not-be-called"

    fake_module.save_trajectory_heatmap = _fake_save
    monkeypatch.setitem(sys.modules, "utils.viz.heatmap", fake_module)

    run_setup = RunSetup(
        agent_color_name="red",
        scene_spec=SceneSpec(
            scene_id="narrow_corridor_mirror",
            mirror_enabled=True,
            layout_kind="corridor",
        ),
    )
    task_spec = SimpleNamespace(task_id="corridor_task")

    saved = controller._save_heatmap_if_supported(
        run_id=1,
        trajectory=[(0.0, 0.0)],
        agent_color="red",
        object_positions_2d={},
        output_dir=str(ROOT / "outputs"),
        run_setup=run_setup,
        task_spec=task_spec,
    )

    captured = capsys.readouterr()
    assert saved is None
    assert called["value"] is False
    assert "Skipping heatmap" in captured.out


def test_save_heatmap_if_supported_calls_writer_for_two_room_cube_task(monkeypatch):
    import runtime.controller as controller

    fake_module = ModuleType("utils.viz.heatmap")
    calls = []

    def _fake_save(*args, **kwargs):
        calls.append((args, kwargs))
        return "fake-path.png"

    fake_module.save_trajectory_heatmap = _fake_save
    monkeypatch.setitem(sys.modules, "utils.viz.heatmap", fake_module)

    run_setup = RunSetup(
        agent_color_name="red",
        scene_spec=SceneSpec(
            scene_id="mirror_room",
            mirror_enabled=True,
            layout_kind="two_room",
        ),
    )
    task_spec = SimpleNamespace(task_id="cube_task")

    saved = controller._save_heatmap_if_supported(
        run_id=2,
        trajectory=[(1.0, 2.0)],
        agent_color="red",
        object_positions_2d={"red": (-10.0, -4.0)},
        output_dir=str(ROOT / "outputs"),
        run_setup=run_setup,
        task_spec=task_spec,
    )

    assert saved == "fake-path.png"
    assert len(calls) == 1


def test_canonical_scripts_and_judge_docstring_use_canonical_entrypoint():
    canonical_targets = [
        ROOT / "experiments" / "evaluation" / "judge.py",
        ROOT / "experiments" / "evaluation" / "judge_reliability.py",
        ROOT / "scripts" / "mujoco" / "eval.sh",
        ROOT / "scripts" / "pipelines" / "pipeline.sh",
    ]

    for path in canonical_targets:
        text = path.read_text()
        if path.name == "judge_reliability.py":
            assert "python -m experiments.evaluation.judge_reliability" in text
            assert "python -m experiments.judge_reliability" not in text
        else:
            assert "python -m experiments.evaluation.judge" in text
            assert "python -m experiments.llm_as_judge" not in text


def test_eval_scripts_do_not_keep_stale_pre_reorg_defaults():
    mujoco_eval = (ROOT / "scripts" / "mujoco" / "eval.sh").read_text()
    mujoco_headless = (ROOT / "scripts" / "mujoco" / "headless.sh").read_text()

    assert "qwen3.5_397b-cloud" not in mujoco_eval
    assert "outputs/mujoco/qwen3.5_397b-cloud" not in mujoco_eval
    assert "+experiment=" in mujoco_headless


def test_readme_describes_non_cube_heatmaps_correctly():
    readme = (ROOT / "README.md").read_text().replace("\n", " ")

    assert "still saves run JSON and heatmaps" not in readme
    assert "skips room-style heatmaps" in readme
