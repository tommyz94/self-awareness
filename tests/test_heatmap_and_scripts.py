from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import os
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


def test_openrouter_matrix_script_dry_run_builds_expected_jobs():
    script = ROOT / "scripts" / "dev" / "openrouter_matrix.py"
    config = ROOT / "scripts" / "dev" / "openrouter_matrix.example.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--models",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "--experiments",
            "e1,e2",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
    )

    assert "Prepared 2 job(s)." in result.stdout
    assert "client=openrouter" in result.stdout
    assert "experiment=e1" in result.stdout
    assert "experiment=e2" in result.stdout
    assert "nav=meta-llama/llama-3.2-11b-vision-instruct" in result.stdout
    assert "num_runs=" not in result.stdout


def test_openrouter_smoke_script_help_renders():
    script = ROOT / "scripts" / "dev" / "openrouter_smoke.py"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--help",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Run minimal OpenRouter smoke test requests." in result.stdout
    assert "--model" in result.stdout
    assert "--model-file" in result.stdout
    assert "--variant" in result.stdout
    assert "--no-image" in result.stdout


def test_openrouter_matrix_script_set_overrides_replace_config_values():
    script = ROOT / "scripts" / "dev" / "openrouter_matrix.py"
    config = ROOT / "scripts" / "dev" / "openrouter_matrix.example.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--models",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "--experiments",
            "e1",
            "--set",
            "seed=123",
            "--set",
            "mujoco.headless=false",
            "--set",
            "print_reasoning=false",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
    )

    assert "seed=123" in result.stdout
    assert "mujoco.headless=false" in result.stdout
    assert "mujoco.headless=true" not in result.stdout
    assert "print_reasoning=false" in result.stdout
    assert "print_reasoning=true" not in result.stdout


def test_openrouter_matrix_script_rejects_experiment_locked_overrides():
    script = ROOT / "scripts" / "dev" / "openrouter_matrix.py"
    config = ROOT / "scripts" / "dev" / "openrouter_matrix.example.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--models",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "--experiments",
            "e1",
            "--set",
            "num_runs=3",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
    )

    assert result.returncode != 0
    assert "Refusing to override experiment-owned preset keys" in result.stderr


def test_openrouter_matrix_script_allows_matrix_nav_for_e7_preset():
    script = ROOT / "scripts" / "dev" / "openrouter_matrix.py"
    config = ROOT / "scripts" / "dev" / "openrouter_matrix.example.json"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--models",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "--experiments",
            "e7",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
    )

    assert "Prepared 1 job(s)." in result.stdout
    assert "experiment=e7" in result.stdout
    assert "nav=meta-llama/llama-3.2-11b-vision-instruct" in result.stdout
    assert "max_steps=" not in result.stdout
