import json
from pathlib import Path
from types import ModuleType, SimpleNamespace
import importlib
import os
import subprocess
import sys

import pytest

from experiments import RunSetup, SceneSpec


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


@pytest.fixture
def launcher_probe(tmp_path):
    """Compose real Hydra configs while replacing model runs and judge requests."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = tmp_path / "calls.jsonl"
    probe = bin_dir / "python"
    probe.write_text(f"#!{sys.executable}\n" + r'''
import json, os, subprocess, sys
from pathlib import Path
from omegaconf import OmegaConf
args = sys.argv[1:]
if args[0] == "-u":
    args = args[1:]
if args[0] == "main.py":
    result = subprocess.run([sys.executable, *args, "--cfg", "job"],
                            capture_output=True, text=True, check=True)
    call = OmegaConf.to_container(OmegaConf.create(result.stdout), resolve=True)
    call["MUJOCO_GL"] = os.environ.get("MUJOCO_GL")
    if os.environ.get("EMIT_RUNS"):
        root = Path(call["outputs"]["dir"]) / "mujoco" / call["experiment_id"] / "session"
        for number in (1, 2):
            target = root / f"run_{number:02}" / f"run{number:02}.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("{}")
else:
    assert args[:2] == ["-m", "experiments.evaluation.judge"]
    call = {"judge": args[2:]}
with open(os.environ["LAUNCHER_LOG"], "a") as stream:
    stream.write(json.dumps(call) + "\n")
''')
    probe.chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
           "LAUNCHER_LOG": str(log)}
    env.pop("MUJOCO_GL", None)
    return env, log


@pytest.mark.parametrize("script,headless", [
    ("scripts/mujoco/run.sh", False), ("scripts/mujoco.sh", False),
    ("scripts/mujoco/headless.sh", True), ("scripts/mujoco_headless.sh", True),
])
def test_launchers_compose_paper_defaults_and_accept_overrides(launcher_probe, script, headless):
    env, log = launcher_probe
    subprocess.run(["bash", str(ROOT / script), "nav=test-model", "experiment=e4"],
                   cwd=ROOT.parent, env=env, check=True, capture_output=True, text=True)
    cfg = json.loads(log.read_text())
    assert cfg["experiment_id"] == "e4"
    assert cfg["nav"] == "test-model"
    assert cfg["num_runs"] == 7 and cfg["max_steps"] == 100 and cfg["history_steps"] == 1
    assert cfg["mujoco"]["headless"] is headless
    assert cfg["MUJOCO_GL"] is None


def test_pipeline_judges_only_its_new_nested_runs(launcher_probe, tmp_path):
    env, log = launcher_probe
    old = tmp_path / "older" / "run01.json"
    old.parent.mkdir()
    old.write_text("{}")
    subprocess.run(["bash", str(ROOT / "scripts/pipeline.sh"), "experiment=e2", "num_runs=2"],
                   cwd=ROOT.parent, check=True, capture_output=True, text=True,
                   env={**env, "NAV_MODEL": "vision-model", "JUDGE_MODEL": "judge-model",
                        "OUTPUT_ROOT": str(tmp_path), "EMIT_RUNS": "1"})
    run, *judges = [json.loads(line) for line in log.read_text().splitlines()]
    assert run["experiment_id"] == "e2" and run["num_runs"] == 2
    assert run["max_steps"] == 100 and run["history_steps"] == 1
    assert run["MUJOCO_GL"] is None
    paths = [Path(j["judge"][0]) for j in judges]
    assert len(paths) == 2 and old not in paths
    assert all(p.is_relative_to(run["outputs"]["dir"]) and p.is_file() for p in paths)
    assert all(j["judge"][2] == "judge-model" for j in judges)


def test_judge_launcher_requires_explicit_run_path(launcher_probe, tmp_path):
    env, log = launcher_probe
    path = str(tmp_path / "selected run.json")
    subprocess.run(["bash", str(ROOT / "scripts/mujoco_eval.sh"), path, "--model", "judge-model"],
                   cwd=ROOT.parent, env=env, check=True, capture_output=True, text=True)
    assert json.loads(log.read_text()) == {"judge": [path, "--model", "judge-model"]}


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


def test_self_correction_plots_help_renders():
    script = ROOT / "scripts" / "dev" / "self_correction_plots.py"

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

    assert "Plot SELF_CORRECTION correlations" in result.stdout
    assert "--output-dir" in result.stdout
    assert "--min-applicable-runs" in result.stdout


def test_openai_batch_script_help_renders():
    script = ROOT / "scripts" / "dev" / "openai_batch.py"

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

    assert "Validate, submit, and inspect OpenAI Batch API jobs." in result.stdout
    assert "validate" in result.stdout
    assert "submit" in result.stdout
    assert "download" in result.stdout


def test_openai_batch_validate_accepts_example_jsonl():
    script = ROOT / "scripts" / "dev" / "openai_batch.py"
    example = ROOT / "scripts" / "dev" / "openai_batch.example.jsonl"

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "validate",
            str(example),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "Validated 2 request(s)" in result.stdout
    assert "Endpoint: /v1/responses" in result.stdout


def test_openai_batch_validate_rejects_duplicate_custom_ids(tmp_path):
    script = ROOT / "scripts" / "dev" / "openai_batch.py"
    bad_jsonl = tmp_path / "duplicate.jsonl"
    bad_jsonl.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "custom_id": "dup-1",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": {"model": "gpt-5.4-mini", "input": "hello"},
                    }
                ),
                json.dumps(
                    {
                        "custom_id": "dup-1",
                        "method": "POST",
                        "url": "/v1/responses",
                        "body": {"model": "gpt-5.4-mini", "input": "goodbye"},
                    }
                ),
            ]
        )
        + "\n"
    )

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "validate",
            str(bad_jsonl),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Duplicate custom_id" in result.stderr


def test_google_vertex_smoke_script_help_renders():
    script = ROOT / "scripts" / "dev" / "google_vertex_smoke.py"

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

    assert "Run minimal Gemini-on-Vertex-AI smoke test requests." in result.stdout
    assert "--project" in result.stdout
    assert "--location" in result.stdout
    assert "--model-file" in result.stdout
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


def test_openrouter_matrix_script_allows_matrix_nav_for_e7_preset(tmp_path):
    script = ROOT / "scripts" / "dev" / "openrouter_matrix.py"
    config = ROOT / "scripts" / "dev" / "openrouter_matrix.example.json"
    data = json.loads(config.read_text())
    data["experiments"].append("e5")
    config = tmp_path / "matrix.json"
    config.write_text(json.dumps(data))

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--config",
            str(config),
            "--models",
            "meta-llama/llama-3.2-11b-vision-instruct",
            "--experiments",
            "e5",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        env={**os.environ, "OPENROUTER_API_KEY": "test-key"},
    )

    assert "Prepared 1 job(s)." in result.stdout
    assert "experiment=e5" in result.stdout
    assert "nav=meta-llama/llama-3.2-11b-vision-instruct" in result.stdout
    assert "max_steps=" not in result.stdout
