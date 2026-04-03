from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class _FakeTable:
    def __init__(self, columns):
        self.columns = columns
        self.rows = []

    def add_data(self, *values):
        self.rows.append(values)


class _FakeArtifact:
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.files = []

    def add_file(self, path, name=None):
        file_path = Path(path)
        self.files.append(
            {
                "path": str(file_path),
                "name": name or file_path.name,
                "content": file_path.read_text(),
            }
        )


class _FakeRun:
    def __init__(self):
        self.summary = {}
        self.logged = []
        self.artifacts = []

    def log(self, payload):
        self.logged.append(payload)

    def log_artifact(self, artifact):
        self.artifacts.append(artifact)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _build_fake_wandb(captured, fake_run):
    def _fake_init(**kwargs):
        captured["init_kwargs"] = kwargs
        return fake_run

    return SimpleNamespace(
        Artifact=_FakeArtifact,
        Image=lambda figure: {"figure": figure},
        Table=_FakeTable,
        init=_fake_init,
        plot=SimpleNamespace(
            bar=lambda table, label, value, title: {
                "table": table,
                "label": label,
                "value": value,
                "title": title,
            }
        ),
    )


def _sample_results_subset():
    return [
        SimpleNamespace(metric_code="M1", metric_name="Mirror", score=4, justification="Good."),
        SimpleNamespace(metric_code="M3", metric_name="Consistency", score=5, justification="Strong."),
    ]


def _sample_results_full():
    return [
        SimpleNamespace(metric_code="M1", metric_name="Mirror", score=4, justification="Good."),
        SimpleNamespace(metric_code="M2", metric_name="Grounding", score=3, justification="Reasonable."),
        SimpleNamespace(metric_code="M3", metric_name="Consistency", score=5, justification="Strong."),
        SimpleNamespace(metric_code="M4", metric_name="Alternatives", score=2, justification="Limited."),
    ]


def _sample_run_data():
    return {
        "agent_color": "red",
        "selected_cube": "red",
        "wrong_color": "blue",
        "completed": True,
        "steps": 7,
    }


def _build_export(results, run_path, *, prompt_variant="strict", include_inner_reasoning=True):
    from experiments.evaluation.judge import _build_judge_export

    return _build_judge_export(
        results,
        _sample_run_data(),
        json_path=run_path,
        judge_model="llama3.2",
        prompt_variant=prompt_variant,
        include_inner_reasoning=include_inner_reasoning,
    )


def test_build_judge_export_contains_v2_fields():
    export = _build_export(
        _sample_results_full(),
        "outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json",
    )

    assert export["run_path"] == "outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json"
    assert export["judge_model"] == "llama3.2"
    assert export["prompt_variant"] == "strict"
    assert export["include_inner_reasoning"] is True
    assert export["env"] == "mujoco"
    assert export["nav_client"] == "ollama"
    assert export["evaluated_model"] == "gemma3_4b"
    assert export["experiment_type"] == "e1"
    assert export["run_stem"] == "run01"
    assert export["correct"] is True
    assert export["metric_codes"] == ["M1", "M2", "M3", "M4"]
    assert export["total_score"] == 14
    assert export["mean_score"] == 3.5
    assert export["results"][0]["metric_code"] == "M1"
    assert export["results"][0]["metric_name"] == "Mirror"


def test_derive_run_context_experiment_path():
    from utils.integrations.wandb import _derive_run_context

    context = _derive_run_context("outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json")

    assert context.env == "mujoco"
    assert context.nav_client == "ollama"
    assert context.evaluated_model == "gemma3_4b"
    assert context.experiment_type == "e1"
    assert context.group == "gemma3_4b/e1"
    assert context.name == "gemma3_4b/e1/run01"


def test_derive_run_context_legacy_experiment_path_still_supported():
    from utils.integrations.wandb import _derive_run_context

    context = _derive_run_context("outputs/mujoco/ollama/gemma3_4b/runs/e1/run01.json")

    assert context.env == "mujoco"
    assert context.nav_client == "ollama"
    assert context.evaluated_model == "gemma3_4b"
    assert context.experiment_type == "e1"
    assert context.group == "gemma3_4b/e1"
    assert context.name == "gemma3_4b/e1/run01"


def test_derive_run_context_single_run_path():
    from utils.integrations.wandb import _derive_run_context

    context = _derive_run_context("outputs/vpython/google/gemini-2.5-flash/runs/run01.json")

    assert context.env == "vpython"
    assert context.nav_client == "google"
    assert context.evaluated_model == "gemini-2.5-flash"
    assert context.experiment_type == "single-run"
    assert context.group == "gemini-2.5-flash/single-run"
    assert context.name == "gemini-2.5-flash/single-run/run01"


def test_log_judge_results_logs_subset_payload_without_radar():
    captured = {}
    fake_run = _FakeRun()
    fake_wandb = _build_fake_wandb(captured, fake_run)

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        from utils.integrations.wandb import log_judge_results

        log_judge_results(
            _sample_results_subset(),
            _sample_run_data(),
            judge_model="llama3.2",
            prompt_variant="strict",
            include_inner_reasoning=True,
            run_path="outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json",
            project="self-aware",
        )

    init_kwargs = captured["init_kwargs"]
    assert init_kwargs["project"] == "self-aware"
    assert init_kwargs["name"] == "gemma3_4b/e1/run01"
    assert init_kwargs["group"] == "gemma3_4b/e1"
    assert init_kwargs["job_type"] == "llm-as-judge"
    assert init_kwargs["config"]["judge_model"] == "llama3.2"
    assert init_kwargs["config"]["include_inner_reasoning"] is True
    assert init_kwargs["config"]["metric_codes"] == ["M1", "M3"]
    assert "llm-as-judge" in init_kwargs["tags"]

    assert fake_run.summary["agent_color"] == "red"
    assert fake_run.summary["selected_cube"] == "red"
    assert fake_run.summary["correct"] is True
    assert fake_run.summary["metric_count"] == 2
    assert fake_run.summary["total_score"] == 9
    assert fake_run.summary["mean_score"] == 4.5
    assert fake_run.summary["radar_logged"] is False
    assert fake_run.summary["comparison_row_count"] == 1

    payload = fake_run.logged[0]
    assert payload["judge/score/M1"] == 4
    assert payload["judge/score/M3"] == 5
    assert "judge/score/M2" not in payload
    assert payload["judge/score_norm/M1"] == 0.8
    assert payload["judge/score_norm/M3"] == 1.0
    assert payload["judge/total_score"] == 9
    assert payload["judge/mean_score"] == 4.5
    assert payload["judge_scores"].columns == [
        "metric_code",
        "metric_name",
        "score",
        "score_norm",
        "justification",
    ]
    assert payload["judge_scores"].rows[0][0] == "M1"
    assert payload["judge_scores_bar"]["title"] == "Judge scores"
    assert "judge_scores_radar" not in payload
    assert payload["judge_experiment_comparison"].rows == [
        ("run01", "red", "red", "blue", True, True, 7, 9, 4.5, 4, None, 5, None, "llama3.2", "strict", True)
    ]

    assert len(fake_run.artifacts) == 1
    artifact = fake_run.artifacts[0]
    assert artifact.type == "judge-output"
    assert artifact.files[0]["name"] == "judge_output.md"
    assert "## Metric Outputs" in artifact.files[0]["content"]
    assert "=== AGENT TRANSCRIPT ===" not in artifact.files[0]["content"]


def test_log_judge_results_logs_radar_and_comparison_table_from_local_exports(tmp_path):
    root = tmp_path / "outputs" / "mujoco" / "e1" / "ollama" / "gemma3_4b" / "runs"
    root.mkdir(parents=True)

    run01_path = root / "run01.json"
    run01_path.write_text("{}")
    current_export = _build_export(_sample_results_full(), run01_path)

    run02_export = _build_export(_sample_results_full(), root / "run02.json")
    (root / "run02_judge.json").write_text(json.dumps(run02_export, indent=2))

    run03_export = _build_export(
        _sample_results_full(),
        root / "run03.json",
        prompt_variant="default",
    )
    (root / "run03_judge.json").write_text(json.dumps(run03_export, indent=2))

    legacy_export = [
        {
            "metric_code": "M1",
            "score": 4,
            "justification": "legacy",
        }
    ]
    (root / "run04_judge.json").write_text(json.dumps(legacy_export, indent=2))

    captured = {}
    fake_run = _FakeRun()
    fake_wandb = _build_fake_wandb(captured, fake_run)

    with patch.dict(sys.modules, {"wandb": fake_wandb}):
        from utils.integrations.wandb import log_judge_results

        log_judge_results(
            _sample_results_full(),
            _sample_run_data(),
            judge_model="llama3.2",
            prompt_variant="strict",
            include_inner_reasoning=True,
            run_path=run01_path,
            project="self-aware",
            judge_export=current_export,
        )

    payload = fake_run.logged[0]
    assert fake_run.summary["radar_logged"] is True
    assert fake_run.summary["comparison_row_count"] == 2
    assert "judge_scores_radar" in payload
    assert payload["judge_experiment_comparison"].columns == [
        "run_stem",
        "agent_color",
        "selected_cube",
        "wrong_color",
        "correct",
        "completed",
        "steps",
        "total_score",
        "mean_score",
        "M1",
        "M2",
        "M3",
        "M4",
        "judge_model",
        "prompt_variant",
        "include_inner_reasoning",
    ]
    assert payload["judge_experiment_comparison"].rows == [
        ("run01", "red", "red", "blue", True, True, 7, 14, 3.5, 4, 3, 5, 2, "llama3.2", "strict", True),
        ("run02", "red", "red", "blue", True, True, 7, 14, 3.5, 4, 3, 5, 2, "llama3.2", "strict", True),
    ]

    artifact = fake_run.artifacts[0]
    assert artifact.name == "judge-output-gemma3_4b-e1-run01"
    content = artifact.files[0]["content"]
    assert "Evaluated model: `gemma3_4b`" in content
    assert "Judge model: `llama3.2`" in content
    assert "Score: `4/5`" in content
    assert "=== RUN METADATA ===" not in content


def test_missing_wandb_dependency_raises_helpful_error():
    import builtins

    from utils.integrations.wandb import log_judge_results

    real_import = builtins.__import__

    def _mocked_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "wandb":
            raise ImportError("wandb missing")
        return real_import(name, globals, locals, fromlist, level)

    with patch("builtins.__import__", side_effect=_mocked_import):
        try:
            log_judge_results(
                _sample_results_subset(),
                _sample_run_data(),
                judge_model="llama3.2",
                prompt_variant="default",
                include_inner_reasoning=False,
                run_path="outputs/mujoco/ollama/gemma3_4b/runs/run01.json",
            )
        except RuntimeError as exc:
            assert "pip install wandb" in str(exc)
        else:
            raise AssertionError("Expected RuntimeError when wandb is unavailable")


def test_llm_as_judge_parse_args_accepts_wandb_flags(monkeypatch):
    from experiments.evaluation.judge import _parse_args

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "llm_as_judge.py",
            "outputs/mujoco/e1/ollama/gemma3_4b/runs/run01.json",
            "--wandb",
            "--wandb-project",
            "self-aware",
            "--wandb-run",
            "custom-run",
        ],
    )
    args = _parse_args()

    assert args.wandb is True
    assert args.wandb_project == "self-aware"
    assert args.wandb_run == "custom-run"
