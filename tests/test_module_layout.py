from __future__ import annotations


def test_repo_no_longer_contains_removed_flat_utils_modules():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    removed_files = [
        root / "utils" / "utils.py",
        root / "utils" / "logger.py",
        root / "utils" / "json_export.py",
        root / "utils" / "heatmap.py",
        root / "utils" / "vpython_capture.py",
        root / "utils" / "wandb_logger.py",
    ]
    search_roots = [
        root / "README.md",
        root / "conf",
        root / "env",
        root / "experiments",
        root / "llm",
        root / "providers",
        root / "runtime",
        root / "scripts",
        root / "tests",
        root / "utils",
        root / "main.py",
    ]
    offenders = []
    removed_snippets = [
        "utils.utils",
        "utils.logger",
        "utils.json_export",
        "utils.heatmap",
        "utils.vpython_capture",
        "utils.wandb_logger",
        "utils/utils.py",
        "utils/logger.py",
        "utils/json_export.py",
        "utils/heatmap.py",
        "utils/vpython_capture.py",
        "utils/wandb_logger.py",
    ]

    for removed_file in removed_files:
        assert not removed_file.exists()

    for path in search_roots:
        paths = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".md", ".sh"}
        )
        for candidate in paths:
            if candidate in {
                Path(__file__).resolve(),
                root / "tests" / "test_folder_docs.py",
                root / "tests" / "test_heatmap_and_scripts.py",
            }:
                continue
            text = candidate.read_text()
            if any(snippet in text for snippet in removed_snippets):
                offenders.append(str(candidate.relative_to(root)))

    assert offenders == []


def test_repo_no_longer_contains_removed_env_and_experiments_wrapper_modules():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    removed_files = [
        root / "env" / "base_env.py",
        root / "env" / "mujoco_env.py",
        root / "env" / "vpython_env.py",
        root / "env" / "vpython_headless.py",
        root / "env" / "environment.py",
        root / "env" / "mjcf_builder.py",
        root / "experiments" / "registry.py",
        root / "experiments" / "metrics.py",
        root / "experiments" / "llm_as_judge.py",
        root / "experiments" / "judge_reliability.py",
        root / "experiments" / "judge_metrics.py",
        root / "experiments" / "judge_prompts.py",
    ]
    removed_snippets = [
        "env.base_env",
        "env.mujoco_env",
        "env.vpython_env",
        "env.vpython_headless",
        "env.environment",
        "env.mjcf_builder",
        "env/base_env.py",
        "env/mujoco_env.py",
        "env/vpython_env.py",
        "env/vpython_headless.py",
        "env/environment.py",
        "env/mjcf_builder.py",
        "experiments.registry",
        "experiments.metrics",
        "experiments.llm_as_judge",
        "experiments.judge_reliability",
        "experiments.judge_metrics",
        "experiments.judge_prompts",
        "experiments/registry.py",
        "experiments/metrics.py",
        "experiments/llm_as_judge.py",
        "experiments/judge_reliability.py",
        "experiments/judge_metrics.py",
        "experiments/judge_prompts.py",
    ]
    search_roots = [
        root / "README.md",
        root / "CLAUDE.md",
        root / "conf",
        root / "env",
        root / "experiments",
        root / "llm",
        root / "providers",
        root / "runtime",
        root / "scripts",
        root / "tests",
        root / "utils",
        root / "main.py",
    ]
    offenders = []

    for removed_file in removed_files:
        assert not removed_file.exists()

    for path in search_roots:
        paths = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".md", ".sh"}
        )
        for candidate in paths:
            if candidate in {
                Path(__file__).resolve(),
                root / "tests" / "test_folder_docs.py",
                root / "tests" / "test_heatmap_and_scripts.py",
            }:
                continue
            text = candidate.read_text()
            if any(snippet in text for snippet in removed_snippets):
                offenders.append(str(candidate.relative_to(root)))

    assert offenders == []


def test_canonical_llm_exports_are_available_from_navigation_package():
    from llm import get_next_move
    from llm.navigation import RESPONSE_SCHEMA, _parse_json_reply
    from llm.navigation.parser import _parse_json_reply as shared_parse_json_reply
    from llm.navigation.schema import RESPONSE_SCHEMA as shared_response_schema
    from llm.navigation.service import get_next_move as shared_get_next_move

    assert get_next_move is shared_get_next_move
    assert RESPONSE_SCHEMA is shared_response_schema
    assert _parse_json_reply is shared_parse_json_reply


def test_judge_metrics_normalize_legacy_and_new_inputs():
    from experiments.evaluation.judge_metrics import METRIC_BY_CODE, normalize_metric_selection

    direct = normalize_metric_selection([METRIC_BY_CODE["M1"], METRIC_BY_CODE["M3"]])
    legacy = normalize_metric_selection(
        [("M1", METRIC_BY_CODE["M1"]), ("ignored", METRIC_BY_CODE["M3"])]
    )

    assert [metric.code for metric in direct] == ["M1", "M3"]
    assert [metric.code for metric in legacy] == ["M1", "M3"]


def test_judge_import_compatibility_surface():
    import experiments.evaluation.judge_reliability as judge_reliability
    import experiments.evaluation.judge as llm_as_judge

    assert llm_as_judge._METRIC_BY_NAME["M1"].code == "M1"
    assert "default" in llm_as_judge.get_prompt_variants()
    assert callable(judge_reliability.evaluate_judge_reliability)


def test_canonical_import_paths_resolve_to_public_surface():
    from llm import get_next_move as canonical_get_next_move
    from runtime.controller import Controller as canonical_controller
    import experiments.evaluation.judge as canonical_judge
    import providers.ollama as canonical_ollama

    assert callable(canonical_get_next_move)
    assert canonical_controller.__name__ == "Controller"
    assert callable(canonical_judge.judge_run)
    assert callable(canonical_ollama.chat)


def test_repo_no_longer_references_removed_apis_package():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    removed_import_prefix = "api" + "s."
    removed_path_prefix = "api" + "s/"
    search_roots = [
        root / "README.md",
        root / "llm",
        root / "providers",
        root / "runtime",
        root / "env",
        root / "experiments",
        root / "utils",
        root / "scripts",
        root / "tests",
    ]
    offenders = []

    for path in search_roots:
        paths = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".md", ".sh"}
        )
        for candidate in paths:
            if candidate == Path(__file__).resolve():
                continue
            text = candidate.read_text()
            if removed_path_prefix in text or removed_import_prefix in text:
                offenders.append(str(candidate.relative_to(root)))

    assert offenders == []


def test_repo_no_longer_references_removed_robots_package():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    search_roots = [
        root / "README.md",
        root / "CLAUDE.md",
        root / "conf",
        root / "env",
        root / "experiments",
        root / "llm",
        root / "providers",
        root / "runtime",
        root / "scripts",
        root / "tests",
        root / "utils",
    ]
    offenders = []
    removed_snippets = [
        "from robots",
        "import robots",
        "robots/",
        "ROBOTS.md",
    ]

    for path in search_roots:
        paths = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".md", ".sh"}
        )
        for candidate in paths:
            if candidate == Path(__file__).resolve():
                continue
            text = candidate.read_text()
            if any(snippet in text for snippet in removed_snippets):
                offenders.append(str(candidate.relative_to(root)))

    assert not (root / "robots").exists()
    assert offenders == []


def test_repo_no_longer_references_removed_llm_legacy_surface():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    removed_files = [
        root / "sitecustomize.py",
        root / "llm" / "chat.py",
        root / "llm" / "navigation_vlm.py",
        root / "llm" / "schemas.py",
        root / "llm" / "vlm_parser.py",
        root / "llm" / "ollama_client.py",
    ]
    search_roots = [
        root / "README.md",
        root / "CLAUDE.md",
        root / "conf",
        root / "env",
        root / "experiments",
        root / "llm",
        root / "providers",
        root / "runtime",
        root / "scripts",
        root / "tests",
        root / "utils",
        root / "main.py",
    ]
    offenders = []
    removed_snippets = [
        "from LLM",
        "import LLM",
        "LLM.chat",
        "LLM.navigation_vlm",
        "LLM.schemas",
        "LLM.vlm_parser",
        "LLM.dispatch",
        "LLM.navigation",
        "llm.chat",
        "llm.navigation_vlm",
        "llm.schemas",
        "llm.vlm_parser",
        "llm.ollama_client",
        "sitecustomize.py",
    ]

    for removed_file in removed_files:
        assert not removed_file.exists()

    for path in search_roots:
        paths = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".md", ".sh"}
        )
        for candidate in paths:
            if candidate == Path(__file__).resolve():
                continue
            text = candidate.read_text()
            if any(snippet in text for snippet in removed_snippets):
                offenders.append(str(candidate.relative_to(root)))

    assert offenders == []


def test_repo_no_longer_contains_removed_controllers_package():
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    search_roots = [
        root / "README.md",
        root / "conf",
        root / "env",
        root / "experiments",
        root / "llm",
        root / "providers",
        root / "runtime",
        root / "scripts",
        root / "tests",
        root / "utils",
        root / "main.py",
    ]
    offenders = []
    removed_snippets = [
        "controllers/",
        "from controllers",
        "import controllers",
    ]

    assert not (root / "controllers").exists()

    for path in search_roots:
        paths = [path] if path.is_file() else sorted(
            candidate for candidate in path.rglob("*")
            if candidate.is_file() and candidate.suffix in {".py", ".md", ".sh"}
        )
        for candidate in paths:
            if candidate == Path(__file__).resolve():
                continue
            text = candidate.read_text()
            if any(snippet in text for snippet in removed_snippets):
                offenders.append(str(candidate.relative_to(root)))

    assert offenders == []


def test_e1_e2_e3_experiment_spec_regression():
    from experiments.specs.registry import get_experiment

    e1 = get_experiment("e1")
    e2 = get_experiment("e2")
    e3 = get_experiment("e3")

    rs1 = e1.run_initializer(1)
    assert e1.task_id == "cube_task"
    assert rs1.scene_spec.mirror_enabled is True
    assert rs1.scene_spec.turn_step_deg == 30.0
    assert rs1.wrong_color is None

    rs2 = e2.run_initializer(1)
    assert e2.task_id == "cube_task"
    assert rs2.scene_spec.mirror_enabled is False
    assert rs2.scene_spec.turn_step_deg == 30.0
    assert rs2.wrong_color is None

    rs3 = e3.run_initializer(1)
    assert e3.task_id == "cube_task"
    assert rs3.scene_spec.mirror_enabled is True
    assert rs3.scene_spec.turn_step_deg == 30.0
    assert rs3.wrong_color is not None
    assert rs3.wrong_color != rs3.agent_color_name


def test_e0_experiment_spec_regression():
    from experiments.specs.registry import get_experiment

    e0 = get_experiment("e0")
    rs0 = e0.run_initializer(1)

    assert e0.task_id == "self_perception_task"
    assert rs0.scene_spec.scene_id == "mirror_room_direct_self_view"
    assert rs0.scene_spec.mirror_enabled is True
    assert rs0.scene_spec.spawn_pos == (-1.2, 0.0)
    assert rs0.scene_spec.spawn_yaw == 0.0
    assert rs0.scene_spec.mirror_window_z == (-7.5, 7.5)
    assert rs0.scene_spec.objects == []


def test_mujoco_asset_paths_resolve_from_new_canonical_locations():
    from env.backends import mujoco as mujoco_backend
    from env.scenes import mujoco_builder

    assert mujoco_backend._CUBE_XML_PATH.exists()
    assert mujoco_builder._HUMANOID_XML_PATH.exists()
