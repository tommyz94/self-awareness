"""Official E1-E5 are isolated from explicitly loaded research prototypes."""
import json
from pathlib import Path
import subprocess
import sys

from hydra import compose, initialize_config_dir
from hydra.errors import MissingConfigException
from omegaconf import OmegaConf
import pytest

from experiments import discover_experiments, get_experiment, invalidate_experiment_cache
from experiments.naming import EXPERIMENTAL_EXPERIMENTS

ROOT = Path(__file__).resolve().parents[1]


def test_official_discovery_and_tasks_have_only_paper_ids():
    registry = discover_experiments(force_reload=True)
    assert set(registry) == {"e1", "e2", "e3", "e4", "e5"}
    assert all(spec.num_runs_default == 7 for spec in registry.values())
    assert [registry[key].task_id for key in sorted(registry)] == ["cube_task"] * 4 + ["explore_task"]
    assert not (ROOT / "experiments" / "specs").exists()


def test_official_loading_does_not_import_experimental_plugins_in_a_clean_process():
    script = """import json,sys
from experiments import discover_experiments
assert set(discover_experiments()) == {'e1','e2','e3','e4','e5'}
print(json.dumps([name for name in sys.modules if name.startswith('experiments.experimental.')]))
"""
    result = subprocess.run([sys.executable, "-c", script], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == []


def test_broken_unrequested_prototype_cannot_break_official_experiments(monkeypatch):
    import experiments.loader as loader
    original = loader._load_experiment_module
    imported = []

    def restricted(name):
        imported.append(name)
        if name.startswith("experiments.experimental."):
            raise ImportError("An unfinished prototype has unavailable dependencies")
        return original(name)

    invalidate_experiment_cache()
    monkeypatch.setattr(loader, "_load_experiment_module", restricted)
    try:
        assert get_experiment("e4").task_id == "cube_task"
        assert get_experiment("e5").task_id == "explore_task"
        assert all(name.startswith("experiments.paper.") for name in imported)
        with pytest.raises(ImportError, match="unfinished prototype"):
            get_experiment("experimental/front_identity")
    finally:
        invalidate_experiment_cache()


@pytest.mark.parametrize("paper_id", ["e1", "e2", "e3", "e4", "e5"])
def test_numeric_and_explicit_paper_presets_are_identical(paper_id):
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        plain = compose(config_name="config", overrides=[f"experiment={paper_id}"])
        explicit = compose(config_name="config", overrides=[f"experiment=paper/{paper_id}"])
    assert OmegaConf.to_container(plain) == OmegaConf.to_container(explicit)
    assert plain.experiment_id == plain.paper_experiment_id == paper_id
    assert plain.nav != "auto"


@pytest.mark.parametrize("retired", ["e0", "e6", "e7", "e8", "e4_1", "extras/corridor"])
def test_retired_runtime_selectors_fail_explicitly(retired):
    with pytest.raises(ValueError, match="Unknown experiment"):
        get_experiment(retired)
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        with pytest.raises(MissingConfigException):
            compose(config_name="config", overrides=[f"experiment={retired}"])


@pytest.mark.parametrize("name", EXPERIMENTAL_EXPERIMENTS)
def test_every_preserved_prototype_requires_explicit_selection(name):
    selected = f"experimental/{name}"
    with initialize_config_dir(config_dir=str(ROOT / "conf"), version_base=None):
        cfg = compose(config_name="config", overrides=[f"experiment={selected}"])
    assert cfg.experiment_id == get_experiment(selected).experiment_id == selected
    assert cfg.outputs.dir == "outputs/experimental"
    assert selected not in discover_experiments()


def test_paper_alias_inherits_launcher_protocol_locks():
    from scripts.dev.openrouter_matrix import _preset_locked_keys
    for paper_id in ["e1", "e2", "e3", "e4", "e5"]:
        direct = _preset_locked_keys(ROOT, paper_id)
        alias = _preset_locked_keys(ROOT, f"paper/{paper_id}")
        assert direct == alias
        assert {"max_steps", "num_runs", "history_steps", "experiment_id"} <= alias


def test_experimental_summary_preserves_its_explicit_identity_scheme(tmp_path):
    from experiments.evaluation.metrics import RunOutcome
    from utils.io.json_export import save_experiment_json
    identifier = "experimental/exploration"
    context = {"experiment_suite": "experimental", "experiment_id_scheme": "experimental-v1"}
    outcome = RunOutcome(1, "red", None, None, False, 1,
                         experiment_id=identifier, task_id="explore_task", run_context=context)
    path = save_experiment_json([outcome], identifier, str(tmp_path))
    data = json.loads(Path(path).read_text())
    assert data["experiment_id"] == identifier
    assert data["run_context"] == context
    assert "paper_experiment_id" not in data
