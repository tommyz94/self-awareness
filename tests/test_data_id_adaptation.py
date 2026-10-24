"""Data-version boundaries must not confuse old cube E5 and new paper E5."""
import pytest

from experiments.evaluation.data_ids import calculator_id


def test_original_and_new_e5_have_different_scoring_task_ids():
    assert calculator_id("e5", {}) == "e5"
    assert calculator_id("e5", {"run_context": {"experiment_id_scheme": "paper-v1"}}) == "e7"


def test_old_paper_alias_metadata_does_not_reinterpret_original_data():
    assert calculator_id("e5", {"paper_experiment_id": "e4"}) == "e5"
    assert calculator_id("e7", {"paper_experiment_id": "e5"}) == "e7"


def test_explicit_manifest_scheme_can_identify_new_canonical_data():
    assert calculator_id("e4", {}, scheme="paper-v1") == "e5"


@pytest.mark.parametrize("metadata,scheme", [
    ({"experiment_id_scheme": "paper-v1"}, "legacy-v1"),
    ({"run_context": {"experiment_id_scheme": "paper-v1", "paper_experiment_id": "e5"}}, None),
    ({"run_context": "invalid"}, None),
    ({"experiment_id_scheme": "experimental-v1"}, None),
])
def test_ambiguous_or_conflicting_data_is_rejected(metadata, scheme):
    with pytest.raises(ValueError):
        calculator_id("e4", metadata, scheme=scheme)
