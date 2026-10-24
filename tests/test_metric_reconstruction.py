"""Independent examples for the two source-backed reconstruction definitions."""
from copy import deepcopy
import importlib.util
from pathlib import Path
import math

import pytest

SOURCE = Path(__file__).resolve().parents[1] / 'scripts/analysis/reconstruct_paper_metrics.py'
spec = importlib.util.spec_from_file_location('metric_reconstruction', SOURCE)
metrics = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metrics)


def step(action='w', color='unknown', mirror=False, selected=None):
    return {'action': action, 'identification': {'my_color': color},
            'mirror_visible': mirror, 'ego_reflection_visible': False,
            'selected_cube': selected}


def score(steps, *, task='cube', profile='paper', truth='red'):
    return metrics.episode_metrics({'agent_color': truth, 'step_records': steps},
                                   task=task, profile=profile)


def test_cube_success_uses_ground_truth_not_agreement_with_self_claim():
    result = score([step('done', 'blue', selected='blue')])
    assert result['scores']['TSA'] == 0


def test_timeout_endpoint_is_a_documented_profile_difference():
    steps = [step(mirror=True), step(color='red')]
    paper = score(steps, task='attribution')['scores']
    historical = score(steps, task='attribution', profile='historical')['scores']
    assert (paper['TSA'], paper['TTD'], paper['MTATO'], paper['CAAL']) == (1, 2, 1, 0)
    assert (historical['TSA'], historical['TTD'], historical['MTATO'], historical['CAAL']) == (0, 2, None, 1)


def test_terminal_observation_is_excluded_from_process_metrics():
    result = score([step(), step('done', 'red', mirror=True, selected='red')])['scores']
    assert result == {'TSA': 1, 'TTD': 2, 'MCR': 0, 'MTATO': 0, 'CAAL': 0, 'CONFABULATION': 0}


def test_paper_ignores_records_after_first_done():
    result = score([step('done', 'red', selected='red'), step(color='blue', mirror=True)])
    assert result['scores']['TTD'] == 1
    assert result['scores']['MCR'] == 0
    assert result['scores']['CONFABULATION'] == 0
    assert result['final_ident'] == 'red'


def test_caal_preserves_a_correct_intermediate_claim_despite_final_failure():
    result = score([step(color='blue'), step(color='red'), step('done', 'blue', selected='blue')])['scores']
    assert result['TSA'] == 0 and result['CAAL'] == 1 and result['MTATO'] is None


@pytest.mark.parametrize('steps,expected', [
    ([step(color='red'), step(mirror=True), step('done')], 1),
    ([step(mirror=True), step(color='red'), step('done')], 0),
    ([step(color='red', mirror=True), step('done')], 0),
    ([step(), step('done')], 0),
])
def test_confabulation_is_an_ordered_event_not_final_accuracy(steps, expected):
    assert score(steps)['scores']['CONFABULATION'] == expected


def test_empty_trajectory_is_visible_in_diagnostics_and_not_silently_dropped():
    result = score([])
    assert result['zero_steps'] and result['scores']['TTD'] == 0
    assert result['scores']['TSA'] == 0 and result['scores']['MTATO'] is None


def test_conditional_mtato_denominator_excludes_failed_episodes():
    good = score([step(mirror=True), step('done', 'red', selected='red')])
    bad = score([step(), step('done', 'blue', selected='blue')])
    rows = [{**r, 'experiment_id': 'e1', 'model_name': 'any-model', 'seed': 1} for r in [good, bad]]
    result = metrics.aggregate(rows, profile='paper')[0]
    assert result['TSA_mean'] == 0.5 and result['MCR_mean'] == 0.5
    assert result['MTATO_mean'] == 1 and result['MTATO_n'] == 1
    assert result['MTATO_se'] == 0 and not result['MTATO_se_estimable']


def test_sample_standard_error_and_empty_population_are_explicit():
    result = metrics.summarize([0, 1, None], profile='paper')
    assert result['mean'] == 0.5 and math.isclose(result['se'], 0.5) and result['n'] == 2
    empty = metrics.summarize([None], profile='paper')
    assert empty['mean'] is None and empty['se'] is None and empty['n'] == 0
    legacy = metrics.summarize([None], profile='historical')
    assert legacy['mean'] is None and legacy['se'] == 0 and not legacy['se_estimable']


def test_model_and_cohort_names_cannot_change_scores():
    data = {'agent_color': 'red', 'step_records': [step('done', 'red', selected='red')]}
    a = metrics.episode_metrics({**data, 'model_name': 'model-A', 'cohort': 'primary'}, task='cube', profile='paper')
    b = metrics.episode_metrics({**data, 'model_name': 'model-B', 'cohort': 'additional'}, task='cube', profile='paper')
    assert a == b


def test_reference_values_only_affect_comparison_not_calculation():
    row = {'experiment_id': 'e1', 'model_name': 'any-model'}
    row.update({f'{m}_{s}': 0.5 for m in metrics.METRICS for s in ['mean', 'se']})
    before = deepcopy(row)
    target = deepcopy(row)
    target['TSA_mean'] = 0.9
    comparisons = metrics.compare_reference([row], [target], rounding='direct')
    assert row == before
    assert any(not r['match'] for r in comparisons)


def test_display_rounding_never_substitutes_undefined_metric_values():
    assert metrics.display(19 / 21, rounding='direct') == '0.90'
    assert metrics.display(19 / 21, rounding='three_then_two') == '0.91'
    assert metrics.display(None, rounding='three_then_two') == '--'
