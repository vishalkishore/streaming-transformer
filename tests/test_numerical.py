import numpy as np
import pytest

from reference.numerical import state_errors, score_errors


def test_complete_layer_gate_is_not_a_short_head_condition_number_test():
    reference = np.ones((4, 3, 32), dtype=np.float32)
    reference[0, 0] = .01
    actual = reference.copy()
    actual[0, 0] += .00003
    head_relative = np.linalg.norm(actual[0, 0] - reference[0, 0]) / np.linalg.norm(reference[0, 0])
    assert head_relative > .002
    result = state_errors(actual, reference, layer=2)
    assert result['passed']
    assert result['relative_l2'] < .002


def test_wrong_ring_slot_and_broad_drift_are_rejected():
    reference = np.random.default_rng(123).normal(size=(4, 7, 32)).astype('f4')
    wrong_slot = reference.copy()
    wrong_slot[:, [0, 1]] = wrong_slot[:, [1, 0]]
    assert not state_errors(wrong_slot, reference, layer=3)['passed']
    assert not state_errors(reference + .02, reference, layer=3)['passed']


def test_nonfinite_and_fp32_errors_remain_failures():
    reference = np.ones(128, dtype='f4')
    actual = reference.copy()
    actual[0] = np.nan
    assert not state_errors(actual, reference, layer=3)['passed']
    actual = reference.copy()
    actual[0] += .001
    assert not state_errors(actual, reference, layer=3, mixed=False)['passed']
    with pytest.raises(ValueError, match='shapes differ'):
        state_errors(np.ones(32), reference, layer=0)


def test_score_gate_uses_absolute_floor_near_zero_and_rejects_drift():
    reference = np.full(128, .04)
    result = score_errors(reference + .0003, reference, layers=2)
    assert result['relative_l2'] > .002
    assert result['normalized_rmse'] < .002
    assert result['passed']
    assert not score_errors(reference + .01, reference, layers=2)['passed']
    assert not score_errors(np.full(128, 10.03), np.full(128, 10.), layers=4)['passed']
