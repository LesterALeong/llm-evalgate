from __future__ import annotations

import pytest

from llm_evalgate.bench.metrics import accuracy, regression_catch_rate
from llm_evalgate.bench.stats import (
    ConfidenceInterval,
    bootstrap_ci,
    min_detectable_effect,
    required_sample_size,
)


def test_bootstrap_ci_is_deterministic_for_fixed_seed():
    pred = [True, False, True, True, False, True, False, True]
    label = [True, False, False, True, False, True, True, True]
    a = bootstrap_ci(accuracy, pred, label, n_resamples=500, seed=0)
    b = bootstrap_ci(accuracy, pred, label, n_resamples=500, seed=0)
    assert a == b
    assert isinstance(a, ConfidenceInterval)


def test_bootstrap_ci_brackets_point_estimate():
    pred = [True, True, True, False, False, True, True, False]
    label = [True, True, False, False, False, True, False, True]
    ci = bootstrap_ci(accuracy, pred, label, n_resamples=1000, seed=1)
    assert ci.low <= ci.point <= ci.high
    assert 0.0 <= ci.low <= 1.0
    assert 0.0 <= ci.high <= 1.0


def test_bootstrap_ci_all_correct_has_tight_interval():
    pred = [True, False, True, False]
    label = [True, False, True, False]
    ci = bootstrap_ci(accuracy, pred, label, n_resamples=500, seed=0)
    assert ci.point == 1.0
    assert ci.low == 1.0
    assert ci.high == 1.0


def test_bootstrap_ci_validates_inputs():
    with pytest.raises(ValueError):
        bootstrap_ci(accuracy, [True], [True, False])
    with pytest.raises(ValueError):
        bootstrap_ci(accuracy, [], [])
    with pytest.raises(ValueError):
        bootstrap_ci(accuracy, [True], [True], n_resamples=0)
    with pytest.raises(ValueError):
        bootstrap_ci(accuracy, [True], [True], alpha=1.5)


def test_bootstrap_ci_works_on_regression_catch_rate():
    pred = [False, False, True, True, False, True]
    label = [False, False, False, True, False, True]
    ci = bootstrap_ci(regression_catch_rate, pred, label, n_resamples=400, seed=3)
    assert ci.low <= ci.point <= ci.high


def test_min_detectable_effect_cannot_see_2pp_at_24_samples():
    # At n=24 the MDE for a 50/50 metric is far above a 2pp gate threshold.
    mde = min_detectable_effect(24, baseline=0.5)
    assert mde > 0.02
    assert 0.25 < mde < 0.32  # ~0.286 from the normal approximation


def test_min_detectable_effect_shrinks_with_n():
    assert min_detectable_effect(1000) < min_detectable_effect(100)


def test_required_sample_size_for_2pp_at_baseline_90():
    n = required_sample_size(0.02, baseline=0.9)
    assert 1500 <= n <= 2000  # ~1764


def test_required_sample_size_inverts_mde():
    # The n that just detects an effect should make MDE(n) ~= that effect.
    effect = 0.05
    n = required_sample_size(effect, baseline=0.5)
    assert min_detectable_effect(n, baseline=0.5) <= effect + 1e-6


def test_power_helpers_validate_inputs():
    with pytest.raises(ValueError):
        min_detectable_effect(0)
    with pytest.raises(ValueError):
        required_sample_size(0.0)
    with pytest.raises(ValueError):
        required_sample_size(0.02, baseline=1.5)
