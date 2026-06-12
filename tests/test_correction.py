from __future__ import annotations

import pytest

from llm_evalgate.judge.calibration import CalibrationReport
from llm_evalgate.judge.correction import (
    CorrectedRate,
    corrected_pass_rate,
    rogan_gladen,
)


def test_rogan_gladen_worked_example():
    # sens=0.9, spec=0.8, observed=0.69 -> ~0.70
    corrected = rogan_gladen(0.69, 0.9, 0.8)
    assert abs(corrected - 0.70) < 0.005


def test_rogan_gladen_perfect_judge_is_identity():
    assert rogan_gladen(0.42, 1.0, 1.0) == pytest.approx(0.42)


def test_rogan_gladen_clamps_to_unit_interval():
    assert rogan_gladen(0.0, 0.9, 0.5) == 0.0   # would go negative
    assert rogan_gladen(1.0, 0.5, 0.9) == 1.0   # would exceed 1


def test_rogan_gladen_raises_at_or_below_chance():
    with pytest.raises(ValueError):
        rogan_gladen(0.5, 0.5, 0.5)  # sens+spec == 1
    with pytest.raises(ValueError):
        rogan_gladen(0.5, 0.3, 0.4)  # sens+spec < 1


def _calibration(tp, fn, tn, fp):
    n_pos = tp + fn
    n_neg = tn + fp
    return CalibrationReport(
        n=n_pos + n_neg,
        pearson=None,
        spearman=None,
        mae=None,
        accuracy=None,
        cohen_kappa=None,
        sensitivity=tp / n_pos if n_pos else None,
        specificity=tn / n_neg if n_neg else None,
        confusion={"tp": tp, "fn": fn, "tn": tn, "fp": fp},
    )


def test_corrected_pass_rate_basic():
    cal = _calibration(tp=90, fn=10, tn=80, fp=20)  # sens 0.9, spec 0.8
    judge_labels = [True] * 69 + [False] * 31  # observed 0.69
    result = corrected_pass_rate(judge_labels, cal, n_resamples=500, seed=0)
    assert isinstance(result, CorrectedRate)
    assert abs(result.observed - 0.69) < 1e-9
    assert abs(result.corrected - 0.70) < 0.01
    assert result.ci_low <= result.corrected <= result.ci_high


def test_perfect_judge_correction_equals_observed():
    cal = _calibration(tp=100, fn=0, tn=100, fp=0)  # sens 1.0, spec 1.0
    judge_labels = [True] * 7 + [False] * 3
    result = corrected_pass_rate(judge_labels, cal, n_resamples=300, seed=0)
    assert abs(result.corrected - 0.7) < 1e-9


def test_ci_widens_with_smaller_calibration_set():
    judge_labels = [True] * 70 + [False] * 30
    big = _calibration(tp=180, fn=20, tn=160, fp=40)    # n_cal = 400
    small = _calibration(tp=18, fn=2, tn=16, fp=4)      # n_cal = 40, same rates
    wide = corrected_pass_rate(judge_labels, small, n_resamples=800, seed=0)
    narrow = corrected_pass_rate(judge_labels, big, n_resamples=800, seed=0)
    assert (wide.ci_high - wide.ci_low) > (narrow.ci_high - narrow.ci_low)


def test_requires_confusion_matrix():
    cal = CalibrationReport(
        n=10, pearson=None, spearman=None, mae=None,
        accuracy=0.8, cohen_kappa=0.6, confusion=None,
    )
    with pytest.raises(ValueError):
        corrected_pass_rate([True, False], cal)


def test_requires_both_classes_in_calibration():
    cal = _calibration(tp=10, fn=5, tn=0, fp=0)  # no human-negative cases
    with pytest.raises(ValueError):
        corrected_pass_rate([True, False], cal)


def test_empty_judge_labels_raises():
    cal = _calibration(tp=9, fn=1, tn=8, fp=2)
    with pytest.raises(ValueError):
        corrected_pass_rate([], cal)


def test_sub_chance_point_calibration_warns_not_raises():
    # sens 0.4, spec 0.5 -> sens+spec = 0.9 <= 1: correction undefined.
    cal = _calibration(tp=4, fn=6, tn=5, fp=5)
    judge_labels = [True] * 6 + [False] * 4
    result = corrected_pass_rate(judge_labels, cal, n_resamples=200, seed=0)
    assert result.warning is not None
    assert "chance" in result.warning
    # falls back to the uncorrected observed rate
    assert result.corrected == result.observed == 0.6
