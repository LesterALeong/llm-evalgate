from __future__ import annotations

import pytest

from llm_evalgate.bench.gate import GateReport, RegressionGate
from llm_evalgate.bench.runner import BenchmarkResult, BenchmarkRunner, BenchSample


class _Grader:
    """Grader that passes a text iff it does not contain ``bad``."""

    def run(self, text: str):
        from llm_evalgate.eval.dimension import DimensionResult

        passed = "bad" not in text
        return DimensionResult(score=1.0 if passed else 0.0, passed=passed, detail="")


def _result(predicted, labels, fingerprint="fp", n=None):
    return BenchmarkResult(
        predicted=list(predicted),
        labels=list(labels),
        metrics={},
        n=n if n is not None else len(predicted),
        dataset_fingerprint=fingerprint,
    )


def test_save_load_roundtrip(tmp_path):
    samples = [BenchSample("good one", True), BenchSample("a bad one", False)]
    result = BenchmarkRunner(_Grader()).run(samples, n_resamples=100, seed=0)
    path = tmp_path / "baseline.json"
    result.save(path)
    loaded = BenchmarkResult.load(path)
    assert loaded.predicted == result.predicted
    assert loaded.labels == result.labels
    assert loaded.metrics == result.metrics
    assert loaded.dataset_fingerprint == result.dataset_fingerprint
    assert loaded.n == result.n


def test_identical_runs_pass_with_zero_delta():
    pred = [True, False, True, True, False, False, True, True, False, True]
    label = [True, False, True, True, False, True, True, False, False, True]
    cur = _result(pred, label)
    base = _result(pred, label)
    gate = RegressionGate(metrics="all", n_resamples=200, seed=0)
    report = gate.check(cur, base)
    assert report.passed
    assert all(row.delta == 0.0 for row in report.rows)


def _synthetic(n, n_wrong):
    """n samples, all label True; first n_wrong predicted False (wrong)."""
    labels = [True] * n
    predicted = [False] * n_wrong + [True] * (n - n_wrong)
    return predicted, labels


def test_large_regression_fails_on_big_dataset():
    base_pred, labels = _synthetic(200, 4)   # accuracy 0.98
    cur_pred, _ = _synthetic(200, 24)        # accuracy 0.88 -> -0.10
    base = _result(base_pred, labels)
    cur = _result(cur_pred, labels)
    gate = RegressionGate(metrics=("accuracy",), threshold=0.02, n_resamples=500, seed=0)
    report = gate.check(cur, base)
    assert not report.passed
    assert report.rows[0].verdict == "fail"
    # regressed indices: where baseline passed (True) but current failed (False)
    assert set(report.regressed_samples) == set(range(4, 24))


def test_small_dataset_regression_warns_not_fails():
    # Same ~10pp drop but on n=24, with significance required -> WARN.
    base_pred, labels = _synthetic(24, 0)   # accuracy 1.0
    cur_pred, _ = _synthetic(24, 3)         # accuracy 0.875 -> -0.125
    base = _result(base_pred, labels)
    cur = _result(cur_pred, labels)
    gate = RegressionGate(
        metrics=("accuracy",), threshold=0.02, require_significance=True,
        n_resamples=500, seed=0,
    )
    report = gate.check(cur, base)
    # delta is past threshold but the one-sided p on such a small set is 0.054 -> warn
    assert report.rows[0].verdict == "warn"
    # power warning must fire for n=24 with a 2pp threshold
    assert any("minimum detectable effect" in w for w in report.warnings)


def test_no_significance_fails_on_any_threshold_breach():
    base_pred, labels = _synthetic(24, 0)
    cur_pred, _ = _synthetic(24, 3)
    gate = RegressionGate(
        metrics=("accuracy",), threshold=0.02, require_significance=False,
        n_resamples=300, seed=0,
    )
    report = gate.check(_result(cur_pred, labels), _result(base_pred, labels))
    assert not report.passed
    assert report.rows[0].verdict == "fail"


def test_fingerprint_mismatch_raises():
    cur = _result([True, False], [True, False], fingerprint="A")
    base = _result([True, False], [True, False], fingerprint="B")
    gate = RegressionGate(metrics=("accuracy",), n_resamples=100, seed=0)
    with pytest.raises(ValueError):
        gate.check(cur, base)


def test_allow_unpaired_proceeds_with_warning():
    cur = _result([True, False, True, True], [True, False, True, True], fingerprint="A")
    base = _result([True, True, True, True], [True, True, True, True], fingerprint="B")
    gate = RegressionGate(
        metrics=("accuracy",), n_resamples=200, seed=0, allow_unpaired=True
    )
    report = gate.check(cur, base)
    assert any("unpaired" in w for w in report.warnings)
    assert report.regressed_samples == []  # unavailable when unpaired


def test_unknown_metric_raises():
    with pytest.raises(ValueError):
        RegressionGate(metrics=("not_a_metric",))


def test_gate_report_table_renders():
    report = GateReport(passed=True, rows=[], regressed_samples=[], warnings=[])
    assert "GateReport: PASS" in report.table()


def _balanced_perfect_vs_small_dip(n=100, flips=4):
    """Perfect baseline vs a current with a small, broad regression.

    Labels alternate True/False; the baseline predicts every sample correctly,
    the current flips ``flips`` of the True-labelled predictions to False. That
    dips accuracy/recall/f1/kappa a few points each -- a near-null change where
    several metrics breach a 2pp threshold with individually-borderline p-values,
    which is exactly where uncorrected multiplicity inflates false failures.
    """
    labels = [i % 2 == 0 for i in range(n)]
    base_pred = list(labels)
    cur_pred = list(labels)
    for i in [j for j in range(n) if labels[j]][:flips]:
        cur_pred[i] = not cur_pred[i]
    return cur_pred, base_pred, labels


def test_correction_default_is_holm():
    cur_pred, base_pred, labels = _balanced_perfect_vs_small_dip()
    report = RegressionGate(metrics="all", n_resamples=200, seed=0).check(
        _result(cur_pred, labels), _result(base_pred, labels)
    )
    assert report.correction == "holm"


def test_correction_turns_spurious_multimetric_fail_into_pass():
    cur_pred, base_pred, labels = _balanced_perfect_vs_small_dip(n=100, flips=4)
    cur, base = _result(cur_pred, labels), _result(base_pred, labels)
    none_report = RegressionGate(
        metrics="all", threshold=0.02, correction="none", n_resamples=1000, seed=0
    ).check(cur, base)
    holm_report = RegressionGate(
        metrics="all", threshold=0.02, correction="holm", n_resamples=1000, seed=0
    ).check(cur, base)
    assert not none_report.passed   # uncorrected: ≥1 metric individually significant
    assert holm_report.passed       # Holm: the same drops are within family noise
    none_fails = {r.metric for r in none_report.rows if r.verdict == "fail"}
    holm_fails = {r.metric for r in holm_report.rows if r.verdict == "fail"}
    assert none_fails               # the demonstration is non-trivial
    assert holm_fails <= none_fails  # correction only ever removes failures


def test_rows_carry_pvalues_and_monotone_adjustment():
    cur_pred, base_pred, labels = _balanced_perfect_vs_small_dip()
    report = RegressionGate(
        metrics="all", correction="holm", n_resamples=300, seed=0
    ).check(_result(cur_pred, labels), _result(base_pred, labels))
    for row in report.rows:
        assert 0.0 <= row.p_value <= 1.0
        assert row.adjusted_p >= row.p_value - 1e-9


def test_bh_alias_is_normalized_in_report():
    cur_pred, base_pred, labels = _balanced_perfect_vs_small_dip()
    report = RegressionGate(
        metrics="all", correction="bh", n_resamples=200, seed=0
    ).check(_result(cur_pred, labels), _result(base_pred, labels))
    assert report.correction == "benjamini_hochberg"


def test_unknown_correction_raises():
    with pytest.raises(ValueError):
        RegressionGate(metrics="all", correction="bonferroni")


def test_single_metric_correction_is_a_noop():
    # With one metric there is nothing to correct: Holm and none must agree.
    base_pred, labels = _synthetic(200, 4)
    cur_pred, _ = _synthetic(200, 24)
    cur, base = _result(cur_pred, labels), _result(base_pred, labels)
    holm_v = (
        RegressionGate(metrics=("accuracy",), correction="holm", n_resamples=300, seed=0)
        .check(cur, base)
        .rows[0]
        .verdict
    )
    none_v = (
        RegressionGate(metrics=("accuracy",), correction="none", n_resamples=300, seed=0)
        .check(cur, base)
        .rows[0]
        .verdict
    )
    assert holm_v == none_v == "fail"


def test_stratified_gate_runs_with_correction():
    cur_pred, base_pred, labels = _balanced_perfect_vs_small_dip()
    metas = [{"group": "a" if i < 50 else "b"} for i in range(len(labels))]
    cur = BenchmarkResult(
        predicted=cur_pred, labels=labels, metrics={}, n=len(labels),
        dataset_fingerprint="fp", metas=metas,
    )
    base = BenchmarkResult(
        predicted=base_pred, labels=labels, metrics={}, n=len(labels),
        dataset_fingerprint="fp", metas=metas,
    )
    report = RegressionGate(metrics="all", n_resamples=200, seed=0).check(
        cur, base, stratify_by="group"
    )
    assert set(report.stratum_rows) == {"a", "b"}
    for rows in report.stratum_rows.values():
        for row in rows:
            assert row.adjusted_p >= row.p_value - 1e-9


def test_cli_exit_codes(tmp_path):
    from llm_evalgate.gate.__main__ import main

    base_pred, labels = _synthetic(200, 4)
    cur_pred, _ = _synthetic(200, 24)
    base = _result(base_pred, labels)
    cur = _result(cur_pred, labels)
    base_path = tmp_path / "base.json"
    cur_path = tmp_path / "cur.json"
    base.save(base_path)
    cur.save(cur_path)

    fail_code = main([str(cur_path), str(base_path), "--metrics", "accuracy",
                      "--threshold", "0.02"])
    assert fail_code == 1
    pass_code = main([str(base_path), str(base_path), "--metrics", "accuracy"])
    assert pass_code == 0
