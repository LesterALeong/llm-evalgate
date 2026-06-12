import math

import pytest

from llm_evalgate.bench import (
    BenchmarkResult,
    BenchmarkRunner,
    BenchSample,
    accuracy,
    all_metrics,
    cohen_kappa,
    confusion_counts,
    f1,
    load_golden,
    precision,
    recall,
    regression_catch_rate,
)
from llm_evalgate.eval import EvalHarness
from llm_evalgate.eval.dimensions import (
    BlocklistDimension,
    ReadabilityDimension,
    SchemaComplianceDimension,
)

# Hand-computed vectors: tp=2, fp=1, tn=2, fn=1.
PREDICTED = [True, True, False, False, True, False]
LABELS = [True, False, False, True, True, False]


# --- metrics correctness ---

def test_confusion_counts():
    assert confusion_counts(PREDICTED, LABELS) == {"tp": 2, "fp": 1, "tn": 2, "fn": 1}


def test_accuracy():
    assert accuracy(PREDICTED, LABELS) == pytest.approx(4 / 6)


def test_precision_recall_f1():
    assert precision(PREDICTED, LABELS) == pytest.approx(2 / 3)
    assert recall(PREDICTED, LABELS) == pytest.approx(2 / 3)
    assert f1(PREDICTED, LABELS) == pytest.approx(2 / 3)


def test_regression_catch_rate():
    # Negative-class recall: tn / (tn + fp) = 2 / 3.
    assert regression_catch_rate(PREDICTED, LABELS) == pytest.approx(2 / 3)


def test_cohen_kappa_partial():
    # observed=4/6, expected=0.5 -> kappa = (0.6667 - 0.5) / 0.5 = 0.3333.
    assert cohen_kappa(PREDICTED, LABELS) == pytest.approx(1 / 3, abs=1e-9)


def test_cohen_kappa_perfect_agreement():
    preds = [True, False, True, False]
    assert cohen_kappa(preds, preds) == pytest.approx(1.0)


def test_all_metrics_keys():
    metrics = all_metrics(PREDICTED, LABELS)
    assert set(metrics) == {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "cohen_kappa",
        "regression_catch_rate",
    }
    assert all(isinstance(v, float) for v in metrics.values())


# --- zero-denominator behavior ---

def test_precision_zero_denominator_returns_zero():
    # No positive predictions -> precision denominator is zero.
    assert precision([False, False], [True, False]) == 0.0


def test_regression_catch_rate_no_regressions_returns_zero():
    assert regression_catch_rate([True, True], [True, True]) == 0.0


# --- input validation ---

def test_length_mismatch_raises():
    with pytest.raises(ValueError):
        accuracy([True, False], [True])


def test_empty_inputs_raise():
    with pytest.raises(ValueError):
        accuracy([], [])


# --- BenchmarkRunner over an in-memory dataset ---

def test_runner_with_blocklist_dimension():
    samples = [
        BenchSample(text="This is a public document.", label=True),
        BenchSample(text="This is an internal document.", label=False),
        BenchSample(text="Another clean public note.", label=True),
    ]
    runner = BenchmarkRunner(BlocklistDimension(terms=["internal"]))
    result = runner.run(samples)
    assert result.predicted == [True, False, True]
    assert result.labels == [True, False, True]
    # All three predictions match their labels -> perfect scores.
    assert result.metrics["accuracy"] == pytest.approx(1.0)
    assert result.metrics["regression_catch_rate"] == pytest.approx(1.0)
    assert result.n == 3


def test_runner_empty_samples_raises():
    runner = BenchmarkRunner(BlocklistDimension(terms=["internal"]))
    with pytest.raises(ValueError):
        runner.run([])


def test_benchmark_result_table_is_aligned():
    result = BenchmarkResult(
        predicted=[True],
        labels=[True],
        metrics={"accuracy": 1.0, "f1": 0.5},
        n=1,
    )
    table = result.table()
    assert "n=1" in table
    assert "1.000" in table
    assert "0.500" in table


# --- golden dataset end to end ---

def test_load_golden_non_empty():
    dataset = load_golden()
    assert isinstance(dataset, list)
    assert len(dataset) > 0
    assert all(isinstance(s, BenchSample) for s in dataset)


def test_runner_over_golden_dataset():
    dataset = load_golden()
    runner = BenchmarkRunner(BlocklistDimension(terms=["confidential"]))
    result = runner.run(dataset)
    assert isinstance(result, BenchmarkResult)
    assert result.n == len(dataset)
    assert len(result.predicted) == len(dataset)
    assert all(not math.isnan(v) for v in result.metrics.values())


def test_golden_deterministic_metrics_are_locked():
    """Pin the README/article benchmark numbers to CI.

    The deterministic harness here is the canonical one from
    examples/benchmark.py. These six values are quoted in the README and the
    article, so an unpinned textstat bump or a dataset edit that moved them must
    fail the suite rather than silently invalidate the writeup.
    """
    harness = EvalHarness([
        BlocklistDimension(terms=["confidential", "internal use only", "[REDACTED]"]),
        SchemaComplianceDimension(required_fields=["title:", "summary:"]),
        ReadabilityDimension(threshold=0.2),
    ])
    result = BenchmarkRunner(harness).run(load_golden(), ci=False)

    assert result.n == 60
    assert result.metrics["accuracy"] == pytest.approx(0.850, abs=1e-3)
    assert result.metrics["precision"] == pytest.approx(0.775, abs=1e-3)
    assert result.metrics["recall"] == pytest.approx(1.000, abs=1e-3)
    assert result.metrics["f1"] == pytest.approx(0.873, abs=1e-3)
    assert result.metrics["cohen_kappa"] == pytest.approx(0.697, abs=1e-3)
    assert result.metrics["regression_catch_rate"] == pytest.approx(0.690, abs=1e-3)


def test_golden_stratification():
    """Per-stratum metrics expose where the deterministic harness fails.

    The semantic stratum is the one deterministic gates cannot catch, so its
    regression-catch-rate is 0 while policy/format/readability are perfect.
    """
    harness = EvalHarness([
        BlocklistDimension(terms=["confidential", "internal use only", "[REDACTED]"]),
        SchemaComplianceDimension(required_fields=["title:", "summary:"]),
        ReadabilityDimension(threshold=0.2),
    ])
    result = BenchmarkRunner(harness).run(
        load_golden(), ci=False, stratify_by="stratum"
    )
    assert result.strata is not None
    assert set(result.strata) >= {
        "clean", "policy", "format", "readability", "semantic", "edge"
    }
    # Deterministic gates miss every semantic regression.
    assert result.strata["semantic"].metrics["regression_catch_rate"] == pytest.approx(0.0)
    # But catch every policy violation.
    assert result.strata["policy"].metrics["regression_catch_rate"] == pytest.approx(1.0)
    # Small strata are flagged in the rendered table.
    table = result.table()
    assert "stratum=semantic" in table


def test_stratify_unknown_key_groups_as_none():
    samples = [
        BenchSample("public a", True, meta={}),
        BenchSample("public b", True, meta={}),
    ]
    result = BenchmarkRunner(BlocklistDimension(terms=["secret"])).run(
        samples, ci=False, stratify_by="stratum"
    )
    assert "(none)" in result.strata
