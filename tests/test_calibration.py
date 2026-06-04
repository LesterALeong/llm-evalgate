import pytest

from llm_evalgate.bench.metrics import mae, pearson, spearman
from llm_evalgate.judge import JudgeDimension
from llm_evalgate.judge.calibration import (
    CalibrationReport,
    CalibrationSample,
    calibrate_judge,
    verbosity_bias,
)


def _judge_from_scores(scores: dict[str, int]) -> JudgeDimension:
    """Build a deterministic judge keyed on a marker substring in the text.

    The text is rendered into the prompt, so the fake ``complete`` can scan the
    prompt for whichever key it contains and return that key's raw score.
    """

    def complete(prompt: str) -> str:
        for key, value in scores.items():
            if key in prompt:
                return f"SCORE: {value}\nREASON: keyed on {key}"
        return "SCORE: 0\nREASON: no key found"

    return JudgeDimension(complete=complete, rubric="r", scale=5, threshold=0.6)


def _length_judge() -> JudgeDimension:
    """A judge whose 1-5 score grows with the text length."""

    def complete(prompt: str) -> str:
        # The text is the prompt tail after the "Text:" marker.
        text = prompt.split("Text:\n", 1)[1]
        raw = min(5, 1 + len(text) // 10)
        return f"SCORE: {raw}\nREASON: length {len(text)}"

    return JudgeDimension(complete=complete, rubric="r", scale=5, threshold=0.6)


def _constant_judge(raw: int) -> JudgeDimension:
    def complete(prompt: str) -> str:
        return f"SCORE: {raw}\nREASON: constant"

    return JudgeDimension(complete=complete, rubric="r", scale=5, threshold=0.6)


# --- pearson / spearman / mae unit tests ---

def test_pearson_perfect_positive():
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)


def test_pearson_perfect_negative():
    assert pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


def test_pearson_zero_variance_returns_zero():
    assert pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) == 0.0


def test_spearman_monotonic_nonlinear_is_one():
    # Perfectly monotonic but nonlinear -> Spearman is 1.0, Pearson is not.
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 4.0, 9.0, 16.0]) == pytest.approx(1.0)


def test_spearman_handles_ties_with_average_ranks():
    # Identical orderings with a tie -> still perfect rank correlation.
    assert spearman([1.0, 2.0, 2.0, 3.0], [10.0, 20.0, 20.0, 30.0]) == pytest.approx(1.0)


def test_mae_known_value():
    assert mae([1.0, 2.0, 3.0], [1.0, 4.0, 3.0]) == pytest.approx(2 / 3)


def test_correlation_length_mismatch_raises():
    with pytest.raises(ValueError):
        pearson([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        spearman([1.0, 2.0], [1.0])
    with pytest.raises(ValueError):
        mae([1.0, 2.0], [1.0])


def test_correlation_empty_raises():
    with pytest.raises(ValueError):
        pearson([], [])
    with pytest.raises(ValueError):
        spearman([], [])
    with pytest.raises(ValueError):
        mae([], [])


# --- calibrate_judge: score-level only ---

def test_calibrate_judge_score_level_tracks_humans():
    # Judge raw scores 2,3,4,5 -> normalized 0.4,0.6,0.8,1.0 vs humans that
    # rise in lockstep -> pearson near 1.0 and a small mae.
    judge = _judge_from_scores({"AAA": 2, "BBB": 3, "CCC": 4, "DDD": 5})
    samples = [
        CalibrationSample(text="AAA", human_score=0.4),
        CalibrationSample(text="BBB", human_score=0.6),
        CalibrationSample(text="CCC", human_score=0.8),
        CalibrationSample(text="DDD", human_score=1.0),
    ]
    report = calibrate_judge(judge, samples)
    assert report.n == 4
    assert report.pearson == pytest.approx(1.0)
    assert report.spearman == pytest.approx(1.0)
    assert report.mae == pytest.approx(0.0, abs=1e-9)
    assert report.accuracy is None
    assert report.cohen_kappa is None


# --- calibrate_judge: label-level only ---

def test_calibrate_judge_label_level_only():
    judge = _judge_from_scores({"PASS": 5, "FAIL": 1})
    samples = [
        CalibrationSample(text="PASS", human_label=True),
        CalibrationSample(text="FAIL", human_label=False),
    ]
    report = calibrate_judge(judge, samples)
    assert report.accuracy == pytest.approx(1.0)
    assert report.cohen_kappa == pytest.approx(1.0)
    assert report.pearson is None
    assert report.spearman is None
    assert report.mae is None


# --- calibrate_judge: both ---

def test_calibrate_judge_both_levels_populated():
    judge = _judge_from_scores({"AAA": 5, "BBB": 1})
    samples = [
        CalibrationSample(text="AAA", human_score=1.0, human_label=True),
        CalibrationSample(text="BBB", human_score=0.2, human_label=False),
    ]
    report = calibrate_judge(judge, samples)
    assert report.pearson is not None
    assert report.spearman is not None
    assert report.mae is not None
    assert report.accuracy is not None
    assert report.cohen_kappa is not None


# --- calibrate_judge: validation ---

def test_calibrate_judge_empty_samples_raises():
    judge = _constant_judge(3)
    with pytest.raises(ValueError):
        calibrate_judge(judge, [])


def test_calibrate_judge_sample_with_neither_raises():
    judge = _constant_judge(3)
    samples = [CalibrationSample(text="AAA", human_score=0.5), CalibrationSample(text="BBB")]
    with pytest.raises(ValueError):
        calibrate_judge(judge, samples)


# --- verbosity_bias ---

def test_verbosity_bias_positive_for_length_judge():
    texts = ["a", "abcdefghij", "abcdefghijabcdefghij", "abcdefghijabcdefghijabcdefghij"]
    assert verbosity_bias(_length_judge(), texts) > 0.9


def test_verbosity_bias_constant_judge_returns_zero():
    texts = ["a", "abcdefghij", "abcdefghijabcdefghij"]
    assert verbosity_bias(_constant_judge(3), texts) == 0.0


def test_verbosity_bias_empty_texts_raises():
    with pytest.raises(ValueError):
        verbosity_bias(_constant_judge(3), [])


# --- CalibrationReport.table ---

def test_calibration_report_table_omits_none_rows():
    report = CalibrationReport(
        n=5,
        pearson=0.912,
        spearman=None,
        mae=0.041,
        accuracy=None,
        cohen_kappa=None,
    )
    table = report.table()
    assert "n=5" in table
    assert "0.912" in table
    assert "0.041" in table
    assert "spearman" not in table
    assert "accuracy" not in table
    assert "cohen_kappa" not in table
