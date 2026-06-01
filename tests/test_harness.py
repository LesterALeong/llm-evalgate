import pytest

from llm_evalgate import Dimension, EvalHarness


class AlwaysPassDimension(Dimension):
    def evaluate(self, text: str) -> tuple[float, str]:
        return 1.0, "always passes"


class AlwaysFailDimension(Dimension):
    def evaluate(self, text: str) -> tuple[float, str]:
        return 0.0, "always fails"


class HalfScoreDimension(Dimension):
    def evaluate(self, text: str) -> tuple[float, str]:
        return 0.5, "half score"


def test_harness_all_pass():
    harness = EvalHarness([AlwaysPassDimension(name="d1"), AlwaysPassDimension(name="d2")])
    report = harness.run("some text")
    assert report.passed
    assert report.failures == {}


def test_harness_one_fail():
    harness = EvalHarness([AlwaysPassDimension(name="pass"), AlwaysFailDimension(name="fail")])
    report = harness.run("some text")
    assert not report.passed
    assert "fail" in report.failures
    assert "pass" not in report.failures


def test_harness_threshold():
    dim = HalfScoreDimension(threshold=0.4, name="half")
    harness = EvalHarness([dim])
    report = harness.run("text")
    assert report.passed

    dim_strict = HalfScoreDimension(threshold=0.6, name="half_strict")
    harness2 = EvalHarness([dim_strict])
    report2 = harness2.run("text")
    assert not report2.passed


def test_harness_empty_dimensions_raises():
    with pytest.raises(ValueError):
        EvalHarness([])


def test_report_str_contains_pass_fail():
    harness = EvalHarness([AlwaysPassDimension(name="p"), AlwaysFailDimension(name="f")])
    report = harness.run("x")
    s = str(report)
    assert "FAIL" in s
    assert "PASS" in s
