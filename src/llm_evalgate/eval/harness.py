from __future__ import annotations

from dataclasses import dataclass

from .dimension import Dimension, DimensionResult


@dataclass
class EvalReport:
    passed: bool
    results: dict[str, DimensionResult]
    text: str

    @property
    def failures(self) -> dict[str, DimensionResult]:
        return {name: r for name, r in self.results.items() if not r.passed}

    @property
    def needs_review(self) -> dict[str, DimensionResult]:
        """Dimensions flagged as too uncertain to trust without a human look."""
        return {name: r for name, r in self.results.items() if r.needs_review}

    def __str__(self) -> str:
        header = f"EvalReport: {'PASS' if self.passed else 'FAIL'}"
        flagged = len(self.needs_review)
        if flagged:
            header += f" ({flagged} flagged for review)"
        lines = [header]
        for name, result in self.results.items():
            status = "PASS" if result.passed else "FAIL"
            marker = " REVIEW" if result.needs_review else ""
            lines.append(
                f"  {status}{marker} [{name}] score={result.score:.3f} - {result.detail}"
            )
        return "\n".join(lines)


class EvalHarness:
    """Run a list of dimensions against text and produce an EvalReport.

    Usage::

        harness = EvalHarness([
            ReadabilityDimension(threshold=0.7),
            BlocklistDimension(terms=["secret", "internal"]),
        ])
        report = harness.run(text)
        if not report.passed:
            raise ValueError(str(report))
    """

    def __init__(self, dimensions: list[Dimension]) -> None:
        if not dimensions:
            raise ValueError("EvalHarness requires at least one dimension.")
        self._dimensions = dimensions

    def run(self, text: str) -> EvalReport:
        results: dict[str, DimensionResult] = {}
        for dim in self._dimensions:
            results[dim.name] = dim.run(text)
        passed = all(r.passed for r in results.values())
        return EvalReport(passed=passed, results=results, text=text)
