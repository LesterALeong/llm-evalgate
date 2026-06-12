from __future__ import annotations

from dataclasses import dataclass

from ..eval.dimension import DimensionResult
from .dimensions.base import TraceDimension
from .trace import AgentTrace


@dataclass
class AgentEvalReport:
    passed: bool
    results: dict[str, DimensionResult]
    trace: AgentTrace

    @property
    def failures(self) -> dict[str, DimensionResult]:
        return {name: r for name, r in self.results.items() if not r.passed}

    @property
    def needs_review(self) -> dict[str, DimensionResult]:
        """Dimensions flagged as too uncertain to trust without a human look."""
        return {name: r for name, r in self.results.items() if r.needs_review}

    def __str__(self) -> str:
        header = f"AgentEvalReport: {'PASS' if self.passed else 'FAIL'}"
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


class AgentEvalHarness:
    """Run a list of trace dimensions against a trace and produce a report.

    Usage::

        harness = AgentEvalHarness([
            ToolSelectionDimension(expected_tools=["search", "calculator"]),
            StepEfficiencyDimension(max_steps=5),
        ])
        report = harness.run(trace)
        if not report.passed:
            raise ValueError(str(report))
    """

    def __init__(self, dimensions: list[TraceDimension]) -> None:
        if not dimensions:
            raise ValueError("AgentEvalHarness requires at least one dimension.")
        self._dimensions = dimensions

    def run(self, trace: AgentTrace) -> AgentEvalReport:
        results: dict[str, DimensionResult] = {}
        for dim in self._dimensions:
            results[dim.name] = dim.run(trace)
        passed = all(r.passed for r in results.values())
        return AgentEvalReport(passed=passed, results=results, trace=trace)
