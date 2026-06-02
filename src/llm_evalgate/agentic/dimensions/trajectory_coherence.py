from __future__ import annotations

from ...judge import JudgeDimension
from ..trace import AgentTrace
from .base import TraceDimension


class TrajectoryCoherenceDimension(TraceDimension):
    """Grade long-horizon reasoning coherence with an LLM-as-judge.

    The trace is serialized with ``AgentTrace.to_text`` and passed to the
    injected ``JudgeDimension``, whose ``(score, detail)`` is returned
    directly. Injecting the judge keeps this dimension offline-testable.
    """

    def __init__(
        self,
        judge: JudgeDimension,
        *,
        threshold: float = 0.6,
        name: str = "trajectory_coherence",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._judge = judge

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        return self._judge.evaluate(trace.to_text())
