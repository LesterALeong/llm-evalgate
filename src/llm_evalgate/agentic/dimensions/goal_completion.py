from __future__ import annotations

from collections.abc import Callable

from ..trace import AgentTrace
from .base import TraceDimension


class GoalCompletionDimension(TraceDimension):
    """Score whether the agent completed its goal.

    If ``checker`` is given it decides pass (1.0) or fail (0.0). Otherwise if
    ``required_substrings`` is given the score is the fraction present in the
    final answer (case-insensitive). Otherwise the score is 1.0 when the
    final answer is non-empty, else 0.0.
    """

    def __init__(
        self,
        checker: Callable[[AgentTrace], bool] | None = None,
        required_substrings: list[str] | None = None,
        *,
        threshold: float = 1.0,
        name: str = "goal_completion",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._checker = checker
        self._required_substrings = required_substrings

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        if self._checker is not None:
            passed = self._checker(trace)
            return (1.0 if passed else 0.0), f"checker -> {passed}"
        if self._required_substrings:
            answer = (trace.final_answer or "").lower()
            present = [s for s in self._required_substrings if s.lower() in answer]
            score = len(present) / len(self._required_substrings)
            return score, f"{len(present)}/{len(self._required_substrings)} substrings present"
        has_answer = bool(trace.final_answer)
        return (1.0 if has_answer else 0.0), f"final answer present: {has_answer}"
