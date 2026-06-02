from __future__ import annotations

from ..trace import AgentTrace
from .base import TraceDimension


class StepEfficiencyDimension(TraceDimension):
    """Score the agent on staying within a step budget without repeating work.

    Base score is 1.0 when ``num_steps <= max_steps``, else ``max_steps /
    num_steps``. When ``penalize_repeats`` is set, repeated identical
    ``(name, sorted-args)`` tool calls subtract ``repeats / total_calls``
    from the score, floored at 0.0.
    """

    def __init__(
        self,
        max_steps: int,
        *,
        penalize_repeats: bool = True,
        threshold: float = 1.0,
        name: str = "step_efficiency",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._max_steps = max_steps
        self._penalize_repeats = penalize_repeats

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        num_steps = trace.num_steps
        if num_steps <= self._max_steps:
            score = 1.0
        else:
            score = self._max_steps / num_steps
        repeats = 0
        if self._penalize_repeats:
            calls = trace.all_tool_calls()
            seen: set[tuple[str, tuple[tuple[str, object], ...]]] = set()
            for call in calls:
                key = (call.name, tuple(sorted(call.args.items())))
                if key in seen:
                    repeats += 1
                else:
                    seen.add(key)
            if repeats and calls:
                score = max(0.0, score - repeats / len(calls))
        detail = f"{num_steps} steps vs budget {self._max_steps}; {repeats} repeated calls"
        return score, detail
