from __future__ import annotations

from ..trace import AgentTrace
from .base import TraceDimension

_MODES = ("subset", "exact", "order")


class ToolSelectionDimension(TraceDimension):
    """Score whether the agent used the expected tools.

    ``mode`` controls the comparison:

    - ``"subset"``: fraction of expected tools present in the trace.
    - ``"exact"``: 1.0 iff the set of used tools equals the expected set.
    - ``"order"``: 1.0 iff expected tools appear as an in-order subsequence.
    """

    def __init__(
        self,
        expected_tools: list[str],
        *,
        mode: str = "subset",
        threshold: float = 1.0,
        name: str = "tool_selection",
    ) -> None:
        if mode not in _MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {_MODES}")
        super().__init__(threshold=threshold, name=name)
        self._expected = expected_tools
        self._mode = mode

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        actual = trace.tool_names()
        actual_set = set(actual)
        matched = [tool for tool in self._expected if tool in actual_set]
        missing = [tool for tool in self._expected if tool not in actual_set]
        if self._mode == "exact":
            score = 1.0 if set(self._expected) == actual_set else 0.0
        elif self._mode == "order":
            score = 1.0 if self._is_subsequence(self._expected, actual) else 0.0
        else:
            score = len(matched) / len(self._expected) if self._expected else 1.0
        detail = f"mode={self._mode}; matched={matched}; missing={missing}"
        return score, detail

    @staticmethod
    def _is_subsequence(expected: list[str], actual: list[str]) -> bool:
        iterator = iter(actual)
        return all(tool in iterator for tool in expected)
