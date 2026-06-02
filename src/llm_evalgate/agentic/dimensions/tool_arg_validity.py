from __future__ import annotations

from collections.abc import Callable

from ..trace import AgentTrace
from .base import TraceDimension


class ToolArgValidityDimension(TraceDimension):
    """Score the fraction of tool calls with valid arguments.

    A tool call is invalid if ``call.error`` is set, or a validator exists
    for ``call.name`` and returns False on ``call.args``. Score is the
    fraction of valid calls. With no tool calls the dimension passes.
    """

    def __init__(
        self,
        validators: dict[str, Callable[[dict], bool]] | None = None,
        *,
        threshold: float = 1.0,
        name: str = "tool_arg_validity",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._validators = validators or {}

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        calls = trace.all_tool_calls()
        if not calls:
            return 1.0, "no tool calls"
        reasons: list[str] = []
        for call in calls:
            if call.error is not None:
                reasons.append(f"{call.name}: error {call.error}")
                continue
            validator = self._validators.get(call.name)
            if validator is not None and not validator(call.args):
                reasons.append(f"{call.name}: invalid args {call.args}")
        invalid = len(reasons)
        score = (len(calls) - invalid) / len(calls)
        detail = f"{invalid}/{len(calls)} invalid"
        if reasons:
            detail += "; " + "; ".join(reasons)
        return score, detail
