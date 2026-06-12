from __future__ import annotations

from dataclasses import dataclass

from ..trace import AgentTrace
from .base import TraceDimension


@dataclass(frozen=True)
class ToolSchema:
    """The argument contract for one tool the agent is allowed to call."""

    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()

    @classmethod
    def from_json_schema(cls, schema: dict) -> ToolSchema:
        """Build a ToolSchema from a standard JSON-Schema tool definition.

        Reads ``properties`` (all known arg names) and ``required`` (the subset
        that must be present) -- the shape used by Anthropic and OpenAI tool
        definitions, so an existing tool spec maps in one line. The schema may be
        the tool's ``input_schema``/``parameters`` or a wrapper dict containing it.
        """
        inner = schema.get("input_schema", schema.get("parameters", schema))
        properties = set(inner.get("properties", {}))
        required = set(inner.get("required", []))
        return cls(
            required=frozenset(required),
            optional=frozenset(properties - required),
        )


class ToolHallucinationDimension(TraceDimension):
    """Catch agents calling tools that do not exist or with malformed arguments.

    A call is hallucinated when the tool is not in the registry, a required
    argument is missing, or (unless ``allow_extra_args``) it passes an argument
    the schema does not declare. Score is the fraction of non-hallucinated calls;
    with no tool calls the dimension passes. Default threshold is 1.0 (zero
    tolerance), since a phantom tool call is a hard correctness failure.
    """

    def __init__(
        self,
        tools: dict[str, ToolSchema],
        *,
        allow_extra_args: bool = False,
        threshold: float = 1.0,
        name: str = "tool_hallucination",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._tools = tools
        self._allow_extra_args = allow_extra_args

    def _issues(self, name: str, args: dict) -> list[str]:
        """Return the hallucination issues for a single call (empty == clean)."""
        schema = self._tools.get(name)
        if schema is None:
            return ["unknown tool"]
        issues: list[str] = []
        arg_names = set(args)
        missing = schema.required - arg_names
        if missing:
            issues.append(f"missing required arg(s) {sorted(missing)}")
        if not self._allow_extra_args:
            extra = arg_names - (schema.required | schema.optional)
            if extra:
                issues.append(f"unexpected arg(s) {sorted(extra)}")
        return issues

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        calls = trace.all_tool_calls()
        if not calls:
            return 1.0, "no tool calls"

        reasons: list[str] = []
        hallucinated = 0
        for call in calls:
            issues = self._issues(call.name, call.args)
            if issues:
                hallucinated += 1
                reasons.append(f"{call.name} ({'; '.join(issues)})")

        score = (len(calls) - hallucinated) / len(calls)
        detail = f"{hallucinated}/{len(calls)} calls hallucinated"
        if reasons:
            detail += "; " + "; ".join(reasons)
        return score, detail
