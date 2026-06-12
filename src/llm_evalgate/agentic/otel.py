from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .trace import AgentStep, AgentTrace, ToolCall

# Attribute names follow the OpenTelemetry GenAI semantic conventions. That spec
# is still incubating, so these are pinned to the 2026 snapshot; if upstream
# renames an attribute, the fixture-backed tests break in one place.
#   operation name: gen_ai.operation.name
#   tool ops mapped to ToolCall:   execute_tool
#   model ops that start a step:   chat, invoke_agent, generate_content,
#                                  text_completion (legacy op name, accepted)
#   tool attrs: gen_ai.tool.name / .call.arguments / .call.result
_OP = "gen_ai.operation.name"
_TOOL_NAME = "gen_ai.tool.name"
_TOOL_ARGS = "gen_ai.tool.call.arguments"
_TOOL_RESULT = "gen_ai.tool.call.result"

_TOOL_OPS = {"execute_tool"}
_MODEL_OPS = {"chat", "invoke_agent", "generate_content", "text_completion"}

# Candidate attributes that carry the user's request / goal, in priority order.
_GOAL_KEYS = (
    "gen_ai.prompt",
    "gen_ai.input.messages",
    "gen_ai.request.messages",
    "input.value",
    "input",
)
# Candidate attributes that carry a model span's output text.
_OUTPUT_KEYS = (
    "gen_ai.completion",
    "gen_ai.output.messages",
    "gen_ai.response.messages",
    "output.value",
    "output",
)


@dataclass
class OtelImportReport:
    spans_total: int = 0
    spans_mapped: int = 0
    spans_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


def trace_from_otel_spans(
    spans: list[dict],
    *,
    goal: str | None = None,
    return_report: bool = False,
) -> AgentTrace | tuple[AgentTrace, OtelImportReport]:
    """Build an :class:`AgentTrace` from exported OpenTelemetry GenAI spans.

    Input is a list of plain span dicts (the JSON shape an OTLP/console exporter
    emits), so no OpenTelemetry SDK dependency is needed. Model-call spans start
    new steps; tool spans (``gen_ai.operation.name == "execute_tool"``) attach to
    the current step. Unknown spans are skipped and counted. The ``goal`` is taken
    from the override, or best-effort from the first span carrying a request
    attribute; if neither is available a ``ValueError`` is raised.

    With ``return_report=True`` a ``(trace, OtelImportReport)`` tuple is returned.
    """
    report = OtelImportReport(spans_total=len(spans))
    ordered = sorted(spans, key=_start_time)

    resolved_goal = goal if goal is not None else _extract_goal(ordered)
    if resolved_goal is None:
        raise ValueError(
            "could not determine a goal from the spans; pass goal=... explicitly."
        )

    steps: list[AgentStep] = []
    final_answer: str | None = None

    for span in ordered:
        attrs = span.get("attributes", {})
        op = attrs.get(_OP)
        if op in _TOOL_OPS:
            call = _tool_call_from_span(span, attrs, report)
            if not steps:
                steps.append(AgentStep())  # orphan tool span: own step
            steps[-1].tool_calls.append(call)
            report.spans_mapped += 1
        elif op in _MODEL_OPS:
            text = _first_text(attrs, _OUTPUT_KEYS)
            steps.append(AgentStep(thought=text))
            if text is not None:
                final_answer = text
            report.spans_mapped += 1
        else:
            report.spans_skipped += 1

    if report.spans_skipped:
        report.warnings.append(
            f"skipped {report.spans_skipped} span(s) with no recognized "
            "gen_ai.operation.name."
        )

    trace = AgentTrace(goal=resolved_goal, steps=steps, final_answer=final_answer)
    return (trace, report) if return_report else trace


def _start_time(span: dict) -> int:
    value = span.get("start_time_unix_nano")
    return int(value) if value is not None else 0


def _extract_goal(spans: list[dict]) -> str | None:
    for span in spans:
        attrs = span.get("attributes", {})
        text = _first_text(attrs, _GOAL_KEYS)
        if text:
            return text
    return None


def _first_text(attrs: dict, keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if key in attrs and attrs[key] is not None:
            return _stringify(attrs[key])
    return None


def _stringify(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = [_message_text(item) for item in value]
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        return _message_text(value)
    return str(value)


def _message_text(item: Any) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("content", "text", "value"):
            if key in item and isinstance(item[key], str):
                return item[key]
        return json.dumps(item, sort_keys=True)
    return str(item)


def _tool_call_from_span(span: dict, attrs: dict, report: OtelImportReport) -> ToolCall:
    name = attrs.get(_TOOL_NAME) or span.get("name", "unknown_tool")
    args = _parse_args(attrs.get(_TOOL_ARGS), name, report)
    result = attrs.get(_TOOL_RESULT)
    if result is None:
        result = _result_from_events(span)
    error = _error_from_status(span)
    return ToolCall(name=name, args=args, result=result, error=error)


def _parse_args(raw: Any, tool_name: str, report: OtelImportReport) -> dict:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return dict(raw)  # copy so the ToolCall does not alias the source span
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            report.warnings.append(
                f"tool '{tool_name}': arguments were not valid JSON; used empty args."
            )
            return {}
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    return {}


def _result_from_events(span: dict) -> Any:
    for event in span.get("events", []):
        attrs = event.get("attributes", {})
        if _TOOL_RESULT in attrs:
            return attrs[_TOOL_RESULT]
    return None


def _error_from_status(span: dict) -> str | None:
    status = span.get("status", {})
    code = str(status.get("code", "")).upper()
    if "ERROR" in code:
        return status.get("message") or "tool span reported ERROR status"
    return None
