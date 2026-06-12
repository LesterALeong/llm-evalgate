from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_evalgate.agentic.dimensions import ToolSelectionDimension
from llm_evalgate.agentic.harness import AgentEvalHarness
from llm_evalgate.agentic.otel import trace_from_otel_spans
from llm_evalgate.agentic.trace import AgentTrace

FIXTURE = Path(__file__).parent / "fixtures" / "otel_agent_run.json"


def _load_fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def test_imports_two_step_run():
    trace = AgentTrace.from_otel(_load_fixture())
    assert trace.goal == "Find the weather in Paris and convert it to Fahrenheit."
    assert trace.num_steps == 2
    names = trace.tool_names()
    assert names == ["get_weather", "celsius_to_f"]
    assert trace.final_answer == "It is about 64.4F in Paris."


def test_tool_args_and_results_parsed():
    trace = AgentTrace.from_otel(_load_fixture())
    calls = trace.all_tool_calls()
    assert calls[0].args == {"city": "Paris"}
    assert calls[0].result == "18C"
    assert calls[1].args == {"c": 18}


def test_error_status_populates_error():
    spans = [
        {"attributes": {"gen_ai.operation.name": "chat", "gen_ai.prompt": "do it",
                        "gen_ai.completion": "ok"}, "start_time_unix_nano": 1},
        {"attributes": {"gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": "{}"},
         "status": {"code": "ERROR", "message": "timeout"}, "start_time_unix_nano": 2},
    ]
    trace = trace_from_otel_spans(spans)
    call = trace.all_tool_calls()[0]
    assert call.error == "timeout"


def test_malformed_arguments_warn_not_raise():
    spans = [
        {"attributes": {"gen_ai.operation.name": "chat", "gen_ai.prompt": "g",
                        "gen_ai.completion": "ok"}, "start_time_unix_nano": 1},
        {"attributes": {"gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "search",
                        "gen_ai.tool.call.arguments": "{not json"},
         "start_time_unix_nano": 2},
    ]
    trace, report = trace_from_otel_spans(spans, return_report=True)
    assert trace.all_tool_calls()[0].args == {}
    assert any("not valid JSON" in w for w in report.warnings)


def test_unknown_spans_skipped_and_counted():
    trace, report = trace_from_otel_spans(_load_fixture(), return_report=True)
    assert report.spans_total == 5
    assert report.spans_skipped == 1  # the postgres bookkeeping span
    assert report.spans_mapped == 4
    assert report.spans_mapped + report.spans_skipped == report.spans_total


def test_missing_goal_raises():
    spans = [
        {"attributes": {"gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": "{}"},
         "start_time_unix_nano": 1},
    ]
    with pytest.raises(ValueError):
        trace_from_otel_spans(spans)


def test_goal_override_wins():
    spans = [
        {"attributes": {"gen_ai.operation.name": "execute_tool",
                        "gen_ai.tool.name": "search", "gen_ai.tool.call.arguments": "{}"},
         "start_time_unix_nano": 1},
    ]
    trace = trace_from_otel_spans(spans, goal="my explicit goal")
    assert trace.goal == "my explicit goal"
    # orphan tool span (no preceding model span) gets its own step
    assert trace.num_steps == 1


def test_imported_trace_runs_through_harness():
    trace = AgentTrace.from_otel(_load_fixture())
    harness = AgentEvalHarness([
        ToolSelectionDimension(expected_tools=["get_weather", "celsius_to_f"], mode="order"),
    ])
    report = harness.run(trace)
    assert report.passed
