from __future__ import annotations

from llm_evalgate.agentic.dimensions.tool_hallucination import (
    ToolHallucinationDimension,
    ToolSchema,
)
from llm_evalgate.agentic.harness import AgentEvalHarness
from llm_evalgate.agentic.trace import AgentStep, AgentTrace, ToolCall


def _trace(*calls):
    return AgentTrace(
        goal="g",
        steps=[AgentStep(tool_calls=list(calls))],
        final_answer="done",
    )


REGISTRY = {
    "get_weather": ToolSchema(required=frozenset({"city"}), optional=frozenset({"units"})),
    "celsius_to_f": ToolSchema(required=frozenset({"c"})),
}


def test_phantom_tool_detected():
    trace = _trace(ToolCall(name="get_wether", args={"city": "Paris"}))
    score, detail = ToolHallucinationDimension(REGISTRY).evaluate(trace)
    assert score == 0.0
    assert "unknown tool" in detail


def test_missing_required_arg_detected():
    trace = _trace(ToolCall(name="get_weather", args={"units": "C"}))
    score, detail = ToolHallucinationDimension(REGISTRY).evaluate(trace)
    assert score == 0.0
    assert "missing required arg" in detail
    assert "city" in detail


def test_unexpected_arg_detected():
    trace = _trace(ToolCall(name="celsius_to_f", args={"c": 18, "bogus": 1}))
    score, detail = ToolHallucinationDimension(REGISTRY).evaluate(trace)
    assert score == 0.0
    assert "unexpected arg" in detail


def test_allow_extra_args_suppresses_extra_only():
    trace = _trace(ToolCall(name="celsius_to_f", args={"c": 18, "bogus": 1}))
    score, _ = ToolHallucinationDimension(REGISTRY, allow_extra_args=True).evaluate(trace)
    assert score == 1.0
    # but still catches missing required even with allow_extra_args
    trace2 = _trace(ToolCall(name="celsius_to_f", args={"bogus": 1}))
    score2, _ = ToolHallucinationDimension(REGISTRY, allow_extra_args=True).evaluate(trace2)
    assert score2 == 0.0


def test_clean_trace_passes():
    trace = _trace(
        ToolCall(name="get_weather", args={"city": "Paris", "units": "C"}),
        ToolCall(name="celsius_to_f", args={"c": 18}),
    )
    dim = ToolHallucinationDimension(REGISTRY)
    score, _ = dim.evaluate(trace)
    assert score == 1.0
    assert dim.run(trace).passed


def test_empty_trace_passes():
    trace = AgentTrace(goal="g", steps=[], final_answer="x")
    score, detail = ToolHallucinationDimension(REGISTRY).evaluate(trace)
    assert score == 1.0
    assert "no tool calls" in detail


def test_partial_score():
    trace = _trace(
        ToolCall(name="get_weather", args={"city": "Paris"}),  # clean
        ToolCall(name="phantom", args={}),                      # bad
    )
    score, _ = ToolHallucinationDimension(REGISTRY).evaluate(trace)
    assert score == 0.5


def test_from_json_schema_anthropic_style():
    tool_def = {
        "name": "get_weather",
        "description": "Look up weather",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string"}, "units": {"type": "string"}},
            "required": ["city"],
        },
    }
    schema = ToolSchema.from_json_schema(tool_def)
    assert schema.required == frozenset({"city"})
    assert schema.optional == frozenset({"units"})


def test_from_json_schema_openai_parameters_key():
    tool_def = {
        "parameters": {
            "properties": {"q": {}, "limit": {}},
            "required": ["q"],
        }
    }
    schema = ToolSchema.from_json_schema(tool_def)
    assert schema.required == frozenset({"q"})
    assert schema.optional == frozenset({"limit"})


def test_composes_in_agent_harness():
    trace = _trace(ToolCall(name="get_weather", args={"city": "Paris"}))
    harness = AgentEvalHarness([ToolHallucinationDimension(REGISTRY)])
    report = harness.run(trace)
    assert report.passed
