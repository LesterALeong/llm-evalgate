import pytest

from llm_evalgate.agentic import (
    AgentEvalHarness,
    AgentStep,
    AgentTrace,
    GoalCompletionDimension,
    StepEfficiencyDimension,
    ToolArgValidityDimension,
    ToolCall,
    ToolSelectionDimension,
    TrajectoryCoherenceDimension,
)
from llm_evalgate.judge import JudgeDimension


def _trace(steps, *, goal="goal", final_answer="done"):
    return AgentTrace(goal=goal, steps=steps, final_answer=final_answer)


def _step(*names, **kwargs):
    args = kwargs.get("args", {})
    return AgentStep(tool_calls=[ToolCall(name=n, args=args) for n in names])


# --- ToolSelectionDimension ---

def test_tool_selection_subset_partial():
    trace = _trace([_step("search")])
    dim = ToolSelectionDimension(["search", "calculator"], mode="subset")
    result = dim.run(trace)
    assert result.score == 0.5
    assert not result.passed
    assert "calculator" in result.detail


def test_tool_selection_subset_full_passes():
    trace = _trace([_step("search", "calculator")])
    dim = ToolSelectionDimension(["search", "calculator"], mode="subset")
    result = dim.run(trace)
    assert result.passed
    assert result.score == 1.0


def test_tool_selection_exact_pass_and_fail():
    expected = ["search", "calculator"]
    passing = _trace([_step("calculator", "search")])
    failing = _trace([_step("search", "calculator", "browser")])
    dim = ToolSelectionDimension(expected, mode="exact")
    assert dim.run(passing).passed
    assert not dim.run(failing).passed


def test_tool_selection_order_pass_and_fail():
    dim = ToolSelectionDimension(["search", "calculator"], mode="order")
    in_order = _trace([_step("search"), _step("browser"), _step("calculator")])
    out_of_order = _trace([_step("calculator"), _step("search")])
    assert dim.run(in_order).passed
    assert not dim.run(out_of_order).passed


def test_tool_selection_unknown_mode_raises():
    with pytest.raises(ValueError):
        ToolSelectionDimension(["search"], mode="bogus")


# --- ToolArgValidityDimension ---

def test_tool_arg_validity_validator_fails():
    trace = AgentTrace(
        goal="g",
        steps=[
            AgentStep(tool_calls=[ToolCall(name="search", args={"query": "x"})]),
            AgentStep(tool_calls=[ToolCall(name="search", args={})]),
        ],
    )
    dim = ToolArgValidityDimension({"search": lambda args: bool(args.get("query"))})
    result = dim.run(trace)
    assert result.score == 0.5
    assert not result.passed
    assert "invalid" in result.detail


def test_tool_arg_validity_errored_call():
    trace = AgentTrace(
        goal="g",
        steps=[
            AgentStep(tool_calls=[ToolCall(name="search", args={"query": "x"})]),
            AgentStep(tool_calls=[ToolCall(name="search", error="timeout")]),
        ],
    )
    dim = ToolArgValidityDimension()
    result = dim.run(trace)
    assert result.score == 0.5
    assert "timeout" in result.detail


def test_tool_arg_validity_no_calls_passes():
    trace = AgentTrace(goal="g", steps=[AgentStep(thought="thinking")])
    result = ToolArgValidityDimension().run(trace)
    assert result.passed
    assert "no tool calls" in result.detail


# --- StepEfficiencyDimension ---

def test_step_efficiency_under_budget_passes():
    trace = _trace([_step("a"), _step("b")])
    result = StepEfficiencyDimension(max_steps=4).run(trace)
    assert result.passed
    assert result.score == 1.0


def test_step_efficiency_over_budget_fails():
    trace = _trace([_step("a"), _step("b"), _step("c"), _step("d")])
    result = StepEfficiencyDimension(max_steps=2).run(trace)
    assert not result.passed
    assert result.score == 0.5


def test_step_efficiency_penalizes_repeats():
    trace = AgentTrace(
        goal="g",
        steps=[
            AgentStep(tool_calls=[ToolCall(name="search", args={"q": "x"})]),
            AgentStep(tool_calls=[ToolCall(name="search", args={"q": "x"})]),
        ],
    )
    result = StepEfficiencyDimension(max_steps=4).run(trace)
    assert result.score == 0.5
    assert "1 repeated calls" in result.detail


def test_step_efficiency_no_penalty_when_disabled():
    trace = AgentTrace(
        goal="g",
        steps=[
            AgentStep(tool_calls=[ToolCall(name="search", args={"q": "x"})]),
            AgentStep(tool_calls=[ToolCall(name="search", args={"q": "x"})]),
        ],
    )
    result = StepEfficiencyDimension(max_steps=4, penalize_repeats=False).run(trace)
    assert result.score == 1.0


# --- GoalCompletionDimension ---

def test_goal_completion_checker():
    trace = _trace([_step("a")], final_answer="anything")
    passing = GoalCompletionDimension(checker=lambda t: t.num_steps == 1)
    failing = GoalCompletionDimension(checker=lambda t: t.num_steps == 5)
    assert passing.run(trace).passed
    assert not failing.run(trace).passed


def test_goal_completion_substrings():
    trace = _trace([_step("a")], final_answer="The answer is 68 Million people.")
    dim = GoalCompletionDimension(required_substrings=["68 million", "missing"])
    result = dim.run(trace)
    assert result.score == 0.5
    assert not result.passed


def test_goal_completion_empty_final_answer():
    trace = AgentTrace(goal="g", steps=[_step("a")], final_answer=None)
    result = GoalCompletionDimension().run(trace)
    assert not result.passed
    assert result.score == 0.0


# --- TrajectoryCoherenceDimension ---

def test_trajectory_coherence_with_fake_judge():
    def complete(prompt: str) -> str:
        return "SCORE: 5\nREASON: coherent"

    judge = JudgeDimension(complete=complete, rubric="Is the trajectory coherent?")
    trace = _trace([_step("search"), _step("calculator")])
    result = TrajectoryCoherenceDimension(judge).run(trace)
    assert result.passed
    assert result.score == 1.0
    assert "coherent" in result.detail


# --- AgentEvalHarness ---

def test_harness_empty_raises():
    with pytest.raises(ValueError):
        AgentEvalHarness([])


def test_harness_combined_pass():
    trace = _trace(
        [_step("search"), _step("calculator")],
        final_answer="68 million",
    )
    harness = AgentEvalHarness(
        [
            ToolSelectionDimension(["search", "calculator"], mode="subset"),
            StepEfficiencyDimension(max_steps=4),
            GoalCompletionDimension(required_substrings=["68 million"]),
        ]
    )
    report = harness.run(trace)
    assert report.passed
    assert not report.failures


def test_harness_combined_fail_and_str():
    trace = _trace([_step("search")], final_answer="68 million")
    harness = AgentEvalHarness(
        [
            ToolSelectionDimension(["search", "calculator"], mode="subset"),
            GoalCompletionDimension(required_substrings=["68 million"]),
        ]
    )
    report = harness.run(trace)
    assert not report.passed
    assert "tool_selection" in report.failures
    text = str(report)
    assert text.startswith("AgentEvalReport: FAIL")
    assert "[tool_selection]" in text
    assert "[goal_completion]" in text


# --- AgentTrace round-trip ---

def test_trace_from_dict_to_text_round_trip():
    data = {
        "goal": "Find the answer.",
        "steps": [
            {
                "thought": "search first",
                "tool_calls": [
                    {"name": "search", "args": {"query": "answer"}, "result": "42"}
                ],
                "observation": "42",
            },
            {
                "tool_calls": [{"name": "calculator", "error": "boom"}],
            },
        ],
        "final_answer": "The answer is 42.",
    }
    trace = AgentTrace.from_dict(data)
    assert trace.num_steps == 2
    assert trace.tool_names() == ["search", "calculator"]
    assert trace.steps[1].thought is None
    text = trace.to_text()
    assert "Goal: Find the answer." in text
    assert "Tool: search args=" in text
    assert "Error: boom" in text
    assert "Final answer: The answer is 42." in text
