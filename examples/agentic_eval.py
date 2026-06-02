"""Evaluate a multi-step tool-using agent trace, offline and runnable.

Run from the repo root::

    python examples/agentic_eval.py
"""

from __future__ import annotations

from llm_evalgate.agentic import (
    AgentEvalHarness,
    AgentStep,
    AgentTrace,
    GoalCompletionDimension,
    StepEfficiencyDimension,
    ToolArgValidityDimension,
    ToolCall,
    ToolSelectionDimension,
)


def build_trace() -> AgentTrace:
    return AgentTrace(
        goal="Find the population of France and convert it to millions.",
        steps=[
            AgentStep(
                thought="I need the current population figure, so I will search.",
                tool_calls=[
                    ToolCall(
                        name="search",
                        args={"query": "population of France"},
                        result="France population is approximately 68000000.",
                    )
                ],
                observation="France population is approximately 68000000.",
            ),
            AgentStep(
                thought="Now I convert the raw figure to millions.",
                tool_calls=[
                    ToolCall(
                        name="calculator",
                        args={"expression": "68000000 / 1000000"},
                        result=68.0,
                    )
                ],
                observation="68.0",
            ),
        ],
        final_answer="The population of France is about 68 million.",
    )


def main() -> None:
    trace = build_trace()
    harness = AgentEvalHarness(
        [
            ToolSelectionDimension(
                expected_tools=["search", "calculator"], mode="order"
            ),
            ToolArgValidityDimension(
                validators={"search": lambda args: bool(args.get("query"))}
            ),
            StepEfficiencyDimension(max_steps=4),
            GoalCompletionDimension(required_substrings=["68 million"]),
        ]
    )
    report = harness.run(trace)
    print(report)

    # To grade long-horizon reasoning coherence with an LLM-as-judge, add a
    # TrajectoryCoherenceDimension wrapping a JudgeDimension. The ``complete``
    # callable is injected, so this stays offline in tests:
    #
    # from llm_evalgate.agentic import TrajectoryCoherenceDimension
    # from llm_evalgate.judge import JudgeDimension
    #
    # def complete(prompt: str) -> str:
    #     return "SCORE: 5\nREASON: the steps follow logically toward the goal"
    #
    # judge = JudgeDimension(complete=complete, rubric="Is the trajectory coherent?")
    # harness = AgentEvalHarness([TrajectoryCoherenceDimension(judge)])


if __name__ == "__main__":
    main()
