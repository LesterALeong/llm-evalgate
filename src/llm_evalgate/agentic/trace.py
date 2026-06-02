from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any] = field(default_factory=dict)
    result: Any = None
    error: str | None = None


@dataclass
class AgentStep:
    thought: str | None = None
    tool_calls: list[ToolCall] = field(default_factory=list)
    observation: str | None = None


@dataclass
class AgentTrace:
    goal: str
    steps: list[AgentStep] = field(default_factory=list)
    final_answer: str | None = None

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    def tool_names(self) -> list[str]:
        """Tool names in execution order across all steps."""
        return [call.name for step in self.steps for call in step.tool_calls]

    def all_tool_calls(self) -> list[ToolCall]:
        return [call for step in self.steps for call in step.tool_calls]

    def to_text(self) -> str:
        """Readable serialization a judge can read."""
        lines = [f"Goal: {self.goal}"]
        for index, step in enumerate(self.steps, start=1):
            lines.append(f"Step {index}:")
            if step.thought is not None:
                lines.append(f"  Thought: {step.thought}")
            for call in step.tool_calls:
                lines.append(f"  Tool: {call.name} args={call.args}")
                if call.error is not None:
                    lines.append(f"    Error: {call.error}")
                elif call.result is not None:
                    lines.append(f"    Result: {call.result}")
            if step.observation is not None:
                lines.append(f"  Observation: {step.observation}")
        if self.final_answer is not None:
            lines.append(f"Final answer: {self.final_answer}")
        return "\n".join(lines)

    @classmethod
    def from_dict(cls, data: dict) -> AgentTrace:
        steps = []
        for step_data in data.get("steps", []):
            tool_calls = [
                ToolCall(
                    name=call["name"],
                    args=call.get("args", {}),
                    result=call.get("result"),
                    error=call.get("error"),
                )
                for call in step_data.get("tool_calls", [])
            ]
            steps.append(
                AgentStep(
                    thought=step_data.get("thought"),
                    tool_calls=tool_calls,
                    observation=step_data.get("observation"),
                )
            )
        return cls(
            goal=data["goal"],
            steps=steps,
            final_answer=data.get("final_answer"),
        )
