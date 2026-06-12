from __future__ import annotations

import statistics
from dataclasses import dataclass

from ...judge.dimension import JudgeDimension
from ..trace import AgentStep, AgentTrace
from .base import TraceDimension

_AGGREGATES = ("mean", "min")

DEFAULT_STEP_RUBRIC = (
    "You are scoring a single step of an agent's trajectory. Given the goal and "
    "everything that happened before it, does THIS step make meaningful progress "
    "toward the goal? Reward steps that gather new information or advance the plan; "
    "penalize redundant, off-track, or no-op steps."
)


@dataclass(frozen=True)
class StepScore:
    index: int
    score: float
    reason: str


class StepProgressDimension(TraceDimension):
    """Process-reward-style scoring: grade each step on progress toward the goal.

    Where :class:`TrajectoryCoherenceDimension` judges the whole trace at once,
    this scores every step against the history that precedes it, so a low score
    localizes *which* step went wrong. Each step is graded by the injected
    ``judge`` (a :class:`JudgeDimension` carrying a progress rubric -- see
    :data:`DEFAULT_STEP_RUBRIC`). Costs one judge call per step.

    ``aggregate`` is ``"mean"`` (overall progress) or ``"min"`` (weakest-link
    gating). ``inefficiency_penalty`` subtracts from the aggregate for each
    repeated identical ``(tool, args)`` call, matching
    :class:`StepEfficiencyDimension`'s repeat detection.
    """

    def __init__(
        self,
        judge: JudgeDimension,
        *,
        aggregate: str = "mean",
        inefficiency_penalty: float = 0.0,
        threshold: float = 0.7,
        name: str = "step_progress",
    ) -> None:
        if aggregate not in _AGGREGATES:
            raise ValueError(
                f"unknown aggregate {aggregate!r}; expected one of {_AGGREGATES}"
            )
        if inefficiency_penalty < 0.0:
            raise ValueError(
                f"inefficiency_penalty must be >= 0; got {inefficiency_penalty}"
            )
        super().__init__(threshold=threshold, name=name)
        self._judge = judge
        self._aggregate = aggregate
        self._inefficiency_penalty = inefficiency_penalty

    def step_scores(self, trace: AgentTrace) -> list[StepScore]:
        """Per-step progress scores, each judged against the preceding history."""
        results: list[StepScore] = []
        for k in range(len(trace.steps)):
            prefix = _render_prefix(trace, k)
            verdict = self._judge.score(prefix)
            results.append(StepScore(index=k + 1, score=verdict.score, reason=verdict.reason))
        return results

    def _repeated_calls(self, trace: AgentTrace) -> int:
        seen: set[tuple[str, tuple]] = set()
        repeats = 0
        for call in trace.all_tool_calls():
            key = (call.name, tuple(sorted(call.args.items())))
            if key in seen:
                repeats += 1
            else:
                seen.add(key)
        return repeats

    def evaluate(self, trace: AgentTrace) -> tuple[float, str]:
        if not trace.steps:
            return 0.0, "no steps: no progress to score"

        scores = self.step_scores(trace)
        values = [s.score for s in scores]
        if self._aggregate == "mean":
            base = statistics.fmean(values)
        else:
            base = min(values)

        repeats = self._repeated_calls(trace)
        penalty = self._inefficiency_penalty * repeats
        score = max(0.0, base - penalty)

        weakest = min(scores, key=lambda s: s.score)
        rendered = ", ".join(f"{v:.2f}" for v in values)
        detail = (
            f"steps={len(scores)} {self._aggregate}={base:.3f}: [{rendered}]; "
            f"weakest step {weakest.index} (judge: \"{weakest.reason}\")"
        )
        if penalty:
            detail += f"; -{penalty:.3f} for {repeats} repeated call(s)"
        return score, detail


def _render_prefix(trace: AgentTrace, k: int) -> str:
    """Serialize the goal, steps before ``k``, and step ``k`` as the current step.

    Only history up to and including step ``k`` is included -- no look-ahead to
    later steps -- so each step is judged on what was known at the time.
    """
    lines = [f"Goal: {trace.goal}"]
    for index in range(k):
        lines.append(f"Step {index + 1}:")
        lines.extend(_render_step(trace.steps[index]))
    lines.append(f"CURRENT STEP {k + 1}:")
    lines.extend(_render_step(trace.steps[k]))
    return "\n".join(lines)


def _render_step(step: AgentStep) -> list[str]:
    out: list[str] = []
    if step.user_message is not None:
        out.append(f"  User: {step.user_message}")
    if step.thought is not None:
        out.append(f"  Thought: {step.thought}")
    for call in step.tool_calls:
        out.append(f"  Tool: {call.name} args={call.args}")
        if call.error is not None:
            out.append(f"    Error: {call.error}")
        elif call.result is not None:
            out.append(f"    Result: {call.result}")
    if step.observation is not None:
        out.append(f"  Observation: {step.observation}")
    return out
