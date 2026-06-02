from __future__ import annotations

import statistics
from collections.abc import Callable
from typing import Any

from ..eval.dimension import Dimension
from .base import DEFAULT_PROMPT_TEMPLATE, JudgeVerdict, parse_verdict, render_prompt

_AGGREGATES = ("mean", "median", "majority")


class JudgeDimension(Dimension):
    """Grade text with an injected model callable (LLM-as-judge).

    ``complete`` maps a rendered prompt string to a model response string.
    Injecting it keeps the dimension offline-testable: tests pass a fake
    callable and no network is touched.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        rubric: str,
        *,
        scale: int = 5,
        threshold: float = 0.6,
        name: str = "judge",
        prompt_template: str | None = None,
        cache: dict[str, JudgeVerdict] | None = None,
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._complete = complete
        self._rubric = rubric
        self._scale = scale
        self._template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self._cache = cache

    def evaluate(self, text: str) -> tuple[float, str]:
        prompt = render_prompt(self._template, self._rubric, self._scale, text)
        if self._cache is not None and prompt in self._cache:
            verdict = self._cache[prompt]
            return verdict.score, verdict.reason
        try:
            raw = self._complete(prompt)
        except Exception as exc:
            return 0.0, f"judge error: {exc}"
        verdict = parse_verdict(raw, self._scale)
        if self._cache is not None:
            self._cache[prompt] = verdict
        return verdict.score, verdict.reason


class JuryDimension(Dimension):
    """Aggregate several JudgeDimension verdicts into one score.

    ``aggregate`` is one of ``"mean"``, ``"median"``, or ``"majority"``. The
    detail string reports the aggregate score, the per-judge scores, and an
    agreement signal: the population standard deviation of the scores (lower
    means more agreement), or the vote split for ``"majority"``.
    """

    def __init__(
        self,
        judges: list[JudgeDimension],
        *,
        aggregate: str = "mean",
        threshold: float = 0.6,
        name: str = "jury",
    ) -> None:
        if not judges:
            raise ValueError("JuryDimension requires at least one judge.")
        if aggregate not in _AGGREGATES:
            raise ValueError(
                f"unknown aggregate {aggregate!r}; expected one of {_AGGREGATES}"
            )
        super().__init__(threshold=threshold, name=name)
        self._judges = judges
        self._aggregate = aggregate

    def evaluate(self, text: str) -> tuple[float, str]:
        scores = [judge.evaluate(text)[0] for judge in self._judges]
        per_judge = ", ".join(f"{s:.3f}" for s in scores)
        if self._aggregate == "mean":
            score = statistics.fmean(scores)
            agreement = f"stdev={statistics.pstdev(scores):.3f}"
        elif self._aggregate == "median":
            score = statistics.median(scores)
            agreement = f"stdev={statistics.pstdev(scores):.3f}"
        else:
            passing = sum(1 for s in scores if s >= self.threshold)
            failing = len(scores) - passing
            score = passing / len(scores)
            agreement = f"votes={passing} pass / {failing} fail"
        detail = (
            f"{self._aggregate} score={score:.3f}; "
            f"per-judge=[{per_judge}]; {agreement}"
        )
        return score, detail


def anthropic_judge(model: str, client: Any = None) -> Callable[[str], str]:
    """Build a ``complete`` callable backed by the Anthropic Messages API.

    Requires the ``[judge]`` extra (the ``anthropic`` package). The import is
    deferred to call time so the base package never hard-depends on it. If no
    ``client`` is given, a default ``anthropic.Anthropic()`` is constructed.
    The returned callable sends ``prompt`` as a single user message and returns
    the text of the first content block.
    """
    import anthropic

    active_client = client if client is not None else anthropic.Anthropic()

    def complete(prompt: str) -> str:
        message = active_client.messages.create(
            model=model,
            max_tokens=512,
            messages=[{"role": "user", "content": prompt}],
        )
        return message.content[0].text

    return complete
