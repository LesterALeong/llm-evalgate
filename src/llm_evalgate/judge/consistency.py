from __future__ import annotations

import statistics
from dataclasses import dataclass
from math import sqrt

from ..eval.dimension import Dimension
from .dimension import JudgeDimension

_AGGREGATES = ("mean", "median")


@dataclass(frozen=True)
class ScoreDistribution:
    mean: float
    stdev: float
    n: int
    ci_low: float
    ci_high: float
    scores: tuple[float, ...]


class SelfConsistencyJudge(Dimension):
    """Sample a base judge N times to surface its score distribution.

    LLM judges are stochastic at temperature > 0, so a single score hides
    uncertainty. This wraps a ``JudgeDimension``, calls its un-cached
    ``score`` ``samples`` times, and reports the mean, spread, and a 95%
    confidence interval. Because ``score`` bypasses the cache, each sample is
    an independent model call, which is what produces the variance.

    Cost: every call to ``distribution`` or ``evaluate`` makes ``samples`` real
    model calls and intentionally does not use the base judge's cache. With a
    live judge that is ``samples`` times the tokens of a single grade per text.
    """

    def __init__(
        self,
        judge: JudgeDimension,
        *,
        samples: int = 5,
        aggregate: str = "mean",
        threshold: float | None = None,
        name: str = "self_consistency",
    ) -> None:
        if samples < 1:
            raise ValueError(f"samples must be >= 1; got {samples}")
        if aggregate not in _AGGREGATES:
            raise ValueError(
                f"unknown aggregate {aggregate!r}; expected one of {_AGGREGATES}"
            )
        resolved_threshold = judge.threshold if threshold is None else threshold
        super().__init__(threshold=resolved_threshold, name=name)
        self._judge = judge
        self._samples = samples
        self._aggregate = aggregate

    def distribution(self, text: str) -> ScoreDistribution:
        scores = [self._judge.score(text).score for _ in range(self._samples)]
        n = len(scores)
        mean = statistics.fmean(scores)
        stdev = statistics.pstdev(scores) if n > 1 else 0.0
        half = 1.96 * stdev / sqrt(n)
        ci_low = max(0.0, mean - half)
        ci_high = min(1.0, mean + half)
        return ScoreDistribution(
            mean=mean,
            stdev=stdev,
            n=n,
            ci_low=ci_low,
            ci_high=ci_high,
            scores=tuple(scores),
        )

    def evaluate(self, text: str) -> tuple[float, str]:
        dist = self.distribution(text)
        if self._aggregate == "mean":
            score = dist.mean
        else:
            score = statistics.median(dist.scores)
        detail = (
            f"mean={dist.mean:.3f}; stdev={dist.stdev:.3f}; "
            f"95% CI=[{dist.ci_low:.3f}, {dist.ci_high:.3f}]; n={dist.n}"
        )
        return score, detail
