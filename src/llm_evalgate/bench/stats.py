from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil
from statistics import NormalDist

__all__ = [
    "ConfidenceInterval",
    "bootstrap_ci",
    "min_detectable_effect",
    "required_sample_size",
    "CorrectionResult",
    "holm",
    "benjamini_hochberg",
    "correct_pvalues",
]


@dataclass(frozen=True)
class ConfidenceInterval:
    """A point estimate with a bootstrap confidence interval."""

    point: float
    low: float
    high: float
    n_resamples: int

    def __str__(self) -> str:
        return f"{self.point:.3f} [{self.low:.3f}, {self.high:.3f}]"


def bootstrap_ci(
    metric_fn: Callable[[list[bool], list[bool]], float],
    predicted: Sequence[bool],
    labels: Sequence[bool],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> ConfidenceInterval:
    """Percentile bootstrap confidence interval for a paired classification metric.

    ``metric_fn`` maps ``(predicted, labels)`` to a float (any of the functions in
    :mod:`llm_evalgate.bench.metrics`). The two sequences are resampled together
    (paired) with replacement ``n_resamples`` times; the interval is the
    ``alpha/2`` and ``1 - alpha/2`` percentiles of the resampled metric values.

    The bootstrap is the right tool here because eval sets are small and metrics
    like F1 or Cohen's kappa are not normally distributed, so a CLT/normal interval
    is unreliable below a few hundred samples (arXiv:2503.01747).

    ``seed`` defaults to ``0`` so results are reproducible by default; pass
    ``seed=None`` to draw from system entropy.
    """
    if len(predicted) != len(labels):
        raise ValueError(
            f"predicted and labels must be equal length; "
            f"got {len(predicted)} and {len(labels)}"
        )
    if not predicted:
        raise ValueError("predicted and labels must be non-empty")
    if n_resamples < 1:
        raise ValueError(f"n_resamples must be >= 1; got {n_resamples}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")

    predicted = list(predicted)
    labels = list(labels)
    point = metric_fn(predicted, labels)

    n = len(predicted)
    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(n_resamples):
        idx = [rng.randrange(n) for _ in range(n)]
        rs_pred = [predicted[i] for i in idx]
        rs_label = [labels[i] for i in idx]
        samples.append(metric_fn(rs_pred, rs_label))
    samples.sort()

    low = _percentile(samples, 100.0 * (alpha / 2))
    high = _percentile(samples, 100.0 * (1 - alpha / 2))
    return ConfidenceInterval(point=point, low=low, high=high, n_resamples=n_resamples)


def _percentile(sorted_values: list[float], pct: float) -> float:
    """Linear-interpolation percentile of an already-sorted list."""
    if not sorted_values:
        raise ValueError("cannot take a percentile of an empty sequence")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (pct / 100.0) * (len(sorted_values) - 1)
    low_idx = int(rank)
    high_idx = min(low_idx + 1, len(sorted_values) - 1)
    frac = rank - low_idx
    return sorted_values[low_idx] * (1 - frac) + sorted_values[high_idx] * frac


def min_detectable_effect(
    n: int,
    *,
    baseline: float = 0.8,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Smallest change in a proportion metric detectable at ``n`` samples.

    Normal-approximation power analysis for a single proportion::

        MDE = (z_{1-alpha/2} + z_{power}) * sqrt(p*(1-p) / n)

    where ``p`` is ``baseline``. This is approximate (it ignores the variance
    change at the alternative), but it is the standard back-of-the-envelope used
    to answer "is my eval set even large enough to see the regression I want to
    gate on?". If the returned MDE exceeds your gate threshold, a difference at
    that threshold is statistically indistinguishable from noise.
    """
    if n < 1:
        raise ValueError(f"n must be >= 1; got {n}")
    if not 0.0 <= baseline <= 1.0:
        raise ValueError(f"baseline must be in [0, 1]; got {baseline}")
    z = _z(alpha, power)
    return z * (baseline * (1 - baseline) / n) ** 0.5


def required_sample_size(
    effect: float,
    *,
    baseline: float = 0.8,
    alpha: float = 0.05,
    power: float = 0.8,
) -> int:
    """Samples needed to detect a proportion change of ``effect`` (inverse of MDE)::

        n = ceil( (z_{1-alpha/2} + z_{power})^2 * p*(1-p) / effect^2 )
    """
    if effect <= 0.0:
        raise ValueError(f"effect must be > 0; got {effect}")
    if not 0.0 <= baseline <= 1.0:
        raise ValueError(f"baseline must be in [0, 1]; got {baseline}")
    z = _z(alpha, power)
    return ceil((z**2) * baseline * (1 - baseline) / (effect**2))


def _z(alpha: float, power: float) -> float:
    """Sum of the two-sided significance z and the power z."""
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1); got {power}")
    normal = NormalDist()
    return normal.inv_cdf(1 - alpha / 2) + normal.inv_cdf(power)


@dataclass(frozen=True)
class CorrectionResult:
    """Per-hypothesis outcome of a multiple-comparisons correction.

    ``adjusted`` are the multiplicity-adjusted p-values, aligned with the input
    order, so a hypothesis is rejected iff ``adjusted[i] <= alpha`` (this also
    holds for the step-down/step-up procedures because the adjusted p-values are
    monotone in the original p-value ranking). ``rejected`` is that comparison
    precomputed.
    """

    method: str
    alpha: float
    pvalues: tuple[float, ...]
    adjusted: tuple[float, ...]
    rejected: tuple[bool, ...]


def _validate_corr_inputs(pvalues: Sequence[float], alpha: float) -> list[float]:
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1); got {alpha}")
    pv = list(pvalues)
    for p in pv:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p-values must be in [0, 1]; got {p}")
    return pv


def holm(pvalues: Sequence[float], *, alpha: float = 0.05) -> CorrectionResult:
    """Holm-Bonferroni step-down correction (controls the family-wise error rate).

    Holm controls the probability of *any* false rejection (FWER) under arbitrary
    dependence between the tests, which is the right guarantee for a gate: the
    question a gate answers is "did I wrongly block this change at all?", not
    "what fraction of my blocks are wrong?". It is uniformly more powerful than a
    plain Bonferroni correction.

    The adjusted p-value for the hypothesis at sorted rank ``k`` (1-based) is
    ``max`` over ranks ``j <= k`` of ``min(1, (m - j + 1) * p_(j))``, enforcing
    monotonicity so the reject set is a prefix of the sorted p-values.
    """
    pv = _validate_corr_inputs(pvalues, alpha)
    m = len(pv)
    if m == 0:
        return CorrectionResult("holm", alpha, (), (), ())
    order = sorted(range(m), key=lambda i: pv[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        factor = m - rank  # m, m-1, ..., 1
        running = max(running, min(1.0, factor * pv[idx]))
        adjusted[idx] = running
    rejected = tuple(adjusted[i] <= alpha for i in range(m))
    return CorrectionResult("holm", alpha, tuple(pv), tuple(adjusted), rejected)


def benjamini_hochberg(
    pvalues: Sequence[float], *, alpha: float = 0.05
) -> CorrectionResult:
    """Benjamini-Hochberg step-up correction (controls the false discovery rate).

    BH controls the expected *proportion* of false rejections among all
    rejections (FDR), which is the standard choice when you are screening many
    hypotheses and can tolerate a known fraction of false positives. It is more
    powerful (rejects more) than Holm, but gives a weaker guarantee, and is valid
    under independence or positive dependence (PRDS) of the tests.

    The adjusted p-value for sorted rank ``k`` (1-based) is ``min`` over ranks
    ``j >= k`` of ``min(1, (m / j) * p_(j))``, enforcing monotonicity.
    """
    pv = _validate_corr_inputs(pvalues, alpha)
    m = len(pv)
    if m == 0:
        return CorrectionResult("benjamini_hochberg", alpha, (), (), ())
    order = sorted(range(m), key=lambda i: pv[i])
    adjusted = [0.0] * m
    running = 1.0
    for rank in range(m - 1, -1, -1):
        idx = order[rank]
        factor = m / (rank + 1)  # m / i, i 1-based
        running = min(running, min(1.0, factor * pv[idx]))
        adjusted[idx] = running
    rejected = tuple(adjusted[i] <= alpha for i in range(m))
    return CorrectionResult("benjamini_hochberg", alpha, tuple(pv), tuple(adjusted), rejected)


def correct_pvalues(
    pvalues: Sequence[float], *, method: str = "holm", alpha: float = 0.05
) -> CorrectionResult:
    """Apply a multiple-comparisons correction by name.

    ``method`` is one of ``"holm"`` (FWER, the default and the right choice for a
    gate), ``"bh"`` / ``"benjamini_hochberg"`` / ``"fdr"`` (FDR), or ``"none"``
    (no correction: each hypothesis is rejected iff its raw p-value ``<= alpha``).
    """
    normalized = method.lower()
    if normalized == "holm":
        return holm(pvalues, alpha=alpha)
    if normalized in {"bh", "benjamini_hochberg", "fdr"}:
        return benjamini_hochberg(pvalues, alpha=alpha)
    if normalized == "none":
        pv = _validate_corr_inputs(pvalues, alpha)
        return CorrectionResult(
            "none", alpha, tuple(pv), tuple(pv), tuple(p <= alpha for p in pv)
        )
    raise ValueError(
        f"unknown correction method {method!r}; use 'holm', 'bh', or 'none'"
    )
