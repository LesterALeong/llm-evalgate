"""Why a regression gate needs a multiple-comparisons correction, offline.

Run from the repo root::

    python examples/multiple_comparisons.py            # ~20-40s
    python examples/multiple_comparisons.py --trials 400   # tighter estimates

A regression gate runs one significance test per metric. Run K of them at
alpha=0.05 with no correction and, under the null (nothing actually regressed),
the chance of at least one false FAIL is ~1 - (1 - 0.05)^K -- about 26% at K=6,
not 5%. This simulation grades many null A/B pairs (baseline and current drawn
from the *same* quality, so the true delta is zero) and reports how often the
gate falsely fails: with no correction, with Holm (controls the family-wise
error rate), and with Benjamini-Hochberg (controls the false discovery rate).

Holm pulls the false-failure rate back toward alpha; "none" leaves it inflated.
The three corrections are evaluated on the *same* sequence of null pairs, so the
differences are due to the correction alone.
"""

from __future__ import annotations

import argparse
import random

from llm_evalgate.bench import BenchmarkResult, RegressionGate


def _grade(rng: random.Random, labels: list[bool], error_rate: float) -> list[bool]:
    """A grader that matches the label except for random i.i.d. errors."""
    return [(not label) if rng.random() < error_rate else label for label in labels]


def _null_pair(
    rng: random.Random, n: int, error_rate: float
) -> tuple[BenchmarkResult, BenchmarkResult]:
    """A baseline and a current graded at the *same* true quality (delta == 0)."""
    labels = [i % 2 == 0 for i in range(n)]
    base = BenchmarkResult(
        predicted=_grade(rng, labels, error_rate), labels=labels,
        metrics={}, n=n, dataset_fingerprint="null",
    )
    cur = BenchmarkResult(
        predicted=_grade(rng, labels, error_rate), labels=labels,
        metrics={}, n=n, dataset_fingerprint="null",
    )
    return cur, base


def false_fail_rate(
    correction: str, *, trials: int, n: int, error_rate: float,
    threshold: float, n_resamples: int, seed: int,
) -> float:
    rng = random.Random(seed)  # same seed per correction -> same null pairs
    fails = 0
    for _ in range(trials):
        cur, base = _null_pair(rng, n, error_rate)
        # The null data (rng) advances every trial; the gate's bootstrap seed is
        # fixed so only the data varies -- do not collapse these into one seed.
        gate = RegressionGate(
            metrics="all", threshold=threshold, correction=correction,
            n_resamples=n_resamples, seed=0,
        )
        if not gate.check(cur, base).passed:
            fails += 1
    return fails / trials


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Monte-Carlo false-failure rate of the regression gate, by correction."
    )
    parser.add_argument("--trials", type=int, default=120)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--error-rate", type=float, default=0.12)
    parser.add_argument("--threshold", type=float, default=0.02)
    parser.add_argument("--n-resamples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(
        f"Null A/B gate over 6 metrics: {args.trials} trials, n={args.n}, "
        f"error_rate={args.error_rate}, threshold={args.threshold}\n"
        "(true delta is zero, so every FAIL is a false positive)\n"
    )
    print(f"{'correction':<22}{'false-fail rate':>16}")
    for correction in ("none", "holm", "bh"):
        rate = false_fail_rate(
            correction, trials=args.trials, n=args.n, error_rate=args.error_rate,
            threshold=args.threshold, n_resamples=args.n_resamples, seed=args.seed,
        )
        print(f"{correction:<22}{rate:>15.1%}")


if __name__ == "__main__":
    main()
