from __future__ import annotations

import argparse
import sys

from ..bench.gate import RegressionGate
from ..bench.runner import BenchmarkResult


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m llm_evalgate.gate",
        description="Fail on a metric regression vs a saved baseline benchmark.",
    )
    parser.add_argument("current", help="path to the current BenchmarkResult JSON")
    parser.add_argument("baseline", help="path to the baseline BenchmarkResult JSON")
    parser.add_argument(
        "--metrics",
        default="accuracy,regression_catch_rate",
        help="comma-separated metric names, or 'all' (default: accuracy,regression_catch_rate)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.02,
        help="max tolerated drop before a metric fails the gate (default: 0.02)",
    )
    parser.add_argument(
        "--no-significance",
        action="store_true",
        help="fail on any drop past the threshold, even within eval noise",
    )
    parser.add_argument(
        "--allow-unpaired",
        action="store_true",
        help="compare even when the two runs used different datasets",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="bootstrap RNG seed (default: 0)",
    )
    args = parser.parse_args(argv)

    metrics: tuple[str, ...] | str
    metrics = "all" if args.metrics.strip() == "all" else tuple(
        m.strip() for m in args.metrics.split(",") if m.strip()
    )

    current = BenchmarkResult.load(args.current)
    baseline = BenchmarkResult.load(args.baseline)
    gate = RegressionGate(
        metrics=metrics,
        threshold=args.threshold,
        require_significance=not args.no_significance,
        seed=args.seed,
        allow_unpaired=args.allow_unpaired,
    )
    report = gate.check(current, baseline)
    print(report.table())
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
