from __future__ import annotations

import random
from dataclasses import dataclass, field

from .metrics import (
    accuracy,
    cohen_kappa,
    f1,
    precision,
    recall,
    regression_catch_rate,
)
from .runner import BenchmarkResult
from .stats import _percentile, min_detectable_effect

# All current metrics are higher-is-better, so a negative delta is a regression.
_METRIC_FNS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cohen_kappa": cohen_kappa,
    "regression_catch_rate": regression_catch_rate,
}


@dataclass(frozen=True)
class GateRow:
    metric: str
    baseline: float
    current: float
    delta: float
    ci_low: float
    ci_high: float
    verdict: str  # "pass" | "warn" | "fail"

    def __str__(self) -> str:
        return (
            f"{self.metric:<22} {self.baseline:6.3f} -> {self.current:6.3f}  "
            f"delta={self.delta:+.3f}  "
            f"CI=[{self.ci_low:+.3f}, {self.ci_high:+.3f}]  {self.verdict.upper()}"
        )


@dataclass(frozen=True)
class GateReport:
    passed: bool
    rows: list[GateRow]
    regressed_samples: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stratum_rows: dict[str, list[GateRow]] = field(default_factory=dict)

    def table(self) -> str:
        lines = [f"GateReport: {'PASS' if self.passed else 'FAIL'}"]
        for row in self.rows:
            lines.append(f"  {row}")
        for stratum, rows in self.stratum_rows.items():
            lines.append(f"  --- stratum={stratum}")
            for row in rows:
                lines.append(f"    {row}")
        if self.regressed_samples:
            shown = ", ".join(str(i) for i in self.regressed_samples[:10])
            extra = len(self.regressed_samples) - 10
            more = f" (+{extra} more)" if extra > 0 else ""
            lines.append(f"  regressed sample indices: {shown}{more}")
        for warning in self.warnings:
            lines.append(f"  WARN: {warning}")
        return "\n".join(lines)


class RegressionGate:
    """Block a change when a metric regresses past a threshold vs a baseline.

    The standard CI discipline is "fail the build when quality drops more than
    N points against the last known-good run". This compares a current
    :class:`BenchmarkResult` to a saved baseline and decides pass/fail per
    metric. When ``require_significance`` is set, a regression only fails the
    gate if the paired-bootstrap CI on the delta lies entirely below zero, so a
    drop that is within eval noise is reported as a warning rather than a hard
    failure -- which matters because small eval sets cannot resolve small deltas.
    """

    def __init__(
        self,
        *,
        metrics: tuple[str, ...] | str = ("accuracy", "regression_catch_rate"),
        threshold: float = 0.02,
        require_significance: bool = True,
        n_resamples: int = 2000,
        alpha: float = 0.05,
        seed: int | None = 0,
        allow_unpaired: bool = False,
        fail_on_stratum: bool = False,
    ) -> None:
        if metrics == "all":
            resolved = tuple(_METRIC_FNS)
        else:
            resolved = tuple(metrics)
            unknown = [m for m in resolved if m not in _METRIC_FNS]
            if unknown:
                raise ValueError(
                    f"unknown metric(s) {unknown}; known: {sorted(_METRIC_FNS)}"
                )
        if not resolved:
            raise ValueError("RegressionGate requires at least one metric.")
        self._metrics = resolved
        self._threshold = threshold
        self._require_significance = require_significance
        self._n_resamples = n_resamples
        self._alpha = alpha
        self._seed = seed
        self._allow_unpaired = allow_unpaired
        self._fail_on_stratum = fail_on_stratum

    def check(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        *,
        stratify_by: str | None = None,
    ) -> GateReport:
        warnings: list[str] = []
        paired = self._is_paired(current, baseline, warnings)

        rows: list[GateRow] = []
        gate_passed = True
        for metric in self._metrics:
            fn = _METRIC_FNS[metric]
            base_val = fn(baseline.predicted, baseline.labels)
            cur_val = fn(current.predicted, current.labels)
            delta = cur_val - base_val
            ci_low, ci_high = self._delta_ci(fn, current, baseline, paired)

            if delta < -self._threshold:
                significant = ci_high < 0.0
                if self._require_significance and not significant:
                    verdict = "warn"
                    warnings.append(
                        f"{metric} dropped {delta:+.3f} (past -{self._threshold:.3f}) "
                        f"but its delta CI [{ci_low:+.3f}, {ci_high:+.3f}] includes 0 "
                        f"-- not separable from eval noise at n={current.n}."
                    )
                else:
                    verdict = "fail"
                    gate_passed = False
            else:
                verdict = "pass"
            rows.append(
                GateRow(
                    metric=metric,
                    baseline=base_val,
                    current=cur_val,
                    delta=delta,
                    ci_low=ci_low,
                    ci_high=ci_high,
                    verdict=verdict,
                )
            )

        mde = min_detectable_effect(current.n)
        if mde > self._threshold:
            warnings.append(
                f"n={current.n}: minimum detectable effect ~= {mde:.3f}; a "
                f"{self._threshold:.3f} threshold cannot be distinguished from noise "
                f"-- decisions below the MDE rely on require_significance."
            )

        stratum_rows: dict[str, list[GateRow]] = {}
        if stratify_by is not None:
            stratum_rows, stratum_failed = self._stratum_rows(
                current, baseline, stratify_by, paired, warnings
            )
            if self._fail_on_stratum and stratum_failed:
                gate_passed = False

        regressed = self._regressed_samples(current, baseline) if paired else []
        return GateReport(
            passed=gate_passed,
            rows=rows,
            regressed_samples=regressed,
            warnings=warnings,
            stratum_rows=stratum_rows,
        )

    def _stratum_rows(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        stratify_by: str,
        paired: bool,
        warnings: list[str],
    ) -> tuple[dict[str, list[GateRow]], bool]:
        if not paired:
            warnings.append("per-stratum deltas require a paired comparison; skipped.")
            return {}, False
        if current.metas is None:
            warnings.append(
                f"current result has no sample metadata; cannot stratify by "
                f"'{stratify_by}'."
            )
            return {}, False

        groups: dict[str, list[int]] = {}
        for i, meta in enumerate(current.metas):
            key = str(meta.get(stratify_by, "(none)"))
            groups.setdefault(key, []).append(i)

        stratum_rows: dict[str, list[GateRow]] = {}
        any_failed = False
        for key in sorted(groups):
            idx = groups[key]
            sub_cur = _subset(current, idx)
            sub_base = _subset(baseline, idx)
            rows: list[GateRow] = []
            for metric in self._metrics:
                fn = _METRIC_FNS[metric]
                base_val = fn(sub_base.predicted, sub_base.labels)
                cur_val = fn(sub_cur.predicted, sub_cur.labels)
                delta = cur_val - base_val
                ci_low, ci_high = self._delta_ci(fn, sub_cur, sub_base, paired=True)
                if delta < -self._threshold and (
                    not self._require_significance or ci_high < 0.0
                ):
                    verdict = "fail" if self._fail_on_stratum else "warn"
                    any_failed = True
                elif delta < -self._threshold:
                    verdict = "warn"
                else:
                    verdict = "pass"
                rows.append(
                    GateRow(metric, base_val, cur_val, delta, ci_low, ci_high, verdict)
                )
            stratum_rows[key] = rows
        return stratum_rows, any_failed

    def _is_paired(
        self,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        warnings: list[str],
    ) -> bool:
        same_fp = (
            current.dataset_fingerprint is not None
            and current.dataset_fingerprint == baseline.dataset_fingerprint
        )
        if same_fp and len(current.predicted) == len(baseline.predicted):
            return True
        if not self._allow_unpaired:
            raise ValueError(
                "current and baseline were graded on different datasets "
                "(fingerprint mismatch); pass allow_unpaired=True to compare anyway."
            )
        warnings.append(
            "dataset fingerprints differ; comparing unpaired -- delta CIs are wider "
            "and per-sample regression listing is unavailable."
        )
        return False

    def _delta_ci(
        self,
        fn,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        paired: bool,
    ) -> tuple[float, float]:
        rng = random.Random(self._seed)
        deltas: list[float] = []
        if paired:
            n = len(current.predicted)
            for _ in range(self._n_resamples):
                idx = [rng.randrange(n) for _ in range(n)]
                cur = fn(
                    [current.predicted[i] for i in idx],
                    [current.labels[i] for i in idx],
                )
                base = fn(
                    [baseline.predicted[i] for i in idx],
                    [baseline.labels[i] for i in idx],
                )
                deltas.append(cur - base)
        else:
            nc = len(current.predicted)
            nb = len(baseline.predicted)
            for _ in range(self._n_resamples):
                ci = [rng.randrange(nc) for _ in range(nc)]
                bi = [rng.randrange(nb) for _ in range(nb)]
                cur = fn(
                    [current.predicted[i] for i in ci],
                    [current.labels[i] for i in ci],
                )
                base = fn(
                    [baseline.predicted[i] for i in bi],
                    [baseline.labels[i] for i in bi],
                )
                deltas.append(cur - base)
        deltas.sort()
        low = _percentile(deltas, 100.0 * (self._alpha / 2))
        high = _percentile(deltas, 100.0 * (1 - self._alpha / 2))
        return low, high

    @staticmethod
    def _regressed_samples(
        current: BenchmarkResult, baseline: BenchmarkResult
    ) -> list[int]:
        return [
            i
            for i, (b, c) in enumerate(zip(baseline.predicted, current.predicted))
            if b and not c
        ]


def _subset(result: BenchmarkResult, idx: list[int]) -> BenchmarkResult:
    """A lightweight BenchmarkResult holding only the predictions at ``idx``.

    Carries the same fingerprint so the paired-bootstrap path stays enabled.
    These sub-results are terminal: ``metas`` is intentionally dropped because a
    stratum is never itself re-stratified.
    """
    return BenchmarkResult(
        predicted=[result.predicted[i] for i in idx],
        labels=[result.labels[i] for i in idx],
        metrics={},
        n=len(idx),
        dataset_fingerprint=result.dataset_fingerprint,
    )
