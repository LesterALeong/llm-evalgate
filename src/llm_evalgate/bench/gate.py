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
from .stats import _percentile, correct_pvalues, min_detectable_effect

# All current metrics are higher-is-better, so a negative delta is a regression.
_METRIC_FNS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cohen_kappa": cohen_kappa,
    "regression_catch_rate": regression_catch_rate,
}

# Accepted spellings on the public API; normalized to a canonical method name.
_CORRECTION_ALIASES = {
    "holm": "holm",
    "bh": "benjamini_hochberg",
    "benjamini_hochberg": "benjamini_hochberg",
    "fdr": "benjamini_hochberg",
    "none": "none",
}


@dataclass(frozen=True)
class GateRow:
    metric: str
    baseline: float
    current: float
    delta: float
    ci_low: float
    ci_high: float
    p_value: float
    adjusted_p: float
    verdict: str  # "pass" | "warn" | "fail"

    def __str__(self) -> str:
        return (
            f"{self.metric:<22} {self.baseline:6.3f} -> {self.current:6.3f}  "
            f"delta={self.delta:+.3f}  "
            f"CI=[{self.ci_low:+.3f}, {self.ci_high:+.3f}]  "
            f"p={self.p_value:.3f} adj_p={self.adjusted_p:.3f}  "
            f"{self.verdict.upper()}"
        )


@dataclass(frozen=True)
class GateReport:
    passed: bool
    rows: list[GateRow]
    regressed_samples: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    stratum_rows: dict[str, list[GateRow]] = field(default_factory=dict)
    correction: str = "none"

    def table(self) -> str:
        head = f"GateReport: {'PASS' if self.passed else 'FAIL'}"
        if self.correction != "none":
            head += f" (correction={self.correction})"
        lines = [head]
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
    gate if it is statistically separable from eval noise -- a drop within noise
    is reported as a warning rather than a hard failure, which matters because
    small eval sets cannot resolve small deltas.

    Significance is a one-sided bootstrap test (H1: the metric dropped). Because
    a gate runs that test on *every* configured metric at once, the raw
    per-metric p-values are corrected for multiple comparisons before the
    decision: with K metrics at alpha=0.05 and no correction, the chance of at
    least one false failure under the null is ~1-(1-0.05)^K, far above 0.05.
    ``correction`` defaults to Holm (controls the family-wise error rate, the
    right guarantee for a gate); pass ``"bh"`` for Benjamini-Hochberg (FDR) or
    ``"none"`` to disable.
    """

    def __init__(
        self,
        *,
        metrics: tuple[str, ...] | str = ("accuracy", "regression_catch_rate"),
        threshold: float = 0.02,
        require_significance: bool = True,
        correction: str = "holm",
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
        canonical = _CORRECTION_ALIASES.get(correction.lower())
        if canonical is None:
            raise ValueError(
                f"unknown correction {correction!r}; use 'holm', 'bh', or 'none'"
            )
        self._metrics = resolved
        self._threshold = threshold
        self._require_significance = require_significance
        self._correction = canonical
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

        rows, significant_breach = self._family_rows(
            self._metrics,
            current,
            baseline,
            paired,
            warnings=warnings,
            breach_is_fail=True,
        )
        gate_passed = not significant_breach

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
            correction=self._correction,
        )

    def _family_rows(
        self,
        metrics: tuple[str, ...],
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        paired: bool,
        *,
        warnings: list[str] | None,
        breach_is_fail: bool,
    ) -> tuple[list[GateRow], bool]:
        """Evaluate one family of metric tests, correcting their p-values together.

        Returns the rows plus whether any metric was a *significant* regression
        (a threshold breach that survived the correction, or any breach when
        ``require_significance`` is off). ``breach_is_fail`` controls whether such
        a breach is labelled ``"fail"`` (top-level) or ``"warn"`` (a stratum that
        is not configured to fail the gate). Pass ``warnings=None`` to suppress
        the per-metric noise warnings (used for strata, which are advisory).
        """
        # Pass 1: per-metric point deltas, CIs, and one-sided bootstrap p-values.
        stats: list[tuple[str, float, float, float, float, float, float]] = []
        for metric in metrics:
            fn = _METRIC_FNS[metric]
            base_val = fn(baseline.predicted, baseline.labels)
            cur_val = fn(current.predicted, current.labels)
            delta = cur_val - base_val
            ci_low, ci_high, p_value = self._delta_stats(fn, current, baseline, paired)
            stats.append((metric, base_val, cur_val, delta, ci_low, ci_high, p_value))

        # Correct the whole family of per-metric tests together.
        corr = correct_pvalues(
            [s[6] for s in stats], method=self._correction, alpha=self._alpha
        )

        # Pass 2: verdicts using the corrected significance.
        rows: list[GateRow] = []
        significant_breach = False
        for (metric, base_val, cur_val, delta, ci_low, ci_high, p_value), adj, rej in zip(
            stats, corr.adjusted, corr.rejected
        ):
            if delta < -self._threshold:
                significant = rej if self._require_significance else True
                if significant:
                    significant_breach = True
                    verdict = "fail" if breach_is_fail else "warn"
                else:
                    verdict = "warn"
                    if warnings is not None:
                        warnings.append(
                            f"{metric} dropped {delta:+.3f} (past "
                            f"-{self._threshold:.3f}) but is not significant after "
                            f"{corr.method} correction (adjusted p={adj:.3f} > "
                            f"alpha={self._alpha:.3f}) -- within eval noise at "
                            f"n={current.n}."
                        )
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
                    p_value=p_value,
                    adjusted_p=adj,
                    verdict=verdict,
                )
            )
        return rows, significant_breach

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
            # Each stratum is its own decision context, so its metrics form their
            # own correction family (correcting across strata too would be far too
            # conservative -- they are separate questions, not one).
            rows, significant_breach = self._family_rows(
                self._metrics,
                sub_cur,
                sub_base,
                paired=True,
                warnings=None,
                breach_is_fail=self._fail_on_stratum,
            )
            any_failed = any_failed or significant_breach
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

    def _delta_stats(
        self,
        fn,
        current: BenchmarkResult,
        baseline: BenchmarkResult,
        paired: bool,
    ) -> tuple[float, float, float]:
        """Bootstrap the delta: return (CI low, CI high, one-sided p-value).

        The CI is the two-sided ``1 - alpha`` percentile interval. The p-value is
        a one-sided bootstrap test for H1: delta < 0 (a regression), estimated as
        the share of resampled deltas that are *not* a regression, with the
        standard ``+1`` in numerator and denominator so it is strictly positive.
        """
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
        n_not_regressed = sum(1 for d in deltas if d >= 0.0)
        p_value = (1 + n_not_regressed) / (len(deltas) + 1)
        deltas.sort()
        low = _percentile(deltas, 100.0 * (self._alpha / 2))
        high = _percentile(deltas, 100.0 * (1 - self._alpha / 2))
        return low, high, p_value

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
