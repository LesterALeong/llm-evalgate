from __future__ import annotations

import random
from dataclasses import dataclass

from ..bench.stats import _percentile
from .calibration import CalibrationReport

__all__ = ["CorrectedRate", "rogan_gladen", "corrected_pass_rate"]


def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    """Bias-correct an observed positive rate for an imperfect classifier.

    The Rogan-Gladen estimator (1978) recovers the true prevalence from the rate
    a noisy classifier reports, given the classifier's sensitivity (true-positive
    rate) and specificity (true-negative rate)::

        true = (observed + specificity - 1) / (sensitivity + specificity - 1)

    An LLM judge is exactly such a noisy classifier, so its raw pass rate is a
    biased estimate of the true pass rate unless sensitivity and specificity are
    both 1. The result is clamped to ``[0, 1]``. Raises ``ValueError`` when
    ``sensitivity + specificity <= 1`` (the judge carries no usable signal and
    the correction is undefined).
    """
    denom = sensitivity + specificity - 1.0
    if denom <= 0.0:
        raise ValueError(
            "sensitivity + specificity must exceed 1 for the Rogan-Gladen "
            f"correction; got sensitivity={sensitivity}, specificity={specificity}"
        )
    corrected = (observed_rate + specificity - 1.0) / denom
    return max(0.0, min(1.0, corrected))


@dataclass(frozen=True)
class CorrectedRate:
    observed: float
    corrected: float
    ci_low: float
    ci_high: float
    n_eval: int
    n_calibration: int
    warning: str | None = None

    def __str__(self) -> str:
        base = (
            f"observed={self.observed:.3f}; corrected={self.corrected:.3f} "
            f"[{self.ci_low:.3f}, {self.ci_high:.3f}] "
            f"(n_eval={self.n_eval}, n_calibration={self.n_calibration})"
        )
        return base if self.warning is None else f"{base}; WARNING: {self.warning}"


def corrected_pass_rate(
    judge_labels: list[bool],
    calibration: CalibrationReport,
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> CorrectedRate:
    """Bias-corrected judge pass rate with a CI over both uncertainty sources.

    ``judge_labels`` are the judge's pass/fail verdicts on the eval set.
    ``calibration`` must carry a confusion matrix (i.e. it was produced by
    :func:`calibrate_judge` on samples with ``human_label`` set).

    The confidence interval propagates *two* independent sources of error, which
    is the whole point of doing this properly: the finite eval set (resampled to
    perturb the observed rate) and the finite calibration set (its confusion
    matrix rows resampled to perturb sensitivity/specificity). Bootstrap draws
    where the resampled judge is no better than chance (sensitivity + specificity
    <= 1) are discarded; if more than 20% are discarded the result carries a
    warning that the judge is too close to chance for the correction to be
    trustworthy.
    """
    if not judge_labels:
        raise ValueError("judge_labels must be non-empty")
    if calibration.confusion is None:
        raise ValueError(
            "calibration must include a confusion matrix; build it with "
            "calibrate_judge on samples that set human_label."
        )

    c = calibration.confusion
    tp, fn, tn, fp = c["tp"], c["fn"], c["tn"], c["fp"]
    n_pos = tp + fn  # human-positive calibration cases
    n_neg = tn + fp  # human-negative calibration cases
    if n_pos == 0 or n_neg == 0:
        raise ValueError(
            "calibration confusion matrix needs both human-positive and "
            "human-negative cases to estimate sensitivity and specificity."
        )

    observed = sum(1 for v in judge_labels if v) / len(judge_labels)
    point_sens = tp / n_pos
    point_spec = tn / n_neg

    # If the point calibration is itself at or below chance, the correction is
    # undefined; return the observed rate with a warning rather than raising.
    if point_sens + point_spec <= 1.0:
        return CorrectedRate(
            observed=observed,
            corrected=observed,
            ci_low=observed,
            ci_high=observed,
            n_eval=len(judge_labels),
            n_calibration=n_pos + n_neg,
            warning=(
                "point calibration is at or below chance "
                f"(sensitivity {point_sens:.3f} + specificity {point_spec:.3f} <= 1); "
                "returning the uncorrected observed rate."
            ),
        )

    point = rogan_gladen(observed, point_sens, point_spec)

    # The two calibration "rows" as 0/1 correctness vectors, resampled
    # nonparametrically: a human-positive case scores 1 if the judge passed it
    # (a true positive), a human-negative case scores 1 if the judge failed it.
    sens_vector = [1] * tp + [0] * fn
    spec_vector = [1] * tn + [0] * fp

    rng = random.Random(seed)
    n_eval = len(judge_labels)
    draws: list[float] = []
    discarded = 0
    for _ in range(n_resamples):
        # (a) resample the eval verdicts -> observed rate
        obs = sum(judge_labels[rng.randrange(n_eval)] for _ in range(n_eval)) / n_eval
        # (b) resample each calibration class -> sensitivity, specificity
        sens = sum(sens_vector[rng.randrange(n_pos)] for _ in range(n_pos)) / n_pos
        spec = sum(spec_vector[rng.randrange(n_neg)] for _ in range(n_neg)) / n_neg
        if sens + spec <= 1.0:
            discarded += 1
            continue
        draws.append(rogan_gladen(obs, sens, spec))

    warning = None
    if discarded > 0.2 * n_resamples:
        warning = (
            f"{discarded}/{n_resamples} bootstrap draws had sensitivity+specificity "
            "<= 1; the judge is close to chance and the correction is unreliable."
        )
    if not draws:
        # Degenerate: every draw was discarded. Fall back to the point estimate.
        return CorrectedRate(
            observed=observed,
            corrected=point,
            ci_low=point,
            ci_high=point,
            n_eval=n_eval,
            n_calibration=n_pos + n_neg,
            warning=warning or "all bootstrap draws were at or below chance.",
        )

    draws.sort()
    ci_low = _percentile(draws, 100.0 * (alpha / 2))
    ci_high = _percentile(draws, 100.0 * (1 - alpha / 2))
    return CorrectedRate(
        observed=observed,
        corrected=point,
        ci_low=ci_low,
        ci_high=ci_high,
        n_eval=n_eval,
        n_calibration=n_pos + n_neg,
        warning=warning,
    )
