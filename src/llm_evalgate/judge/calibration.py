from __future__ import annotations

from dataclasses import dataclass

from ..bench.metrics import accuracy, cohen_kappa, confusion_counts, mae, pearson, spearman
from .dimension import JudgeDimension


@dataclass
class CalibrationSample:
    text: str
    human_score: float | None = None
    human_label: bool | None = None


@dataclass(frozen=True)
class CalibrationReport:
    n: int
    pearson: float | None
    spearman: float | None
    mae: float | None
    accuracy: float | None
    cohen_kappa: float | None
    sensitivity: float | None = None
    specificity: float | None = None
    confusion: dict[str, int] | None = None

    def table(self) -> str:
        """Render an aligned report; ``n=<n>`` then each non-None metric."""
        metrics = [
            (name, value)
            for name, value in (
                ("pearson", self.pearson),
                ("spearman", self.spearman),
                ("mae", self.mae),
                ("accuracy", self.accuracy),
                ("cohen_kappa", self.cohen_kappa),
                ("sensitivity", self.sensitivity),
                ("specificity", self.specificity),
            )
            if value is not None
        ]
        lines = [f"n={self.n}"]
        if metrics:
            width = max(len(name) for name, _ in metrics)
            for name, value in metrics:
                lines.append(f"{name.ljust(width)}  {value:.3f}")
        return "\n".join(lines)


def calibrate_judge(
    judge: JudgeDimension, samples: list[CalibrationSample]
) -> CalibrationReport:
    if not samples:
        raise ValueError("samples must be non-empty")
    if any(s.human_score is None and s.human_label is None for s in samples):
        raise ValueError(
            "each sample must set at least one of human_score or human_label"
        )

    judge_scores = [judge.score(s.text).score for s in samples]

    score_pearson = score_spearman = score_mae = None
    if all(s.human_score is not None for s in samples):
        human_scores = [s.human_score for s in samples]
        score_pearson = pearson(judge_scores, human_scores)
        score_spearman = spearman(judge_scores, human_scores)
        score_mae = mae(judge_scores, human_scores)

    label_accuracy = label_kappa = None
    sensitivity = specificity = None
    confusion = None
    if all(s.human_label is not None for s in samples):
        human_labels = [s.human_label for s in samples]
        judge_labels = [s >= judge.threshold for s in judge_scores]
        label_accuracy = accuracy(judge_labels, human_labels)
        label_kappa = cohen_kappa(judge_labels, human_labels)
        confusion = confusion_counts(judge_labels, human_labels)
        pos = confusion["tp"] + confusion["fn"]  # human-positive total
        neg = confusion["tn"] + confusion["fp"]  # human-negative total
        sensitivity = confusion["tp"] / pos if pos else None
        specificity = confusion["tn"] / neg if neg else None

    return CalibrationReport(
        n=len(samples),
        pearson=score_pearson,
        spearman=score_spearman,
        mae=score_mae,
        accuracy=label_accuracy,
        cohen_kappa=label_kappa,
        sensitivity=sensitivity,
        specificity=specificity,
        confusion=confusion,
    )


def verbosity_bias(judge: JudgeDimension, texts: list[str]) -> float:
    """Pearson correlation between character length and judge score.

    A high positive value means the judge rewards longer outputs regardless of
    quality.
    """
    if not texts:
        raise ValueError("texts must be non-empty")
    lengths = [float(len(text)) for text in texts]
    scores = [judge.score(text).score for text in texts]
    return pearson(lengths, scores)
