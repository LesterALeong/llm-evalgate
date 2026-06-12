from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .metrics import (
    accuracy,
    all_metrics,
    cohen_kappa,
    f1,
    precision,
    recall,
    regression_catch_rate,
)
from .stats import ConfidenceInterval, bootstrap_ci

# Maps each metric name produced by ``all_metrics`` to its function, so a
# confidence interval can be bootstrapped per metric.
_METRIC_FNS = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "cohen_kappa": cohen_kappa,
    "regression_catch_rate": regression_catch_rate,
}


@dataclass
class BenchSample:
    text: str
    label: bool
    meta: dict[str, Any] = field(default_factory=dict)


_SMALL_STRATUM = 10  # below this, per-stratum metrics are too noisy to trust


@dataclass
class BenchmarkResult:
    predicted: list[bool]
    labels: list[bool]
    metrics: dict[str, float]
    n: int
    intervals: dict[str, ConfidenceInterval] | None = None
    dataset_fingerprint: str | None = None
    created_at: str | None = None
    metas: list[dict[str, Any]] | None = None
    strata: dict[str, BenchmarkResult] | None = None

    def _metric_lines(self) -> list[str]:
        width = max(len(name) for name in self.metrics)
        lines = []
        for name, value in self.metrics.items():
            line = f"{name.ljust(width)}  {value:.3f}"
            if self.intervals is not None and name in self.intervals:
                ci = self.intervals[name]
                line += f"  [{ci.low:.3f}, {ci.high:.3f}]"
            lines.append(line)
        return lines

    def table(self) -> str:
        """Render an aligned metric/value table, with CIs and strata when present."""
        lines = [f"n={self.n}"]
        lines.extend(self._metric_lines())
        if self.strata:
            for name, sub in self.strata.items():
                lines.append(f"--- stratum={name} (n={sub.n})")
                lines.extend(f"  {line}" for line in sub._metric_lines())
                if sub.n < _SMALL_STRATUM:
                    lines.append(
                        f"  WARN: n={sub.n} < {_SMALL_STRATUM}; per-stratum metrics "
                        "are noisy."
                    )
        return "\n".join(lines)

    def save(self, path: str | Path) -> None:
        """Persist this result as JSON so it can serve as a gate baseline."""
        payload = {
            "predicted": self.predicted,
            "labels": self.labels,
            "metrics": self.metrics,
            "n": self.n,
            "dataset_fingerprint": self.dataset_fingerprint,
            "created_at": self.created_at,
            "metas": self.metas,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkResult:
        """Load a result previously written with :meth:`save`."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            predicted=data["predicted"],
            labels=data["labels"],
            metrics=data["metrics"],
            n=data["n"],
            intervals=None,
            dataset_fingerprint=data.get("dataset_fingerprint"),
            created_at=data.get("created_at"),
            metas=data.get("metas"),
        )


def _build_result(
    predicted: list[bool],
    labels: list[bool],
    metas: list[dict[str, Any]],
    *,
    ci: bool,
    n_resamples: int,
    seed: int | None,
    fingerprint: str | None = None,
    created_at: str | None = None,
    strata: dict[str, BenchmarkResult] | None = None,
) -> BenchmarkResult:
    metrics = all_metrics(predicted, labels)
    intervals: dict[str, ConfidenceInterval] | None = None
    if ci:
        intervals = {
            name: bootstrap_ci(fn, predicted, labels, n_resamples=n_resamples, seed=seed)
            for name, fn in _METRIC_FNS.items()
        }
    return BenchmarkResult(
        predicted=predicted,
        labels=labels,
        metrics=metrics,
        n=len(predicted),
        intervals=intervals,
        dataset_fingerprint=fingerprint,
        created_at=created_at,
        metas=metas,
        strata=strata,
    )


def fingerprint_samples(samples: list[BenchSample]) -> str:
    """Stable sha256 over the ordered sample texts.

    Used to detect when two benchmark runs were graded on different datasets,
    which would make a paired comparison meaningless.
    """
    digest = hashlib.sha256()
    for sample in samples:
        digest.update(sample.text.encode("utf-8"))
        digest.update(b"\x00")
    return digest.hexdigest()


class BenchmarkRunner:
    """Grade labeled samples and score the grader against the human labels.

    ``grader`` is anything with a ``run(text)`` method returning an object
    that has a ``.passed`` bool, i.e. a ``Dimension`` or an ``EvalHarness``.
    """

    def __init__(self, grader: Any) -> None:
        self._grader = grader

    def run(
        self,
        samples: list[BenchSample],
        *,
        ci: bool = True,
        n_resamples: int = 2000,
        seed: int | None = 0,
        stratify_by: str | None = None,
    ) -> BenchmarkResult:
        if not samples:
            raise ValueError("BenchmarkRunner requires at least one sample.")
        predicted = [self._grader.run(sample.text).passed for sample in samples]
        labels = [sample.label for sample in samples]
        metas = [dict(sample.meta) for sample in samples]

        strata: dict[str, BenchmarkResult] | None = None
        if stratify_by is not None:
            strata = self._build_strata(
                predicted, labels, metas, stratify_by, ci, n_resamples, seed
            )

        return _build_result(
            predicted,
            labels,
            metas,
            ci=ci,
            n_resamples=n_resamples,
            seed=seed,
            fingerprint=fingerprint_samples(samples),
            created_at=datetime.now(timezone.utc).isoformat(),
            strata=strata,
        )

    @staticmethod
    def _build_strata(
        predicted, labels, metas, stratify_by, ci, n_resamples, seed
    ) -> dict[str, BenchmarkResult]:
        groups: dict[str, list[int]] = {}
        for i, meta in enumerate(metas):
            key = str(meta.get(stratify_by, "(none)"))
            groups.setdefault(key, []).append(i)
        strata = {}
        for key in sorted(groups):
            idx = groups[key]
            strata[key] = _build_result(
                [predicted[i] for i in idx],
                [labels[i] for i in idx],
                [metas[i] for i in idx],
                ci=ci,
                n_resamples=n_resamples,
                seed=seed,
            )
        return strata


def load_golden(path: str | None = None) -> list[BenchSample]:
    """Load a JSONL golden dataset into ``BenchSample`` objects.

    Defaults to the bundled ``datasets/golden_eval.jsonl``. Each line is a
    JSON object ``{"text": ..., "label": bool, "meta": {...}}`` where ``meta``
    is optional.
    """
    if path is None:
        dataset_path = Path(__file__).resolve().parent / "datasets" / "golden_eval.jsonl"
    else:
        dataset_path = Path(path)
    samples: list[BenchSample] = []
    with dataset_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            samples.append(
                BenchSample(
                    text=record["text"],
                    label=record["label"],
                    meta=record.get("meta", {}),
                )
            )
    return samples
