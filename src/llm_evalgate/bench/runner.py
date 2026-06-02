from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .metrics import all_metrics


@dataclass
class BenchSample:
    text: str
    label: bool
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    predicted: list[bool]
    labels: list[bool]
    metrics: dict[str, float]
    n: int

    def table(self) -> str:
        """Render an aligned metric/value table with values to 3 decimals."""
        width = max(len(name) for name in self.metrics)
        lines = [f"n={self.n}"]
        for name, value in self.metrics.items():
            lines.append(f"{name.ljust(width)}  {value:.3f}")
        return "\n".join(lines)


class BenchmarkRunner:
    """Grade labeled samples and score the grader against the human labels.

    ``grader`` is anything with a ``run(text)`` method returning an object
    that has a ``.passed`` bool, i.e. a ``Dimension`` or an ``EvalHarness``.
    """

    def __init__(self, grader: Any) -> None:
        self._grader = grader

    def run(self, samples: list[BenchSample]) -> BenchmarkResult:
        if not samples:
            raise ValueError("BenchmarkRunner requires at least one sample.")
        predicted = [self._grader.run(sample.text).passed for sample in samples]
        labels = [sample.label for sample in samples]
        metrics = all_metrics(predicted, labels)
        return BenchmarkResult(
            predicted=predicted,
            labels=labels,
            metrics=metrics,
            n=len(samples),
        )


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
