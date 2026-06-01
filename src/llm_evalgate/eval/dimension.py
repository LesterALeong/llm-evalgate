from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class DimensionResult:
    score: float
    passed: bool
    detail: str


class Dimension(ABC):
    """Base class for a single eval dimension.

    Subclass this and implement ``evaluate``. The harness calls ``run``,
    which applies the threshold and returns a ``DimensionResult``.
    """

    def __init__(self, threshold: float = 1.0, name: str | None = None) -> None:
        self.threshold = threshold
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(self, text: str) -> tuple[float, str]:
        """Return (score, detail). Score is in [0.0, 1.0]."""

    def run(self, text: str) -> DimensionResult:
        score, detail = self.evaluate(text)
        return DimensionResult(score=score, passed=score >= self.threshold, detail=detail)
