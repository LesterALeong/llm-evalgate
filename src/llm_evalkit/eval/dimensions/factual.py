from __future__ import annotations

import re

from ..dimension import Dimension


class FactualGroundingDimension(Dimension):
    """Check that numeric claims in LLM output are traceable to evidence.

    For each number extracted from the text, we check whether a value
    within ``rel_tolerance`` of it appears in the evidence list.  Score
    is the fraction of numeric claims that are grounded.

    If no evidence is supplied the dimension is skipped (returns 1.0).
    If the text contains no numbers, it also passes.
    """

    def __init__(
        self,
        evidence: list[float] | None = None,
        rel_tolerance: float = 0.02,
        threshold: float = 0.85,
        name: str = "factual_grounding",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._evidence = evidence or []
        self._rel_tolerance = rel_tolerance

    def _numbers_in_text(self, text: str) -> list[float]:
        raw = re.findall(r"[\d,]+(?:\.\d+)?", text)
        results = []
        for r in raw:
            try:
                results.append(float(r.replace(",", "")))
            except ValueError:
                pass
        return results

    def _is_grounded(self, value: float) -> bool:
        for ev in self._evidence:
            if ev == 0:
                continue
            if abs(value - ev) / abs(ev) <= self._rel_tolerance:
                return True
        return False

    def evaluate(self, text: str) -> tuple[float, str]:
        if not self._evidence:
            return 1.0, "skipped (no evidence supplied)"
        numbers = self._numbers_in_text(text)
        if not numbers:
            return 1.0, "no numeric claims found"
        grounded = [n for n in numbers if self._is_grounded(n)]
        score = len(grounded) / len(numbers)
        return score, f"{len(grounded)}/{len(numbers)} numeric claims grounded"
