from __future__ import annotations

import textstat

from ..dimension import Dimension


class ReadabilityDimension(Dimension):
    """Pass when Flesch Reading Ease score maps to a grade <= max_grade.

    ``threshold`` is a normalised [0, 1] score derived from Flesch Reading
    Ease, where 1.0 = very easy and 0.0 = very difficult.  The default
    threshold of 0.3 accepts most professional prose up to ~college level.
    """

    def __init__(self, threshold: float = 0.3, name: str = "readability") -> None:
        super().__init__(threshold=threshold, name=name)

    def evaluate(self, text: str) -> tuple[float, str]:
        if not text.strip():
            return 0.0, "empty text"
        ease = textstat.flesch_reading_ease(text)
        # Flesch ease: 100=very easy, 0=very hard. Normalise to [0, 1].
        score = max(0.0, min(1.0, ease / 100.0))
        grade = textstat.flesch_kincaid_grade(text)
        return score, f"Flesch ease={ease:.1f}, FK grade={grade:.1f}"
