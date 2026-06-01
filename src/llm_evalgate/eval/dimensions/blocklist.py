from __future__ import annotations

import re

from ..dimension import Dimension


class BlocklistDimension(Dimension):
    """Fail if any prohibited term appears in the text (case-insensitive).

    Score is 1.0 when clean, 0.0 when any term is found.  Useful for
    preventing confidential identifiers, brand names, or internal jargon
    from leaking into LLM output.
    """

    def __init__(
        self,
        terms: list[str],
        threshold: float = 1.0,
        name: str = "blocklist",
        case_sensitive: bool = False,
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        flags = 0 if case_sensitive else re.IGNORECASE
        self._patterns = [re.compile(re.escape(t), flags) for t in terms]
        self._terms = terms

    def evaluate(self, text: str) -> tuple[float, str]:
        found = [t for t, p in zip(self._terms, self._patterns) if p.search(text)]
        if found:
            return 0.0, f"prohibited terms found: {found}"
        return 1.0, "clean"
