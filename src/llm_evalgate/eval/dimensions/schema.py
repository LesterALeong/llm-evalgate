from __future__ import annotations

from ..dimension import Dimension


class SchemaComplianceDimension(Dimension):
    """Check that required fields are present in the text.

    Useful for structured LLM outputs (JSON, YAML, markdown with
    required sections) where missing fields are a hard failure.

    ``required_fields`` is a list of strings that must each appear
    verbatim somewhere in the text.
    """

    def __init__(
        self,
        required_fields: list[str],
        threshold: float = 1.0,
        name: str = "schema_compliance",
    ) -> None:
        super().__init__(threshold=threshold, name=name)
        self._required = required_fields

    def evaluate(self, text: str) -> tuple[float, str]:
        missing = [f for f in self._required if f not in text]
        if missing:
            score = 1.0 - len(missing) / len(self._required)
            return score, f"missing fields: {missing}"
        return 1.0, f"all {len(self._required)} required fields present"
