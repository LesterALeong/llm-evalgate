from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from ..eval.dimension import Dimension

DEFAULT_EXTRACT_TEMPLATE = (
    "Decompose the text below into its distinct, atomic factual claims.\n"
    "Each claim must be independently verifiable and stand on its own.\n"
    "List one claim per line, each prefixed with 'CLAIM: '.\n"
    "If the text makes no factual claims, reply with exactly 'NONE'.\n\n"
    "Text:\n{text}\n"
)

DEFAULT_VERIFY_TEMPLATE = (
    "You are checking whether a claim is supported by the evidence.\n\n"
    "Evidence:\n{evidence}\n\n"
    "Claim:\n{claim}\n\n"
    "Reply with exactly two lines and nothing else:\n"
    "VERDICT: <SUPPORTED|UNSUPPORTED|CONTRADICTED>\n"
    "REASON: <one sentence>\n"
)

_CLAIM_RE = re.compile(r"^\s*claim\s*:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
_VERDICT_RE = re.compile(r"verdict\s*:\s*(supported|unsupported|contradicted)", re.IGNORECASE)


@dataclass(frozen=True)
class ClaimVerdict:
    claim: str
    verdict: str  # "supported" | "unsupported" | "contradicted"
    reason: str


class ClaimFaithfulnessDimension(Dimension):
    """Claim-level groundedness: is every factual claim supported by the evidence?

    Unlike :class:`FactualGroundingDimension`, which only traces *numbers*, this
    decomposes the text into atomic claims and verifies each against the provided
    evidence with a model. It is the canonical RAG faithfulness check: a fluent,
    well-formatted answer that invents a qualitative fact scores low here.

    ``complete`` maps a prompt to a response string (inject a fake in tests, the
    Anthropic adapter in production). ``evidence`` is either a fixed list of
    context strings or a callable that retrieves context for the text under eval.

    Cost: ``1 + min(num_claims, max_claims)`` model calls per text. This is a
    judge-tier check; keep cheap deterministic gates ahead of it.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        evidence: list[str] | Callable[[str], list[str]],
        *,
        threshold: float = 0.9,
        max_claims: int = 20,
        unsupported_credit: float = 0.0,
        extract_template: str | None = None,
        verify_template: str | None = None,
        name: str = "claim_faithfulness",
    ) -> None:
        if not 0.0 <= unsupported_credit <= 1.0:
            raise ValueError(
                f"unsupported_credit must be in [0, 1]; got {unsupported_credit}"
            )
        if max_claims < 1:
            raise ValueError(f"max_claims must be >= 1; got {max_claims}")
        super().__init__(threshold=threshold, name=name)
        self._complete = complete
        self._evidence = evidence
        self._max_claims = max_claims
        self._unsupported_credit = unsupported_credit
        self._extract_template = extract_template or DEFAULT_EXTRACT_TEMPLATE
        self._verify_template = verify_template or DEFAULT_VERIFY_TEMPLATE

    def _resolve_evidence(self, text: str) -> str:
        evidence = self._evidence(text) if callable(self._evidence) else self._evidence
        return "\n".join(f"- {item}" for item in evidence)

    def extract_claims(self, text: str) -> list[str]:
        """Call the model once and parse atomic claims. Never raises."""
        prompt = self._extract_template.format(text=text)
        try:
            raw = self._complete(prompt)
        except Exception as exc:  # noqa: BLE001 - fail closed, surface the reason
            raise _ExtractionError(str(exc)) from exc
        claims = [m.strip() for m in _CLAIM_RE.findall(raw) if m.strip()]
        return claims

    def verify_claim(self, claim: str, evidence: str) -> ClaimVerdict:
        """Verify one claim against evidence. Never raises; failures are UNSUPPORTED."""
        prompt = self._verify_template.format(evidence=evidence, claim=claim)
        try:
            raw = self._complete(prompt)
        except Exception as exc:  # noqa: BLE001 - fail closed
            return ClaimVerdict(claim, "unsupported", f"verify error: {exc}")
        match = _VERDICT_RE.search(raw)
        if match is None:
            return ClaimVerdict(claim, "unsupported", "parse failure: no VERDICT found")
        return ClaimVerdict(claim, match.group(1).lower(), _reason(raw))

    def evaluate(self, text: str) -> tuple[float, str]:
        try:
            claims = self.extract_claims(text)
        except _ExtractionError as exc:
            return 0.0, f"claim extraction failed: {exc}"

        if not claims:
            return 1.0, "no factual claims found"

        truncated = len(claims) > self._max_claims
        claims = claims[: self._max_claims]
        evidence = self._resolve_evidence(text)

        verdicts = [self.verify_claim(claim, evidence) for claim in claims]
        supported = sum(1 for v in verdicts if v.verdict == "supported")
        unsupported = sum(1 for v in verdicts if v.verdict == "unsupported")
        contradicted = sum(1 for v in verdicts if v.verdict == "contradicted")
        total = len(verdicts)

        score = (supported + self._unsupported_credit * unsupported) / total

        detail = (
            f"claims={total}: {supported} supported, {unsupported} unsupported, "
            f"{contradicted} contradicted; score={score:.3f}"
        )
        contradictions = [v for v in verdicts if v.verdict == "contradicted"]
        if contradictions:
            detail += "; CONTRADICTED: " + "; ".join(
                f'"{v.claim}" ({v.reason})' for v in contradictions
            )
        if truncated:
            detail += f"; note: truncated to first {self._max_claims} claims"
        return score, detail


class _ExtractionError(Exception):
    """Internal signal that the extraction model call failed."""


def _reason(raw: str) -> str:
    match = re.search(r"reason\s*:\s*(.+)", raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""
