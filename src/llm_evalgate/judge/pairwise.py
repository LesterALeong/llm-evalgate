from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_PAIRWISE_TEMPLATE = (
    "You are an impartial judge comparing two responses.\n\n"
    "Criteria:\n{criteria}\n\n"
    "Decide which response better satisfies the criteria.\n"
    "Reply with exactly two lines and nothing else:\n"
    "WINNER: <First|Second|Tie>\n"
    "REASON: <one sentence>\n\n"
    "First response:\n{first}\n\n"
    "Second response:\n{second}\n"
)

_WINNER_RE = re.compile(r"winner\s*:\s*(first|second|tie)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class PairwiseResult:
    winner: str
    consistent: bool
    order_ab_winner: str
    order_ba_winner: str
    reason: str
    raw: tuple[str, str]

    @property
    def needs_review(self) -> bool:
        """True when the verdict flipped on order swap, so it is not trustworthy."""
        return not self.consistent


def parse_pairwise(raw: str) -> tuple[str, str]:
    """Parse a pairwise response into ``(slot, reason)``, never raising.

    ``slot`` is ``"first"``, ``"second"``, or ``"tie"`` based on the WINNER
    line (case-insensitive). If no WINNER line is found, ``slot`` defaults to
    ``"tie"``. ``reason`` is the REASON text, or a fallback string when absent.
    """
    winner_match = _WINNER_RE.search(raw)
    reason_match = _REASON_RE.search(raw)
    reason = reason_match.group(1).strip() if reason_match else ""
    if winner_match is None:
        slot = "tie"
        if not reason:
            reason = "parse failure: no WINNER found in judge response"
    else:
        slot = winner_match.group(1).lower()
        if not reason:
            reason = "no REASON found in judge response"
    return slot, reason


# Maps the positional slot the model picked back to the real A/B label, keyed
# by which original answer was presented First.
_SLOT_TO_LABEL = {
    "A": {"first": "A", "second": "B", "tie": "tie"},
    "B": {"first": "B", "second": "A", "tie": "tie"},
}


class PairwiseJudge:
    """Pairwise LLM-as-judge with optional position-bias debiasing.

    ``complete`` maps a rendered prompt string to a model response string.
    Injecting it keeps the judge offline-testable: tests pass a fake callable
    and no network is touched. When ``debias_position`` is True the comparison
    is run in both orders and a verdict is only trusted when it agrees across
    the swap; otherwise the result falls back to a tie.
    """

    def __init__(
        self,
        complete: Callable[[str], str],
        *,
        criteria: str,
        prompt_template: str | None = None,
        debias_position: bool = True,
        name: str = "pairwise",
    ) -> None:
        self._complete = complete
        self._criteria = criteria
        self._template = prompt_template or DEFAULT_PAIRWISE_TEMPLATE
        self._debias_position = debias_position
        self.name = name

    def _judge_once(self, first: str, second: str) -> tuple[str, str]:
        """Render, call the model once, and parse. Returns ``(slot, reason)``, never raises."""
        prompt = self._template.format(
            criteria=self._criteria, first=first, second=second
        )
        try:
            raw = self._complete(prompt)
        except Exception as exc:
            return "tie", f"judge error: {exc}"
        return parse_pairwise(raw)

    def compare(self, a: str, b: str) -> PairwiseResult:
        """Compare answers ``a`` and ``b``, returning a PairwiseResult.

        Order AB presents ``a`` as First and ``b`` as Second. When debiasing,
        order BA also presents ``b`` as First and ``a`` as Second; the two
        verdicts must agree on the same real answer to be trusted.
        """
        ab_slot, ab_reason = self._judge_once(a, b)
        order_ab_winner = _SLOT_TO_LABEL["A"][ab_slot]
        # _judge_once parses the prompt internally; re-render here only to keep
        # a faithful raw record of what the model was shown and answered.
        ab_raw = f"WINNER: {ab_slot}\nREASON: {ab_reason}"

        if not self._debias_position:
            # Single-order verdict: consistency is not checked, so it is
            # reported as True and the BA fields mirror the AB result.
            return PairwiseResult(
                winner=order_ab_winner,
                consistent=True,
                order_ab_winner=order_ab_winner,
                order_ba_winner=order_ab_winner,
                reason=ab_reason,
                raw=(ab_raw, ""),
            )

        ba_slot, ba_reason = self._judge_once(b, a)
        order_ba_winner = _SLOT_TO_LABEL["B"][ba_slot]
        ba_raw = f"WINNER: {ba_slot}\nREASON: {ba_reason}"

        if order_ab_winner == order_ba_winner:
            winner = order_ab_winner
            consistent = True
        else:
            winner = "tie"
            consistent = False
        reason = f"AB: {ab_reason} | BA: {ba_reason}"

        return PairwiseResult(
            winner=winner,
            consistent=consistent,
            order_ab_winner=order_ab_winner,
            order_ba_winner=order_ba_winner,
            reason=reason,
            raw=(ab_raw, ba_raw),
        )


def position_bias_rate(
    judge: PairwiseJudge, pairs: list[tuple[str, str]]
) -> float:
    """Fraction of pairs whose verdict flipped when the order was swapped.

    For each ``(a, b)`` pair the judge is run in both orders directly via
    ``_judge_once`` (ignoring its ``debias_position`` flag). A pair is counted
    as inconsistent when the AB-order pick and the BA-order pick disagree on
    the real answer, which is the position-bias signature. Raises ``ValueError``
    on empty input.
    """
    if not pairs:
        raise ValueError("position_bias_rate requires at least one pair.")
    inconsistent = 0
    for a, b in pairs:
        ab_slot, _ = judge._judge_once(a, b)
        ba_slot, _ = judge._judge_once(b, a)
        order_ab_winner = _SLOT_TO_LABEL["A"][ab_slot]
        order_ba_winner = _SLOT_TO_LABEL["B"][ba_slot]
        if order_ab_winner != order_ba_winner:
            inconsistent += 1
    return inconsistent / len(pairs)
