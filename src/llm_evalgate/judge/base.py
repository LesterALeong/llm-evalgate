from __future__ import annotations

import re
from dataclasses import dataclass

DEFAULT_PROMPT_TEMPLATE = (
    "You are an impartial grader. Grade the text below against the rubric.\n\n"
    "Rubric:\n{rubric}\n\n"
    "Grade on a 1 to {scale} scale, where 1 is worst and {scale} is best.\n"
    "Reply with exactly two lines and nothing else:\n"
    "SCORE: <n>\n"
    "REASON: <one sentence>\n\n"
    "Text:\n{text}\n"
)

_SCORE_RE = re.compile(r"score\s*:\s*([-+]?\d*\.?\d+)", re.IGNORECASE)
_REASON_RE = re.compile(r"reason\s*:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True)
class JudgeVerdict:
    score: float
    reason: str
    raw: str


def parse_verdict(raw: str, scale: int) -> JudgeVerdict:
    """Parse a model response into a JudgeVerdict, never raising.

    Looks for ``SCORE:`` followed by a number and ``REASON:`` text. The score
    is normalized to [0.0, 1.0] by dividing by ``scale`` and clamping. When
    ``scale`` is 1 the value is treated as an already-normalized 0-1 float.
    If no score can be parsed, returns score 0.0 with an explanatory reason.
    """
    score_match = _SCORE_RE.search(raw)
    reason_match = _REASON_RE.search(raw)
    reason = reason_match.group(1).strip() if reason_match else ""
    if score_match is None:
        return JudgeVerdict(
            score=0.0,
            reason="parse failure: no SCORE found in judge response",
            raw=raw,
        )
    value = float(score_match.group(1))
    normalized = value if scale == 1 else value / scale
    clamped = max(0.0, min(1.0, normalized))
    if not reason:
        reason = "no REASON found in judge response"
    return JudgeVerdict(score=clamped, reason=reason, raw=raw)


def render_prompt(template: str, rubric: str, scale: int, text: str) -> str:
    return template.format(rubric=rubric, scale=scale, text=text)
