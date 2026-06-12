from __future__ import annotations

from llm_evalgate.eval.harness import EvalHarness
from llm_evalgate.judge.faithfulness import ClaimFaithfulnessDimension


def _scripted(responses):
    """A fake ``complete`` that returns queued responses in order."""
    queue = list(responses)
    calls = []

    def complete(prompt: str) -> str:
        calls.append(prompt)
        return queue.pop(0)

    complete.calls = calls
    return complete


EXTRACT_3 = "CLAIM: a\nCLAIM: b\nCLAIM: c"


def test_three_claims_strict_scoring():
    complete = _scripted([
        EXTRACT_3,
        "VERDICT: SUPPORTED\nREASON: ok",
        "VERDICT: UNSUPPORTED\nREASON: not in evidence",
        "VERDICT: CONTRADICTED\nREASON: evidence says otherwise",
    ])
    dim = ClaimFaithfulnessDimension(complete, evidence=["e"], unsupported_credit=0.0)
    score, detail = dim.evaluate("text")
    assert abs(score - 1 / 3) < 1e-9
    assert "CONTRADICTED" in detail
    assert '"c"' in detail


def test_unsupported_credit_changes_score():
    complete = _scripted([
        EXTRACT_3,
        "VERDICT: SUPPORTED\nREASON: ok",
        "VERDICT: UNSUPPORTED\nREASON: x",
        "VERDICT: CONTRADICTED\nREASON: y",
    ])
    dim = ClaimFaithfulnessDimension(complete, evidence=["e"], unsupported_credit=0.5)
    score, _ = dim.evaluate("text")
    assert abs(score - (1 + 0.5) / 3) < 1e-9


def test_no_claims_scores_one():
    complete = _scripted(["NONE"])
    dim = ClaimFaithfulnessDimension(complete, evidence=["e"])
    score, detail = dim.evaluate("text")
    assert score == 1.0
    assert "no factual claims" in detail


def test_extraction_failure_scores_zero():
    def boom(prompt: str) -> str:
        raise RuntimeError("model down")

    dim = ClaimFaithfulnessDimension(boom, evidence=["e"])
    score, detail = dim.evaluate("text")
    assert score == 0.0
    assert "extraction failed" in detail


def test_unparseable_verdict_counts_unsupported():
    complete = _scripted([
        "CLAIM: only one",
        "the model rambled without a verdict line",
    ])
    dim = ClaimFaithfulnessDimension(complete, evidence=["e"])
    score, _ = dim.evaluate("text")
    assert score == 0.0  # unsupported, strict credit


def test_callable_evidence_receives_text():
    seen = {}

    def retriever(text: str):
        seen["text"] = text
        return ["retrieved evidence"]

    complete = _scripted([
        "CLAIM: x",
        "VERDICT: SUPPORTED\nREASON: ok",
    ])
    dim = ClaimFaithfulnessDimension(complete, evidence=retriever)
    dim.evaluate("the answer under eval")
    assert seen["text"] == "the answer under eval"


def test_max_claims_truncation_reported():
    extract = "\n".join(f"CLAIM: c{i}" for i in range(5))
    responses = [extract] + ["VERDICT: SUPPORTED\nREASON: ok"] * 2
    complete = _scripted(responses)
    dim = ClaimFaithfulnessDimension(complete, evidence=["e"], max_claims=2)
    score, detail = dim.evaluate("text")
    assert score == 1.0  # both checked claims supported
    assert "truncated to first 2" in detail


def test_composes_in_eval_harness():
    complete = _scripted([
        "CLAIM: x",
        "VERDICT: SUPPORTED\nREASON: ok",
    ])
    dim = ClaimFaithfulnessDimension(complete, evidence=["e"], threshold=0.9)
    harness = EvalHarness([dim])
    report = harness.run("text")
    assert report.passed
