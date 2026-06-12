# Tier 2 Spec — Coverage Holes: Faithfulness, Tool Hallucination, Review Routing

Status: DRAFT — awaiting Lester's review. No code written.
Target version: 0.4.0 (ship with Tier 1) or 0.5.0 if staged.
Theme: three real failure modes the library currently cannot express — RAG
unfaithfulness beyond numerics, agents inventing tools, and judge verdicts that are
too unstable to trust silently.

Sources: interviewstack llm-engineering ch4 (RAG faithfulness as the canonical gate),
llm-orchestration ch5 (RAGAS mechanics, validation nodes), ch6-evaluation
(low-confidence routing to humans); FActScore-style claim decomposition (ACL 2025
claim-extraction line of work); 2025 agent-hallucination literature (tool-misuse as the
top agent failure class).

---

## T2.4 ClaimFaithfulnessDimension — claim-level groundedness

### Problem
`FactualGroundingDimension` (eval/dimensions/factual.py) only checks **numbers** against
evidence. A fluent answer that invents a qualitative fact ("the contract allows early
termination") sails through every deterministic gate. Both courses treat
faithfulness — every claim supported by the provided context — as *the* RAG generation
metric, with mechanical scoring (a claim is in the chunks or it is not).

### Design

New module `src/llm_evalgate/judge/faithfulness.py`. It is a `Dimension` (drops into
`EvalHarness` next to deterministic gates) but lives under judge/ because it requires a
`complete` callable — same injection pattern as `JudgeDimension`, so it stays
offline-testable with a fake callable.

```python
class ClaimFaithfulnessDimension(Dimension):
    def __init__(
        self,
        complete: Callable[[str], str],
        evidence: list[str] | Callable[[str], list[str]],  # static or per-text retriever
        *,
        threshold: float = 0.9,
        max_claims: int = 20,
        unsupported_credit: float = 0.0,   # 0.0 = strict FActScore-style
        name: str = "claim_faithfulness",
    ) -> None: ...
```

**Two-stage pipeline:**

1. **Extract** — one model call. Prompt: decompose the text into atomic, independently
   verifiable factual claims, one per line, `CLAIM: <text>` format. Parse with a
   tolerant regex (same philosophy as `parse_pairwise`); cap at `max_claims` (cost
   bound), note truncation in detail. Zero claims extracted → score 1.0 with detail
   "no factual claims found" (an answer with no claims cannot be unfaithful —
   matches FactualGroundingDimension's no-numbers convention).
2. **Verify** — one model call **per claim** (bounded by `max_claims`). Prompt presents
   the evidence and one claim, demands exactly
   `VERDICT: <SUPPORTED|UNSUPPORTED|CONTRADICTED>` + `REASON: <one sentence>`.
   Unparseable verdict → counted UNSUPPORTED with the parse failure recorded (fail
   closed, same as JudgeDimension).

**Scoring:**

```
score = (supported + unsupported_credit * unsupported) / total_claims
```

CONTRADICTED always scores 0 and is reported separately in the detail string —
the extrinsic/intrinsic hallucination split from the course material:

```
claims=7: 5 supported, 1 unsupported, 1 contradicted; score=0.714
  CONTRADICTED: "the contract allows early termination" (evidence says 90-day notice)
```

**Error behavior:** never raises into the pipeline. Extraction-call exception →
score 0.0 with reason. Per-claim verification exception → that claim UNSUPPORTED.

**Cost note (document in README):** 1 + min(claims, max_claims) model calls per text.
This is inherently a judge-tier check; the README's "deterministic first" framing
stays intact.

### Acceptance criteria
- [ ] Fake-complete test: 3 claims extracted, verdicts S/U/C → score 1/3 (strict),
      2/3 with `unsupported_credit=0.5`; detail lists the contradicted claim.
- [ ] No-claims text → 1.0. Extraction failure → 0.0 with reason.
- [ ] Callable evidence (retriever) is invoked with the text under eval.
- [ ] `max_claims` truncation works and is reported.
- [ ] Composes inside `EvalHarness` with deterministic dimensions; report renders.
- [ ] Fully offline test suite (no model calls).

### Files
- NEW `src/llm_evalgate/judge/faithfulness.py`
- EDIT `src/llm_evalgate/judge/__init__.py`, `src/llm_evalgate/__init__.py`
- NEW `tests/test_faithfulness.py`
- EDIT `README.md`; NEW example or extend `examples/judge_eval.py`

---

## T2.5 ToolHallucinationDimension — agents inventing tools/args

### Problem
Nothing in agentic/dimensions/ catches an agent calling a tool that does not exist, or
calling a real tool with missing-required / unknown arguments. `ToolSelectionDimension`
checks expected tools were used; `ToolArgValidityDimension` runs user validators and
checks `.error` — neither validates against the *available tool schema*. Tool-call
hallucination is the most-cited agent failure mode in the 2025 literature.

### Design

New module `src/llm_evalgate/agentic/dimensions/tool_hallucination.py`:

```python
@dataclass(frozen=True)
class ToolSchema:
    required: frozenset[str] = frozenset()
    optional: frozenset[str] = frozenset()

class ToolHallucinationDimension(TraceDimension):
    def __init__(
        self,
        tools: dict[str, ToolSchema],     # the agent's actual tool registry
        *,
        allow_extra_args: bool = False,
        threshold: float = 1.0,           # zero tolerance by default
        name: str = "tool_hallucination",
    ) -> None: ...
```

A call is **hallucinated** when any of:
1. `call.name not in tools` — phantom tool;
2. a `required` arg is absent from `call.args`;
3. `allow_extra_args=False` and an arg is in neither `required` nor `optional`.

`score = 1 - hallucinated_calls / total_calls`; no tool calls → 1.0 (consistent with
ToolArgValidityDimension). Detail names each offender with its category:

```
2/5 calls hallucinated: get_wether (unknown tool); get_weather missing required arg 'city'
```

Convenience constructor `ToolSchema.from_json_schema(schema: dict)` that reads a
standard JSON-Schema tool definition (`properties` + `required`) — this is the shape
people already have from Anthropic/OpenAI tool definitions, so adoption is one line.

### Acceptance criteria
- [ ] Phantom tool, missing-required, and unexpected-arg each detected and labeled.
- [ ] `allow_extra_args=True` suppresses category 3 only.
- [ ] Clean trace → 1.0 PASS; empty trace → 1.0.
- [ ] `from_json_schema` handles a real Anthropic-style tool definition fixture.
- [ ] Composes in `AgentEvalHarness` with the existing five dimensions.

### Files
- NEW `src/llm_evalgate/agentic/dimensions/tool_hallucination.py`
- EDIT `src/llm_evalgate/agentic/__init__.py` (+ dimensions/__init__.py), root exports
- EDIT `tests/test_agentic.py` (or NEW `tests/test_tool_hallucination.py`)
- EDIT `README.md` agentic table; EDIT `examples/agentic_eval.py`

---

## T2.6 Low-confidence routing — `needs_review`

### Problem
The library already *detects* unstable verdicts — wide `SelfConsistencyJudge` spread,
`PairwiseJudge` order-flips (`consistent=False`) — but only prints them. The course
discipline is: an ambiguous judge verdict should be routed to a human, not silently
trusted as pass/fail. There is no programmatic hook for that today.

### Design

**Core change:** `DimensionResult` (eval/dimension.py) gains
`needs_review: bool = False` — additive, default keeps every existing constructor call
and test valid.

**Producers:**

1. `SelfConsistencyJudge` gains two opt-in knobs (both default `None` = off):
   - `max_stdev: float | None` — flag when `dist.stdev > max_stdev` (the score is
     a coin flip);
   - `review_margin: float | None` — flag when `|score - threshold| < review_margin`
     *and* the 95% CI straddles the threshold (the verdict is within noise of the
     gate line).
   `evaluate()` keeps returning `(score, detail)`; `run()` is overridden to set
   `needs_review` and append the trigger to detail (`"REVIEW: stdev 0.21 > 0.15"`).
2. `JuryDimension` gains `max_disagreement: float | None` — same pattern, keyed on the
   existing agreement/spread measure.
3. `PairwiseResult` gains a `needs_review` property: `not self.consistent`. (No
   behavior change — the order-flip-→ -tie fallback stays; this is just the explicit
   routing signal.)

**Consumers:**

4. `EvalReport` / `AgentEvalReport` gain `needs_review: list[DimensionResult]`
   (computed) and render a `REVIEW` marker line per flagged dimension:

   ```
   EvalReport: PASS (1 dimension flagged for review)
     PASS [blocklist]         score=1.000 - ...
     PASS [self_consistency]  score=0.640 - ... REVIEW: CI [0.55, 0.73] straddles threshold 0.6
   ```

   `report.passed` semantics unchanged — review is orthogonal to pass/fail; the caller
   decides what to do with flagged items (route to human queue, log, block).

### Acceptance criteria
- [ ] `DimensionResult` default keeps all existing tests green untouched.
- [ ] stdev trigger, margin trigger, jury-disagreement trigger each unit-tested with
      fake judges (deterministic score sequences).
- [ ] Margin trigger does NOT fire when CI is clear of threshold even if score is close.
- [ ] Report rendering shows the marker; `report.needs_review` lists flagged results.
- [ ] `passed` unchanged by review flags.

### Files
- EDIT `src/llm_evalgate/eval/dimension.py`, `eval/harness.py`,
  `agentic/harness.py`, `judge/consistency.py`, `judge/base.py` (jury),
  `judge/pairwise.py`
- EDIT `tests/test_consistency.py`, `tests/test_judge.py`, `tests/test_harness.py`,
  `tests/test_pairwise.py`
- EDIT `README.md` (judge-reliability section: "route, don't trust")

---

## Tier 2 open questions

1. **Verification batching (T2.4):** spec says one call per claim (simple, robust
   parsing). A `batch=True` single-call mode would cut tokens ~10x but parsing gets
   fragile. Recommended: per-claim now, batch later if cost complaints materialize.
2. **Where does faithfulness live** — `judge/` (needs a model) or `eval/dimensions/`
   (it's a Dimension)? Spec says judge/. Cosmetic; implementer may argue.
3. **T2.6 scope check:** touching 6 files for a bool flag is the honest cost of doing
   routing properly. Alternative is a wrapper class (`ReviewRouter(judge, ...)`), less
   invasive but a second way of doing things. Recommended: the flag.
