# Tier 3 Spec — Extensions: Step-Level Scoring, OTel Import, Stratified Golden Sets

Status: DRAFT — awaiting Lester's review. No code written.
Target version: 0.5.0 (do NOT bundle with Tier 1/2 — each of these is independently
shippable and none should block the statistical-rigor release).
Theme: adoption levers and depth. Step-level agent scoring (process-reward style),
importing real-world traces instead of hand-built ones, and making the golden-set
discipline from the courses expressible.

Sources: AgentPRM / process-reward-model line of work (2025, step-level promise+progress);
OpenTelemetry GenAI semantic conventions (incubating, 2025-26); interviewstack
llm-orchestration ch7 (traces/spans, stratified sampling) and ch11 (30/30/20 golden-set
composition, quarterly refresh).

---

## T3.7 StepProgressDimension — process-reward-style step scoring

### Problem
All five agentic dimensions score the trajectory as a whole; `TrajectoryCoherenceDimension`
is one judge call over the full serialized trace. For long-horizon agents the signal you
want is *which step* went wrong — outcome-level scoring can't localize failure. The 2025
process-reward literature (AgentPRM and successors) scores each step on whether it makes
progress toward the goal given the history so far.

### Design

New module `src/llm_evalgate/agentic/dimensions/step_progress.py`:

```python
class StepProgressDimension(TraceDimension):
    def __init__(
        self,
        judge: JudgeDimension,            # injected, same pattern as TrajectoryCoherence
        *,
        aggregate: str = "mean",          # "mean" | "min" (min = weakest-link gating)
        inefficiency_penalty: float = 0.0,  # per repeated tool call, same key as
                                            # StepEfficiencyDimension (name + sorted args)
        threshold: float = 0.7,
        name: str = "step_progress",
    ) -> None: ...
```

**Per-step judging:** for step *k*, render a prefix view — goal, steps 1..k-1
(serialized like `AgentTrace.to_text()`), then step *k* highlighted — and ask the
injected judge to grade ONE thing: *does this step make meaningful progress toward the
goal given everything before it?* (rubric passed to the judge; a sensible default
rubric ships in the module). One judge call per step; cost documented (n_steps calls).

**Aggregation:**

```
base  = mean(step_scores)            # or min(step_scores)
score = max(0.0, base - inefficiency_penalty * repeated_calls)
```

**Detail localizes failure** — this is the entire value over TrajectoryCoherence:

```
steps=5 mean=0.72: [0.9, 0.9, 0.2, 0.8, 0.8]; weakest step 3 (judge: "re-queried the
same API with identical args; no new information")
```

A `step_scores(trace) -> list[StepScore]` public method exposes the raw per-step
results (`StepScore(index, score, reason)`) for programmatic use (e.g. feeding a
fine-tuning signal or a dashboard).

**Multi-turn support (minimal, shared with T3.8):** `AgentStep` gains
`user_message: str | None = None` — a new user input arriving mid-trace.
`to_text()` renders it (`User: ...`); `from_dict` reads it. This is deliberately the
smallest possible multi-turn representation; full conversation modeling is out of scope.

### Acceptance criteria
- [ ] Fake judge returning scripted scores → correct mean/min aggregation; penalty math
      matches StepEfficiencyDimension's repeat detection on a trace with a repeated call.
- [ ] Prefix rendering for step k contains steps 1..k-1 and NOT steps k+1.. (no
      look-ahead — test asserts this explicitly).
- [ ] Empty-steps trace → score 0.0 with reason (no steps = no progress).
- [ ] `user_message` round-trips through `from_dict`/`to_text`; absent → rendering
      unchanged (existing to_text tests stay green).
- [ ] Composes in `AgentEvalHarness`.

### Files
- NEW `src/llm_evalgate/agentic/dimensions/step_progress.py`
- EDIT `src/llm_evalgate/agentic/trace.py` (user_message), agentic exports, root exports
- NEW `tests/test_step_progress.py`; EDIT `tests/test_agentic.py`
- EDIT `README.md` agentic table; EDIT `examples/agentic_eval.py`

---

## T3.8 AgentTrace.from_otel — import real traces

### Problem
Every `AgentTrace` today is hand-built. Real agent runs already emit traces — LangSmith,
MLflow, and anything OpenTelemetry-instrumented exports spans following the OTel GenAI
semantic conventions. An importer turns "score your agent" from a data-entry exercise
into one line against data people already have. (Course ch7 treats trace
instrumentation as table stakes; this is the bridge from their traces to our evals.)

### Design

New module `src/llm_evalgate/agentic/otel.py`. **No OTel SDK dependency** — input is
plain exported span dicts (the JSON shape of an OTLP/console exporter), keeping the
zero-dep core.

```python
@classmethod-style function (exposed as AgentTrace.from_otel via thin wrapper)
def trace_from_otel_spans(
    spans: list[dict],
    *,
    goal: str | None = None,    # override; else best-effort from root span input
) -> AgentTrace: ...
```

**Mapping (GenAI semconv, tolerant best-effort):**

| OTel | AgentTrace |
|---|---|
| span with `gen_ai.operation.name == "execute_tool"` | `ToolCall` |
| `gen_ai.tool.name` | `ToolCall.name` |
| `gen_ai.tool.call.arguments` (JSON string or dict) | `ToolCall.args` |
| `gen_ai.tool.call.result` / span events | `ToolCall.result` |
| span `status.code == ERROR` (+ message) | `ToolCall.error` |
| span with `gen_ai.operation.name` in `{"chat","invoke_agent","generate_content"}` | step boundary; output text → `thought`/`final_answer` |
| root span input attribute / first user message | `goal` (when not overridden) |

Rules:
- Order by `start_time_unix_nano` (fall back to list order when absent).
- Each model-call span starts a new `AgentStep`; tool spans attach to the current step;
  orphan tool spans (no preceding model span) get their own step.
- Last model span's output text → `final_answer`.
- Unknown spans are skipped, **counted**, and reported: the function returns the trace,
  and a companion `OtelImportReport` (spans_total, spans_mapped, spans_skipped,
  warnings) is available via `trace_from_otel_spans(..., return_report=True)`.
- Missing goal and no override → `ValueError` with a clear message (goal is required
  by every dimension).

**Stability caveat (document prominently):** the GenAI semconv is still incubating;
attribute names are pinned to the 2026 spec snapshot and the module docstring says so.
Tests run against checked-in fixture files, so upstream renames break loudly in one
place.

### Acceptance criteria
- [ ] Fixture: realistic OTLP-JSON export of a 2-step tool-using agent run →
      AgentTrace with correct goal, 2 steps, tool names/args/results, final_answer.
- [ ] Error-status tool span → `ToolCall.error` populated.
- [ ] JSON-string arguments parsed; malformed argument JSON → args `{}` + warning in
      report (never raises).
- [ ] Unknown spans skipped and counted; report numbers add up.
- [ ] No goal anywhere → ValueError; `goal=` override wins.
- [ ] Imported trace runs through `AgentEvalHarness` end-to-end in a test.

### Files
- NEW `src/llm_evalgate/agentic/otel.py`; EDIT `agentic/trace.py` (from_otel wrapper)
- NEW `tests/test_otel.py` + `tests/fixtures/otel_agent_run.json`
- EDIT exports, `README.md` (one-liner: "score your existing LangSmith/OTel traces")

---

## T3.9 Stratified golden sets — per-stratum metrics + a bigger bundled set

### Problem
Course discipline: golden sets are stratified (30 production / 30 edge / 20 hard),
metrics are tracked per stratum (a new failing category hides inside a good average),
and sets are refreshed. `BenchSample.meta` exists but nothing uses it; the bundled set
is 24 samples with no strata — which Tier 1's power warning will correctly call too
small for any tight gate.

### Design

**Per-stratum metrics** — `BenchmarkRunner.run()` gains `stratify_by: str | None = None`
(a `meta` key, conventional value `"stratum"`):

```python
result = BenchmarkRunner(harness).run(samples, stratify_by="stratum")
result.strata  # dict[str, BenchmarkResult] — full result per stratum, CIs included
print(result.table())
# n=60  accuracy 0.883 [0.800, 0.950] ...
# --- stratum=policy (n=15)    accuracy 1.000 [1.000, 1.000] ...
# --- stratum=semantic (n=15)  accuracy 0.667 [0.400, 0.867] ...
```

- Samples missing the key go to stratum `"(none)"`.
- Tiny strata get the honest treatment: n < 10 → per-stratum power warning in table.
- `RegressionGate.check()` (T1.2) gains `stratify_by` too: gate the overall metrics as
  before, but **report** per-stratum deltas so "fine on average, collapsed on one
  category" (the ch12 distribution-drift failure) is visible in the gate output.
  Per-stratum deltas warn, never fail, by default (`fail_on_stratum=False`) — small
  strata are too noisy to block on.

**Bundled golden set v2** — grow `datasets/golden_eval.jsonl` 24 → ~60, every sample
tagged `meta.stratum` with one of:
- `policy` (blocklist-type violations), `format` (schema/structure), `readability`,
- `semantic` (fluent-but-wrong — the regressions deterministic gates miss),
- `edge` (empty-ish, very long, unicode, near-threshold cases).

Composition mirrors the course shape: ~half clean passes, ~half failures spread across
strata. All synthetic, hand-written (no licensing issues), each with a one-line
`meta.why` so the set is self-documenting. `load_golden()` unchanged (meta already
supported). README benchmark numbers regenerated.

**Refresh discipline** — not code: add a short "maintaining your golden set" README
section (stratify, track per-stratum, refresh quarterly, version the file like code).
The library expresses the mechanics; the cadence is the user's job.

### Acceptance criteria
- [ ] `stratify_by` populates `result.strata`; per-stratum metrics match independently
      computed values; missing-key samples land in `(none)`.
- [ ] Table renders overall + per-stratum blocks; n<10 warning fires.
- [ ] Gate report shows per-stratum deltas; a constructed one-stratum collapse is
      visible in the report while the overall gate still passes.
- [ ] New golden set: 60 lines, all valid JSONL, every sample has stratum + why;
      README example output regenerated and committed in the same change.
- [ ] `load_golden()` and all existing call sites unchanged.

### Files
- EDIT `src/llm_evalgate/bench/runner.py`, `bench/gate.py`
- REPLACE `src/llm_evalgate/bench/datasets/golden_eval.jsonl` (v2)
- EDIT `tests/test_bench.py` (+ stratification tests), `examples/benchmark.py`,
  `README.md`

---

## Tier 3 open questions

1. **Sequencing:** recommended order T3.9 → T3.7 → T3.8 (T3.9 completes the Tier 1
   gate story; T3.8 is the most speculative because the semconv is still moving).
2. **`min` vs `mean` default aggregate for StepProgress?** Spec says mean; `min` is the
   stricter weakest-link gate. Mean matches the PRM literature's trajectory scoring.
3. **OTel scope:** import only (spec) vs also *exporting* AgentTrace as OTel spans.
   Export is deferred — it drags in instrumentation concerns the library deliberately
   avoids.
4. **Golden set authorship:** 60 hand-written samples is a half-day content task. Worth
   it for the README credibility, but it can be trimmed to 40 if time-boxed.
