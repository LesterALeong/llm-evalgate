# llm-evalgate

**Eval gates with error bars.** Deterministic gates, a calibrated LLM-as-judge, agentic-trace evals, and a statistically honest regression gate for LLM pipelines — the whole eval surface, in CI, with a confidence interval on every number.

[![CI](https://github.com/LesterALeong/llm-evalgate/actions/workflows/ci.yml/badge.svg)](https://github.com/LesterALeong/llm-evalgate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-evalgate)](https://pypi.org/project/llm-evalgate/)
[![Python](https://img.shields.io/pypi/pyversions/llm-evalgate)](https://pypi.org/project/llm-evalgate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

Most LLM eval tooling hands you a score. Almost none of it tells you whether that score is real. `llm-evalgate` treats your eval like a measurement instrument: every benchmark metric ships with a bootstrap confidence interval, the regression gate only fails CI on a drop that clears the noise floor, and an LLM judge is something you calibrate and de-bias before you trust it — not a vibe with an API bill. It's a small, composable library that covers the full eval surface and runs in CI.

This rigor isn't academic. `llm-evalgate` was extracted and hardened inside a production multi-agent system where a wrong answer doesn't fail a test, it costs real money — so "looks fine" was never good enough. The deterministic gates, the fail-closed judges, and the statistical regression check are the patterns that survived that environment. (Only the reliability engineering is open-sourced here; no strategy, signal, or position data lives in this repo.)

It gives you four things:

- **Eval gates**: code-only quality dimensions that run the same way every time. Drop them into any pipeline, run them in CI, get a pass/fail with a reason. No tokens burned, no model drift.
- **LLM-as-judge and jury**: when a check needs semantic judgment a regex cannot give, grade with a model, or a panel of models, and read the agreement back. Same `Dimension` interface, so judges compose with gates in one harness.
- **Agentic evals**: score agent traces, not just final text. Tool selection, argument validity, step efficiency, goal completion, and judge-backed trajectory coherence for long-horizon reasoning.
- **Benchmarking and a real gate**: measure your eval against labeled data — accuracy, precision/recall, Cohen's kappa, and regression-catch-rate, each with a bootstrap confidence interval — then fail CI on a statistically significant regression against a saved baseline. Power helpers tell you when your dataset is too small to trust.

Plus **reliability primitives** (retry with backoff, model fallback chains, circuit breaker) for pipelines that hold up in production.

Design rule: deterministic gates are the cheap, CI-friendly backbone; reach for a judge only where semantic nuance demands it; benchmark both so the eval is trustworthy rather than assumed.

## Install

```bash
pip install llm-evalgate          # core: gates, agentic evals, benchmarking
pip install "llm-evalgate[judge]" # adds the Anthropic adapter for live LLM-as-judge
```

The judge and jury work with any callable that maps a prompt string to a response string, so the core package needs no model SDK and the test suite runs fully offline. The `[judge]` extra only adds a convenience adapter for live calls.

## Quickstart

### Eval gates

```python
from llm_evalgate import EvalHarness
from llm_evalgate.eval.dimensions import BlocklistDimension, ReadabilityDimension, SchemaComplianceDimension

harness = EvalHarness([
    BlocklistDimension(terms=["confidential", "internal use only"]),
    ReadabilityDimension(threshold=0.3),
    SchemaComplianceDimension(required_fields=["title:", "summary:"]),
])

report = harness.run(llm_output)

if not report.passed:
    print(report)
    # EvalReport: FAIL
    #   FAIL [blocklist] score=0.000 - prohibited terms found: ['confidential']
    #   PASS [readability] score=0.612 - Flesch ease=61.2, FK grade=8.4
    #   PASS [schema_compliance] score=1.000 - all 2 required fields present
```

### Custom dimension

```python
from llm_evalgate import Dimension

class JsonDimension(Dimension):
    def evaluate(self, text: str) -> tuple[float, str]:
        import json
        try:
            json.loads(text)
            return 1.0, "valid JSON"
        except json.JSONDecodeError as e:
            return 0.0, f"invalid JSON: {e}"

harness = EvalHarness([JsonDimension(threshold=1.0)])
report = harness.run('{"key": "value"}')
assert report.passed
```

### Retry

```python
from llm_evalgate.reliable import retry

@retry(max_attempts=3, backoff=2.0)
def call_llm(prompt: str) -> str:
    return client.messages.create(...)
```

### Fallback chain

```python
from llm_evalgate.reliable import with_fallback, with_fallback_chain

# two-model fallback
result = with_fallback(
    primary=lambda: call_model("claude-opus-4-8", prompt),
    fallback=lambda: call_model("claude-sonnet-4-6", prompt),
)

# ordered chain - first success wins
result = with_fallback_chain([
    lambda: call_model("claude-opus-4-8", prompt),
    lambda: call_model("claude-sonnet-4-6", prompt),
    lambda: call_model("claude-haiku-4-5", prompt),
])
```

### Circuit breaker

```python
from llm_evalgate.reliable import CircuitBreaker, CircuitOpenError

breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)

try:
    with breaker:
        result = call_llm(prompt)
except CircuitOpenError:
    result = cached_response  # serve from cache while circuit is open
```

## Built-in dimensions

| Dimension | What it checks | Default threshold |
|---|---|---|
| `BlocklistDimension` | No prohibited terms in output | 1.0 (zero tolerance) |
| `ReadabilityDimension` | Flesch Reading Ease score | 0.3 (college-level prose) |
| `SchemaComplianceDimension` | Required fields are present | 1.0 (all fields) |
| `FactualGroundingDimension` | Numeric claims traceable to evidence | 0.85 |

All dimensions follow the same interface: `evaluate(text) -> (score, detail)`. Writing a new one is ten lines.

## LLM-as-judge and jury

Some quality dimensions cannot be checked with code: tone, helpfulness, faithfulness to a brief. For those, grade with a model. `JudgeDimension` is a regular `Dimension`, so it drops into the same harness next to your deterministic gates.

```python
from llm_evalgate import JudgeDimension

# `complete` is any callable mapping a prompt to a response string.
# Inject your own, or use the optional Anthropic adapter (see below).
judge = JudgeDimension(
    complete=my_model_call,
    rubric="Is the answer factually correct and on-topic for the question?",
    scale=5,            # model grades 1..5; normalized to [0, 1]
    threshold=0.6,
)

report = judge.run("The capital of France is Paris.")
# PASS [judge] score=1.000 - clear and correct
```

Want a panel instead of a single grader? `JuryDimension` runs several judges and aggregates the verdicts, and reports how much they agreed (a tight spread means the eval is stable; a wide one means the dimension is ambiguous).

```python
from llm_evalgate import JudgeDimension, JuryDimension

jury = JuryDimension(
    judges=[
        JudgeDimension(complete=opus_call, rubric=rubric, name="opus"),
        JudgeDimension(complete=sonnet_call, rubric=rubric, name="sonnet"),
        JudgeDimension(complete=haiku_call, rubric=rubric, name="haiku"),
    ],
    aggregate="mean",   # or "median", "majority"
)
report = jury.run(answer)
```

For live grading without writing the adapter yourself, install `llm-evalgate[judge]` and use `anthropic_judge`:

```python
from llm_evalgate.judge import anthropic_judge
from llm_evalgate import JudgeDimension

judge = JudgeDimension(complete=anthropic_judge("claude-sonnet-4-6"), rubric=rubric)
```

The judge never raises into your pipeline: a model error or an unparseable response scores `0.0` with the reason recorded, so a flaky grader fails closed instead of crashing the run.

## Judge reliability: measure and de-bias your judge

An LLM judge is itself a model, with all the variance and bias that implies. Trusting it blind is how a bad eval ships. These three tools measure and correct for that. See [`examples/judge_calibration.py`](examples/judge_calibration.py) for a runnable, offline demo.

### Uncertainty (self-consistency)

A judge at temperature > 0 returns a different score each call, and a single number hides that. `SelfConsistencyJudge` samples the judge N times and reports the spread and a 95% confidence interval, so you know whether a score of 0.62 is solid or a coin flip.

```python
from llm_evalgate import JudgeDimension, SelfConsistencyJudge

judge = JudgeDimension(complete=my_model_call, rubric="Is the answer faithful to the source?")
robust = SelfConsistencyJudge(judge, samples=5)

report = robust.run(answer)
# PASS [self_consistency] score=0.640 - mean=0.640; stdev=0.080; 95% CI=[0.570, 0.710]; n=5
dist = robust.distribution(answer)   # ScoreDistribution(mean, stdev, n, ci_low, ci_high, scores)
```

### Position bias (pairwise, order-swapped)

When a judge compares two answers, it tends to favor whichever is shown first. `PairwiseJudge` runs the comparison in both orders and only trusts a verdict that survives the swap; if the judge flips, the result is reported as a tie with `consistent=False`. `position_bias_rate` measures how often a judge flips across a set of pairs, so you can quantify the bias before you rely on the judge.

```python
from llm_evalgate import PairwiseJudge, position_bias_rate

pj = PairwiseJudge(complete=my_model_call, criteria="Which answer is more helpful?")
result = pj.compare(answer_a, answer_b)
# result.winner in {"A", "B", "tie"}; result.consistent is False when order flipped the call

bias = position_bias_rate(pj, [(a1, b1), (a2, b2), ...])   # 0.0 = unbiased, 1.0 = pure position bias
```

### Calibration (judge vs human)

The only real test of a judge is agreement with human labels. `calibrate_judge` scores a labeled set and reports the correlation (Pearson, Spearman) and error (MAE) against human scores, plus accuracy and Cohen's kappa against human pass/fail. `verbosity_bias` checks the failure mode where a judge just rewards length.

```python
from llm_evalgate import JudgeDimension, CalibrationSample, calibrate_judge, verbosity_bias

judge = JudgeDimension(complete=my_model_call, rubric="Grade the answer.")
report = calibrate_judge(judge, [
    CalibrationSample(text=a1, human_score=0.9, human_label=True),
    CalibrationSample(text=a2, human_score=0.2, human_label=False),
    # ...
])
print(report.table())   # pearson / spearman / mae / accuracy / cohen_kappa

length_corr = verbosity_bias(judge, [a1, a2, a3])   # high = the judge is rewarding length, not quality
```

A judge you have calibrated, sampled for uncertainty, and checked for position and verbosity bias is one you can defend in a design review. An uncalibrated judge is a vibe with an API bill.

## Agentic evaluation

Agents are not just their final answer, they are a trajectory of tool calls. `llm-evalgate` evaluates the whole trace.

```python
from llm_evalgate import (
    AgentTrace, AgentStep, ToolCall, AgentEvalHarness,
    ToolSelectionDimension, ToolArgValidityDimension,
    StepEfficiencyDimension, GoalCompletionDimension,
)

trace = AgentTrace(
    goal="Find the weather in Paris and convert it to Fahrenheit.",
    steps=[
        AgentStep(
            thought="Look up the weather.",
            tool_calls=[ToolCall(name="get_weather", args={"city": "Paris"}, result="18C")],
        ),
        AgentStep(
            thought="Convert to Fahrenheit.",
            tool_calls=[ToolCall(name="celsius_to_f", args={"c": 18}, result="64.4F")],
        ),
    ],
    final_answer="It is about 64.4F in Paris.",
)

harness = AgentEvalHarness([
    ToolSelectionDimension(expected_tools=["get_weather", "celsius_to_f"], mode="order"),
    ToolArgValidityDimension(validators={"celsius_to_f": lambda a: "c" in a}),
    StepEfficiencyDimension(max_steps=3),
    GoalCompletionDimension(required_substrings=["F"]),
])

report = harness.run(trace)
if not report.passed:
    print(report)
```

| Trace dimension | What it scores |
|---|---|
| `ToolSelectionDimension` | Did the agent call the expected tools (as a subset, exact set, or in order)? |
| `ToolArgValidityDimension` | Were tool arguments well-formed, and did any call error? |
| `ToolHallucinationDimension` | Did it call a tool that exists, with arguments the schema allows? |
| `StepEfficiencyDimension` | Did it finish within a step budget, without looping on repeated calls? |
| `GoalCompletionDimension` | Did the final answer satisfy a checker or required content? |
| `StepProgressDimension` | Judge-backed: does each step make progress, and which step is the weakest? |
| `TrajectoryCoherenceDimension` | Judge-backed: does the reasoning path hang together over long horizons? |

`TrajectoryCoherenceDimension` and `StepProgressDimension` wrap a `JudgeDimension`, so reasoning quality is scored by a model while the structural checks above stay deterministic. `ToolHallucinationDimension` takes a tool registry and catches phantom tools and bad arguments — build the registry straight from your existing tool definitions with `ToolSchema.from_json_schema(tool_def)`. `StepProgressDimension` scores every step against the history before it and names the weakest one, so a failure is localized rather than reported as a single trajectory number.

You do not have to hand-build traces. If your agent is instrumented with the OpenTelemetry GenAI semantic conventions (LangSmith, MLflow, and most OTel exporters emit them), import the spans directly:

```python
from llm_evalgate import AgentTrace

trace = AgentTrace.from_otel(exported_spans)   # plain span dicts; no OTel SDK needed
report = harness.run(trace)
```

## Benchmarking: is your eval any good?

An eval you have not measured is a guess. Point a `BenchmarkRunner` at a labeled dataset and it tells you how well your gate matches human judgment, including the metric that matters most in production: the fraction of real regressions it actually catches.

```python
from llm_evalgate import BenchmarkRunner, EvalHarness, load_golden
from llm_evalgate.eval.dimensions import (
    BlocklistDimension, SchemaComplianceDimension, ReadabilityDimension,
)

harness = EvalHarness([
    BlocklistDimension(terms=["confidential", "internal use only", "[REDACTED]"]),
    SchemaComplianceDimension(required_fields=["title:", "summary:"]),
    ReadabilityDimension(threshold=0.2),
])

result = BenchmarkRunner(harness).run(load_golden())
print(result.table())
```

Run against the bundled 60-sample stratified golden set (see [`examples/benchmark.py`](examples/benchmark.py)), the deterministic harness alone scores (95% bootstrap CIs shown):

```
n=60
accuracy               0.850  [0.750, 0.933]
precision              0.775  [0.641, 0.900]
recall                 1.000  [1.000, 1.000]
f1                     0.873  [0.781, 0.947]
cohen_kappa            0.697  [0.516, 0.864]
regression_catch_rate  0.690  [0.500, 0.857]
```

It catches every formatting, policy, and readability violation but misses all 9 semantic regressions, because those failures are semantic (a fluent, well-formatted answer that is simply wrong). Add an LLM judge to the same harness and the catch rate closes. That is the whole thesis: deterministic gates are necessary but not sufficient, and the benchmark is how you prove where the line is.

Those `[low, high]` columns are 95% bootstrap confidence intervals, on by default. They are not decoration: at `n=60` the minimum detectable effect is about `0.145`, so a 2-point regression is inside the noise. The library tells you that rather than letting you pretend a point estimate is precise — see [confidence intervals and the regression gate](#confidence-intervals-power-and-the-regression-gate) below.

## Confidence intervals, power, and the regression gate

A benchmark number with no error bar is a guess with a decimal point. Every `BenchmarkRunner` metric ships with a percentile bootstrap CI, and two power helpers tell you whether your dataset can even resolve the regression you want to gate on.

```python
from llm_evalgate.bench import min_detectable_effect, required_sample_size

min_detectable_effect(60)        # ~0.145 — a 60-sample set can't see a 2pp drop
required_sample_size(0.02)       # ~3100  — that's what a 2pp gate actually needs
```

The **regression gate** is the piece that makes the package name true. Save a known-good run as a baseline, then fail the build when a metric regresses past a threshold — but only when the drop clears the noise floor, computed with a paired bootstrap on the same resampled rows.

```python
from llm_evalgate import BenchmarkRunner, RegressionGate

baseline = BenchmarkRunner(harness).run(golden)
baseline.save("baseline.json")               # commit this

# ... later, on a PR ...
current = BenchmarkRunner(new_harness).run(golden)
gate = RegressionGate(metrics="all", threshold=0.02, require_significance=True)
report = gate.check(current, BenchmarkResult.load("baseline.json"))
if not report.passed:
    raise SystemExit(report.table())
```

A dataset fingerprint guards against the classic mistake of diffing two runs that were not on the same data. A drop past the threshold whose delta CI still includes zero is reported as a `WARN`, not a hard `FAIL`, so a small eval set does not block merges on noise — and a power warning fires when the dataset is too small for the threshold. Run it in CI with the bundled entry point:

```bash
python -m llm_evalgate.gate current.json baseline.json --threshold 0.02   # exit 1 on regression
```

## Correct the verdict, not just the judge

Calibration tells you the judge's sensitivity and specificity against human labels. The Rogan–Gladen estimator then *uses* them: it converts the judge's raw pass rate into a bias-corrected estimate of the true pass rate, with a CI that propagates uncertainty from both the eval set and the finite calibration set.

```python
from llm_evalgate import calibrate_judge, corrected_pass_rate

cal = calibrate_judge(judge, labeled_samples)   # now carries sensitivity/specificity
judge_labels = [judge.run(t).passed for t in production_outputs]
rate = corrected_pass_rate(judge_labels, cal)
# observed=0.690; corrected=0.704 [0.631, 0.770] (n_eval=200, n_calibration=120)
```

A judge at 95% sensitivity and 60% specificity reporting a 78% pass rate is really seeing about 64% — an 8-point bias an uncorrected number hides.

## Claim-level faithfulness

`FactualGroundingDimension` traces numbers; `ClaimFaithfulnessDimension` traces every claim. It decomposes an answer into atomic claims and grades each as supported, unsupported, or contradicted against the evidence — the canonical RAG faithfulness check, and a regular `Dimension` so it drops into the same harness.

```python
from llm_evalgate import ClaimFaithfulnessDimension

faith = ClaimFaithfulnessDimension(complete=my_model_call, evidence=retrieved_chunks, threshold=0.9)
report = faith.run(answer)
# claims=7: 5 supported, 1 unsupported, 1 contradicted; score=0.714
#   CONTRADICTED: "the contract allows early termination" (evidence says 90-day notice)
```

## Routing uncertain verdicts to a human

An unstable judge verdict should go to a person, not be trusted as pass/fail. `SelfConsistencyJudge`, `JuryDimension`, and `PairwiseJudge` can now flag a result as `needs_review` — when the score spread is too wide, the CI straddles the threshold, the jury disagrees, or a pairwise verdict flips on order swap. The flag is orthogonal to pass/fail; the report surfaces it so you decide what to route.

```python
from llm_evalgate import SelfConsistencyJudge

robust = SelfConsistencyJudge(judge, samples=5, max_stdev=0.15, review_margin=0.1)
result = robust.run(answer)
if result.needs_review:
    send_to_human_queue(answer)
```

## Deterministic gates first, judge when you need to

Reach for the cheapest check that works:

- **Deterministic gates** (`BlocklistDimension`, `SchemaComplianceDimension`, and your own) are pure functions: same input, same pass/fail, every run. No model calls, no network, no randomness. Run them on every commit in CI without burning tokens, and keep an audit trail that does not depend on a model that may drift.
- **Judges and juries** are the opt-in layer for the semantic checks code cannot express. They cost tokens and carry variance, so use them where they earn it, and benchmark them so the variance is known rather than assumed.

## Composing with a pipeline

```python
from llm_evalgate import EvalHarness
from llm_evalgate.eval.dimensions import BlocklistDimension, ReadabilityDimension
from llm_evalgate.reliable import retry, with_fallback

harness = EvalHarness([
    BlocklistDimension(terms=["[REDACTED]", "TODO"]),
    ReadabilityDimension(threshold=0.2),
])

@retry(max_attempts=3, backoff=2.0)
def generate(prompt: str) -> str:
    return with_fallback(
        primary=lambda: call_model("claude-opus-4-8", prompt),
        fallback=lambda: call_model("claude-sonnet-4-6", prompt),
    )

output = generate(prompt)
report = harness.run(output)
if not report.passed:
    raise ValueError(f"Output failed eval gate:\n{report}")
```

## Who built this

[Lester Leong](https://github.com/LesterALeong) — data scientist and quant. `llm-evalgate` is the eval and reliability layer extracted from a multi-agent system that trades real money, where a wrong answer has real downside rather than just a red test. Issues and PRs welcome.

## License

MIT
