# llm-evalgate

Deterministic eval gates and reliability primitives for LLM pipelines.

[![CI](https://github.com/LesterALeong/llm-evalgate/actions/workflows/ci.yml/badge.svg)](https://github.com/LesterALeong/llm-evalgate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-evalgate)](https://pypi.org/project/llm-evalgate/)
[![Python](https://img.shields.io/pypi/pyversions/llm-evalgate)](https://pypi.org/project/llm-evalgate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

Most LLM eval tooling is either heavy enterprise SaaS or a pile of one-off scripts. `llm-evalgate` is a small, composable library that covers the full eval surface and runs in CI.

It gives you four things:

- **Eval gates**: code-only quality dimensions that run the same way every time. Drop them into any pipeline, run them in CI, get a pass/fail with a reason. No tokens burned, no model drift.
- **LLM-as-judge and jury**: when a check needs semantic judgment a regex cannot give, grade with a model, or a panel of models, and read the agreement back. Same `Dimension` interface, so judges compose with gates in one harness.
- **Agentic evals**: score agent traces, not just final text. Tool selection, argument validity, step efficiency, goal completion, and judge-backed trajectory coherence for long-horizon reasoning.
- **Benchmarking**: measure your eval against labeled data. Accuracy, precision/recall, Cohen's kappa, and regression-catch-rate, so you know whether your gate actually catches the failures that matter.

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
| `StepEfficiencyDimension` | Did it finish within a step budget, without looping on repeated calls? |
| `GoalCompletionDimension` | Did the final answer satisfy a checker or required content? |
| `TrajectoryCoherenceDimension` | Judge-backed: does the reasoning path hang together over long horizons? |

`TrajectoryCoherenceDimension` wraps a `JudgeDimension`, so long-horizon reasoning quality is scored by a model while the structural checks above stay deterministic.

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

Run against the bundled 24-sample golden set (see [`examples/benchmark.py`](examples/benchmark.py)), the deterministic harness alone scores:

```
n=24
accuracy               0.833
precision              0.750
recall                 1.000
f1                     0.857
cohen_kappa            0.667
regression_catch_rate  0.667
```

It catches every formatting and policy violation but misses 4 of 12 regressions, because those failures are semantic (a fluent, well-formatted answer that is simply wrong). Add an LLM judge to the same harness and the catch rate closes to `1.000`. That is the whole thesis: deterministic gates are necessary but not sufficient, and the benchmark is how you prove where the line is.

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

## License

MIT
