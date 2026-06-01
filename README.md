# llm-evalgate

Deterministic eval gates and reliability primitives for LLM pipelines.

[![CI](https://github.com/LesterALeong/llm-evalgate/actions/workflows/ci.yml/badge.svg)](https://github.com/LesterALeong/llm-evalgate/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/llm-evalgate)](https://pypi.org/project/llm-evalgate/)
[![Python](https://img.shields.io/pypi/pyversions/llm-evalgate)](https://pypi.org/project/llm-evalgate/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

---

Most LLM eval tooling is either LLM-as-judge (non-deterministic, expensive, not CI-friendly) or a heavy enterprise suite. `llm-evalkit` is neither.

It gives you two things:

- **Eval gates**: code-only quality dimensions that run the same way every time. Drop them into any pipeline, run them in CI, get a pass/fail with a reason.
- **Reliability primitives**: retry with backoff, model fallback chains, and a circuit breaker. The building blocks for LLM pipelines that hold up in production.

## Install

```bash
pip install llm-evalgate
```

## Quickstart

### Eval gates

```python
from llm_evalkit import EvalHarness
from llm_evalkit.eval.dimensions import BlocklistDimension, ReadabilityDimension, SchemaComplianceDimension

harness = EvalHarness([
    BlocklistDimension(terms=["confidential", "internal use only"]),
    ReadabilityDimension(threshold=0.3),
    SchemaComplianceDimension(required_fields=["title:", "summary:"]),
])

report = harness.run(llm_output)

if not report.passed:
    print(report)
    # EvalReport: FAIL
    #   FAIL [blocklist] score=0.000 — prohibited terms found: ['confidential']
    #   PASS [readability] score=0.612 — Flesch ease=61.2, FK grade=8.4
    #   PASS [schema_compliance] score=1.000 — all 2 required fields present
```

### Custom dimension

```python
from llm_evalkit import Dimension

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
from llm_evalkit.reliable import retry

@retry(max_attempts=3, backoff=2.0)
def call_llm(prompt: str) -> str:
    return client.messages.create(...)
```

### Fallback chain

```python
from llm_evalkit.reliable import with_fallback, with_fallback_chain

# two-model fallback
result = with_fallback(
    primary=lambda: call_model("claude-opus-4-8", prompt),
    fallback=lambda: call_model("claude-sonnet-4-6", prompt),
)

# ordered chain — first success wins
result = with_fallback_chain([
    lambda: call_model("claude-opus-4-8", prompt),
    lambda: call_model("claude-sonnet-4-6", prompt),
    lambda: call_model("claude-haiku-4-5", prompt),
])
```

### Circuit breaker

```python
from llm_evalkit.reliable import CircuitBreaker, CircuitOpenError

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

## Why deterministic?

LLM-as-judge eval is useful for research. In production pipelines, you need:

- The same input to produce the same pass/fail result every run
- CI to catch regressions without burning tokens on every commit
- An audit trail that doesn't depend on a model that may drift

`llm-evalkit` eval dimensions are pure functions. No model calls, no network, no randomness.

## Composing with a pipeline

```python
from llm_evalkit import EvalHarness
from llm_evalkit.eval.dimensions import BlocklistDimension, ReadabilityDimension
from llm_evalkit.reliable import retry, with_fallback

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
