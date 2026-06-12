# Tier 1 Spec — Statistical Rigor + A Real Gate

Status: DRAFT — awaiting Lester's review. No code written.
Target version: 0.4.0
Theme: the library is named *evalgate* but today it reports point estimates and has no
baseline-diff gate. Tier 1 makes every number honest (confidence intervals, power
warnings) and adds the missing gate. Everything here is offline, deterministic with a
seed, and pure-stdlib.

Sources: interviewstack llm-orchestration ch7 (eval-as-CI, 2pp gate), ch12 eval-gap case
(80 examples ≈ ±4.8pp CI); Anthropic "Adding Error Bars to Evals" (Miller, 2024);
"Don't Use CLT in LLM Evals With Fewer Than a Few Hundred" (arXiv 2503.01747 — verified).

---

## T1.1 Bootstrap confidence intervals + power helpers

### Problem
`BenchmarkResult.metrics` (src/llm_evalgate/bench/runner.py:19) are point estimates.
On the bundled 24-sample golden set, `regression_catch_rate 0.667` is 8/12 — the
binomial 95% CI is roughly [0.39, 0.88]. Nothing tells the user that.

### Design

New module `src/llm_evalgate/bench/stats.py`:

```python
@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float
    n_resamples: int

def bootstrap_ci(
    metric_fn: Callable[[list[bool], list[bool]], float],
    predicted: list[bool],
    labels: list[bool],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int | None = 0,
) -> ConfidenceInterval: ...
```

- Percentile bootstrap: resample indices with replacement using `random.Random(seed)`,
  recompute `metric_fn` per resample, return the (alpha/2, 1-alpha/2) percentiles.
- `seed=0` default → deterministic by default (on-brand); `seed=None` → entropy.
- Degenerate resamples (e.g. zero-denominator precision returning 0.0 per the existing
  convention in metrics.py) simply feed the distribution; no special-casing.
- Pure stdlib (`random`, `statistics`). **No numpy dependency** — keep the dep list at
  `textstat` only.

Power helpers, same module (normal-approximation, documented as approximate):

```python
def min_detectable_effect(n: int, *, baseline: float = 0.8,
                          alpha: float = 0.05, power: float = 0.8) -> float:
    # ≈ (1.96 + 0.84) * sqrt(p*(1-p) / n) for a proportion metric

def required_sample_size(effect: float, *, baseline: float = 0.8,
                         alpha: float = 0.05, power: float = 0.8) -> int:
    # inverse of the above, ceil'd
```

### Integration with BenchmarkRunner

`BenchmarkRunner.run()` gains `ci: bool = True`, `n_resamples: int = 2000`,
`seed: int | None = 0`. `BenchmarkResult` gains
`intervals: dict[str, ConfidenceInterval] | None`.

`BenchmarkResult.table()` renders intervals when present:

```
n=24
accuracy               0.833  [0.667, 0.958]
...
regression_catch_rate  0.667  [0.417, 0.917]
```

**Recommendation: CIs ON by default.** The whole thesis of the library is honest
measurement; pre-1.0 a table-format change is acceptable. Existing tests that assert
on `table()` output get updated. (Flag for review — if you'd rather not break the
format, flip the default to `ci=False`.)

### Acceptance criteria
- [ ] `bootstrap_ci` is deterministic for a fixed seed; two runs produce identical CIs.
- [ ] CI for a metric on a hand-computable tiny set brackets the point estimate.
- [ ] `min_detectable_effect(24)` returns ≈ 0.22–0.24 at baseline 0.5 (i.e. it correctly
      says a 24-sample set cannot see 2pp).
- [ ] `required_sample_size(0.02, baseline=0.9)` returns a value in the ~1,500–2,000
      range (sanity-checked against a standard power calculator).
- [ ] `table()` with intervals stays aligned; without (`ci=False`) is byte-identical to
      today's output.
- [ ] No new runtime dependencies; suite still runs fully offline.

### Files
- NEW `src/llm_evalgate/bench/stats.py`
- EDIT `src/llm_evalgate/bench/runner.py` (run signature, BenchmarkResult fields, table)
- EDIT `src/llm_evalgate/bench/__init__.py`, `src/llm_evalgate/__init__.py` (exports)
- NEW `tests/test_stats.py`; EDIT `tests/test_bench.py`
- EDIT `README.md` (benchmark section shows intervals + a "can your dataset even
  detect your threshold?" callout)

---

## T1.2 RegressionGate — baseline diff with significance

### Problem
There is no gate. The course-standard CI discipline is "block merge on >2pp regression
vs baseline," and the library has no way to express it: no baseline persistence, no
delta, no decision.

### Design

New module `src/llm_evalgate/bench/gate.py`.

**Baseline persistence** (on `BenchmarkResult`):

```python
result.save(path)                    # JSON: predicted, labels, metrics, n,
                                     # dataset_fingerprint, created_at, lib version
BenchmarkResult.load(path)           # classmethod
```

`dataset_fingerprint` = sha256 over the ordered sample texts, computed in
`BenchmarkRunner.run()` and stored on the result. It exists to prevent the classic
silent error: diffing two runs that weren't on the same dataset.

**The gate:**

```python
gate = RegressionGate(
    metrics=("accuracy", "regression_catch_rate"),  # or "all"
    threshold=0.02,                 # course-standard 2pp
    require_significance=True,      # delta CI must exclude 0 to FAIL
    n_resamples=2000,
    seed=0,
)
report = gate.check(current, baseline)   # -> GateReport
```

Decision rule per metric (higher-is-better assumed; all six current metrics qualify):
- `delta = current - baseline`
- **FAIL** when `delta < -threshold` AND (if `require_significance`) the paired-bootstrap
  95% CI on delta lies entirely below 0.
- `delta < -threshold` but CI overlaps 0 → **WARN** (reported, does not fail the gate):
  "regression observed but not separable from eval noise at n=24."

Paired bootstrap: requires matching fingerprints; resample shared indices, compute the
metric for baseline-predictions and current-predictions on the same resample, take the
delta distribution. If fingerprints differ: raise `ValueError` by default;
`allow_unpaired=True` falls back to independent resampling with a prominent warning in
the report.

**GateReport:**

```python
@dataclass(frozen=True)
class GateReport:
    passed: bool
    rows: list[GateRow]        # metric, baseline, current, delta, ci, verdict
    regressed_samples: list[int]  # indices where baseline passed and current failed
    warnings: list[str]        # unpaired fallback, power warning, etc.
    def table(self) -> str: ...
```

`regressed_samples` carries indices + `meta` of samples that flipped pass→fail, so a CI
log names the failing examples (course pattern: "block + comment with failing examples").

**Power warning (ties T1.1 to T1.2):** at construction-check time, if
`min_detectable_effect(n)` > `threshold`, append a warning:
`"n=24: minimum detectable effect ≈ 0.23; a 0.02 threshold cannot be distinguished
from noise — gate decisions below MDE rely on require_significance."`

**CLI (small, stdlib argparse):**

```
python -m llm_evalgate.gate current.json baseline.json --threshold 0.02
```

Prints `GateReport.table()`, exits 0/1. This is what makes it drop into any CI system
with two lines of YAML. (Flag for review: keep or cut the CLI.)

### Acceptance criteria
- [ ] save → load round-trips a `BenchmarkResult` losslessly.
- [ ] Identical runs → gate PASSES with all deltas 0.
- [ ] A constructed 5pp-worse run on n=200 synthetic data → FAIL, regressed sample
      indices correct.
- [ ] Same 5pp drop on n=24 with `require_significance=True` → WARN not FAIL, and the
      power warning fires.
- [ ] Mismatched fingerprints raise; `allow_unpaired=True` proceeds with warning.
- [ ] CLI exits 1 on FAIL, 0 on PASS; output readable.

### Files
- NEW `src/llm_evalgate/bench/gate.py`, NEW `src/llm_evalgate/gate/__main__.py`
  (or `bench/__main__.py` — implementer's choice, document it)
- EDIT `src/llm_evalgate/bench/runner.py` (fingerprint, save/load)
- EDIT exports, NEW `tests/test_gate.py`, EDIT `README.md` (new top-level section —
  this is the headline feature of 0.4.0)

---

## T1.3 Judge bias correction (Rogan–Gladen)

### Problem
`calibrate_judge` (src/llm_evalgate/judge/calibration.py:46) reports agreement
(accuracy/kappa) but never *uses* the calibration to correct downstream judge verdicts.
A judge with 90% sensitivity / 80% specificity measuring a true 70% pass rate reports
~69% — close — but at 95%/60% it reports ~78.5%, an 8.5pp distortion. The fix is the
Rogan–Gladen estimator (1978; the standard correction for imperfect classifiers —
recently revived for LLM judges in "How to Correctly Report LLM-as-a-Judge
Evaluations," arXiv 2511.21140).

### Design

**Extend `CalibrationReport`** (backward-compatible, new fields default `None`):
- `sensitivity: float | None` — P(judge pass | human pass) = TP/(TP+FN)
- `specificity: float | None` — P(judge fail | human fail) = TN/(TN+FP)
- `confusion: dict[str, int] | None` — the raw counts (needed for CI resampling)

Computed in `calibrate_judge` whenever `human_label`s are present (reuses
`confusion_counts` from bench.metrics). `table()` gains the two rows.

**New module `src/llm_evalgate/judge/correction.py`:**

```python
def rogan_gladen(observed_rate: float, sensitivity: float, specificity: float) -> float:
    # (observed + specificity - 1) / (sensitivity + specificity - 1), clamped to [0,1]
    # raises ValueError when sensitivity + specificity <= 1 (judge no better than chance)

@dataclass(frozen=True)
class CorrectedRate:
    observed: float
    corrected: float
    ci_low: float
    ci_high: float
    n_eval: int
    n_calibration: int

def corrected_pass_rate(
    judge_labels: list[bool],
    calibration: CalibrationReport,   # must carry confusion counts
    *,
    n_resamples: int = 2000,
    seed: int | None = 0,
) -> CorrectedRate: ...
```

CI propagates **both** uncertainty sources, which is the entire point: per bootstrap
resample, (a) resample the eval verdicts → `p_obs*`, (b) resample the calibration
confusion matrix rows → `sens*`, `spec*`, (c) compute `rogan_gladen(p_obs*, sens*,
spec*)`. Percentile CI over the resamples. Resamples where `sens* + spec* <= 1` are
dropped; if more than 20% drop, the result carries a warning (judge is too close to
chance for the correction to be meaningful) — surfaced via a `warning: str | None`
field on `CorrectedRate`.

Reuses `bootstrap_ci` machinery/seeding conventions from T1.1 where practical.

### Acceptance criteria
- [ ] Worked example: sens=0.9, spec=0.8, observed=0.69 → corrected ≈ 0.70.
- [ ] Perfect judge (sens=spec=1.0) → corrected == observed, CI matches plain bootstrap.
- [ ] sens+spec <= 1 raises with a clear message.
- [ ] CI is wider when the calibration set is smaller (e.g. n_cal=20 vs 200), holding
      the eval set fixed — this demonstrates two-source propagation works.
- [ ] `calibrate_judge` output unchanged for existing callers (new fields additive);
      existing tests pass untouched.

### Files
- EDIT `src/llm_evalgate/judge/calibration.py`
- NEW `src/llm_evalgate/judge/correction.py`
- EDIT exports, NEW `tests/test_correction.py`, EDIT `tests/test_calibration.py`
- EDIT `README.md` (judge-reliability section gets a "correct the verdict, don't just
  measure the judge" subsection); EDIT `examples/judge_calibration.py`

---

## Tier 1 open questions (answer before /ship)

1. **CIs default ON in `BenchmarkRunner.run()`?** Recommended yes (table format changes).
2. **Keep the gate CLI** (`python -m llm_evalgate.gate`)? Recommended yes — it's the
   CI-integration story — but it's severable.
3. **Golden-set growth now or with Tier 3?** The power warning will (correctly) say the
   bundled 24-sample set is too small for a 2pp gate. Honest options: grow it in T3.9,
   or have the README demo use a larger threshold. Recommended: leave 24 for now, let
   the warning itself be the README teaching moment.
