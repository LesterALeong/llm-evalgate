# llm-evalgate Improvement Program — Spec Review Gate

Source: interviewstack course mining (llm-engineering ch4/ch6/ch9, llm-orchestration
ch5/ch7/ch10/ch11/ch12) + 2025-26 eval research sweep, diffed against the v0.3.0
codebase on 2026-06-12.

**Status: specs written, NOT approved. No implementation until Lester reviews.**

## Specs

- [ ] Review [tier1-statistical-gate.md](specs/tier1-statistical-gate.md) — bootstrap
      CIs + power helpers, RegressionGate w/ baseline persistence + paired bootstrap +
      CLI, Rogan–Gladen judge bias correction. Target 0.4.0. **Headline release.**
- [ ] Review [tier2-coverage.md](specs/tier2-coverage.md) — ClaimFaithfulnessDimension
      (claim-level RAG groundedness), ToolHallucinationDimension (phantom tools/args),
      needs_review low-confidence routing. Ships with 0.4.0 or as 0.5.0.
- [ ] Review [tier3-extensions.md](specs/tier3-extensions.md) — StepProgressDimension
      (process-reward step scoring), AgentTrace.from_otel importer, stratified golden
      sets + 60-sample bundled set v2. Target 0.5.0, independently severable.

## Open questions needing Lester's call (collected from the three specs)

1. T1: bootstrap CIs ON by default in BenchmarkRunner (changes table format)? Rec: yes.
2. T1: keep the `python -m llm_evalgate.gate` CLI? Rec: yes.
3. T1/T3: bundled golden set stays at 24 until Tier 3 (power warning becomes the README
   teaching moment)? Rec: yes.
4. T2: claim verification one-call-per-claim (robust) vs batched (cheap)? Rec: per-claim.
5. T2: needs_review as a DimensionResult field (6-file touch) vs wrapper class? Rec: field.
6. T3: sequencing T3.9 → T3.7 → T3.8? OTel import-only (no export)? Rec: yes to both.

## After approval

- [ ] Run Tier 1 through /ship (pm-spec gate is satisfied by these specs → architect →
      implementer → reviewer → qa), one feature per implementer pass
- [ ] Tier 2 same pipeline
- [ ] Tier 3 same pipeline, sequenced per spec
- [ ] Version bumps + README regeneration per tier; verify citations (esp. 2026 arXiv
      IDs from the research sweep) before reusing any of them in Medium articles

## Review section

Shipped all three tiers as **v0.4.0** (2026-06-12).

**Delivered (9 features):**
- T1.1 `bench/stats.py` — `bootstrap_ci`, `min_detectable_effect`, `required_sample_size`; CIs on by default in `BenchmarkRunner`.
- T1.2 `bench/gate.py` + `gate/__main__.py` — `RegressionGate` (baseline save/load, dataset fingerprint, paired/unpaired delta bootstrap, significance gating, power warning) + `python -m llm_evalgate.gate` CLI.
- T1.3 `judge/correction.py` — Rogan–Gladen `corrected_pass_rate` with two-source (eval + calibration) nonparametric bootstrap CI; `calibrate_judge` now carries sensitivity/specificity/confusion.
- T2.4 `judge/faithfulness.py` — `ClaimFaithfulnessDimension` (atomic-claim decomposition, supported/unsupported/contradicted).
- T2.5 `agentic/.../tool_hallucination.py` — `ToolHallucinationDimension` + `ToolSchema.from_json_schema`.
- T2.6 `needs_review` routing — `DimensionResult.needs_review` + producers (SelfConsistency stdev/margin, Jury disagreement, Pairwise flip) + report surfacing.
- T3.7 `agentic/.../step_progress.py` — `StepProgressDimension` (per-step progress, weakest-step localization, no look-ahead) + `AgentStep.user_message`.
- T3.8 `agentic/otel.py` — `AgentTrace.from_otel` / `trace_from_otel_spans` (no OTel SDK dep).
- T3.9 stratification — `BenchmarkRunner.run(stratify_by=...)`, gate per-stratum deltas, 60-sample stratified golden set v2.

**Decisions taken (open questions resolved as recommended):** CIs ON by default; gate CLI kept; golden set grown to 60 now (T3.9); per-claim verification; `needs_review` as a field; OTel import-only.

**Gates:** reviewer loop run 2 rounds — R1 APPROVED with 5 nits, all 5 fixed, R2 APPROVED clean. QA: 195 tests pass (was 118), ruff clean, 4 examples run, `python -m build` produces 0.4.0 wheel+sdist, wheel contents verified (all modules + 60-line dataset), gate CLI smoke PASS with correct power warning. Locked-metric test updated to n=60 numbers (accuracy 0.850, regression_catch_rate 0.690 — deterministic harness misses exactly the 9 semantic regressions).

**Follow-ups (none blocking):** `test_ci_widens_with_smaller_calibration_set` uses a strict `>` on bootstrap CI widths — deterministic under seed, but noted as the first place to look if it ever flakes.
