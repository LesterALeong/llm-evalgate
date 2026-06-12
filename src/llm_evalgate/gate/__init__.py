"""CLI entry point for the regression gate.

Run ``python -m llm_evalgate.gate current.json baseline.json`` in CI to fail a
build on a metric regression. The gate types themselves live in
:mod:`llm_evalgate.bench.gate` and are re-exported here for convenience.
"""

from ..bench.gate import GateReport, GateRow, RegressionGate

__all__ = ["GateReport", "GateRow", "RegressionGate"]
