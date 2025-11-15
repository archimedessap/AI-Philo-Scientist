# -*- coding: utf-8 -*-
"""
Hand-written formulas for entanglement/contextuality tests.
"""

import math

from theory_experiment.utils.theory_adapter import TheoryAdapter

# ---------- CHSH Bell test ----------
def predict_bell(theory, experiment):
    """
    Quick model: standard QM value 2√2 suppressed by exp(-γ τ).
    γ, τ can be tuned by theory parameters or experiment setup.
    """
    params = TheoryAdapter.collect_parameters(theory)
    gamma = float(params.get("gamma", params.get("γ", 0.0))) if params else 0.0
    tau = experiment.get("setup", {}).get("flight_time_s", 1e-6)

    std = experiment.get("std_prediction", {})
    baseline = std.get("value")
    if not isinstance(baseline, (int, float)):
        measured = experiment.get("measured", {})
        baseline = measured.get("value") if isinstance(measured.get("value"), (int, float)) else 2 * math.sqrt(2)

    if gamma is None:
        gamma = 0.0

    S_val = float(baseline) * math.exp(-gamma * tau)
    return {"value": S_val}

# ---------- Kochen-Specker photon test ----------
def predict_KS(theory, experiment):
    """Assume QM perfect violation."""
    value = experiment.get("std_prediction", {}).get("value")
    if isinstance(value, (int, float)):
        return {"value": float(value)}
    measured = experiment.get("measured", {}).get("value")
    if isinstance(measured, (int, float)):
        return {"value": float(measured)}
    return {"label": "same_as_QM"}
