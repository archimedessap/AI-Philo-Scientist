# -*- coding: utf-8 -*-
"""
Hand-written formulas for entanglement/contextuality tests.
"""

import math

# ---------- CHSH Bell test ----------
def predict_bell(theory, experiment):
    """
    Quick model: standard QM value 2√2 suppressed by exp(-γ τ).
    γ, τ can be tuned by theory parameters or experiment setup.
    """
    params = theory.get("parameters", {})
    gamma  = params.get("γ", {}).get("value", 0.0)   # collapse rate (s⁻¹)
    tau    = experiment["setup"].get("flight_time_s", 1e-6)  # default 1 µs

    S_qm   = 2 * math.sqrt(2)
    S_val  = S_qm * math.exp(-gamma * tau)
    return {"value": S_val}

# ---------- Kochen-Specker photon test ----------
def predict_KS(theory, experiment):
    """Assume QM perfect violation."""
    return {"label": "same_as_QM"}
