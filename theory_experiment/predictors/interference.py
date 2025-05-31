# theory_experiment/predictors/interference.py
# -*- coding: utf-8 -*-
"""
Hand-written formulas for interference-type experiments.
"""

import math

def _visibility_csl(setup, gamma, rc):
    """CSL-induced visibility suppression."""
    d = setup.get("slit_sep_µm", 0.2) * 1e-6          # slit separation (m)
    E = setup.get("electron_energy_keV", 50) * 1e3 * 1.602e-19  # kinetic energy (J)
    m_e = 9.11e-31
    v = math.sqrt(2*E/m_e)                            # velocity
    L = setup.get("screen_dist_cm", 10) * 1e-2        # to screen (m)
    t = L / v                                         # flight time
    gamma_eff = gamma * (d/rc)**2
    return math.exp(-gamma_eff * t)

# ---------- public API ----------

def predict_double_slit(theory, experiment):
    """Return expected fringe visibility for electron double-slit."""
    p = theory.get("parameters", {})
    gamma = p.get("γ", {}).get("value", 0.0)          # s⁻¹
    rc    = p.get("rc", {}).get("value", 1e-7)        # m
    vis   = _visibility_csl(experiment["setup"], gamma, rc)
    return {"value": vis}

def predict_c60(theory, experiment):
    """Reuse same formula (different mass not considered in quick model)."""
    return predict_double_slit(theory, experiment)

def predict_neutron_grav(theory, experiment):
    g=9.81; h=experiment["setup"].get("height_m",0.03)
    m=1.675e-27; hbar=1.055e-34
    T=experiment["setup"].get("time_s",0.01)
    dphi = m*g*h*T/hbar
    return {"value": dphi}

def predict_neutron_plain(theory, experiment):
    return {"label":"same_as_QM"}
