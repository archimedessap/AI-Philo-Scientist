# theory_experiment/predictors/interference.py
# -*- coding: utf-8 -*-
"""
Hand-written formulas for interference-type experiments.
"""

import math
from theory_experiment.utils.theory_adapter import TheoryAdapter

AMU_TO_KG = 1.66053906660e-27
ELECTRON_MASS = 9.10938356e-31

GAMMA_KEYS = ("csl_gamma", "gamma_csl", "lambda_csl", "lambda", "gamma", "γ")
RC_KEYS = ("rc", "r_c", "csl_rc", "correlation_length")


def _flatten_params(theory):
    params = TheoryAdapter.collect_parameters(theory)
    return {k: float(v) for k, v in params.items() if isinstance(v, (int, float))}


def _get_param(params, keys, default=None):
    for key in keys:
        if key in params:
            value = params[key]
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    return default


def _slit_separation_m(setup, default):
    if not isinstance(setup, dict):
        return default
    if "slit_sep_m" in setup:
        return float(setup["slit_sep_m"])
    if "slit_sep_um" in setup:
        return float(setup["slit_sep_um"]) * 1e-6
    if "slit_sep_nm" in setup:
        return float(setup["slit_sep_nm"]) * 1e-9
    if "path_sep_nm" in setup:
        return float(setup["path_sep_nm"]) * 1e-9
    return default


def _flight_path_m(setup, default):
    if not isinstance(setup, dict):
        return default
    if "screen_dist_m" in setup:
        return float(setup["screen_dist_m"])
    if "screen_dist_cm" in setup:
        return float(setup["screen_dist_cm"]) * 1e-2
    if "flight_length_m" in setup:
        return float(setup["flight_length_m"])
    if "propagation_length_m" in setup:
        return float(setup["propagation_length_m"])
    return default


def _beam_velocity(setup, particle_mass):
    if isinstance(setup, dict):
        if "beam_velocity_ms" in setup:
            return max(float(setup["beam_velocity_ms"]), 1e-6)
        if "velocity_ms" in setup:
            return max(float(setup["velocity_ms"]), 1e-6)
        if "electron_energy_keV" in setup:
            energy_j = float(setup["electron_energy_keV"]) * 1e3 * 1.602176634e-19
            return math.sqrt(max(2.0 * energy_j / particle_mass, 1e-9))
    # fallback to thermal speed ~200 m/s for heavy molecules
    return 200.0


def _mass_from_setup(setup, fallback):
    if isinstance(setup, dict):
        if "mass_kg" in setup:
            return float(setup["mass_kg"])
        if "mass_amu" in setup:
            return float(setup["mass_amu"]) * AMU_TO_KG
    return fallback


def _std_visibility(experiment):
    std = experiment.get("std_prediction", {}) if isinstance(experiment, dict) else {}
    return float(std.get("value", 1.0))


def _baseline_visibility(experiment):
    std = experiment.get("std_prediction", {}) if isinstance(experiment, dict) else {}
    std_val = std.get("value")
    if isinstance(std_val, (int, float)):
        return float(std_val)
    measured = experiment.get("measured", {}) if isinstance(experiment, dict) else {}
    measured_val = measured.get("value")
    if isinstance(measured_val, (int, float)):
        return float(measured_val)
    return 1.0


def _visibility_with_csl(theory, experiment, particle_mass):
    params = _flatten_params(theory)
    gamma = _get_param(params, GAMMA_KEYS, None)
    rc = _get_param(params, RC_KEYS, None)

    has_math_mod = TheoryAdapter.has_math_modification(theory)

    if gamma is None or rc is None or rc <= 0:
        if has_math_mod:
            return {
                "status": "missing_parameters",
                "message": "理论声明修改了量子数学结构，但未提供有效的 CSL 参数 (gamma/rc)。",
                "required": {
                    "gamma_keys": GAMMA_KEYS,
                    "rc_keys": RC_KEYS
                }
            }
        baseline = _baseline_visibility(experiment)
        return {"value": baseline}

    setup = experiment.get("setup", {}) if isinstance(experiment, dict) else {}
    slit_sep = _slit_separation_m(setup, default=1e-7)
    path_length = _flight_path_m(setup, default=0.1)
    mass = _mass_from_setup(setup, particle_mass)
    velocity = _beam_velocity(setup, particle_mass)
    time_of_flight = max(path_length / velocity, 0.0)

    m0 = AMU_TO_KG  # 1 amu reference
    mass_factor = (mass / m0) ** 2 if mass > 0 else 1.0
    separation_factor = (slit_sep / rc) ** 2 if rc > 0 else 0.0
    suppression = math.exp(-max(gamma, 0.0) * mass_factor * separation_factor * time_of_flight)
    std_vis = _std_visibility(experiment)
    visibility = max(min(std_vis * suppression, 1.0), 0.0)
    return {"value": visibility}


# ---------- public API ----------

def predict_double_slit(theory, experiment):
    """Return expected fringe visibility for electron double-slit."""
    return _visibility_with_csl(theory, experiment, ELECTRON_MASS)


def predict_c60(theory, experiment):
    """CSL-aware visibility prediction for fullerene interference."""
    setup = experiment.get("setup", {}) if isinstance(experiment, dict) else {}
    mass = _mass_from_setup(setup, 720 * AMU_TO_KG)
    return _visibility_with_csl(theory, experiment, mass)

def predict_neutron_grav(theory, experiment):
    g=9.81; h=experiment["setup"].get("height_m",0.03)
    m=1.675e-27; hbar=1.055e-34
    T=experiment["setup"].get("time_s",0.01)
    dphi = m*g*h*T/hbar
    return {"value": dphi}

def predict_neutron_plain(theory, experiment):
    return {"label":"same_as_QM"}
