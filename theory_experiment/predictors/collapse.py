# -*- coding: utf-8 -*-
"""
collapse.py
------------
手写公式模块：计算 CSL (Continuous Spontaneous Localization)
对宏观系统的附加加热率 / 扩散率等观测量。

参考：Adler & Bassi, Science 325 (2009); Bahrami et al., PRA 91 (2015).
"""

import math

# ---------- 共用物理常数 ----------
HBAR = 1.054571817e-34   # J·s
M0   = 1.66053906660e-27 # 1 amu in kg
PI   = math.pi

# ---------- 1. Optomechanical collapse bound ----------
def predict_optomech(theory, experiment):
    """
    观测量：附加加热率 [Hz]
    Γ_CSL ≈ γ * (m / m0)^2 * ħ / (4 π m ω r_c^2)
    其中 m0 = 1 amu
    """
    p = theory.get("parameters", {})
    gamma = p.get("γ", {}).get("value", 0.0)     # s^-1
    rc    = p.get("rc", {}).get("value", 1e-7)   # m

    setup = experiment["setup"]
    m     = setup.get("mass_ng", 0) * 1e-9       # kg
    omega = setup.get("freq_kHz", 0) * 1e3 * 2 * PI     # rad/s

    heating_rate = gamma * (m / M0)**2 * HBAR / (4 * PI * m * omega * rc**2)
    # 转 Hz (divide by 2π)
    return {"value": heating_rate / (2 * PI)}

# ---------- 2. Ultra-cold CSL bound ----------
def predict_ultra_cold(theory, experiment):
    """
    观测量：位置扩散率 D  (m^2 / s)
    D_CSL = ħ^2 γ / (4 m^2 r_c^2)
    """
    p = theory.get("parameters", {})
    gamma = p.get("γ", {}).get("value", 0.0)
    rc    = p.get("rc", {}).get("value", 1e-7)

    m = experiment["setup"].get("mass_kg", 1.0e-25)   # 若实验无质量字段

    D = (HBAR**2 * gamma) / (4 * m**2 * rc**2)
    return {"value": D}
