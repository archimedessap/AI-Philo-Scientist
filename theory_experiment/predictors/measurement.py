# -*- coding: utf-8 -*-
"""
measurement.py
---------------
手写公式：量子 Zeno 效应与 Leggett–Garg 不等式 (LGI) 解析近似。
"""

import math

# ---------- 1. Quantum Zeno survival probability ----------
def predict_zeno(theory, experiment):
    """
    P(t) ≈ exp[- (λ / N) * t]
    λ 原始衰减率（可从实验 setup 给出）
    N 观测次数 = total_time / Δt
    """
    setup = experiment["setup"]
    lam   = setup.get("lambda_raw", 1.0)            # s^-1
    N     = setup.get("measure_times", 100)
    t_tot = setup.get("total_time_s", 1.0)

    lam_eff = lam / max(N, 1)
    P = math.exp(-lam_eff * t_tot)
    return {"value": P}

# ---------- 2. Leggett–Garg K-function ----------
def predict_lgi(theory, experiment):
    """
    K = 1.5 * exp(-Γ τ)
    CSL/塌缩过程使宏观关联指数衰减
    """
    params = theory.get("parameters", {})
    gamma  = params.get("γ", {}).get("value", 0.0)   # s^-1

    tau = experiment["setup"].get("correlation_time_s", 1e-6)
    K = 1.5 * math.exp(-gamma * tau)
    return {"value": K}
