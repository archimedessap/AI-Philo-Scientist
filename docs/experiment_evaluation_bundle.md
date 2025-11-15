# Experiment Evaluation Bundle

This file consolidates the current experiment-evaluation pipeline so it can be provided to another assistant for review or modification. It contains:

1. High-level overview of the evaluation flow.
2. Source code for the evaluator and routing predictors.
3. Current experiment dataset.
4. Integration points in the end-to-end pipeline.

---

## 1. Overview

- `theory_experiment/experiment_evaluator.py`: orchestrates experiment predictions, computes chi-square scores, applies success thresholds, and aggregates diagnostics such as residuals and z-scores.
- `theory_experiment/predictors/`: provides hand-written predictors or LLM fallbacks for different experiment categories. `auto_predictor.py` routes experiment IDs to specialized modules (`interference.py`, `measurement.py`, `entanglement.py`, `collapse.py`).
- `theory_experiment/data/experiments.json`: list of benchmark experiments with measured values, uncertainties, and standard QM predictions.
- Integration touchpoints:
  * `run_direct_synthesis.py` + `theory_generation/short_card_generator.py`: generate theories.
  * `demo/demo_1.py` and `run_full_cycle.py`: run experiment evaluation (and role evaluations) via `ExperimentEvaluator` and predictor routing.
  * `theory_experiment/utils/theory_adapter.py`: adapts theory JSON into predictor-friendly format, extracts parameters, and flags math modifications.

---

## 2. Evaluator Core Logic

```python
default_path = "theory_experiment/experiment_evaluator.py"
```
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
theory_experiment/experiment_evaluator.py
----------------------------------------
• 对单个理论 JSON 逐条实验做预测
• 统计 χ²、成功率、最终分数
• 若 predictor 返回 {"status":"unpredictable"} → χ²=None（不计入均值）
"""

import json, importlib
from statistics import mean
from pathlib import Path

from typing import Optional, Tuple

# 内联实现验证器，而不是从外部导入
class SchemaValidator:
    """简单的理论格式验证器"""
    def validate_theory(self, theory):
        """验证理论JSON格式"""
        # 最基本的验证：确保必要字段存在
        required_fields = ["name"]
        missing = [field for field in required_fields if field not in theory]
        
        if missing:
            print(f"[WARN] 理论缺少必要字段: {', '.join(missing)}")
            return False
        return True

# --------------------------------------------------------------------
class ExperimentEvaluator:
    CHI2_THRESHOLD = 4.0   # <4 视作"兼容"

    def __init__(self, experiments_path="theory_experiment/data/experiments.jsonl"):
        self.experiments = self._load_experiments(experiments_path)
        self.schema_validator = SchemaValidator()

    # ------------------ load experiments -----------------------------
    def _load_experiments(self, path):
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(path)

        if p.suffix == ".jsonl":
            with p.open(encoding="utf-8") as f:
                return [json.loads(line) for line in f if line.strip()]
        elif p.suffix == ".json":
            return json.loads(p.read_text(encoding="utf-8"))

        raise ValueError("experiments file must be .jsonl or .json")

    # ------------------ predictor loader -----------------------------
    def _load_predictor(self, theory_json, predictor_module):
        # predictor_module 既可是模块对象，也可以是点分字符串
        if hasattr(predictor_module, "Predictor"):
            return predictor_module.Predictor(theory_json)

        if isinstance(predictor_module, str):
            mod = importlib.import_module(predictor_module)
            return mod.Predictor(theory_json)

        raise TypeError("predictor_module must be module obj or str")

    # ------------------ χ² helper ------------------------------------
    def _extract_effective_prediction(self, exp: dict, pred: dict) -> Optional[float]:
        if isinstance(pred, dict) and pred.get("status") in ["unpredictable", "error", "missing_parameters"]:
            return None

        if isinstance(pred, dict) and pred.get("label") == "same_as_QM":
            std_qm_value = exp.get("std_prediction", {}).get("value")
            if isinstance(std_qm_value, (int, float)):
                return float(std_qm_value)
            return None

        if isinstance(pred, dict) and "value" in pred and isinstance(pred["value"], (int, float)):
            return float(pred["value"])

        return None

    def _calculate_chi2(self, exp, pred, effective_predicted_value=None):
        if effective_predicted_value is None:
            effective_predicted_value = self._extract_effective_prediction(exp, pred)

        if effective_predicted_value is None:
            return None

        measured_data = exp.get("measured")
        if not isinstance(measured_data, dict):
            return None

        if "upper_bound" in measured_data:
            upper_bound = measured_data["upper_bound"]
            if not isinstance(upper_bound, (int, float)):
                return None
            return 0.0 if effective_predicted_value <= upper_bound else self.CHI2_THRESHOLD * 2.0

        if "lower_bound" in measured_data:
            lower_bound = measured_data["lower_bound"]
            if not isinstance(lower_bound, (int, float)):
                return None
            return 0.0 if effective_predicted_value >= lower_bound else self.CHI2_THRESHOLD * 2.0

        if "value" in measured_data and "sigma" in measured_data:
            actual_value = measured_data["value"]
            sigma = measured_data["sigma"]
            if not isinstance(actual_value, (int, float)) or not isinstance(sigma, (int, float)):
                return None
            if sigma < 0:
                return None
            if sigma == 0:
                return 0.0 if effective_predicted_value == actual_value else self.CHI2_THRESHOLD * 2.0
            diff = effective_predicted_value - actual_value
            return (diff ** 2) / (sigma ** 2)

        return None

    def _residual_metrics(self, exp: dict, effective_pred: Optional[float]) -> Tuple[Optional[float], Optional[float]]:
        if effective_pred is None:
            return None, None
        measured = exp.get("measured")
        if not isinstance(measured, dict):
            return None, None
        if "value" in measured and "sigma" in measured:
            actual = measured["value"]
            sigma = measured["sigma"]
            if isinstance(actual, (int, float)) and isinstance(sigma, (int, float)) and sigma > 0:
                residual = effective_pred - actual
                z_score = residual / sigma
                return residual, z_score
            if isinstance(actual, (int, float)) and sigma == 0:
                residual = effective_pred - actual
                z_score = float("inf") if residual != 0 else 0.0
                return residual, z_score
        return None, None

    # ------------------ public API -----------------------------------
    async def evaluate_theory(self, theory_json, predictor_module):
        # schema 验证(非阻断)
        self.schema_validator.validate_theory(theory_json)

        # 如果predictor_module有predict方法，则它已经是实例化的预测器
        if hasattr(predictor_module, 'predict'):
            pred_inst = predictor_module
        else:
            # 否则使用_load_predictor加载预测器
            pred_inst = self._load_predictor(theory_json, predictor_module)

        chi2_list, pred_results = [], []
        success_cnt = 0
        outlier_ids = []
        failed_ids = []

        for exp in self.experiments:
            try:
                pred   = pred_inst.predict(exp)
                effective_pred = self._extract_effective_prediction(exp, pred)
                chi2   = self._calculate_chi2(exp, pred, effective_pred)
                residual, z_score = self._residual_metrics(exp, effective_pred)

                status_value = pred.get("status") if isinstance(pred, dict) else None

                result = {
                    "experiment_id": exp["id"],
                    "prediction": pred,
                    "chi2_result": chi2,
                    "success": (chi2 is not None and chi2 < self.CHI2_THRESHOLD),
                    "effective_prediction": effective_pred,
                    "residual": residual,
                    "z_score": z_score,
                    "std_prediction": exp.get("std_prediction", {}).get("value"),
                    "measured": exp.get("measured"),
                    "status": status_value
                }
                pred_results.append(result)

                if result["success"]:
                    success_cnt += 1
                else:
                    failed_ids.append(exp["id"])
                if chi2 is not None:
                    chi2_list.append(chi2)

                if status_value == "missing_parameters":
                    result.setdefault("warnings", []).append(
                        "缺少关键参数，无法给出有效预测")

                if z_score is not None and abs(z_score) >= 3:
                    result.setdefault("warnings", []).append(
                        f"|z|={abs(z_score):.2f} 表明与实验数据存在明显张力")
                    outlier_ids.append(exp["id"])

            except Exception as e:
                pred_results.append({
                    "experiment_id": exp["id"],
                    "prediction": {"status":"error", "msg": str(e)},
                    "chi2_result": None,
                    "success": False
                })
                failed_ids.append(exp["id"])

        # 统计
        success_rate = success_cnt / len(self.experiments)
        avg_chi2 = mean(chi2_list) if chi2_list else None

        if avg_chi2 is None:
            final_score = 0.0
        elif avg_chi2 < self.CHI2_THRESHOLD:
            final_score = 10 * (1 - avg_chi2 / self.CHI2_THRESHOLD)
        else:
            final_score = max(0.0, 5 * (2 - avg_chi2 / self.CHI2_THRESHOLD))

        return {
            "theory_name": theory_json.get("name", "Unnamed"),
            "theory_id": theory_json.get("id", ""),
            "experiments_evaluated": len(self.experiments),
            "successful_predictions": success_cnt,
            "success_rate": success_rate,
            "average_chi2": avg_chi2,
            "final_score": round(final_score, 2),
            "prediction_results": pred_results,
            "evaluation_type": "experiment",
            "failed_experiments": failed_ids,
            "outlier_experiments": outlier_ids
        }
```

---

## 3. Predictor Router and Modules

### 3.1 `theory_experiment/predictors/auto_predictor.py`
```python
# -*- coding: utf-8 -*-
"""
Router predictor:
• 实验 ID 命中手写字典 → 调用对应函数
• 否则回退到 LLM-driven Predictor
"""
import sys
from . import interference, entanglement, collapse, measurement
from .llm_predictor import LLMPredictor

# 获取通过 CLI 注入的模型名；默认 deepseek-chat
LLM_MODEL_NAME = getattr(sys.modules[__name__], "__llm_model__", "deepseek-chat")

# ---------- 手写公式映射 ----------
HAND_WRITTEN = {
    # --- Interference ---
    "double_slit_electron": interference.predict_double_slit,
    "c60_molecule_interf":  interference.predict_c60,
    "neutron_interfer_grav": interference.predict_neutron_grav,
    "neutron_interferometer": interference.predict_neutron_plain,

    # --- Entanglement / Contextuality ---
    "bell_loophole_free":   entanglement.predict_bell,
    "kochen_specker_photon": entanglement.predict_KS,

    # --- Measurement effects ---
    "quantum_zeno_ion":     measurement.predict_zeno,
    "leggett_garg_supercond": measurement.predict_lgi,

    # --- Collapse bounds ---
    "optomech_collapse_bound": collapse.predict_optomech,
    "ultra_cold_csl_bound":    collapse.predict_ultra_cold,
}

# ---------- Router ----------
class Predictor:
    def __init__(self, theory_json, model_name="deepseek-chat", model_source="openai"):
        self.theory = theory_json
        self.llm_pred = LLMPredictor(theory_json, model_name=model_name, model_source=model_source)

    def predict(self, experiment: dict):
        if "prediction_map" in self.theory:
            func_or_flag = self.theory["prediction_map"].get(experiment["id"], "LLM_auto")
            if func_or_flag == "LLM_auto":
                return self.llm_pred.predict(experiment)
            elif func_or_flag in HAND_WRITTEN:
                try:
                    return HAND_WRITTEN[func_or_flag](self.theory, experiment)
                except Exception as e:
                    print(f"[WARN] prediction_map指定的处理函数失败 ({func_or_flag}): {e}")
                    return self.llm_pred.predict(experiment)
        
        func = HAND_WRITTEN.get(experiment["id"])
        if func:
            try:
                return func(self.theory, experiment)
            except Exception as e:
                print(f"[WARN] hand-written predictor failed ({experiment['id']}): {e}")
        return self.llm_pred.predict(experiment)
```

### 3.2 `theory_experiment/predictors/interference.py`
```python
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


def _visibility_with_csl(theory, experiment, particle_mass):
    params = _flatten_params(theory)
    gamma = _get_param(params, GAMMA_KEYS, None)
    rc = _get_param(params, RC_KEYS, None)

    needs_params = TheoryAdapter.needs_parameter_support(theory)

    if gamma is None or rc is None or rc <= 0:
        if needs_params:
            return {
                "status": "missing_parameters",
                "message": "理论声明修改了量子数学结构，但未提供有效的 CSL 参数 (gamma/rc)。",
                "required": {
                    "gamma_keys": GAMMA_KEYS,
                    "rc_keys": RC_KEYS
                }
            }
        return {"label": "same_as_QM"}

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
```

### 3.3 `theory_experiment/predictors/measurement.py`
```python
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
```

### 3.4 `theory_experiment/predictors/entanglement.py`
```python
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
```

### 3.5 `theory_experiment/predictors/collapse.py`
```python
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
```

---

## 4. Experiment Dataset

`theory_experiment/data/experiments.json`

```json
[
{"id":"double_slit_electron",
 "category":"interference",
 "observable":"fringe_visibility",
 "std_prediction":{"value":0.95,"units":""},
 "measured":{"value":0.94,"sigma":0.02,"units":""},
 "conditions":{"electron_energy_keV":50},
 "setup":{"electron_energy_keV":50,
           "slit_sep_um":0.2,
           "screen_dist_cm":10.0},
 "refs":["Jönsson 1961 Am. J. Phys. 42 4"]}
 ,
{"id":"c60_molecule_interf",
 "category":"interference",
 "observable":"V(mass=720u)",
 "std_prediction":{"value":0.88,"units":""},
 "measured":{"value":0.86,"sigma":0.03,"units":""},
 "conditions":{"mass_amu":720},
 "setup":{"mass_amu":720,
           "slit_sep_nm":100.0,
           "flight_length_m":1.0,
           "beam_velocity_ms":200.0},
 "refs":["Arndt et al. Nature 401 (1999) 680"]}
,
{"id":"bell_loophole_free",
 "category":"entanglement",
 "observable":"CHSH_S",
 "std_prediction":{"value":0.707,"units":"corr"},
 "measured":{"value":0.705,"sigma":0.012,"units":"corr"},
 "conditions":{"distance_m":1300},
 "setup":{"distance_m":1300, "flight_time_s":4.3e-6},
 "refs":["10.1038/nature15759"],
 "name": "无漏洞贝尔测试",
 "description": "..."}
,
{"id":"delayed_choice_qw",
 "category":"measurement",
 "observable":"which_path_prob",
 "std_prediction":{"value":0.50,"units":""},
 "measured":{"value":0.51,"sigma":0.02,"units":""},
 "conditions":{"choice_delay_ns":30},
 "setup":{"choice_delay_ns":30},
 "refs":["Ma et al. PNAS 110 (2013) 1221"]}
,
{"id":"leggett_garg_supercond",
 "category":"macro_realism",
 "observable":"LGI_K",
 "std_prediction":{"value":1.40,"units":""},
 "measured":{"value":1.35,"sigma":0.10,"units":""},
 "conditions":{"device":"RF-SQUID"},
 "setup":{"device":"RF-SQUID", "correlation_time_s":1e-9},
 "refs":["Palacios-Lavado PRL 123 (2019) 250403"]}
,
{"id":"quantum_zeno_ion",
 "category":"measurement",
 "observable":"decay_rate_ratio",
 "std_prediction":{"value":0.10,"units":""},
 "measured":{"value":0.11,"sigma":0.01,"units":""},
 "conditions":{"probe_interval_us":0.1},
 "setup":{"probe_interval_us":0.1, "lambda_raw":1.0, "measure_times":100, "total_time_s":1.0},
 "refs":["Itano et al. PR A 41 (1990) 2295"]}
,
{"id":"neutron_interfer_grav",
 "category":"phase",
 "observable":"grav_phase_shift_rad",
 "std_prediction":{"value":1.10,"units":"rad"},
 "measured":{"value":1.11,"sigma":0.05,"units":"rad"},
 "conditions":{"height_cm":2.5},
 "setup":{"height_cm":2.5, "height_m":0.025, "time_s":0.01},
 "refs":["Colella, Overhauser, Werner 1975"]}
,
{"id":"optomech_collapse_bound",
 "category":"collapse_bounds",
 "observable":"collapse_rate_bound",
 "std_prediction":{"value":0.0,"units":"s^-1"},
 "measured":{"upper_bound":3e-8,"units":"s^-1"},
 "conditions":{"mass_kg":2e-11},
 "setup":{"mass_kg":2e-11, "mass_ng":20, "freq_kHz":10},
 "refs":["Vinante et al. PRL 119 (2017) 110401"]}
,
{"id":"spin_cat_NV",
 "category":"macro_superposition",
 "observable":"coherence_time_us",
 "std_prediction":{"value":120.0,"units":"µs"},
 "measured":{"value":118.0,"sigma":5.0,"units":"µs"},
 "conditions":{"mass_amu":3e4},
 "setup":{"mass_amu":3e4},
 "refs":["Kehayias et al. PR A 91 (2015) 012107"]}
,
{"id":"kochen_specker_photon",
 "category":"contextuality",
 "observable":"KS_violation",
 "std_prediction":{"value":0.85,"units":""},
 "measured":{"value":0.84,"sigma":0.03,"units":""},
 "conditions":{"qubits":3},
 "setup":{"qubits":3},
 "refs":["Liu et al. PRL 121 (2018) 190404"]}
,
{"id":"quantum_erasure_mzi",
 "category":"measurement",
 "observable":"erased_fringe_visibility",
 "std_prediction":{"value":0.95,"units":""},
 "measured":{"value":0.93,"sigma":0.03,"units":""},
 "conditions":{"delay_ns":25},
 "setup":{"delay_ns":25},
 "refs":["Kim et al. PRL 84 (2000) 1"]}
,
{"id":"ultra_cold_csl_bound",
 "category":"collapse_bounds",
 "observable":"lambda_upper_bound",
 "std_prediction":{"value":0.0,"units":"s^-1"},
 "measured":{"upper_bound":1e-9,"units":"s^-1"},
 "conditions":{"mass_kg":1e-25,"temperature_nK":100},
 "setup":{"mass_kg":1e-25,"temperature_nK":100},
 "refs":["Toroš & Bassi PRL 116 (2016) 160403"]}
,
{"id":"macroscopic_interference",
 "category":"interference",
 "observable":"visibility",
 "std_prediction":{"value":0.98,"units":""},
 "measured":{"value":0.95,"sigma":0.05,"units":""},
 "conditions":{"mass_kg":2.5e-25},
 "setup":{"mass_kg":2.5e-25},
 "refs":["10.1038/nature16155"]}
,   
{"id":"neutron_interferometer",
 "category":"interference",
 "observable":"visibility",
 "std_prediction":{"value":0.935,"units":""},
 "measured":{"value":0.941,"sigma":0.018,"units":""},
 "conditions":{"mass_kg":1.67e-27},
 "setup":{"mass_kg":1.67e-27},
 "refs":["10.1103/PhysRevLett.34.1472"],
 "name": "中子干涉实验",
 "description": "..."
}
]
```

---

## 5. Adapter & Integration Helpers

### 5.1 `theory_experiment/utils/theory_adapter.py`
```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论适配器

将不同格式的理论JSON适配为统一格式。
"""

import re
from typing import Any, Dict, Iterable, List, Optional

class TheoryAdapter:
    """理论格式适配器"""
    
    @staticmethod
    def adapt(theory_json):
        """将理论JSON适配为标准格式
        
        Args:
            theory_json: 原始理论JSON
            
        Returns:
            适配后的理论JSON
        """
        if not theory_json:
            # 处理空输入
            return {"dynamics": {"type": "linear", "parameters": {}}}
            
        adapted = theory_json.copy()
        
        # 确保有dynamics字段
        if "dynamics" not in adapted:
            adapted["dynamics"] = {}
            
            # 从名称和描述推断类型
            theory_name = adapted.get("name", "").lower()
            description = adapted.get("description", "").lower()
            
            # 推断dynamics.type
            if any(x in theory_name or x in description for x in ["grw", "ghirardi"]):
                adapted["dynamics"]["type"] = "GRW"
            elif any(x in theory_name or x in description for x in ["csl", "continuous spontaneous"]):
                adapted["dynamics"]["type"] = "linear+CSL"
            elif "thermal" in theory_name and "pilot" in theory_name:
                adapted["dynamics"]["type"] = "thermal" 
            elif any(x in theory_name or x in description for x in ["bohm", "pilot", "guide"]):
                adapted["dynamics"]["type"] = "bohmian"
            elif any(x in theory_name or x in description for x in ["nonlinear", "非线性"]):
                adapted["dynamics"]["type"] = "nonlinear_Schr"
            else:
                adapted["dynamics"]["type"] = "linear"
                
        # 确保有parameters字段
        if "parameters" not in adapted["dynamics"]:
            adapted["dynamics"]["parameters"] = {}
            
            # 从其他地方提取参数
            if "math_core" in adapted and "params" in adapted["math_core"]:
                adapted["dynamics"]["parameters"].update(adapted["math_core"]["params"])
                
            elif "mathematical_formulation" in adapted:
                math_form = adapted["mathematical_formulation"]
                if "parameters" in math_form:
                    adapted["dynamics"]["parameters"].update(math_form["parameters"])
                elif "params" in math_form:
                    adapted["dynamics"]["parameters"].update(math_form["params"])
                    
        # 处理None值参数
        params = adapted["dynamics"]["parameters"]
        for key, value in list(params.items()):
            if value is None:
                params[key] = 0
        
        return adapted

    @staticmethod
    def _coerce_numeric(value: Any) -> Any:
        """将常见的 {"value": x} 结构或字符串数值转换为浮点数。"""
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, dict):
            if "value" in value:
                try:
                    return float(value["value"])
                except (TypeError, ValueError):
                    return None
            if "mean" in value:
                try:
                    return float(value["mean"])
                except (TypeError, ValueError):
                    return None
        if isinstance(value, str):
            stripped = value.strip()
            try:
                return float(stripped)
            except ValueError:
                return None
        return None

    @staticmethod
    def collect_parameters(theory_json: Dict[str, Any]) -> Dict[str, float]:
        """聚合理论中分散的参数并返回展平后的数值字典。"""
        if not isinstance(theory_json, dict):
            return {}

        aggregated: Dict[str, float] = {}

        def merge_params(container: Any):
            if isinstance(container, dict):
                for key, raw in container.items():
                    numeric = TheoryAdapter._coerce_numeric(raw)
                    if numeric is not None and key not in aggregated:
                        aggregated[key] = numeric
                return

            if isinstance(container, list):
                for item in container:
                    if not isinstance(item, dict):
                        continue
                    name = item.get("name") or item.get("id")
                    for field in ("value", "default", "default_value", "mean"):
                        if field in item:
                            numeric = TheoryAdapter._coerce_numeric(item[field])
                            if numeric is not None:
                                aggregated[name or field] = numeric
                                break

        for src in TheoryAdapter._parameter_sources(theory_json):
            merge_params(src)

        return aggregated

    @staticmethod
    def _collect_math_relations(theory_json: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        relations = []
        direct = theory_json.get('math_relation_to_SQM') or theory_json.get('math_relation_to_sqm')
        if isinstance(direct, dict):
            relations.append(direct)

        machine_summary = theory_json.get('machine_summary')
        if isinstance(machine_summary, dict):
            rel = TheoryAdapter._normalize_math_relation(machine_summary)
            if isinstance(rel, dict):
                relations.append(rel)

        card_workflow = theory_json.get('card_workflow')
        if isinstance(card_workflow, dict):
            machine = card_workflow.get('machine_summary')
            if isinstance(machine, dict):
                rel = TheoryAdapter._normalize_math_relation(machine)
                if isinstance(rel, dict):
                    relations.append(rel)

        return relations

    @staticmethod
    def _normalize_math_relation(machine_summary: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        relation = machine_summary.get('math_relation_to_SQM') or machine_summary.get('math_relation_to_sqm')
        if isinstance(relation, dict):
            return relation

        meta = machine_summary.get('meta', {}) if isinstance(machine_summary, dict) else {}
        dynamics = machine_summary.get('dynamics', {}) if isinstance(machine_summary, dict) else {}
        if isinstance(meta, dict):
            relation_type = meta.get('relation_to_SQM')
            if relation_type:
                if relation_type == 'minimal_change':
                    relation_type = 'no_change'
                math_change = relation_type != 'no_change'
                return {
                    'type': relation_type,
                    'math_change': math_change,
                    'equations_summary': dynamics.get('equation') or dynamics.get('commentary') or ''
                }
        return None

    @staticmethod
    def has_math_modification(theory_json: Dict[str, Any]) -> bool:
        for relation in TheoryAdapter._collect_math_relations(theory_json):
            math_change = relation.get('math_change')
            if math_change is not None:
                return bool(math_change)
            relation_type = relation.get('type')
            if relation_type in {'modified_dynamics', 'modified_logic', 'retrocausal', 'extension', 'modification'}:
                return True

        metadata = theory_json.get('metadata', {})
        if isinstance(metadata, dict):
            classification = metadata.get('mathematical_classification', {})
            if isinstance(classification, dict):
                if classification.get('math_change') is True:
                    return True
                uses_standard = classification.get('uses_standard_qm_math')
                if uses_standard is False:
                    return True

        legacy = theory_json.get('mathematical_relation_to_sqm') or theory_json.get('mathematical_relation_to_SQM')
        if isinstance(legacy, dict):
            legacy_type = legacy.get('type')
            if legacy_type in {'extension', 'modification', 'modified_dynamics', 'modified_logic'}:
                return True
            if legacy.get('math_change') is True:
                return True

        return False

    @staticmethod
    def _contains_range(container: Any) -> bool:
        if isinstance(container, dict):
            if any(key in container for key in ('range', 'bounds', 'interval')):
                return True
            keys = container.keys()
            if {'min', 'max'}.issubset(keys) or {'lower', 'upper'}.issubset(keys):
                return True
            for value in container.values():
                if TheoryAdapter._contains_range(value):
                    return True
        if isinstance(container, list):
            if len(container) == 2 and all(isinstance(v, (int, float)) for v in container):
                return True
            return any(TheoryAdapter._contains_range(item) for item in container)
        return False

    @staticmethod
    @staticmethod
    def _parameter_sources(theory_json: Dict[str, Any]) -> List[Any]:
        sources: List[Any] = []
        if not isinstance(theory_json, dict):
            return sources

        if theory_json.get('parameters') is not None:
            sources.append(theory_json.get('parameters'))

        math_core = theory_json.get('math_core')
        if isinstance(math_core, dict):
            for key in ('params', 'parameters'):
                if math_core.get(key) is not None:
                    sources.append(math_core.get(key))

        math_form = theory_json.get('mathematical_formulation')
        if isinstance(math_form, dict):
            for key in ('parameters', 'params'):
                if math_form.get(key) is not None:
                    sources.append(math_form.get(key))

        adapted = TheoryAdapter.adapt(theory_json)
        if isinstance(adapted, dict):
            dynamics = adapted.get('dynamics', {})
            if isinstance(dynamics, dict) and dynamics.get('parameters') is not None:
                sources.append(dynamics.get('parameters'))

        return sources

    @staticmethod
    def has_parameter_support(theory_json: Dict[str, Any]) -> bool:
        if TheoryAdapter.collect_parameters(theory_json):
            return True

        for src in TheoryAdapter._parameter_sources(theory_json):
            if src and TheoryAdapter._contains_range(src):
                return True
        return False

    @staticmethod
    def needs_parameter_support(theory_json: Dict[str, Any]) -> bool:
        for src in TheoryAdapter._parameter_sources(theory_json):
            if not src:
                continue
            if isinstance(src, dict) and src:
                return True
            if isinstance(src, list) and len(src) > 0:
                return True
        return False
```

### 5.2 Runner Integration (`demo/demo_1.py` excerpt)
```python
#!/usr/bin/env python3
# coding: utf-8
"""
量子理论评估工具 - 多理论多实验评估器
可以指定LLM模型，评估多个量子理论对多个实验的预测能力

"""
import sys, os, json, asyncio, argparse, glob, re, time
from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from theory_generation.llm_interface import LLMInterface
# Instrument correction is optional; import lazily when needed to allow offline runs
# 导入新的角色评估模块
from demo.auto_role_evaluation import run_role_evaluation_for_theories

def load_theories_from_sources(theories_path: str, schema_version: str = "2.1") -> dict:
    """
    从目录或单个文件加载理论数据
    
    Args:
        theories_path: 理论目录或单个理论文件的路径
        schema_version: 要求加载的理论schema版本 ('any' to load all)
        
    Returns:
        dict: 理论名称到 (理论数据, 文件路径) 元组的映射
    """
    theories = {}
    
    # 确定是目录还是文件
    if os.path.isdir(theories_path):
        theory_files = glob.glob(os.path.join(theories_path, "*.json"))
        print(f"[INFO] 在目录 {theories_path} 中找到 {len(theory_files)} 个理论文件")
    elif os.path.isfile(theories_path) and theories_path.endswith('.json'):
        theory_files = [theories_path]
        print(f"[INFO] 正在加载单个理论文件: {theories_path}")
    else:
        print(f"[ERROR] 无效的理论路径: {theories_path}")
        return {}
        
    for theory_file in theory_files:
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            
            # 兼容包含理论列表的JSON文件
            if isinstance(theory, list):
                theory_list = theory
            else:
                theory_list = [theory]
            
            for t in theory_list:
                # 检查schema版本
                load_any_schema = schema_version is None or schema_version.lower() == 'any'
                file_schema_version = t.get("metadata", {}).get("schema_version")
                
                if not load_any_schema and file_schema_version != schema_version:
                    print(f"[WARN] 跳过理论 '{t.get('name', '未命名')}': schema版本不匹配 (需要 {schema_version}, 文件为 {file_schema_version})")
                    continue

                theory_name = t.get("name", os.path.basename(theory_file))
                if theory_name in theories:
                    print(f"[WARN] 发现重复的理论名称 '{theory_name}'，将覆盖旧版本。")
                # 保存理论数据和其原始文件路径
                theories[theory_name] = (t, theory_file)

        except Exception as e:
            print(f"[ERROR] 加载理论文件 {theory_file} 时出错: {str(e)}")
            
    return theories

async def evaluate_theory_experiment(theory, setup_exp, measured_data, llm, args, output_prefix=None, corrected_setup_exp=None):
    """评估单个理论对单个实验的预测能力"""
    # 导入数学分类器
    import sys
    sys.path.append('.')
    from utils.mathematical_classifier import MathematicalClassifier
    
    # 检查理论是否使用标准QM数学
    classifier = MathematicalClassifier()
    
    # 检查理论是否已有分类标注
    existing_classification = theory.get("metadata", {}).get("mathematical_classification")
    if existing_classification:
        uses_standard_qm = existing_classification.get("uses_standard_qm_math", False)
        math_type = existing_classification.get("type", "unknown")
    else:
        # 进行实时分类
        classification, analysis = classifier.classify_theory_mathematics(theory)
        uses_standard_qm = classification == "standard_qm"
        math_type = classification
    
    theory_name = theory.get("name", "未知理论")
    exp_id = setup_exp["id"]
    
    # 如果使用标准QM数学，跳过实验评估，直接给予完美分数
    if uses_standard_qm:
        print(f"[INFO] 理论'{theory_name}'使用标准量子力学数学，跳过实验评估")
        print(f"[INFO] 数学分类: {math_type}")
        
        # 创建虚拟的完美评估结果
        measured = measured_data[exp_id]["value"]
        sigma = measured_data[exp_id]["sigma"]
        
        structured_output = {
            "theory_name": theory_name,
            "experiment_id": exp_id,
            "derivation": f"理论'{theory_name}'使用标准量子力学数学形式，因此预测与实验完全一致。",
            "predicted_value": float(measured),  # 预测值等于测量值
            "measured_value": float(measured),
            "sigma": float(sigma),
            "chi2": 0.0,  # 完美匹配
            "success": True,  # 100%成功
            "chi2_threshold": float(args.chi2_threshold if hasattr(args, 'chi2_threshold') else 4.0),
            "mathematical_classification": {
                "type": math_type,
                "uses_standard_qm_math": True,
                "skipped_experiment": True  # 标记跳过了实验评估
            },
            "model_info": {
                "source": args.model_source,
                "name": args.model_name,
                "temperature": float(args.temperature)
            }
        }
        
        # 确定输出文件路径
        if args.output_dir:
            if output_prefix:
                filename_prefix = output_prefix
            else:
                theory_filename = theory_name.replace(" ", "_").lower()
                exp_filename = exp_id.replace(" ", "_").lower()
                filename_prefix = f"{theory_filename}_vs_{exp_filename}"
            
            raw_output_file = os.path.join(args.output_dir, f"{filename_prefix}_response_raw.txt")
            output_file = os.path.join(args.output_dir, f"{filename_prefix}_evaluation.json")
        else:
            raw_output_file = args.raw_output_file
            output_file = args.output_file
        
        # 确保目录存在
        output_dir = os.path.dirname(raw_output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
        
        # 保存简化的原始响应
        with open(raw_output_file, "w", encoding="utf-8") as f:
            f.write(f"理论'{theory_name}'使用标准量子力学数学，自动通过实验验证。\n")
            f.write(f"数学分类: {math_type}\n")
            f.write(f"预测值: {measured}\n")
            f.write(f"实验值: {measured}\n")
            f.write(f"χ²值: 0.0 (完美匹配)\n")
        
        # 保存结构化输出
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(structured_output, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 标准QM理论自动通过: χ²=0.0, 成功率=100%")
        print(f"[INFO] 评估结果已保存到: {output_file}")
        
        return structured_output
    
    # 对于非标准QM理论，继续原有的评估流程
    print(f"[INFO] 理论'{theory_name}'使用修改的量子力学数学，进行完整实验评估")
    print(f"[INFO] 数学分类: {math_type}")
    
    # 获取实验ID
    exp_id = setup_exp["id"]
    
    # 检查是否有对应的测量数据
    if measured_data and exp_id in measured_data:
        complete_exp = setup_exp.copy()
        complete_exp["measured"] = {
            "value": measured_data[exp_id]["value"],
            "sigma": measured_data[exp_id]["sigma"]
        }
    else:
        print(f"[ERROR] 在测量数据中找不到实验ID: {exp_id}")
        return None
    
    # 提取实验目标
    exp_type = setup_exp.get("type", setup_exp.get("category", "未知类型"))
    exp_target = setup_exp.get("target_value", setup_exp.get("observable", "未定义目标"))
    
    # 使用Schema v2.1更新Prompt
    prompt = f"""
    You are a quantum physics expert tasked with evaluating a theoretical model against experimental data.
    
    ## Theory: {theory.get("name", "Unknown Theory")} (Schema Version: {theory.get("metadata", {}).get("schema_version", "N/A")})
    
    ### Core Identity
    - **UID:** {theory.get("metadata", {}).get("uid", "N/A")}
    - **Lineage:** {json.dumps(theory.get("metadata", {}).get("lineage", {}), indent=2)}
    - **Relation to SQM:** {theory.get("mathematical_relation_to_sqm", "Not specified")}

    ### Core Principles
```

### 5.3 Full-cycle orchestration (`run_full_cycle.py` excerpt)
```python
        args.synthesis_model_source = 'openai'
    elif 'deepseek' in args.synthesis_model_name.lower() and args.synthesis_model_source is None:
        args.synthesis_model_source = 'deepseek'

    # 如果用户没有显式指定evaluation_model_source，则根据evaluation_model_name推断
    if 'gemini' in args.evaluation_model_name.lower() and args.evaluation_model_source is None:
        args.evaluation_model_source = 'google'
    elif 'gpt' in args.evaluation_model_name.lower() and args.evaluation_model_source is None:
        args.evaluation_model_source = 'openai'
    elif 'deepseek' in args.evaluation_model_name.lower() and args.evaluation_model_source is None:
        args.evaluation_model_source = 'deepseek'

    # 1. 创建本次运行的专属主目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    main_run_dir = os.path.join(args.base_output_dir, f"run_{timestamp}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"主运行目录已创建: {main_run_dir}")

    # =================================================================
    # 阶段一: 理论合成
    # =================================================================
    synthesis_output_dir = os.path.join(main_run_dir, "1_synthesis_output")
    os.makedirs(synthesis_output_dir, exist_ok=True)

    synthesis_command = [
        "python", "run_direct_synthesis.py",
        "--generation_method", args.generation_method,
        "--model_source", args.synthesis_model_source,
        "--model_name", args.synthesis_model_name,
        "--output_dir", synthesis_output_dir
    ]

    if args.generation_method == "direct":
        synthesis_command.extend([
            "--theories_dir", args.existing_theories_dir,
            "--max_pairs", str(args.max_pairs_to_analyze),
            "--variants_per_contradiction", str(args.variants_per_contradiction),
        ])
    else:
        synthesis_command.extend([
            "--cards_dir", args.cards_dir,
            "--card_schema", args.card_schema,
            "--contradiction_schema", args.contradiction_schema,
            "--new_interpretation_schema", args.new_interpretation_schema,
            "--short_card_query", args.short_card_query,
            "--short_card_task_hint", args.short_card_task_hint,
            "--short_card_topk", str(args.short_card_topk),
            "--short_card_machine_temperature", str(args.short_card_machine_temperature),
            "--short_card_human_temperature", str(args.short_card_human_temperature),
        ])
        if args.short_card_constraints:
            synthesis_command.extend(["--short_card_constraints", args.short_card_constraints])
        if args.short_card_human_model_source:
            synthesis_command.extend(["--short_card_human_model_source", args.short_card_human_model_source])
        if args.short_card_human_model_name:
            synthesis_command.extend(["--short_card_human_model_name", args.short_card_human_model_name])

    synthesis_return_code, synthesis_output = run_command(synthesis_command, "理论合成")

    if synthesis_return_code != 0:
        print("\n[FATAL] 理论合成阶段失败，无法继续。请检查以上日志。")
        sys.exit(1)

    # 从合成脚本的输出中解析出可评估理论的目录
    eval_ready_path_match = re.search(r"标准格式的评估理论文件已保存到: (.*)", synthesis_output)
    if not eval_ready_path_match:
        print("\n[FATAL] 无法从合成脚本的输出中找到可评估理论的路径，无法继续。")
        sys.exit(1)
        
    eval_ready_theories_path = eval_ready_path_match.group(1).strip()
    print(f"\n[INFO] 成功解析出新理论路径: {eval_ready_theories_path}")

    # =================================================================
    # 阶段二: 理论评估
    # =================================================================
    evaluation_output_dir = os.path.join(main_run_dir, "2_evaluation_output")
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    evaluation_command = [
        "python", "demo/demo_1.py",
        "--theory_path", eval_ready_theories_path,
        "--experiment_dir", args.experiment_dir,
        "--output_dir", evaluation_output_dir,
        "--model_source", args.evaluation_model_source,
        "--model_name", args.evaluation_model_name,
        "--run_role_evaluation",
        "--role_success_threshold", str(args.role_eval_threshold)
    ]

    if args.use_instrument_correction:
        evaluation_command.append("--use_instrument_correction")

    evaluation_return_code, _ = run_command(evaluation_command, "理论评估")

    if evaluation_return_code != 0:
        print("\n[FATAL] 理论评估阶段失败。请检查以上日志。")
        sys.exit(1)
        
    print_banner("全周期运行成功完成！")
    print(f"所有结果已保存在: {main_run_dir}")

```

---

*End of bundle.*
