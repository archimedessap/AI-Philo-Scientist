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
