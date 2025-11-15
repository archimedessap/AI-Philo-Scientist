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
