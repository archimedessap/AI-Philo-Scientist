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
    def _calculate_chi2(self, exp, pred):
        # 1. 处理预测状态：如果预测本身表明是错误或不可预测的
        if isinstance(pred, dict) and pred.get("status") in ["unpredictable", "error"]:
            return None

        effective_predicted_value = None

        # 2. 从预测中确定 effective_predicted_value
        # 情况 A: 预测是 "same_as_QM"
        if isinstance(pred, dict) and pred.get("label") == "same_as_QM":
            # 理论的预测被认为是该实验的标准QM预测
            std_qm_value = exp.get("std_prediction", {}).get("value")
            if std_qm_value is None:
                # 如果实验没有定义标准QM预测，
                # "same_as_QM" 无法评估。
                return None 
            effective_predicted_value = std_qm_value
        
        # 情况 B: 预测提供了直接的数值 "value"
        elif isinstance(pred, dict) and "value" in pred and isinstance(pred["value"], (int, float)):
            effective_predicted_value = pred["value"]
        
        # 3. 如果无法确定有效的 effective_predicted_value
        if effective_predicted_value is None:
            # 如果 pred 不是 "same_as_QM" 且没有数值 "value"，
            # 或者 pred 是 "same_as_QM" 但实验缺少 std_prediction，则发生此情况。
            return None

        # 4. 将 effective_predicted_value 与 exp["measured"] 进行比较
        measured_data = exp.get("measured") # 获取 'measured' 字典
        if not isinstance(measured_data, dict):
            # 如果 'measured' 字段缺失或不是字典，则无法比较。
            return None

        # 情形 1: 测量数据是上限
        if "upper_bound" in measured_data:
            upper_bound = measured_data["upper_bound"]
            if not isinstance(upper_bound, (int, float)): return None # 无效的界限类型
            # 如果预测值在上限内（含上限），则视为兼容（χ²=0）
            # 否则，给予一个较大的惩罚性χ²值
            return 0.0 if effective_predicted_value <= upper_bound else self.CHI2_THRESHOLD * 2.0

        # 情形 2: 测量数据是下限
        elif "lower_bound" in measured_data:
            lower_bound = measured_data["lower_bound"]
            if not isinstance(lower_bound, (int, float)): return None # 无效的界限类型
            # 如果预测值在下限外（含下限），则视为兼容（χ²=0）
            # 否则，给予一个较大的惩罚性χ²值
            return 0.0 if effective_predicted_value >= lower_bound else self.CHI2_THRESHOLD * 2.0

        # 情形 3: 测量数据是带有 sigma 的值
        elif "value" in measured_data and "sigma" in measured_data:
            actual_value = measured_data["value"]
            sigma = measured_data["sigma"]
            
            # 验证测量值和 sigma 的类型
            if not isinstance(actual_value, (int, float)) or not isinstance(sigma, (int, float)):
                return None 
            
            if sigma < 0: # Sigma 不能为负
                return None
            if sigma == 0:
                # 如果 sigma 为零，预测必须精确匹配。
                return 0.0 if effective_predicted_value == actual_value else self.CHI2_THRESHOLD * 2.0
            
            # 标准卡方计算
            diff = effective_predicted_value - actual_value
            return (diff**2) / (sigma**2)

        # 回退：如果 'measured' 字典格式无法识别以进行比较
        return None

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

        for exp in self.experiments:
            try:
                pred   = pred_inst.predict(exp)
                chi2   = self._calculate_chi2(exp, pred)

                result = {
                    "experiment_id": exp["id"],
                    "prediction": pred,
                    "chi2_result": chi2,
                    "success": (chi2 is not None and chi2 < self.CHI2_THRESHOLD)
                }
                pred_results.append(result)

                if result["success"]:
                    success_cnt += 1
                if chi2 is not None:
                    chi2_list.append(chi2)

            except Exception as e:
                pred_results.append({
                    "experiment_id": exp["id"],
                    "prediction": {"status":"error", "msg": str(e)},
                    "chi2_result": None,
                    "success": False
                })

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
            "evaluation_type": "experiment"
        }
