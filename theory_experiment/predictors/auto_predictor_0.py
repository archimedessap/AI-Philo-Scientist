#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
自动预测路由

根据理论类型和实验类别自动选择合适的预测公式。
"""

import importlib
import math
import numpy as np

class AutoPredictor:
    """自动预测路由类"""
    
    def __init__(self, theory_json):
        """初始化预测器
        
        Args:
            theory_json: 理论JSON数据
        """
        self.theory = theory_json
        self.dynamics_type = self._get_dynamics_type()
        self.parameters = self._extract_parameters()
        print(f"[DEBUG] 理论类型: {self.dynamics_type}, 参数: {self.parameters}")
        
    def _get_dynamics_type(self):
        """获取理论动力学类型"""
        # 1. 首先检查是否直接指定了dynamics.type
        if "dynamics" in self.theory and "type" in self.theory["dynamics"]:
            return self.theory["dynamics"]["type"]
            
        # 2. 尝试从理论描述推断类型
        # 安全获取文本，确保是字符串类型
        theory_name = self._safe_get_text(self.theory.get("name", ""))
        core_principles = self._safe_get_text(self.theory.get("core_principles", ""))
        description = self._safe_get_text(self.theory.get("description", ""))
        
        # 合并所有文本以提高检测准确性
        all_text = f"{theory_name} {core_principles} {description}"
        
        # 区分主要类型的理论
        # 给每种类型一个优先级顺序，避免错误分类
        
        # 非线性模型 (最高优先级检测)
        if any(x in all_text for x in ["nonlinear", "非线性", "修改薛定谔方程"]):
            return "nonlinear_Schr"
            
        # GRW/CSL类模型
        if any(x in all_text for x in ["grw", "ghirardi", "spontaneous collapse", "自发坍缩"]):
            return "GRW"
        if any(x in all_text for x in ["csl", "continuous spontaneous", "连续自发"]):
            return "linear+CSL"
            
        # 导航波理论
        if "thermal" in all_text and any(x in all_text for x in ["pilot", "guide", "导航波", "bohm", "玻姆"]):
            return "thermal"  # 热力学导航波
        elif any(x in all_text for x in ["bohm", "pilot", "guide", "导航波", "玻姆", "隐变量"]):
            return "bohmian"  # 玻姆理论
            
        # 相对态/多世界
        if any(x in all_text for x in ["many-worlds", "many worlds", "多世界", "everett", "相对态"]):
            return "many_worlds"
            
        # 哥本哈根及其变体
        if any(x in all_text for x in ["copenhagen", "哥本哈根", "波函数坍缩", "波包塌缩"]):
            return "copenhagen"
            
        # 检查关系/信息类理论
        if any(x in all_text for x in ["relational", "关系", "information", "信息"]):
            return "relational"
        
        # 默认为多世界解释 (最保守的预测，与QM几乎一致)
        return "many_worlds"
    
    def _safe_get_text(self, value):
        """安全获取文本，确保返回字符串类型"""
        if isinstance(value, str):
            return value.lower()
        elif isinstance(value, list):
            # 如果是列表，将其所有元素连接成字符串
            return " ".join([self._safe_get_text(item) for item in value])
        elif value is None:
            return ""
        else:
            # 其他类型转为字符串
            return str(value).lower()
    
    def _extract_parameters(self):
        """提取理论参数"""
        # 优先从dynamics.parameters获取
        if "dynamics" in self.theory and "parameters" in self.theory["dynamics"]:
            return self.theory["dynamics"]["parameters"]
            
        # 其次从math_core.params获取
        if "math_core" in self.theory and "params" in self.theory["math_core"]:
            return self.theory["math_core"]["params"]
            
        # 从mathematical_formulation获取
        if "mathematical_formulation" in self.theory:
            math_form = self.theory["mathematical_formulation"]
            if isinstance(math_form, dict):
                if "parameters" in math_form:
                    return math_form["parameters"]
                elif "params" in math_form:
                    return math_form["params"]
                
        # 没有发现参数，返回基于理论类型的默认参数
        return self._get_default_parameters()
    
    def _get_default_parameters(self):
        """根据理论类型获取默认参数"""
        if self.dynamics_type == "GRW":
            return {"lambda": 1e-16, "rc": 1e-7}
        elif self.dynamics_type == "linear+CSL":
            return {"lambda": 1e-17, "rc": 1e-7}
        elif self.dynamics_type == "thermal":
            return {"gamma": 0.5, "T_effective": 1e-12}
        elif self.dynamics_type == "nonlinear_Schr":
            return {"nonlinear_factor": 0.1}
        return {}
    
    def predict(self, experiment):
        """预测实验结果
        
        Args:
            experiment: 实验数据
            
        Returns:
            预测结果
        """
        # 获取实验类别和ID
        exp_id = experiment["id"]
        category = experiment.get("category", "")
        
        # 获取标准QM预测值(如果有)
        std_prediction = experiment.get("std_prediction", {}).get("value", None)
        
        # 注意调试输出
        print(f"[DEBUG] 预测实验: {exp_id}, 类型: {self.dynamics_type}")
        
        # 根据理论类型和实验类别生成实际预测值
        try:
            # 波函数坍缩相关实验
            if category == "collapse_bounds" or "collapse" in exp_id:
                # 不同理论对坍缩率有不同预测
                if self.dynamics_type in ["GRW", "linear+CSL"]:
                    lambda_csl = float(self.parameters.get("lambda", 1e-16))
                    rc = float(self.parameters.get("rc", 1e-7))
                    mass_kg = self._get_mass_from_experiment(experiment)
                    m0 = 1e-27  # 参考质量
                    collapse_rate = lambda_csl * (mass_kg/m0)**2
                    return {"value": collapse_rate, "units": "s^-1"}
                elif self.dynamics_type == "thermal":
                    gamma = float(self.parameters.get("gamma", 0.5))
                    T_eff = float(self.parameters.get("T_effective", 1e-12))
                    mass_kg = self._get_mass_from_experiment(experiment)
                    collapse_rate = gamma * 1.380649e-23 * T_eff * mass_kg / 1.0545718e-34
                    return {"value": collapse_rate, "units": "s^-1"}
                elif self.dynamics_type == "nonlinear_Schr":
                    nonlinear_factor = float(self.parameters.get("nonlinear_factor", 0.1))
                    mass_kg = self._get_mass_from_experiment(experiment)
                    collapse_rate = nonlinear_factor * 1e-8 * mass_kg
                    return {"value": collapse_rate, "units": "s^-1"}
                else:
                    # 线性理论预测无坍缩
                    return {"value": 0.0, "units": "s^-1"}
            
            # Bell实验(量子纠缠)
            elif category == "entanglement" or "bell" in exp_id:
                # Bell不等式违背程度
                if self.dynamics_type in ["copenhagen", "many_worlds", "relational"]:
                    # 标准量子力学预测
                    return {"value": 2 * math.sqrt(2), "units": ""}
                elif self.dynamics_type == "bohmian":
                    # 玻姆理论与标准QM相同
                    return {"value": 2 * math.sqrt(2), "units": ""}
                elif self.dynamics_type == "thermal":
                    # 热力学玻姆理论可能有轻微偏差
                    gamma = float(self.parameters.get("gamma", 0.5))
                    bell_value = 2 * math.sqrt(2) * (1 - 0.05 * gamma)
                    return {"value": bell_value, "units": ""}
                elif self.dynamics_type in ["GRW", "linear+CSL"]:
                    # 坍缩模型可能降低关联性
                    lambda_csl = float(self.parameters.get("lambda", 1e-16))
                    reduction_factor = min(0.2, lambda_csl * 1e14)
                    bell_value = 2 * math.sqrt(2) * (1 - reduction_factor)
                    return {"value": bell_value, "units": ""}
                else:
                    # 默认标准量子预测
                    return {"value": 2 * math.sqrt(2), "units": ""}
            
            # 干涉实验
            elif category == "interference" or "interf" in exp_id:
                if std_prediction is None:
                    # 没有标准预测值，生成典型值
                    std_visibility = 0.95  # 假设标准可见度为0.95
                else:
                    std_visibility = std_prediction
                
                # 不同理论对干涉条纹可见度的预测
                if self.dynamics_type in ["GRW", "linear+CSL"]:
                    lambda_csl = float(self.parameters.get("lambda", 1e-16))
                    rc = float(self.parameters.get("rc", 1e-7))
                    mass_kg = self._get_mass_from_experiment(experiment)
                    flight_time_s = experiment["conditions"].get("flight_time_s", 0.01)
                    m0 = 1e-27
                    gamma = lambda_csl * (mass_kg/m0)**2
                    decoherence_factor = math.exp(-gamma * flight_time_s)
                    return {"value": std_visibility * decoherence_factor, "units": ""}
                elif self.dynamics_type == "thermal":
                    gamma = float(self.parameters.get("gamma", 0.5))
                    T_eff = float(self.parameters.get("T_effective", 1e-12))
                    mass_kg = self._get_mass_from_experiment(experiment)
                    temperature_K = experiment["conditions"].get("temperature_K", 300)
                    decoherence_factor = math.exp(-gamma * 1.380649e-23 * T_eff * mass_kg / (1.0545718e-34**2))
                    return {"value": std_visibility * decoherence_factor, "units": ""}
                elif self.dynamics_type == "nonlinear_Schr":
                    nonlinear_factor = float(self.parameters.get("nonlinear_factor", 0.1))
                    # 非线性会轻微降低干涉可见度
                    return {"value": std_visibility * (1.0 - 0.1 * nonlinear_factor), "units": ""}
                else:
                    # 标准量子力学预测
                    return {"value": std_visibility, "units": ""}
                    
            # 量子测量实验(包括泽诺实验)
            elif category == "measurement" or "zeno" in exp_id or "choice" in exp_id:
                if self.dynamics_type in ["copenhagen"]:
                    # 哥本哈根解释强调测量导致坍缩
                    return {"value": 0.99, "units": ""}  # 略高的测量效率
                elif self.dynamics_type in ["many_worlds", "relational"]:
                    # 多世界理论无实际坍缩
                    return {"value": 0.95, "units": ""}  # 标准测量效率
                elif self.dynamics_type in ["GRW", "linear+CSL"]:
                    # 自发坍缩可能影响测量结果
                    lambda_csl = float(self.parameters.get("lambda", 1e-16))
                    reduction_factor = min(0.1, lambda_csl * 1e14)
                    return {"value": 0.95 * (1 - reduction_factor), "units": ""}
                else:
                    # 默认标准预测
                    return {"value": 0.95, "units": ""}
                    
            # 宏观叠加态实验
            elif category == "macro_superposition" or "cat" in exp_id:
                # 提取实验数据
                mass_kg = self._get_mass_from_experiment(experiment)
                
                # 不同理论对宏观叠加态的预测
                if self.dynamics_type in ["GRW", "linear+CSL"]:
                    lambda_csl = float(self.parameters.get("lambda", 1e-16))
                    rc = float(self.parameters.get("rc", 1e-7))
                    m0 = 1e-27
                    collapse_rate = lambda_csl * (mass_kg/m0)**2
                    coherence_time = 1.0 / collapse_rate if collapse_rate > 0 else 1e6
                    return {"value": coherence_time, "units": "s"}
                elif self.dynamics_type == "thermal":
                    gamma = float(self.parameters.get("gamma", 0.5))
                    T_eff = float(self.parameters.get("T_effective", 1e-12))
                    coherence_time = 1.0 / (gamma * 1.380649e-23 * T_eff * mass_kg / 1.0545718e-34)
                    return {"value": coherence_time, "units": "s"}
                elif self.dynamics_type == "nonlinear_Schr":
                    nonlinear_factor = float(self.parameters.get("nonlinear_factor", 0.1))
                    coherence_time = 1.0 / (nonlinear_factor * 1e-8 * mass_kg) if nonlinear_factor > 0 else 1e6
                    return {"value": coherence_time, "units": "s"}
                else:
                    # 线性理论预测无限相干时间(实际上受环境退相干限制)
                    return {"value": 1e6, "units": "s"}  # 非常长的相干时间
            
            # 其他类型实验，返回标准预测值附近的结果
            else:
                if std_prediction is not None:
                    # 根据理论类型微调预测值
                    if self.dynamics_type in ["GRW", "linear+CSL", "thermal", "nonlinear_Schr"]:
                        # 非标准理论可能有微小偏差
                        adjustment = 0.05 * std_prediction  # 5%偏差
                        return {"value": std_prediction - adjustment, "units": experiment["std_prediction"].get("units", "")}
                    else:
                        # 标准理论应该匹配QM预测
                        return {"value": std_prediction, "units": experiment["std_prediction"].get("units", "")}
                
                # 如果没有足够信息，使用标准预测
                print(f"[WARN] 对实验 {exp_id} 使用标准预测")
                return {"label": "same_as_QM"}
                
        except Exception as e:
            print(f"[ERROR] 预测 {exp_id} 失败: {str(e)}")
            import traceback
            traceback.print_exc()
            # 返回无法预测
            return {
                "status": "error",
                "message": f"预测失败: {str(e)}",
                "value": None  # 加入空值而非标准QM预测
            }
    
    def _get_mass_from_experiment(self, experiment):
        """从实验条件提取质量(kg)"""
        conditions = experiment.get("conditions", {})
        if "mass_kg" in conditions:
            return conditions["mass_kg"]
        elif "mass_amu" in conditions:
            return conditions["mass_amu"] * 1.66053886e-27
        elif "electron_energy_keV" in conditions:
            # 从电子能量计算有效质量
            energy_J = conditions["electron_energy_keV"] * 1.602176634e-16
            return 2 * energy_J / (299792458**2)
        return 9.1093837e-31  # 默认电子质量

# 兼容默认Predictor接口
class Predictor(AutoPredictor):
    pass
