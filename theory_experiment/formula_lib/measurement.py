#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
干涉实验公式库

计算不同理论下的干涉条纹可见度。
"""

import numpy as np
import math

# 物理常数
KB = 1.380649e-23  # 玻尔兹曼常数 (J/K)
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)

def predict_visibility(dynamics_type, params, experiment):
    """预测干涉条纹可见度
    
    Args:
        dynamics_type: 理论动力学类型
        params: 理论参数
        experiment: 实验数据
        
    Returns:
        干涉条纹可见度 (0-1)
    """
    try:
        # 提取实验数据
        mass_kg = get_mass_kg(experiment["conditions"])
        temperature_K = experiment["conditions"].get("temperature_K", 300)
        flight_time_s = experiment["conditions"].get("flight_time_s", 0.01)
        
        # 标准QM预测
        std_visibility = experiment["std_prediction"]["value"]
        
        # CSL类模型
        if dynamics_type in ["GRW", "linear+CSL"]:
            lambda_csl = float(params.get("lambda", 1e-16))
            rc = float(params.get("rc", 1e-7))
            
            # CSL衰减因子: exp(-Γt), Γ = λ(m/m0)²/rc²
            m0 = 1e-27  # 参考质量
            gamma = lambda_csl * (mass_kg/m0)**2
            decoherence_factor = math.exp(-gamma * flight_time_s)
            
            return std_visibility * decoherence_factor
            
        # 热力学类模型
        elif dynamics_type == "thermal":
            gamma = float(params.get("gamma", 1.0))
            T_eff = float(params.get("T_effective", temperature_K))
            
            # 热力学衰减因子: exp(-γ k_B T m / ħ^2)
            decoherence_factor = np.exp(-gamma * KB * T_eff * mass_kg / (HBAR**2))
            return std_visibility * decoherence_factor
            
        # 玻姆类模型 - 干涉与标准QM相同
        elif dynamics_type == "bohmian":
            return std_visibility
            
        # 非线性薛定谔类
        elif dynamics_type == "nonlinear_Schr":
            # 非线性项系数
            nonlinear_factor = float(params.get("nonlinear_factor", 0))
            
            # 简化近似: 干涉可见度略微降低
            return std_visibility * (1.0 - 0.1 * nonlinear_factor)
            
        # 默认返回标准QM预测
        return std_visibility
        
    except Exception as e:
        print(f"[ERROR] 计算干涉实验预测时出错: {str(e)}")
        print(f"理论类型: {dynamics_type}, 参数: {params}")
        import traceback
        traceback.print_exc()
        return experiment["std_prediction"]["value"]  # 失败时返回标准预测

def get_mass_kg(conditions):
    """从实验条件提取质量(kg)"""
    if "mass_kg" in conditions:
        return conditions["mass_kg"]
    elif "mass_amu" in conditions:
        return conditions["mass_amu"] * 1.66053886e-27
    elif "electron_energy_keV" in conditions:
        # 从电子能量计算有效质量
        energy_J = conditions["electron_energy_keV"] * 1.602176634e-16
        return 2 * energy_J / (299792458**2)
    return 9.1093837e-31  # 默认电子质量
