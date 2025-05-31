#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
坍缩实验公式库

计算不同理论下的坍缩率和相干时间。
"""

import math

# 物理常数
KB = 1.380649e-23  # 玻尔兹曼常数 (J/K)
HBAR = 1.0545718e-34  # 约化普朗克常数 (J·s)

def predict_collapse_rate(dynamics_type, params, experiment):
    """预测波函数坍缩率
    
    Args:
        dynamics_type: 理论动力学类型
        params: 理论参数
        experiment: 实验数据
        
    Returns:
        坍缩率 (1/s)
    """
    try:
        # 提取实验数据
        mass_kg = get_mass_kg(experiment["conditions"])
        
        # CSL类模型
        if dynamics_type in ["GRW", "linear+CSL"]:
            lambda_csl = float(params.get("lambda", 1e-16))
            rc = float(params.get("rc", 1e-7))
            
            # CSL坍缩率: λ(m/m0)²/rc²
            m0 = 1e-27  # 参考质量
            return lambda_csl * (mass_kg/m0)**2
            
        # 热力学类模型
        elif dynamics_type == "thermal":
            gamma = float(params.get("gamma", 1.0))
            T_eff = float(params.get("T_effective", 1e-12))
            
            # 热力学坍缩率: γ k_B T m / ħ
            return gamma * KB * T_eff * mass_kg / HBAR
            
        # 无坍缩理论
        elif dynamics_type in ["linear", "bohmian"]:
            return 0.0
            
        # 非线性薛定谔类
        elif dynamics_type == "nonlinear_Schr":
            # 非线性项可能导致有效坍缩
            nonlinear_factor = float(params.get("nonlinear_factor", 0))
            return nonlinear_factor * 1e-8 * mass_kg  # 简化近似
            
        # 默认无坍缩
        return 0.0
        
    except Exception as e:
        print(f"[ERROR] 计算坍缩率预测时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return 0.0  # 失败时默认无坍缩

def predict_coherence_time(dynamics_type, params, experiment):
    """预测量子相干时间
    
    Args:
        dynamics_type: 理论动力学类型
        params: 理论参数
        experiment: 实验数据
        
    Returns:
        相干时间 (s) 或 "same_as_QM"
    """
    try:
        # 提取实验数据
        mass_kg = get_mass_kg(experiment["conditions"])
        
        # 标准QM预测
        std_time = experiment["std_prediction"]["value"]
        
        # 对于有坍缩的理论，相干时间与坍缩率成反比
        collapse_rate = predict_collapse_rate(dynamics_type, params, experiment)
        
        if collapse_rate > 0:
            # 简化近似: 相干时间 ~ 1/坍缩率
            coherence_time = 1.0 / collapse_rate
            return min(coherence_time, std_time)
        
        # 无坍缩理论，返回标准预测
        return "same_as_QM"
        
    except Exception as e:
        print(f"[ERROR] 计算相干时间预测时出错: {str(e)}")
        return "same_as_QM"  # 失败时返回标准预测

def get_mass_kg(conditions):
    """从实验条件提取质量(kg)"""
    if "mass_kg" in conditions:
        return conditions["mass_kg"]
    elif "mass_amu" in conditions:
        return conditions["mass_amu"] * 1.66053886e-27
    return 9.1093837e-31  # 默认电子质量
