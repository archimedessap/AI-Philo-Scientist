#!/usr/bin/env python3
# coding: utf-8
"""
仪器修正模块 - 处理理论预测值到实验测量值的转换

按照"仪器属于实验层"的原则：
1. 仪器参数对所有理论统一使用
2. 支持两步式（先校准后评比）和联合拟合两种模式
3. 误差传播包含仪器校准不确定度
"""

import numpy as np
import json
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

@dataclass
class InstrumentParams:
    """仪器修正参数"""
    detection_efficiency: float = 1.0
    background_noise: float = 0.0
    systematic_bias: float = 0.0
    resolution_smearing: float = 0.0
    
    # 校准不确定度
    detection_efficiency_sigma: float = 0.0
    background_noise_sigma: float = 0.0
    systematic_bias_sigma: float = 0.0

class InstrumentCorrector:
    """仪器修正器"""
    
    def __init__(self):
        self.calibrated_params = {}  # {experiment_id: InstrumentParams}
        
    def load_instrument_params(self, experiment_setup: dict) -> InstrumentParams:
        """从实验设置中加载仪器参数"""
        corrections = experiment_setup.get("instrument_corrections", {})
        
        return InstrumentParams(
            detection_efficiency=corrections.get("detection_efficiency", 1.0),
            background_noise=corrections.get("background_noise", 0.0),
            systematic_bias=corrections.get("systematic_bias", 0.0),
            resolution_smearing=corrections.get("resolution_smearing", 0.0),
            detection_efficiency_sigma=corrections.get("calibration_uncertainty", {}).get("detection_efficiency_sigma", 0.0),
            background_noise_sigma=corrections.get("calibration_uncertainty", {}).get("background_noise_sigma", 0.0),
            systematic_bias_sigma=corrections.get("calibration_uncertainty", {}).get("systematic_bias_sigma", 0.0)
        )
    
    def apply_instrument_response(self, 
                                theory_prediction: float, 
                                params: InstrumentParams) -> float:
        """
        应用仪器响应函数：O_meas = η * O_pred + B + systematic_bias
        
        Args:
            theory_prediction: 理论预测值
            params: 仪器参数
            
        Returns:
            修正后的预测值
        """
        # 基本线性响应模型
        corrected = (params.detection_efficiency * theory_prediction + 
                    params.background_noise + 
                    params.systematic_bias)
        
        # 可选：添加分辨率展宽效应（这里简化为高斯展宽的均值保持）
        # 实际应用中可能需要更复杂的卷积
        
        return corrected
    
    def calculate_total_uncertainty(self, 
                                   theory_prediction: float,
                                   measurement_sigma: float,
                                   params: InstrumentParams) -> float:
        """
        计算包含仪器校准不确定度的总误差
        
        σ_total² = σ_meas² + (δη * O_pred)² + δB² + δsys²
        """
        instrumental_variance = (
            (params.detection_efficiency_sigma * theory_prediction)**2 +
            params.background_noise_sigma**2 +
            params.systematic_bias_sigma**2
        )
        
        total_sigma = np.sqrt(measurement_sigma**2 + instrumental_variance)
        return total_sigma
    
    def calibrate_two_step(self, 
                          experiment_setup: dict,
                          measurement_data: dict,
                          baseline_theory_prediction: float) -> InstrumentParams:
        """
        两步式校准：先用标准QM（或已知理论）校准仪器参数
        
        Args:
            experiment_setup: 实验设置
            measurement_data: 测量数据
            baseline_theory_prediction: 基准理论（如标准QM）的预测值
            
        Returns:
            校准后的仪器参数
        """
        measured_value = measurement_data["value"]
        
        # 简单线性拟合：measured = η * predicted + B
        # 这里假设systematic_bias已知或为0，只拟合η和B
        
        if "instrument_corrections" in experiment_setup:
            # 如果有先验信息，使用它们作为初值
            params = self.load_instrument_params(experiment_setup)
        else:
            # 否则从数据拟合（这里简化为直接计算）
            # 实际应用中应该用多个标定点进行最小二乘拟合
            detection_efficiency = measured_value / baseline_theory_prediction
            background_noise = 0.0  # 需要独立测量
            
            params = InstrumentParams(
                detection_efficiency=detection_efficiency,
                background_noise=background_noise,
                systematic_bias=0.0,
                detection_efficiency_sigma=0.05,  # 估计值
                background_noise_sigma=0.01
            )
        
        return params
    
    def evaluate_with_correction(self,
                               theory_prediction: float,
                               experiment_setup: dict,
                               measurement_data: dict,
                               instrument_params: Optional[InstrumentParams] = None) -> Dict:
        """
        带仪器修正的理论评估
        
        Returns:
            评估结果字典，包含修正前后的χ²值
        """
        if instrument_params is None:
            instrument_params = self.load_instrument_params(experiment_setup)
        
        # 应用仪器响应
        corrected_prediction = self.apply_instrument_response(theory_prediction, instrument_params)
        
        # 计算总不确定度
        total_sigma = self.calculate_total_uncertainty(
            theory_prediction, 
            measurement_data["sigma"], 
            instrument_params
        )
        
        # 计算χ²值
        measured_value = measurement_data["value"]
        chi2_raw = ((theory_prediction - measured_value) / measurement_data["sigma"])**2
        chi2_corrected = ((corrected_prediction - measured_value) / total_sigma)**2
        
        return {
            "theory_prediction": theory_prediction,
            "corrected_prediction": corrected_prediction,
            "measured_value": measured_value,
            "measurement_sigma": measurement_data["sigma"],
            "total_sigma": total_sigma,
            "chi2_raw": chi2_raw,
            "chi2_corrected": chi2_corrected,
            "instrument_params": {
                "detection_efficiency": instrument_params.detection_efficiency,
                "background_noise": instrument_params.background_noise,
                "systematic_bias": instrument_params.systematic_bias
            }
        }

def example_usage():
    """使用示例"""
    corrector = InstrumentCorrector()
    
    # 模拟实验设置和数据
    experiment_setup = {
        "id": "test_experiment",
        "instrument_corrections": {
            "detection_efficiency": 0.92,
            "background_noise": 0.01,
            "systematic_bias": -0.02,
            "calibration_uncertainty": {
                "detection_efficiency_sigma": 0.03,
                "background_noise_sigma": 0.002
            }
        }
    }
    
    measurement_data = {"value": 0.89, "sigma": 0.02}
    theory_prediction = 0.95
    
    # 评估
    result = corrector.evaluate_with_correction(
        theory_prediction, experiment_setup, measurement_data
    )
    
    print("评估结果:")
    print(f"理论预测值: {result['theory_prediction']:.3f}")
    print(f"修正后预测值: {result['corrected_prediction']:.3f}")
    print(f"测量值: {result['measured_value']:.3f}")
    print(f"原始χ²: {result['chi2_raw']:.3f}")
    print(f"修正后χ²: {result['chi2_corrected']:.3f}")

if __name__ == "__main__":
    example_usage() 