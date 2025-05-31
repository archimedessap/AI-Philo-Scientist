#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
实验兼容性验证器

集成已有的ExperimentEvaluator，优先验证理论与现有实验数据的兼容性，
然后再评估预测能力和新实验设计可行性。
"""

import os
import json
from typing import Dict, Any

class ExperimentCompatibilityValidator:
    """验证理论与现有实验数据的兼容性并评估预测能力"""
    
    def __init__(self, llm_interface, experiments_path="theory_experiment/data/experiments.jsonl"):
        """
        初始化实验兼容性验证器
        
        Args:
            llm_interface: LLM接口实例
            experiments_path: 实验数据文件路径
        """
        self.llm = llm_interface
        
        # 导入已有的实验评估器
        try:
            from theory_validation.experimetal_validation.experiment_evaluator import ExperimentEvaluator
            self.experiment_evaluator = ExperimentEvaluator(experiments_path)
            print(f"[INFO] 已成功加载实验评估器")
        except ImportError:
            print(f"[WARN] 无法导入已有的ExperimentEvaluator，将使用有限功能模式")
            self.experiment_evaluator = None
    
    async def validate(self, theory: Dict) -> Dict:
        """
        验证理论与实验数据的兼容性
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 验证结果
        """
        theory_name = theory.get("name", theory.get("theory_name", "未命名理论"))
        print(f"[INFO] 验证理论与实验数据的兼容性: {theory_name}")
        
        # 第一阶段：使用已有实验评估器验证与现有数据的兼容性
        if self.experiment_evaluator:
            compatibility_results = self.experiment_evaluator.evaluate_theory(theory)
            
            # 提取关键指标
            avg_chi2 = compatibility_results.get('avg_chi2', 10.0)  # 默认较高值表示不兼容
            conflicts = compatibility_results.get('conflicts', [])
            
            # 基于χ²值计算基础兼容性得分(0-10)
            # χ²值越小越好，0表示完美匹配
            # 假设χ²阈值为4.0（与ExperimentEvaluator保持一致）
            CHI2_THRESHOLD = 4.0
            compatibility_score = max(0, 10 - (avg_chi2 * 2.5))  # 转换为0-10分
            
            # 计算冲突百分比
            experiment_count = len(compatibility_results.get('detailed_results', []))
            conflict_percentage = (len(conflicts) / experiment_count) * 100 if experiment_count > 0 else 0
            
            # 第二阶段：仅当基本兼容性达标时才进行预测能力评估
            prediction_score = 0.0
            prediction_analysis = ""
            suggested_experiments = []
            
            # 如果基本兼容性分数超过6分（较好兼容），则评估预测能力
            if compatibility_score >= 6.0:
                prediction_results = await self._evaluate_prediction_capability(theory)
                prediction_score = prediction_results.get("prediction_score", 0.0)
                prediction_analysis = prediction_results.get("analysis", "")
                suggested_experiments = prediction_results.get("suggested_experiments", [])
            else:
                prediction_analysis = "理论与现有实验数据的兼容性不足，需要先解决数据匹配问题"
            
            # 综合实验评分，基础兼容性占70%，预测能力占30%
            overall_score = compatibility_score * 0.7 + prediction_score * 0.3
            
            # 建立最终评估结果
            return {
                "overall_score": overall_score,
                "dimension_scores": {
                    "data_compatibility": compatibility_score,
                    "prediction_capability": prediction_score,
                    "experimental_testability": prediction_score * 0.8  # 预测能力的子集
                },
                "compatibility_analysis": {
                    "avg_chi2": avg_chi2,
                    "conflict_count": len(conflicts),
                    "conflict_percentage": conflict_percentage,
                    "conflicts": conflicts
                },
                "prediction_analysis": prediction_analysis,
                "raw_experiment_results": compatibility_results,
                "suggested_experiments": suggested_experiments,
                "recommendations": self._generate_recommendations(
                    theory, compatibility_score, prediction_score, conflicts
                )
            }
        else:
            # 如果没有实验评估器，使用LLM进行基于知识的评估
            return await self._fallback_validation(theory)
    
    async def _evaluate_prediction_capability(self, theory: Dict) -> Dict:
        """
        评估理论的预测能力和可测试性
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 预测能力评估结果
        """
        # 提取核心要素
        theory_name = theory.get("name", theory.get("theory_name", "未命名理论"))
        
        # 提取理论预测（如果存在）
        predictions = theory.get("experimental_predictions", [])
        if isinstance(predictions, list):
            predictions_text = "\n".join([f"- {p}" for p in predictions])
        else:
            predictions_text = str(predictions)
        
        # 构建提示，重点评估预测能力和可测试性
        prompt = f"""
        作为量子物理实验专家，请评估以下理论的预测能力和实验可测试性。该理论已通过与现有实验数据的基本兼容性验证。

        理论名称: {theory_name}

        理论提出的实验预测:
        {predictions_text}

        请深入分析该理论的实验预测能力，特别关注:
        1. 预测的具体性和精确性: 预测是否足够具体，可被精确测量
        2. 预测的独特性: 是否与标准量子力学的预测有明显区别
        3. 实验设计可行性: 是否可以设计具体实验来验证这些预测
        4. 成本和技术可行性: 验证预测所需的技术和资源复杂度

        请返回JSON格式的评估结果:
        {{
          "prediction_score": 预测能力评分(0-10),
          "analysis": "对预测能力的总体分析",
          "predictions_assessment": [
            {{
              "prediction": "从理论中提取的具体预测",
              "specificity": 具体性评分(0-10),
              "uniqueness": 独特性评分(0-10),
              "testability": 可测试性评分(0-10)
            }}
          ],
          "suggested_experiments": [
            {{
              "name": "建议实验名称",
              "description": "实验设计简述",
              "expected_outcome": "根据该理论预期的结果",
              "distinguishing_power": "与标准量子力学预测的区别性",
              "feasibility": 技术可行性评分(0-10)
            }}
          ]
        }}

        如果理论没有提供足够的预测信息，请基于理论内容推导可能的预测。评分标准:
        - 10分: 多个精确、可测量、独特的预测，并提供可行实验方案
        - 7-9分: 有具体预测，可设计实验验证，具有一定独特性
        - 4-6分: 预测不够精确或独特性有限
        - 1-3分: 预测模糊或难以测试
        - 0分: 无有意义预测
        """
        
        # 调用LLM评估
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # 解析结果
        result = self.llm.extract_json(response)
        
        if not result:
            print(f"[WARN] 无法解析LLM响应为有效JSON，将使用默认结果")
            # 创建默认结果
            result = {
                "prediction_score": 5.0,
                "analysis": "无法评估理论的预测能力",
                "predictions_assessment": [],
                "suggested_experiments": []
            }
        
        return result
    
    def _generate_recommendations(self, theory, compatibility_score, prediction_score, conflicts):
        """生成改进建议"""
        recommendations = []
        
        # 基于兼容性分数的建议
        if compatibility_score < 5.0:
            recommendations.append("理论需要重大修改以符合现有实验观测")
            if conflicts:
                recommendations.append(f"特别关注与以下实验的冲突: {', '.join(conflicts[:3])}")
        elif compatibility_score < 7.0:
            recommendations.append("理论需要调整以更好地匹配现有实验数据")
        
        # 基于预测能力的建议
        if prediction_score < 4.0:
            recommendations.append("提高理论的预测能力，提出更具体、可验证的预测")
        elif prediction_score < 7.0:
            recommendations.append("增强预测的独特性，使其与标准量子力学有明显区别")
        
        # 如果两项评分差异大
        if abs(compatibility_score - prediction_score) > 3.0:
            if compatibility_score > prediction_score:
                recommendations.append("理论与数据兼容性良好，但需要增强预测能力")
            else:
                recommendations.append("理论有良好的预测，但需要更好地解释现有实验数据")
        
        return recommendations
    
    async def _fallback_validation(self, theory: Dict) -> Dict:
        """
        当无法使用已有实验评估器时的备用验证方法
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 备用验证结果
        """
        # 构建提示
        prompt = f"""
        作为量子物理实验专家，请评估以下理论与现有量子力学实验数据的兼容性，以及其预测能力。

        理论名称: {theory.get('name', '未命名理论')}
        理论描述: {theory.get('core_principles', '')}
        
        请首先分析该理论与以下关键量子实验的兼容性:
        1. 双缝实验
        2. EPR悖论和Bell不等式实验
        3. 量子擦除实验
        4. 延迟选择实验
        5. 量子隧穿实验
        
        然后分析理论的预测能力和可测试性。
        
        以JSON格式返回评估结果:
        {{
          "overall_score": 总体评分(0-10),
          "dimension_scores": {{
            "data_compatibility": 与现有数据兼容性评分(0-10),
            "prediction_capability": 预测能力评分(0-10),
            "experimental_testability": 可测试性评分(0-10)
          }},
          "compatibility_analysis": "与关键实验兼容性的分析",
          "prediction_analysis": "预测能力分析",
          "suggested_experiments": [
            {{
              "name": "建议实验名称",
              "description": "实验设计简述",
              "expected_outcome": "预期结果"
            }}
          ],
          "recommendations": [
            "改进建议1",
            "改进建议2"
          ]
        }}
        
        请严格按照实验物理学的标准进行评估，优先考虑与现有数据的兼容性，然后是预测能力。
        """
        
        # 调用LLM进行评估
        response = await self.llm.query_async(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        # 解析结果
        result = self.llm.extract_json(response)
        
        if not result:
            print(f"[WARN] 无法解析LLM响应为有效JSON，将使用默认结果")
            # 创建默认结果
            result = {
                "overall_score": 5.0,
                "dimension_scores": {
                    "data_compatibility": 5.0,
                    "prediction_capability": 5.0,
                    "experimental_testability": 5.0
                },
                "compatibility_analysis": "无法评估与现有数据的兼容性",
                "prediction_analysis": "无法评估预测能力",
                "suggested_experiments": [],
                "recommendations": ["建议使用实验评估器进行更准确的评估"]
            }
        
        return result
