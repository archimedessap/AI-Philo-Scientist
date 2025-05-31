#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论验证框架

对生成的量子理论假说进行多角度验证，
评估理论的一致性、解释力、预测能力和实验可行性。
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional
import asyncio

class TheoryValidationFramework:
    """对量子理论假说进行多角度验证的框架"""
    
    def __init__(self, llm_interface):
        """
        初始化验证框架
        
        Args:
            llm_interface: LLM接口实例
        """
        self.llm = llm_interface
        self.validators = []  # 存储不同的验证器
        self.validation_results = {}  # 存储验证结果
    
    def register_validator(self, validator):
        """
        注册验证器
        
        Args:
            validator: 验证器实例
        """
        self.validators.append(validator)
        print(f"[INFO] 注册验证器: {validator.__class__.__name__}")
    
    async def validate_theory(self, theory: Dict) -> Dict:
        """
        对单个理论进行全面验证
        
        Args:
            theory: 理论数据
            
        Returns:
            Dict: 验证结果
        """
        theory_id = theory.get("id", "unknown")
        theory_name = theory.get("name", theory.get("theory_name", "未命名理论"))
        
        print(f"[INFO] 开始验证理论: {theory_name} (ID: {theory_id})")
        
        # 验证结果结构
        validation_result = {
            "theory_id": theory_id,
            "theory_name": theory_name,
            "overall_score": 0.0,
            "dimension_scores": {},
            "validator_details": {},
            "recommendations": []
        }
        
        # 运行每个验证器
        total_score = 0.0
        
        for validator in self.validators:
            validator_name = validator.__class__.__name__
            print(f"[INFO] 运行验证器: {validator_name}")
            
            try:
                # 运行验证
                result = await validator.validate(theory)
                
                # 汇总结果
                validation_result["validator_details"][validator_name] = result
                validation_result["dimension_scores"].update(result.get("dimension_scores", {}))
                
                # 收集推荐
                if "recommendations" in result:
                    validation_result["recommendations"].extend(result["recommendations"])
                
                # 计算该验证器的平均分
                validator_score = result.get("overall_score", 0.0)
                total_score += validator_score
                
                print(f"[INFO] 验证器 {validator_name} 完成，分数: {validator_score:.2f}/10")
            except Exception as e:
                print(f"[ERROR] 验证器 {validator_name} 失败: {str(e)}")
                validation_result["validator_details"][validator_name] = {
                    "error": str(e),
                    "overall_score": 0.0
                }
        
        # 计算总体分数 (平均)
        if self.validators:
            validation_result["overall_score"] = total_score / len(self.validators)
        
        # 存储结果
        self.validation_results[theory_id] = validation_result
        
        print(f"[INFO] 理论 {theory_name} 验证完成，总体分数: {validation_result['overall_score']:.2f}/10")
        return validation_result
    
    async def validate_multiple_theories(self, theories: List[Dict]) -> Dict:
        """
        对多个理论进行验证并比较
        
        Args:
            theories: 理论列表
            
        Returns:
            Dict: 包含所有验证结果和比较的字典
        """
        results = {}
        
        for theory in theories:
            result = await self.validate_theory(theory)
            theory_id = result["theory_id"]
            results[theory_id] = result
        
        # 执行理论间比较
        comparison = await self.compare_validation_results(results)
        
        return {
            "individual_results": results,
            "comparison": comparison
        }
    
    async def compare_validation_results(self, results: Dict) -> Dict:
        """
        比较多个理论的验证结果
        
        Args:
            results: 验证结果字典 {theory_id: result}
            
        Returns:
            Dict: 比较结果
        """
        # 按总体分数排序
        sorted_theories = sorted(
            results.items(),
            key=lambda x: x[1]["overall_score"],
            reverse=True
        )
        
        # 提取维度评分进行比较
        dimension_comparisons = {}
        all_dimensions = set()
        
        # 收集所有维度
        for _, result in results.items():
            all_dimensions.update(result["dimension_scores"].keys())
        
        # 对每个维度进行比较
        for dimension in all_dimensions:
            dimension_scores = []
            
            for theory_id, result in results.items():
                theory_name = result["theory_name"]
                score = result["dimension_scores"].get(dimension, 0.0)
                dimension_scores.append({
                    "theory_id": theory_id,
                    "theory_name": theory_name,
                    "score": score
                })
            
            # 按分数排序
            dimension_scores.sort(key=lambda x: x["score"], reverse=True)
            dimension_comparisons[dimension] = dimension_scores
        
        return {
            "ranked_theories": [
                {
                    "theory_id": theory_id,
                    "theory_name": result["theory_name"],
                    "overall_score": result["overall_score"],
                    "strongest_dimensions": self._get_strongest_dimensions(result, top=3),
                    "weakest_dimensions": self._get_weakest_dimensions(result, bottom=3)
                }
                for theory_id, result in sorted_theories
            ],
            "dimension_comparisons": dimension_comparisons,
            "best_theory_by_dimension": self._get_best_theory_by_dimension(results)
        }
    
    def _get_strongest_dimensions(self, result: Dict, top: int = 3) -> List[Dict]:
        """获取理论的最强维度"""
        dimensions = [(dim, score) for dim, score in result["dimension_scores"].items()]
        dimensions.sort(key=lambda x: x[1], reverse=True)
        
        return [{"dimension": dim, "score": score} for dim, score in dimensions[:top]]
    
    def _get_weakest_dimensions(self, result: Dict, bottom: int = 3) -> List[Dict]:
        """获取理论的最弱维度"""
        dimensions = [(dim, score) for dim, score in result["dimension_scores"].items()]
        dimensions.sort(key=lambda x: x[1])
        
        return [{"dimension": dim, "score": score} for dim, score in dimensions[:bottom]]
    
    def _get_best_theory_by_dimension(self, results: Dict) -> Dict:
        """获取每个维度的最佳理论"""
        best_by_dimension = {}
        all_dimensions = set()
        
        # 收集所有维度
        for _, result in results.items():
            all_dimensions.update(result["dimension_scores"].keys())
        
        # 对每个维度找出最佳理论
        for dimension in all_dimensions:
            best_theory = None
            best_score = -1
            
            for theory_id, result in results.items():
                score = result["dimension_scores"].get(dimension, 0.0)
                if score > best_score:
                    best_score = score
                    best_theory = {
                        "theory_id": theory_id,
                        "theory_name": result["theory_name"],
                        "score": score
                    }
            
            if best_theory:
                best_by_dimension[dimension] = best_theory
        
        return best_by_dimension
    
    def save_results(self, output_dir: str) -> None:
        """
        保存验证结果
        
        Args:
            output_dir: 输出目录
        """
        # 确保目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存所有结果
        results_file = os.path.join(output_dir, "validation_results.json")
        try:
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(self.validation_results, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 验证结果已保存到: {results_file}")
        except Exception as e:
            print(f"[ERROR] 保存验证结果失败: {str(e)}")
