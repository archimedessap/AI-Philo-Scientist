#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论综合评估整合器

整合角色评估和实验评估的结果，生成最终的综合评分和报告。
"""

import os
import sys
import json
import argparse
import asyncio
from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
from theory_experiment.experiment_evaluator import ExperimentEvaluator
from theory_generation.llm_interface import LLMInterface

class TheoryEvaluationIntegrator:
    """理论评估整合器类"""
    
    def __init__(self, role_weight=0.4, experiment_weight=0.6):
        """
        初始化整合器
        
        Args:
            role_weight: 角色评估权重
            experiment_weight: 实验评估权重
        """
        self.role_weight = role_weight
        self.experiment_weight = experiment_weight
        
    async def evaluate_theory(self, theory, llm=None, experiments_path=None):
        """
        综合评估一个理论
        
        Args:
            theory: 理论对象
            llm: LLM接口
            experiments_path: 实验数据路径
            
        Returns:
            综合评估结果
        """
        # 初始化评估器
        if not llm:
            llm = LLMInterface(model_source="deepseek", model_name="deepseek-chat")
        
        role_evaluator = TheoryEvaluator(llm)
        experiment_evaluator = ExperimentEvaluator(experiments_path=experiments_path)
        
        # 独立进行两种评估
        role_result = await role_evaluator.evaluate_theory(theory)
        experiment_result = await experiment_evaluator.evaluate_theory(theory)
        
        # 整合结果
        integrated_result = self._integrate_results(theory, role_result, experiment_result)
        
        return integrated_result
    
    def _integrate_results(self, theory, role_result, experiment_result):
        """整合两种评估结果"""
        # 提取评分
        role_score = role_result.get('average_score', 0)
        experiment_score = experiment_result.get('final_score', 0)
        
        # 计算加权总分
        weighted_score = (
            self.role_weight * role_score + 
            self.experiment_weight * experiment_score
        )
        
        # 创建整合结果
        integrated_result = {
            'theory_name': theory.get('name', '未命名理论'),
            'role_evaluation': role_result,
            'experiment_evaluation': experiment_result,
            'role_score': role_score,
            'experiment_score': experiment_score,
            'role_weight': self.role_weight,
            'experiment_weight': self.experiment_weight,
            'integrated_score': round(weighted_score, 2),
            'evaluation_time': role_result.get('evaluation_time')
        }
        
        return integrated_result

async def run_integrated_evaluation(theory_file, output_file=None, 
                                   role_weight=0.4, experiment_weight=0.6,
                                   experiments_path=None):
    """运行整合评估"""
    # 加载理论
    with open(theory_file, 'r', encoding='utf-8') as f:
        theories = json.load(f)
    
    # 初始化评估器
    integrator = TheoryEvaluationIntegrator(
        role_weight=role_weight,
        experiment_weight=experiment_weight
    )
    llm = LLMInterface(model_source="deepseek", model_name="deepseek-chat")
    
    # 执行评估
    results = []
    if isinstance(theories, list):
        for theory in theories:
            result = await integrator.evaluate_theory(
                theory, 
                llm=llm,
                experiments_path=experiments_path
            )
            results.append(result)
    else:
        result = await integrator.evaluate_theory(
            theories, 
            llm=llm,
            experiments_path=experiments_path
        )
        results = [result]
    
    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="理论综合评估整合器")
    parser.add_argument("--theory_file", required=True, help="理论文件路径")
    parser.add_argument("--output_file", help="输出文件路径")
    parser.add_argument("--role_weight", type=float, default=0.4, help="角色评估权重")
    parser.add_argument("--experiment_weight", type=float, default=0.6, help="实验评估权重")
    parser.add_argument("--experiments_path", help="实验数据文件路径")
    
    args = parser.parse_args()
    asyncio.run(run_integrated_evaluation(
        args.theory_file, 
        args.output_file,
        args.role_weight,
        args.experiment_weight,
        args.experiments_path
    ))

if __name__ == "__main__":
    main()
