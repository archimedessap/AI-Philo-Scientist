#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论综合评估整合器

按照先实验筛选、后角色评估的顺序生成综合结果：
• 若理论改动了量子力学的数学结构，必须先通过实验评估且拿到满分；
• 仅在通过实验筛选后，才执行角色评估，并以角色评估成绩作为最终分数；
• 纯解释型（无数学改动）的理论直接走角色评估。
"""

import os
import json
import argparse
import asyncio
from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
from theory_experiment.experiment_evaluator import ExperimentEvaluator
from theory_generation.llm_interface import LLMInterface

class TheoryEvaluationIntegrator:
    """理论评估整合器类"""

    def __init__(self, experiment_pass_score: float = 10.0, pass_tolerance: float = 1e-6):
        """初始化整合器.

        Args:
            experiment_pass_score: 实验评估需要达到的分数阈值（默认满分10分）。
            pass_tolerance: 浮点比较的容差，避免舍入误判。
        """
        self.experiment_pass_score = experiment_pass_score
        self.pass_tolerance = pass_tolerance

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

        requires_experiment = role_evaluator.has_math_modification(theory)
        experiment_result = None
        experiment_passed = True
        evaluation_route = 'role_only'

        # 数学有改动 → 先做实验筛选
        if requires_experiment:
            evaluation_route = 'experiment_gate'
            experiment_result = await experiment_evaluator.evaluate_theory(theory)
            experiment_score = experiment_result.get('final_score') if isinstance(experiment_result, dict) else None
            if experiment_score is None:
                experiment_passed = False
            else:
                experiment_passed = experiment_score + self.pass_tolerance >= self.experiment_pass_score

            if not experiment_passed:
                return self._integrate_results(
                    theory=theory,
                    role_result=None,
                    experiment_result=experiment_result,
                    requires_experiment=True,
                    experiment_passed=False,
                    evaluation_route=evaluation_route
                )

            evaluation_route = 'experiment_then_role'

        # 通过实验筛选或无数学改动 → 角色评估决定最终得分
        role_result = await role_evaluator.evaluate_theory(theory)

        return self._integrate_results(
            theory=theory,
            role_result=role_result,
            experiment_result=experiment_result,
            requires_experiment=requires_experiment,
            experiment_passed=experiment_passed,
            evaluation_route=evaluation_route
        )

    def _integrate_results(self, *, theory, role_result, experiment_result,
                           requires_experiment: bool, experiment_passed: bool,
                           evaluation_route: str):
        """整合评估结果并生成报告。"""
        role_score = None
        if isinstance(role_result, dict):
            role_score = role_result.get('overall_score')
            if role_score is not None:
                role_score = float(role_score)

        experiment_score = None
        if isinstance(experiment_result, dict):
            experiment_score = experiment_result.get('final_score')
            if experiment_score is not None:
                experiment_score = float(experiment_score)

        final_score = round(role_score, 2) if role_score is not None else 0.0

        status = 'evaluated'
        if requires_experiment and not experiment_passed:
            status = 'filtered_by_experiment'
            final_score = 0.0

        return {
            'theory_name': theory.get('name', '未命名理论'),
            'theory_id': theory.get('id'),
            'math_modification': requires_experiment,
            'experiment_required': requires_experiment,
            'experiment_passed': experiment_passed if requires_experiment else None,
            'experiment_threshold': self.experiment_pass_score,
            'experiment_evaluation': experiment_result,
            'role_evaluation': role_result,
            'role_score': role_score,
            'experiment_score': experiment_score,
            'role_weight': 1.0,
            'experiment_weight': 0.0,
            'evaluation_route': evaluation_route,
            'status': status,
            'final_score': final_score,
            'integrated_score': final_score,
            'evaluation_time': role_result.get('evaluation_time') if isinstance(role_result, dict) else None
        }

async def run_integrated_evaluation(theory_file, output_file=None,
                                   experiments_path=None,
                                   experiment_pass_score: float = 10.0,
                                   pass_tolerance: float = 1e-6):
    """运行整合评估"""
    # 加载理论
    with open(theory_file, 'r', encoding='utf-8') as f:
        theories = json.load(f)
    
    # 初始化评估器
    integrator = TheoryEvaluationIntegrator(
        experiment_pass_score=experiment_pass_score,
        pass_tolerance=pass_tolerance
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
    parser.add_argument("--experiments_path", help="实验数据文件路径")
    parser.add_argument("--experiment_pass_score", type=float, default=10.0,
                        help="实验评估筛选所需最低分数（默认10分）")
    parser.add_argument("--pass_tolerance", type=float, default=1e-6,
                        help="实验分数比较的容差，用于避免浮点误差导致的误判")

    args = parser.parse_args()
    asyncio.run(run_integrated_evaluation(
        args.theory_file, 
        args.output_file,
        experiments_path=args.experiments_path,
        experiment_pass_score=args.experiment_pass_score,
        pass_tolerance=args.pass_tolerance
    ))

if __name__ == "__main__":
    main()
