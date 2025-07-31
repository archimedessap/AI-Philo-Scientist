#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_feedback_loop.py - 简单的评估反馈循环
=========================================

将评估结果反馈给理论生成器，生成改进的理论版本
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import argparse

from theory_generation.generation_hub import get_generation_hub
from demo.demo_1 import load_theories_from_sources
from demo.auto_role_evaluation import run_role_evaluation_for_theories
from theory_generation.llm_interface import LLMInterface
from utils.logging_config import get_logger

logger = get_logger(__name__)


class SimpleFeedbackLoop:
    """简单的评估反馈循环"""
    
    def __init__(self, model_source: str = "google", model_name: str = "gemini-2.5-pro"):
        self.model_source = model_source
        self.model_name = model_name
        self.llm_interface = LLMInterface(model_source=model_source, model_name=model_name)
        self.generation_hub = get_generation_hub()
        
    def extract_improvement_suggestions(self, evaluation_results: Dict) -> Dict[str, List[str]]:
        """从评估结果中提取改进建议"""
        suggestions = {
            'weaknesses': [],
            'improvement_suggestions': [],
            'questions': []
        }
        
        # 从三个角色的评估中提取信息
        for role in ['physicist', 'philosopher', 'mathematician']:
            if role in evaluation_results:
                role_eval = evaluation_results[role]
                
                # 收集弱点
                if 'weaknesses' in role_eval:
                    suggestions['weaknesses'].extend(role_eval['weaknesses'])
                
                # 收集改进建议
                if 'improvement_suggestions' in role_eval:
                    suggestions['improvement_suggestions'].append(
                        f"[{role}] {role_eval['improvement_suggestions']}"
                    )
                
                # 收集问题
                if 'questions' in role_eval:
                    suggestions['questions'].extend(role_eval['questions'])
        
        # 添加总体建议
        if 'summary' in evaluation_results and 'improvement_directions' in evaluation_results['summary']:
            suggestions['improvement_suggestions'].append(
                f"[综合] {evaluation_results['summary']['improvement_directions']}"
            )
        
        return suggestions
    
    def generate_improvement_prompt(self, original_theory: Dict, suggestions: Dict) -> str:
        """生成改进提示"""
        prompt = f"""基于以下评估反馈，请改进这个量子力学诠释理论。

原始理论：
名称：{original_theory.get('name', '未知')}
描述：{original_theory.get('description', '无描述')}

评估发现的主要问题：
{chr(10).join(f'- {w}' for w in suggestions['weaknesses'][:5])}

具体改进建议：
{chr(10).join(suggestions['improvement_suggestions'])}

需要回答的关键问题：
{chr(10).join(f'- {q}' for q in suggestions['questions'][:3])}

请生成改进版本的理论，要求：
1. 解决上述指出的主要问题
2. 采纳合理的改进建议
3. 回答提出的关键问题
4. 保持原理论的核心创新点
5. 增强数学形式化和实验可验证性

请以JSON格式输出改进后的理论，包含以下字段：
- name: 理论名称（可以添加版本号如v2）
- description: 详细描述
- core_assumptions: 核心假设列表
- mathematical_formalism: 数学形式化
- empirical_predictions: 实验预测
- improvements_made: 相对原版本的改进说明
"""
        return prompt
    
    async def generate_improved_theory(self, original_theory: Dict, suggestions: Dict) -> Dict:
        """生成改进的理论"""
        prompt = self.generate_improvement_prompt(original_theory, suggestions)
        
        logger.info(f"生成改进版本: {original_theory.get('name', '未知')}")
        
        try:
            response = await self.llm_interface.generate(prompt)
            
            # 解析响应
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            
            improved_theory = json.loads(response)
            
            # 添加元数据
            improved_theory['generation_metadata'] = {
                'original_theory': original_theory.get('name'),
                'improvement_based_on': 'evaluation_feedback',
                'model': f"{self.model_source}/{self.model_name}"
            }
            
            return improved_theory
            
        except Exception as e:
            logger.error(f"生成改进理论失败: {e}")
            return None
    
    async def run_feedback_loop(self, theory_path: str, output_dir: str, iterations: int = 1):
        """运行反馈循环"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 加载原始理论
        theories = load_theories_from_sources(theory_path)
        if not theories:
            logger.error("未找到理论文件")
            return
        
        # 处理每个理论
        for theory_name, (theory_data, _) in theories.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"处理理论: {theory_name}")
            
            current_theory = theory_data
            
            for iteration in range(iterations):
                logger.info(f"\n--- 迭代 {iteration + 1}/{iterations} ---")
                
                # 1. 评估当前理论
                logger.info("执行三角色评估...")
                eval_output_dir = output_path / f"iteration_{iteration + 1}" / "evaluation"
                eval_output_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存当前理论以供评估
                current_theory_path = eval_output_dir / f"{theory_name}_v{iteration + 1}.json"
                with open(current_theory_path, 'w', encoding='utf-8') as f:
                    json.dump(current_theory, f, ensure_ascii=False, indent=2)
                
                # 运行评估
                await run_role_evaluation_for_theories(
                    str(eval_output_dir),
                    model_source=self.model_source,
                    model_name=self.model_name,
                    output_dir=str(eval_output_dir / "results")
                )
                
                # 2. 读取评估结果
                eval_result_path = eval_output_dir / "results" / f"{current_theory.get('name', theory_name)}_role_evaluation.json"
                if not eval_result_path.exists():
                    logger.error(f"未找到评估结果: {eval_result_path}")
                    continue
                
                with open(eval_result_path, 'r', encoding='utf-8') as f:
                    evaluation_results = json.load(f)
                
                # 3. 提取改进建议
                suggestions = self.extract_improvement_suggestions(evaluation_results)
                
                # 保存建议
                suggestions_path = output_path / f"iteration_{iteration + 1}" / "suggestions.json"
                with open(suggestions_path, 'w', encoding='utf-8') as f:
                    json.dump(suggestions, f, ensure_ascii=False, indent=2)
                
                logger.info(f"提取了 {len(suggestions['weaknesses'])} 个问题和 {len(suggestions['improvement_suggestions'])} 条建议")
                
                # 4. 生成改进版本
                logger.info("生成改进版本...")
                improved_theory = await self.generate_improved_theory(current_theory, suggestions)
                
                if improved_theory:
                    # 保存改进版本
                    improved_path = output_path / f"iteration_{iteration + 1}" / "improved_theory.json"
                    with open(improved_path, 'w', encoding='utf-8') as f:
                        json.dump(improved_theory, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"✅ 生成改进版本: {improved_theory.get('name')}")
                    
                    # 更新当前理论为改进版本
                    current_theory = improved_theory
                else:
                    logger.error("生成改进版本失败")
                    break
            
            # 保存最终版本
            final_path = output_path / f"{theory_name}_final.json"
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(current_theory, f, ensure_ascii=False, indent=2)
            
            logger.info(f"\n✅ 完成 {theory_name} 的反馈循环优化")


async def main():
    parser = argparse.ArgumentParser(description="简单评估反馈循环")
    parser.add_argument("--theory", required=True, help="理论文件或目录路径")
    parser.add_argument("--output", default="output_feedback_loop", help="输出目录")
    parser.add_argument("--iterations", type=int, default=2, help="反馈循环次数")
    parser.add_argument("--model_source", default="google", help="模型来源")
    parser.add_argument("--model_name", default="gemini-2.5-pro", help="模型名称")
    
    args = parser.parse_args()
    
    # 创建反馈循环实例
    feedback_loop = SimpleFeedbackLoop(
        model_source=args.model_source,
        model_name=args.model_name
    )
    
    # 运行反馈循环
    await feedback_loop.run_feedback_loop(
        theory_path=args.theory,
        output_dir=args.output,
        iterations=args.iterations
    )


if __name__ == "__main__":
    asyncio.run(main())