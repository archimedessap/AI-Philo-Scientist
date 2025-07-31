#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
simple_feedback_loop_v2.py - 改进的评估反馈循环
===========================================

使用正确的评估流程，包括实验评估和角色评估
"""

import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List
import argparse

from theory_generation.generation_hub import get_generation_hub
from demo.demo_1 import load_theories_from_sources
from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
from theory_generation.llm_interface import LLMInterface
from utils.logging_config import get_logger

logger = get_logger(__name__)


class ImprovedFeedbackLoop:
    """改进的评估反馈循环"""
    
    def __init__(self, model_source: str = "google", model_name: str = "gemini-2.5-pro"):
        self.model_source = model_source
        self.model_name = model_name
        self.llm_interface = LLMInterface(model_source=model_source, model_name=model_name)
        self.generation_hub = get_generation_hub()
        self.theory_evaluator = TheoryEvaluator(self.llm_interface)
        
    async def evaluate_theory_with_roles(self, theory: Dict, output_dir: Path) -> Dict:
        """对单个理论进行三角色评估"""
        logger.info(f"评估理论: {theory.get('name', '未知')}")
        
        # 准备理论格式
        theory_formatted = {
            "name": theory.get("name", "Unknown Theory"),
            "content": theory.get("content") or self._format_theory_content(theory),
            "philosophy": theory.get("description", ""),
            "core_assumptions": theory.get("core_assumptions", []),
            "mathematical_formalism": theory.get("mathematical_formalism", ""),
            "empirical_predictions": theory.get("empirical_predictions", [])
        }
        
        # 执行三角色评估
        evaluation_results = await self.theory_evaluator.evaluate_theory(theory_formatted)
        
        # 保存评估结果
        result_path = output_dir / f"{theory.get('name', 'theory')}_role_evaluation.json"
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
        
        return evaluation_results
    
    def _format_theory_content(self, theory: Dict) -> str:
        """格式化理论内容"""
        content_parts = []
        
        if theory.get("description"):
            content_parts.append(f"Description: {theory['description']}")
        
        if theory.get("core_assumptions"):
            content_parts.append("\nCore Assumptions:")
            for i, assumption in enumerate(theory["core_assumptions"], 1):
                content_parts.append(f"{i}. {assumption}")
        
        if theory.get("mathematical_formalism"):
            content_parts.append(f"\nMathematical Formalism:\n{theory['mathematical_formalism']}")
        
        if theory.get("empirical_predictions"):
            content_parts.append("\nEmpirical Predictions:")
            for i, prediction in enumerate(theory["empirical_predictions"], 1):
                content_parts.append(f"{i}. {prediction}")
        
        return "\n".join(content_parts)
    
    def extract_comprehensive_feedback(self, evaluation_results: Dict) -> Dict[str, List[str]]:
        """从评估结果中提取全面的反馈"""
        feedback = {
            'strengths': [],
            'weaknesses': [],
            'mathematical_issues': [],
            'experimental_needs': [],
            'conceptual_problems': [],
            'improvement_suggestions': [],
            'questions': []
        }
        
        # 收集三个角色的评估
        roles = ['physicist', 'philosopher', 'mathematician']
        
        for role in roles:
            if role in evaluation_results.get('evaluations', {}):
                role_eval = evaluation_results['evaluations'][role]
                
                # 收集优点
                if 'strengths' in role_eval:
                    feedback['strengths'].extend([f"[{role}] {s}" for s in role_eval['strengths']])
                
                # 收集弱点
                if 'weaknesses' in role_eval:
                    for weakness in role_eval['weaknesses']:
                        feedback['weaknesses'].append(f"[{role}] {weakness}")
                        
                        # 分类弱点
                        weakness_lower = weakness.lower()
                        if any(term in weakness_lower for term in ['数学', 'math', '公式', 'equation', 'formalism']):
                            feedback['mathematical_issues'].append(weakness)
                        elif any(term in weakness_lower for term in ['实验', 'experiment', '预测', 'prediction', 'test']):
                            feedback['experimental_needs'].append(weakness)
                        elif any(term in weakness_lower for term in ['概念', 'concept', '定义', 'definition', '清晰', 'clear']):
                            feedback['conceptual_problems'].append(weakness)
                
                # 收集改进建议
                if 'improvement_suggestions' in role_eval:
                    feedback['improvement_suggestions'].append(f"[{role}] {role_eval['improvement_suggestions']}")
                
                # 收集问题
                if 'questions' in role_eval:
                    feedback['questions'].extend(role_eval['questions'])
        
        # 添加总体评估
        if 'summary' in evaluation_results:
            summary = evaluation_results['summary']
            if 'improvement_directions' in summary:
                feedback['improvement_suggestions'].append(f"[综合建议] {summary['improvement_directions']}")
            if 'potential_value' in summary:
                feedback['strengths'].append(f"[总体评价] {summary['potential_value']}")
        
        return feedback
    
    def generate_comprehensive_improvement_prompt(self, original_theory: Dict, feedback: Dict) -> str:
        """生成综合改进提示"""
        prompt = f"""基于详细的三角色评估反馈，请改进以下量子力学诠释理论。

## 原始理论
名称：{original_theory.get('name', '未知')}
描述：{original_theory.get('description', '无描述')}

## 评估反馈汇总

### 理论优点（需要保持）：
{chr(10).join(f'• {s}' for s in feedback['strengths'][:5])}

### 主要问题：
{chr(10).join(f'• {w}' for w in feedback['weaknesses'][:8])}

### 数学形式化问题：
{chr(10).join(f'• {m}' for m in feedback['mathematical_issues'][:3])}

### 实验预测需求：
{chr(10).join(f'• {e}' for e in feedback['experimental_needs'][:3])}

### 概念澄清需求：
{chr(10).join(f'• {c}' for c in feedback['conceptual_problems'][:3])}

### 具体改进建议：
{chr(10).join(feedback['improvement_suggestions'])}

### 需要回答的关键问题：
{chr(10).join(f'• {q}' for q in feedback['questions'][:5])}

## 改进要求

请生成改进版本的理论，确保：

1. **解决数学问题**：提供完整的数学框架，包括：
   - 基本方程和算符定义
   - 演化规律的数学表达
   - 与标准量子力学的数学关系

2. **增强实验可验证性**：提供：
   - 具体的、可量化的实验预测
   - 与现有实验的对比
   - 区分于其他诠释的关键实验

3. **澄清核心概念**：
   - 所有关键概念的精确定义
   - 概念间的逻辑关系
   - 避免循环定义和模糊表述

4. **保持创新性**：
   - 保留原理论的核心创新点
   - 在改进中不失去独特视角
   - 平衡完善性与原创性

请以JSON格式输出改进后的理论，包含以下字段：
{{
  "name": "理论名称（建议加v2或improved）",
  "description": "改进后的详细描述",
  "core_assumptions": ["核心假设1", "核心假设2", ...],
  "mathematical_formalism": "完整的数学形式化描述",
  "key_equations": ["关键方程1", "关键方程2", ...],
  "empirical_predictions": ["具体预测1", "具体预测2", ...],
  "conceptual_definitions": {{
    "概念1": "精确定义",
    "概念2": "精确定义"
  }},
  "improvements_made": "相对原版本的具体改进",
  "preserved_innovations": "保留的创新要素"
}}
"""
        return prompt
    
    async def generate_improved_theory(self, original_theory: Dict, feedback: Dict) -> Dict:
        """生成改进的理论"""
        prompt = self.generate_comprehensive_improvement_prompt(original_theory, feedback)
        
        logger.info(f"基于综合反馈生成改进版本: {original_theory.get('name', '未知')}")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_interface.query_async(messages)
            
            # 解析响应
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            
            improved_theory = json.loads(response)
            
            # 添加元数据
            improved_theory['generation_metadata'] = {
                'original_theory': original_theory.get('name'),
                'improvement_method': 'comprehensive_feedback',
                'feedback_sources': ['physicist', 'philosopher', 'mathematician'],
                'model': f"{self.model_source}/{self.model_name}"
            }
            
            # 确保格式兼容
            if 'key_equations' in improved_theory and 'mathematical_formalism' in improved_theory:
                improved_theory['mathematical_formalism'] += "\n\nKey Equations:\n" + "\n".join(improved_theory['key_equations'])
            
            return improved_theory
            
        except Exception as e:
            logger.error(f"生成改进理论失败: {e}")
            return None
    
    async def run_feedback_loop(self, theory_path: str, output_dir: str, iterations: int = 1):
        """运行改进的反馈循环"""
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
                
                # 运行角色评估
                evaluation_results = await self.evaluate_theory_with_roles(current_theory, eval_output_dir)
                
                # 2. 提取综合反馈
                feedback = self.extract_comprehensive_feedback(evaluation_results)
                
                # 保存反馈
                feedback_path = output_path / f"iteration_{iteration + 1}" / "extracted_feedback.json"
                with open(feedback_path, 'w', encoding='utf-8') as f:
                    json.dump(feedback, f, ensure_ascii=False, indent=2)
                
                logger.info(f"提取反馈: {len(feedback['weaknesses'])} 个问题, "
                          f"{len(feedback['mathematical_issues'])} 个数学问题, "
                          f"{len(feedback['experimental_needs'])} 个实验需求")
                
                # 3. 生成改进版本
                logger.info("基于反馈生成改进版本...")
                improved_theory = await self.generate_improved_theory(current_theory, feedback)
                
                if improved_theory:
                    # 保存改进版本
                    improved_path = output_path / f"iteration_{iteration + 1}" / "improved_theory.json"
                    with open(improved_path, 'w', encoding='utf-8') as f:
                        json.dump(improved_theory, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"✅ 生成改进版本: {improved_theory.get('name')}")
                    
                    # 如果有更多迭代，评估改进版本
                    if iteration < iterations - 1:
                        logger.info("评估改进版本...")
                        improved_eval = await self.evaluate_theory_with_roles(
                            improved_theory, 
                            output_path / f"iteration_{iteration + 1}" / "improved_evaluation"
                        )
                        
                        # 比较改进前后的分数
                        original_scores = self._extract_scores(evaluation_results)
                        improved_scores = self._extract_scores(improved_eval)
                        
                        logger.info(f"分数变化: {original_scores} → {improved_scores}")
                    
                    # 更新当前理论为改进版本
                    current_theory = improved_theory
                else:
                    logger.error("生成改进版本失败")
                    break
            
            # 保存最终版本
            final_path = output_path / f"{theory_name}_final_improved.json"
            with open(final_path, 'w', encoding='utf-8') as f:
                json.dump(current_theory, f, ensure_ascii=False, indent=2)
            
            # 生成改进报告
            report = self._generate_improvement_report(theory_data, current_theory, iterations)
            report_path = output_path / f"{theory_name}_improvement_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"\n✅ 完成 {theory_name} 的反馈循环优化")
    
    def _extract_scores(self, evaluation_results: Dict) -> Dict[str, float]:
        """提取评估分数"""
        scores = {}
        for role in ['physicist', 'philosopher', 'mathematician']:
            if role in evaluation_results.get('evaluations', {}):
                scores[role] = evaluation_results['evaluations'][role].get('score', 0)
        return scores
    
    def _generate_improvement_report(self, original_theory: Dict, final_theory: Dict, iterations: int) -> str:
        """生成改进报告"""
        report = f"""# 理论改进报告

## 原始理论
- **名称**: {original_theory.get('name', '未知')}
- **描述**: {original_theory.get('description', '无')[:200]}...

## 最终版本
- **名称**: {final_theory.get('name', '未知')}
- **迭代次数**: {iterations}

## 主要改进

### 数学形式化
{final_theory.get('improvements_made', '无具体说明')}

### 实验预测
- 原始: {len(original_theory.get('empirical_predictions', []))} 个预测
- 改进后: {len(final_theory.get('empirical_predictions', []))} 个预测

### 概念定义
{len(final_theory.get('conceptual_definitions', {}))} 个核心概念得到澄清

## 保留的创新要素
{final_theory.get('preserved_innovations', '未说明')}
"""
        return report


async def main():
    parser = argparse.ArgumentParser(description="改进的评估反馈循环")
    parser.add_argument("--theory", required=True, help="理论文件或目录路径")
    parser.add_argument("--output", default="output_feedback_loop_v2", help="输出目录")
    parser.add_argument("--iterations", type=int, default=2, help="反馈循环次数")
    parser.add_argument("--model_source", default="google", help="模型来源")
    parser.add_argument("--model_name", default="gemini-2.5-pro", help="模型名称")
    
    args = parser.parse_args()
    
    # 创建反馈循环实例
    feedback_loop = ImprovedFeedbackLoop(
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