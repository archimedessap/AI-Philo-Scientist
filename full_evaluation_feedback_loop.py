#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_evaluation_feedback_loop.py - 完整评估反馈循环
==============================================

集成实验评估和角色评估的完整反馈循环系统
"""

import json
import asyncio
import subprocess
from pathlib import Path
from typing import Dict, Any, List, Optional
import argparse

from theory_generation.generation_hub import get_generation_hub
from demo.demo_1 import load_theories_from_sources
from theory_generation.llm_interface import LLMInterface
from utils.logging_config import get_logger

logger = get_logger(__name__)


class FullEvaluationFeedbackLoop:
    """完整评估反馈循环（包括实验评估和角色评估）"""
    
    def __init__(self, 
                 model_source: str = "google", 
                 model_name: str = "gemini-2.5-pro",
                 experiment_dir: str = "data/experiments"):
        self.model_source = model_source
        self.model_name = model_name
        self.experiment_dir = experiment_dir
        self.llm_interface = LLMInterface(model_source=model_source, model_name=model_name)
        self.generation_hub = get_generation_hub()
        
    def run_full_evaluation(self, theories_dir: str, output_dir: str, 
                          include_experiments: bool = True,
                          role_threshold: float = 0.3) -> bool:
        """运行完整评估（实验+角色）"""
        cmd = [
            "python", "demo/demo_1.py",
            "--theory_path", theories_dir,
            "--output_dir", output_dir,
            "--experiment_dir", self.experiment_dir,  # 总是需要这个参数
            "--model_source", self.model_source,
            "--model_name", self.model_name,
            "--run_role_evaluation",
            "--role_success_threshold", str(role_threshold)
        ]
        
        if not include_experiments:
            # 如果不包含实验，设置最大实验数为0
            cmd.extend(["--max_experiments", "0"])
        
        logger.info(f"运行完整评估: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                logger.error(f"评估失败: {result.stderr}")
                return False
            logger.info("评估完成")
            return True
        except Exception as e:
            logger.error(f"运行评估命令失败: {e}")
            return False
    
    def extract_comprehensive_feedback(self, eval_output_dir: Path) -> Dict:
        """从评估结果中提取综合反馈"""
        feedback = {
            'experimental_results': {},
            'role_evaluations': {},
            'combined_insights': {
                'strengths': [],
                'weaknesses': [],
                'mathematical_gaps': [],
                'experimental_needs': [],
                'conceptual_issues': [],
                'improvement_suggestions': []
            }
        }
        
        # 1. 读取实验评估结果
        exp_summary_path = eval_output_dir / "final_evaluation_summary.json"
        if exp_summary_path.exists():
            with open(exp_summary_path, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
                
            # 提取实验结果
            for theory_name, theory_data in exp_data.items():
                if 'experiments' in theory_data:
                    feedback['experimental_results'][theory_name] = {
                        'success_rate': theory_data.get('success_rate', 0),
                        'successful_experiments': [
                            exp for exp, result in theory_data['experiments'].items()
                            if result.get('match_level', 0) >= 0.8
                        ],
                        'failed_experiments': [
                            exp for exp, result in theory_data['experiments'].items()
                            if result.get('match_level', 0) < 0.8
                        ]
                    }
                    
                    # 基于实验失败添加改进需求
                    if feedback['experimental_results'][theory_name]['failed_experiments']:
                        feedback['combined_insights']['experimental_needs'].append(
                            f"改进对以下实验的预测: {', '.join(feedback['experimental_results'][theory_name]['failed_experiments'])}"
                        )
        
        # 2. 读取角色评估结果
        role_eval_dir = eval_output_dir / "role_evaluations"
        if role_eval_dir.exists():
            for eval_file in role_eval_dir.glob("*_role_evaluation.json"):
                theory_name = eval_file.stem.replace('_role_evaluation', '')
                
                with open(eval_file, 'r', encoding='utf-8') as f:
                    role_data = json.load(f)
                
                feedback['role_evaluations'][theory_name] = role_data
                
                # 提取角色反馈
                if 'evaluations' in role_data:
                    for role, eval_data in role_data['evaluations'].items():
                        # 收集优点
                        if 'strengths' in eval_data:
                            feedback['combined_insights']['strengths'].extend(
                                [f"[{role}] {s}" for s in eval_data['strengths']]
                            )
                        
                        # 收集缺点并分类
                        if 'weaknesses' in eval_data:
                            for weakness in eval_data['weaknesses']:
                                feedback['combined_insights']['weaknesses'].append(f"[{role}] {weakness}")
                                
                                # 分类缺点
                                weakness_lower = weakness.lower()
                                if any(term in weakness_lower for term in ['数学', 'math', '公式', 'equation']):
                                    feedback['combined_insights']['mathematical_gaps'].append(weakness)
                                elif any(term in weakness_lower for term in ['概念', 'concept', '定义', 'definition']):
                                    feedback['combined_insights']['conceptual_issues'].append(weakness)
                        
                        # 收集改进建议
                        if 'improvement_suggestions' in eval_data:
                            feedback['combined_insights']['improvement_suggestions'].append(
                                f"[{role}] {eval_data['improvement_suggestions']}"
                            )
        
        # 3. 读取综合排名（如果有）
        combined_ranking_path = eval_output_dir / "combined_rankings.json"
        if combined_ranking_path.exists():
            with open(combined_ranking_path, 'r', encoding='utf-8') as f:
                ranking_data = json.load(f)
                feedback['combined_ranking'] = ranking_data
        
        return feedback
    
    def generate_feedback_informed_improvement_prompt(self, 
                                                    original_theory: Dict, 
                                                    feedback: Dict) -> str:
        """基于综合反馈生成改进提示"""
        # 获取理论名称
        theory_name = original_theory.get('name', '未知理论')
        
        # 获取该理论的具体反馈
        exp_results = feedback['experimental_results'].get(theory_name, {})
        role_eval = feedback['role_evaluations'].get(theory_name, {})
        
        prompt = f"""基于实验评估和专家角色评估的综合反馈，请改进以下量子力学诠释理论。

## 原始理论
名称：{theory_name}
描述：{original_theory.get('description', '无描述')}

## 评估结果汇总

### 实验评估结果
- 成功率：{exp_results.get('success_rate', 0)*100:.1f}%
- 成功的实验：{', '.join(exp_results.get('successful_experiments', ['无']))}
- 失败的实验：{', '.join(exp_results.get('failed_experiments', ['无']))}

### 专家评估摘要
"""
        
        # 添加各角色的评分
        if role_eval and 'evaluations' in role_eval:
            for role in ['physicist', 'philosopher', 'mathematician']:
                if role in role_eval['evaluations']:
                    score = role_eval['evaluations'][role].get('score', 0)
                    prompt += f"- {role}评分：{score}/10\n"
        
        prompt += f"""
### 综合反馈

#### 理论优点（需保持）：
{chr(10).join(f'• {s}' for s in feedback['combined_insights']['strengths'][:5])}

#### 主要问题：
{chr(10).join(f'• {w}' for w in feedback['combined_insights']['weaknesses'][:8])}

#### 数学形式化缺陷：
{chr(10).join(f'• {m}' for m in feedback['combined_insights']['mathematical_gaps'][:3]) or '• 无特定数学问题'}

#### 实验预测改进需求：
{chr(10).join(f'• {e}' for e in feedback['combined_insights']['experimental_needs'][:3]) or '• 无特定实验需求'}

#### 概念澄清需求：
{chr(10).join(f'• {c}' for c in feedback['combined_insights']['conceptual_issues'][:3]) or '• 无特定概念问题'}

#### 具体改进建议：
{chr(10).join(feedback['combined_insights']['improvement_suggestions'][:5])}

## 改进要求

请生成显著改进的理论版本，必须：

1. **提升实验兼容性**：
   - 特别关注失败的实验，提供更准确的预测
   - 保持并强化成功实验的解释
   - 添加定量预测公式

2. **增强数学严格性**：
   - 提供完整的数学框架
   - 定义所有算符和状态空间
   - 建立与标准量子力学的明确关系

3. **澄清核心概念**：
   - 为所有关键术语提供精确定义
   - 消除循环定义和模糊表述
   - 建立清晰的概念层次

4. **保持创新特色**：
   - 不要失去原理论的独特视角
   - 在改进中保持核心创新
   - 平衡完善与原创

5. **提高可验证性**：
   - 提出至少3个具体的、可量化的预测
   - 说明如何通过实验区分于其他诠释
   - 包含可能的反驳条件

请以JSON格式输出改进后的理论：
{{
  "name": "{theory_name} (Enhanced)",
  "description": "改进后的详细描述",
  "core_assumptions": ["假设1", "假设2", ...],
  "mathematical_formalism": "完整的数学描述，包括基本方程",
  "key_equations": {{
    "state_evolution": "状态演化方程",
    "measurement": "测量过程方程",
    "predictions": "预测计算公式"
  }},
  "empirical_predictions": [
    {{
      "experiment": "实验名称",
      "prediction": "具体预测",
      "quantitative_value": "定量值"
    }}
  ],
  "conceptual_clarifications": {{
    "term1": "精确定义",
    "term2": "精确定义"
  }},
  "improvements_summary": "相对原版本的关键改进",
  "experimental_compatibility": {{
    "double_slit": "解释",
    "bell_test": "解释",
    "delayed_choice": "解释"
  }}
}}
"""
        return prompt
    
    async def generate_improved_theory(self, original_theory: Dict, feedback: Dict) -> Dict:
        """基于综合反馈生成改进理论"""
        prompt = self.generate_feedback_informed_improvement_prompt(original_theory, feedback)
        
        logger.info(f"基于完整评估反馈生成改进版本: {original_theory.get('name', '未知')}")
        
        try:
            messages = [{"role": "user", "content": prompt}]
            response = await self.llm_interface.query_async(messages)
            
            # 解析响应
            if response.startswith("```json") and response.endswith("```"):
                response = response[7:-3].strip()
            
            improved_theory = json.loads(response)
            
            # 确保格式兼容性
            if 'key_equations' in improved_theory:
                # 将key_equations整合到mathematical_formalism
                equations_text = "\n\nKey Equations:\n"
                for eq_name, eq_formula in improved_theory['key_equations'].items():
                    equations_text += f"- {eq_name}: {eq_formula}\n"
                improved_theory['mathematical_formalism'] += equations_text
            
            # 添加元数据
            improved_theory['generation_metadata'] = {
                'improvement_method': 'full_evaluation_feedback',
                'original_theory': original_theory.get('name'),
                'experimental_success_rate': feedback['experimental_results'].get(
                    original_theory.get('name', ''), {}
                ).get('success_rate', 0),
                'model': f"{self.model_source}/{self.model_name}"
            }
            
            return improved_theory
            
        except Exception as e:
            logger.error(f"生成改进理论失败: {e}")
            return None
    
    async def run_feedback_loop(self, 
                              theory_path: str, 
                              output_dir: str, 
                              iterations: int = 1,
                              include_experiments: bool = True):
        """运行完整的反馈循环"""
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
            improvement_history = []
            
            for iteration in range(iterations):
                logger.info(f"\n--- 迭代 {iteration + 1}/{iterations} ---")
                
                iteration_dir = output_path / f"iteration_{iteration + 1}"
                iteration_dir.mkdir(exist_ok=True)
                
                # 1. 保存当前理论
                current_theory_path = iteration_dir / "current_theory.json"
                with open(current_theory_path, 'w', encoding='utf-8') as f:
                    json.dump(current_theory, f, ensure_ascii=False, indent=2)
                
                # 2. 运行完整评估
                logger.info(f"运行{'完整' if include_experiments else '角色'}评估...")
                eval_output_dir = iteration_dir / "evaluation"
                
                # 创建临时理论目录
                temp_theory_dir = iteration_dir / "temp_theories"
                temp_theory_dir.mkdir(exist_ok=True)
                temp_theory_path = temp_theory_dir / f"{current_theory.get('name', 'theory')}.json"
                with open(temp_theory_path, 'w', encoding='utf-8') as f:
                    json.dump(current_theory, f, ensure_ascii=False, indent=2)
                
                # 运行评估
                success = self.run_full_evaluation(
                    str(temp_theory_dir),
                    str(eval_output_dir),
                    include_experiments=include_experiments
                )
                
                if not success:
                    logger.error("评估失败，跳过此迭代")
                    continue
                
                # 3. 提取综合反馈
                logger.info("提取评估反馈...")
                feedback = self.extract_comprehensive_feedback(eval_output_dir)
                
                # 保存反馈
                feedback_path = iteration_dir / "extracted_feedback.json"
                with open(feedback_path, 'w', encoding='utf-8') as f:
                    json.dump(feedback, f, ensure_ascii=False, indent=2)
                
                # 记录改进历史
                improvement_entry = {
                    'iteration': iteration + 1,
                    'weaknesses_count': len(feedback['combined_insights']['weaknesses']),
                    'suggestions_count': len(feedback['combined_insights']['improvement_suggestions'])
                }
                
                # 只有在有实验结果时才添加成功率
                if include_experiments and feedback['experimental_results']:
                    improvement_entry['experimental_success_rate'] = feedback['experimental_results'].get(
                        current_theory.get('name', ''), {}
                    ).get('success_rate', 0)
                
                improvement_history.append(improvement_entry)
                
                # 4. 生成改进版本
                logger.info("生成改进版本...")
                improved_theory = await self.generate_improved_theory(current_theory, feedback)
                
                if improved_theory:
                    # 保存改进版本
                    improved_path = iteration_dir / "improved_theory.json"
                    with open(improved_path, 'w', encoding='utf-8') as f:
                        json.dump(improved_theory, f, ensure_ascii=False, indent=2)
                    
                    logger.info(f"✅ 生成改进版本: {improved_theory.get('name')}")
                    
                    # 更新当前理论
                    current_theory = improved_theory
                else:
                    logger.error("生成改进版本失败")
                    break
            
            # 保存最终结果
            final_dir = output_path / "final_results"
            final_dir.mkdir(exist_ok=True)
            
            # 保存最终理论
            final_theory_path = final_dir / f"{theory_name}_final.json"
            with open(final_theory_path, 'w', encoding='utf-8') as f:
                json.dump(current_theory, f, ensure_ascii=False, indent=2)
            
            # 保存改进历史
            history_path = final_dir / f"{theory_name}_improvement_history.json"
            with open(history_path, 'w', encoding='utf-8') as f:
                json.dump(improvement_history, f, ensure_ascii=False, indent=2)
            
            # 生成改进报告
            report = self._generate_improvement_report(
                theory_data, current_theory, improvement_history
            )
            report_path = final_dir / f"{theory_name}_improvement_report.md"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(report)
            
            logger.info(f"\n✅ 完成 {theory_name} 的完整反馈循环优化")
    
    def _generate_improvement_report(self, 
                                   original_theory: Dict, 
                                   final_theory: Dict, 
                                   history: List[Dict]) -> str:
        """生成详细的改进报告"""
        report = f"""# 理论改进报告 - 完整评估反馈循环

## 原始理论
- **名称**: {original_theory.get('name', '未知')}
- **初始描述**: {original_theory.get('description', '无')[:300]}...

## 最终版本
- **名称**: {final_theory.get('name', '未知')}
- **总迭代次数**: {len(history)}

## 改进轨迹

| 迭代 | 实验成功率 | 问题数 | 建议数 |
|------|-----------|--------|--------|
"""
        
        for record in history:
            report += f"| {record['iteration']} | {record['experimental_success_rate']*100:.1f}% | "
            report += f"{record['weaknesses_count']} | {record['suggestions_count']} |\n"
        
        report += f"""

## 关键改进

### 数学形式化
{final_theory.get('improvements_summary', '未提供详细说明')}

### 实验兼容性
原始理论实验成功率: {history[0].get('experimental_success_rate', 0)*100:.1f if history else 0:.1f}% (如果有实验评估)
最终理论（预期）: 需要重新评估以确认

### 概念澄清
{len(final_theory.get('conceptual_clarifications', {}))} 个核心概念得到精确定义

## 保留的创新要素
{final_theory.get('description', '')[:200]}...

## 建议后续步骤
1. 对最终版本进行完整的实验评估，验证改进效果
2. 进行更深入的数学一致性检查
3. 设计区分性实验，验证独特预测
"""
        return report


async def main():
    parser = argparse.ArgumentParser(description="完整评估反馈循环")
    parser.add_argument("--theory", required=True, help="理论文件或目录路径")
    parser.add_argument("--output", default="output_full_feedback_loop", help="输出目录")
    parser.add_argument("--iterations", type=int, default=2, help="反馈循环次数")
    parser.add_argument("--model_source", default="google", help="模型来源")
    parser.add_argument("--model_name", default="gemini-2.5-pro", help="模型名称")
    parser.add_argument("--experiment_dir", default="data/experiments", help="实验数据目录")
    parser.add_argument("--skip_experiments", action="store_true", help="跳过实验评估，只做角色评估")
    
    args = parser.parse_args()
    
    # 创建反馈循环实例
    feedback_loop = FullEvaluationFeedbackLoop(
        model_source=args.model_source,
        model_name=args.model_name,
        experiment_dir=args.experiment_dir
    )
    
    # 运行反馈循环
    await feedback_loop.run_feedback_loop(
        theory_path=args.theory,
        output_dir=args.output,
        iterations=args.iterations,
        include_experiments=not args.skip_experiments
    )


if __name__ == "__main__":
    asyncio.run(main())