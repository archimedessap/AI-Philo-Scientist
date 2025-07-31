#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_feedback_quick.py - 快速测试评估反馈机制
==========================================

一个简化的测试脚本，用于快速验证评估反馈流程
"""

import argparse
import asyncio
import json
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append('.')

from theory_generation.llm_interface import LLMInterface
from simple_feedback_loop_v2 import ImprovedFeedbackLoop
from utils.logging_config import get_logger

logger = get_logger(__name__)


async def quick_feedback_test(
    theory_path: str,
    output_dir: str = "output_quick_feedback_test",
    model_source: str = "google",
    model_name: str = "gemini-2.5-flash",
    skip_improvement: bool = False
):
    """快速测试评估反馈机制"""
    
    print(f"\n{'='*60}")
    print("快速评估反馈测试")
    print(f"{'='*60}\n")
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 初始化反馈循环
    feedback_loop = ImprovedFeedbackLoop(
        model_source=model_source,
        model_name=model_name
    )
    
    # 加载理论
    print(f"1. 加载理论文件: {theory_path}")
    with open(theory_path, 'r', encoding='utf-8') as f:
        theory = json.load(f)
    
    theory_name = theory.get('name', '未知理论')
    print(f"   理论名称: {theory_name}")
    print(f"   描述: {theory.get('description', '无')[:100]}...")
    
    # 运行角色评估
    print(f"\n2. 运行三角色评估...")
    eval_result = await feedback_loop.evaluate_theory_with_roles(theory, output_path)
    
    # 显示评估结果
    print(f"\n3. 评估结果:")
    if 'evaluations' in eval_result:
        for role, eval_data in eval_result['evaluations'].items():
            score = eval_data.get('score', 0)
            print(f"   - {role}: {score}/10")
    
    # 提取反馈
    print(f"\n4. 提取综合反馈...")
    feedback = feedback_loop.extract_comprehensive_feedback(eval_result)
    
    # 显示反馈摘要
    print(f"\n   优点 ({len(feedback['strengths'])}):")
    for s in feedback['strengths'][:3]:
        print(f"   • {s}")
    
    print(f"\n   问题 ({len(feedback['weaknesses'])}):")
    for w in feedback['weaknesses'][:3]:
        print(f"   • {w}")
    
    print(f"\n   改进建议 ({len(feedback['improvement_suggestions'])}):")
    for s in feedback['improvement_suggestions'][:3]:
        print(f"   • {s}")
    
    # 保存反馈报告
    feedback_path = output_path / f"{theory_name}_feedback_summary.json"
    with open(feedback_path, 'w', encoding='utf-8') as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)
    print(f"\n   反馈报告已保存: {feedback_path}")
    
    # 生成改进版本（可选）
    if not skip_improvement:
        print(f"\n5. 基于反馈生成改进版本...")
        improved_theory = await feedback_loop.generate_improved_theory(theory, feedback)
        
        if improved_theory:
            improved_path = output_path / f"{theory_name}_improved.json"
            with open(improved_path, 'w', encoding='utf-8') as f:
                json.dump(improved_theory, f, ensure_ascii=False, indent=2)
            print(f"   改进版本已保存: {improved_path}")
            
            # 显示改进摘要
            if 'improvements_summary' in improved_theory:
                print(f"\n   主要改进:")
                print(f"   {improved_theory['improvements_summary']}")
        else:
            print(f"   生成改进版本失败")
    
    print(f"\n{'='*60}")
    print(f"测试完成! 结果保存在: {output_path}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="快速测试评估反馈机制",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "--theory", 
        default="data/theories_test/T_MW_many-worlds_(everett)_interpretation.json",
        help="理论文件路径"
    )
    
    parser.add_argument(
        "--output", 
        default="output_quick_feedback_test",
        help="输出目录"
    )
    
    parser.add_argument(
        "--model_source",
        default="google",
        choices=["google", "openai", "deepseek", "anthropic"],
        help="LLM提供商"
    )
    
    parser.add_argument(
        "--model_name",
        default="gemini-2.5-flash",
        help="模型名称"
    )
    
    parser.add_argument(
        "--skip_improvement",
        action="store_true",
        help="跳过生成改进版本（只做评估和反馈提取）"
    )
    
    args = parser.parse_args()
    
    # 运行测试
    asyncio.run(quick_feedback_test(
        theory_path=args.theory,
        output_dir=args.output,
        model_source=args.model_source,
        model_name=args.model_name,
        skip_improvement=args.skip_improvement
    ))


if __name__ == "__main__":
    main()