#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
角色专家评估执行器

独立运行多角色专家对理论的评估，不依赖实验评估模块。
支持选择不同的LLM模型。
"""

import os
import sys
import json
import argparse
import asyncio
from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
from theory_generation.llm_interface import LLMInterface

async def run_role_evaluation(theory_file, output_file=None, 
                             model_source="deepseek", model_name="deepseek-chat"):
    """
    运行角色评估
    
    Args:
        theory_file: 理论文件路径
        output_file: 输出文件路径
        model_source: LLM模型来源
        model_name: LLM模型名称
    """
    print(f"[INFO] 使用{model_source}模型: {model_name}")
    
    # 加载理论
    with open(theory_file, 'r', encoding='utf-8') as f:
        theories = json.load(f)
    
    # 初始化LLM接口
    llm = LLMInterface(model_source=model_source, model_name=model_name)
    
    # 初始化评估器
    evaluator = TheoryEvaluator(llm)
    
    # 执行评估
    if isinstance(theories, list):
        results = await evaluator.evaluate_theories(theories)
    else:
        results = [await evaluator.evaluate_theory(theories, predictor_module=None)]
    
    # 保存结果
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    
    return results

def main():
    parser = argparse.ArgumentParser(description="角色专家评估执行器")
    parser.add_argument("--theory_file", required=True, help="理论文件路径")
    parser.add_argument("--output_file", help="输出文件路径")
    parser.add_argument("--model_source", default="deepseek", 
                       help="LLM模型来源 (例如: openai, deepseek)")
    parser.add_argument("--model_name", default="deepseek-chat", 
                       help="LLM模型名称 (例如: gpt-4, deepseek-chat)")
    
    args = parser.parse_args()
    asyncio.run(run_role_evaluation(
        args.theory_file, 
        args.output_file,
        args.model_source,
        args.model_name
    ))

if __name__ == "__main__":
    main()
