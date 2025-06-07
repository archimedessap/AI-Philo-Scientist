#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行量子诠释理论比较
python run_theory_comparison.py --model_source deepseek --model_name deepseek-chat --max_pairs 10
比较不同量子诠释理论，找出关键矛盾点。
"""

import os
import argparse
import asyncio
from theory_generation.llm_interface import LLMInterface
from theory_generation.constraint_embedding.theory_comparator import TheoryComparator

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def main():
    parser = argparse.ArgumentParser(description="比较量子诠释理论，找出矛盾点")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "ollama", "groq", "auto"],
                        help="LLM模型来源")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="LLM模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥（可选，默认使用.env中的配置）")
    
    # 输入输出参数
    parser.add_argument("--theories_dir", type=str, default="data/generated_theories",
                        help="量子诠释理论目录")
    parser.add_argument("--output_dir", type=str, default="data/theory_analysis",
                        help="结果输出目录")
    parser.add_argument("--max_pairs", type=int, default=None,
                        help="最大比较对数，默认比较所有可能组合")
    parser.add_argument("--specific_pair", type=str, default=None,
                        help="只比较特定的理论对，格式为'理论1,理论2'")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口 - 修改为与当前实现兼容的方式
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=2.0
    )
    
    # 然后设置模型源和名称
    llm.set_model(args.model_source, args.model_name)
    
    # 显示当前使用的模型信息
    print(f"[INFO] 当前使用的模型: {llm.model_source} - {llm.model_name}")
    
    # 创建理论比较器
    comparator = TheoryComparator(llm)
    
    # 加载理论
    print(f"\n[步骤1] 从 {args.theories_dir} 加载量子诠释理论")
    comparator.load_theories(args.theories_dir)
    
    # 比较理论
    if args.specific_pair:
        theory_names = args.specific_pair.split(',')
        if len(theory_names) != 2:
            print(f"[ERROR] 特定理论对格式错误: {args.specific_pair}，应为'理论1,理论2'")
            return
            
        theory1, theory2 = theory_names
        print(f"\n[步骤2] 比较特定理论对: {theory1} vs {theory2}")
        
        result = await comparator.compare_theories(theory1, theory2)
        if result:
            print(f"[成功] 找到 {len(result.get('contradictions', []))} 个矛盾点")
    else:
        print(f"\n[步骤2] 比较理论对 (最多 {args.max_pairs if args.max_pairs else '无限制'} 对)")
        results = await comparator.compare_all_pairs(args.max_pairs)
        print(f"[成功] 共比较了 {len(results)} 对理论")
    
    # 保存结果
    output_path = os.path.join(args.output_dir, "theory_contradictions.json")
    comparator.save_contradictions(output_path)
    
    print(f"\n[完成] 理论比较完成，结果已保存到: {output_path}")
    print("可以使用此结果继续运行向量空间探索步骤。")

if __name__ == "__main__":
    asyncio.run(main()) 