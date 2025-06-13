#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论直接合成程序
python run_direct_synthesis.py --model_source deepseek --model_name deepseek-chat
分析量子理论矛盾点，并直接调用LLM合成新的理论假说，
不依赖向量空间操作，而是利用LLM的认知能力直接放松矛盾。
"""

import os
import json
import argparse
import asyncio
import time
from theory_generation.llm_interface import LLMInterface
from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from theory_generation.direct_synthesis.hypothesis_generator import HypothesisGenerator

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def main():
    parser = argparse.ArgumentParser(description="理论直接合成程序")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "google"],
                        help="模型来源")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="模型名称")
    
    # 输入参数
    parser.add_argument("--theories_dir", type=str, 
                        default="data/theories_v2.1",
                        help="理论文件目录")
    parser.add_argument("--specific_pair", type=str, default=None,
                        help="特定理论对，格式为'理论1,理论2'")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="最大比较对数，默认为3")
    parser.add_argument("--schema_version", type=str, default="2.1",
                        help="要加载的理论schema版本，设置为'any'可加载所有版本")
    
    # 生成参数
    parser.add_argument("--variants_per_contradiction", type=int, default=3,
                        help="每个矛盾点的假说变体数量")
    parser.add_argument("--diversity_level", type=float, default=0.7,
                        help="变体多样性等级(0.0-1.0)")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/synthesized_theories",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    synthesis_dir = os.path.join(args.output_dir, f"synthesis_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_directory_exists(synthesis_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=1.0
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 1. 加载理论数据
    print(f"\n[步骤1] 从 {args.theories_dir} 加载理论数据")
    analyzer = ContradictionAnalyzer(llm)
    
    load_schema_version = None if args.schema_version.lower() == 'any' else args.schema_version
    analyzer.load_theories(args.theories_dir, schema_version=load_schema_version)
    
    if not analyzer.theories:
        print("[ERROR] 未加载到理论数据，程序终止")
        return
    
    # 2. Contradiction Analysis - 确定要比较的理论对
    print(f"\n[步骤2] 分析理论矛盾点")
    theory_pairs = []
    
    if args.specific_pair:
        # 使用指定的理论对
        theory_names = args.specific_pair.split(',')
        if len(theory_names) != 2:
            print(f"[ERROR] 理论对格式错误: {args.specific_pair}，应为'理论1,理论2'")
            return
        theory_pairs.append((theory_names[0], theory_names[1]))
    else:
        # 自动选择理论对
        theory_names = list(analyzer.theories.keys())
        if len(theory_names) < 2:
            print("[ERROR] 至少需要2个理论才能进行比较")
            return
            
        import random
        from itertools import combinations
        
        # 生成所有可能的理论对并随机选择
        all_pairs = list(combinations(theory_names, 2))
        random.shuffle(all_pairs)
        theory_pairs = all_pairs[:args.max_pairs]
    
    print(f"[INFO] 将分析 {len(theory_pairs)} 对理论的矛盾点")
    
    # 3. 对每对理论进行矛盾分析
    all_analyses = []
    for theory1, theory2 in theory_pairs:
        analysis = await analyzer.find_contradictions(theory1, theory2)
        if "error" not in analysis:
            all_analyses.append(analysis)
            
            # 保存单个分析结果
            pair_name = f"{theory1}_vs_{theory2}".replace(" ", "_")
            analysis_dir = os.path.join(synthesis_dir, pair_name)
            ensure_directory_exists(analysis_dir)
            
            analysis_file = os.path.join(analysis_dir, "contradiction_analysis.json")
            with open(analysis_file, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
    
    # 保存所有分析结果
    analyses_file = os.path.join(synthesis_dir, "all_contradiction_analyses.json")
    with open(analyses_file, 'w', encoding='utf-8') as f:
        json.dump(all_analyses, f, ensure_ascii=False, indent=2)
    
    # 4. 基于矛盾生成新假说
    print(f"\n[步骤3] 基于矛盾点合成新理论")
    generator = HypothesisGenerator(llm)
    
    for analysis in all_analyses:
        theory1 = analysis.get("theory1")
        theory2 = analysis.get("theory2")
        pair_name = f"{theory1}_vs_{theory2}".replace(" ", "_")
        
        print(f"[INFO] 处理矛盾: {theory1} vs {theory2}")
        
        # 为该理论对创建输出目录
        pair_dir = os.path.join(synthesis_dir, pair_name)
        hypotheses_dir = os.path.join(pair_dir, "hypotheses")
        ensure_directory_exists(hypotheses_dir)
        
        # 生成多个假说变体
        hypotheses = await generator.generate_multiple_hypotheses(
            contradiction=analysis,
            variants_count=args.variants_per_contradiction,
            diversity_level=args.diversity_level
        )
        
        # 保存生成的假说
        for i, hypothesis in enumerate(hypotheses):
            hypothesis_name = hypothesis.get("name", f"新理论_{i+1}")
            safe_name = hypothesis_name.replace(" ", "_").replace("/", "_").lower()
            
            # 保存到文件
            hypothesis_file = os.path.join(hypotheses_dir, f"{safe_name}.json")
            with open(hypothesis_file, 'w', encoding='utf-8') as f:
                json.dump(hypothesis, f, ensure_ascii=False, indent=2)
        
        # 保存所有假说到一个文件
        all_file = os.path.join(hypotheses_dir, "all_variants.json")
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(hypotheses, f, ensure_ascii=False, indent=2)
            
        print(f"[INFO] 为 {theory1} vs {theory2} 生成了 {len(hypotheses)} 个理论假说")
    
    # 5. 汇总所有生成的假说
    all_hypotheses = generator.generated_hypotheses
    if all_hypotheses:
        summary_file = os.path.join(synthesis_dir, "all_synthesized_theories.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(all_hypotheses, f, ensure_ascii=False, indent=2)
        
        # 创建标准格式的理论文件，用于评估
        eval_theories_dir = os.path.join(synthesis_dir, "eval_ready_theories")
        ensure_directory_exists(eval_theories_dir)
        
        for hypothesis in all_hypotheses:
            # 获取理论名
            theory_name = hypothesis.get("name", "未命名理论")
            safe_name = theory_name.replace(" ", "_").replace("/", "_").lower()
            
            # Schema v2.1: 直接保存完整的假说对象，因为它已经符合新格式
            eval_theory = hypothesis
            
            # 保存标准格式的理论文件
            eval_file = os.path.join(eval_theories_dir, f"{safe_name}.json")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(eval_theory, f, ensure_ascii=False, indent=2)
        
        print(f"\n[完成] 总共合成了 {len(all_hypotheses)} 个新理论，已保存到: {synthesis_dir}")
        print(f"[INFO] 标准格式的评估理论文件已保存到: {eval_theories_dir}")
    else:
        print("\n[完成] 未生成任何新理论")

if __name__ == "__main__":
    asyncio.run(main())
