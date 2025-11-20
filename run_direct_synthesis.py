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
import glob
import shutil
from pathlib import Path
from theory_generation.llm_interface import LLMInterface
from theory_generation.direct_synthesis.contradiction_analyzer import ContradictionAnalyzer
from theory_generation.direct_synthesis.hypothesis_generator import HypothesisGenerator
from theory_generation.short_card_generator import (
    ShortCardGenerationConfig,
    convert_to_legacy_schema,
    generate_short_card_theory,
)

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def load_theories_from_directory(theories_dir):
    """从目录加载理论文件"""
    theories = {}
    
    theory_files = glob.glob(os.path.join(theories_dir, "*.json"))
    print(f"[INFO] 在目录 {theories_dir} 中找到 {len(theory_files)} 个理论文件")
    
    for theory_file in theory_files:
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            
            theory_name = theory.get("name", os.path.basename(theory_file))
            theories[theory_name] = theory
            
        except Exception as e:
            print(f"[ERROR] 加载理论文件 {theory_file} 时出错: {str(e)}")
    
    return theories

def slugify(value: str) -> str:
    import re

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "theory"


async def main():
    parser = argparse.ArgumentParser(description="理论直接合成程序")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "xai", "google"],
                        help="模型来源")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="模型名称")
    
    # 生成模式
    parser.add_argument("--generation_method", type=str, default="direct",
                        choices=["direct", "short_card"],
                        help="理论生成方法：direct（传统矛盾对）或 short_card（短卡联合分析）")

    # 输入参数（direct 模式）
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

    # 短卡模式参数
    parser.add_argument("--cards_dir", type=str, default="cards", help="短卡目录")
    parser.add_argument("--card_schema", type=str, default="schemas/card.schema.json", help="短卡Schema路径")
    parser.add_argument("--contradiction_schema", type=str, default="schemas/contradiction.schema.json", help="矛盾表Schema路径")
    parser.add_argument("--new_interpretation_schema", type=str, default="schemas/new_interpretation.schema.json", help="新诠释结构化输出Schema")
    parser.add_argument("--short_card_constraints", type=str, default=None, help="短卡模式约束文件JSON")
    parser.add_argument("--short_card_contradictions", type=str, default=None, help="预先生成的矛盾表JSON路径，提供后将跳过LLM矛盾分析")
    parser.add_argument("--short_card_query", type=str, default="短卡联合分析生成新理论", help="短卡模式的任务描述")
    parser.add_argument("--short_card_task_hint", type=str, default="", help="短卡模式额外提示")
    parser.add_argument("--short_card_topk", type=int, default=-1, help="短卡模式使用的卡片数量，-1 表示全部")
    parser.add_argument("--short_card_machine_temperature", type=float, default=0.4, help="短卡模式结构化输出温度")
    parser.add_argument("--short_card_human_temperature", type=float, default=0.6, help="短卡模式人类文本输出温度")
    parser.add_argument("--short_card_human_model_source", type=str, default=None, help="短卡模式人类写作模型来源")
    parser.add_argument("--short_card_human_model_name", type=str, default=None, help="短卡模式人类写作模型名称")
    parser.add_argument("--num_theories", type=int, default=1, help="要生成的新理论数量（短卡模式适用）")

    args = parser.parse_args()
    
    # 确保输出目录存在
    synthesis_dir = os.path.join(args.output_dir, f"synthesis_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_directory_exists(synthesis_dir)
    
    if args.generation_method == "short_card":
        config = ShortCardGenerationConfig(
            query=args.short_card_query,
            task_hint=args.short_card_task_hint,
            cards_dir=Path(args.cards_dir),
            card_schema=Path(args.card_schema),
            contradiction_schema=Path(args.contradiction_schema),
            new_schema=Path(args.new_interpretation_schema),
            constraints_path=Path(args.short_card_constraints) if args.short_card_constraints else None,
            topk=args.short_card_topk,
            machine_model_source=args.model_source,
            machine_model_name=args.model_name,
            human_model_source=args.short_card_human_model_source,
            human_model_name=args.short_card_human_model_name,
            machine_temperature=args.short_card_machine_temperature,
            human_temperature=args.short_card_human_temperature,
            precomputed_contradictions_path=Path(args.short_card_contradictions) if args.short_card_contradictions else None,
        )

        if args.num_theories < 1:
            raise ValueError("--num_theories 必须为正整数")

        short_dir = os.path.join(synthesis_dir, "short_card_rag")
        ensure_directory_exists(short_dir)
        eval_ready_dir = os.path.join(synthesis_dir, "eval_ready_theories")
        ensure_directory_exists(eval_ready_dir)

        synthesized_theories = []

        for idx in range(args.num_theories):
            result = await generate_short_card_theory(config)
            legacy_theory, machine_summary = convert_to_legacy_schema(result)

            suffix = "" if args.num_theories == 1 else f"_{idx + 1}"

            contradictions_path = os.path.join(short_dir, f"contradictions{suffix or ''}.json")
            with open(contradictions_path, "w", encoding="utf-8") as f:
                json.dump(result.get("contradictions", {}), f, ensure_ascii=False, indent=2)

            raw_path = os.path.join(short_dir, f"raw_new_interpretation{suffix or ''}.json")
            with open(raw_path, "w", encoding="utf-8") as f:
                json.dump({
                    "machine_summary": machine_summary,
                    "writeup": result.get("writeup"),
                    "selected_cards": result.get("selected_card_ids", [])
                }, f, ensure_ascii=False, indent=2)

            base_slug = slugify(legacy_theory['name'])
            slug_candidate = base_slug
            counter = 1
            while os.path.exists(os.path.join(eval_ready_dir, f"{slug_candidate}.json")):
                slug_candidate = f"{base_slug}_{counter}"
                counter += 1

            eval_ready_path = os.path.join(eval_ready_dir, f"{slug_candidate}.json")
            with open(eval_ready_path, "w", encoding="utf-8") as f:
                json.dump(legacy_theory, f, ensure_ascii=False, indent=2)

            synthesized_theories.append(legacy_theory)

            print(f"[INFO] 成功生成短卡驱动理论({idx + 1}/{args.num_theories}): {legacy_theory['name']}")
            print(f"[INFO] 标准格式文件: {eval_ready_path}")

        with open(os.path.join(synthesis_dir, "all_synthesized_theories.json"), "w", encoding="utf-8") as f:
            json.dump(synthesized_theories, f, ensure_ascii=False, indent=2)

        print("标准格式的评估理论文件已保存到: {}".format(eval_ready_dir))
        return

    # 初始化LLM接口（direct 模式）
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
    
    # 导入数学分类器
    import sys
    sys.path.append('.')
    from utils.mathematical_classifier import MathematicalClassifier
    classifier = MathematicalClassifier()
    
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
            # 添加数学分类标注
            hypothesis = classifier.annotate_theory_with_classification(hypothesis)
            
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
            # 确保理论有数学分类标注
            if "mathematical_classification" not in hypothesis.get("metadata", {}):
                hypothesis = classifier.annotate_theory_with_classification(hypothesis)
            
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
        
        # 生成数学分类统计
        classification_stats = {"standard_qm": 0, "modified_qm": 0, "extended_qm": 0}
        for hypothesis in all_hypotheses:
            math_classification = hypothesis.get("metadata", {}).get("mathematical_classification", {})
            math_type = math_classification.get("type", "unknown")
            if math_type in classification_stats:
                classification_stats[math_type] += 1
        
        print(f"\n📊 数学分类统计:")
        for math_type, count in classification_stats.items():
            print(f"   {math_type}: {count}")
    else:
        print("\n[完成] 未生成任何新理论")

if __name__ == "__main__":
    asyncio.run(main())
