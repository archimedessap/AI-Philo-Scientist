#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Direct theory synthesis runner.

Example:
python run_direct_synthesis.py --model_source deepseek --model_name deepseek-chat

Analyses contradictions among quantum theories and asks an LLM to propose
new hypotheses without relying on vector-space operations.
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
    """Ensure the target directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] Created directory: {directory}")

def load_theories_from_directory(theories_dir):
    """Load theory JSON files from a directory."""
    theories = {}
    
    theory_files = glob.glob(os.path.join(theories_dir, "*.json"))
    print(f"[INFO] Found {len(theory_files)} theory files in {theories_dir}")
    
    for theory_file in theory_files:
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            
            theory_name = theory.get("name", os.path.basename(theory_file))
            theories[theory_name] = theory
            
        except Exception as e:
            print(f"[ERROR] Failed to load theory file {theory_file}: {str(e)}")
    
    return theories

def slugify(value: str) -> str:
    import re

    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value or "theory"


async def main():
    parser = argparse.ArgumentParser(description="Direct theory synthesis runner")
    
    # LLM parameters
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "xai", "google"],
                        help="Model provider to use for LLM calls.")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="Model name to use.")
    
    # Generation mode
    parser.add_argument("--generation_method", type=str, default="direct",
                        choices=["direct", "short_card"],
                        help="Theory generation method: 'direct' or 'short_card'.")

    # Input parameters (direct mode)
    parser.add_argument("--theories_dir", type=str, 
                        default="data/theories_v2.1",
                        help="Directory containing source theories.")
    parser.add_argument("--specific_pair", type=str, default=None,
                        help="Specific theory pair to analyse, format 'theory1,theory2'.")
    parser.add_argument("--max_pairs", type=int, default=3,
                        help="Maximum number of theory pairs to analyse when selecting automatically.")
    parser.add_argument("--schema_version", type=str, default="2.1",
                        help="Theory schema version to load; use 'any' to load all versions.")
    
    # Generation parameters
    parser.add_argument("--variants_per_contradiction", type=int, default=3,
                        help="Number of hypothesis variants per contradiction.")
    parser.add_argument("--diversity_level", type=float, default=0.7,
                        help="Controls diversity of generated variants (0.0-1.0).")
    
    # Output parameters
    parser.add_argument("--output_dir", type=str, default="data/synthesized_theories",
                        help="Directory to store synthesis outputs.")

    # Short-card parameters
    parser.add_argument("--cards_dir", type=str, default="cards", help="Directory containing short-card JSON files.")
    parser.add_argument("--card_schema", type=str, default="schemas/card.schema.json", help="Schema path for short-card files.")
    parser.add_argument("--contradiction_schema", type=str, default="schemas/contradiction.schema.json", help="Schema path for contradiction tables.")
    parser.add_argument("--new_interpretation_schema", type=str, default="schemas/new_interpretation.schema.json", help="Schema path for synthesized interpretations.")
    parser.add_argument("--short_card_constraints", type=str, default=None, help="Optional JSON file with custom constraints for short-card mode.")
    parser.add_argument("--short_card_contradictions", type=str, default=None, help="Optional precomputed contradiction table to skip analysis.")
    parser.add_argument("--short_card_query", type=str, default="Short-card joint analysis for new theory creation", help="Task description passed to the short-card workflow.")
    parser.add_argument("--short_card_task_hint", type=str, default="", help="Additional hint for the short-card workflow.")
    parser.add_argument("--short_card_topk", type=int, default=-1, help="Number of cards to use in short-card mode; -1 means all available cards.")
    parser.add_argument("--short_card_machine_temperature", type=float, default=0.4, help="Sampling temperature for the structured output in short-card mode.")
    parser.add_argument("--short_card_human_temperature", type=float, default=0.6, help="Sampling temperature for the human-readable write-up in short-card mode.")
    parser.add_argument("--short_card_human_model_source", type=str, default=None, help="Model source for the human-readable write-up (short-card mode).")
    parser.add_argument("--short_card_human_model_name", type=str, default=None, help="Model name for the human-readable write-up (short-card mode).")
    parser.add_argument("--num_theories", type=int, default=1, help="Number of theories to generate in short-card mode.")

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
            raise ValueError("--num_theories must be a positive integer")

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

            print(f"[INFO] Generated short-card theory ({idx + 1}/{args.num_theories}): {legacy_theory['name']}")
            print(f"[INFO] Evaluation-ready file: {eval_ready_path}")

        with open(os.path.join(synthesis_dir, "all_synthesized_theories.json"), "w", encoding="utf-8") as f:
            json.dump(synthesized_theories, f, ensure_ascii=False, indent=2)

        print("Evaluation-ready theory files saved to: {}".format(eval_ready_dir))
        return

    # Initialise LLM interface (direct mode)
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=1.0
    )

    # Display current model info
    model_info = llm.get_current_model_info()
    print(f"[INFO] Using model: {model_info['source']} - {model_info['name']}")

    # Step 1: Load theories
    print(f"\n[Step 1] Loading theories from {args.theories_dir}")
    analyzer = ContradictionAnalyzer(llm)

    load_schema_version = None if args.schema_version.lower() == 'any' else args.schema_version
    analyzer.load_theories(args.theories_dir, schema_version=load_schema_version)

    if not analyzer.theories:
        print("[ERROR] No theory data loaded; aborting.")
        return
    
    # Step 2: Determine theory pairs for contradiction analysis
    print("\n[Step 2] Analysing theory contradictions")
    theory_pairs = []
    
    if args.specific_pair:
        # Use the specified pair
        theory_names = args.specific_pair.split(',')
        if len(theory_names) != 2:
            print(f"[ERROR] Invalid theory pair format: {args.specific_pair}; expected 'theory1,theory2'")
            return
        theory_pairs.append((theory_names[0], theory_names[1]))
    else:
        # Automatically choose theory pairs
        theory_names = list(analyzer.theories.keys())
        if len(theory_names) < 2:
            print("[ERROR] At least two theories are required for comparison")
            return
            
        import random
        from itertools import combinations
        
        # Generate all possible pairs and select randomly
        all_pairs = list(combinations(theory_names, 2))
        random.shuffle(all_pairs)
        theory_pairs = all_pairs[:args.max_pairs]
    
    print(f"[INFO] Analysing {len(theory_pairs)} theory contradictions")

    # Step 3: Analyse contradictions and generate hypotheses
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
    print(f"\n[Step 3] Synthesising new theories from contradictions")
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
        
        print(f"[INFO] Processing contradiction: {theory1} vs {theory2}")
        
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
            
            hypothesis_name = hypothesis.get("name", f"new_theory_{i+1}")
            safe_name = hypothesis_name.replace(" ", "_").replace("/", "_").lower()
            
            # 保存到文件
            hypothesis_file = os.path.join(hypotheses_dir, f"{safe_name}.json")
            with open(hypothesis_file, 'w', encoding='utf-8') as f:
                json.dump(hypothesis, f, ensure_ascii=False, indent=2)
        
        # 保存所有假说到一个文件
        all_file = os.path.join(hypotheses_dir, "all_variants.json")
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(hypotheses, f, ensure_ascii=False, indent=2)
            
        print(f"[INFO] Generated {len(hypotheses)} candidate theories for {theory1} vs {theory2}")
    
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
            theory_name = hypothesis.get("name", "Unnamed theory")
            safe_name = theory_name.replace(" ", "_").replace("/", "_").lower()
            
            # Schema v2.1: 直接保存完整的假说对象，因为它已经符合新格式
            eval_theory = hypothesis
            
            # 保存标准格式的理论文件
            eval_file = os.path.join(eval_theories_dir, f"{safe_name}.json")
            with open(eval_file, 'w', encoding='utf-8') as f:
                json.dump(eval_theory, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] Synthesized {len(all_hypotheses)} new theories; saved to: {synthesis_dir}")
        print(f"[INFO] Evaluation-ready theory files saved to: {eval_theories_dir}")
        
        # 生成数学分类统计
        classification_stats = {"standard_qm": 0, "modified_qm": 0, "extended_qm": 0}
        for hypothesis in all_hypotheses:
            math_classification = hypothesis.get("metadata", {}).get("mathematical_classification", {})
            math_type = math_classification.get("type", "unknown")
            if math_type in classification_stats:
                classification_stats[math_type] += 1
        
        print(f"\n[INFO] Mathematical classification summary:")
        for math_type, count in classification_stats.items():
            print(f"   {math_type}: {count}")
    else:
        print("\n[INFO] No theories were generated")

if __name__ == "__main__":
    asyncio.run(main())
