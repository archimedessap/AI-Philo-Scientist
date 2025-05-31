#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
概念和公式提取与合并脚本

功能:
1. 从原始文献中提取概念和公式
2. 使用LLM生成先验理论描述
3. 对比和融合提取的与生成的概念和公式
4. 保存结果到CSV文件供后续处理

用法:
python scripts/extract_and_merge_concepts.py --input_dir data/raw_literature --output_dir data/processed_concepts --theories "Copenhagen Interpretation,Many-Worlds Interpretation"
"""

import os
import sys
import argparse
from typing import List, Dict, Tuple

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core_embedding.concept_extractor import ConceptExtractor
from theory_generation.prior_theory_generator import PriorTheoryGenerator
from core_embedding.concept_merger import ConceptMerger
from theory_generation.llm_interface import LLMInterface

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        print(f"[INFO] 创建目录: {directory}")
        os.makedirs(directory, exist_ok=True)

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="概念和公式提取与合并工具")
    parser.add_argument("--input_dir", default="data/raw_literature",
                      help="输入文献目录")
    parser.add_argument("--output_dir", default="data/processed_concepts",
                      help="输出目录")
    parser.add_argument("--theories", default="Copenhagen Interpretation,Many-Worlds Interpretation",
                      help="要生成的理论列表，用逗号分隔")
    parser.add_argument("--model_source", default="openai",
                      choices=["openai", "ollama", "deepseek", "groq", "auto"],
                      help="LLM模型来源，使用'auto'自动选择可用的最佳模型")
    parser.add_argument("--model_name", default="gpt-4o-mini",
                      help="LLM模型名称")
    parser.add_argument("--api_key", default=None,
                      help="API密钥（可选，默认从环境变量读取）")
    parser.add_argument("--skip_extraction", action="store_true",
                      help="跳过从文献提取概念和公式步骤")
    parser.add_argument("--skip_generation", action="store_true",
                      help="跳过使用LLM生成理论步骤")
    parser.add_argument("--skip_merge", action="store_true",
                      help="跳过合并结果步骤")
    parser.add_argument("--auto_fallback", action="store_true",
                      help="启用自动回退到备用模型（当主选模型失败时）")
    
    args = parser.parse_args()
    
    # 创建必要的目录
    ensure_directory_exists(args.output_dir)
    if not args.skip_extraction:
        ensure_directory_exists(args.input_dir)
    
    # 检查是否在输入目录中有文献
    if not args.skip_extraction and len(os.listdir(args.input_dir)) == 0:
        print(f"[WARN] 输入目录 {args.input_dir} 为空，您可能需要添加一些PDF或文本文件")
    
    # 检查是否启用自动模型选择
    auto_select = args.model_source == "auto"
    if auto_select:
        print("[INFO] 启用自动模型选择...")
        model_source = None  # 将由LLMInterface自动选择
    else:
        model_source = args.model_source
    
    # 创建LLM接口
    llm = LLMInterface(
        model_source=model_source,
        model_name=args.model_name,
        api_key=args.api_key,
        auto_select=auto_select,
        enable_fallback=args.auto_fallback
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_model_info()
    print(f"[INFO] 使用模型: {model_info['model_source']}/{model_info['model_name']}")
    
    # 初始化变量
    extracted_concepts = []
    extracted_formulas = []
    generated_concepts = []
    generated_formulas = []
    
    # Step 1: 从文献中提取概念和公式
    if not args.skip_extraction:
        print("\n" + "="*80)
        print("步骤1: 从文献中提取概念和公式")
        print("="*80)
        
        # 检查输入目录是否存在
        if not os.path.isdir(args.input_dir):
            print(f"[ERROR] 输入目录不存在: {args.input_dir}")
            return
        
        # 初始化概念提取器
        extractor = ConceptExtractor(llm_interface=llm)
        
        # 从文献中提取概念和公式
        extracted_concepts, extracted_formulas = extractor.extract_from_directory(args.input_dir)
        
        print(f"[INFO] 从文献中提取了 {len(extracted_concepts)} 个概念和 {len(extracted_formulas)} 个公式")
        
        # 保存原始提取结果(可选)
        if extracted_concepts or extracted_formulas:
            # 确保子目录存在
            extract_dir = os.path.join(args.output_dir, "extracted")
            ensure_directory_exists(extract_dir)
            extractor.save_to_csv(extract_dir)
    
    # Step 2: 使用LLM生成先验理论
    if not args.skip_generation:
        print("\n" + "="*80)
        print("步骤2: 使用LLM生成先验理论")
        print("="*80)
        
        # 初始化理论生成器
        generator = PriorTheoryGenerator(llm_interface=llm)
        
        # 解析理论名称列表
        theory_names = [name.strip() for name in args.theories.split(",") if name.strip()]
        
        if not theory_names:
            print("[WARN] 未指定理论名称，将使用默认理论")
            theory_names = ["Copenhagen Interpretation", "Many-Worlds Interpretation"]
        
        # 生成理论
        theories = generator.generate_multiple_theories(theory_names)
        
        # 从生成的理论中提取概念和公式
        generated_concepts = generator.concepts
        generated_formulas = generator.formulas
        
        print(f"[INFO] 生成了 {len(theories)} 个理论，包含 {len(generated_concepts)} 个概念和 {len(generated_formulas)} 个公式")
        
        # 保存生成的理论和概念(可选)
        if generated_concepts or generated_formulas:
            # 确保子目录存在
            generated_dir = os.path.join(args.output_dir, "generated")
            ensure_directory_exists(generated_dir)
            generator.save_to_csv(generated_dir)
    
    # Step 3: 合并提取的和生成的概念和公式
    if not args.skip_merge:
        print("\n" + "="*80)
        print("步骤3: 合并提取的和生成的概念和公式")
        print("="*80)
        
        # 初始化概念合并器
        merger = ConceptMerger(llm_interface=llm)
        
        # 合并概念
        merged_concepts = merger.merge_concepts(extracted_concepts, generated_concepts)
        
        # 合并公式
        merged_formulas = merger.merge_formulas(extracted_formulas, generated_formulas)
        
        print(f"[INFO] 合并后得到 {len(merged_concepts)} 个概念和 {len(merged_formulas)} 个公式")
        
        # 保存合并结果
        if merged_concepts or merged_formulas:
            merger.save_to_csv(args.output_dir)
    
    print("\n" + "="*80)
    print("处理完成!")
    print("="*80)
    print(f"结果保存在 {args.output_dir}")

if __name__ == "__main__":
    main() 