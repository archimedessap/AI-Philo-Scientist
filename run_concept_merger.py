#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行概念融合器
python run_concept_merger.py --extracted_data data/extracted_concepts/extracted_data.json --generated_theories data/generated_theories/all_quantum_interpretations.json
融合从文献中提取的概念和公式，以及从理论生成器生成的概念和公式。
"""

import os
import argparse
from core_embedding.concept_merger import ConceptMerger

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def main():
    parser = argparse.ArgumentParser(description="融合提取的概念和生成的理论")
    
    # 输入参数
    parser.add_argument("--extracted_data", type=str, default="data/extracted_concepts/extracted_data.json",
                        help="从文献中提取的概念和公式数据")
    parser.add_argument("--generated_theories", type=str, default="data/generated_theories/all_quantum_interpretations.json",
                        help="生成的理论数据")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/merged_concepts",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 创建概念融合器
    merger = ConceptMerger()
    
    # 加载提取的数据
    print(f"\n[步骤1] 加载从文献中提取的概念和公式")
    if os.path.exists(args.extracted_data):
        merger.load_extracted_data(args.extracted_data)
    else:
        print(f"[WARN] 提取数据文件不存在: {args.extracted_data}")
        print(f"[INFO] 将继续处理，但没有提取的概念")
    
    # 加载生成的理论
    print(f"\n[步骤2] 加载生成的量子力学诠释理论")
    if os.path.exists(args.generated_theories):
        merger.load_generated_theories(args.generated_theories)
    else:
        print(f"[ERROR] 生成的理论文件不存在: {args.generated_theories}")
        return
    
    # 融合概念和公式
    print(f"\n[步骤3] 融合概念和公式")
    merger.merge_all()
    
    # 保存融合后的数据
    print(f"\n[步骤4] 保存融合后的数据")
    merger.save_to_csv(args.output_dir)
    
    print("\n[完成] 概念融合过程完成!")

if __name__ == "__main__":
    main() 