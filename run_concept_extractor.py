#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行概念提取器
python run_concept_extractor.py --use_preprocessed --model_source deepseek --model_name deepseek-chat --disable_fallback
从文献资料中提取量子理论相关的概念和公式。
"""

import os
import argparse
import glob
from theory_generation.llm_interface import LLMInterface
from data_preparation.concept_extraction.concept_extractor import ConceptExtractor

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def main():
    parser = argparse.ArgumentParser(description="从文献中提取概念和公式")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="deepseek",
                        choices=["openai", "deepseek", "ollama", "groq", "auto"],
                        help="LLM模型来源")
    parser.add_argument("--model_name", type=str, default="deepseek-chat",
                        help="LLM模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥（可选，默认使用.env中的配置）")
    
    # 输入输出参数
    parser.add_argument("--input_dir", type=str, default="data/raw_literature",
                        help="输入文献目录")
    parser.add_argument("--preprocessed_dir", type=str, default="data/preprocessed",
                        help="预处理后的文件目录")
    parser.add_argument("--output_dir", type=str, default="data/extracted_concepts",
                        help="输出目录")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")
    
    # 添加新的命令行参数
    parser.add_argument("--request_interval", type=float, default=1.0,
                      help="API请求之间的最小间隔(秒)")
    parser.add_argument("--raise_api_error", action="store_true",
                      help="API错误时直接抛出异常而不尝试回退")
    parser.add_argument("--disable_fallback", action="store_true",
                      help="禁用模型自动回退功能，当指定模型失败时不会尝试其他模型")
    parser.add_argument("--use_preprocessed", action="store_true",
                      help="使用预处理后的文件而不是原始文献")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=args.request_interval
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 创建概念提取器
    extractor = ConceptExtractor(llm)
    
    # 从预处理文件或原始文献提取
    if args.use_preprocessed:
        # 检查预处理目录是否存在
        if not os.path.exists(args.preprocessed_dir):
            print(f"[错误] 预处理目录不存在: {args.preprocessed_dir}")
            return
        
        # 查找所有JSON文件
        json_files = glob.glob(os.path.join(args.preprocessed_dir, "*.json"))
        if not json_files:
            print(f"[错误] 在预处理目录中未找到JSON文件: {args.preprocessed_dir}")
            return
        
        # 从预处理文件提取概念和公式
        print(f"\n[步骤1] 从预处理文件提取概念和公式: {args.preprocessed_dir}")
        print(f"[INFO] 在 {args.preprocessed_dir} 中找到了 {len(json_files)} 个文件")
        
        for json_file in json_files:
            print(f"[INFO] 处理文件: {os.path.basename(json_file)}")
            extractor.extract_from_preprocessed(json_file)
    else:
        # 检查输入目录是否存在
        if not os.path.exists(args.input_dir):
            print(f"[错误] 输入目录不存在: {args.input_dir}")
            return
        
        # 从目录提取概念和公式
        print(f"\n[步骤1] 从目录提取概念和公式: {args.input_dir}")
        concepts, formulas = extractor.extract_from_directory(args.input_dir)
    
    print(f"[INFO] 去重后共有 {len(extractor.extracted_concepts)} 个概念和 {len(extractor.extracted_formulas)} 个公式")
    print(f"[结果] 总共提取到 {len(extractor.extracted_concepts)} 个概念和 {len(extractor.extracted_formulas)} 个公式")
    
    # 保存提取的概念和公式
    print(f"\n[步骤2] 保存提取的概念和公式到: {args.output_dir}")
    extractor.save_to_csv(args.output_dir)
    print(f"[INFO] 提取的概念和公式已保存")
    
    print("\n[完成] 概念提取过程完成!")

if __name__ == "__main__":
    main()