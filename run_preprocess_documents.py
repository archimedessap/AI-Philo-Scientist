#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
预处理量子理论文献

从PDF和Word文件中提取、分割和清洗文本，确保每个部分不超过指定的token限制。
"""

import os
import argparse
import glob
from document_preprocessor import DocumentPreprocessor

def main():
    parser = argparse.ArgumentParser(description="预处理量子理论文献，分割成适合LLM处理的片段")
    
    parser.add_argument("--input_dir", type=str, default="data/raw_literature",
                        help="原始文献目录")
    parser.add_argument("--output_dir", type=str, default="data/preprocessed",
                        help="预处理后文档的保存目录")
    parser.add_argument("--max_tokens", type=int, default=50000,
                        help="每个文档片段的最大token数量")
    parser.add_argument("--single_file", type=str, default=None,
                        help="处理单个文件（提供完整路径）")
    
    args = parser.parse_args()
    
    # 创建预处理器
    preprocessor = DocumentPreprocessor(
        output_dir=args.output_dir,
        max_tokens=args.max_tokens
    )
    
    # 处理单个文件或整个目录
    if args.single_file:
        if not os.path.exists(args.single_file):
            print(f"[ERROR] 文件不存在: {args.single_file}")
            return
            
        print(f"[INFO] 处理单个文件: {args.single_file}")
        preprocessor.preprocess_file(args.single_file)
    else:
        # 确保输入目录存在
        if not os.path.isdir(args.input_dir):
            print(f"[ERROR] 输入目录不存在: {args.input_dir}")
            return
            
        # 查找所有PDF和Word文件
        supported_files = []
        for ext in ['.pdf', '.docx', '.doc']:
            supported_files.extend(glob.glob(os.path.join(args.input_dir, "**", f"*{ext}"), recursive=True))
        
        if not supported_files:
            print(f"[WARN] 在 {args.input_dir} 中未找到支持的文件")
            return
            
        print(f"[INFO] 在 {args.input_dir} 中找到 {len(supported_files)} 个文件")
        
        # 处理每个文件
        for i, file_path in enumerate(supported_files, 1):
            file_ext = os.path.splitext(file_path)[1].lower()
            file_type = "PDF" if file_ext == '.pdf' else "Word"
            print(f"[INFO] ({i}/{len(supported_files)}) 处理{file_type}文件: {os.path.basename(file_path)}")
            preprocessor.preprocess_file(file_path)
    
    print(f"\n[完成] 文档预处理完毕！预处理后的文件保存在: {args.output_dir}")
    print("[提示] 使用这些预处理后的文件进行概念提取时，可能会得到更好的结果")
    print("[依赖] 确保已安装必要的库：pip install pypdf python-docx")

if __name__ == "__main__":
    main() 