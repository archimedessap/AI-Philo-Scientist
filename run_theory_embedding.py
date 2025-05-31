#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行理论嵌入器
python run_embedding.py --model_source openai --model_name text-embedding-3-small --input_file data/merged_concepts/merged_data.json
将量子诠释理论嵌入到高维向量空间，便于后续的理论创新分析。
"""

import os
import json
import argparse
import asyncio
import glob
from theory_generation.llm_interface import LLMInterface
from core_embedding.embedding import ConceptEmbedder

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def main():
    parser = argparse.ArgumentParser(description="生成量子诠释理论的嵌入向量")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "ollama", "groq", "auto"],
                        help="LLM模型来源")
    parser.add_argument("--model_name", type=str, default="text-embedding-3-small",
                        help="LLM嵌入模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥（可选，默认使用.env中的配置）")
    
    # 输入输出参数
    parser.add_argument("--theories_dir", type=str, default="data/generated_theories",
                        help="理论JSON文件所在目录")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="嵌入向量的输出目录")
    parser.add_argument("--request_interval", type=float, default=0.5,
                        help="API请求之间的最小间隔(秒)")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=None if args.model_source == "auto" else args.model_source,
        model_name=args.model_name,
        api_key=args.api_key,
        auto_select=args.model_source == "auto",
        enable_fallback=False,
        request_interval=args.request_interval
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 创建概念嵌入器
    embedder = ConceptEmbedder(llm)
    
    # 加载现有嵌入向量(如果存在)
    embeddings_file = os.path.join(args.output_dir, "theory_embeddings.pkl")
    if os.path.exists(embeddings_file):
        embedder.load_embeddings(args.output_dir)
    
    # 加载所有理论JSON文件
    theory_files = glob.glob(os.path.join(args.theories_dir, "*.json"))
    print(f"[INFO] 在目录 {args.theories_dir} 中找到 {len(theory_files)} 个理论文件")
    
    # 准备理论列表
    theories = []
    for file_path in theory_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
                
            # 确保理论有name字段
            if isinstance(theory_data, dict) and "name" in theory_data:
                theories.append(theory_data)
            # 处理特殊情况：all_quantum_interpretations.json可能包含多个理论
            elif isinstance(theory_data, list):
                for item in theory_data:
                    if isinstance(item, dict) and "name" in item:
                        theories.append(item)
        except Exception as e:
            print(f"[WARNING] 无法加载理论文件 {file_path}: {str(e)}")
    
    print(f"[INFO] 共加载 {len(theories)} 个理论进行嵌入")
    
    # 嵌入理论
    if theories:
        print(f"[INFO] 开始嵌入理论...")
        try:
            theory_embeddings = await embedder.embed_theories(theories)
            print(f"[SUCCESS] 成功嵌入 {len(theory_embeddings)} 个理论")
        except Exception as e:
            print(f"[ERROR] 理论嵌入失败: {str(e)}")
    
    # 保存嵌入向量
    print(f"[INFO] 保存嵌入向量到 {args.output_dir}")
    embedder.save_embeddings(args.output_dir)
    
    print(f"[完成] 理论嵌入完成！")

if __name__ == "__main__":
    asyncio.run(main()) 