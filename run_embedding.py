#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行概念嵌入器
python run_embedding.py --model_source openai --model_name text-embedding-3-small --request_interval 2.0 --input_file data/merged_concepts/merged_data.json
将融合后的概念、公式和理论嵌入到高维向量空间，并保存结果。
"""

import os
import json
import argparse
import asyncio
from theory_generation.llm_interface import LLMInterface
from core_embedding.embedding import ConceptEmbedder

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def load_merged_data(merged_data_path):
    """加载融合后的数据"""
    if not os.path.exists(merged_data_path):
        print(f"[ERROR] 融合数据文件不存在: {merged_data_path}")
        return None, None, None
        
    try:
        with open(merged_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        concepts = data.get('concepts', [])
        formulas = data.get('formulas', [])
        
        # 提取理论信息
        theories = []
        theory_names = set()
        
        # 从概念中提取理论信息
        for concept in concepts:
            theory_list = concept.get('theories', [])
            for theory in theory_list:
                if theory and theory not in theory_names:
                    theory_names.add(theory)
                    theories.append({
                        "name": theory,
                        "description": f"Quantum interpretation theory: {theory}",
                        "philosophical_assumptions": "Extracted from concept relationships"
                    })
        
        print(f"[INFO] 已加载 {len(concepts)} 个概念, {len(formulas)} 个公式, 识别到 {len(theories)} 个理论")
        return concepts, formulas, theories
    except Exception as e:
        print(f"[ERROR] 加载融合数据失败: {str(e)}")
        return None, None, None

async def main():
    parser = argparse.ArgumentParser(description="生成概念、公式和理论的嵌入向量")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "ollama", "groq", "auto"],
                        help="LLM模型来源")
    parser.add_argument("--model_name", type=str, default="text-embedding-3-small",
                        help="LLM嵌入模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥（可选，默认使用.env中的配置）")
    
    # 输入输出参数
    parser.add_argument("--input_file", type=str, default="data/merged_concepts/merged_data.json",
                        help="融合数据的输入文件")
    parser.add_argument("--output_dir", type=str, default="data/embeddings",
                        help="嵌入向量的输出目录")
    parser.add_argument("--visualize", action="store_true",
                        help="是否生成可视化图表")
    parser.add_argument("--request_interval", type=float, default=0.5,
                        help="API请求之间的最小间隔(秒)")
    parser.add_argument("--disable_fallback", action="store_true",
                        help="禁用模型自动回退功能")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=2.0
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 创建概念嵌入器
    embedder = ConceptEmbedder(llm)
    
    # 加载融合后的数据
    print(f"\n[步骤1] 加载融合数据: {args.input_file}")
    concepts, formulas, theories = load_merged_data(args.input_file)
    
    if not concepts and not formulas:
        print("[ERROR] 没有找到可嵌入的概念或公式，程序终止")
        return
    
    # 嵌入概念
    if concepts:
        print(f"\n[步骤2] 嵌入 {len(concepts)} 个概念")
        try:
            concept_embeddings = await embedder.embed_concepts(concepts)
            print(f"[SUCCESS] 成功嵌入 {len(concept_embeddings)} 个概念")
        except Exception as e:
            print(f"[ERROR] 概念嵌入失败: {str(e)}")
    
    # 嵌入公式
    if formulas:
        print(f"\n[步骤3] 嵌入 {len(formulas)} 个公式")
        try:
            formula_embeddings = await embedder.embed_formulas(formulas)
            print(f"[SUCCESS] 成功嵌入 {len(formula_embeddings)} 个公式")
        except Exception as e:
            print(f"[ERROR] 公式嵌入失败: {str(e)}")
    
    # 嵌入理论
    if theories:
        print(f"\n[步骤4] 嵌入 {len(theories)} 个理论")
        try:
            theory_embeddings = await embedder.embed_theories(theories)
            print(f"[SUCCESS] 成功嵌入 {len(theory_embeddings)} 个理论")
        except Exception as e:
            print(f"[ERROR] 理论嵌入失败: {str(e)}")
    
    # 保存嵌入向量
    print(f"\n[步骤5] 保存嵌入向量")
    concept_path, formula_path, theory_path = embedder.save_embeddings(args.output_dir)
    print(f"[INFO] 嵌入向量已保存到: {args.output_dir}")
    
    # 生成可视化
    if args.visualize:
        print(f"\n[步骤6] 生成嵌入空间可视化")
        try:
            viz_path = os.path.join(args.output_dir, "embeddings_visualization.png")
            embedder.visualize_embeddings(viz_path)
            print(f"[SUCCESS] 可视化图表已保存到: {viz_path}")
        except Exception as e:
            print(f"[ERROR] 生成可视化失败: {str(e)}")
    
    print("\n[完成] 概念、公式和理论嵌入流程已完成!")

if __name__ == "__main__":
    asyncio.run(main()) 