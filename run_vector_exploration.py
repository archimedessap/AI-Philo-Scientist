#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行理论向量空间探索

在嵌入空间中表示理论矛盾点，通过放松约束生成新的理论可能性。
"""

import os
import argparse
import asyncio
import json
import traceback
import numpy as np
from theory_generation.llm_interface import LLMInterface
from theory_generation.vector_space_explorer import VectorSpaceExplorer

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def main():
    parser = argparse.ArgumentParser(description="运行向量空间探索")
    parser.add_argument("--model_source", type=str, default="deepseek", help="模型来源 (openai 或 deepseek)")
    parser.add_argument("--model_name", type=str, default="deepseek-chat", help="模型名称")
    parser.add_argument("--exploration_method", type=str, default="contradiction", 
                        choices=["contradiction", "list_theories"],
                        help="探索方法")
    parser.add_argument("--theory1", type=str, help="第一个理论名称")
    parser.add_argument("--theory2", type=str, help="第二个理论名称")
    parser.add_argument("--num_variants", type=int, default=3, help="生成的理论变体数量")
    parser.add_argument("--temperature", type=float, default=0.7, help="LLM生成温度")
    parser.add_argument("--output_dir", type=str, default="data/new_theories", help="输出目录")
    
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化组件
    llm = LLMInterface()
    llm.set_model(args.model_source, args.model_name)
    explorer = VectorSpaceExplorer(llm)
    
    # 加载数据
    print("[步骤1] 加载理论嵌入和矛盾点数据")
    embeddings_path = os.path.join("data", "embeddings", "theory_embeddings.pkl")
    contradictions_path = os.path.join("data", "theory_analysis", "theory_contradictions.json")
    
    if not explorer.load_embeddings(embeddings_path):
        print("[错误] 加载理论嵌入失败，退出程序")
        return
    
    if not explorer.load_contradictions(contradictions_path):
        print("[警告] 加载矛盾点数据失败，某些功能可能受限")
    
    # 仅列出可用理论名称
    if args.exploration_method == "list_theories":
        explorer.print_available_theories()
        return
    
    print(f"\n[步骤2] 根据选择的方法生成新理论")
    
    if args.exploration_method == "contradiction":
        # 如果指定了理论名称，使用指定的理论
        if args.theory1 and args.theory2:
            if args.theory1 not in explorer.theory_embeddings or args.theory2 not in explorer.theory_embeddings:
                print(f"[错误] 指定的理论名称不存在，请使用 --exploration_method list_theories 查看可用理论")
                explorer.print_available_theories()
                return
                
            try:
                # 这里改为await
                print(f"[INFO] 使用矛盾点放松+最近邻方法生成新理论: {args.theory1} vs {args.theory2}")
                new_theories = await explorer.generate_relaxed_theories(
                    args.theory1, args.theory2, 
                    model_source=args.model_source,
                    model_name=args.model_name,
                    num_variants=args.num_variants,
                    temperature=args.temperature
                )
                
                # 保存结果
                output_file = os.path.join(args.output_dir, f"relaxed_theories_{args.theory1}_vs_{args.theory2}.json")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        "theory1": args.theory1,
                        "theory2": args.theory2,
                        "method": "contradiction_relaxation_with_nearest",
                        "model_source": args.model_source,
                        "model_name": args.model_name,
                        "generated_theories": new_theories
                    }, f, ensure_ascii=False, indent=2)
                
                print(f"[成功] 已生成 {len(new_theories)} 个新理论变体，保存到: {output_file}")
                
                # 在输出结果时，添加实际使用的模型信息
                print(f"[INFO] 使用的模型来源: {args.model_source}")
                print(f"[INFO] 使用的模型名称: {args.model_name}")
                
            except Exception as e:
                print(f"[错误] 处理理论对 {args.theory1} vs {args.theory2} 时出错: {str(e)}")
                print("[错误] 详细错误信息:")
                traceback.print_exc()
            
            return  # 成功完成，退出程序
        else:
            print("[错误] 使用 contradiction 方法需要指定 --theory1 和 --theory2 参数")
            return
    
    else:
        print(f"[错误] 不支持的探索方法: {args.exploration_method}")
        print("支持的方法: contradiction, list_theories")

if __name__ == "__main__":
    asyncio.run(main()) 