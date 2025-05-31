#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行理论生成器
python run_theory_generator.py --model_source deepseek --model_name deepseek-chat --num_theories 25 --disable_fallback
生成量子力学的各种诠释理论。
"""

import os
import argparse
import json
from theory_generation.llm_interface import LLMInterface
from theory_generation.prior_theory_generator import PriorTheoryGenerator

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def main():
    parser = argparse.ArgumentParser(description="生成量子力学的诠释理论")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="deepseek",
                        choices=["openai", "deepseek", "ollama", "groq", "auto"],
                        help="LLM模型来源")
    parser.add_argument("--model_name", type=str, default="deepseek-chat",
                        help="LLM模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥（可选，默认使用.env中的配置）")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/generated_theories",
                        help="输出目录")
    parser.add_argument("--num_theories", type=int, default=20,
                        help="生成理论的最大数量")
    parser.add_argument("--verbose", action="store_true",
                        help="显示详细信息")
    
    # API控制参数
    parser.add_argument("--request_interval", type=float, default=1.0,
                      help="API请求之间的最小间隔(秒)")
    parser.add_argument("--raise_api_error", action="store_true",
                      help="API错误时直接抛出异常而不尝试回退")
    parser.add_argument("--disable_fallback", action="store_true",
                      help="禁用模型自动回退功能，当指定模型失败时不会尝试其他模型")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    api_key = args.api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        print("[错误] 未提供API密钥，请使用--api_key参数或设置环境变量")
        return

    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=args.request_interval
    )
    llm.enable_fallback = not args.disable_fallback
    llm.raise_api_error = args.raise_api_error
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 创建理论生成器
    generator = PriorTheoryGenerator(llm)
    
    # 生成量子力学诠释理论列表
    print(f"\n[步骤1] 生成量子力学的诠释理论列表")
    theory_list = generator.generate_quantum_interpretation_list(max_theories=args.num_theories)
    print(f"[INFO] 已生成{len(theory_list)}个量子力学诠释理论名称")
    
    # 保存理论列表
    theory_list_path = os.path.join(args.output_dir, "quantum_interpretations_list.json")
    with open(theory_list_path, 'w', encoding='utf-8') as f:
        json.dump(theory_list, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 理论列表已保存到: {theory_list_path}")
    
    # 为每个理论生成详细描述
    print(f"\n[步骤2] 为每个理论生成详细描述")
    all_theories = []
    
    for i, theory_name in enumerate(theory_list, 1):
        print(f"[INFO] ({i}/{len(theory_list)}) 生成理论: {theory_name}")
        try:
            theory_details = generator.generate_quantum_interpretation_details(theory_name)
            all_theories.append(theory_details)
            
            # 保存单个理论详情
            theory_filename = theory_name.lower().replace(' ', '_').replace('-', '_')
            theory_path = os.path.join(args.output_dir, f"{theory_filename}.json")
            with open(theory_path, 'w', encoding='utf-8') as f:
                json.dump(theory_details, f, ensure_ascii=False, indent=2)
            
            print(f"[SUCCESS] 已保存理论: {theory_name}")
            
        except Exception as e:
            import traceback
            print(f"[ERROR] 处理理论 '{theory_name}' 时发生错误:")
            print(traceback.format_exc())
            print(f"继续处理下一个理论...")
    
    # 保存所有理论
    all_theories_path = os.path.join(args.output_dir, "all_quantum_interpretations.json")
    with open(all_theories_path, 'w', encoding='utf-8') as f:
        json.dump(all_theories, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 所有理论已保存到: {all_theories_path}")
    
    print("\n[完成] 量子力学诠释理论生成完成!")

if __name__ == "__main__":
    main()