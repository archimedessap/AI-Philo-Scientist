#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
理论验证程序

对生成的量子理论假说进行多角度验证，
集成已有实验评估方法并添加更多验证维度。
"""

import os
import json
import argparse
import asyncio
import time
from theory_generation.llm_interface import LLMInterface
from theory_validation.validation_framework import TheoryValidationFramework
from theory_validation.validators.consistency_validator import ConsistencyValidator
from theory_validation.validators.experiment_compatibility_validator import ExperimentCompatibilityValidator
from theory_validation.validators.agent_evaluation_validator import AgentEvaluationValidator

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def load_theories(theories_path):
    """从目录或文件加载理论数据"""
    theories = []
    
    # 判断是单个文件还是目录
    if os.path.isfile(theories_path) and theories_path.endswith('.json'):
        # 单个JSON文件
        try:
            with open(theories_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # 判断是单个理论还是理论列表
            if isinstance(data, list):
                theories.extend(data)
            else:
                theories.append(data)
                
        except Exception as e:
            print(f"[ERROR] 加载理论文件失败: {str(e)}")
    else:
        # 目录中的所有JSON文件
        for root, _, files in os.walk(theories_path):
            for file in files:
                if file.endswith('.json'):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                        # 判断是单个理论还是理论列表
                        if isinstance(data, list):
                            theories.extend(data)
                        else:
                            theories.append(data)
                            
                    except Exception as e:
                        print(f"[ERROR] 加载理论文件 {file} 失败: {str(e)}")
    
    print(f"[INFO] 共加载了 {len(theories)} 个理论")
    return theories

async def main():
    parser = argparse.ArgumentParser(description="理论验证程序")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="deepseek",
                        choices=["openai", "deepseek"],
                        help="模型来源")
    parser.add_argument("--model_name", type=str, default="deepseek-chat",
                        help="模型名称")
    
    # 输入参数
    parser.add_argument("--theories_path", type=str, required=True,
                        help="理论文件或目录路径")
    parser.add_argument("--specific_theory", type=str, default=None,
                        help="要验证的特定理论名称或ID")
    parser.add_argument("--experiments_file", type=str, 
                        default="theory_experiment/data/experiments.jsonl",
                        help="实验数据文件路径")
    
    # 验证参数
    parser.add_argument("--skip_consistency", action="store_true",
                        help="跳过一致性验证")
    parser.add_argument("--skip_experiment", action="store_true",
                        help="跳过实验验证")
    parser.add_argument("--skip_agent", action="store_true",
                        help="跳过专家角色评估")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, default="data/validated_theories",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    validation_dir = os.path.join(args.output_dir, f"validation_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_directory_exists(validation_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=1.0
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 1. 加载理论数据
    print(f"\n[步骤1] 从 {args.theories_path} 加载理论数据")
    theories = await load_theories(args.theories_path)
    
    if not theories:
        print("[ERROR] 未加载到理论数据，程序终止")
        return
    
    # 2. 如果指定了特定理论，筛选出来
    if args.specific_theory:
        filtered_theories = []
        for theory in theories:
            theory_name = theory.get("name", theory.get("theory_name", ""))
            theory_id = theory.get("id", "")
            
            if (args.specific_theory.lower() in theory_name.lower() or 
                args.specific_theory == theory_id):
                filtered_theories.append(theory)
        
        if filtered_theories:
            theories = filtered_theories
            print(f"[INFO] 已筛选出 {len(theories)} 个匹配的理论")
        else:
            print(f"[WARN] 未找到匹配的理论: {args.specific_theory}，将验证所有理论")
    
    # 3. 初始化验证框架和验证器
    print(f"\n[步骤2] 初始化验证框架")
    framework = TheoryValidationFramework(llm)
    
    # 注册验证器
    if not args.skip_consistency:
        framework.register_validator(ConsistencyValidator(llm))
    
    if not args.skip_experiment:
        framework.register_validator(ExperimentCompatibilityValidator(llm, args.experiments_file))
    
    if not args.skip_agent:
        framework.register_validator(AgentEvaluationValidator(llm))
    
    # 4. 执行验证
    print(f"\n[步骤3] 开始理论验证")
    all_results = await framework.validate_multiple_theories(theories)
    
    # 5. 保存验证结果
    print(f"\n[步骤4] 保存验证结果")
    
    # 保存个别理论结果
    for theory_id, result in all_results["individual_results"].items():
        theory_name = result["theory_name"].replace(" ", "_").lower()
        output_file = os.path.join(validation_dir, f"{theory_name}_validation.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[INFO] 保存理论 {theory_name} 的验证结果到: {output_file}")
        except Exception as e:
            print(f"[ERROR] 保存理论 {theory_name} 的验证结果失败: {str(e)}")
    
    # 保存比较结果
    comparison_file = os.path.join(validation_dir, "theories_comparison.json")
    try:
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(all_results["comparison"], f, ensure_ascii=False, indent=2)
        print(f"[INFO] 保存理论比较结果到: {comparison_file}")
    except Exception as e:
        print(f"[ERROR] 保存理论比较结果失败: {str(e)}")
    
    # 6. 打印摘要
    print(f"\n[步骤5] 验证摘要")
    
    # 按总体分数排序
    ranked_theories = all_results["comparison"]["ranked_theories"]
    
    print(f"理论排名 (按总体评分):")
    for i, theory in enumerate(ranked_theories, 1):
        print(f"{i}. {theory['theory_name']}: {theory['overall_score']:.2f}/10")
    
    print(f"\n[完成] 理论验证完成，结果已保存到: {validation_dir}")

if __name__ == "__main__":
    asyncio.run(main())
