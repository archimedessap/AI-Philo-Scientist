#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
运行量子理论评估
python run_theory_evaluation.py --model_source deepseek --model_name deepseek-chat --theories_file data/innovated_theories/deepseek_chat/all_innovated_theories.json --output_dir data/evaluated_theories/deepseek_chat
从物理学家、哲学家和数学家多个角度评估生成的新量子诠释理论。
"""

import os
import argparse
import asyncio
import json
from theory_generation.llm_interface import LLMInterface
from theory_generation.theory_evaluator import TheoryEvaluator
import glob

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def safe_get_nested(obj, path, default=''):
    """安全获取嵌套字段，无论是字典还是字符串"""
    if isinstance(obj, dict) and path in obj:
        value = obj[path]
        if isinstance(value, dict):
            return value  # 返回整个字典以便进一步嵌套访问
        return value  # 返回非字典值
    return default  # 字段不存在或obj不是字典

async def main():
    parser = argparse.ArgumentParser(description="评估新生成的量子诠释理论")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="openai",
                        choices=["openai", "deepseek", "ollama", "groq", "auto"],
                        help="LLM模型来源")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini",
                        help="LLM模型名称")
    parser.add_argument("--api_key", type=str, default=None,
                        help="API密钥（可选，默认使用.env中的配置）")
    
    # 输入输出参数
    parser.add_argument("--theories_file", type=str, 
                        default="data/new_theories/new_theories.json",
                        help="要评估的理论文件")
    parser.add_argument("--output_dir", type=str, default="data/evaluated_theories",
                        help="结果输出目录")
    parser.add_argument("--top_n", type=int, default=None,
                        help="只评估前N个理论，默认评估所有理论")
    parser.add_argument("--extract_threshold", type=float, default=7.0,
                        help="提取高分理论的分数阈值")
    parser.add_argument("--immediate_extract", action="store_true", default=True,
                        help="立即提取高分理论，不等待全部评估完成")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    ensure_directory_exists(args.output_dir)
    
    # 初始化LLM接口
    llm = LLMInterface(
        model_source=args.model_source,
        model_name=args.model_name,
        request_interval=1.0
    )
    
    # 显示当前使用的模型信息
    model_info = llm.get_current_model_info()
    print(f"[INFO] 当前使用的模型: {model_info['source']} - {model_info['name']}")
    
    # 创建理论评估器
    evaluator = TheoryEvaluator(llm)
    
    # 加载理论
    if not os.path.exists(args.theories_file):
        print(f"[ERROR] 理论文件不存在: {args.theories_file}")
        return
        
    try:
        with open(args.theories_file, 'r', encoding='utf-8') as f:
            theories = json.load(f)
            
        if args.top_n and args.top_n < len(theories):
            theories = theories[:args.top_n]
            
        print(f"\n[步骤1] 加载了 {len(theories)} 个要评估的理论")
    except Exception as e:
        print(f"[ERROR] 加载理论文件失败: {str(e)}")
        return
    
    # 评估理论
    print("\n[步骤2] 开始多角色评估")
    
    # 创建保存单个理论评估结果的目录
    individual_results_dir = os.path.join(args.output_dir, "individual_theories")
    ensure_directory_exists(individual_results_dir)
    
    # 修改:逐个评估并立即保存每个理论的结果
    results = []
    for i, theory in enumerate(theories):
        print(f"\n[评估中] 理论 {i+1}/{len(theories)}: {theory.get('name', '未命名')}")
        try:
            # 评估单个理论
            result = await evaluator.evaluate_theory(theory)
            results.append(result)
            
            # 为每个理论创建单独的评估文件
            theory_name = theory.get('name', 'unnamed').replace(' ', '_').lower()
            theory_id = theory.get('id', f'theory_{i}')
            individual_output = os.path.join(individual_results_dir, f"{theory_id}_{theory_name}_evaluation.json")
            
            # 保存单个理论评估结果
            with open(individual_output, 'w', encoding='utf-8') as f:
                json.dump(result, f, ensure_ascii=False, indent=2)
            print(f"[保存] 理论评估结果已保存至: {individual_output}")
            
            # 额外：创建简化版本用于实验测试(仅包含理论和评估得分)
            experiment_ready = {
                "theory": theory,
                "evaluation": {
                    "overall_score": result.get('overall_score', 0),
                    "recommendation": result.get('recommendation', ''),
                    "evaluator_feedback": result.get('evaluator_feedback', {})
                }
            }
            experiment_file = os.path.join(individual_results_dir, f"{theory_id}_{theory_name}_for_experiment.json")
            with open(experiment_file, 'w', encoding='utf-8') as f:
                json.dump(experiment_ready, f, ensure_ascii=False, indent=2)
            print(f"[保存] 用于实验测试的文件已保存至: {experiment_file}")
            
            # 添加：即时提取高分理论
            if args.immediate_extract:
                score = result.get('overall_score', 0)
                if score >= args.extract_threshold:
                    # 生成输出文件名，包含评分
                    sanitized_name = theory_name.lower().replace(" ", "_").replace("-", "_")
                    high_quality_dir = "data/high_quality_theories"
                    ensure_directory_exists(high_quality_dir)
                    
                    # 保存转换后的理论数据
                    filename = f"{theory_id}_score_{score:.1f}_{sanitized_name}.json"
                    output_path = os.path.join(high_quality_dir, filename)
                    
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(theory, f, ensure_ascii=False, indent=2)
                    
                    print(f"[即时提取] 高分理论 ({score:.1f}分) 已保存到: {output_path}")
        
        except Exception as e:
            print(f"[ERROR] 评估理论 '{theory.get('name', '未命名')}' 失败: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # 保存所有评估结果(保留原有功能)
    output_path = os.path.join(args.output_dir, "theory_evaluations.json")
    evaluator.save_evaluation_results(output_path)
    
    # 提取排名前三的理论
    sorted_theories = sorted(results, key=lambda x: x.get('overall_score', 0), reverse=True)
    top_theories = sorted_theories[:3] if len(sorted_theories) >= 3 else sorted_theories
    
    print(f"\n[完成] 评估完成，结果已保存到: {output_path}")
    print("\n评分最高的理论:")
    for i, theory in enumerate(top_theories):
        name = theory.get('theory_name', 'Unknown')
        score = theory.get('overall_score', 0)
        recommendation = theory.get('recommendation', 'No recommendation')
        print(f"{i+1}. {name} (评分: {score:.2f}/10)")
        print(f"   推荐意见: {recommendation[:100]}..." if len(recommendation) > 100 else f"   推荐意见: {recommendation}")
        print()

    # 评估完成后，筛选并导出高分理论
    print("\n[步骤4] 筛选高分理论")
    extract_high_scoring_theories(args.output_dir, threshold=args.extract_threshold)

def extract_high_scoring_theories(output_dir, threshold=7.0, destination_dir="data/high_quality_theories"):
    """筛选并导出高分理论
    
    Args:
        output_dir: 评估结果存放目录
        threshold: 分数阈值，默认7.0
        destination_dir: 高分理论输出目录
    """
    # 确保输出目录存在
    ensure_directory_exists(destination_dir)
    
    # 查找所有评估结果文件
    evaluated_files = glob.glob(os.path.join(output_dir, "**/*.json"), recursive=True)
    
    high_scoring_count = 0
    
    for file_path in evaluated_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查是否包含评估结果和分数
            if "evaluation" in data and "overall_score" in data["evaluation"]:
                score = data["evaluation"]["overall_score"]
                
                # 筛选高分理论
                if score >= threshold:
                    # 理论数据格式转换：从嵌套格式转为根级格式
                    if "theory" in data:
                        theory_data = data["theory"]
                    else:
                        continue  # 跳过格式不正确的文件
                    
                    # 生成输出文件名，包含评分
                    theory_id = theory_data.get("id", "unknown")
                    theory_name = theory_data.get("name", "unnamed")
                    sanitized_name = theory_name.lower().replace(" ", "_").replace("-", "_")
                    filename = f"{theory_id}_score_{score:.1f}_{sanitized_name}.json"
                    output_path = os.path.join(destination_dir, filename)
                    
                    # 保存转换后的理论数据
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(theory_data, f, ensure_ascii=False, indent=2)
                    
                    high_scoring_count += 1
                    print(f"[INFO] 导出高分理论: {theory_name} (评分: {score:.2f})")
        
        except Exception as e:
            print(f"[ERROR] 处理文件 {file_path} 时出错: {str(e)}")
    
    print(f"[完成] 共发现 {high_scoring_count} 个高分理论（分数≥{threshold}），已保存至 {destination_dir}")

if __name__ == "__main__":
    asyncio.run(main()) 