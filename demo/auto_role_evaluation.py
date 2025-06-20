#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
基于已有实验评估结果的自动角色评估脚本

读取之前的实验评估排名结果，自动对成功率≥80%的理论进行多角色评估
"""

import os
import sys
import json
import argparse
import asyncio
from typing import List, Dict, Any

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from theory_validation.agent_validation.theory_evaluator import TheoryEvaluator
from theory_generation.llm_interface import LLMInterface

async def run_role_evaluation_for_theories(
    high_success_theories: List[Dict[str, Any]],
    all_theories_definitions: Dict[str, Any],
    output_dir: str,
    model_source: str = "deepseek",
    model_name: str = "deepseek-chat"
):
    """
    对给定的高成功率理论列表进行角色评估。
    
    Args:
        high_success_theories: 经实验评估筛选出的高成功率理论信息列表。
                                (e.g., [{'theory_name': 'T1', 'success_rate': 0.9, ...}, ...])
        all_theories_definitions: 包含所有理论定义的字典。
                                (e.g., {'T1': {'name': 'T1', 'philosophy': ...}, ...})
        output_dir: 角色评估结果的输出目录。
        model_source: LLM模型来源。
        model_name: LLM模型名称。
    """
    print("\n" + "="*50)
    print("🚀 开始进行多角色评估...")
    print("="*50)
    print(f"[INFO] 收到 {len(high_success_theories)} 个理论进行角色评估。")
    print(f"[INFO] 使用模型: {model_source}/{model_name}")
    
    # 1. 初始化LLM和角色评估器
    try:
        llm = LLMInterface(model_name=model_name, model_source=model_source)
        role_evaluator = TheoryEvaluator(llm)
        print(f"[INFO] 已初始化角色评估器")
    except Exception as e:
        print(f"[ERROR] 初始化角色评估器失败: {str(e)}")
        return
    
    # 2. 创建输出目录
    role_output_dir = os.path.join(output_dir, "role_evaluations")
    os.makedirs(role_output_dir, exist_ok=True)
    
    # 3. 执行角色评估
    role_results = []
    
    for rank_info in high_success_theories:
        theory_name = rank_info['theory_name']
        
        if theory_name not in all_theories_definitions:
            print(f"[WARNING] 未找到理论 '{theory_name}' 的定义，跳过角色评估")
            continue
            
        theory_data = all_theories_definitions[theory_name]
        
        print(f"\n[INFO] 正在对理论 '{theory_name}' 进行角色评估...")
        print(f"       实验成功率: {rank_info['success_rate']*100:.1f}%, 平均χ²: {rank_info['average_chi2']:.4f}")
        
        try:
            # 执行角色评估
            # 将理论数据适配为角色评估器期望的格式
            adapted_theory = {
                'name': theory_data.get('name', theory_name),
                'core_principles': theory_data.get('summary', ''),
                'detailed_description': f"哲学立场: {theory_data.get('philosophy', {}).get('ontology', '')}\\n测量解释: {theory_data.get('philosophy', {}).get('measurement', '')}",
                'quantum_phenomena_explanation': {
                    'wave_function_collapse': theory_data.get('philosophy', {}).get('measurement', ''),
                    'measurement_problem': theory_data.get('philosophy', {}).get('measurement', ''),
                    'non_locality': theory_data.get('philosophy', {}).get('ontology', '')
                },
                'philosophical_stance': theory_data.get('philosophy', {}),
                'mathematical_formulation': theory_data.get('formalism', {}),
                'parameters': theory_data.get('parameters', {}),
                'semantics': theory_data.get('semantics', {})
            }
            
            # 初始化结果
            role_result = {
                'theory_name': theory_name,
                'theory_id': f"AUTO_{hash(theory_name)}",
                'evaluations': {},
                'avg_chi2': 0,
                'conflicts': [],
                'detailed_results': []
            }
            
            # 直接调用各个角色的评估方法
            for role_id, role_info in role_evaluator.evaluation_roles.items():
                print(f"       正在进行{role_info['name']}评估...")
                eval_result = await role_evaluator._evaluate_as_role(adapted_theory, role_id, role_info)
                role_result['evaluations'][role_id] = eval_result
                score = eval_result.get('score', 0)
                print(f"       {role_info['name']}评估完成，得分: {score}/10")
            
            # 如果有评估结果，生成总结
            if role_result['evaluations']:
                summary = await role_evaluator._generate_evaluation_summary(adapted_theory, role_result['evaluations'])
                role_result['summary'] = summary
            
            # 添加实验成功率信息
            role_result['experiment_success_rate'] = rank_info['success_rate']
            role_result['average_chi2'] = rank_info['average_chi2']
            role_result['experiments_count'] = rank_info['experiments_count']
            
            role_results.append(role_result)
            
            # 保存单个理论的角色评估结果
            theory_role_file = os.path.join(
                role_output_dir,
                f"{theory_name.replace(' ', '_').lower()}_role_evaluation.json"
            )
            with open(theory_role_file, "w", encoding="utf-8") as f:
                json.dump(role_result, f, ensure_ascii=False, indent=2)
            
            print(f"[INFO] 理论 '{theory_name}' 的角色评估结果已保存到: {theory_role_file}")
            
        except Exception as e:
            print(f"[ERROR] 评估理论 '{theory_name}' 的角色时出错: {str(e)}")
    
    # 4. 保存综合结果和排名
    if role_results:
        # 保存角色评估汇总
        role_summary_file = os.path.join(role_output_dir, "role_evaluation_summary.json")
        with open(role_summary_file, "w", encoding="utf-8") as f:
            json.dump(role_results, f, ensure_ascii=False, indent=2)
        
        # 计算综合排名
        combined_rankings = []
        for role_result in role_results:
            theory_name = role_result.get('theory_name', 'Unknown Theory')
            
            # 计算角色评估平均分
            role_scores = []
            if 'evaluations' in role_result and isinstance(role_result['evaluations'], dict):
                for role, eval_data in role_result['evaluations'].items():
                    if isinstance(eval_data, dict) and 'score' in eval_data:
                        score = eval_data['score']
                        if score is not None:
                            role_scores.append(score)
            
            avg_role_score = sum(role_scores) / len(role_scores) if role_scores else 0
            
            # 综合评分 = 实验成功率 * 0.6 + 角色评估分 * 0.4
            combined_score = (
                role_result['experiment_success_rate'] * 0.6 + 
                avg_role_score / 10.0 * 0.4  # 角色评估分通常是1-10分，归一化到0-1
            )
            
            # 收集角色详细评分
            role_details = {}
            if 'evaluations' in role_result and isinstance(role_result['evaluations'], dict):
                for role, eval_data in role_result['evaluations'].items():
                    if isinstance(eval_data, dict) and 'score' in eval_data:
                        role_details[role] = eval_data.get('score', 0)
            
            combined_rankings.append({
                'theory_name': theory_name,
                'experiment_success_rate': role_result['experiment_success_rate'],
                'average_chi2': role_result['average_chi2'],
                'experiments_count': role_result['experiments_count'],
                'average_role_score': avg_role_score,
                'combined_score': combined_score,
                'role_details': role_details
            })
        
        # 按综合评分排序
        combined_rankings.sort(key=lambda x: -x['combined_score'])
        
        # 保存综合排名
        combined_ranking_file = os.path.join(role_output_dir, "combined_rankings.json")
        with open(combined_ranking_file, "w", encoding="utf-8") as f:
            json.dump(combined_rankings, f, ensure_ascii=False, indent=2)
        
        print(f"\n[INFO] 角色评估汇总已保存到: {role_summary_file}")
        print(f"[INFO] 综合排名已保存到: {combined_ranking_file}")
        
        # 打印综合排名
        print("\n" + "="*120)
        print("🏆 综合排名（实验成功率 60% + 角色评估 40%）")
        print("="*120)
        print(f"{'排名':<4} {'理论名称':<40} {'实验成功率':<12} {'角色平均分':<12} {'综合评分':<12} {'详细角色评分'}")
        print("-"*120)
        
        for i, rank in enumerate(combined_rankings, 1):
            role_detail_items = rank.get('role_details', {}).items()
            role_detail = ", ".join([f"{role_evaluator.evaluation_roles.get(role, {}).get('name', '未知')}:{score:.1f}" for role, score in role_detail_items if score > 0])
            print(f"{i:<4} {rank['theory_name']:<40} {rank['experiment_success_rate']*100:>8.1f}% "
                  f"{rank['average_role_score']:>10.1f} {rank['combined_score']:>10.3f}    {role_detail}")
        
        print("-"*120)
        print(f"共完成 {len(combined_rankings)} 个理论的角色评估。")
        
        # 输出最佳理论的详细信息
        if combined_rankings:
            best_theory = combined_rankings[0]
            print(f"\n🥇 最佳理论: {best_theory['theory_name']}")
            print(f"   实验成功率: {best_theory['experiment_success_rate']*100:.1f}%")
            print(f"   平均χ²值: {best_theory['average_chi2']:.4f}")
            print(f"   角色评估平均分: {best_theory['average_role_score']:.1f}/10")
            print(f"   综合评分: {best_theory['combined_score']:.3f}")
        
        return combined_ranking_file
    else:
        print("[WARNING] 没有成功完成的角色评估结果")
        return None

async def standalone_auto_role_evaluation(ranking_files: List[str], theories_dir: str, output_dir: str, 
                              success_threshold: float = 0.8, model_source: str = "deepseek", 
                              model_name: str = "deepseek-chat"):
    """
    基于实验评估排名结果进行自动角色评估 (独立运行模式)
    
    Args:
        ranking_files: 一个或多个理论排名文件路径的列表
        theories_dir: 包含理论定义文件的目录路径
        output_dir: 输出目录
        success_threshold: 成功率阈值 (默认0.8即80%)
        model_source: LLM模型来源
        model_name: LLM模型名称
    """
    print(f"[INFO] 以独立模式运行角色评估...")
    print(f"[INFO] 成功率阈值: {success_threshold*100:.0f}%")
    
    # 1. 读取并合并所有理论排名结果
    all_rankings = []
    for rank_file in ranking_files:
        try:
            with open(rank_file, 'r', encoding='utf-8') as f:
                all_rankings.extend(json.load(f))
            print(f"[INFO] 已加载排名文件: {rank_file}")
        except Exception as e:
            print(f"[ERROR] 读取排名文件 {rank_file} 失败: {str(e)}")
            continue
    
    if not all_rankings:
        print("[ERROR] 未加载任何排名数据，程序退出。")
        return
    
    # 2. 读取理论定义
    all_theories_definitions = {}
    try:
        theory_files = [f for f in os.listdir(theories_dir) if f.endswith('.json')]
        print(f"[INFO] 在目录 '{theories_dir}' 中找到 {len(theory_files)} 个理论定义文件。")
        for file_name in theory_files:
            file_path = os.path.join(theories_dir, file_name)
            with open(file_path, 'r', encoding='utf-8') as f:
                theory_data = json.load(f)
                theory_name = theory_data.get('name')
                if theory_name:
                    all_theories_definitions[theory_name] = theory_data
                else:
                    print(f"[WARN] 文件 {file_name} 中缺少理论名称 'name'。")
    except Exception as e:
        print(f"[ERROR] 读取理论定义目录 '{theories_dir}' 失败: {str(e)}")
        return
    
    # 3. 去重并筛选高成功率理论
    seen_theories = set()
    high_success_theories = []
    for rank in all_rankings:
        t_name = rank.get('theory_name')
        s_rate = rank.get('success_rate')

        if t_name and s_rate is not None and t_name not in seen_theories:
            if s_rate >= success_threshold:
                high_success_theories.append(rank)
            seen_theories.add(t_name)
    
    # 按成功率和χ²排序
    high_success_theories.sort(key=lambda x: (-x['success_rate'], x.get('average_chi2', float('inf'))))

    # 4. 如果有符合条件的理论，则运行角色评估
    if high_success_theories:
        await run_role_evaluation_for_theories(
            high_success_theories=high_success_theories,
            all_theories_definitions=all_theories_definitions,
            output_dir=output_dir,
            model_source=model_source,
            model_name=model_name
        )
    else:
        print(f"[INFO] 没有理论达到 {success_threshold*100:.0f}% 的成功率阈值，角色评估结束。")

def main():
    parser = argparse.ArgumentParser(description="自动进行多角色评估")
    parser.add_argument('ranking_files', nargs='+', help="一个或多个实验评估排名文件的路径 (JSON格式)")
    parser.add_argument('--theories_dir', required=True, help="包含所有理论定义文件的目录路径")
    parser.add_argument('--output_dir', default="data/role_evaluation_results", help="评估结果的输出目录")
    parser.add_argument('--threshold', type=float, default=0.6, help="进行角色评估的实验成功率阈值 (例如 0.6 表示 60%)")
    
    # LLM 相关参数
    parser.add_argument("--model_source", type=str, default="openai", choices=['openai', 'google', 'deepseek'], help="LLM API的服务商")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", help="具体的LLM模型名称")
    
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    asyncio.run(standalone_auto_role_evaluation(
        ranking_files=args.ranking_files,
        theories_dir=args.theories_dir,
        output_dir=args.output_dir,
        success_threshold=args.threshold,
        model_source=args.model_source,
        model_name=args.model_name
    ))

if __name__ == "__main__":
    main() 