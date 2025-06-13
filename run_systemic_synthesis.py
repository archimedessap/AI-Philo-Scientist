#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
系统性理论合成程序

基于多个量子理论的共同假设分析，识别系统性缺陷，
并生成突破现有框架的革命性量子理论。

使用方法:
python run_systemic_synthesis.py --model_source deepseek --model_name deepseek-chat
"""

import os
import json
import argparse
import asyncio
import time
import glob
from typing import Dict, List

from theory_generation.llm_interface import LLMInterface
from theory_generation.systemic_synthesis.common_assumption_analyzer import CommonAssumptionAnalyzer
from theory_generation.systemic_synthesis.revolutionary_generator import RevolutionaryGenerator

def load_theories_from_sources(theories_dir: str, theories_json_file: str = None, schema_version: str = "2.1") -> Dict[str, Dict]:
    """
    从多个来源加载理论数据
    
    Args:
        theories_dir: 理论目录
        theories_json_file: 可选的理论JSON文件
        schema_version: 要求加载的理论schema版本
        
    Returns:
        Dict[str, Dict]: 理论名称到理论数据的映射
    """
    theories = {}
    
    # 从目录加载理论
    if theories_dir and os.path.exists(theories_dir):
        theory_files = glob.glob(os.path.join(theories_dir, "*.json"))
        print(f"[INFO] 在目录 {theories_dir} 中找到 {len(theory_files)} 个理论文件")
        
        for theory_file in theory_files:
            try:
                with open(theory_file, 'r', encoding='utf-8') as f:
                    theory = json.load(f)
                
                # 检查schema版本
                file_schema_version = theory.get("metadata", {}).get("schema_version")
                if schema_version and file_schema_version != schema_version:
                    print(f"[WARN] 跳过文件 {os.path.basename(theory_file)}: schema版本不匹配 (需要 {schema_version}, 文件为 {file_schema_version})")
                    continue

                theory_name = theory.get("name", os.path.basename(theory_file))
                theories[theory_name] = theory
            except Exception as e:
                print(f"[ERROR] 加载理论文件 {theory_file} 时出错: {str(e)}")
    
    # 从JSON文件加载理论
    if theories_json_file and os.path.exists(theories_json_file):
        try:
            with open(theories_json_file, 'r', encoding='utf-8') as f:
                theories_list = json.load(f)
            
            if isinstance(theories_list, list):
                print(f"[INFO] 从 {theories_json_file} 加载了 {len(theories_list)} 个理论")
                for i, theory in enumerate(theories_list):
                    theory_name = theory.get("name", f"理论_{i+1}")
                    theories[theory_name] = theory
            else:
                print(f"[WARNING] {theories_json_file} 不包含理论数组")
        except Exception as e:
            print(f"[ERROR] 加载理论JSON文件 {theories_json_file} 时出错: {str(e)}")
    
    return theories

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

async def main():
    parser = argparse.ArgumentParser(description="系统性理论合成程序")
    
    # LLM接口参数
    parser.add_argument("--model_source", type=str, default="deepseek",
                        choices=["openai", "deepseek", "google"],
                        help="模型来源")
    parser.add_argument("--model_name", type=str, default="deepseek-chat",
                        help="模型名称")
    
    # 输入参数
    parser.add_argument("--theories_dir", type=str, 
                        default="data/theories_v2.1",
                        help="理论文件目录")
    parser.add_argument("--additional_theories_json", type=str, default=None,
                        help="额外的理论JSON文件（包含理论数组）")
    parser.add_argument("--schema_version", type=str, default="2.1",
                        help="要加载的理论schema版本，设置为'any'可加载所有版本")
    parser.add_argument("--theory_subset", type=str, default=None,
                        help="要分析的理论子集，逗号分隔，如'copenhagen,many_worlds'")
    
    # 生成参数
    parser.add_argument("--max_theories_to_analyze", type=int, default=8,
                        help="最大分析理论数量")
    parser.add_argument("--max_breakthrough_targets", type=int, default=5,
                        help="最大突破目标数量")
    parser.add_argument("--theories_per_target", type=int, default=2,
                        help="每个突破目标生成的理论数量")
    parser.add_argument("--create_unified_theory", action="store_true",
                        help="是否创建统一的综合理论")
    
    # 输出参数
    parser.add_argument("--output_dir", type=str, 
                        default="data/systemic_synthesis",
                        help="输出目录")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    synthesis_dir = os.path.join(args.output_dir, f"systemic_synthesis_{time.strftime('%Y%m%d_%H%M%S')}")
    ensure_directory_exists(synthesis_dir)
    
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
    print(f"\n[步骤1] 加载理论数据")
    load_schema_version = None if args.schema_version.lower() == 'any' else args.schema_version
    theories = load_theories_from_sources(
        args.theories_dir, 
        args.additional_theories_json,
        schema_version=load_schema_version
    )
    
    if not theories:
        print("[ERROR] 未加载到理论数据，程序终止")
        return
    
    print(f"[INFO] 总共加载了 {len(theories)} 个理论")
    
    # 确定要分析的理论子集
    if args.theory_subset:
        theory_names = [name.strip() for name in args.theory_subset.split(',')]
        theory_names = [name for name in theory_names if name in theories]
    else:
        theory_names = list(theories.keys())[:args.max_theories_to_analyze]
    
    print(f"[INFO] 将分析以下 {len(theory_names)} 个理论: {', '.join(theory_names)}")
    
    # 2. 共同假设分析
    print(f"\n[步骤2] 分析多理论共同假设和系统性缺陷")
    analyzer = CommonAssumptionAnalyzer(llm)
    analyzer.load_theories(theories)
    
    # 进行共同假设分析
    analysis_result = await analyzer.analyze_common_assumptions(theory_names)
    
    if "error" in analysis_result:
        print(f"[ERROR] 共同假设分析失败: {analysis_result['error']}")
        return
    
    # 保存分析结果
    analysis_dir = os.path.join(synthesis_dir, "assumption_analysis")
    ensure_directory_exists(analysis_dir)
    analyzer.save_analysis_results(analysis_dir)
    
    # 3. 识别突破目标
    print(f"\n[步骤3] 识别突破性目标")
    breakthrough_result = await analyzer.identify_breakthrough_targets(analysis_result)
    
    if "error" in breakthrough_result:
        print(f"[ERROR] 突破目标识别失败: {breakthrough_result['error']}")
        return
    
    # 保存突破目标
    breakthrough_file = os.path.join(analysis_dir, "breakthrough_targets.json")
    with open(breakthrough_file, 'w', encoding='utf-8') as f:
        json.dump(breakthrough_result, f, ensure_ascii=False, indent=2)
    print(f"[INFO] 突破目标已保存到: {breakthrough_file}")
    
    # 获取突破目标列表
    breakthrough_targets = breakthrough_result.get("breakthrough_targets", [])
    if not breakthrough_targets:
        print("[ERROR] 未识别到突破目标")
        return
    
    # 限制突破目标数量
    breakthrough_targets = breakthrough_targets[:args.max_breakthrough_targets]
    print(f"[INFO] 将基于 {len(breakthrough_targets)} 个突破目标生成革命性理论")
    
    # 4. 生成革命性理论
    print(f"\n[步骤4] 生成革命性理论")
    generator = RevolutionaryGenerator(llm)
    
    # 为每个突破目标生成理论
    revolutionary_theories = await generator.generate_multiple_revolutionary_theories(
        breakthrough_targets=breakthrough_targets,
        variants_per_target=args.theories_per_target
    )
    
    print(f"[INFO] 成功生成 {len(revolutionary_theories)} 个革命性理论")
    
    # 5. 可选：创建统一综合理论
    if args.create_unified_theory and len(breakthrough_targets) > 1:
        print(f"\n[步骤5] 创建统一综合理论")
        unified_theory = await generator.create_synthesis_theory(breakthrough_targets[:3])  # 最多使用前3个目标
        
        if "error" not in unified_theory:
            print(f"[INFO] 成功创建统一综合理论: {unified_theory.get('name', '未命名')}")
        else:
            print(f"[ERROR] 统一理论创建失败: {unified_theory['error']}")
    
    # 6. 保存革命性理论
    print(f"\n[步骤6] 保存革命性理论")
    theories_dir = os.path.join(synthesis_dir, "revolutionary_theories")
    ensure_directory_exists(theories_dir)
    generator.save_revolutionary_theories(theories_dir)
    
    # 7. 创建标准格式的理论文件，用于评估
    print(f"\n[步骤7] 创建标准格式理论文件")
    eval_theories_dir = os.path.join(synthesis_dir, "eval_ready_theories")
    ensure_directory_exists(eval_theories_dir)
    
    for theory in generator.generated_theories:
        # Schema v2.1: 直接保存完整的理论对象，因为它已经符合新格式
        eval_theory = theory
        
        # 保存标准格式文件
        theory_name = theory.get("name", "未命名理论")
        safe_name = theory_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "").lower()
        
        eval_file = os.path.join(eval_theories_dir, f"{safe_name}.json")
        with open(eval_file, 'w', encoding='utf-8') as f:
            json.dump(eval_theory, f, ensure_ascii=False, indent=2)
    
    # 8. 生成总结报告
    print(f"\n[步骤8] 生成总结报告")
    summary_report = {
        "synthesis_summary": {
            "analyzed_theories": theory_names,
            "breakthrough_targets_identified": len(breakthrough_targets),
            "revolutionary_theories_generated": len(revolutionary_theories),
            "has_unified_theory": args.create_unified_theory and "unified_theory" in locals(),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "key_findings": {
            "most_critical_deficiency": analysis_result.get("meta_analysis", {}).get("most_critical_deficiency", ""),
            "revolutionary_potential": analysis_result.get("meta_analysis", {}).get("revolutionary_potential", ""),
            "top_breakthrough_targets": [t.get("target_name", "") for t in breakthrough_targets[:3]]
        },
        "generated_theories_summary": [
            {
                "name": t.get("name", ""),
                "target": t.get("generated_from", {}).get("breakthrough_target", ""),
                "classification": t.get("theory_type", {}).get("classification", ""),
                "paradigm_shift_level": t.get("revolutionary_assessment", {}).get("paradigm_shift_level", "")
            }
            for t in revolutionary_theories
        ],
        "next_steps": {
            "experimental_evaluation": f"使用demo_1.py评估生成的理论文件: {eval_theories_dir}",
            "theory_refinement": "基于评估结果进一步优化理论",
            "collaboration_opportunities": "与实验物理学家合作验证预测"
        }
    }
    
    summary_file = os.path.join(synthesis_dir, "synthesis_summary.json")
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    
    print(f"\n[完成] 系统性合成完成!")
    print(f"输出目录: {synthesis_dir}")
    print(f"生成的革命性理论数量: {len(revolutionary_theories)}")
    print(f"标准格式理论文件: {eval_theories_dir}")
    print(f"下一步建议: 使用以下命令评估生成的理论:")
    print(f"python demo/demo_1.py --model_source {args.model_source} --model_name {args.model_name} --theory_dir {eval_theories_dir} --experiment_dir demo/experiments --output_dir demo/outputs/revolutionary_evaluation")

if __name__ == "__main__":
    asyncio.run(main()) 