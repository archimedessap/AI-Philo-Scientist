#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
增强版Demo脚本

将基于七大标准的理论生成器整合到现有的实验评估流程中，
生成更高质量的量子理论并进行完整的实验验证。
"""

import os
import sys
import json
import asyncio
import argparse
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from theory_generation.enhanced_theory_generator import EnhancedTheoryGenerator
from theory_generation.llm_interface import LLMInterface
from demo.demo_1 import load_experiments_from_directory, evaluate_theory_experiment

async def generate_and_evaluate_enhanced_theories(args):
    """生成增强理论并进行完整评估"""
    
    print("=" * 80)
    print("增强版量子理论生成与评估系统")
    print("=" * 80)
    
    # 初始化组件
    print("[INFO] 初始化系统组件...")
    llm = LLMInterface()
    generator = EnhancedTheoryGenerator(llm)
    
    # 加载实验数据
    print("[INFO] 加载实验数据...")
    experiments_dir = project_root / "demo" / "experiments"
    experiments = load_experiments_from_directory(str(experiments_dir))
    print(f"[INFO] 已加载 {len(experiments)} 个实验")
    
    # 定义理论对比和矛盾点
    theory_comparisons = [
        {
            "theory1": "哥本哈根诠释",
            "theory2": "多世界诠释",
            "contradictions": [
                {
                    "dimension": "波函数本体论地位",
                    "theory1_position": "波函数是计算工具，不对应物理实在",
                    "theory2_position": "波函数是完全的物理实在",
                    "core_tension": "波函数的存在论地位问题",
                    "importance_score": 9
                },
                {
                    "dimension": "测量过程",
                    "theory1_position": "通过波函数坍缩解释测量",
                    "theory2_position": "通过宇宙分裂避免坍缩",
                    "core_tension": "测量是否引起真实的物理变化",
                    "importance_score": 8
                }
            ]
        },
        {
            "theory1": "玻姆力学", 
            "theory2": "关系量子力学",
            "contradictions": [
                {
                    "dimension": "隐变量存在性",
                    "theory1_position": "存在确定的隐变量决定量子行为",
                    "theory2_position": "只有关系性质是实在的，无隐变量",
                    "core_tension": "是否存在潜在的确定性结构",
                    "importance_score": 8
                },
                {
                    "dimension": "非局域性",
                    "theory1_position": "通过量子势实现非局域关联",
                    "theory2_position": "通过关系网络实现相关性",
                    "core_tension": "非局域关联的机制问题",
                    "importance_score": 7
                }
            ]
        }
    ]
    
    # 定义不同的生成配置
    generation_configs = [
        {
            "name": "实验导向型",
            "config": {
                "focus_on_standards": ["experimental_distinguishability", "explanatory_completeness"],
                "mathematical_depth": "moderate",
                "novelty_level": "incremental",
                "experimental_orientation": True
            }
        },
        {
            "name": "数学严谨型",
            "config": {
                "focus_on_standards": ["mathematical_consistency", "conceptual_clarity"],
                "mathematical_depth": "advanced", 
                "novelty_level": "incremental",
                "experimental_orientation": False
            }
        },
        {
            "name": "哲学创新型",
            "config": {
                "focus_on_standards": ["philosophical_coherence", "intuitive_comprehensibility"],
                "mathematical_depth": "moderate",
                "novelty_level": "moderate",
                "experimental_orientation": True
            }
        }
    ]
    
    # 创建输出目录
    output_dir = project_root / "demo" / "outputs" / "enhanced_evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # 为每个理论对比和配置生成理论
    for comparison in theory_comparisons:
        theory1 = comparison["theory1"]
        theory2 = comparison["theory2"]
        contradictions = comparison["contradictions"]
        
        print(f"\n" + "=" * 60)
        print(f"理论对比: {theory1} vs {theory2}")
        print("=" * 60)
        
        comparison_results = {
            "comparison": f"{theory1} vs {theory2}",
            "theories": []
        }
        
        for config_info in generation_configs:
            config_name = config_info["name"]
            config = config_info["config"]
            
            print(f"\n[INFO] 生成 {config_name} 理论...")
            
            # 生成理论
            theory = await generator.generate_high_quality_theory(
                theory1, theory2, contradictions, config
            )
            
            if "error" in theory:
                print(f"[ERROR] {config_name} 理论生成失败: {theory['error']}")
                continue
            
            print(f"[INFO] 成功生成: {theory['name']}")
            print(f"[INFO] 质量评分: {theory['quality_assessment']['overall_score']:.2f}/10")
            
            # 评估理论在所有实验上的表现
            print(f"[INFO] 评估理论在 {len(experiments)} 个实验上的表现...")
            
            theory_results = {
                "theory_name": theory["name"],
                "config_type": config_name,
                "quality_score": theory["quality_assessment"]["overall_score"],
                "experiments": [],
                "summary": {}
            }
            
            successful_predictions = 0
            total_chi2 = 0
            experiment_count = 0
            
            for exp_name, exp_data in experiments.items():
                print(f"  评估实验: {exp_name}")
                
                # 进行实验评估
                result = await evaluate_theory_experiment(
                    theory, 
                    exp_data["setup"],
                    exp_data["measured"],
                    llm,
                    args,
                    output_prefix=f"enhanced_{theory['name'].replace(' ', '_')}"
                )
                
                if result:
                    theory_results["experiments"].append({
                        "experiment": exp_name,
                        "predicted_value": result.get("predicted_value", 0),
                        "measured_value": result.get("measured_value", 0),
                        "chi2": result.get("chi2", float('inf')),
                        "success": result.get("success", False)
                    })
                    
                    if result.get("success", False):
                        successful_predictions += 1
                    
                    total_chi2 += result.get("chi2", 0)
                    experiment_count += 1
            
            # 计算总体评估指标
            success_rate = successful_predictions / experiment_count if experiment_count > 0 else 0
            average_chi2 = total_chi2 / experiment_count if experiment_count > 0 else float('inf')
            
            theory_results["summary"] = {
                "success_rate": success_rate,
                "average_chi2": average_chi2,
                "total_experiments": experiment_count,
                "successful_predictions": successful_predictions
            }
            
            comparison_results["theories"].append(theory_results)
            
            print(f"[INFO] {config_name} 理论评估完成:")
            print(f"  成功率: {success_rate:.1%}")
            print(f"  平均χ²: {average_chi2:.2f}")
            
            # 保存单个理论结果
            theory_file = output_dir / f"{theory['name'].replace(' ', '_').lower()}.json"
            with open(theory_file, 'w', encoding='utf-8') as f:
                json.dump({
                    "theory": theory,
                    "evaluation_results": theory_results
                }, f, ensure_ascii=False, indent=2)
        
        all_results.append(comparison_results)
    
    # 生成综合报告
    print(f"\n" + "=" * 80)
    print("综合评估报告")
    print("=" * 80)
    
    for comparison_result in all_results:
        print(f"\n理论对比: {comparison_result['comparison']}")
        print("-" * 50)
        
        # 按成功率排序
        theories = sorted(
            comparison_result["theories"],
            key=lambda x: x["summary"]["success_rate"],
            reverse=True
        )
        
        for i, theory in enumerate(theories, 1):
            print(f"{i}. {theory['theory_name']} ({theory['config_type']})")
            print(f"   质量评分: {theory['quality_score']:.2f}/10")
            print(f"   实验成功率: {theory['summary']['success_rate']:.1%}")
            print(f"   平均χ²: {theory['summary']['average_chi2']:.2f}")
    
    # 保存完整结果
    results_file = output_dir / "enhanced_evaluation_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n[INFO] 完整结果已保存到: {results_file}")
    print(f"[INFO] 各个理论详情请查看: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="增强版量子理论生成与评估")
    parser.add_argument("--model-source", default="deepseek", 
                       choices=["openai", "deepseek", "claude"],
                       help="LLM模型源")
    parser.add_argument("--model-name", default="deepseek-reasoner",
                       help="具体模型名称")
    parser.add_argument("--temperature", type=float, default=0.2,
                       help="生成温度")
    parser.add_argument("--max-theories", type=int, default=6,
                       help="最大生成理论数量")
    
    args = parser.parse_args()
    
    try:
        asyncio.run(generate_and_evaluate_enhanced_theories(args))
    except KeyboardInterrupt:
        print("\n[INFO] 程序被用户中断")
    except Exception as e:
        print(f"\n[ERROR] 程序执行失败: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 