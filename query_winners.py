#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
query_winners.py
================
查询演进运行中的最终晋级理论信息

用法:
python query_winners.py --run_dir output_clean_evolution/run_20250620_143718
python query_winners.py --run_dir output_clean_evolution/run_20250620_143718 --show_full_theory
python query_winners.py --list_all_runs
"""

import json
import argparse
from pathlib import Path
import glob
from datetime import datetime


def list_all_runs():
    """列出所有演进运行"""
    runs_dir = Path("output_clean_evolution")
    if not runs_dir.exists():
        print("❌ 未找到演进运行目录")
        return
    
    run_dirs = sorted(runs_dir.glob("run_*"))
    if not run_dirs:
        print("❌ 未找到任何演进运行")
        return
    
    print("🗂️  所有演进运行:")
    print("-" * 80)
    
    for run_dir in run_dirs:
        manifest_path = run_dir / "run_manifest.json"
        if manifest_path.exists():
            try:
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    manifest = json.load(f)
                
                # 统计信息
                total_theories = len(manifest.get('theories', {}))
                generations = manifest.get('generations', {})
                max_gen = max(map(int, generations.keys())) if generations else 0
                
                # 最终胜者
                final_winners = []
                if generations and str(max_gen) in generations:
                    final_winners = generations[str(max_gen)].get('promoted_theories', [])
                
                print(f"📁 {run_dir.name}")
                print(f"   🧬 总理论数: {total_theories}")
                print(f"   🔄 代际数: {max_gen + 1}")
                print(f"   🏆 最终胜者: {len(final_winners)} 个")
                
                if final_winners:
                    for winner_id in final_winners:
                        winner_data = manifest['theories'].get(winner_id, {})
                        name = winner_data.get('theory_name', 'Unknown')
                        score = winner_data.get('score', 0)
                        print(f"      • {name}: {score:.1%}")
                print()
                
            except Exception as e:
                print(f"❌ 无法读取 {run_dir.name}: {e}")
        else:
            print(f"⚠️  {run_dir.name}: 未找到清单文件")


def query_run_winners(run_dir: str, show_full_theory: bool = False):
    """查询指定运行的胜者信息"""
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    
    if not manifest_path.exists():
        print(f"❌ 未找到清单文件: {manifest_path}")
        return
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取清单文件: {e}")
        return
    
    # 基本信息
    run_id = manifest.get('run_id', 'Unknown')
    theories = manifest.get('theories', {})
    lineage = manifest.get('lineage', {})
    generations = manifest.get('generations', {})
    
    print(f"🏃 运行ID: {run_id}")
    print(f"📍 路径: {run_path}")
    print(f"🧬 总理论数: {len(theories)}")
    print(f"🔄 总代际数: {len(generations)}")
    print()
    
    # 找到最终代际
    if not generations:
        print("❌ 未找到代际信息")
        return
    
    max_gen = max(map(int, generations.keys()))
    final_gen_data = generations[str(max_gen)]
    final_winners = final_gen_data.get('promoted_theories', [])
    
    if not final_winners:
        print("❌ 未找到最终胜者")
        return
    
    print(f"🏆 最终胜者 (Generation {max_gen}):")
    print("=" * 80)
    
    for i, winner_id in enumerate(final_winners, 1):
        winner_data = theories.get(winner_id, {})
        
        # 基本信息
        name = winner_data.get('theory_name', 'Unknown Theory')
        score = winner_data.get('score', 0)
        generation = winner_data.get('generation', 0)
        status = winner_data.get('status', 'Unknown')
        file_path = winner_data.get('file_path', '')
        
        print(f"\n🥇 胜者 #{i}: {name}")
        print(f"   📊 最终分数: {score:.1%}")
        print(f"   🧬 代际: {generation}")
        print(f"   📋 状态: {status}")
        
        # 血缘追踪
        ancestry = []
        current_id = winner_id
        while current_id in lineage:
            parent_id = lineage[current_id]
            parent_data = theories.get(parent_id, {})
            parent_name = parent_data.get('theory_name', 'Unknown')
            parent_score = parent_data.get('score', 0)
            ancestry.append(f"{parent_name} ({parent_score:.1%})")
            current_id = parent_id
        
        if ancestry:
            print(f"   🧬 血缘: {' ← '.join(reversed(ancestry))}")
        
        # 精炼信息
        if 'refinement_parent' in winner_data:
            parent_id = winner_data['refinement_parent']
            parent_data = theories.get(parent_id, {})
            parent_score = parent_data.get('score', 0)
            improvement = score - parent_score
            print(f"   🔧 精炼自: {parent_data.get('theory_name', 'Unknown')}")
            print(f"   📈 改进: {improvement:+.1%} ({parent_score:.1%} → {score:.1%})")
        
        print(f"   📁 文件路径: {file_path}")
        
        # 显示完整理论内容
        if show_full_theory:
            print(f"\n📖 完整理论内容:")
            print("-" * 60)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    theory_content = json.load(f)
                
                # 格式化显示关键部分
                print(f"🏷️  名称: {theory_content.get('name', 'N/A')}")
                print(f"📝 摘要: {theory_content.get('summary', 'N/A')}")
                
                # 核心原理
                core_principles = theory_content.get('core_principles', {})
                if core_principles:
                    print(f"\n🎯 核心原理:")
                    for key, value in core_principles.items():
                        if isinstance(value, list):
                            print(f"   {key}:")
                            for item in value:
                                print(f"     • {item}")
                        else:
                            print(f"   {key}: {value}")
                
                # 预测
                predictions = theory_content.get('predictions_and_verifiability', {})
                if predictions:
                    print(f"\n🔮 预测与验证:")
                    deviations = predictions.get('deviations_from_sqm', [])
                    if deviations:
                        print(f"   与标准量子力学的偏差:")
                        for dev in deviations:
                            print(f"     • {dev.get('prediction_name', 'Unknown')}")
                            print(f"       {dev.get('description', 'No description')}")
                
            except Exception as e:
                print(f"❌ 无法读取理论文件: {e}")


def main():
    parser = argparse.ArgumentParser(description="查询演进运行的胜者理论")
    parser.add_argument("--run_dir", type=str, help="运行目录路径")
    parser.add_argument("--show_full_theory", action="store_true", help="显示完整理论内容")
    parser.add_argument("--list_all_runs", action="store_true", help="列出所有运行")
    
    args = parser.parse_args()
    
    if args.list_all_runs:
        list_all_runs()
    elif args.run_dir:
        query_run_winners(args.run_dir, args.show_full_theory)
    else:
        print("❌ 请指定 --run_dir 或使用 --list_all_runs")


if __name__ == "__main__":
    main() 