#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_winner_theory.py
========================
导出最终胜者理论的完整JSON内容

用法:
python export_winner_theory.py --run_dir output_clean_evolution/run_20250620_143718
python export_winner_theory.py --run_dir output_clean_evolution/run_20250620_143718 --output_file my_theory.json
python export_winner_theory.py --run_dir output_clean_evolution/run_20250620_143718 --theory_index 0
"""

import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime


def export_winner_theory(run_dir: str, output_file: str = None, theory_index: int = 0):
    """导出最终胜者理论的JSON内容"""
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    
    if not manifest_path.exists():
        print(f"❌ 未找到清单文件: {manifest_path}")
        return False
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取清单文件: {e}")
        return False
    
    # 找到最终胜者
    generations = manifest.get('generations', {})
    if not generations:
        print("❌ 未找到代际信息")
        return False
    
    max_gen = max(map(int, generations.keys()))
    final_winners = generations[str(max_gen)].get('promoted_theories', [])
    
    if not final_winners:
        print("❌ 未找到最终胜者")
        return False
    
    if theory_index >= len(final_winners):
        print(f"❌ 理论索引 {theory_index} 超出范围。只有 {len(final_winners)} 个胜者理论。")
        return False
    
    # 获取指定索引的胜者
    winner_id = final_winners[theory_index]
    winner_data = manifest['theories'][winner_id]
    theory_name = winner_data.get('theory_name', 'Unknown')
    theory_file_path = winner_data.get('file_path', '')
    score = winner_data.get('score', 0)
    
    print(f"🏆 正在导出胜者理论 #{theory_index + 1}: {theory_name}")
    print(f"📊 分数: {score:.1%}")
    print(f"📁 源文件: {theory_file_path}")
    
    # 检查源文件是否存在
    source_path = Path(theory_file_path)
    if not source_path.exists():
        print(f"❌ 理论文件不存在: {source_path}")
        return False
    
    # 确定输出文件名
    if output_file is None:
        # 基于理论名称和时间戳生成文件名
        safe_name = theory_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"exported_theory_{safe_name}_{timestamp}.json"
    
    output_path = Path(output_file)
    
    try:
        # 读取原始理论内容
        with open(source_path, 'r', encoding='utf-8') as f:
            theory_content = json.load(f)
        
        # 添加导出元数据
        export_metadata = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "source_run": manifest.get('run_id', 'Unknown'),
                "source_file": str(source_path),
                "final_score": score,
                "generation": winner_data.get('generation', 0),
                "theory_id": winner_id
            }
        }
        
        # 将导出信息添加到理论元数据中
        if 'metadata' not in theory_content:
            theory_content['metadata'] = {}
        theory_content['metadata']['export_info'] = export_metadata['export_info']
        
        # 保存到输出文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(theory_content, f, ensure_ascii=False, indent=2)
        
        print(f"✅ 理论已成功导出到: {output_path.absolute()}")
        print(f"📏 文件大小: {output_path.stat().st_size} 字节")
        
        # 显示简要信息
        print(f"\n📖 导出理论摘要:")
        print(f"   🏷️  名称: {theory_content.get('name', 'N/A')}")
        print(f"   🔬 Schema版本: {theory_content.get('metadata', {}).get('schema_version', 'N/A')}")
        print(f"   🎯 与SQM关系: {theory_content.get('mathematical_relation_to_sqm', 'N/A')}")
        
        # 显示核心信息
        summary = theory_content.get('summary', '')
        if summary:
            print(f"   📝 摘要: {summary[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"❌ 导出过程中发生错误: {e}")
        return False


def export_all_winners(run_dir: str, output_dir: str = None):
    """导出所有最终胜者理论"""
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    
    if not manifest_path.exists():
        print(f"❌ 未找到清单文件: {manifest_path}")
        return False
    
    try:
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
    except Exception as e:
        print(f"❌ 无法读取清单文件: {e}")
        return False
    
    # 找到最终胜者
    generations = manifest.get('generations', {})
    if not generations:
        print("❌ 未找到代际信息")
        return False
    
    max_gen = max(map(int, generations.keys()))
    final_winners = generations[str(max_gen)].get('promoted_theories', [])
    
    if not final_winners:
        print("❌ 未找到最终胜者")
        return False
    
    # 创建输出目录
    if output_dir is None:
        run_id = manifest.get('run_id', 'unknown_run')
        output_dir = f"exported_theories_{run_id}"
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"🗂️  正在导出 {len(final_winners)} 个胜者理论到: {output_path.absolute()}")
    
    success_count = 0
    for i, winner_id in enumerate(final_winners):
        winner_data = manifest['theories'][winner_id]
        theory_name = winner_data.get('theory_name', f'theory_{i+1}')
        safe_name = theory_name.replace(" ", "_").replace("(", "").replace(")", "").lower()
        output_file = output_path / f"{i+1:02d}_{safe_name}.json"
        
        print(f"\n--- 导出理论 {i+1}/{len(final_winners)} ---")
        if export_winner_theory(run_dir, str(output_file), i):
            success_count += 1
    
    print(f"\n🎉 批量导出完成！成功导出 {success_count}/{len(final_winners)} 个理论")
    return success_count == len(final_winners)


def main():
    parser = argparse.ArgumentParser(description="导出最终胜者理论的完整JSON内容")
    parser.add_argument("--run_dir", type=str, required=True, help="运行目录路径")
    parser.add_argument("--output_file", type=str, help="输出文件路径（可选）")
    parser.add_argument("--theory_index", type=int, default=0, help="理论索引（从0开始，默认导出第一个胜者）")
    parser.add_argument("--export_all", action="store_true", help="导出所有胜者理论")
    parser.add_argument("--output_dir", type=str, help="批量导出时的输出目录")
    
    args = parser.parse_args()
    
    if args.export_all:
        export_all_winners(args.run_dir, args.output_dir)
    else:
        export_winner_theory(args.run_dir, args.output_file, args.theory_index)


if __name__ == "__main__":
    main() 