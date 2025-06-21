#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resume_evaluation.py
====================
从中断的评估阶段恢复运行，基于已有的实验评估结果继续进行角色评估

用法:
python resume_evaluation.py --run_dir output_clean_evolution/run_20250620_160034
"""

import json
import argparse
import subprocess
import glob
from pathlib import Path
import os

def collect_experimental_results(run_dir: str):
    """收集已有的实验评估结果"""
    run_path = Path(run_dir)
    results_dir = run_path / "generation_0" / "evaluation" / "results"
    
    if not results_dir.exists():
        print(f"❌ 未找到评估结果目录: {results_dir}")
        return None
    
    # 查找所有实验评估JSON文件
    eval_files = list(results_dir.glob("*_evaluation.json"))
    
    print(f"📊 找到 {len(eval_files)} 个实验评估文件")
    
    # 按理论分组
    theory_results = {}
    for eval_file in eval_files:
        filename = eval_file.name
        # 提取理论名称（文件名格式：theory_name_vs_experiment_evaluation.json）
        parts = filename.replace("_evaluation.json", "").split("_vs_")
        if len(parts) >= 2:
            theory_name = parts[0]
            if theory_name not in theory_results:
                theory_results[theory_name] = []
            
            # 读取评估结果
            try:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    result = json.load(f)
                theory_results[theory_name].append(result)
            except Exception as e:
                print(f"⚠️ 无法读取文件 {eval_file}: {e}")
    
    return theory_results

def generate_experimental_summary(theory_results: dict, output_file: str):
    """基于实验评估结果生成临时总结文件"""
    summary_data = []
    
    for theory_name, results in theory_results.items():
        if not results:
            continue
            
        # 计算成功率和平均chi2
        total_experiments = len(results)
        successful_experiments = 0
        chi2_sum = 0.0
        chi2_count = 0
        
        for result in results:
            # 优先使用修正后的结果
            success = result.get('success_corrected', result.get('success', False))
            if success:
                successful_experiments += 1
            
            chi2 = result.get('chi2_corrected', result.get('chi2', 0))
            if chi2 is not None:
                chi2_sum += chi2
                chi2_count += 1
        
        success_rate = successful_experiments / total_experiments if total_experiments > 0 else 0
        average_chi2 = chi2_sum / chi2_count if chi2_count > 0 else float('inf')
        
        # 生成与demo_1.py兼容的格式
        theory_summary = {
            "theory_name": theory_name.replace("_", " ").title(),
            "success_rate": success_rate,
            "average_chi2": average_chi2,
            "experiments_count": total_experiments,
            "file_path": f"dummy_path_{theory_name}.json"  # 占位符
        }
        
        summary_data.append(theory_summary)
    
    # 按成功率排序
    summary_data.sort(key=lambda x: (-x['success_rate'], x['average_chi2']))
    
    # 保存总结文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 实验评估总结已保存到: {output_file}")
    return summary_data

def run_role_evaluation_only(run_dir: str, experimental_summary: list):
    """只运行角色评估部分"""
    run_path = Path(run_dir)
    theories_dir = run_path / "generation_0" / "evaluation" / "theories"
    
    if not theories_dir.exists():
        print(f"❌ 未找到理论目录: {theories_dir}")
        return False
    
    # 筛选成功率达到阈值的理论
    role_eval_threshold = 0.1  # 降低阈值确保有理论进入角色评估
    eligible_theories = [t for t in experimental_summary if t['success_rate'] >= role_eval_threshold]
    
    if not eligible_theories:
        print(f"❌ 没有理论达到角色评估阈值 {role_eval_threshold*100:.0f}%")
        return False
    
    print(f"🎭 {len(eligible_theories)} 个理论符合角色评估条件:")
    for theory in eligible_theories:
        print(f"   • {theory['theory_name']}: {theory['success_rate']*100:.1f}%")
    
    # 运行角色评估
    results_dir = run_path / "generation_0" / "evaluation" / "results"
    
    # 使用更快的模型和更低的阈值
    cmd = [
        "python", "demo/auto_role_evaluation.py",
        "--theories_dir", str(theories_dir),
        "--output_dir", str(results_dir),
        "--model_source", "google",
        "--model_name", "gemini-1.5-pro",  # 使用更快的模型
        "--success_threshold", "0.05"  # 降低阈值
    ]
    
    print(f"🚀 运行角色评估: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ 角色评估完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 角色评估失败: {e}")
        print(f"输出: {e.stdout}")
        print(f"错误: {e.stderr}")
        return False

def update_manifest_with_results(run_dir: str):
    """更新清单文件，包含评估结果"""
    run_path = Path(run_dir)
    manifest_path = run_path / "run_manifest.json"
    results_dir = run_path / "generation_0" / "evaluation" / "results"
    
    # 查找最新的角色评估结果
    role_eval_files = list(results_dir.glob("**/combined_rankings.json"))
    if not role_eval_files:
        print("⚠️ 未找到角色评估结果文件")
        return False
    
    latest_file = max(role_eval_files, key=os.path.getmtime)
    print(f"📊 使用角色评估结果: {latest_file}")
    
    # 更新清单
    import manifest_tools
    try:
        manifest = manifest_tools.load_manifest(manifest_path)
        manifest_tools.update_manifest_with_evaluation(manifest, str(latest_file))
        manifest_tools.save_manifest(manifest, manifest_path)
        print("✅ 清单文件已更新")
        return True
    except Exception as e:
        print(f"❌ 更新清单失败: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="从中断的评估阶段恢复运行")
    parser.add_argument("--run_dir", type=str, required=True, help="运行目录路径")
    
    args = parser.parse_args()
    
    print(f"🔄 恢复运行: {args.run_dir}")
    
    # 1. 收集已有的实验评估结果
    theory_results = collect_experimental_results(args.run_dir)
    if not theory_results:
        return
    
    # 2. 生成实验评估总结
    temp_summary_file = Path(args.run_dir) / "temp_experimental_summary.json"
    experimental_summary = generate_experimental_summary(theory_results, str(temp_summary_file))
    
    # 3. 运行角色评估
    if run_role_evaluation_only(args.run_dir, experimental_summary):
        # 4. 更新清单文件
        update_manifest_with_results(args.run_dir)
        print("🎉 恢复评估完成！")
    else:
        print("❌ 恢复评估失败")

if __name__ == "__main__":
    main() 