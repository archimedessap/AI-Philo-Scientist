#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完成Generation 2的评估
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# 导入清单管理工具
import manifest_tools

def main():
    """完成Generation 2的评估"""
    
    # 设置路径
    run_dir = Path("output_clean_evolution/run_20250620_091005")
    manifest_path = run_dir / "run_manifest.json"
    gen2_dir = run_dir / "generation_2"
    
    # 加载清单
    print("加载清单...")
    manifest = manifest_tools.load_manifest(manifest_path)
    
    # 查找Generation 2的未评估理论
    unevaluated = {}
    for theory_id, theory_data in manifest['theories'].items():
        if (theory_data['generation'] == 2 and 
            theory_data.get('status') in ['created', 'refined']):
            unevaluated[theory_id] = theory_data
    
    if not unevaluated:
        print("没有找到Generation 2的待评估理论")
        return
        
    print(f"找到 {len(unevaluated)} 个待评估理论:")
    for theory_id, theory_data in unevaluated.items():
        print(f"  📋 {theory_id[:8]}: {theory_data['theory_name']}")
    
    # 创建评估目录
    eval_dir = gen2_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    temp_theories_dir = eval_dir / "theories"
    temp_theories_dir.mkdir(exist_ok=True)
    
    # 复制理论文件
    for theory_id, theory_data in unevaluated.items():
        src_path = Path(theory_data['file_path'])
        dest_path = temp_theories_dir / f"{theory_id}_{src_path.name}"
        shutil.copy(src_path, dest_path)
        print(f"复制理论文件: {src_path.name}")
    
    # 运行评估
    eval_output_dir = eval_dir / "results"
    cmd = [
        "python", "demo/demo_1.py",
        "--theory_path", str(temp_theories_dir),
        "--experiment_dir", "demo/experiments/",
        "--output_dir", str(eval_output_dir),
        "--model_source", "google",
        "--model_name", "gemini-2.5-pro",
        "--run_role_evaluation",
        "--role_success_threshold", "0.1",
        "--use_instrument_correction"
    ]
    
    print(f"运行评估命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=True, text=True)
        print("评估成功完成")
        
        # 更新分数
        ranking_files = list(eval_output_dir.rglob("combined_rankings.json"))
        
        if ranking_files:
            ranking_file = ranking_files[0]
            print(f"从评估结果更新分数: {ranking_file}")
            
            manifest_tools.update_manifest_with_evaluation(manifest, str(ranking_file))
            
            # 标记为已评估
            for theory_id in unevaluated.keys():
                manifest['theories'][theory_id]['status'] = 'evaluated'
            
            # 选择优胜者
            promoted_ids = manifest_tools.select_best_theories_for_next_gen(
                manifest, current_gen=2, top_n=2, min_score=0.3
            )
            
            for theory_id in promoted_ids:
                manifest['theories'][theory_id]['status'] = 'promoted'
            
            # 更新代际信息
            if 'generations' not in manifest:
                manifest['generations'] = {}
            
            manifest['generations']['2'] = {
                'completed_at': '2025-06-20T10:30:00.000000',
                'promoted_count': len(promoted_ids),
                'promoted_theories': promoted_ids
            }
            
            manifest_tools.save_manifest(manifest, manifest_path)
            print(f"Generation 2 完成，{len(promoted_ids)} 个理论晋级")
            
            # 打印最终排名
            print("\n" + "="*60)
            print("🏆 Generation 2 最终排名")
            print("="*60)
            
            gen2_theories = [(tid, data) for tid, data in manifest['theories'].items() 
                           if data['generation'] == 2 and 'score' in data]
            gen2_theories.sort(key=lambda x: x[1].get('score', 0), reverse=True)
            
            for i, (theory_id, theory_data) in enumerate(gen2_theories, 1):
                score = theory_data.get('score', 0)
                status = theory_data.get('status', 'unknown')
                print(f"{i}. {theory_data['theory_name']}")
                print(f"   分数: {score:.3f} | 状态: {status}")
                print()
        else:
            print("未找到评估结果文件")
            
    except subprocess.CalledProcessError as e:
        print(f"评估失败: {e}")
        return False

if __name__ == "__main__":
    main() 