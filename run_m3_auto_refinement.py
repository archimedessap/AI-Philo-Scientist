#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_m3_auto_refinement.py (Manifest-Aware Adapter)
===================================================
This script acts as a manifest-aware adapter for the legacy
`run_refinement_loop.py` script.

It does the following:
1. Reads the central manifest and a list of theory IDs.
2. Creates temporary input files (`summary.json`, theories dir) in the
   format expected by the legacy script.
3. Calls `run_refinement_loop.py`.
4. After completion, parses the output of the legacy script.
5. Registers any new, improved theory variants back into the central manifest.
"""

from __future__ import annotations
import sys
import argparse
import json
import shutil
import glob
from pathlib import Path
import subprocess
import os

# NEW: Import manifest tools
try:
    import manifest_tools
    MANIFEST_AVAILABLE = True
except ImportError:
    MANIFEST_AVAILABLE = False
    print("[WARN] manifest_tools not available, running in legacy mode")

def run_cmd(cmd: list[str]):
    """Wraps subprocess.call and prints the command."""
    print("\n$ " + " ".join(cmd))
    return subprocess.call(cmd)

def ensure_directory_exists(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"[INFO] 创建目录: {directory}")

def load_theories_from_directory(theories_dir):
    """从目录加载理论文件"""
    theories = {}
    
    theory_files = glob.glob(os.path.join(theories_dir, "*.json"))
    print(f"[INFO] 在目录 {theories_dir} 中找到 {len(theory_files)} 个理论文件")
    
    for theory_file in theory_files:
        try:
            with open(theory_file, 'r', encoding='utf-8') as f:
                theory = json.load(f)
            
            theory_name = theory.get("name", os.path.basename(theory_file))
            theories[theory_name] = theory
            
        except Exception as e:
            print(f"[ERROR] 加载理论文件 {theory_file} 时出错: {str(e)}")
    
    return theories

def main():
    parser = argparse.ArgumentParser(description="M3自动精炼系统")
    
    # 支持多种调用方式的参数
    parser.add_argument("--run_dir", type=str, help="运行目录（robust_evolution_runner模式）")
    parser.add_argument("--target_generation", type=int, help="目标代际（robust_evolution_runner模式）")
    parser.add_argument("--input_theories_dir", type=str, help="输入理论目录（直接模式）")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--summary_file", type=str, help="Summary文件路径（文件模式）")
    parser.add_argument("--theories_root", type=str, help="理论根目录（文件模式）")
    parser.add_argument("--manifest-path", type=str, help="Manifest文件路径（manifest模式）")
    parser.add_argument("--theory-ids", type=str, help="理论ID列表（manifest模式）")
    parser.add_argument("--output-dir", type=str, help="输出目录（manifest模式）")
    
    # 通用参数
    parser.add_argument("--target_score", type=float, default=0.7, help="目标分数阈值")
    parser.add_argument("--improvement_threshold", type=float, default=0.02, help="改进阈值")
    parser.add_argument("--min_improvement", type=float, help="最小改进阈值（兼容性）")
    parser.add_argument("--max_iterations", type=int, default=3, help="最大迭代次数")
    parser.add_argument("--max_iters", type=int, help="最大迭代次数（兼容性）")
    parser.add_argument("--eval_mode", type=str, default="real", choices=["fast", "real"], help="评估模式")
    parser.add_argument("--model_source", type=str, default="deepseek", help="LLM来源")
    parser.add_argument("--model_name", type=str, default="deepseek-reasoner", help="LLM模型")
    
    # 其他兼容性参数
    parser.add_argument("--top_n", type=int, default=5, help="选择前N个理论")
    parser.add_argument("--top-n", type=int, help="选择前N个理论（manifest模式）")
    parser.add_argument("--min-improve", type=float, help="最小改进阈值（manifest模式）")
    parser.add_argument("--max-iters", type=int, help="最大迭代次数（manifest模式）")
    parser.add_argument("--judge_model_source", type=str, help="评审模型来源")
    parser.add_argument("--judge_model_name", type=str, help="评审模型名称")
    parser.add_argument("--dialog_model_source", type=str, help="对话模型来源")
    parser.add_argument("--dialog_model_name", type=str, help="对话模型名称")

    args = parser.parse_args()

    # 参数标准化和兼容性处理
    max_iterations = args.max_iters or getattr(args, 'max-iters', None) or args.max_iterations
    improvement_threshold = args.min_improvement or getattr(args, 'min-improve', None) or args.improvement_threshold
    output_dir = args.output_dir or getattr(args, 'output-dir', None)
    
    # 根据调用模式确定输入和输出
    if args.run_dir and args.target_generation is not None:
        # robust_evolution_runner模式
        print(f"[INFO] 运行在robust_evolution_runner模式")
        print(f"[INFO] 运行目录: {args.run_dir}")
        print(f"[INFO] 目标代际: {args.target_generation}")
        
        # 在这种模式下，我们需要从manifest中找到上一代的优胜理论
        run_dir = Path(args.run_dir)
        manifest_path = run_dir / "run_manifest.json"
        
        if not manifest_path.exists():
            print(f"[ERROR] 未找到manifest文件: {manifest_path}")
            return
        
        if not MANIFEST_AVAILABLE:
            print("[ERROR] robust_evolution_runner模式需要manifest_tools")
            return
        
        # 设置输出目录
        if not output_dir:
            output_dir = str(run_dir / f"generation_{args.target_generation}" / "refinement")
        
        # 加载manifest并找到需要精炼的理论
        manifest = manifest_tools.load_manifest(manifest_path)
        
        # 找到上一代的promoted理论
        prev_generation = args.target_generation - 1
        promoted_theories = {}
        for theory_id, theory_data in manifest['theories'].items():
            if (theory_data['generation'] == prev_generation and 
                theory_data.get('status') == 'promoted'):
                promoted_theories[theory_id] = theory_data
        
        if not promoted_theories:
            print(f"[INFO] 第{prev_generation}代没有找到promoted理论，精炼完成")
            return
        
        print(f"[INFO] 找到 {len(promoted_theories)} 个需要精炼的理论")
        
        # 创建临时目录和文件
        ensure_directory_exists(output_dir)
        temp_dir = Path(output_dir) / "temp"
        temp_dir.mkdir(exist_ok=True)
        temp_theories_dir = temp_dir / "theories"
        temp_theories_dir.mkdir(exist_ok=True)
        
        # 复制理论文件并创建summary
        summary_data = []
        for theory_id, theory_data in promoted_theories.items():
            src_path = Path(theory_data['file_path'])
            dest_path = temp_theories_dir / src_path.name
            shutil.copy(src_path, dest_path)
            
            summary_data.append({
                "theory_name": theory_data['theory_name'],
                "file_path": str(dest_path),
                "success_rate": theory_data.get('score', 0.8),
                "composite_score": theory_data.get('score', 0.8)
            })
        
        temp_summary_path = temp_dir / "summary.json"
        with open(temp_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
            
        summary_file = str(temp_summary_path)
        theories_root = str(temp_theories_dir)
        top_n = len(promoted_theories)
        
    elif args.input_theories_dir:
        # 直接模式
        print("[INFO] 运行在直接模式")
        theories_root = args.input_theories_dir
        
        if not output_dir:
            print("[ERROR] 直接模式需要指定--output_dir")
            return
        
        # 创建临时summary文件
        ensure_directory_exists(output_dir)
        temp_summary_path = os.path.join(output_dir, "temp_summary.json")
        
        theories = load_theories_from_directory(theories_root)
        summary_data = []
        for theory_name, theory_data in theories.items():
            # 找到对应的文件路径
            theory_files = glob.glob(os.path.join(theories_root, "*.json"))
            theory_file = None
            for tf in theory_files:
                with open(tf, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if data.get("name") == theory_name:
                        theory_file = tf
                        break
            
            if theory_file:
                summary_data.append({
                    "theory_name": theory_name,
                    "file_path": theory_file,
                    "success_rate": 0.8,
                    "composite_score": 0.8
                })
        
        with open(temp_summary_path, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=2)
        
        summary_file = temp_summary_path
        top_n = len(summary_data)
        
    elif args.summary_file and args.theories_root:
        # 文件模式（如run_feedback_cycle.py调用）
        print("[INFO] 运行在文件模式")
        summary_file = args.summary_file
        theories_root = args.theories_root
        
        if not output_dir:
            print("[ERROR] 文件模式需要指定输出目录")
            return
        
        # 从summary文件读取理论数量
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary_data = json.load(f)
        top_n = getattr(args, 'top-n', None) or args.top_n or len(summary_data)
        
    else:
        print("[ERROR] 必须指定运行模式的参数")
        print("支持的模式：")
        print("1. robust_evolution_runner模式: --run_dir + --target_generation")
        print("2. 直接模式: --input_theories_dir + --output_dir")
        print("3. 文件模式: --summary_file + --theories_root + --output_dir")
        return
    
    # 调用run_refinement_loop.py
    refinement_output = os.path.join(output_dir, "refinement_output")
    
    cmd = [
        "python", "run_refinement_loop.py",
        "--summary_file", summary_file,
        "--theories_root", theories_root,
        "--output_root", refinement_output,
        "--top_n", str(top_n),
        "--max_iters", str(max_iterations),
        "--min_improve", str(improvement_threshold),
        "--eval_mode", args.eval_mode,
        "--judge_model_source", args.judge_model_source or args.model_source,
        "--judge_model_name", args.judge_model_name or args.model_name,
        "--dialog_model_source", args.dialog_model_source or args.model_source,
        "--dialog_model_name", args.dialog_model_name or args.model_name
    ]

    print(f"[INFO] 调用精炼循环...")
    result = run_cmd(cmd)
    
    if result == 0:
        print(f"[INFO] 精炼完成，结果保存在: {refinement_output}")

        # 如果是robust_evolution_runner模式，需要将结果注册回manifest
        if args.run_dir and args.target_generation is not None and MANIFEST_AVAILABLE:
            print("[INFO] 注册精炼结果到manifest...")
            # 查找精炼结果文件
            depth_runs_dir = Path(refinement_output) / "depth_runs"
            if depth_runs_dir.exists():
                improved_files = list(depth_runs_dir.rglob("improved_*.json"))
                manifest = manifest_tools.load_manifest(manifest_path)
                
                for improved_file in improved_files:
                    try:
                        with open(improved_file, 'r', encoding='utf-8') as f:
                            theory_data = json.load(f)
                        
                        # 注册新的精炼理论
                        theory_id = manifest_tools.register_theory_in_manifest(
                            manifest, theory_data, improved_file, generation=args.target_generation
                        )
                        theory_name = theory_data.get('name', 'Unknown')
                        print(f"  ✅ 精炼理论: {theory_name} -> {theory_id}")
                        
                    except Exception as e:
                        print(f"  ❌ 无法注册精炼理论 {improved_file}: {e}")
                
                manifest_tools.save_manifest(manifest, manifest_path)
                print(f"[INFO] 精炼结果已注册到manifest")
            
    else:
        print(f"[ERROR] 精炼失败，返回码: {result}")

if __name__ == "__main__":
    main() 