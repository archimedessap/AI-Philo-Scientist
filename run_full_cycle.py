#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UniversalTheoryGen - 端到端全周期运行器
========================================
该脚本作为项目总控制器，一键执行从"理论合成"到"综合评估"的全过程。

工作流程:
1.  **理论合成**: 调用 `run_direct_synthesis.py` 脚本，基于现有理论的矛盾分析，生成一批新的理论。
2.  **理论评估**: 待新理论生成后，自动捕获其输出路径，并调用 `demo/demo_1.py` 脚本，
    对这批新理论进行实验评估和多角色评估，最终输出综合排名。

示例:
python run_full_cycle.py \
  --max_pairs_to_analyze 20 \
  --variants_per_contradiction 3 \
  --synthesis_model_name "gemini-2.5-pro-preview-06-05" \
  --evaluation_model_name "deepseek-reasoner" \
  --use_instrument_correction
"""

import os
import sys
import argparse
import subprocess
import time
import re

def print_banner(text):
    """打印一个漂亮的横幅"""
    line = "=" * (len(text) + 4)
    print(f"\n{line}")
    print(f"| {text} |")
    print(f"{line}\n")

def run_command(command, description):
    """运行一个子命令并实时打印输出"""
    print_banner(f"正在执行: {description}")
    print(f"命令行: {' '.join(command)}\n")
    
    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1
        )
        
        output_lines = []
        for line in iter(process.stdout.readline, ''):
            sys.stdout.write(line)
            output_lines.append(line)
        
        process.wait()
        
        if process.returncode != 0:
            print(f"\n[ERROR] {description} 失败，返回码: {process.returncode}")
            return None, "".join(output_lines)
            
        print(f"\n[SUCCESS] {description} 完成。")
        return process.returncode, "".join(output_lines)

    except FileNotFoundError:
        print(f"\n[ERROR] 命令未找到: {command[0]}。请确保脚本路径正确且文件可执行。")
        return None, f"Command not found: {command[0]}"
    except Exception as e:
        print(f"\n[CRITICAL] 运行命令时发生未知错误: {e}")
        return None, str(e)


def main():
    parser = argparse.ArgumentParser(
        description="UniversalTheoryGen - 端到端全周期运行器",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- 总控参数 ---
    control_group = parser.add_argument_group('Overall Control Parameters')
    control_group.add_argument("--base_output_dir", type=str, default="data/full_cycle_runs", help="所有运行结果的根输出目录")

    # --- 理论合成参数 (Synthesis Parameters) ---
    synthesis_group = parser.add_argument_group('Phase 1: Theory Synthesis')
    synthesis_group.add_argument("--existing_theories_dir", type=str, default="data/theories_v2.1", help="用于分析矛盾的现有理论目录")
    synthesis_group.add_argument("--max_pairs_to_analyze", type=int, default=1, help="合成阶段分析的最大理论对数")
    synthesis_group.add_argument("--variants_per_contradiction", type=int, default=3, help="每个矛盾点生成的新理论变体数量")
    synthesis_group.add_argument("--synthesis_model_source", type=str, default="google", choices=["openai", "deepseek", "google"], help="用于理论合成的LLM来源")
    synthesis_group.add_argument("--synthesis_model_name", type=str, default="gemini-1.5-pro-latest", help="用于理论合成的具体模型名称")
    
    # --- 理论评估参数 (Evaluation Parameters) ---
    evaluation_group = parser.add_argument_group('Phase 2: Theory Evaluation')
    evaluation_group.add_argument("--experiment_dir", type=str, default="demo/experiments/", help="用于评估的实验数据目录")
    evaluation_group.add_argument("--use_instrument_correction", action='store_true', default=True, help="在实验评估中启用仪器修正模型（默认启用）")
    evaluation_group.add_argument("--evaluation_model_source", type=str, default="openai", choices=["openai", "deepseek", "google"], help="用于理论评估的LLM来源")
    evaluation_group.add_argument("--evaluation_model_name", type=str, default="gpt-4o-mini", help="用于理论评估的具体模型名称")
    evaluation_group.add_argument("--role_eval_threshold", type=float, default=0.6, help="实验成功率阈值，超过该值的理论将进行多角色评估")

    args = parser.parse_args()

    # --- 智能推断模型来源 ---
    # 如果用户没有显式指定synthesis_model_source，则根据synthesis_model_name推断
    if 'gemini' in args.synthesis_model_name.lower() and args.synthesis_model_source is None:
        args.synthesis_model_source = 'google'
    elif 'gpt' in args.synthesis_model_name.lower() and args.synthesis_model_source is None:
        args.synthesis_model_source = 'openai'
    elif 'deepseek' in args.synthesis_model_name.lower() and args.synthesis_model_source is None:
        args.synthesis_model_source = 'deepseek'

    # 如果用户没有显式指定evaluation_model_source，则根据evaluation_model_name推断
    if 'gemini' in args.evaluation_model_name.lower() and args.evaluation_model_source is None:
        args.evaluation_model_source = 'google'
    elif 'gpt' in args.evaluation_model_name.lower() and args.evaluation_model_source is None:
        args.evaluation_model_source = 'openai'
    elif 'deepseek' in args.evaluation_model_name.lower() and args.evaluation_model_source is None:
        args.evaluation_model_source = 'deepseek'

    # 1. 创建本次运行的专属主目录
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    main_run_dir = os.path.join(args.base_output_dir, f"run_{timestamp}")
    os.makedirs(main_run_dir, exist_ok=True)
    print(f"主运行目录已创建: {main_run_dir}")

    # =================================================================
    # 阶段一: 理论合成
    # =================================================================
    synthesis_output_dir = os.path.join(main_run_dir, "1_synthesis_output")
    os.makedirs(synthesis_output_dir, exist_ok=True)

    synthesis_command = [
        "python", "run_direct_synthesis.py",
        "--theories_dir", args.existing_theories_dir,
        "--max_pairs", str(args.max_pairs_to_analyze),
        "--variants_per_contradiction", str(args.variants_per_contradiction),
        "--model_source", args.synthesis_model_source,
        "--model_name", args.synthesis_model_name,
        "--output_dir", synthesis_output_dir
    ]

    synthesis_return_code, synthesis_output = run_command(synthesis_command, "理论合成")

    if synthesis_return_code != 0:
        print("\n[FATAL] 理论合成阶段失败，无法继续。请检查以上日志。")
        sys.exit(1)

    # 从合成脚本的输出中解析出可评估理论的目录
    eval_ready_path_match = re.search(r"标准格式的评估理论文件已保存到: (.*)", synthesis_output)
    if not eval_ready_path_match:
        print("\n[FATAL] 无法从合成脚本的输出中找到可评估理论的路径，无法继续。")
        sys.exit(1)
        
    eval_ready_theories_path = eval_ready_path_match.group(1).strip()
    print(f"\n[INFO] 成功解析出新理论路径: {eval_ready_theories_path}")

    # =================================================================
    # 阶段二: 理论评估
    # =================================================================
    evaluation_output_dir = os.path.join(main_run_dir, "2_evaluation_output")
    os.makedirs(evaluation_output_dir, exist_ok=True)
    
    evaluation_command = [
        "python", "demo/demo_1.py",
        "--theory_path", eval_ready_theories_path,
        "--experiment_dir", args.experiment_dir,
        "--output_dir", evaluation_output_dir,
        "--model_source", args.evaluation_model_source,
        "--model_name", args.evaluation_model_name,
        "--run_role_evaluation",
        "--role_success_threshold", str(args.role_eval_threshold)
    ]

    if args.use_instrument_correction:
        evaluation_command.append("--use_instrument_correction")

    evaluation_return_code, _ = run_command(evaluation_command, "理论评估")

    if evaluation_return_code != 0:
        print("\n[FATAL] 理论评估阶段失败。请检查以上日志。")
        sys.exit(1)
        
    print_banner("全周期运行成功完成！")
    print(f"所有结果已保存在: {main_run_dir}")


if __name__ == "__main__":
    main() 