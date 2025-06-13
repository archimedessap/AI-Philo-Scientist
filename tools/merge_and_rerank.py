#!/usr/bin/env python3
# coding: utf-8
"""
合并、重排序和重新评估工具

此脚本用于将多个评估运行(run)的结果合并，并对合并后的完整数据集进行最终的排序和可选的角色扮演评估。
这在评估过程因任何原因中断并从断点恢复后特别有用。

用法:
python tools/merge_and_rerank.py \
    --run_dirs evaluation_results/run_20240523_100000 evaluation_results/run_20240523_113000 \
    --output_dir evaluation_results/merged_run_20240523 \
    --theory_path data/theories_v2.1 \
    --run_role_evaluation
"""

import sys
import os
import json
import asyncio
import argparse
import glob
import time
from pathlib import Path

# 将根目录添加到sys.path以导入项目模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 从demo脚本中导入必要的函数
from demo.auto_role_evaluation import run_role_evaluation_for_theories
from demo.demo_1 import load_theories_from_sources

def find_summary_files(run_dirs):
    """在给定的运行目录中查找所有_summary.json文件"""
    summary_files = []
    for run_dir in run_dirs:
        # 使用glob递归查找所有匹配的文件
        search_pattern = os.path.join(run_dir, "**", "_summary.json")
        found_files = glob.glob(search_pattern, recursive=True)
        if not found_files:
            print(f"[WARN] 在目录 {run_dir} 中没有找到 '_summary.json' 文件。")
        summary_files.extend(found_files)
    return summary_files

def merge_results(summary_files):
    """合并所有找到的摘要文件中的结果"""
    all_results = []
    seen_results = set() # 用于跟踪已经添加的 (theory_name, experiment_id) 对，防止重复

    for f_path in summary_files:
        try:
            with open(f_path, 'r', encoding='utf-8') as f:
                results = json.load(f)
                for result in results:
                    # 创建一个唯一标识符来检测重复
                    identifier = (result.get('theory_name'), result.get('experiment_id'))
                    if identifier not in seen_results:
                        all_results.append(result)
                        seen_results.add(identifier)
                    else:
                        print(f"[INFO] 发现重复评估结果，将跳过: {identifier[0]} vs {identifier[1]}")
        except Exception as e:
            print(f"[ERROR] 读取或解析文件 {f_path} 时出错: {e}")
    return all_results

async def main():
    parser = argparse.ArgumentParser(
        description="Merge, Rerank, and Re-evaluate multiple evaluation runs.",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # --- 主要参数 ---
    main_group = parser.add_argument_group('Main Parameters')
    main_group.add_argument("--run_dirs", type=str, nargs='+', required=True, help="List of run directories to merge.")
    main_group.add_argument("--output_dir", type=str, default=None, help="Directory to save the merged results. Defaults to a new directory in 'evaluation_results'.")
    main_group.add_argument("--theory_path", type=str, required=True, help="Path to the original directory of theory definitions, needed for role evaluation.")
    main_group.add_argument("--schema_version", type=str, default="2.1", help="Schema version of theories to load.")

    # --- 角色评估参数 (从demo_1.py复制) ---
    role_eval_group = parser.add_argument_group('Role-playing Evaluation (Optional)')
    role_eval_group.add_argument("--run_role_evaluation", action='store_true', help="Run role-playing evaluation on the merged and ranked results.")
    role_eval_group.add_argument("--role_success_threshold", type=float, default=0.75, help="Success rate threshold for a theory to be passed to role evaluation.")
    role_eval_group.add_argument("--role_model_source", type=str, default="deepseek", choices=["openai", "deepseek", "google"], help="LLM provider for role evaluation.")
    role_eval_group.add_argument("--role_model_name", type=str, default="deepseek-reasoner", help="Specific model name for role evaluation.")
    
    args = parser.parse_args()

    # --- 1. 设置输出目录 ---
    if args.output_dir:
        output_dir = args.output_dir
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join("evaluation_results", f"merged_run_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"[SETUP] Merged results will be saved in: {output_dir}")

    # --- 2. 查找并合并结果 ---
    print(f"\n[MERGE] Searching for summary files in: {args.run_dirs}")
    summary_files = find_summary_files(args.run_dirs)
    if not summary_files:
        print("[ERROR] No summary files found. Exiting.")
        return
    
    print(f"[MERGE] Found {len(summary_files)} summary files. Merging...")
    all_results = merge_results(summary_files)
    print(f"[MERGE] Merged a total of {len(all_results)} unique evaluation results.")

    # --- 3. 生成最终报告 (逻辑从demo_1.py复用) ---
    if all_results:
        # 汇总每个理论的表现
        theory_performance = {}
        for result in all_results:
            t_name = result['theory_name']
            if t_name not in theory_performance:
                theory_performance[t_name] = {'success_count': 0, 'total_count': 0, 'chi2_sum': 0, 'chi2_list': []}
            
            theory_performance[t_name]['total_count'] += 1
            
            # 优先使用修正后的成功状态，如果不存在则使用原始成功状态
            is_success = result.get('success_corrected', result.get('success'))
            if is_success:
                theory_performance[t_name]['success_count'] += 1
            
            chi2 = result.get('chi2_corrected') or result.get('chi2')
            if chi2 is not None:
                theory_performance[t_name]['chi2_sum'] += chi2
                theory_performance[t_name]['chi2_list'].append(chi2)

        # 计算成功率和平均chi2
        ranked_theories = []
        for t_name, perf_data in theory_performance.items():
            success_rate = (perf_data['success_count'] / perf_data['total_count']) if perf_data['total_count'] > 0 else 0
            average_chi2 = (perf_data['chi2_sum'] / len(perf_data['chi2_list'])) if perf_data['chi2_list'] else float('inf')
            ranked_theories.append({
                "theory_name": t_name,
                "success_rate": success_rate,
                "average_chi2": average_chi2,
                "experiments_count": perf_data['total_count']
            })
            
        # 按成功率降序，平均chi2升序排序
        ranked_theories.sort(key=lambda x: (-x['success_rate'], x['average_chi2']))

        # 保存合并后的完整结果和排名
        final_summary_file = os.path.join(output_dir, "final_evaluation_summary.json")
        with open(final_summary_file, "w", encoding="utf-8") as f:
            json.dump(ranked_theories, f, ensure_ascii=False, indent=2)
        
        full_results_file = os.path.join(output_dir, "all_merged_results.json")
        with open(full_results_file, "w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)

        print(f"\n[FINAL] Overall experimental evaluation summary saved to: {final_summary_file}")
        print(f"[FINAL] All merged raw results saved to: {full_results_file}")

        # 打印实验评估排名
        print("\n" + "="*100)
        print("📊 实验评估排名 (合并后)")
        print("="*100)
        print(f"{'排名':<5} {'理论名称':<40} {'成功率':<15} {'平均χ²':<15} {'实验数':<10}")
        print("-"*100)
        for i, rank in enumerate(ranked_theories, 1):
            print(f"{i:<5} {rank['theory_name']:<40} {rank['success_rate']*100:14.1f}% {rank['average_chi2']:<15.4f} {rank['experiments_count']:<10}")
        print("-"*100)
        
        # --- 4. 自动运行角色评估 (如果启用) ---
        if args.run_role_evaluation:
            print("\n[ROLE-PLAY] Preparing for role-playing evaluation...")
            # 加载完整的理论定义
            print(f"[LOAD] Loading full theory definitions from: {args.theory_path}")
            schema_to_load = None if args.schema_version.lower() == 'any' else args.schema_version
            all_theories_definitions = load_theories_from_sources(args.theory_path, schema_version=schema_to_load)
            
            if not all_theories_definitions:
                print("[ERROR] Cannot run role evaluation without theory definitions. Exiting role evaluation.")
                return

            # 筛选高成功率理论
            high_success_theories = [
                r for r in ranked_theories 
                if r['success_rate'] >= args.role_success_threshold
            ]
            
            if high_success_theories:
                print(f"[ROLE-PLAY] Found {len(high_success_theories)} theories meeting the {args.role_success_threshold*100:.0f}% success threshold.")
                # 调用角色评估模块
                await run_role_evaluation_for_theories(
                    high_success_theories=high_success_theories,
                    all_theories_definitions=all_theories_definitions,
                    output_dir=output_dir, # 在合并后的新目录中输出
                    model_source=args.role_model_source,
                    model_name=args.role_model_name
                )
            else:
                print(f"\n[INFO] 没有理论达到 {args.role_success_threshold*100:.0f}% 的成功率阈值，跳过角色评估。")

    else:
        print("\n[FINAL] No evaluation results found after merging.")

    print("\nMerge and Rerank process finished.")

if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[INFO] 运行被用户中断。")
    except Exception as e:
        print(f"[CRITICAL ERROR] 发生未处理的异常: {e}")
        import traceback
        traceback.print_exc() 