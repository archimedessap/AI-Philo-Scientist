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
import json
from pathlib import Path

from utils.global_theory_registry import GlobalTheoryRegistry

DEFAULT_EXPERIMENT_SKIPS = {"fullerene_decoherence_hornberger2003"}

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


def auto_register_high_scoring_theories(
    main_run_dir: str,
    evaluation_output_dir: str,
    threshold: float,
    generation_method: str,
    synthesis_model_source: str,
    synthesis_model_name: str,
    registry_dir: str,
    visuals_dir: str,
):
    if threshold <= 0:
        return

    evaluation_path = Path(evaluation_output_dir)
    if not evaluation_path.exists():
        return

    run_id = Path(main_run_dir).name
    registry = GlobalTheoryRegistry(registry_dir)
    registered_names = []

    eval_dirs = sorted([d for d in evaluation_path.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not eval_dirs:
        return

    for eval_dir in eval_dirs:
        final_summary_file = eval_dir / "final_evaluation_summary.json"
        combined_file = eval_dir / "role_evaluations" / "combined_rankings.json"
        role_summary_file = eval_dir / "role_evaluations" / "role_evaluation_summary.json"

        if not final_summary_file.exists() or not combined_file.exists():
            continue

        final_data = json.loads(final_summary_file.read_text(encoding="utf-8"))
        combined_data = json.loads(combined_file.read_text(encoding="utf-8"))
        role_summary = json.loads(role_summary_file.read_text(encoding="utf-8")) if role_summary_file.exists() else []

        final_by_name = {entry.get("theory_name"): entry for entry in final_data}
        role_by_name = {entry.get("theory_name"): entry for entry in role_summary}

        for item in combined_data:
            score = item.get("combined_score", 0)
            if score < threshold:
                continue

            theory_name = item.get("theory_name")
            if not theory_name:
                continue

            final_entry = final_by_name.get(theory_name)
            if not final_entry:
                continue

            theory_file = Path(final_entry.get("file_path", ""))
            if not theory_file.exists():
                candidate = Path(main_run_dir) / "1_synthesis_output"
                matches = list(candidate.glob(f"**/{theory_file.name}")) if theory_file.name else []
                if matches:
                    theory_file = matches[0]
            if not theory_file.exists():
                print(f"[WARN] 自动注册失败，未找到理论文件: {theory_name}")
                continue

            experimental_results = {
                "success_rate": item.get("experiment_success_rate", final_entry.get("success_rate", 0.0)),
                "average_chi2": item.get("average_chi2", final_entry.get("average_chi2", 0.0)),
                "experiments_count": item.get("experiments_count", final_entry.get("experiments_count", 0)),
                "evaluation_method": "full_cycle" if generation_method == "direct" else "short_card_full_cycle",
            }

            role_entry = role_by_name.get(theory_name, {})
            role_details = item.get("role_details", {})
            role_evaluation_results = {
                "physicist_score": role_details.get("physicist", role_entry.get("evaluations", {}).get("physicist", {}).get("score", 0.0)),
                "philosopher_score": role_details.get("philosopher", role_entry.get("evaluations", {}).get("philosopher", {}).get("score", 0.0)),
                "mathematician_score": role_details.get("mathematician", role_entry.get("evaluations", {}).get("mathematician", {}).get("score", 0.0)),
                "composite_score": item.get("combined_score", 0.0),
                "full_role_evaluation": role_entry,
            }

            evaluation_payload = {
                "run_id": run_id,
                "theory_name": theory_name,
                "final_summary": final_entry,
                "combined_ranking": item,
                "role_evaluation": role_entry,
                "experimental_results": experimental_results,
                "role_evaluation_results": role_evaluation_results,
                "generation_method": generation_method,
            }

            try:
                registry.register_scored_theory(
                    run_id=run_id,
                    run_path=str(Path(main_run_dir)),
                    theory_name=theory_name,
                    theory_file=theory_file,
                    evaluation_payload=evaluation_payload,
                    composite_score=score,
                    success_rate=item.get("experiment_success_rate", final_entry.get("success_rate", 0.0)),
                    source_type="evolved",
                    generation_model_source=synthesis_model_source,
                    generation_model_name=synthesis_model_name,
                )
                registered_names.append(theory_name)
                print(f"[INFO] 自动注册理论: {theory_name} (综合评分 {score:.3f})")
            except Exception as exc:
                print(f"[WARN] 自动注册理论失败 {theory_name}: {exc}")

    if registered_names:
        try:
            from utils.global_theory_visualizer import GlobalTheoryVisualizer

            visualizer = GlobalTheoryVisualizer(registry_dir=registry_dir, output_dir=visuals_dir)
            visualizer.generate_comprehensive_report()
        except Exception as exc:
            print(f"[WARN] 更新理论可视化失败: {exc}")


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
    synthesis_group.add_argument("--generation_method", type=str, default="direct", choices=["direct", "short_card"], help="理论合成方式")
    synthesis_group.add_argument("--existing_theories_dir", type=str, default="data/theories_v2.1", help="用于分析矛盾的现有理论目录 (direct 模式)")
    synthesis_group.add_argument("--max_pairs_to_analyze", type=int, default=1, help="合成阶段分析的最大理论对数")
    synthesis_group.add_argument("--variants_per_contradiction", type=int, default=1, help="每个矛盾点生成的新理论变体数量")
    synthesis_group.add_argument("--synthesis_model_source", type=str, default="google", choices=["openai", "deepseek", "xai", "google"], help="用于理论合成的LLM来源")
    synthesis_group.add_argument("--synthesis_model_name", type=str, default="gemini-2.5-pro", help="用于理论合成的具体模型名称")
    synthesis_group.add_argument("--cards_dir", type=str, default="cards", help="短卡目录 (short_card 模式)")
    synthesis_group.add_argument("--card_schema", type=str, default="schemas/card.schema.json", help="短卡Schema")
    synthesis_group.add_argument("--contradiction_schema", type=str, default="schemas/contradiction.schema.json", help="短卡矛盾Schema")
    synthesis_group.add_argument("--new_interpretation_schema", type=str, default="schemas/new_interpretation.schema.json", help="短卡新理论Schema")
    synthesis_group.add_argument("--short_card_query", type=str, default="Short-card joint analysis for new theory creation", help="Task description used by the short-card workflow.")
    synthesis_group.add_argument("--short_card_task_hint", type=str, default="", help="短卡模式额外提示")
    synthesis_group.add_argument("--short_card_topk", type=int, default=-1, help="短卡模式使用的卡片数量，-1 表示全部")
    synthesis_group.add_argument("--short_card_constraints", type=str, default=None, help="短卡模式约束文件")
    synthesis_group.add_argument("--short_card_machine_temperature", type=float, default=0.4, help="短卡模式结构化输出温度")
    synthesis_group.add_argument("--short_card_human_temperature", type=float, default=0.6, help="短卡模式人类写作温度")
    synthesis_group.add_argument("--short_card_human_model_source", type=str, default=None, help="短卡模式人类写作模型来源")
    synthesis_group.add_argument("--short_card_human_model_name", type=str, default=None, help="短卡模式人类写作模型名称")
    synthesis_group.add_argument("--short_card_contradictions", type=str, default=None, help="预先生成的矛盾表JSON文件，提供后跳过矛盾分析阶段")
    synthesis_group.add_argument("--num_theories", type=int, default=1, help="短卡模式生成理论数量")
    
    # --- 理论评估参数 (Evaluation Parameters) ---
    evaluation_group = parser.add_argument_group('Phase 2: Theory Evaluation')
    evaluation_group.add_argument("--experiment_dir", type=str, default="demo/experiments/", help="用于评估的实验数据目录")
    evaluation_group.add_argument("--use_instrument_correction", action='store_true', default=True, help="在实验评估中启用仪器修正模型（默认启用）")
    evaluation_group.add_argument("--evaluation_model_source", type=str, default="google", choices=["openai", "deepseek", "xai", "google"], help="用于理论评估的LLM来源")
    evaluation_group.add_argument("--evaluation_model_name", type=str, default="gemini-2.5-pro", help="用于理论评估的具体模型名称")
    evaluation_group.add_argument("--role_eval_threshold", type=float, default=0.6, help="实验成功率阈值，超过该值的理论将进行多角色评估")
    evaluation_group.add_argument(
        "--role_eval_models",
        type=str,
        default=None,
        help="角色评估时使用的模型列表，如 'openai:gpt-4o-mini,google:gemini-2.5-flash'",
    )
    evaluation_group.add_argument(
        "--auto_register_threshold",
        type=float,
        default=0.7,
        help="综合评分达到该阈值时自动注册到全局理论库（<=0 表示不自动注册）",
    )
    evaluation_group.add_argument(
        "--auto_register_registry",
        type=str,
        default="global_theory_registry_multi",
        help="自动注册时目标理论库目录",
    )
    evaluation_group.add_argument(
        "--auto_register_visuals",
        type=str,
        default="theory_visuals_multi",
        help="自动注册后可视化输出目录",
    )
    evaluation_group.add_argument(
        "--skip_experiments",
        nargs='*',
        default=None,
        help="评估阶段要跳过的实验ID列表",
    )
    evaluation_group.add_argument(
        "--include_fullerene_experiment",
        action='store_true',
        help="默认跳过的富勒烯退相干实验重新加入评估。",
    )

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
        "--generation_method", args.generation_method,
        "--model_source", args.synthesis_model_source,
        "--model_name", args.synthesis_model_name,
        "--output_dir", synthesis_output_dir
    ]

    if args.generation_method == "direct":
        synthesis_command.extend([
            "--theories_dir", args.existing_theories_dir,
            "--max_pairs", str(args.max_pairs_to_analyze),
            "--variants_per_contradiction", str(args.variants_per_contradiction),
        ])
    else:
        synthesis_command.extend([
            "--cards_dir", args.cards_dir,
            "--card_schema", args.card_schema,
            "--contradiction_schema", args.contradiction_schema,
            "--new_interpretation_schema", args.new_interpretation_schema,
            "--short_card_query", args.short_card_query,
            "--short_card_task_hint", args.short_card_task_hint,
            "--short_card_topk", str(args.short_card_topk),
            "--short_card_machine_temperature", str(args.short_card_machine_temperature),
            "--short_card_human_temperature", str(args.short_card_human_temperature),
        ])
        if args.short_card_constraints:
            synthesis_command.extend(["--short_card_constraints", args.short_card_constraints])
        if args.short_card_human_model_source:
            synthesis_command.extend(["--short_card_human_model_source", args.short_card_human_model_source])
        if args.short_card_human_model_name:
            synthesis_command.extend(["--short_card_human_model_name", args.short_card_human_model_name])
        if args.short_card_contradictions:
            synthesis_command.extend(["--short_card_contradictions", args.short_card_contradictions])
        if args.num_theories and args.num_theories != 1:
            synthesis_command.extend(["--num_theories", str(args.num_theories)])

    synthesis_return_code, synthesis_output = run_command(synthesis_command, "理论合成")

    if synthesis_return_code != 0:
        print("\n[FATAL] 理论合成阶段失败，无法继续。请检查以上日志。")
        sys.exit(1)

    # 从合成脚本的输出中解析出可评估理论的目录
    eval_ready_path_match = re.search(r"Evaluation-ready theory files saved to: (.*)", synthesis_output)
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

    skip_set = set(DEFAULT_EXPERIMENT_SKIPS)
    if args.skip_experiments:
        skip_set.update(args.skip_experiments)
    if args.include_fullerene_experiment:
        skip_set.discard("fullerene_decoherence_hornberger2003")
    if skip_set:
        evaluation_command.append("--skip_experiments")
        evaluation_command.extend(sorted(skip_set))

    if args.use_instrument_correction:
        evaluation_command.append("--use_instrument_correction")

    if args.role_eval_models:
        evaluation_command.extend(["--role_eval_models", args.role_eval_models])

    evaluation_return_code, _ = run_command(evaluation_command, "理论评估")

    if evaluation_return_code != 0:
        print("\n[FATAL] 理论评估阶段失败。请检查以上日志。")
        sys.exit(1)

    try:
        auto_register_high_scoring_theories(
            main_run_dir=main_run_dir,
            evaluation_output_dir=evaluation_output_dir,
            threshold=args.auto_register_threshold,
            generation_method=args.generation_method,
            synthesis_model_source=args.synthesis_model_source,
            synthesis_model_name=args.synthesis_model_name,
            registry_dir=args.auto_register_registry,
            visuals_dir=args.auto_register_visuals,
        )
    except Exception as exc:
        print(f"[WARN] 自动注册高分理论失败: {exc}")
        
    print_banner("全周期运行成功完成！")
    print(f"所有结果已保存在: {main_run_dir}")


if __name__ == "__main__":
    main() 
