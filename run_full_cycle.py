#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
UniversalTheoryGen - 端到端全周期运行器 (多代迭代版)
====================================================
基于 `run_evolution_cycle` 的清单管理能力，将理论合成、评估与多轮精炼整合到同一入口脚本中。
无论是直接矛盾分析（direct）还是短卡模式（short_card），均会先完成首轮评估，并根据评估反馈
触发后续多代优化，支持断点续跑与理论血缘追踪。
"""

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

from run_evolution_cycle import EvolutionOrchestrator
from utils.global_theory_registry import GlobalTheoryRegistry

DEFAULT_EXPERIMENT_SKIPS = {"fullerene_decoherence_hornberger2003"}


def print_banner(text: str):
    """打印一个简单横幅，提升可读性。"""
    line = "=" * (len(text) + 4)
    print(f"\n{line}")
    print(f"| {text} |")
    print(f"{line}\n")


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
                search_root = Path(main_run_dir)
                matches = list(search_root.glob(f"**/{theory_file.name}")) if theory_file.name else []
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


class FullCycleEvolutionOrchestrator(EvolutionOrchestrator):
    """
    在原有多代调度器基础上，加入 full-cycle 的输入输出结构、短卡支持与评估附加参数。
    """

    def __init__(self, config: Dict, evaluation_options: Dict):
        self.evaluation_options = evaluation_options
        self.generation_method = config.get("generation_method", "direct")
        self.generation_eval_dirs: Dict[int, Path] = {}
        super().__init__(config)

    def _call_synthesis_stage(self, gen_dir: Path):
        """调用理论合成阶段（支持 direct / short_card 两种模式）。"""
        synthesis_dir = gen_dir / "1_synthesis_output"
        synthesis_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "run_direct_synthesis.py",
            "--generation_method",
            self.generation_method,
            "--model_source",
            self.config["synthesis_model_source"],
            "--model_name",
            self.config["synthesis_model_name"],
            "--output_dir",
            str(synthesis_dir),
        ]

        if self.generation_method == "direct":
            cmd.extend(
                [
                    "--theories_dir",
                    self.config.get("existing_theories_dir", self.config["initial_theories_dir"]),
                    "--max_pairs",
                    str(self.config["max_pairs_to_analyze"]),
                    "--variants_per_contradiction",
                    str(self.config["variants_per_contradiction"]),
                ]
            )
        else:
            cmd.extend(
                [
                    "--cards_dir",
                    self.config["cards_dir"],
                    "--card_schema",
                    self.config["card_schema"],
                    "--contradiction_schema",
                    self.config["contradiction_schema"],
                    "--new_interpretation_schema",
                    self.config["new_interpretation_schema"],
                    "--short_card_query",
                    self.config["short_card_query"],
                    "--short_card_task_hint",
                    self.config["short_card_task_hint"],
                    "--short_card_topk",
                    str(self.config["short_card_topk"]),
                    "--short_card_machine_temperature",
                    str(self.config["short_card_machine_temperature"]),
                    "--short_card_human_temperature",
                    str(self.config["short_card_human_temperature"]),
                ]
            )

            if self.config.get("short_card_constraints"):
                cmd.extend(["--short_card_constraints", self.config["short_card_constraints"]])
            if self.config.get("short_card_human_model_source"):
                cmd.extend(["--short_card_human_model_source", self.config["short_card_human_model_source"]])
            if self.config.get("short_card_human_model_name"):
                cmd.extend(["--short_card_human_model_name", self.config["short_card_human_model_name"]])
            if self.config.get("short_card_contradictions"):
                cmd.extend(["--short_card_contradictions", self.config["short_card_contradictions"]])
            if self.config.get("num_theories") and self.config.get("num_theories") != 1:
                cmd.extend(["--num_theories", str(self.config["num_theories"])])

        result = self._execute_command("理论合成", cmd)
        if result["success"]:
            result["output_dir"] = synthesis_dir
        return result

    def _evaluate_generation(self, generation: int, gen_dir: Path):
        """评估指定代际的理论，复用 full-cycle 的评估命令结构。"""
        unevaluated = self._get_unevaluated_theories(generation)
        if not unevaluated:
            print(f"[INFO] Generation {generation} 没有需要评估的理论")
            return True

        print(f"[INFO] 发现 {len(unevaluated)} 个待评估理论")
        for theory_id, theory_data in unevaluated.items():
            print(f"  - {theory_id}: {theory_data['theory_name']} (状态: {theory_data.get('status', 'unknown')})")

        evaluation_root = gen_dir / "2_evaluation_output"
        evaluation_root.mkdir(exist_ok=True)

        temp_theories_dir = evaluation_root / "theories"
        temp_theories_dir.mkdir(exist_ok=True)

        for theory_id, theory_data in unevaluated.items():
            src_path = Path(theory_data["file_path"])
            dest_path = temp_theories_dir / f"{theory_id}_{src_path.name}"
            shutil.copy(src_path, dest_path)
            print(f"[COPY] {src_path.name} -> {dest_path}")

        eval_output_dir = evaluation_root / "results"
        eval_output_dir.mkdir(exist_ok=True)

        cmd = [
            "python",
            "demo/demo_1.py",
            "--theory_path",
            str(temp_theories_dir),
            "--experiment_dir",
            self.config["experiment_dir"],
            "--output_dir",
            str(eval_output_dir),
            "--model_source",
            self.config["evaluation_model_source"],
            "--model_name",
            self.config["evaluation_model_name"],
            "--run_role_evaluation",
        ]

        role_threshold = self.evaluation_options.get("role_eval_threshold")
        if role_threshold is not None:
            cmd.extend(["--role_success_threshold", str(role_threshold)])

        skip_set = set(DEFAULT_EXPERIMENT_SKIPS)
        if self.evaluation_options.get("skip_experiments"):
            skip_set.update(self.evaluation_options["skip_experiments"])
        if self.evaluation_options.get("include_fullerene_experiment"):
            skip_set.discard("fullerene_decoherence_hornberger2003")
        if skip_set:
            cmd.append("--skip_experiments")
            cmd.extend(sorted(skip_set))

        if self.evaluation_options.get("use_instrument_correction"):
            cmd.append("--use_instrument_correction")

        if self.evaluation_options.get("role_eval_models"):
            cmd.extend(["--role_eval_models", self.evaluation_options["role_eval_models"]])

        result = self._execute_command("理论评估", cmd)
        if not result["success"]:
            return False

        success = self._update_scores_from_evaluation(eval_output_dir)
        if not success:
            return False

        self.generation_eval_dirs[generation] = eval_output_dir
        self._mark_generation_complete(generation)

        return True

    def get_latest_evaluation_dir(self) -> Optional[Path]:
        """返回最新一代评估结果目录，用于自动注册。"""
        if not self.generation_eval_dirs:
            return None
        latest_generation = max(self.generation_eval_dirs.keys())
        return self.generation_eval_dirs[latest_generation]


def build_evolution_config(args: argparse.Namespace) -> Dict:
    dialog_model_source = args.dialog_model_source or args.evaluation_model_source
    dialog_model_name = args.dialog_model_name or args.evaluation_model_name

    return {
        "output_root": args.base_output_dir,
        "initial_theories_dir": args.existing_theories_dir,
        "existing_theories_dir": args.existing_theories_dir,
        "experiment_dir": args.experiment_dir,
        "max_generations": args.max_generations,
        "promotion_min_score": args.promotion_min_score,
        "top_n_survivors": args.top_n_survivors,
        "max_pairs_to_analyze": args.max_pairs_to_analyze,
        "variants_per_contradiction": args.variants_per_contradiction,
        "synthesis_model_source": args.synthesis_model_source,
        "synthesis_model_name": args.synthesis_model_name,
        "evaluation_model_source": args.evaluation_model_source,
        "evaluation_model_name": args.evaluation_model_name,
        "dialog_model_source": dialog_model_source,
        "dialog_model_name": dialog_model_name,
        "use_instrument_correction": args.use_instrument_correction,
        "max_refinement_iters": args.max_refinement_iters,
        "min_improvement": args.min_improvement,
        "generation_method": args.generation_method,
        "cards_dir": args.cards_dir,
        "card_schema": args.card_schema,
        "contradiction_schema": args.contradiction_schema,
        "new_interpretation_schema": args.new_interpretation_schema,
        "short_card_query": args.short_card_query,
        "short_card_task_hint": args.short_card_task_hint,
        "short_card_topk": args.short_card_topk,
        "short_card_constraints": args.short_card_constraints,
        "short_card_machine_temperature": args.short_card_machine_temperature,
        "short_card_human_temperature": args.short_card_human_temperature,
        "short_card_human_model_source": args.short_card_human_model_source,
        "short_card_human_model_name": args.short_card_human_model_name,
        "short_card_contradictions": args.short_card_contradictions,
        "num_theories": args.num_theories,
    }


def build_evaluation_options(args: argparse.Namespace) -> Dict:
    return {
        "role_eval_threshold": args.role_eval_threshold,
        "skip_experiments": args.skip_experiments or [],
        "include_fullerene_experiment": args.include_fullerene_experiment,
        "role_eval_models": args.role_eval_models,
        "use_instrument_correction": args.use_instrument_correction,
    }


def main():
    parser = argparse.ArgumentParser(
        description="UniversalTheoryGen - 端到端全周期运行器（多代演进版）",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # --- 总控参数 ---
    control_group = parser.add_argument_group("Overall Control Parameters")
    control_group.add_argument("--base_output_dir", type=str, default="data/full_cycle_runs", help="所有运行结果的根输出目录")
    control_group.add_argument("--max_generations", type=int, default=1, help="演进的最大代数（>=2 时开启多轮优化）")
    control_group.add_argument("--promotion_min_score", type=float, default=0.6, help="晋级到下一代所需的最低综合分")
    control_group.add_argument("--top_n_survivors", type=int, default=2, help="每代保留用于精炼的理论数量")
    control_group.add_argument("--max_refinement_iters", type=int, default=3, help="单次精炼的最大迭代次数")
    control_group.add_argument("--min_improvement", type=float, default=0.05, help="精炼时认定有效提升的最小分值")

    # --- 理论合成参数 (Synthesis Parameters) ---
    synthesis_group = parser.add_argument_group("Phase 1: Theory Synthesis")
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
    synthesis_group.add_argument("--short_card_query", type=str, default="短卡联合分析生成新理论", help="短卡模式任务描述")
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
    evaluation_group = parser.add_argument_group("Phase 2: Theory Evaluation")
    evaluation_group.add_argument("--experiment_dir", type=str, default="demo/experiments/", help="用于评估的实验数据目录")
    evaluation_group.add_argument("--use_instrument_correction", action="store_true", default=True, help="在实验评估中启用仪器修正模型（默认启用）")
    evaluation_group.add_argument("--evaluation_model_source", type=str, default="google", choices=["openai", "deepseek", "xai", "google"], help="用于理论评估的LLM来源")
    evaluation_group.add_argument("--evaluation_model_name", type=str, default="gemini-2.5-pro", help="用于理论评估的具体模型名称")
    evaluation_group.add_argument("--dialog_model_source", type=str, default=None, help="精炼深度对话的模型来源（默认继承评估模型）")
    evaluation_group.add_argument("--dialog_model_name", type=str, default=None, help="精炼深度对话的模型名称（默认继承评估模型）")
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
        nargs="*",
        default=None,
        help="评估阶段要跳过的实验ID列表",
    )
    evaluation_group.add_argument(
        "--include_fullerene_experiment",
        action="store_true",
        help="默认跳过的富勒烯退相干实验重新加入评估。",
    )

    args = parser.parse_args()

    # --- 智能推断模型来源 ---
    if args.synthesis_model_source is None:
        if "gemini" in args.synthesis_model_name.lower():
            args.synthesis_model_source = "google"
        elif "gpt" in args.synthesis_model_name.lower():
            args.synthesis_model_source = "openai"
        elif "deepseek" in args.synthesis_model_name.lower():
            args.synthesis_model_source = "deepseek"

    if args.evaluation_model_source is None:
        if "gemini" in args.evaluation_model_name.lower():
            args.evaluation_model_source = "google"
        elif "gpt" in args.evaluation_model_name.lower():
            args.evaluation_model_source = "openai"
        elif "deepseek" in args.evaluation_model_name.lower():
            args.evaluation_model_source = "deepseek"

    config = build_evolution_config(args)
    evaluation_options = build_evaluation_options(args)

    orchestrator = FullCycleEvolutionOrchestrator(config, evaluation_options)
    success = orchestrator.run_full_evolution()

    if not success:
        print("\n[FATAL] 全周期演进流程失败。请检查以上日志。")
        sys.exit(1)

    main_run_dir = str(orchestrator.run_root)
    latest_eval_dir = orchestrator.get_latest_evaluation_dir()

    if latest_eval_dir:
        try:
            auto_register_high_scoring_theories(
                main_run_dir=main_run_dir,
                evaluation_output_dir=str(latest_eval_dir),
                threshold=args.auto_register_threshold,
                generation_method=args.generation_method,
                synthesis_model_source=args.synthesis_model_source,
                synthesis_model_name=args.synthesis_model_name,
                registry_dir=args.auto_register_registry,
                visuals_dir=args.auto_register_visuals,
            )
        except Exception as exc:
            print(f"[WARN] 自动注册高分理论失败: {exc}")

    print_banner("全周期演进流程完成")
    print(f"所有结果已保存在: {main_run_dir}")


if __name__ == "__main__":
    main()
