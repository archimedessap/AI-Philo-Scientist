#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_eval_dialogue_cycle.py
=========================
实现如下闭环：
  0) 初始：run_full_cycle.py 生成新理论并评估
     └► 得到 eval_ready_theories + final_evaluation_summary.json
     └► 立即执行 run_m3_auto_refinement.py 做首轮对话改写

  1..N) 迭代：
     a) demo/demo_1.py  对当前理论池重新评估，生成新的 summary
     b) run_m3_auto_refinement.py  多轮对话改写
     c) 用改写版本更新理论池，计算平均提升
     d) 若 avg_delta < stop_delta (默认 0.03) 或达到最大代数则停止

用法示例：
python run_eval_dialogue_cycle.py \
  --generations 1 \
  --initial_theories_dir data/theories_v2.1 \
  --output_root data/dialog_cycle_runs_test_gemini \
  --max_pairs_to_analyze 1 \
  --variants_per_contradiction 1 \
  --top_n 1 --max_iters 1 --stop_delta 0.03 \
  --role_eval_threshold 0.5 \
  --synthesis_model_source google  --synthesis_model_name gemini-2.5-pro-preview-06-05 \
  --evaluation_model_source google --evaluation_model_name gemini-2.5-pro-preview-06-05 \
  --dialog_model_source google     --dialog_model_name gemini-2.5-pro-preview-06-05
"""

from __future__ import annotations
import subprocess
import sys
import argparse
from pathlib import Path
import glob
import json
import shutil
import time


# -----------------------------------------------------------------------------
# 工具函数
# -----------------------------------------------------------------------------

def run_cmd(cmd: list[str]):
    print("\n$ " + " ".join(cmd))
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"[FATAL] 命令失败: {' '.join(cmd)}")
        sys.exit(ret)


def find_unique(pattern: str) -> Path:
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"未找到匹配: {pattern}")
    if len(matches) > 1:
        raise RuntimeError(f"匹配到多个路径: {matches}")
    return Path(matches[0])


def collect_improved(depth_runs_dir: Path, dest_dir: Path):
    """复制 depth_runs 中最后 iter 的 improved_*.json 到 dest_dir，返回理论及其 delta 列表"""
    from tools.refinement.schema_validator import is_valid_theory_json

    dest_dir.mkdir(parents=True, exist_ok=True)
    deltas = []

    for theory_dir in depth_runs_dir.iterdir():
        if not theory_dir.is_dir():
            continue
        # 读取 summary
        summary_file = theory_dir / "summary.json"
        if not summary_file.exists():
            continue
        with open(summary_file, "r", encoding="utf-8") as f:
            sdata = json.load(f)
        baseline = sdata["scores"][0]
        final_score = sdata["final_score"]
        deltas.append(final_score - baseline)

        # 找 improved 文件
        iter_dirs = sorted([p for p in theory_dir.iterdir() if p.is_dir() and p.name.startswith("iter_")],
                           key=lambda p: int(p.name.split("_")[1]))
        last_iter = iter_dirs[-1] if iter_dirs else None
        src_json = None
        if last_iter:
            improved = list(last_iter.glob("improved_*.json"))
            if improved:
                src_json = improved[0]
        if src_json is None:
            continue
        # 校验
        with open(src_json, "r", encoding="utf-8") as f:
            tdata = json.load(f)
        if not is_valid_theory_json(tdata):
            continue
        # 统一文件命名，避免不同阶段字符串替换规则不一致
        try:
            # 与 candidate_selector.slugify 保持一致
            from tools.refinement.candidate_selector import slugify  # type: ignore
        except ImportError:
            # 回退：简单替换非法字符
            import re

            def slugify(text: str) -> str:
                return re.sub(r"[^0-9a-zA-Z]+", "_", text).strip("_")[:120]

        slug_name = slugify(tdata.get("name", theory_dir.name)) + ".json"
        shutil.copy(src_json, dest_dir / slug_name)
    # 计算平均提升
    avg_delta = sum(deltas)/len(deltas) if deltas else 0
    return avg_delta, len(deltas)

# -----------------------------------------------------------------------------
# 主逻辑
# -----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("评估↔对话多轮循环控制脚本")
    parser.add_argument("--generations", type=int, default=5, help="最大循环代数（含首代）")
    parser.add_argument("--initial_theories_dir", required=True, help="初始已有理论池目录")
    parser.add_argument("--output_root", default="data/dialog_cycle_runs", help="总输出根目录")

    parser.add_argument("--stop_delta", type=float, default=0.03, help="平均提升低于该阈值则提前停止")

    # --- 生成阶段参数 ---
    parser.add_argument("--max_pairs_to_analyze", type=int, default=5, help="run_direct_synthesis.py: 分析的最大矛盾对数")
    parser.add_argument("--variants_per_contradiction", type=int, default=1, help="run_direct_synthesis.py: 每个矛盾生成的变体数")
    parser.add_argument("--synthesis_model_source", default="google")
    parser.add_argument("--synthesis_model_name", default="gemini-1.5-pro-latest")

    # --- 评估阶段 LLM 参数（Judge）---
    parser.add_argument("--evaluation_model_source", default="deepseek")
    parser.add_argument("--evaluation_model_name", default="deepseek-reasoner")

    # --- 对话优化阶段 LLM 参数 ---
    parser.add_argument("--dialog_model_source", default="deepseek")
    parser.add_argument("--dialog_model_name", default="deepseek-reasoner")

    parser.add_argument("--top_n", type=int, default=3)
    parser.add_argument("--max_iters", type=int, default=2)
    parser.add_argument("--min_improve", type=float, default=0.03)

    # --- 角色评估阈值 ---
    parser.add_argument("--role_eval_threshold", type=float, default=0.6, help="实验成功率达到该值才进行角色评估 (透传给 run_full_cycle.py)")

    args = parser.parse_args()

    root = Path(args.output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)

    # ---------------- Generation 0 -----------------
    gen0_dir = root / "generation_0"
    gen0_dir.mkdir(exist_ok=True)

    # Phase 0A: run_full_cycle
    full_cycle_dir = gen0_dir / "full_cycle"
    cmd_full = [
        "python", "run_full_cycle.py",
        "--existing_theories_dir", args.initial_theories_dir,
        "--base_output_dir", str(full_cycle_dir),
        "--max_pairs_to_analyze", str(args.max_pairs_to_analyze),
        "--variants_per_contradiction", str(args.variants_per_contradiction),
        "--synthesis_model_source", args.synthesis_model_source,
        "--synthesis_model_name", args.synthesis_model_name,
        "--evaluation_model_source", args.evaluation_model_source,
        "--evaluation_model_name", args.evaluation_model_name,
        "--role_eval_threshold", str(args.role_eval_threshold),
        "--synthesis_model_source", args.synthesis_model_source,
        "--synthesis_model_name", args.synthesis_model_name
    ]
    run_cmd(cmd_full)

    # 定位路径
    run_subdir = find_unique(str(full_cycle_dir / "run_*/"))
    summary_file = find_unique(str(run_subdir / "2_evaluation_output" / "*/final_evaluation_summary.json"))
    theories_root = find_unique(str(run_subdir / "1_synthesis_output" / "*/eval_ready_theories"))

    # Phase 0B: 首轮对话改写
    refinement_dir = gen0_dir / "refinement"
    cmd_refine0 = [
        "python", "run_m3_auto_refinement.py",
        "--summary_file", str(summary_file),
        "--theories_root", str(theories_root),
        "--output_dir", str(refinement_dir),
        "--top_n", str(args.top_n),
        "--eval_mode", "real",
        "--max_iters", str(args.max_iters),
        "--min_improve", str(args.min_improve),
        "--judge_model_source", args.evaluation_model_source,
        "--judge_model_name", args.evaluation_model_name,
        "--dialog_model_source", args.dialog_model_source,
        "--dialog_model_name", args.dialog_model_name
    ]
    run_cmd(cmd_refine0)

    # 兼容 run_m3_auto_refinement 的输出结构：可能带 run_*/
    candidates = list(refinement_dir.glob("run_*/depth_output/depth_runs"))
    depth_runs_dir = candidates[0] if candidates else (refinement_dir / "depth_output/depth_runs")

    pool_dir = gen0_dir / "theory_pool"
    avg_delta, count = collect_improved(depth_runs_dir, pool_dir)
    print(f"[GEN0] 收集 {count} 个改写理论，平均提升 {avg_delta:.3f}")

    # ---------------- Subsequent Generations -----------------
    theories_dir_for_next = pool_dir

    for gen in range(1, args.generations):
        print("\n" + "="*80)
        print(f"Generation {gen}")
        print("="*80)
        gen_dir = root / f"generation_{gen}"
        gen_dir.mkdir(exist_ok=True)

        # Phase A: 重新评估
        eval_dir = gen_dir / "evaluation"
        eval_dir.mkdir(exist_ok=True)
        cmd_eval = [
            "python", "demo/demo_1.py",
            "--theory_path", str(theories_dir_for_next),
            "--experiment_dir", "demo/experiments/",
            "--output_dir", str(eval_dir),
            "--model_source", args.evaluation_model_source,
            "--model_name", args.evaluation_model_name,
            "--run_role_evaluation",
            "--role_model_source", args.evaluation_model_source,
            "--role_model_name", args.evaluation_model_name
        ]
        run_cmd(cmd_eval)
        summary_file = find_unique(str(eval_dir / "run_*/final_evaluation_summary.json"))

        # Phase B: 对话改写
        refinement_dir = gen_dir / "refinement"
        cmd_refine = [
            "python", "run_m3_auto_refinement.py",
            "--summary_file", str(summary_file),
            "--theories_root", str(theories_dir_for_next),
            "--output_dir", str(refinement_dir),
            "--top_n", str(args.top_n),
            "--eval_mode", "real",
            "--max_iters", str(args.max_iters),
            "--min_improve", str(args.min_improve),
            "--judge_model_source", args.evaluation_model_source,
            "--judge_model_name", args.evaluation_model_name,
            "--dialog_model_source", args.dialog_model_source,
            "--dialog_model_name", args.dialog_model_name
        ]
        run_cmd(cmd_refine)

        # 查找 depth_runs 目录
        candidates = list(refinement_dir.glob("run_*/depth_output/depth_runs"))
        depth_runs_dir = candidates[0] if candidates else (refinement_dir / "depth_output/depth_runs")

        next_pool = gen_dir / "theory_pool"
        avg_delta, count = collect_improved(depth_runs_dir, next_pool)
        print(f"[GEN{gen}] 收集 {count} 个改写理论，平均提升 {avg_delta:.3f}")

        if avg_delta < args.stop_delta:
            print(f"[STOP] 平均提升 {avg_delta:.3f} < stop_delta {args.stop_delta}, 提前结束循环")
            break
        theories_dir_for_next = next_pool
        time.sleep(1)

    print("\n🎉 评估↔对话循环完成，全部结果位于:", root)


if __name__ == "__main__":
    main() 